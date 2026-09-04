"""Alert dispatch: snapshot, evidence clip, persistence, live push, PTZ, C2.

Everything slow (JPEG/MP4 encode, HTTP to a C2 system, ONVIF SOAP) happens off
the analytics thread. The pipeline hands over a frame copy and returns to
decoding immediately.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..config import CLIP_DIR, PTZ, SNAP_DIR, settings
from .. import db


class EventBus:
    """Thread -> asyncio bridge. Workers `publish()`; the WebSocket endpoint
    drains via `subscribe()`."""

    def __init__(self, maxsize: int = 256) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, message: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(message)
            except queue.Full:
                pass          # a stalled client must not back-pressure analytics

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


bus = EventBus()


def save_snapshot(frame: np.ndarray, camera_id: str, event_ts: float) -> str:
    day = time.strftime("%Y%m%d", time.localtime(event_ts))
    folder = SNAP_DIR / day
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{camera_id}_{int(event_ts * 1000)}.jpg"
    path = folder / name
    cv2.imwrite(str(path), frame,
                [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
    return f"{day}/{name}"


class ClipRecorder:
    """Rolling pre-roll buffer per camera; writes pre+post MP4 on demand."""

    def __init__(self, fps: float = 8.0) -> None:
        self.fps = max(2.0, fps)
        self.buffer: deque = deque(maxlen=int(self.fps * settings.clip_pre_seconds))
        self._active: list[tuple[Path, list[np.ndarray], int]] = []
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        self.buffer.append(frame)
        with self._lock:
            done = []
            for i, (path, frames, remaining) in enumerate(self._active):
                frames.append(frame)
                self._active[i] = (path, frames, remaining - 1)
                if remaining - 1 <= 0:
                    done.append(i)
            for i in reversed(done):
                path, frames, _ = self._active.pop(i)
                threading.Thread(target=self._write, args=(path, frames),
                                 daemon=True).start()

    def start(self, camera_id: str, event_ts: float) -> str:
        day = time.strftime("%Y%m%d", time.localtime(event_ts))
        folder = CLIP_DIR / day
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{camera_id}_{int(event_ts * 1000)}.mp4"
        with self._lock:
            self._active.append((folder / name, list(self.buffer),
                                 int(self.fps * settings.clip_post_seconds)))
        return f"{day}/{name}"

    def _write(self, path: Path, frames: list[np.ndarray]) -> None:
        if not frames:
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (w, h))
        if not writer.isOpened():
            return
        for f in frames:
            writer.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        writer.release()


def post_to_c2(payload: dict) -> None:
    """Fire-and-forget push to an existing command-and-control system."""
    url = settings.c2_webhook_url
    if not url:
        return

    def _send() -> None:
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload, default=str).encode(),
                headers={"content-type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
            db.audit("c2.push", {"event_id": payload.get("id"), "url": url})
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            db.audit("c2.push_failed", {"error": str(exc), "url": url})

    threading.Thread(target=_send, daemon=True).start()


# --------------------------------------------------------------------------
# PTZ slew-to-target
# --------------------------------------------------------------------------
def ptz_command_for(box: tuple[int, int, int, int],
                    frame_shape: tuple[int, int],
                    deadzone: float = 0.08) -> dict:
    """Proportional pan/tilt velocities that centre `box` in the frame.

    Returns velocities in ONVIF's normalised [-1, 1] space, plus a zoom hint
    based on how small the target is."""
    h, w = frame_shape[:2]
    cx = (box[0] + box[2]) / 2.0 / max(1, w)
    cy = (box[1] + box[3]) / 2.0 / max(1, h)
    ex, ey = cx - 0.5, cy - 0.5
    pan = 0.0 if abs(ex) < deadzone else max(-1.0, min(1.0, ex * 2.2))
    tilt = 0.0 if abs(ey) < deadzone else max(-1.0, min(1.0, -ey * 2.2))
    target_frac = (box[3] - box[1]) / max(1, h)
    zoom = 0.35 if target_frac < 0.18 else (-0.3 if target_frac > 0.62 else 0.0)
    return {"pan": round(pan, 3), "tilt": round(tilt, 3),
            "zoom": round(zoom, 3), "target_fraction": round(target_frac, 3)}


_SOAP = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
 <s:Header><Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
  <UsernameToken><Username>{user}</Username>
  <Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>
  <Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce}</Nonce>
  <Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>
  </UsernameToken></Security></s:Header>
 <s:Body><ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{profile}</ProfileToken>
  <Velocity><PanTilt x="{pan}" y="{tilt}" xmlns="http://www.onvif.org/ver10/schema"/>
  <Zoom x="{zoom}" xmlns="http://www.onvif.org/ver10/schema"/></Velocity>
 </ContinuousMove></s:Body></s:Envelope>"""


def _ws_security(user: str, password: str) -> dict:
    import base64
    import hashlib
    import os
    from datetime import datetime, timezone
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(hashlib.sha1(
        nonce + created.encode() + password.encode()).digest()).decode()
    return {"user": user, "digest": digest,
            "nonce": base64.b64encode(nonce).decode(), "created": created}


class PTZController:
    """Slews a PTZ head onto the highest-threat target.

    `settings.ptz_simulate` (the default) records exactly the command that would
    have gone out and surfaces it in the UI, so the control path is verifiable
    without a physical head on the bench. The live ONVIF branch builds a valid
    ContinuousMove envelope but has not been exercised against real hardware
    here — treat it as untested until you point it at a camera.
    """

    def __init__(self, cooldown: float = 1.0) -> None:
        self.last_sent = 0.0
        self.cooldown = cooldown
        self.history: deque = deque(maxlen=25)

    def slew(self, camera_id: str, ptz: PTZ, box: tuple[int, int, int, int],
             frame_shape: tuple[int, int], reason: str = "") -> dict | None:
        now = time.time()
        if now - self.last_sent < self.cooldown:
            return None
        cmd = ptz_command_for(box, frame_shape)
        if cmd["pan"] == 0.0 and cmd["tilt"] == 0.0 and cmd["zoom"] == 0.0:
            return None
        self.last_sent = now
        record = {"ts": now, "camera_id": camera_id, "reason": reason,
                  "command": cmd,
                  "mode": "simulated" if settings.ptz_simulate else "live"}
        self.history.append(record)
        db.audit("ptz.slew", record)
        if settings.ptz_simulate or not ptz.enabled:
            return record
        threading.Thread(target=self._send, args=(ptz, cmd), daemon=True).start()
        return record

    @staticmethod
    def _send(ptz: PTZ, cmd: dict) -> None:
        import os
        password = os.environ.get(ptz.password_env, "") if ptz.password_env else ""
        body = _SOAP.format(profile="Profile_1", pan=cmd["pan"],
                            tilt=cmd["tilt"], zoom=cmd["zoom"],
                            **_ws_security(ptz.username, password)).encode()
        url = f"http://{ptz.host}:{ptz.port}/onvif/ptz_service"
        try:
            req = urllib.request.Request(url, data=body, headers={
                "content-type": "application/soap+xml; charset=utf-8"})
            urllib.request.urlopen(req, timeout=4).read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            db.audit("ptz.failed", {"host": ptz.host, "error": str(exc)})
