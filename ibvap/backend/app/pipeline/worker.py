"""Per-camera analytics worker.

One thread per camera runs the whole chain: ingest -> enhance -> detect+track ->
pose -> action -> zone logic -> face/ANPR -> threat -> annotate -> dispatch.

Frames are never queued. The worker always pulls the newest frame from the
source and paces itself to `settings.target_fps`; if inference cannot keep up it
processes fewer frames rather than falling behind real time, which is the right
trade for a surveillance feed.
"""
from __future__ import annotations

import threading
import time

import cv2

from ..config import Camera, settings
from .. import db
from . import annotate as viz
from .actions import SUSPICIOUS, classify
from .anpr import ANPRWorker, is_valid
from .caption import caption, scene_summary
from .detect import Detector, attach_carried_objects
from .dispatch import ClipRecorder, PTZController, bus, post_to_c2, save_snapshot
from .enhance import Enhancer
from .faces import FaceEngine
from .ingest import VideoSource
from .pose import PoseEstimator
from .threat import assess
from .tracks import TrackStore
from .zones import ZoneEngine
from .reid import bpb_extractor, reid_engine

# One OCR process-wide: EasyOCR holds ~0.5 GB, and plate reads are bursty.
_anpr: ANPRWorker | None = None
_anpr_lock = threading.Lock()


def anpr_worker() -> ANPRWorker:
    global _anpr
    with _anpr_lock:
        if _anpr is None:
            _anpr = ANPRWorker()
        return _anpr

class CameraWorker(threading.Thread):
    def __init__(self, camera: Camera) -> None:
        super().__init__(name=f"cam-{camera.id}", daemon=True)
        self.camera = camera
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._raw_jpeg: bytes | None = None

        self.zones = ZoneEngine(camera.zones)
        self.tracks = TrackStore()
        self.overlay = viz.ZoneOverlayCache()
        self.clips = ClipRecorder(fps=settings.target_fps)
        self.ptz = PTZController()

        self.detector: Detector | None = None
        self.pose: PoseEstimator | None = None
        self.enhancer: Enhancer | None = None
        self.faces: FaceEngine | None = None
        self.source: VideoSource | None = None

        self.fps = 0.0
        self.processed = 0
        self.summary = "Starting..."
        self.is_night = False
        self.enhanced = False
        self.luma = 0.0
        self.top_threat: tuple[int, str] = (0, "")
        self.error = ""
        self.started_ts = 0.0

    # ------------------------------------------------------------------ state
    @property
    def status(self) -> dict:
        return {
            "id": self.camera.id, "name": self.camera.name,
            "sector": self.camera.sector, "enabled": self.camera.enabled,
            "running": self.is_alive() and not self._stop.is_set(),
            "fps": round(self.fps, 1), "frames": self.processed,
            "is_night": self.is_night, "enhanced": self.enhanced,
            "luma": round(self.luma, 1), "tracks": self.tracks.active,
            "top_threat": {"score": self.top_threat[0],
                           "level": self.top_threat[1]},
            "summary": self.summary, "error": self.error,
            "uptime": round(time.time() - self.started_ts, 1) if self.started_ts else 0,
            "source": self.source.status if self.source else {},
            "zone_occupancy": self.zones.occupancy(self.tracks.tracks),
            "ptz_history": list(self.ptz.history)[-5:],
        }

    def frame_jpeg(self, raw: bool = False) -> bytes | None:
        with self._frame_lock:
            return self._raw_jpeg if raw else self._jpeg

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ setup
    def _setup(self) -> bool:
        try:
            self.enhancer = Enhancer()
            self.detector = Detector()
            self.pose = PoseEstimator()
            self.faces = FaceEngine()
        except Exception as exc:
            self.error = f"model load failed: {exc.__class__.__name__}: {exc}"
            db.audit("camera.model_error",
                     {"camera": self.camera.id, "error": self.error})
            return False
        anpr_worker()                       # kick off the lazy OCR load early
        self.source = VideoSource(self.camera.source, loop=self.camera.loop,
                                  name=self.camera.id)
        if not self.source.start():
            self.error = self.source.status.get("error") or "source failed"
            db.audit("camera.source_error",
                     {"camera": self.camera.id, "error": self.error})
            return False
        self.started_ts = time.time()
        db.audit("camera.start", {"camera": self.camera.id,
                                  "source": self.camera.source})
        return True

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        if not self._setup():
            return
        interval = 1.0 / max(1.0, settings.target_fps)
        last_idx, t_prev = -1, time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, frame, idx = self.source.read() if self.source else (False, None, -1)
            if not ok or frame is None or idx == last_idx:
                self._stop.wait(0.01)
                continue
            last_idx = idx
            try:
                self._process(frame.copy())
                self.error = ""
            except Exception as exc:        # one bad frame must not kill a camera
                self.error = f"{exc.__class__.__name__}: {exc}"
            self.processed += 1
            dt = time.perf_counter() - t_prev
            t_prev = time.perf_counter()
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if not self.fps else 0.8 * self.fps + 0.2 * inst
            sleep = interval - (time.perf_counter() - t0)
            if sleep > 0:
                self._stop.wait(sleep)
        if self.source:
            self.source.close()
        self.tracks.tracks.clear()
        db.audit("camera.stop", {"camera": self.camera.id})

    # --------------------------------------------------------------- one frame
    def _process(self, frame) -> None:
        now = time.time()
        h, w = frame.shape[:2]
        self.zones.project(w, h)

        inference_frame, self.is_night, self.luma = self.enhancer.process(frame)
        self.enhanced = inference_frame is not frame

        dets = self.detector(inference_frame)
        attach_carried_objects(dets)
        if dets and self.processed % max(1, settings.pose_stride) == 0:
            self.pose.annotate(inference_frame, dets)

        occupancy = self.zones.occupancy(self.tracks.tracks)
        plate_watch = self._plate_watchlist()
        pending, threat_by_track = [], {}
        for det in dets:
            if det.group == "carried":
                continue
            track = self.tracks.get(det, now)
            if det.group == "person":
                classify(det, track)
                try:
                    features, crop_b64 = bpb_extractor.extract(frame, det.box)
                    assoc = reid_engine.associate_and_track(
                        self.camera.id, det.track_id, det.box, features, crop_b64, w, h
                    )
                    det.attrs.append(assoc["worker_name"])
                except Exception:
                    pass
            hits = self.zones.evaluate(det, track, now)
            in_zone = max((occupancy.get(hit.zone_id, 1) for hit in hits),
                          default=1)
            watch = ""
            if track.plate and track.plate in plate_watch:
                watch = f"plate {track.plate}"
            th = assess(det, track, hits, self.is_night, in_zone, watch)
            threat_by_track[det.track_id] = (th.score, th.level)
            pending.append((det, track, hits, th, watch))

        found_faces = self._run_faces(inference_frame, dets)
        self._run_anpr(inference_frame, dets, pending)

        zone_counts = {z.name: occupancy.get(z.id, 0) for z in self.camera.zones}
        self.summary = scene_summary(dets, zone_counts)
        self.top_threat = max((t for t in threat_by_track.values()),
                              key=lambda t: t[0], default=(0, ""))

        display = frame.copy()
        if settings.blur_faces and found_faces:
            self.faces.blur(display, found_faces)
        self.overlay.build(self.camera.zones, self.zones._pixels, display.shape[:2])
        self.overlay.blend(display)
        viz.draw_detections(display, dets, threat_by_track)
        viz.draw_faces(display, found_faces)
        viz.draw_hud(display, camera_name=self.camera.name,
                     sector=self.camera.sector, fps=self.fps,
                     tracks=self.tracks.active, is_night=self.is_night,
                     enhanced=self.enhanced, luma=self.luma,
                     top_threat=self.top_threat, summary=self.summary,
                     anpr_ready=anpr_worker().ready)

        self.clips.push(display)
        for det, track, hits, th, watch in pending:
            self._maybe_emit(display, det, track, hits, th, watch, now)
        self._publish_frames(display, frame)
        self.tracks.expire(now)

    # -------------------------------------------------------------- sub-stages
    def _publish_frames(self, display, raw) -> None:
        q = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
        scale = min(1.0, settings.stream_max_width / max(1, display.shape[1]))
        if scale < 0.99:
            size = (int(display.shape[1] * scale), int(display.shape[0] * scale))
            display = cv2.resize(display, size)
            raw = cv2.resize(raw, size)
        ok1, buf1 = cv2.imencode(".jpg", display, q)
        ok2, buf2 = cv2.imencode(".jpg", raw, q)
        with self._frame_lock:
            if ok1:
                self._jpeg = buf1.tobytes()
            if ok2:
                self._raw_jpeg = buf2.tobytes()

    def _plate_watchlist(self) -> set[str]:
        ts, vals = getattr(self, "_watch_cache", (0.0, set()))
        if time.time() - ts > 10.0:
            try:
                vals = db.watchlist_values("plate")
            except Exception:
                vals = set()
            self._watch_cache = (time.time(), vals)
        return vals

    def _run_faces(self, frame, dets) -> list:
        if not self.faces or not self.faces.available:
            return []
        if "face" not in self.camera.analytics:
            return []
        if self.processed % max(1, settings.face_stride):
            return getattr(self, "_last_faces", [])
        boxes = [d.box for d in dets if d.group == "person"]
        found = self.faces.detect_in(frame, boxes)
        if settings.face_watchlist_enabled and self.faces.recognizer:
            for f in found:
                emb = self.faces.embed(frame, f)
                if emb is not None:
                    f.label, f.match_score = self.faces.match(emb)
        self._last_faces = found
        return found

    def _run_anpr(self, frame, dets, pending) -> None:
        worker = anpr_worker()
        for tid, plate, conf in worker.drain():
            st = self.tracks.tracks.get(tid)
            if st is None or conf <= st.plate_conf:
                continue
            st.plate, st.plate_conf = plate, conf
            hit = plate in self._plate_watchlist()
            db.insert_plate({"camera_id": self.camera.id, "plate": plate,
                             "confidence": round(conf, 3),
                             "vehicle_class": st.name,
                             "watchlist_hit": 1 if hit else 0})
            bus.publish({"type": "plate", "camera_id": self.camera.id,
                         "camera_name": self.camera.name, "plate": plate,
                         "confidence": round(conf, 3), "valid": is_valid(plate),
                         "watchlist_hit": hit, "ts": time.time()})
        if "anpr" not in self.camera.analytics or not worker.ready:
            return
        if self.processed % max(1, settings.anpr_stride):
            return
        for d in dets:
            if d.group != "vehicle":
                continue
            st = self.tracks.tracks.get(d.track_id)
            if st is not None and st.plate_conf >= 0.75:
                continue
            if worker.submit(d.track_id, frame, d):
                break               # at most one crop per stride

    # ------------------------------------------------------------------ events
    def _maybe_emit(self, display, det, track, hits, th, watch, now) -> None:
        """Alert policy. Each rule has its own cooldown key so a zone breach and
        a suspicious gait do not suppress one another, but neither repeats every
        frame while the target stays in view."""
        fired = False
        for hit in hits:
            key = f"{hit.kind}:{hit.zone_id}"
            if track.cooled_down(key, now):
                track.mark_event(key, now)
                self._emit(display, det, track, hits, th, hit.kind, now, hit)
                fired = True
        if not fired and det.action in SUSPICIOUS and det.action_conf >= 0.5:
            if track.cooled_down(f"act:{det.action}", now):
                track.mark_event(f"act:{det.action}", now)
                self._emit(display, det, track, hits, th, "activity", now)
                fired = True
        if watch and track.cooled_down("watchlist", now, 30.0):
            track.mark_event("watchlist", now)
            self._emit(display, det, track, hits, th, "watchlist", now)
            fired = True
        if (not fired and self.is_night and det.group in ("person", "vehicle")
                and track.cooled_down("night", now, 45.0)):
            track.mark_event("night", now)
            self._emit(display, det, track, hits, th, "night_motion", now)

    def _emit(self, display, det, track, hits, th, kind, now, zone=None) -> None:
        snapshot = save_snapshot(display, self.camera.id, now)
        clip = self.clips.start(self.camera.id, now) if th.score >= 60 else None
        text = caption(det, track, hits, self.camera.name, self.camera.sector,
                       self.is_night, now, self.enhanced)
        event = {
            "ts": now, "camera_id": self.camera.id,
            "camera_name": self.camera.name, "sector": self.camera.sector,
            "kind": kind, "object_class": det.name, "track_id": det.track_id,
            "action": det.action if det.group == "person" else None,
            "zone_id": zone.zone_id if zone else None,
            "zone_name": zone.zone_name if zone else None,
            "threat_score": th.score, "threat_level": th.level,
            "caption": text, "snapshot": snapshot, "clip": clip,
            "is_night": 1 if self.is_night else 0,
            "meta": {
                "threat_breakdown": th.contributions,
                "detection": det.as_dict(),
                "zone_hits": [h.as_dict() for h in hits],
                "plate": track.plate, "plate_confidence": track.plate_conf,
                "enhanced": self.enhanced, "luma": round(self.luma, 1),
                "caption_source": "rule-based-template",
                "track_age_s": round(track.age, 1),
                "speed_body_heights_s": round(track.speed_body_heights(), 2),
                "enhancer": self.enhancer.backend.name if self.enhancer else None,
            },
        }
        event["id"] = db.insert_event(event)
        event["iso"] = db.iso_now(now)
        bus.publish({"type": "event", **event})
        if th.score >= 70:
            record = self.ptz.slew(self.camera.id, self.camera.ptz, det.box,
                                   display.shape[:2], reason=kind)
            if record:
                bus.publish({"type": "ptz", **record})
        if th.score >= 60:
            post_to_c2(event)
