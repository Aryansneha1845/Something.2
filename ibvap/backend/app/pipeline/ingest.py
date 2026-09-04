"""Video ingest: RTSP / file / webcam, normalised behind one interface.

Two behaviours matter for a surveillance workload and are easy to get wrong:

* RTSP  — OpenCV buffers frames internally, so a slow consumer drifts further
          and further behind real time. We run a grabber thread that keeps only
          the newest frame; the analytics loop is therefore always looking at
          "now" and simply skips whatever it could not keep up with.
* files — used for demos and for replaying recorded incidents. Here we do the
          opposite and pace to the file's native FPS so motion looks correct,
          optionally looping forever.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Force TCP for RTSP (UDP loses packets over long BOP microwave links) and cap
# the socket timeout. Must be set before the first VideoCapture is created.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


class VideoSource:
    """A resilient frame source. Thread-safe: call read() from one consumer."""

    def __init__(self, source: str, loop: bool = True, name: str = "src") -> None:
        self.source_str = source
        self.name = name
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_idx = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reconnects = 0
        self._last_error = ""
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.is_live = not self._is_file(source)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _is_file(source: str) -> bool:
        return not source.startswith(("rtsp://", "http://", "https://")) \
            and not source.isdigit()

    def _open(self) -> bool:
        src: str | int = self.source_str
        api = cv2.CAP_ANY
        if self.source_str.isdigit():
            src = int(self.source_str)
            if sys.platform == "win32":
                api = cv2.CAP_DSHOW          # avoids ~2s MSMF startup stall
        elif self.source_str.startswith("rtsp://"):
            api = cv2.CAP_FFMPEG
        elif self._is_file(self.source_str):
            p = Path(self.source_str)
            if not p.exists():
                from ..config import MEDIA_DIR, PROJECT_DIR
                candidates = [
                    MEDIA_DIR / p.name,
                    PROJECT_DIR / self.source_str,
                    MEDIA_DIR / self.source_str,
                ]
                for c in candidates:
                    if c.exists():
                        src = str(c)
                        break
            else:
                src = str(p.resolve())
        cap = cv2.VideoCapture(src, api)
        if not cap.isOpened():
            self._last_error = f"cannot open source {self.source_str!r}"
            cap.release()
            return False
        if self.is_live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and 1.0 < fps < 121.0 else 25.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self._cap = cap
        self._last_error = ""
        return True

    # ------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        if not self._open():
            return False
        self._thread = threading.Thread(
            target=self._grab_loop, name=f"grab-{self.name}", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------ grab thread
    def _grab_loop(self) -> None:
        frame_interval = 1.0 / self.fps
        idx = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                if not self._reconnect():
                    break
                continue
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                if not self.is_live and self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if not self._reconnect():
                    break
                continue
            idx += 1
            with self._lock:
                self._latest = frame
                self._latest_idx = idx
            if not self.is_live:
                # pace file playback to its native frame rate
                sleep = frame_interval - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        with self._lock:
            self._latest = None

    def _reconnect(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        backoff = min(1.0 + self._reconnects * 1.5, 15.0)
        self._reconnects += 1
        self._last_error = f"reconnecting in {backoff:.0f}s (attempt {self._reconnects})"
        if self._stop.wait(backoff):
            return False
        return self._open()

    # ------------------------------------------------------------------ read
    def read(self) -> tuple[bool, np.ndarray | None, int]:
        """Return the newest available frame. Never blocks on decode."""
        with self._lock:
            if self._latest is None:
                return False, None, -1
            return True, self._latest, self._latest_idx

    @property
    def status(self) -> dict:
        return {
            "source": self.source_str, "live": self.is_live,
            "fps": round(self.fps, 1), "width": self.width,
            "height": self.height, "reconnects": self._reconnects,
            "error": self._last_error,
            "connected": self._cap is not None and self._latest is not None,
        }
