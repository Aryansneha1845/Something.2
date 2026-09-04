"""Camera manager coordinating workers, live MJPEG streams, and config persistence."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator, Generator

from .config import Camera, _camera_from_dict, load_cameras, save_cameras
from .pipeline.worker import CameraWorker

logger = logging.getLogger("ibvap.manager")


class CameraManager:
    """Manages all active CameraWorker threads and camera registry persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, CameraWorker] = {}
        self._cameras: dict[str, Camera] = {}

    def initialize(self) -> None:
        """Load cameras from disk and boot workers for enabled cameras."""
        with self._lock:
            cams = load_cameras()
            self._cameras = {c.id: c for c in cams}
            for cam in cams:
                if cam.enabled:
                    self._start_worker_locked(cam)

    def _start_worker_locked(self, camera: Camera) -> CameraWorker:
        if camera.id in self._workers:
            old = self._workers[camera.id]
            if old.is_alive():
                old.stop()
        worker = CameraWorker(camera)
        worker.start()
        self._workers[camera.id] = worker
        logger.info("Started analytics worker for camera %s (%s)", camera.id, camera.name)
        return worker

    def get_worker(self, camera_id: str) -> CameraWorker | None:
        with self._lock:
            return self._workers.get(camera_id)

    def get_camera(self, camera_id: str) -> Camera | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def list_cameras(self) -> list[dict]:
        """Return combined configuration and live worker status for all cameras."""
        with self._lock:
            result = []
            for cid, cam in self._cameras.items():
                worker = self._workers.get(cid)
                status = worker.status if worker else {
                    "id": cam.id,
                    "name": cam.name,
                    "sector": cam.sector,
                    "enabled": cam.enabled,
                    "running": False,
                    "fps": 0.0,
                    "frames": 0,
                    "is_night": False,
                    "enhanced": False,
                    "luma": 0.0,
                    "tracks": 0,
                    "top_threat": {"score": 0, "level": "clear"},
                    "summary": "Offline" if not cam.enabled else "Stopped",
                    "error": "",
                    "uptime": 0,
                    "source": {"source": cam.source, "connected": False},
                    "zone_occupancy": {},
                    "ptz_history": [],
                }
                item = {
                    "id": cam.id,
                    "name": cam.name,
                    "source": cam.source,
                    "sector": cam.sector,
                    "enabled": cam.enabled,
                    "loop": cam.loop,
                    "map_x": cam.map_x,
                    "map_y": cam.map_y,
                    "zones": [z.__dict__ for z in cam.zones],
                    "ptz": cam.ptz.__dict__,
                    "analytics": cam.analytics,
                    "status": status,
                }
                result.append(item)
            return result

    def add_or_update_camera(self, data: dict) -> dict:
        """Create or update a camera definition, persist to disk, and sync worker."""
        with self._lock:
            cam = _camera_from_dict(dict(data))
            self._cameras[cam.id] = cam
            save_cameras(list(self._cameras.values()))

            if cam.enabled:
                self._start_worker_locked(cam)
            else:
                if cam.id in self._workers:
                    self._workers[cam.id].stop()
                    del self._workers[cam.id]

            return {
                "id": cam.id,
                "name": cam.name,
                "sector": cam.sector,
                "enabled": cam.enabled,
            }

    def delete_camera(self, camera_id: str) -> bool:
        """Delete camera config and terminate worker."""
        with self._lock:
            if camera_id not in self._cameras:
                return False
            if camera_id in self._workers:
                self._workers[camera_id].stop()
                del self._workers[camera_id]
            del self._cameras[camera_id]
            save_cameras(list(self._cameras.values()))
            return True

    def start_camera(self, camera_id: str) -> bool:
        with self._lock:
            cam = self._cameras.get(camera_id)
            if not cam:
                return False
            cam.enabled = True
            save_cameras(list(self._cameras.values()))
            self._start_worker_locked(cam)
            return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            cam = self._cameras.get(camera_id)
            if not cam:
                return False
            cam.enabled = False
            save_cameras(list(self._cameras.values()))
            worker = self._workers.get(camera_id)
            if worker:
                worker.stop()
                del self._workers[camera_id]
            return True

    def shutdown(self) -> None:
        """Stop all workers gracefully."""
        with self._lock:
            for cid, w in self._workers.items():
                try:
                    w.stop()
                except Exception:
                    pass
            self._workers.clear()

    def get_standby_jpeg(self, label: str = "CAMERA INITIALIZING...") -> bytes:
        import cv2
        import numpy as np
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (20, 25, 30)
        cv2.rectangle(img, (10, 10), (630, 350), (40, 55, 70), 2)
        cv2.putText(img, "IBVAP TACTICAL SURVEILLANCE", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 1)
        cv2.putText(img, label, (190, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 180, 200), 1)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    async def stream_mjpeg(self, camera_id: str, raw: bool = False, max_fps: float = 8.0) -> AsyncGenerator[bytes, None]:
        """Yield multipart MJPEG stream frames asynchronously without blocking the event loop."""
        interval = 1.0 / max(1.0, max_fps)
        standby_sent = False
        while True:
            worker = self.get_worker(camera_id)
            if worker and worker.is_alive():
                frame = worker.frame_jpeg(raw=raw)
                if frame is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                        frame + b"\r\n"
                    )
                elif frame is None and not standby_sent:
                    standby = self.get_standby_jpeg(f"{camera_id}: INITIALIZING AI PIPELINE...")
                    standby_sent = True
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(standby)).encode() + b"\r\n\r\n" +
                        standby + b"\r\n"
                    )
            else:
                standby = self.get_standby_jpeg(f"{camera_id}: STANDBY")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(standby)).encode() + b"\r\n\r\n" +
                    standby + b"\r\n"
                )
            await asyncio.sleep(interval)


manager = CameraManager()
