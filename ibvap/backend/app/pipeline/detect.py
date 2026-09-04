"""Object detection + multi-object tracking (YOLO11 + ByteTrack).

One `Detector` instance per camera: ultralytics keeps tracker state inside the
model object, so sharing one instance across camera threads would splice tracks
from different scenes together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import settings

# COCO class -> semantic group used by the rest of the pipeline.
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "train"}
ANIMAL_CLASSES = {"dog", "cat", "horse", "sheep", "cow", "elephant", "bear",
                  "zebra", "giraffe", "bird"}
CARRIED_CLASSES = {"backpack", "handbag", "suitcase", "umbrella", "bottle",
                   "cell phone", "laptop"}

GROUP_OF: dict[str, str] = {}
for _s, _g in ((PERSON_CLASSES, "person"), (VEHICLE_CLASSES, "vehicle"),
               (ANIMAL_CLASSES, "animal"), (CARRIED_CLASSES, "carried")):
    GROUP_OF.update({c: _g for c in _s})

KEEP_CLASSES = set(GROUP_OF)


@dataclass
class Detection:
    track_id: int
    name: str
    group: str
    conf: float
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in frame pixels
    keypoints: np.ndarray | None = None      # (17, 3) x, y, conf — filled by pose
    action: str = "unknown"
    action_conf: float = 0.0
    attrs: list[str] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot(self) -> tuple[float, float]:
        """Ground-contact point — the right anchor for zone tests, since a
        person's bounding-box centre can sit outside a fence line they have
        not actually crossed."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, float(y2))

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_dict(self) -> dict:
        return {"track_id": self.track_id, "class": self.name,
                "group": self.group, "conf": round(self.conf, 3),
                "box": list(self.box), "action": self.action,
                "action_conf": round(self.action_conf, 2), "attrs": self.attrs}


class Detector:
    def __init__(self, weights: str | None = None) -> None:
        from ultralytics import YOLO
        self.model = YOLO(weights or settings.detect_model, task="detect")
        self.names: dict[int, str] = self.model.names
        self._keep_ids = [i for i, n in self.names.items() if n in KEEP_CLASSES]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.track(
            frame, persist=True, verbose=False, tracker="bytetrack.yaml",
            imgsz=settings.imgsz, conf=settings.conf, iou=settings.iou,
            device=settings.device, classes=self._keep_ids,
        )
        if not res:
            return []
        boxes = res[0].boxes
        if boxes is None or boxes.id is None:
            return []
        out: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        h, w = frame.shape[:2]
        for box, tid, cid, cf in zip(xyxy, ids, clss, confs):
            name = self.names.get(int(cid), str(cid))
            x1 = int(max(0, min(box[0], w - 1)))
            y1 = int(max(0, min(box[1], h - 1)))
            x2 = int(max(0, min(box[2], w - 1)))
            y2 = int(max(0, min(box[3], h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(
                track_id=int(tid), name=name,
                group=GROUP_OF.get(name, "other"),
                conf=float(cf), box=(x1, y1, x2, y2),
            ))
        return out

    def reset(self) -> None:
        """Drop tracker state (used when a source reconnects)."""
        pred = getattr(self.model, "predictor", None)
        for tracker in (getattr(pred, "trackers", None) or []):
            try:
                tracker.reset()
            except Exception:
                pass


def attach_carried_objects(dets: list[Detection]) -> None:
    """Mark people whose box substantially contains a carried-object box.

    Cheap stand-in for a dedicated 'carrying' classifier, and it reads well in
    an alert caption ('person carrying a backpack')."""
    people = [d for d in dets if d.group == "person"]
    items = [d for d in dets if d.group == "carried"]
    for p in people:
        px1, py1, px2, py2 = p.box
        for it in items:
            ix1, iy1, ix2, iy2 = it.box
            ox = max(0, min(px2, ix2) - max(px1, ix1))
            oy = max(0, min(py2, iy2) - max(py1, iy1))
            if it.area and (ox * oy) / it.area > 0.6:
                tag = f"carrying:{it.name}"
                if tag not in p.attrs:
                    p.attrs.append(tag)
