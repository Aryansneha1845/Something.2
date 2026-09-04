"""Pose estimation (YOLO11-pose) and keypoint bookkeeping.

Runs a second pass over the frame only when the detector found people, then
matches the pose model's person boxes back onto the *tracked* detections by IoU
so keypoints inherit a stable track id.
"""
from __future__ import annotations

import numpy as np

from ..config import settings
from .detect import Detection

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

SKELETON: tuple[tuple[int, int], ...] = (
    (L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB),
    (R_ELB, R_WRI), (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK),
    (NOSE, L_SHO), (NOSE, R_SHO), (NOSE, L_EYE), (NOSE, R_EYE),
    (L_EYE, L_EAR), (R_EYE, R_EAR),
)

KP_MIN_CONF = 0.35


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class PoseEstimator:
    def __init__(self, weights: str | None = None) -> None:
        from ultralytics import YOLO
        self.model = YOLO(weights or settings.pose_model, task="pose")

    def annotate(self, frame: np.ndarray, dets: list[Detection]) -> int:
        """Attach `keypoints` to every person detection we can match.

        Returns the number of detections that received keypoints."""
        people = [d for d in dets if d.group == "person"]
        if not people:
            return 0
        res = self.model.predict(
            frame, verbose=False, imgsz=settings.imgsz,
            conf=max(0.25, settings.conf - 0.1), device=settings.device,
        )
        if not res or res[0].keypoints is None or res[0].boxes is None:
            return 0
        kps = res[0].keypoints.data.cpu().numpy()          # (n, 17, 3)
        pboxes = res[0].boxes.xyxy.cpu().numpy()
        used: set[int] = set()
        matched = 0
        for det in people:
            best, best_iou = -1, 0.3
            for i, pb in enumerate(pboxes):
                if i in used:
                    continue
                score = _iou(det.box, (int(pb[0]), int(pb[1]),
                                       int(pb[2]), int(pb[3])))
                if score > best_iou:
                    best, best_iou = i, score
            if best >= 0:
                used.add(best)
                det.keypoints = kps[best]
                matched += 1
        return matched


def kp(points: np.ndarray, idx: int) -> tuple[float, float, float] | None:
    """Return (x, y, conf) for a keypoint, or None if it is not confident."""
    if points is None or idx >= len(points):
        return None
    x, y, c = float(points[idx][0]), float(points[idx][1]), float(points[idx][2])
    return (x, y, c) if c >= KP_MIN_CONF else None


def mean_point(points: np.ndarray, idxs: tuple[int, ...]) -> tuple[float, float] | None:
    vals = [kp(points, i) for i in idxs]
    good = [v for v in vals if v is not None]
    if not good:
        return None
    return (sum(v[0] for v in good) / len(good),
            sum(v[1] for v in good) / len(good))


def visible_count(points: np.ndarray) -> int:
    if points is None:
        return 0
    return int((points[:, 2] >= KP_MIN_CONF).sum())
