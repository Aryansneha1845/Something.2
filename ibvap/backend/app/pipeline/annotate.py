"""Frame annotation — the operator's view of what the pipeline decided.

The zone overlay is rasterised once per frame size and cached, then alpha-blended
each frame: redrawing translucent polygons per frame costs a full-frame
addWeighted, which is measurable on a 4-core box.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import Zone
from .detect import Detection
from .faces import Face
from .pose import KP_MIN_CONF, SKELETON

GROUP_COLOR = {"person": (0, 200, 255), "vehicle": (255, 175, 0),
               "animal": (130, 205, 130), "carried": (200, 200, 200),
               "other": (180, 180, 180)}
LEVEL_COLOR = {"CRITICAL": (60, 60, 255), "HIGH": (0, 140, 255),
               "MEDIUM": (0, 215, 255), "LOW": (130, 220, 130)}
SEVERITY_COLOR = {5: (60, 60, 255), 4: (0, 120, 255), 3: (0, 200, 255),
                  2: (0, 220, 180), 1: (140, 220, 140)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _dashed_line(img, p1, p2, color, thickness=2, dash=14) -> None:
    dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if dist == 0:
        return
    for i in range(0, dist, dash * 2):
        a = i / dist
        b = min(1.0, (i + dash) / dist)
        pa = (int(p1[0] + (p2[0] - p1[0]) * a), int(p1[1] + (p2[1] - p1[1]) * a))
        pb = (int(p1[0] + (p2[0] - p1[0]) * b), int(p1[1] + (p2[1] - p1[1]) * b))
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def _label(img, text, org, color, scale=0.45, pad=4) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    y = max(th + pad, y)
    cv2.rectangle(img, (x, y - th - pad), (x + tw + pad * 2, y + base),
                  color, -1)
    cv2.putText(img, text, (x + pad, y - 1), FONT, scale, (15, 15, 20), 1,
                cv2.LINE_AA)


class ZoneOverlayCache:
    def __init__(self) -> None:
        self._key: tuple = ()
        self._layer: np.ndarray | None = None
        self._labels: list[tuple[str, tuple[int, int], tuple] ] = []

    def build(self, zones: list[Zone], pixels: dict[str, list],
              shape: tuple[int, int]) -> None:
        key = (shape, tuple((z.id, z.severity, tuple(map(tuple, pixels.get(z.id, ()))))
                            for z in zones))
        if key == self._key and self._layer is not None:
            return
        layer = np.zeros((shape[0], shape[1], 3), np.uint8)
        labels = []
        for z in zones:
            pts = pixels.get(z.id) or []
            color = SEVERITY_COLOR.get(z.severity, (0, 200, 255))
            if z.kind == "line" and len(pts) >= 2:
                _dashed_line(layer, pts[0], pts[1], color, 3)
                labels.append((f"{z.name} [S{z.severity}]", pts[0], color))
            elif len(pts) >= 3:
                arr = np.array(pts, np.int32)
                cv2.fillPoly(layer, [arr], tuple(int(c * 0.35) for c in color))
                cv2.polylines(layer, [arr], True, color, 2, cv2.LINE_AA)
                top = min(pts, key=lambda p: p[1])
                labels.append((f"{z.name} [S{z.severity}]", top, color))
        self._key, self._layer, self._labels = key, layer, labels

    def blend(self, frame: np.ndarray) -> None:
        if self._layer is None or self._layer.shape != frame.shape:
            return
        mask = self._layer.any(axis=2)
        frame[mask] = cv2.addWeighted(frame, 0.55, self._layer, 0.45, 0)[mask]
        for text, org, color in self._labels:
            _label(frame, text, (org[0] + 4, org[1] - 6), color, 0.42)


def draw_pose(frame: np.ndarray, kps: np.ndarray, color=(0, 255, 180)) -> None:
    for a, b in SKELETON:
        if kps[a][2] >= KP_MIN_CONF and kps[b][2] >= KP_MIN_CONF:
            cv2.line(frame, (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])), color, 2, cv2.LINE_AA)
    for x, y, c in kps:
        if c >= KP_MIN_CONF:
            cv2.circle(frame, (int(x), int(y)), 3, (40, 40, 255), -1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, dets: list[Detection],
                    threat_by_track: dict[int, tuple[int, str]],
                    show_pose: bool = True) -> None:
    for d in dets:
        if d.group == "carried":
            continue
        score, level = threat_by_track.get(d.track_id, (0, ""))
        color = LEVEL_COLOR[level] if level else GROUP_COLOR.get(d.group,
                                                                (180, 180, 180))
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        bits = [f"{d.name} #{d.track_id}"]
        if d.group == "person" and d.action not in ("unknown", "n/a"):
            bits.append(f"{d.action} {d.action_conf:.2f}")
        if level:
            bits.append(f"{level} {score}")
        _label(frame, " | ".join(bits), (x1, y1 - 4), color)
        for a in d.attrs:
            if a.startswith("carrying:"):
                _label(frame, a.replace("carrying:", "carrying "),
                       (x1, y2 + 16), (200, 200, 200), 0.4)
        if show_pose and d.keypoints is not None:
            draw_pose(frame, d.keypoints)


def draw_faces(frame: np.ndarray, faces: list[Face]) -> None:
    for f in faces:
        color = (60, 60, 255) if f.label else (230, 230, 120)
        cv2.rectangle(frame, (f.box[0], f.box[1]), (f.box[2], f.box[3]),
                      color, 1)
        if f.label:
            _label(frame, f"WATCHLIST: {f.label} {f.match_score:.2f}",
                   (f.box[0], f.box[1] - 3), color, 0.4)


def draw_hud(frame: np.ndarray, *, camera_name: str, sector: str, fps: float,
             tracks: int, is_night: bool, enhanced: bool, luma: float,
             top_threat: tuple[int, str] | None, summary: str,
             anpr_ready: bool) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (22, 24, 30), -1)
    cv2.rectangle(frame, (0, h - 26), (w, h), (22, 24, 30), -1)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"{sector}  |  {camera_name}", (10, 22), FONT, 0.55,
                (235, 235, 240), 1, cv2.LINE_AA)
    right = f"{stamp}   {fps:4.1f} fps   tracks:{tracks}"
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.48, 1)
    cv2.putText(frame, right, (w - tw - 10, 22), FONT, 0.48,
                (190, 195, 205), 1, cv2.LINE_AA)

    badges: list[tuple[str, tuple[int, int, int]]] = []
    if is_night:
        badges.append((f"NIGHT {luma:.0f}", (255, 190, 90)))
    if enhanced:
        badges.append(("ENHANCED", (150, 230, 150)))
    if not anpr_ready:
        badges.append(("ANPR LOADING", (140, 140, 150)))
    bx = 10
    for text, color in badges:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.4, 1)
        cv2.rectangle(frame, (bx, 40), (bx + tw + 10, 40 + th + 8), color, -1)
        cv2.putText(frame, text, (bx + 5, 40 + th + 2), FONT, 0.4,
                    (20, 20, 25), 1, cv2.LINE_AA)
        bx += tw + 16

    if top_threat and top_threat[0] >= 30:
        score, level = top_threat
        color = LEVEL_COLOR.get(level, (0, 200, 255))
        bw, bh_ = 190, 46
        x0, y0 = w - bw - 10, 42
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh_), (18, 20, 26), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh_), color, 2)
        cv2.putText(frame, f"THREAT {level}", (x0 + 10, y0 + 18), FONT, 0.5,
                    color, 1, cv2.LINE_AA)
        fill = int((bw - 20) * min(100, score) / 100)
        cv2.rectangle(frame, (x0 + 10, y0 + 26), (x0 + 10 + fill, y0 + 36),
                      color, -1)
        cv2.putText(frame, f"{score}", (x0 + bw - 40, y0 + 36), FONT, 0.45,
                    (235, 235, 240), 1, cv2.LINE_AA)

    if summary:
        text = summary if len(summary) < 120 else summary[:117] + "..."
        cv2.putText(frame, text, (10, h - 8), FONT, 0.44, (205, 210, 220), 1,
                    cv2.LINE_AA)
