"""Action recognition from pose geometry + track motion.

Why not ST-GCN / a trained skeleton-action model: the classes that matter for
this problem ("crawling towards the fence", "climbing it") have no off-the-shelf
labelled dataset, a skeleton-GCN needs one, and there is no GPU here. Instead we
score a small set of scale-invariant geometric features that separate exactly
those classes, and expose every feature in the event payload so an operator can
see *why* a call was made. Thresholds live in `THRESHOLDS` and are tunable at
runtime.

All features are normalised by torso length or bounding-box height, so a target
20 m away scores the same as one at 200 m.
"""
from __future__ import annotations

import math

import numpy as np

from .detect import Detection
from .pose import (L_ANK, L_HIP, L_KNE, L_SHO, L_WRI, R_ANK, R_HIP, R_KNE,
                   R_SHO, R_WRI, kp, mean_point, visible_count)
from .tracks import TrackState

ACTIONS = ("standing", "walking", "running", "crouching", "crawling",
           "climbing", "fallen", "unknown")

# Actions that should raise an alert on their own merit, with severity weight.
SUSPICIOUS: dict[str, int] = {
    "crawling": 30, "climbing": 35, "fallen": 20, "running": 15,
}

THRESHOLDS: dict[str, float] = {
    "tilt_crawl": 55.0,        # deg from vertical: torso pitched over
    "tilt_fallen": 68.0,
    "aspect_prone": 1.05,      # bbox w/h above this => body is horizontal
    "speed_walk": 0.35,        # body-heights / second
    "speed_run": 1.55,
    "vert_climb": 0.12,        # body-heights / second of upward motion
    "extent_stand": 2.0,       # (ankle_y - shoulder_y) / torso_len
    "extent_crouch": 1.55,
    "extent_crawl": 1.05,
    "height_ratio_low": 0.68,  # bbox height vs this track's own standing max
}


def _tilt_degrees(shoulder: tuple[float, float],
                  hip: tuple[float, float]) -> float:
    """Angle of the torso away from vertical, 0 = upright, 90 = horizontal."""
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -dy / length))))


def geometry(det: Detection, track: TrackState) -> dict[str, float]:
    """Scale-invariant descriptors. Missing values are reported as NaN so the
    scorer can tell 'not upright' apart from 'not measurable'."""
    nan = float("nan")
    f: dict[str, float] = {
        "aspect": det.width / max(1.0, det.height),
        "speed": track.speed_body_heights(1.0),
        "vertical_rate": track.vertical_rate(1.5),
        "height_ratio": det.height / max(1.0, track.max_box_height),
        "tilt": nan, "extent": nan, "knee_lift": nan, "wrist_lift": nan,
        "kp_visible": 0.0,
    }
    kps = det.keypoints
    if kps is None:
        return f
    f["kp_visible"] = float(visible_count(kps))
    shoulder = mean_point(kps, (L_SHO, R_SHO))
    hip = mean_point(kps, (L_HIP, R_HIP))
    if shoulder is None or hip is None:
        return f
    torso = max(1.0, math.dist(shoulder, hip))
    f["tilt"] = _tilt_degrees(shoulder, hip)
    ankle = mean_point(kps, (L_ANK, R_ANK))
    if ankle is not None:
        f["extent"] = (ankle[1] - shoulder[1]) / torso
    knee = mean_point(kps, (L_KNE, R_KNE))
    if knee is not None:
        # positive => knee lifted above the hip line (rungs / fence footholds)
        f["knee_lift"] = (hip[1] - knee[1]) / torso
    wrist = mean_point(kps, (L_WRI, R_WRI))
    if wrist is not None:
        # positive => hands raised above the shoulders (gripping / hauling up)
        f["wrist_lift"] = (shoulder[1] - wrist[1]) / torso
    return f


def _ok(v: float) -> bool:
    return not (v is None or math.isnan(v))


def _ramp(v: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear in between."""
    if hi <= lo:
        return 1.0 if v >= hi else 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def score_actions(f: dict[str, float]) -> dict[str, float]:
    """Soft score per action in 0..1. Deliberately overlapping — the caller
    takes the argmax, and the margin becomes the confidence."""
    T = THRESHOLDS
    s = dict.fromkeys(ACTIONS, 0.0)
    has_pose = _ok(f["tilt"])
    speed, aspect = f["speed"], f["aspect"]

    # --- body-is-horizontal family -----------------------------------------
    prone = 0.30 * _ramp(aspect, T["aspect_prone"], 1.9)
    prone += 0.15 * (1.0 - _ramp(f["height_ratio"], 0.45, T["height_ratio_low"]))
    if has_pose:
        prone += 0.55 * _ramp(f["tilt"], T["tilt_crawl"], 85.0)
    if _ok(f["extent"]):
        prone += 0.25 * (1.0 - _ramp(f["extent"], 0.5, T["extent_crawl"]))
    prone = min(1.0, prone)

    moving = _ramp(speed, 0.08, 0.45)
    # Crawling is prone *and still making ground*; fallen is prone and static.
    s["crawling"] = prone * (0.45 + 0.55 * moving)
    s["fallen"] = prone * (1.0 - moving) * (0.6 + 0.4 * _ramp(aspect, 1.2, 2.0))

    # --- climbing -----------------------------------------------------------
    climb = 0.45 * _ramp(f["vertical_rate"], T["vert_climb"], 0.5)
    if _ok(f["wrist_lift"]):
        climb += 0.30 * _ramp(f["wrist_lift"], 0.05, 0.80)
    if _ok(f["knee_lift"]):
        climb += 0.25 * _ramp(f["knee_lift"], 0.10, 0.70)
    if has_pose:                      # a climber stays broadly upright
        climb *= 1.0 - 0.6 * _ramp(f["tilt"], T["tilt_crawl"], 85.0)
    s["climbing"] = min(1.0, climb)

    # --- upright family -----------------------------------------------------
    upright = 1.0 - prone
    if _ok(f["extent"]):
        crouch = 1.0 - _ramp(f["extent"], T["extent_crouch"], T["extent_stand"])
    else:
        crouch = 1.0 - _ramp(f["height_ratio"], T["height_ratio_low"], 0.92)
    s["running"] = upright * _ramp(speed, T["speed_walk"] * 2, T["speed_run"])
    s["walking"] = upright * (1.0 - s["running"]) * _ramp(
        speed, T["speed_walk"] * 0.6, T["speed_walk"] * 2.2)
    s["crouching"] = upright * crouch * (1.0 - 0.5 * moving)
    s["standing"] = upright * (1.0 - crouch) * (
        1.0 - _ramp(speed, 0.05, T["speed_walk"]))
    return s


def classify(det: Detection, track: TrackState) -> tuple[str, float, dict]:
    """Classify one person detection, smoothed over the track's recent frames."""
    if det.group != "person":
        return "n/a", 0.0, {}
    f = geometry(det, track)
    scores = score_actions(f)
    label = max(scores, key=lambda k: scores[k])
    top = scores[label]
    if top < 0.22 or len(track.history) < 2:
        label, top = "unknown", 0.0
    # A pose-less call is a weaker call; say so rather than hiding it.
    if not _ok(f["tilt"]) and label in ("crawling", "climbing", "fallen"):
        top *= 0.7
    smoothed, conf = track.vote_action(label, top)
    det.action, det.action_conf = smoothed, conf
    f["scores"] = {k: round(v, 3) for k, v in scores.items() if v > 0.05}
    f["raw_label"] = label
    return smoothed, conf, f


def describe(action: str) -> str:
    return {
        "crawling": "crawling", "climbing": "climbing",
        "fallen": "lying motionless", "running": "running",
        "walking": "walking", "standing": "standing still",
        "crouching": "crouching", "unknown": "moving",
    }.get(action, action)
