"""Threat scoring.

A single 0-100 number that fuses zone severity, action class, time of day and
watchlist status, returned alongside the itemised contributions that produced
it. The breakdown is not decoration: an operator who cannot see why a score is
85 will stop trusting the score, and a reviewer needs it to audit a call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .actions import SUSPICIOUS
from .detect import Detection
from .tracks import TrackState
from .zones import ZoneHit

LEVELS = (("CRITICAL", 80), ("HIGH", 60), ("MEDIUM", 30), ("LOW", 0))

BASE_BY_GROUP = {"person": 25, "vehicle": 20, "animal": 5, "other": 5}
KIND_WEIGHT = {"intrusion": 15, "crossing": 20, "loiter": 8}


@dataclass
class Threat:
    score: int = 0
    level: str = "LOW"
    contributions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"score": self.score, "level": self.level,
                "contributions": self.contributions}


def level_for(score: int) -> str:
    for name, floor in LEVELS:
        if score >= floor:
            return name
    return "LOW"


def assess(det: Detection, track: TrackState, hits: list[ZoneHit],
           is_night: bool = False, zone_occupancy: int = 1,
           watchlist_hit: str = "") -> Threat:
    parts: list[dict] = []

    def add(label: str, points: int) -> None:
        if points:
            parts.append({"factor": label, "points": int(points)})

    add(f"{det.group} detected", BASE_BY_GROUP.get(det.group, 5))

    if hits:
        worst = max(hits, key=lambda h: h.severity * 10 + KIND_WEIGHT.get(h.kind, 0))
        add(f"zone '{worst.zone_name}' severity {worst.severity}",
            min(40, worst.severity * 8))
        add(f"{worst.kind} event", KIND_WEIGHT.get(worst.kind, 0))
        if worst.kind == "loiter" and worst.dwell:
            add(f"loitering {worst.dwell:.0f}s", min(10, int(worst.dwell / 3)))

    if det.action in SUSPICIOUS:
        # Weight the action by how confidently it was called.
        add(f"action: {det.action}",
            int(SUSPICIOUS[det.action] * max(0.4, det.action_conf)))

    if is_night:
        add("night-time movement", 12)
    if any(a.startswith("carrying:") for a in det.attrs):
        add("carrying an object", 5)
    if zone_occupancy > 1:
        add(f"{zone_occupancy} targets in zone", min(15, 5 * (zone_occupancy - 1)))
    if watchlist_hit:
        add(f"watchlist match ({watchlist_hit})", 30)
    if track.age > 30 and track.net_displacement(30) > 6:
        add("sustained track movement", 5)

    score = max(0, min(100, sum(p["points"] for p in parts)))
    track.peak_threat = max(track.peak_threat, score)
    return Threat(score=score, level=level_for(score), contributions=parts)
