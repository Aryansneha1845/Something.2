"""Per-track state: motion history, action smoothing, zone occupancy, cooldowns.

Everything time-dependent lives here so the individual analytics modules stay
stateless and testable.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from ..config import settings
from .detect import Detection


@dataclass
class ZoneOccupancy:
    inside: bool = False
    since: float = 0.0
    frames_inside: int = 0
    entered_count: int = 0
    last_alert_ts: float = 0.0
    last_side: int = 0            # for line crossings: sign of the last test


@dataclass
class TrackState:
    track_id: int
    name: str
    group: str
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=settings.track_history))
    action_votes: deque = field(default_factory=lambda: deque(maxlen=9))
    action: str = "unknown"
    action_conf: float = 0.0
    max_box_height: float = 1.0
    zones: dict[str, ZoneOccupancy] = field(default_factory=dict)
    last_event_ts: dict[str, float] = field(default_factory=dict)
    plate: str | None = None
    plate_conf: float = 0.0
    face_count: int = 0
    peak_threat: int = 0
    attrs: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ update
    def update(self, det: Detection, now: float) -> None:
        self.last_seen = now
        self.history.append((now, det.foot, det.box))
        self.max_box_height = max(self.max_box_height, float(det.height))
        for a in det.attrs:
            self.attrs.add(a)

    # ----------------------------------------------------------------- motion
    @property
    def age(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def speed_body_heights(self, window: float = 1.0) -> float:
        """Foot-point speed in body-heights per second — scale invariant, so a
        distant runner and a close one score alike."""
        if len(self.history) < 2:
            return 0.0
        t_end, p_end, box_end = self.history[-1]
        cutoff = t_end - window
        start = None
        for rec in reversed(self.history):
            start = rec
            if rec[0] <= cutoff:
                break
        if start is None or start[0] >= t_end:
            return 0.0
        dt = t_end - start[0]
        dist = math.dist(p_end, start[1])
        h = max(1.0, float(box_end[3] - box_end[1]))
        return (dist / h) / dt if dt > 0 else 0.0

    def vertical_rate(self, window: float = 1.5) -> float:
        """Upward foot movement in body-heights/second (positive = rising).
        A climber's feet leave the ground; a walker's do not."""
        if len(self.history) < 2:
            return 0.0
        t_end, p_end, box_end = self.history[-1]
        cutoff = t_end - window
        start = None
        for rec in reversed(self.history):
            start = rec
            if rec[0] <= cutoff:
                break
        if start is None or start[0] >= t_end:
            return 0.0
        dt = t_end - start[0]
        dy = start[1][1] - p_end[1]          # image y grows downward
        h = max(1.0, float(box_end[3] - box_end[1]))
        return (dy / h) / dt if dt > 0 else 0.0

    def net_displacement(self, window: float = 8.0) -> float:
        if len(self.history) < 2:
            return 0.0
        t_end, p_end, box_end = self.history[-1]
        cutoff = t_end - window
        start = None
        for rec in reversed(self.history):
            start = rec
            if rec[0] <= cutoff:
                break
        h = max(1.0, float(box_end[3] - box_end[1]))
        return math.dist(p_end, start[1]) / h if start else 0.0

    # ---------------------------------------------------------------- helpers
    def vote_action(self, label: str, conf: float) -> tuple[str, float]:
        """Majority vote over a short window: single-frame pose noise should not
        raise an intrusion alert."""
        self.action_votes.append(label)
        counts: dict[str, int] = {}
        for v in self.action_votes:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts, key=lambda k: counts[k])
        share = counts[best] / len(self.action_votes)
        self.action = best
        self.action_conf = round(conf * share, 3)
        return self.action, self.action_conf

    def cooled_down(self, kind: str, now: float, seconds: float | None = None) -> bool:
        gap = settings.event_cooldown_seconds if seconds is None else seconds
        return (now - self.last_event_ts.get(kind, 0.0)) >= gap

    def mark_event(self, kind: str, now: float) -> None:
        self.last_event_ts[kind] = now

    def zone(self, zone_id: str) -> ZoneOccupancy:
        return self.zones.setdefault(zone_id, ZoneOccupancy())


class TrackStore:
    """Track table for one camera, with expiry of stale tracks."""

    def __init__(self, ttl: float = 5.0) -> None:
        self.ttl = ttl
        self.tracks: dict[int, TrackState] = {}

    def get(self, det: Detection, now: float | None = None) -> TrackState:
        now = now if now is not None else time.time()
        st = self.tracks.get(det.track_id)
        if st is None:
            st = TrackState(track_id=det.track_id, name=det.name,
                            group=det.group, first_seen=now, last_seen=now)
            self.tracks[det.track_id] = st
        st.update(det, now)
        return st

    def expire(self, now: float | None = None) -> list[TrackState]:
        now = now if now is not None else time.time()
        dead = [tid for tid, st in self.tracks.items()
                if now - st.last_seen > self.ttl]
        return [self.tracks.pop(tid) for tid in dead]

    @property
    def active(self) -> int:
        return len(self.tracks)
