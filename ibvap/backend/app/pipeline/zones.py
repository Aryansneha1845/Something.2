"""Virtual fence logic: polygon breach, directional line crossing, loitering.

Zones are stored normalised (0..1) and projected to pixels at runtime, so the
same configuration works whether the stream negotiates 1080p or drops to CIF.

Tests use the detection's *foot point* rather than its centre: a person standing
just outside a fence line has a bounding-box centre that can fall inside it,
which is the classic source of false intrusion alerts.
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from ..config import Zone, settings
from .detect import Detection
from .tracks import TrackState


@dataclass
class ZoneHit:
    kind: str                  # "intrusion" | "crossing" | "loiter"
    zone_id: str
    zone_name: str
    severity: int
    dwell: float = 0.0
    direction: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "zone_id": self.zone_id,
                "zone_name": self.zone_name, "severity": self.severity,
                "dwell": round(self.dwell, 1), "direction": self.direction}


class ZoneEngine:
    """Projected zones for one camera at one frame size."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones
        self._size: tuple[int, int] = (0, 0)
        self._geom: dict[str, Polygon | LineString] = {}
        self._pixels: dict[str, list[tuple[int, int]]] = {}

    def project(self, width: int, height: int) -> None:
        if (width, height) == self._size or width <= 0 or height <= 0:
            return
        self._size = (width, height)
        self._geom.clear()
        self._pixels.clear()
        for z in self.zones:
            pts = [(p[0] * width, p[1] * height) for p in z.points]
            if z.kind == "line" and len(pts) >= 2:
                self._geom[z.id] = LineString(pts[:2])
            elif len(pts) >= 3:
                poly = Polygon(pts)
                self._geom[z.id] = poly if poly.is_valid else poly.buffer(0)
            self._pixels[z.id] = [(int(x), int(y)) for x, y in pts]

    def pixels(self, zone_id: str) -> list[tuple[int, int]]:
        return self._pixels.get(zone_id, [])

    # ------------------------------------------------------------------ tests
    @staticmethod
    def _applies(z: Zone, det: Detection) -> bool:
        return not z.classes or det.name in z.classes or det.group in z.classes

    @staticmethod
    def _side(line: LineString, pt: tuple[float, float]) -> int:
        (ax, ay), (bx, by) = list(line.coords)[:2]
        cross = (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax)
        return 1 if cross > 0 else (-1 if cross < 0 else 0)

    def evaluate(self, det: Detection, track: TrackState,
                 now: float) -> list[ZoneHit]:
        hits: list[ZoneHit] = []
        foot = Point(det.foot)
        for z in self.zones:
            geom = self._geom.get(z.id)
            if geom is None or not self._applies(z, det):
                continue
            occ = track.zone(z.id)

            if z.kind == "line":
                side = self._side(geom, det.foot)
                prev = occ.last_side
                occ.last_side = side or prev
                if prev and side and side != prev:
                    direction = "in" if side > 0 else "out"
                    if z.direction in ("both", direction):
                        occ.entered_count += 1
                        hits.append(ZoneHit("crossing", z.id, z.name,
                                            z.severity, direction=direction))
                continue

            inside = geom.contains(foot)
            if inside:
                if not occ.inside:
                    occ.inside, occ.since, occ.frames_inside = True, now, 0
                    occ.entered_count += 1
                occ.frames_inside += 1
                dwell = now - occ.since
                if occ.frames_inside == settings.intrusion_min_frames:
                    hits.append(ZoneHit("intrusion", z.id, z.name, z.severity))
                elif (dwell >= settings.loiter_seconds
                      and track.net_displacement(settings.loiter_seconds) < 1.2
                      and track.cooled_down(f"loiter:{z.id}", now,
                                            settings.loiter_seconds)):
                    track.mark_event(f"loiter:{z.id}", now)
                    hits.append(ZoneHit("loiter", z.id, z.name,
                                        max(1, z.severity - 1), dwell=dwell))
            elif occ.inside:
                occ.inside = False
                occ.frames_inside = 0
        return hits

    def occupancy(self, tracks: dict[int, TrackState]) -> dict[str, int]:
        counts = {z.id: 0 for z in self.zones}
        for st in tracks.values():
            for zid, occ in st.zones.items():
                if occ.inside and zid in counts:
                    counts[zid] += 1
        return counts


def default_zones(width_frac: float = 0.62) -> list[Zone]:
    """Sensible starting geometry so a new camera is useful before anyone has
    drawn anything: a fence line across the upper third and a restricted
    approach box in front of it."""
    return [
        Zone(id="fence-line", name="Fence Line", kind="line", severity=5,
             points=[[0.05, 0.42], [0.95, 0.38]], direction="both"),
        Zone(id="approach", name="Approach Apron", kind="polygon", severity=3,
             points=[[0.08, 0.45], [0.92, 0.41], [0.98, 0.95], [0.02, 0.95]]),
    ]
