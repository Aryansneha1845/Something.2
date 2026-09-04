"""Central configuration and camera registry for IBVAP.

Deliberately dependency-free (stdlib + dataclasses) so the config layer can be
imported by tooling and tests without pulling in torch/opencv.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent          # .../backend
PROJECT_DIR = BASE_DIR.parent                              # .../ibvap
DATA_DIR = BASE_DIR / "data"
SNAP_DIR = DATA_DIR / "snapshots"
CLIP_DIR = DATA_DIR / "clips"
MODEL_DIR = BASE_DIR / "models"
MEDIA_DIR = PROJECT_DIR / "media"
CAMERAS_FILE = BASE_DIR / "cameras.json"
DB_PATH = DATA_DIR / "ibvap.db"

for _d in (DATA_DIR, SNAP_DIR, CLIP_DIR, MODEL_DIR, MEDIA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: Any) -> Any:
    raw = os.environ.get(f"IBVAP_{key}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # --- models -------------------------------------------------------------
    detect_model: str = str(MODEL_DIR / "yolo11n.pt")
    pose_model: str = str(MODEL_DIR / "yolo11n-pose.pt")
    device: str = "cpu"

    # --- inference budget (tuned for 4-core CPU, no GPU) --------------------
    imgsz: int = 512
    conf: float = 0.35
    iou: float = 0.5
    target_fps: float = 8.0        # analytic rate; ingest drops frames to match
    pose_stride: int = 2           # run pose every Nth analysed frame
    anpr_stride: int = 12
    face_stride: int = 6
    caption_stride: int = 8

    # --- low-light ----------------------------------------------------------
    enhance_auto: bool = True
    night_luma_threshold: float = 70.0   # mean Y below this => "night mode"
    clahe_clip: float = 2.5
    gamma: float = 1.6

    # --- event logic --------------------------------------------------------
    loiter_seconds: float = 6.0
    intrusion_min_frames: int = 2
    event_cooldown_seconds: float = 8.0   # per (camera, track, kind)
    track_history: int = 90
    clip_pre_seconds: float = 3.0
    clip_post_seconds: float = 3.0

    # --- privacy / policy ---------------------------------------------------
    blur_faces: bool = False              # UI toggle; watchlist crops exempt
    face_watchlist_enabled: bool = False   # off until an operator enrols someone
    retain_events_days: int = 30

    # --- output -------------------------------------------------------------
    jpeg_quality: int = 72
    stream_max_width: int = 960

    # --- integrations -------------------------------------------------------
    ptz_simulate: bool = True
    c2_webhook_url: str = ""              # POSTs alert JSON to a C2 system
    llm_captions: bool = False            # opt-in VLM/LLM caption upgrade

    def __post_init__(self) -> None:
        for f_name, f_val in list(self.__dict__.items()):
            setattr(self, f_name, _env(f_name.upper(), f_val))


settings = Settings()


# --------------------------------------------------------------------------
# Camera registry
# --------------------------------------------------------------------------
@dataclass
class Zone:
    """A virtual fence. Points are NORMALISED (0..1) so they survive any
    resolution change between configuration and runtime."""
    id: str
    name: str
    kind: str = "polygon"              # "polygon" | "line"
    points: list[list[float]] = field(default_factory=list)
    severity: int = 3                  # 1..5, feeds the threat score
    direction: str = "both"            # line only: "in" | "out" | "both"
    classes: list[str] = field(default_factory=list)  # [] => any class


@dataclass
class PTZ:
    enabled: bool = False
    host: str = ""
    port: int = 80
    username: str = ""
    # NOTE: never stored here in a real deployment; read from a secret store.
    password_env: str = ""


@dataclass
class Camera:
    id: str
    name: str
    source: str                        # rtsp:// | file path | webcam index
    sector: str = "UNASSIGNED"
    enabled: bool = True
    loop: bool = True                  # loop video files (demo convenience)
    map_x: float = 0.5                 # normalised position on the sector map
    map_y: float = 0.5
    zones: list[Zone] = field(default_factory=list)
    ptz: PTZ = field(default_factory=PTZ)
    analytics: list[str] = field(default_factory=lambda: [
        "person", "vehicle", "pose", "intrusion", "anpr", "face", "night"
    ])

    @property
    def is_file(self) -> bool:
        return not self.source.startswith(("rtsp://", "http://", "https://")) \
            and not self.source.isdigit()


def _camera_from_dict(d: dict) -> Camera:
    zones = [Zone(**z) for z in d.pop("zones", [])]
    ptz = PTZ(**d.pop("ptz", {})) if isinstance(d.get("ptz"), dict) else PTZ()
    return Camera(zones=zones, ptz=ptz, **d)


def load_cameras() -> list[Camera]:
    if not CAMERAS_FILE.exists():
        return []
    raw = json.loads(CAMERAS_FILE.read_text(encoding="utf-8"))
    return [_camera_from_dict(dict(c)) for c in raw.get("cameras", [])]


def save_cameras(cams: list[Camera]) -> None:
    def enc(o: Any) -> Any:
        if hasattr(o, "__dataclass_fields__"):
            return {k: enc(v) for k, v in o.__dict__.items()}
        if isinstance(o, list):
            return [enc(i) for i in o]
        return o
    payload = {"cameras": [enc(c) for c in cams]}
    CAMERAS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
