"""Multi-Camera Synthetic Video Generator for Cross-Camera ReID & Tracking.

Generates 3 synchronized multi-camera streams with overlapping & handover FOVs:
- Camera #1 (Entrance & Gate): Worker A (Orange Vest) enters and moves right; Worker C (Navy) in background.
- Camera #2 (Inner Yard): Worker A transitions in from Cam 1; Worker B (Lime-Green Vest) crosses.
- Camera #3 (Access Corridor): Worker A arrives from Cam 2; Worker B follows into facility.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def draw_worker(frame: np.ndarray, x: int, y: int, height: int, vest_color: tuple[int, int, int], helmet_color: tuple[int, int, int], pants_color: tuple[int, int, int], frame_num: int = 0, posture: str = "walk") -> None:
    """Draw a construction/border worker with realistic helmet, high-vis vest, and pants."""
    skin = (140, 175, 220)
    w = int(height * 0.35)

    if posture == "crouch":
        # Crouching posture
        cw = int(w * 1.3)
        ch = int(height * 0.6)
        # Torso / Vest
        cv2.ellipse(frame, (x, y + ch // 2), (cw // 2, ch // 2), -15, 0, 360, vest_color, -1)
        # High-vis reflective stripe
        cv2.ellipse(frame, (x, y + ch // 2), (cw // 2 - 4, ch // 2 - 4), -15, 0, 360, (230, 240, 240), 2)
        # Head / Helmet
        cv2.circle(frame, (x + 5, y + 10), int(height * 0.13), helmet_color, -1)
        # Pants / bent legs
        cv2.line(frame, (x - 10, y + ch - 12), (x - 18, y + ch), pants_color, 7)
        cv2.line(frame, (x + 10, y + ch - 12), (x + 16, y + ch), pants_color, 7)
        return

    # Walking posture
    # Head & Helmet (top 20%)
    head_y = y + int(height * 0.12)
    head_r = int(height * 0.1)
    cv2.circle(frame, (x, head_y), head_r, skin, -1)
    # Safety Helmet on top
    cv2.ellipse(frame, (x, head_y - int(head_r * 0.3)), (int(head_r * 1.15), int(head_r * 0.85)), 0, 180, 360, helmet_color, -1)
    cv2.line(frame, (x - int(head_r * 1.2), head_y - int(head_r * 0.3)), (x + int(head_r * 1.2), head_y - int(head_r * 0.3)), helmet_color, 3)

    # Torso with High-Vis Safety Vest (20% - 60%)
    torso_top = y + int(height * 0.22)
    torso_bot = y + int(height * 0.60)
    torso_w = int(w * 0.75)
    cv2.rectangle(frame, (x - torso_w // 2, torso_top), (x + torso_w // 2, torso_bot), vest_color, -1)
    # High-vis reflective horizontal silver stripes
    cv2.line(frame, (x - torso_w // 2, torso_top + int((torso_bot - torso_top) * 0.45)),
             (x + torso_w // 2, torso_top + int((torso_bot - torso_top) * 0.45)), (240, 245, 250), 3)
    cv2.line(frame, (x - torso_w // 2, torso_top + int((torso_bot - torso_top) * 0.75)),
             (x + torso_w // 2, torso_top + int((torso_bot - torso_top) * 0.75)), (240, 245, 250), 3)

    # Legs & Pants (60% - 100%)
    leg_sweep = math.sin(frame_num * 0.35) * (w * 0.8)
    cv2.line(frame, (x - torso_w // 4, torso_bot), (int(x - torso_w // 4 + leg_sweep), y + height), pants_color, 8)
    cv2.line(frame, (x + torso_w // 4, torso_bot), (int(x + torso_w // 4 - leg_sweep), y + height), pants_color, 8)

    # Arms
    arm_color = vest_color
    cv2.line(frame, (x - torso_w // 2, torso_top + 10), (int(x - torso_w // 2 - leg_sweep * 0.7), torso_top + int(height * 0.26)), arm_color, 6)
    cv2.line(frame, (x + torso_w // 2, torso_top + 10), (int(x + torso_w // 2 + leg_sweep * 0.7), torso_top + int(height * 0.26)), arm_color, 6)


# Distinct worker appearance specifications (BGR)
WORKER_A = {
    "name": "Worker A",
    "vest": (20, 110, 245),      # Bright High-Vis Orange
    "helmet": (245, 245, 250),   # White Helmet
    "pants": (120, 70, 40),      # Blue Denim
}

WORKER_B = {
    "name": "Worker B",
    "vest": (40, 230, 160),      # High-Vis Lime Green
    "helmet": (30, 220, 250),    # Yellow Helmet
    "pants": (40, 50, 60),       # Dark Navy Pants
}

WORKER_C = {
    "name": "Worker C",
    "vest": (70, 55, 45),        # Dark Tactical Navy
    "helmet": (180, 100, 30),    # Blue Helmet
    "pants": (90, 95, 100),      # Grey Workwear
}


def generate_multicam_videos(fps: int = 20, seconds: int = 15, width: int = 640, height: int = 360) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = fps * seconds

    p1 = MEDIA_DIR / "cam1_entrance.mp4"
    p2 = MEDIA_DIR / "cam2_yard.mp4"
    p3 = MEDIA_DIR / "cam3_corridor.mp4"

    out1 = cv2.VideoWriter(str(p1), fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(str(p2), fourcc, fps, (width, height))
    out3 = cv2.VideoWriter(str(p3), fourcc, fps, (width, height))

    print("Generating 3 synchronized multi-camera surveillance feeds...")

    for f in range(total_frames):
        t = f / total_frames  # 0.0 to 1.0 progress

        # ---------------- CAM 1: Entrance & Gate ----------------
        f1 = np.zeros((height, width, 3), dtype=np.uint8)
        f1[:] = (55, 60, 65)  # Concrete pavement
        # Draw gate road & booth
        cv2.rectangle(f1, (0, 0), (int(width * 0.25), int(height * 0.4)), (40, 45, 50), -1)
        cv2.putText(f1, "CAM #1: MAIN ENTRANCE", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)
        cv2.line(f1, (0, int(height * 0.75)), (width, int(height * 0.75)), (180, 180, 180), 2)

        # Worker C crouching in background near booth
        draw_worker(f1, int(width * 0.18), int(height * 0.52), 85, WORKER_C["vest"], WORKER_C["helmet"], WORKER_C["pants"], f, posture="crouch")

        # Worker A enters from left (t=0..0.5) and moves to right exit toward Yard
        if t <= 0.65:
            prog_a = t / 0.65
            ax = int(40 + prog_a * (width - 20))
            ay = int(height * 0.55 + prog_a * (height * 0.15))
            ah = int(115 + prog_a * 25)
            draw_worker(f1, ax, ay, ah, WORKER_A["vest"], WORKER_A["helmet"], WORKER_A["pants"], f)

        out1.write(f1)

        # ---------------- CAM 2: Inner Yard ----------------
        f2 = np.zeros((height, width, 3), dtype=np.uint8)
        f2[:] = (50, 52, 55)  # Industrial yard gravel
        cv2.rectangle(f2, (int(width * 0.75), 0), (width, int(height * 0.5)), (35, 40, 45), -1)
        cv2.putText(f2, "CAM #2: INNER YARD", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 179, 0), 1)
        cv2.line(f2, (0, int(height * 0.45)), (width, int(height * 0.45)), (100, 105, 110), 2)

        # Worker A enters from left (arrives from Cam 1 at t=0.25..0.85)
        if 0.25 <= t <= 0.90:
            prog_a2 = (t - 0.25) / 0.65
            ax2 = int(20 + prog_a2 * (width - 40))
            ay2 = int(height * 0.48 + prog_a2 * 30)
            draw_worker(f2, ax2, ay2, 120, WORKER_A["vest"], WORKER_A["helmet"], WORKER_A["pants"], f)

        # Worker B (Lime Green) walks across in foreground (t=0.1..0.8)
        if 0.10 <= t <= 0.80:
            prog_b = (t - 0.10) / 0.70
            bx = int(width - 40 - prog_b * (width - 80))
            by = int(height * 0.65 - prog_b * 20)
            draw_worker(f2, bx, by, 135, WORKER_B["vest"], WORKER_B["helmet"], WORKER_B["pants"], f)

        out2.write(f2)

        # ---------------- CAM 3: Access Corridor ----------------
        f3 = np.zeros((height, width, 3), dtype=np.uint8)
        f3[:] = (45, 48, 50)  # Tiled corridor
        cv2.putText(f3, "CAM #M: ACCESS CORRIDOR", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 1)
        # Perspective hallway lines
        cv2.line(f3, (int(width * 0.3), 0), (0, height), (70, 75, 80), 2)
        cv2.line(f3, (int(width * 0.7), 0), (width, height), (70, 75, 80), 2)

        # Worker A arrives from Cam 2 yard (t=0.55..1.0) and advances forward
        if t >= 0.50:
            prog_a3 = (t - 0.50) / 0.50
            ax3 = int(width * 0.42 + math.sin(prog_a3 * 2.0) * 40)
            ay3 = int(height * 0.35 + prog_a3 * (height * 0.42))
            ah3 = int(80 + prog_a3 * 75)
            draw_worker(f3, ax3, ay3, ah3, WORKER_A["vest"], WORKER_A["helmet"], WORKER_A["pants"], f)

        # Worker B arrives shortly behind (t=0.70..1.0)
        if t >= 0.65:
            prog_b3 = (t - 0.65) / 0.35
            bx3 = int(width * 0.58 + math.cos(prog_b3 * 2.0) * 30)
            by3 = int(height * 0.28 + prog_b3 * (height * 0.38))
            bh3 = int(70 + prog_b3 * 65)
            draw_worker(f3, bx3, by3, bh3, WORKER_B["vest"], WORKER_B["helmet"], WORKER_B["pants"], f)

        out3.write(f3)

    out1.release()
    out2.release()
    out3.release()
    print("Multi-camera video feeds generated successfully!")


if __name__ == "__main__":
    generate_multicam_videos()
