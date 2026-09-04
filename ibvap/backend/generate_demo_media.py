"""Synthetic surveillance scenario generator for IBVAP.

Generates realistic demo video feeds for testing without physical cameras:
1. border_fence.mp4: Person approaching fence, crouching/crawling, and breaching tripwire.
2. gate_checkpoint.mp4: Vehicle with license plate pulling up to a security checkpoint gate.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def draw_human_figure(frame: np.ndarray, x: int, y: int, height: int, posture: str = "walk", frame_num: int = 0) -> None:
    """Draw a realistic human silhouette with limbs in proper proportion for YOLO & Pose estimation."""
    color = (45, 50, 55) # Dark tactical clothing
    skin = (140, 175, 220) # BGR skin tone

    if posture == "crawl":
        # Horizontal elongated body for crawling posture
        w = int(height * 0.9)
        h = int(height * 0.35)
        # Torso
        cv2.ellipse(frame, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, color, -1)
        # Head low near front
        cv2.circle(frame, (x + w - 10, y + h // 3), int(h * 0.35), skin, -1)
        # Arms forward
        arm_phase = math.sin(frame_num * 0.4) * 15
        cv2.line(frame, (x + w - 15, y + h // 2), (x + w + int(arm_phase), y + h), color, 6)
        # Legs back
        cv2.line(frame, (x + 10, y + h // 2), (x - 20 - int(arm_phase), y + h), color, 6)
    elif posture == "crouch":
        w = int(height * 0.5)
        h = int(height * 0.6)
        # Torso hunched
        cv2.ellipse(frame, (x, y + h // 2), (w // 2, h // 2), -15, 0, 360, color, -1)
        # Head
        cv2.circle(frame, (x + 5, y + 10), int(height * 0.14), skin, -1)
        # Bent legs
        cv2.line(frame, (x - 10, y + h - 15), (x - 20, y + h), color, 7)
        cv2.line(frame, (x + 10, y + h - 15), (x + 15, y + h), color, 7)
    else: # walk / stand
        w = int(height * 0.3)
        h = height
        # Head
        cv2.circle(frame, (x, y + int(h * 0.12)), int(h * 0.11), skin, -1)
        # Torso
        torso_top = y + int(h * 0.22)
        torso_bot = y + int(h * 0.6)
        cv2.line(frame, (x, torso_top), (x, torso_bot), color, int(w * 0.7))
        # Legs walking animation
        leg_sweep = math.sin(frame_num * 0.35) * (w * 0.8)
        cv2.line(frame, (x, torso_bot), (int(x + leg_sweep), y + h), color, 7)
        cv2.line(frame, (x, torso_bot), (int(x - leg_sweep), y + h), color, 7)
        # Arms
        cv2.line(frame, (x, torso_top + 10), (int(x - leg_sweep * 0.8), torso_top + int(h * 0.25)), color, 5)
        cv2.line(frame, (x, torso_top + 10), (int(x + leg_sweep * 0.8), torso_top + int(h * 0.25)), color, 5)


def generate_border_fence_video(output_path: Path, width: int = 960, height: int = 540, fps: int = 20, seconds: int = 14) -> None:
    """Generate border perimeter video with person crawling and breaching fence."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    total_frames = fps * seconds

    print(f"Generating {output_path.name} ({total_frames} frames)...")

    for f in range(total_frames):
        # Base background: dusk border outpost landscape
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky gradient (dusk / twilight low-light)
        for row in range(int(height * 0.55)):
            r = int(25 + 40 * (row / (height * 0.55)))
            g = int(35 + 45 * (row / (height * 0.55)))
            b = int(50 + 60 * (row / (height * 0.55)))
            frame[row, :] = (b, g, r)

        # Ground terrain (arid sand/gravel)
        for row in range(int(height * 0.55), height):
            depth = (row - height * 0.55) / (height * 0.45)
            frame[row, :] = (int(45 + 30 * depth), int(60 + 35 * depth), int(75 + 40 * depth))

        # Patrol track road
        pts_road = np.array([
            [int(width * 0.1), height],
            [int(width * 0.8), height],
            [int(width * 0.6), int(height * 0.55)],
            [int(width * 0.35), int(height * 0.55)],
        ], np.int32)
        cv2.fillPoly(frame, [pts_road], (40, 50, 60))

        # Security perimeter fence line (mesh posts and barbed wire)
        fence_y = int(height * 0.65)
        for post_x in range(40, width, 90):
            cv2.line(frame, (post_x, fence_y + 40), (post_x, fence_y - 90), (70, 75, 80), 4)
            # Barb tops
            cv2.line(frame, (post_x, fence_y - 90), (post_x + 15, fence_y - 105), (90, 95, 100), 3)

        # Chainlink horizontal wires
        for wy in range(fence_y - 80, fence_y + 40, 18):
            cv2.line(frame, (0, wy), (width, wy), (60, 65, 70), 1)

        # Watchtower in background
        cv2.rectangle(frame, (int(width * 0.82), int(height * 0.35)), (int(width * 0.9), int(height * 0.55)), (30, 35, 40), -1)
        cv2.rectangle(frame, (int(width * 0.8), int(height * 0.3)), (int(width * 0.92), int(height * 0.36)), (50, 55, 60), -1)

        # Person movement trajectory
        # 0..60 (0..3s): Scene calm, person appears at far left walking
        # 60..150 (3..7.5s): Approaches fence
        # 150..220 (7.5..11s): Crouches and crawls toward fence gap
        # 220..total: Breaches past fence line into restricted zone
        if f >= 30:
            progress = min(1.0, (f - 30) / (total_frames - 30))
            px = int(80 + progress * 580)
            py_base = int(height * 0.62 + progress * 90)

            if f < 140:
                # Walking toward perimeter
                p_height = int(90 + progress * 35)
                draw_human_figure(frame, px, py_base - p_height, p_height, posture="walk", frame_num=f)
            elif f < 210:
                # Crouching & crawling
                p_height = int(80 + progress * 20)
                draw_human_figure(frame, px, py_base - int(p_height * 0.4), p_height, posture="crawl", frame_num=f)
            else:
                # Low sprint breach past fence
                p_height = int(95 + progress * 30)
                draw_human_figure(frame, px, py_base - p_height, p_height, posture="crouch", frame_num=f)

        # Add subtle CCTV camera scanline / grain
        noise = np.random.randint(-5, 6, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()
    print(f"Finished {output_path.name}")


def generate_checkpoint_video(output_path: Path, width: int = 960, height: int = 540, fps: int = 20, seconds: int = 12) -> None:
    """Generate checkpoint gate video with vehicle approaching and readable Indian plate."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    total_frames = fps * seconds

    print(f"Generating {output_path.name} ({total_frames} frames)...")

    plate_text = "DL 01 AB 1234"

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Asphalt checkpoint lane
        frame[:] = (75, 75, 80)

        # Lane markings
        cv2.line(frame, (int(width * 0.1), 0), (int(width * 0.1), height), (220, 220, 220), 4)
        cv2.line(frame, (int(width * 0.9), 0), (int(width * 0.9), height), (220, 220, 220), 4)
        # Yellow curb / barrier
        cv2.rectangle(frame, (0, 0), (int(width * 0.08), height), (30, 160, 200), -1)
        cv2.rectangle(frame, (int(width * 0.92), 0), (width, height), (30, 160, 200), -1)

        # Security boom barrier across the lane
        barrier_y = int(height * 0.38)
        # Guard cabin
        cv2.rectangle(frame, (int(width * 0.02), barrier_y - 80), (int(width * 0.18), barrier_y + 40), (45, 55, 65), -1)
        cv2.putText(frame, "SSB POST 4", (int(width * 0.03), barrier_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1)

        # Boom barrier pole (red and white stripes)
        for bx in range(int(width * 0.15), int(width * 0.85), 40):
            color = (255, 255, 255) if ((bx // 40) % 2 == 0) else (30, 30, 220)
            cv2.line(frame, (bx, barrier_y), (bx + 38, barrier_y), color, 8)

        # Stop line
        cv2.line(frame, (int(width * 0.12), barrier_y + 70), (int(width * 0.88), barrier_y + 70), (240, 240, 240), 8)
        cv2.putText(frame, "STOP", (int(width * 0.44), barrier_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        # Vehicle motion: drives up from bottom toward the stop line
        # f=0..160 (approaches and stops at line)
        progress = min(1.0, (f / (total_frames * 0.75)))
        # Car position and scale
        car_y = int(height * 0.9 - progress * (height * 0.42))
        car_w = int(240 + (1.0 - progress) * 120)
        car_h = int(140 + (1.0 - progress) * 70)
        car_x = int(width * 0.5 - car_w // 2)

        # Draw vehicle body (SUV / Sedan)
        car_color = (190, 195, 200) # Silver white
        # Main chassis
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), car_color, -1)
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 50, 50), 3)

        # Windshield
        ws_margin = int(car_w * 0.12)
        cv2.rectangle(frame, (car_x + ws_margin, car_y + 10), (car_x + car_w - ws_margin, car_y + int(car_h * 0.35)), (40, 45, 50), -1)

        # Headlights
        hl_w = int(car_w * 0.15)
        cv2.rectangle(frame, (car_x + 10, car_y + int(car_h * 0.45)), (car_x + 10 + hl_w, car_y + int(car_h * 0.65)), (180, 240, 255), -1)
        cv2.rectangle(frame, (car_x + car_w - 10 - hl_w, car_y + int(car_h * 0.45)), (car_x + car_w - 10, car_y + int(car_h * 0.65)), (180, 240, 255), -1)

        # License Plate Banner on front bumper
        plate_w = int(car_w * 0.42)
        plate_h = int(car_h * 0.18)
        plate_x = car_x + (car_w - plate_w) // 2
        plate_y = car_y + int(car_h * 0.72)

        # High-contrast white plate with black border
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (245, 245, 250), -1)
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (20, 20, 20), 2)
        # IND blue strip on left
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + int(plate_w * 0.12), plate_y + plate_h), (180, 50, 20), -1)

        # Plate text
        font_scale = max(0.4, (plate_h / 32.0))
        cv2.putText(
            frame, plate_text,
            (plate_x + int(plate_w * 0.16), plate_y + int(plate_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 2, cv2.LINE_AA
        )

        out.write(frame)

    out.release()
    print(f"Finished {output_path.name}")


if __name__ == "__main__":
    generate_border_fence_video(MEDIA_DIR / "border_fence.mp4")
    generate_checkpoint_video(MEDIA_DIR / "gate_checkpoint.mp4")
    print("Demo media generated successfully in:", MEDIA_DIR)
