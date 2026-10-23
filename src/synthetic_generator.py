"""Synthetic traffic video generator for testing and validation."""

import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_traffic_video(
    output_path: str = "videos/sample_traffic.mp4",
    duration_sec: int = 15,
    fps: int = 30,
    width: int = 1280,
    height: int = 720
) -> str:
    """
    Generate a clean synthetic road video with moving vehicles crossing a line.
    
    Args:
        output_path: Output video file path.
        duration_sec: Video length in seconds.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path to generated video file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Vehicle profiles: (spawn_frame, x, speed_y, width, height, color, label)
    # We stagger vehicles so traffic starts LOW, transitions to MEDIUM, then bursts to HIGH
    vehicles = [
        # Phase 1: Sparse vehicles (LOW traffic)
        {"spawn": 10, "x": 350, "y": -80, "vy": 4, "w": 65, "h": 110, "color": (180, 50, 50), "type": "car"},
        {"spawn": 40, "x": 600, "y": -50, "vy": 5, "w": 30, "h": 60, "color": (50, 180, 50), "type": "motorcycle"},
        {"spawn": 75, "x": 850, "y": -120, "vy": 3.5, "w": 85, "h": 170, "color": (40, 40, 200), "type": "bus"},
        
        # Phase 2: Moderate flow (MEDIUM traffic)
        {"spawn": 110, "x": 380, "y": -90, "vy": 4.5, "w": 70, "h": 120, "color": (200, 150, 30), "type": "car"},
        {"spawn": 130, "x": 620, "y": -90, "vy": 4.0, "w": 65, "h": 115, "color": (160, 60, 160), "type": "car"},
        {"spawn": 150, "x": 860, "y": -130, "vy": 3.0, "w": 90, "h": 190, "color": (80, 130, 200), "type": "truck"},
        {"spawn": 170, "x": 420, "y": -50, "vy": 5.5, "w": 28, "h": 65, "color": (30, 200, 200), "type": "motorcycle"},

        # Phase 3: Clustered surge inside ROI (HIGH traffic)
        {"spawn": 200, "x": 320, "y": -90, "vy": 3.0, "w": 70, "h": 120, "color": (190, 70, 70), "type": "car"},
        {"spawn": 210, "x": 500, "y": -90, "vy": 3.2, "w": 68, "h": 115, "color": (70, 190, 70), "type": "car"},
        {"spawn": 220, "x": 680, "y": -90, "vy": 3.1, "w": 72, "h": 125, "color": (70, 70, 190), "type": "car"},
        {"spawn": 230, "x": 880, "y": -90, "vy": 2.8, "w": 85, "h": 160, "color": (190, 150, 40), "type": "truck"},
        {"spawn": 240, "x": 400, "y": -50, "vy": 4.0, "w": 30, "h": 60, "color": (40, 200, 200), "type": "motorcycle"},
        {"spawn": 250, "x": 580, "y": -90, "vy": 3.0, "w": 66, "h": 110, "color": (120, 120, 120), "type": "car"},
        {"spawn": 260, "x": 760, "y": -90, "vy": 2.9, "w": 68, "h": 118, "color": (180, 100, 50), "type": "car"},
        {"spawn": 270, "x": 920, "y": -120, "vy": 2.5, "w": 90, "h": 180, "color": (50, 150, 180), "type": "bus"},
    ]

    for f in range(total_frames):
        # Draw asphalt road
        frame = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        # Road shoulders
        cv2.rectangle(frame, (0, 0), (200, height), (35, 75, 40), -1)      # Green grass left
        cv2.rectangle(frame, (1080, 0), (width, height), (35, 75, 40), -1) # Green grass right
        cv2.line(frame, (200, 0), (200, height), (220, 220, 220), 4)       # Left solid white line
        cv2.line(frame, (1080, 0), (1080, height), (220, 220, 220), 4)     # Right solid white line

        # 3 Lane dividers (dashed lines)
        lane_step = 60
        dash_offset = (f * 4) % lane_step
        for lane_x in (420, 640, 860):
            for y_start in range(-dash_offset, height, lane_step):
                y_end = min(height, y_start + 30)
                if y_end > 0:
                    cv2.line(frame, (lane_x, max(0, y_start)), (lane_x, y_end), (240, 240, 240), 2)

        # Draw active vehicles
        for v in vehicles:
            if f >= v["spawn"]:
                cur_y = v["y"] + (f - v["spawn"]) * v["vy"]
                if cur_y < height + 100:
                    x1 = int(v["x"] - v["w"] / 2)
                    y1 = int(cur_y - v["h"] / 2)
                    x2 = int(v["x"] + v["w"] / 2)
                    y2 = int(cur_y + v["h"] / 2)

                    # Vehicle body
                    cv2.rectangle(frame, (x1, y1), (x2, y2), v["color"], -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)
                    # Windshield / windows
                    win_y1 = int(y1 + v["h"] * 0.2)
                    win_y2 = int(y1 + v["h"] * 0.45)
                    cv2.rectangle(frame, (x1 + 6, win_y1), (x2 - 6, win_y2), (200, 220, 230), -1)
                    # Headlights
                    cv2.circle(frame, (x1 + 8, y2 - 4), 4, (100, 255, 255), -1)
                    cv2.circle(frame, (x2 - 8, y2 - 4), 4, (100, 255, 255), -1)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="videos/sample_traffic.mp4")
    parser.add_argument("--duration", type=int, default=15)
    args = parser.parse_args()
    res = generate_synthetic_traffic_video(args.out, duration_sec=args.duration)
    print(f"Generated synthetic video at: {res}")
