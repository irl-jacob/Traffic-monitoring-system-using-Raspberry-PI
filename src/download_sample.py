"""Helper script to download real-world sample traffic footage or generate synthetic clip."""

import argparse
import sys
import urllib.request
from pathlib import Path
from src.synthetic_generator import generate_synthetic_traffic_video

# Direct public domain / CC0 traffic footage URLs
SAMPLE_VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"


def get_traffic_video(output_path: str = "videos/road.mp4", fallback_synthetic: bool = True) -> str:
    """
    Ensure a sample traffic video exists at output_path.
    Attempts to download a sample traffic clip; falls back to generating a synthetic video.
    """
    target = Path(output_path)
    if target.exists() and target.stat().st_size > 1000:
        print(f"Video already exists at: {target}")
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample road video from {SAMPLE_VIDEO_URL}...")
    try:
        urllib.request.urlretrieve(SAMPLE_VIDEO_URL, str(target))
        if target.exists() and target.stat().st_size > 10000:
            print(f"Successfully downloaded road video ({target.stat().st_size / (1024*1024):.1f} MB) to {target}")
            return str(target)
    except Exception as e:
        print(f"Download failed ({e}).")

    if fallback_synthetic:
        print(f"Generating synthetic traffic video at: {target}")
        generate_synthetic_traffic_video(str(target), duration_sec=20)
        return str(target)

    raise RuntimeError(f"Could not acquire traffic video for {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="videos/road.mp4")
    args = parser.parse_args()
    get_traffic_video(args.out)
