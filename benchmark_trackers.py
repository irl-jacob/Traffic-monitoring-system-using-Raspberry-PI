"""Benchmark script comparing ByteTrack vs BoT-SORT and model variants."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_benchmark(video_path: str, duration_limit_sec: int = 30):
    print("=" * 70)
    print(" TRAFFIC MONITORING - TRACKER & MODEL BENCHMARK")
    print("=" * 70)
    print(f"Testing video source: {video_path}")

    configs = [
        {"name": "ByteTrack (Nano)", "tracker": "bytetrack.yaml", "model": "yolo26n.pt"},
        {"name": "BoT-SORT (Nano)", "tracker": "botsort.yaml", "model": "yolo26n.pt"},
    ]

    results = []
    for cfg in configs:
        print(f"\n---> Running benchmark configuration: {cfg['name']}...")
        tmp_summary = f"outputs/summary_bench_{cfg['tracker'].replace('.', '_')}.json"
        cmd = [
            sys.executable,
            "main.py",
            "--source", video_path,
            "--tracker", cfg["tracker"],
            "--model", cfg["model"],
            "--device", "auto"
        ]

        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"FAILED with return code {res.returncode}")
            print(f"Error output:\n{res.stderr}")
            continue

        # Read summary.json
        with open("outputs/summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)

        results.append({
            "name": cfg["name"],
            "tracker": cfg["tracker"],
            "model": summary.get("model", cfg["model"]),
            "fps": summary.get("average_fps", 0.0),
            "total_count": summary.get("total_count", 0),
            "peak_level": summary.get("peak_traffic_level", "LOW"),
            "frames": summary.get("total_frames_processed", 0)
        })

    print("\n" + "=" * 70)
    print(f"{'Configuration':<25} | {'FPS':<8} | {'Total Count':<12} | {'Peak Level':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} | {r['fps']:<8.1f} | {r['total_count']:<12} | {r['peak_level']:<10}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="videos/road.mp4")
    args = parser.parse_args()
    run_benchmark(args.source)
