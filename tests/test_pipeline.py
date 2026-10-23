"""End-to-end integration test running pipeline on synthetic video."""

import json
import subprocess
import sys
from pathlib import Path
import pytest


def test_end_to_end_pipeline_execution(tmp_path: Path):
    """Generate a short synthetic traffic video and run main.py in headless mode."""
    # Ensure src and root are in python path
    repo_root = Path(__file__).resolve().parent.parent
    synthetic_script = repo_root / "src" / "synthetic_generator.py"
    main_script = repo_root / "main.py"

    sample_video = tmp_path / "test_traffic.mp4"
    output_video = tmp_path / "output_demo.mp4"
    log_csv = tmp_path / "test_log.csv"
    summary_json = tmp_path / "test_summary.json"
    temp_config = tmp_path / "test_config.yaml"

    # Write a test configuration
    config_content = f"""
video:
  source: "{sample_video}"
  imgsz: 320
  conf: 0.15
  frame_skip: 2

model:
  weights: "yolo26n.pt"
  device: "cpu"
  tracker: "bytetrack.yaml"

counting_line:
  p1: [100, 360]
  p2: [1180, 360]

roi_polygon:
  points:
    - [100, 150]
    - [1180, 150]
    - [1180, 550]
    - [100, 550]

traffic_level:
  flow_window_sec: 10
  hysteresis_sec: 1.0
  flow_thresholds:
    low_max: 5.0
    medium_max: 15.0
  density_thresholds:
    low_max: 2
    medium_max: 5

logging:
  csv_path: "{log_csv}"
  summary_path: "{summary_json}"
  log_interval_sec: 0.5
"""
    with open(temp_config, "w", encoding="utf-8") as f:
        f.write(config_content)

    # 1. Generate 4-second synthetic clip
    gen_cmd = [
        sys.executable,
        str(synthetic_script),
        "--out", str(sample_video),
        "--duration", "4"
    ]
    gen_res = subprocess.run(gen_cmd, capture_output=True, text=True)
    assert gen_res.returncode == 0, f"Synthetic generator failed: {gen_res.stderr}"
    assert sample_video.exists(), "Sample video file was not created"

    # 2. Run main.py headless
    run_cmd = [
        sys.executable,
        str(main_script),
        "--config", str(temp_config),
        "--save", str(output_video)
    ]
    run_res = subprocess.run(run_cmd, capture_output=True, text=True)
    assert run_res.returncode == 0, f"main.py execution failed: {run_res.stderr}\nOutput: {run_res.stdout}"

    # 3. Assert artifacts were created
    assert output_video.exists(), "Output annotated video was not created"
    assert log_csv.exists(), "log.csv was not created"
    assert summary_json.exists(), "summary.json was not created"

    # 4. Verify summary.json content
    with open(summary_json, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert "total_count" in summary_data
    assert "per_class_counts" in summary_data
    assert "average_fps" in summary_data
    assert "peak_traffic_level" in summary_data
    assert summary_data["total_frames_processed"] > 0
