"""Unit tests for TrafficLogger (CSV and JSON summary)."""

import json
import tempfile
from pathlib import Path
from src.logger import TrafficLogger


def test_logger_csv_and_summary(tmp_path=None):
    if tmp_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(temp_dir.name)
    else:
        temp_dir = None

    csv_file = tmp_path / "test_log.csv"
    summary_file = tmp_path / "test_summary.json"

    logger = TrafficLogger(
        csv_path=str(csv_file),
        summary_path=str(summary_file),
        log_interval_sec=1.0,
        vehicle_classes=["car", "motorcycle", "bus", "truck", "bicycle"]
    )

    # Log at t=0.0s
    logger.log_step(
        timestamp_sec=0.0,
        total_count=1,
        class_counts={"car": 1, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0},
        density=1,
        flow_per_min=12.0,
        traffic_level="MEDIUM"
    )

    # Log at t=0.5s -> should NOT log due to interval=1.0s
    logger.log_step(
        timestamp_sec=0.5,
        total_count=1,
        class_counts={"car": 1, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0},
        density=1,
        flow_per_min=12.0,
        traffic_level="MEDIUM"
    )

    # Log at t=1.2s -> interval (1.2 - 0.0 >= 1.0) elapsed, should log
    logger.log_step(
        timestamp_sec=1.2,
        total_count=2,
        class_counts={"car": 2, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0},
        density=2,
        flow_per_min=24.0,
        traffic_level="HIGH"
    )

    logger.close()

    # Verify CSV rows
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3  # Header + 2 logged rows
    assert "timestamp_sec,total_count,car,motorcycle,bus,truck,bicycle,in_roi_density,flow_per_minute,traffic_level" in lines[0]
    assert "0.00,1,1,0,0,0,0,1,12.00,MEDIUM" in lines[1]
    assert "1.20,2,2,0,0,0,0,2,24.00,HIGH" in lines[2]

    # Test summary JSON write
    logger.write_summary(
        total_count=2,
        class_counts={"car": 2, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0},
        direction_counts={"In": 2, "Out": 0},
        traffic_level_metrics={
            "peak_level": "HIGH",
            "time_spent_in_levels_sec": {"LOW": 0.0, "MEDIUM": 1.2, "HIGH": 0.0},
            "time_spent_in_levels_pct": {"LOW": 0.0, "MEDIUM": 100.0, "HIGH": 0.0}
        },
        average_fps=30.0,
        total_duration_sec=1.2,
        total_frames_processed=36,
        model_name="yolo26n.pt",
        tracker_name="bytetrack.yaml"
    )

    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_count"] == 2
    assert data["per_class_counts"]["car"] == 2
    assert data["peak_traffic_level"] == "HIGH"
    assert data["tracker"] == "bytetrack.yaml"
    assert data["average_fps"] == 30.0
