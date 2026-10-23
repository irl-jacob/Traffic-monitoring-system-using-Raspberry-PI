"""Metrics logging to CSV (per-second) and summary JSON output."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class TrafficLogger:
    """Handles 1-second cadence CSV logging and end-of-run JSON summary."""

    def __init__(
        self,
        csv_path: str = "outputs/log.csv",
        summary_path: str = "outputs/summary.json",
        log_interval_sec: float = 1.0,
        vehicle_classes: Optional[List[str]] = None
    ):
        """
        Initialize the logger.

        Args:
            csv_path: Path to CSV output file.
            summary_path: Path to summary JSON file.
            log_interval_sec: Interval in seconds between CSV row flushes.
            vehicle_classes: List of vehicle class names to include in CSV columns.
        """
        self.csv_path = Path(csv_path)
        self.summary_path = Path(summary_path)
        self.log_interval_sec = float(log_interval_sec)
        self.vehicle_classes = vehicle_classes or ["car", "motorcycle", "bus", "truck", "bicycle"]

        self.last_log_sec: Optional[float] = None
        self.csv_file = None
        self.csv_writer = None

        self._init_csv()

    def _init_csv(self):
        """Create output directory and initialize CSV with header."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        header = [
            "timestamp_sec",
            "total_count",
            *self.vehicle_classes,
            "in_roi_density",
            "flow_per_minute",
            "traffic_level"
        ]
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(header)
        self.csv_file.flush()

    def log_step(
        self,
        timestamp_sec: float,
        total_count: int,
        class_counts: Dict[str, int],
        density: int,
        flow_per_min: float,
        traffic_level: str,
        force: bool = False
    ):
        """
        Write a row to log.csv if interval has elapsed or if forced.

        Args:
            timestamp_sec: Current video timestamp in seconds.
            total_count: Total unique vehicles counted so far.
            class_counts: Dictionary of per-class count values.
            density: Current number of vehicles in ROI.
            flow_per_min: Current flow rate (veh/min).
            traffic_level: Current traffic level string ('LOW', 'MEDIUM', 'HIGH').
            force: Force writing a row regardless of interval.
        """
        if (
            self.last_log_sec is None
            or (timestamp_sec - self.last_log_sec) >= self.log_interval_sec
            or force
        ):
            row = [
                f"{timestamp_sec:.2f}",
                total_count,
                *[class_counts.get(cls_name, 0) for cls_name in self.vehicle_classes],
                density,
                f"{flow_per_min:.2f}",
                traffic_level
            ]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.last_log_sec = timestamp_sec

    def close(self):
        """Close the CSV file handle safely."""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()

    def write_summary(
        self,
        total_count: int,
        class_counts: Dict[str, int],
        direction_counts: Dict[str, int],
        traffic_level_metrics: Dict,
        average_fps: float,
        total_duration_sec: float,
        total_frames_processed: int,
        model_name: str,
        tracker_name: str
    ):
        """
        Write end-of-run summary metrics to outputs/summary.json.
        """
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "model": model_name,
            "tracker": tracker_name,
            "total_frames_processed": total_frames_processed,
            "total_duration_sec": round(total_duration_sec, 2),
            "average_fps": round(average_fps, 2),
            "total_count": total_count,
            "per_class_counts": {k: class_counts.get(k, 0) for k in self.vehicle_classes},
            "counts_by_direction": direction_counts,
            "peak_traffic_level": traffic_level_metrics.get("peak_level", "LOW"),
            "time_spent_in_levels_sec": traffic_level_metrics.get("time_spent_in_levels_sec", {}),
            "time_spent_in_levels_pct": traffic_level_metrics.get("time_spent_in_levels_pct", {})
        }

        with open(self.summary_path, mode="w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
