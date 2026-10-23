#!/usr/bin/env python3
"""
Traffic Monitoring System with YOLO
Course Use Case 15: Vehicle Detection + Tracking + Counting + Traffic Level Analysis

Main CLI entrypoint and orchestrator loop.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

from src.detector_tracker import DetectorTracker
from src.line_counter import LineCounter
from src.logger import TrafficLogger
from src.overlay import VideoOverlay
from src.traffic_level import TrafficLevelEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrafficMonitor")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.is_file():
        logger.warning(f"Config file not found at {config_path}; using internal defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with precedence over config.yaml."""
    parser = argparse.ArgumentParser(
        description="Traffic Monitoring System with YOLO (Course Use Case 15)"
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file or webcam index (e.g. 0).")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--show", action="store_true",
                        help="Show live visualization window.")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save annotated output video (e.g. outputs/demo.mp4).")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO model weights (e.g. yolo26n.pt, yolo11n.pt, yolov8n.pt).")
    parser.add_argument("--tracker", type=str, default=None,
                        help="Tracker YAML config (e.g. bytetrack.yaml or botsort.yaml).")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold for vehicle detection.")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference image resolution size.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference ('auto', 'cpu', 'cuda:0').")
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="Process every Nth frame (useful for CPU / Raspberry Pi).")
    parser.add_argument("--line-coords", type=int, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Override counting line coordinates: x1 y1 x2 y2.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactive mouse calibration on the first frame.")
    return parser.parse_args()


def scale_point_to_frame(pt: Tuple[float, float], width: int, height: int, base_w: int = 1280, base_h: int = 720) -> Tuple[int, int]:
    """Adapt coordinates to frame dimensions if necessary."""
    x, y = pt
    if isinstance(x, float) and isinstance(y, float) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(x * width), int(y * height)
    if (width != base_w or height != base_h) and (x > width or y > height):
        scale_x = width / float(base_w)
        scale_y = height / float(base_h)
        return int(x * scale_x), int(y * scale_y)
    return int(min(max(0, x), width)), int(min(max(0, y), height))


def interactive_calibrate(frame: cv2.Mat) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Interactive mouse callback to set counting line endpoints."""
    clone = frame.copy()
    points: List[Tuple[int, int]] = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 255), -1)
            if len(points) == 2:
                cv2.line(clone, points[0], points[1], (0, 255, 255), 2)
            cv2.imshow("Calibrate Counting Line (Click 2 points, then press ENTER)", clone)

    window_name = "Calibrate Counting Line (Click 2 points, then press ENTER)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, clone)

    logger.info("Click 2 points on the frame to define the counting line, then press ENTER.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32, 27) and len(points) == 2:  # ENTER or SPACE or ESC
            break
        if key == ord('r'):  # Reset
            points.clear()
            clone = frame.copy()
            cv2.imshow(window_name, clone)

    cv2.destroyWindow(window_name)
    return points[0], points[1]


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve settings (CLI overrides config.yaml, then internal defaults)
    video_cfg = cfg.get("video", {})
    model_cfg = cfg.get("model", {})
    line_cfg = cfg.get("counting_line", {})
    roi_cfg = cfg.get("roi_polygon", {})
    level_cfg = cfg.get("traffic_level", {})
    log_cfg = cfg.get("logging", {})

    source_input = args.source if args.source is not None else video_cfg.get("source", "videos/road.mp4")
    # Parse integer webcam index if digits
    source = int(source_input) if str(source_input).isdigit() else str(source_input)

    model_weights = args.model or model_cfg.get("weights", "yolo26n.pt")
    tracker = args.tracker or model_cfg.get("tracker", "bytetrack.yaml")
    conf = args.conf if args.conf is not None else video_cfg.get("conf", 0.25)
    imgsz = args.imgsz if args.imgsz is not None else video_cfg.get("imgsz", 640)
    device = args.device or model_cfg.get("device", "auto")
    frame_skip = args.frame_skip if args.frame_skip is not None else video_cfg.get("frame_skip", 1)
    frame_skip = max(1, int(frame_skip))

    # Counting line coordinates
    if args.line_coords:
        line_p1 = (args.line_coords[0], args.line_coords[1])
        line_p2 = (args.line_coords[2], args.line_coords[3])
    else:
        line_p1 = tuple(line_cfg.get("p1", [100, 420]))
        line_p2 = tuple(line_cfg.get("p2", [1180, 420]))

    # ROI polygon
    roi_points = [tuple(p) for p in roi_cfg.get("points", [
        [100, 220], [1180, 220], [1180, 620], [100, 620]
    ])]

    # Logging paths
    csv_path = log_cfg.get("csv_path", "outputs/log.csv")
    summary_path = log_cfg.get("summary_path", "outputs/summary.json")
    log_interval_sec = log_cfg.get("log_interval_sec", 1.0)

    # Initialize video capture
    logger.info(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source '{source}'. Please verify the path or camera index.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if video_fps <= 0 or video_fps > 120:
        video_fps = 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Source specs: {width}x{height} @ {video_fps:.1f} FPS (Total frames: {total_video_frames})")

    # Scale coordinates to video resolution if not explicitly overridden by CLI
    if not args.line_coords:
        line_p1 = scale_point_to_frame(line_p1, width, height)
        line_p2 = scale_point_to_frame(line_p2, width, height)
    roi_points = [scale_point_to_frame(pt, width, height) for pt in roi_points]

    # Interactive calibration if requested
    if args.calibrate and args.show:
        ret, first_frame = cap.read()
        if ret:
            line_p1, line_p2 = interactive_calibrate(first_frame)
            logger.info(f"Calibrated counting line: P1={line_p1}, P2={line_p2}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize core modules
    detector = DetectorTracker(
        weights=model_weights,
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        device=device
    )

    line_counter = LineCounter(
        p1=line_p1,
        p2=line_p2,
        classes=list(detector.class_id_to_name.values())
    )

    traffic_engine = TrafficLevelEngine(
        flow_window_sec=level_cfg.get("flow_window_sec", 30.0),
        hysteresis_sec=level_cfg.get("hysteresis_sec", 2.0),
        flow_thresholds=level_cfg.get("flow_thresholds"),
        density_thresholds=level_cfg.get("density_thresholds"),
        roi_polygon=roi_points
    )

    overlay = VideoOverlay(
        line_p1=line_p1,
        line_p2=line_p2,
        roi_polygon=roi_points
    )

    logger_handler = TrafficLogger(
        csv_path=csv_path,
        summary_path=summary_path,
        log_interval_sec=log_interval_sec,
        vehicle_classes=["car", "motorcycle", "bus", "truck", "bicycle"]
    )

    # Initialize video writer if --save is requested
    writer = None
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = video_fps / frame_skip
        writer = cv2.VideoWriter(str(save_path), fourcc, effective_fps, (width, height))
        logger.info(f"Saving output video to: {save_path} @ {effective_fps:.1f} FPS")

    # Processing loop state
    frame_idx = 0
    processed_count = 0
    start_real_time = time.time()
    last_fps_time = time.time()
    current_fps = 0.0
    paused = False

    logger.info("Traffic monitoring started. Press 'q' to quit, 'p' to pause/resume.")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream reached.")
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                processed_count += 1
                timestamp_sec = frame_idx / video_fps

                # 1. Detect and track
                tracked_vehicles = detector.track(frame)
                active_centroids = [v.centroid for v in tracked_vehicles]

                # 2. Update line crossing counter
                crossing_on_frame = False
                for v in tracked_vehicles:
                    event = line_counter.update(
                        track_id=v.track_id,
                        class_name=v.class_name,
                        current_centroid=v.centroid,
                        frame_idx=frame_idx,
                        timestamp=timestamp_sec
                    )
                    if event is not None:
                        crossing_on_frame = True
                        traffic_engine.record_crossing(timestamp_sec)

                line_counter.cleanup_old_tracks(frame_idx)

                # 3. Update traffic level engine
                traffic_level, flow_per_min, density = traffic_engine.update(
                    current_time=timestamp_sec,
                    active_centroids=active_centroids,
                    new_crossing_count=0  # already recorded above
                )

                # Calculate smoothed FPS
                now = time.time()
                dt = now - last_fps_time
                if dt >= 0.5:
                    current_fps = (processed_count / (now - start_real_time))
                    last_fps_time = now

                # 4. Log metrics to CSV
                logger_handler.log_step(
                    timestamp_sec=timestamp_sec,
                    total_count=line_counter.total_count,
                    class_counts=line_counter.class_counts,
                    density=density,
                    flow_per_min=flow_per_min,
                    traffic_level=traffic_level.name
                )

                # 5. Render overlays
                overlay.draw_roi(frame)
                overlay.draw_counting_line(frame, is_crossing_active=crossing_on_frame)
                overlay.draw_vehicles(frame, tracked_vehicles)
                overlay.draw_hud(
                    frame=frame,
                    traffic_level=traffic_level.name,
                    total_count=line_counter.total_count,
                    class_counts=line_counter.class_counts,
                    direction_counts=line_counter.direction_counts,
                    density_in_roi=density,
                    flow_per_minute=flow_per_min,
                    fps=current_fps,
                    timestamp_sec=timestamp_sec
                )

                # 6. Save video frame
                if writer is not None:
                    writer.write(frame)

                # 7. Display window
                if args.show:
                    cv2.imshow("Traffic Monitoring System (Course Use Case 15)", frame)

            # Keyboard controls
            if args.show:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q'):
                    logger.info("User requested exit ('q').")
                    break
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"Playback {'paused' if paused else 'resumed'}.")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user (SIGINT).")

    finally:
        total_time_sec = (frame_idx / video_fps) if video_fps > 0 else 0.0
        traffic_engine.finalize(total_time_sec)

        # Write final log entry and summary.json
        logger_handler.close()
        avg_fps = (processed_count / (time.time() - start_real_time)) if (time.time() - start_real_time) > 0 else 0.0
        traffic_engine_summary = traffic_engine.get_summary_metrics()

        logger_handler.write_summary(
            total_count=line_counter.total_count,
            class_counts=line_counter.class_counts,
            direction_counts=line_counter.direction_counts,
            traffic_level_metrics=traffic_engine_summary,
            average_fps=avg_fps,
            total_duration_sec=total_time_sec,
            total_frames_processed=processed_count,
            model_name=detector.weights,
            tracker_name=tracker
        )

        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

        logger.info("Execution complete.")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"CSV log saved to: {csv_path}")
        if args.save:
            logger.info(f"Output video saved to: {args.save}")


if __name__ == "__main__":
    main()
