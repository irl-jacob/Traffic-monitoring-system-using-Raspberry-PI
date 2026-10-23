"""YOLO object detector and tracker wrapper with dynamic vehicle class resolution."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard target vehicle class names in COCO
TARGET_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}


@dataclass
class TrackedVehicle:
    """Represents a detected and tracked vehicle in the current frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]             # (cx, cy)


class DetectorTracker:
    """Wrapper around Ultralytics YOLO for vehicle tracking."""

    def __init__(
        self,
        weights: str = "yolo26n.pt",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "auto"
    ):
        """
        Initialize detector and tracker.

        Args:
            weights: Path or name of YOLO weights (e.g. yolo26n.pt, yolo11n.pt, yolov8n.pt).
            tracker: Tracker config ('bytetrack.yaml' or 'botsort.yaml').
            conf: Confidence threshold.
            imgsz: Input image dimension for inference.
            device: 'auto', 'cpu', 'cuda', etc.
        """
        self.weights = weights
        self.tracker = tracker
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = self._resolve_device(device)

        self.model = self._load_model(weights)
        self.vehicle_classes, self.class_id_to_name = self._resolve_vehicle_classes()

        logger.info(
            f"Initialized DetectorTracker with model={self.weights}, tracker={self.tracker}, "
            f"device={self.device}, classes={list(self.class_id_to_name.values())}"
        )

    def _resolve_device(self, device_str: str) -> str:
        """Resolve device string, falling back to CPU if CUDA is unavailable."""
        if device_str == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device_str

    def _load_model(self, weights: str):
        """Load YOLO model with automatic fallback to popular pretrained weights if needed."""
        from ultralytics import YOLO

        candidates = [weights]
        # Include popular fallbacks if the primary weight fails to download or initialize
        if weights.startswith("yolo26"):
            candidates.extend(["yolo11n.pt", "yolov8n.pt"])
        elif weights.startswith("yolo11"):
            candidates.append("yolov8n.pt")

        last_error = None
        for candidate in candidates:
            try:
                logger.info(f"Loading YOLO weights from: {candidate}")
                model = YOLO(candidate)
                self.weights = candidate
                return model
            except Exception as e:
                logger.warning(f"Could not load weights '{candidate}': {e}")
                last_error = e

        raise RuntimeError(f"Failed to load any YOLO weights: {last_error}")

    def _resolve_vehicle_classes(self) -> Tuple[List[int], Dict[int, str]]:
        """
        Dynamically query model.names to extract vehicle class IDs.
        Ensures compliance with requirement not to hardcode class indices.
        """
        model_names = self.model.names  # Dict[int, str]
        vehicle_ids: List[int] = []
        id_to_name: Dict[int, str] = {}

        for cls_id, name in model_names.items():
            clean_name = str(name).lower().strip()
            if clean_name in TARGET_VEHICLE_NAMES:
                vehicle_ids.append(int(cls_id))
                id_to_name[int(cls_id)] = clean_name

        if not vehicle_ids:
            logger.warning("No standard vehicle classes found in model.names; using all classes.")
            vehicle_ids = list(model_names.keys())
            id_to_name = {int(k): str(v).lower() for k, v in model_names.items()}

        return vehicle_ids, id_to_name

    def track(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Run detection and tracking on a single frame.

        Args:
            frame: BGR numpy image frame.

        Returns:
            List of TrackedVehicle objects detected and tracked in this frame.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            classes=self.vehicle_classes,
            tracker=self.tracker,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        tracked_vehicles: List[TrackedVehicle] = []
        if not results or len(results) == 0:
            return tracked_vehicles

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracked_vehicles

        xyxy_arr = boxes.xyxy.cpu().numpy()
        ids_arr = boxes.id.int().cpu().numpy()
        cls_arr = boxes.cls.int().cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()

        for xyxy, track_id, cls_id, conf in zip(xyxy_arr, ids_arr, cls_arr, conf_arr):
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cls_name = self.class_id_to_name.get(cls_id, str(cls_id))

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=int(track_id),
                    class_id=int(cls_id),
                    class_name=cls_name,
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    centroid=(cx, cy)
                )
            )

        return tracked_vehicles
