"""Traffic level computation engine with rolling window flow, ROI density, and hysteresis."""

import time
from collections import deque
from enum import IntEnum
from typing import Deque, Dict, List, Optional, Sequence, Tuple


class TrafficLevel(IntEnum):
    """Traffic congestion levels ordered by severity."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @classmethod
    def from_string(cls, name: str) -> "TrafficLevel":
        return cls[name.upper()]

    def to_string(self) -> str:
        return self.name


def is_point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    """
    Ray-casting algorithm to test whether a point is inside a 2D polygon.
    Works independently of OpenCV (portable for testing and pure Python).
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class TrafficLevelEngine:
    """Computes traffic metrics and evaluates traffic level with hysteresis."""

    def __init__(
        self,
        flow_window_sec: float = 30.0,
        hysteresis_sec: float = 2.0,
        flow_thresholds: Optional[Dict[str, float]] = None,
        density_thresholds: Optional[Dict[str, int]] = None,
        roi_polygon: Optional[Sequence[Tuple[float, float]]] = None
    ):
        """
        Initialize the traffic level engine.

        Args:
            flow_window_sec: Rolling window length in seconds for flow calculation.
            hysteresis_sec: Debounce time in seconds required before changing level.
            flow_thresholds: Dict with 'low_max' and 'medium_max' for flow (veh/min).
            density_thresholds: Dict with 'low_max' and 'medium_max' for vehicle count in ROI.
            roi_polygon: List of (x, y) coordinates defining the ROI polygon.
        """
        self.flow_window_sec = float(flow_window_sec)
        self.hysteresis_sec = float(hysteresis_sec)

        self.flow_thresholds = flow_thresholds or {"low_max": 8.0, "medium_max": 20.0}
        self.density_thresholds = density_thresholds or {"low_max": 3, "medium_max": 8}
        self.roi_polygon = [tuple(p) for p in (roi_polygon or [])]

        # Crossing timestamps deque for rolling window
        self.crossing_timestamps: Deque[float] = deque()

        # Engine state
        self.start_timestamp: Optional[float] = None
        self.last_update_timestamp: Optional[float] = None

        self.reported_level: TrafficLevel = TrafficLevel.LOW
        self.candidate_level: Optional[TrafficLevel] = None
        self.candidate_since: Optional[float] = None

        # Cumulative metrics
        self.time_in_level: Dict[TrafficLevel, float] = {
            TrafficLevel.LOW: 0.0,
            TrafficLevel.MEDIUM: 0.0,
            TrafficLevel.HIGH: 0.0
        }
        self.peak_level: TrafficLevel = TrafficLevel.LOW

    def record_crossing(self, timestamp: float):
        """Record a line crossing event timestamp."""
        self.crossing_timestamps.append(timestamp)

    def compute_flow_per_minute(self, current_time: float) -> float:
        """Calculate rolling window flow rate scaled to vehicles per minute."""
        # Evict timestamps older than current_time - flow_window_sec
        cutoff = current_time - self.flow_window_sec
        while self.crossing_timestamps and self.crossing_timestamps[0] < cutoff:
            self.crossing_timestamps.popleft()

        count = len(self.crossing_timestamps)
        if self.start_timestamp is None:
            self.start_timestamp = current_time

        elapsed = current_time - self.start_timestamp
        effective_window = min(self.flow_window_sec, max(1.0, elapsed))

        flow_per_min = (count / effective_window) * 60.0
        return flow_per_min

    def compute_density_in_roi(self, vehicle_centroids: Sequence[Tuple[float, float]]) -> int:
        """Count how many vehicle centroids are currently inside the ROI polygon."""
        if not self.roi_polygon:
            return 0

        density = sum(1 for c in vehicle_centroids if is_point_in_polygon(c, self.roi_polygon))
        return density

    def classify_flow(self, flow: float) -> TrafficLevel:
        """Classify flow rate into LOW, MEDIUM, or HIGH."""
        if flow < self.flow_thresholds["low_max"]:
            return TrafficLevel.LOW
        elif flow <= self.flow_thresholds["medium_max"]:
            return TrafficLevel.MEDIUM
        return TrafficLevel.HIGH

    def classify_density(self, density: int) -> TrafficLevel:
        """Classify vehicle density in ROI into LOW, MEDIUM, or HIGH."""
        if density <= self.density_thresholds["low_max"]:
            return TrafficLevel.LOW
        elif density <= self.density_thresholds["medium_max"]:
            return TrafficLevel.MEDIUM
        return TrafficLevel.HIGH

    def update(
        self,
        current_time: float,
        active_centroids: Sequence[Tuple[float, float]],
        new_crossing_count: int = 0
    ) -> Tuple[TrafficLevel, float, int]:
        """
        Update traffic engine with current frame state.

        Args:
            current_time: Video timestamp in seconds (or monotonic time).
            active_centroids: List of (cx, cy) centroids for all currently tracked vehicles.
            new_crossing_count: Number of crossings detected on this frame.

        Returns:
            Tuple of (reported_level, flow_per_minute, density_in_roi).
        """
        if self.start_timestamp is None:
            self.start_timestamp = current_time

        # Update cumulative time in level
        if self.last_update_timestamp is not None:
            dt = max(0.0, current_time - self.last_update_timestamp)
            self.time_in_level[self.reported_level] += dt
        self.last_update_timestamp = current_time

        for _ in range(new_crossing_count):
            self.record_crossing(current_time)

        flow_per_min = self.compute_flow_per_minute(current_time)
        density = self.compute_density_in_roi(active_centroids)

        flow_lvl = self.classify_flow(flow_per_min)
        density_lvl = self.classify_density(density)

        # Combined level: highest severity of flow or density
        raw_level = max(flow_lvl, density_lvl)

        # Apply hysteresis
        if raw_level == self.reported_level:
            self.candidate_level = None
            self.candidate_since = None
        else:
            if self.candidate_level != raw_level:
                self.candidate_level = raw_level
                self.candidate_since = current_time
            elif self.candidate_since is not None and (current_time - self.candidate_since) >= self.hysteresis_sec:
                self.reported_level = raw_level
                self.candidate_level = None
                self.candidate_since = None

        if self.reported_level > self.peak_level:
            self.peak_level = self.reported_level

        return self.reported_level, flow_per_min, density

    def finalize(self, end_time: Optional[float] = None):
        """Flush final time accounting at the end of the video."""
        if self.last_update_timestamp is not None and end_time is not None:
            dt = max(0.0, end_time - self.last_update_timestamp)
            self.time_in_level[self.reported_level] += dt
            self.last_update_timestamp = end_time

    def get_summary_metrics(self) -> Dict:
        """Return cumulative summary metrics for report generation."""
        total_time = sum(self.time_in_level.values())
        return {
            "peak_level": self.peak_level.name,
            "time_spent_in_levels_sec": {
                lvl.name: round(dur, 2) for lvl, dur in self.time_in_level.items()
            },
            "time_spent_in_levels_pct": {
                lvl.name: round((dur / total_time * 100.0) if total_time > 0 else 0.0, 1)
                for lvl, dur in self.time_in_level.items()
            },
            "total_monitoring_duration_sec": round(total_time, 2)
        }
