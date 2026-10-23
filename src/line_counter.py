"""Line crossing counter module for vehicle detection and tracking."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CrossingEvent:
    """Record of a vehicle crossing event."""
    timestamp: float
    track_id: int
    class_name: str
    direction: str
    centroid: Tuple[float, float]


def _ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Signed area / cross product of triangle ABC."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """Check if segment (p1, p2) intersects segment (p3, p4)."""
    d1 = _ccw(p1, p2, p3)
    d2 = _ccw(p1, p2, p4)
    d3 = _ccw(p3, p4, p1)
    d4 = _ccw(p3, p4, p2)

    # Segments intersect if endpoints of each segment are on opposite sides of the other
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


class LineCounter:
    """Tracks vehicle centroids and counts unique vehicles crossing a virtual line."""

    def __init__(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        classes: Optional[List[str]] = None,
        max_missing_frames: int = 120
    ):
        """
        Initialize the line crossing counter.

        Args:
            p1: First endpoint of the virtual line (x1, y1).
            p2: Second endpoint of the virtual line (x2, y2).
            classes: Optional list of class names to initialize per-class counters.
            max_missing_frames: Frames before dropping inactive track history.
        """
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.max_missing_frames = max_missing_frames

        # Counting state
        self.counted_ids: Set[int] = set()
        self.total_count: int = 0
        self.class_counts: Dict[str, int] = {c: 0 for c in (classes or ["car", "motorcycle", "bus", "truck", "bicycle"])}
        self.direction_counts: Dict[str, int] = {"In": 0, "Out": 0}

        # Track history: track_id -> (centroid, last_seen_frame)
        self.track_history: Dict[int, Tuple[Tuple[float, float], int]] = {}
        self.events: List[CrossingEvent] = []

    def update(
        self,
        track_id: int,
        class_name: str,
        current_centroid: Tuple[float, float],
        frame_idx: int = 0,
        timestamp: float = 0.0
    ) -> Optional[CrossingEvent]:
        """
        Update a vehicle's position and check if it crossed the virtual line.

        Args:
            track_id: Unique tracking ID from tracker.
            class_name: Vehicle class name (e.g. 'car', 'motorcycle').
            current_centroid: Current centroid (cx, cy).
            frame_idx: Current frame index.
            timestamp: Current video timestamp in seconds.

        Returns:
            CrossingEvent if a crossing occurred on this update, else None.
        """
        event: Optional[CrossingEvent] = None

        if track_id in self.track_history:
            prev_centroid, _ = self.track_history[track_id]

            # Only evaluate crossing if not already counted (FR4: count each vehicle once)
            if track_id not in self.counted_ids:
                if segments_intersect(self.p1, self.p2, prev_centroid, current_centroid):
                    # Determine crossing direction based on signed cross-product
                    # Moving from d_prev < 0 to d_curr > 0 is "In", opposite is "Out"
                    d_prev = _ccw(self.p1, self.p2, prev_centroid)
                    direction = "In" if d_prev < 0 else "Out"

                    # Register count
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                    self.direction_counts[direction] = self.direction_counts.get(direction, 0) + 1

                    event = CrossingEvent(
                        timestamp=timestamp,
                        track_id=track_id,
                        class_name=class_name,
                        direction=direction,
                        centroid=current_centroid
                    )
                    self.events.append(event)

        # Update centroid & seen frame
        self.track_history[track_id] = (current_centroid, frame_idx)
        return event

    def cleanup_old_tracks(self, current_frame_idx: int):
        """Purge tracks that have been inactive for more than max_missing_frames."""
        stale_ids = [
            tid for tid, (_, last_seen) in self.track_history.items()
            if current_frame_idx - last_seen > self.max_missing_frames
        ]
        for tid in stale_ids:
            del self.track_history[tid]

    def reset(self):
        """Reset all counters and tracking history."""
        self.counted_ids.clear()
        self.total_count = 0
        for k in self.class_counts:
            self.class_counts[k] = 0
        for k in self.direction_counts:
            self.direction_counts[k] = 0
        self.track_history.clear()
        self.events.clear()
