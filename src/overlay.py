"""Visual overlay module for rendering bounding boxes, counting line, ROI, and HUD panel."""

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

# Color definitions in BGR
COLOR_LOW = (50, 205, 50)       # Green
COLOR_MEDIUM = (0, 165, 255)     # Amber / Orange
COLOR_HIGH = (30, 30, 220)       # Red

COLOR_PANEL_BG = (20, 24, 28)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_DIM = (180, 180, 180)
COLOR_LINE = (255, 191, 0)       # Deep sky blue / Cyan
COLOR_ROI = (180, 105, 255)      # Violet / Orchid

# Class colors for bounding boxes
CLASS_COLORS = {
    "car": (255, 120, 0),        # Azure
    "motorcycle": (0, 220, 220), # Yellow
    "bus": (200, 50, 200),       # Magenta
    "truck": (255, 80, 80),      # Blue/Teal
    "bicycle": (50, 255, 150)    # Mint green
}


class VideoOverlay:
    """Renders all visual telemetry, detections, ROI, line, and HUD panel onto video frames."""

    def __init__(
        self,
        line_p1: Tuple[int, int],
        line_p2: Tuple[int, int],
        roi_polygon: Optional[Sequence[Tuple[int, int]]] = None
    ):
        """
        Initialize the overlay drawer.

        Args:
            line_p1: (x1, y1) endpoint of the counting line.
            line_p2: (x2, y2) endpoint of the counting line.
            roi_polygon: List of (x, y) vertices for the ROI polygon.
        """
        self.line_p1 = tuple(line_p1)
        self.line_p2 = tuple(line_p2)
        self.roi_polygon = [tuple(p) for p in (roi_polygon or [])]

    def draw_roi(self, frame: np.ndarray, alpha: float = 0.15):
        """Draw semi-transparent filled ROI polygon and dashed/solid border."""
        if not self.roi_polygon or len(self.roi_polygon) < 3:
            return

        pts = np.array(self.roi_polygon, dtype=np.int32).reshape((-1, 1, 2))

        # Filled transparent overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], COLOR_ROI)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Crisp polygon border
        cv2.polylines(frame, [pts], isClosed=True, color=COLOR_ROI, thickness=2, lineType=cv2.LINE_AA)

        # Label inside or near first vertex
        label_pos = (self.roi_polygon[0][0] + 8, self.roi_polygon[0][1] + 20)
        cv2.putText(
            frame,
            "ROI (Density Zone)",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_ROI,
            1,
            cv2.LINE_AA
        )

    def draw_counting_line(self, frame: np.ndarray, is_crossing_active: bool = False):
        """Draw virtual counting line and direction arrows."""
        color = (0, 255, 255) if is_crossing_active else COLOR_LINE
        thickness = 3 if is_crossing_active else 2

        p1 = (int(self.line_p1[0]), int(self.line_p1[1]))
        p2 = (int(self.line_p2[0]), int(self.line_p2[1]))

        cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(frame, p1, 5, color, -1)
        cv2.circle(frame, p2, 5, color, -1)

        # Label line
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        cv2.putText(
            frame,
            "COUNTING LINE",
            (mid_x - 50, mid_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    def draw_vehicles(self, frame: np.ndarray, vehicles: Sequence):
        """
        Draw bounding boxes, track IDs, class names, and centroids for active vehicles.

        Args:
            frame: Target BGR image frame.
            vehicles: Iterable of TrackedVehicle objects.
        """
        for v in vehicles:
            x1, y1, x2, y2 = [int(coord) for coord in v.bbox]
            color = CLASS_COLORS.get(v.class_name.lower(), (200, 200, 200))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Centroid point
            cx, cy = int(v.centroid[0]), int(v.centroid[1])
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Label banner
            label = f"#{v.track_id} {v.class_name} {v.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - h - 6)), (x1 + w + 4, max(h + 6, y1)), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, max(h, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (10, 10, 10),
                1,
                cv2.LINE_AA
            )

    def draw_hud(
        self,
        frame: np.ndarray,
        traffic_level: str,
        total_count: int,
        class_counts: Dict[str, int],
        direction_counts: Dict[str, int],
        density_in_roi: int,
        flow_per_minute: float,
        fps: float,
        timestamp_sec: float
    ):
        """
        Draw informational telemetry HUD panel on the upper-left of the frame.
        """
        # Determine level color
        lvl_upper = traffic_level.upper()
        if lvl_upper == "HIGH":
            badge_color = COLOR_HIGH
        elif lvl_upper == "MEDIUM":
            badge_color = COLOR_MEDIUM
        else:
            badge_color = COLOR_LOW

        # Panel geometry
        panel_x = 16
        panel_y = 16
        panel_w = 340
        panel_h = 240

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLOR_PANEL_BG,
            -1
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (80, 80, 80),
            1,
            cv2.LINE_AA
        )

        # Title
        cv2.putText(
            frame,
            "TRAFFIC MONITORING",
            (panel_x + 12, panel_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA
        )

        # Traffic Level Badge
        badge_w = 110
        badge_h = 28
        badge_x = panel_x + panel_w - badge_w - 12
        badge_y = panel_y + 8
        cv2.rectangle(
            frame,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            badge_color,
            -1
        )
        (text_w, text_h), _ = cv2.getTextSize(lvl_upper, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_x = badge_x + (badge_w - text_w) // 2
        text_y = badge_y + (badge_h + text_h) // 2 - 1
        cv2.putText(
            frame,
            lvl_upper,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Line separator
        sep_y = panel_y + 42
        cv2.line(frame, (panel_x + 10, sep_y), (panel_x + panel_w - 10, sep_y), (60, 65, 70), 1)

        # Metrics rows
        y_cursor = sep_y + 22
        line_height = 22

        # 1. Flow & Density
        cv2.putText(
            frame,
            f"Flow: {flow_per_minute:.1f} veh/min",
            (panel_x + 12, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"In ROI: {density_in_roi}",
            (panel_x + 200, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA
        )

        # 2. Total Count & Directions
        y_cursor += line_height
        cv2.putText(
            frame,
            f"Total Counted: {total_count}",
            (panel_x + 12, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (50, 255, 255),
            1,
            cv2.LINE_AA
        )
        in_c = direction_counts.get("In", 0)
        out_c = direction_counts.get("Out", 0)
        cv2.putText(
            frame,
            f"(In: {in_c} | Out: {out_c})",
            (panel_x + 190, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA
        )

        # 3. Class Counts Header / Breakdown
        y_cursor += line_height
        cv2.putText(
            frame,
            "Vehicle Breakdown:",
            (panel_x + 12, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA
        )

        y_cursor += line_height
        car_c = class_counts.get("car", 0)
        bike_c = class_counts.get("motorcycle", 0) + class_counts.get("bicycle", 0)
        bus_c = class_counts.get("bus", 0)
        truck_c = class_counts.get("truck", 0)

        cv2.putText(
            frame,
            f"Cars: {car_c}   Bikes: {bike_c}",
            (panel_x + 24, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA
        )
        y_cursor += line_height - 2
        cv2.putText(
            frame,
            f"Buses: {bus_c}   Trucks: {truck_c}",
            (panel_x + 24, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA
        )

        # 4. Status Bar (FPS / Timestamp)
        y_cursor += line_height + 4
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}  |  Time: {timestamp_sec:.1f}s",
            (panel_x + 12, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (150, 160, 170),
            1,
            cv2.LINE_AA
        )
