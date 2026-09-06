from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.schemas.metrics import CongestionState, TrafficMetrics

# Display order for the per-class breakdown in the summary panel; anything not listed here
# (an unexpected class name) is appended afterwards rather than dropped.
_CLASS_DISPLAY_ORDER = ["car", "bus", "truck", "motorcycle", "bicycle"]
_CLASS_DISPLAY_LABEL = {
    "car": "Cars",
    "bus": "Buses",
    "truck": "Trucks",
    "motorcycle": "Motorcycles",
    "bicycle": "Bicycles",
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_CONGESTION_COLORS = {
    CongestionState.LOW: (110, 200, 90),
    CongestionState.MODERATE: (0, 179, 255),
    CongestionState.HIGH: (60, 60, 220),
}

_PERSON_COLOR = (222, 196, 60)
_VEHICLE_COLORS = {
    "car": (231, 158, 40),
    "bus": (168, 76, 173),
    "truck": (0, 149, 255),
    "motorcycle": (66, 66, 214),
    "bicycle": (110, 190, 90),
}
_FALLBACK_VEHICLE_COLOR = (190, 190, 190)

_PANEL_BG = (32, 30, 28)
_PANEL_TEXT = (240, 240, 240)
_PANEL_MUTED_TEXT = (165, 165, 165)

# Sizing is derived from frame width relative to this reference so labels and the panel stay
# legible from small clips through 4K, instead of a fixed pixel size tuned for one resolution.
# _MIN_SCALE is well above 1.0 because "legible at 1x" on a 1280px-wide frame still reads as
# small text on a modern display -- most everything is bumped up from there.
_REFERENCE_WIDTH = 1280.0
_MIN_SCALE = 1.3
_MAX_SCALE = 4.0

# Below this fraction of the frame's width, a vehicle is small/distant enough that a handful
# of them packed into one lane near the horizon sit only a few pixels apart -- a full
# "#id class - N km/h" banner for each one overlaps its neighbors into an unreadable smear.
# These get a compact "#id"-only label instead: shorter, and drawn as outlined text with no
# filled background, so even where two labels do overlap it's a couple of thin glyphs on top
# of each other, not one opaque block hiding several vehicles.
_COMPACT_LABEL_MAX_WIDTH_RATIO = 0.065


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    if class_name == "person":
        return _PERSON_COLOR
    return _VEHICLE_COLORS.get(class_name, _FALLBACK_VEHICLE_COLOR)


def _text_color_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (20, 20, 20) if sum(color) > 380 else (245, 245, 245)


def _put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _put_label_outlined(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    """Like _put_label, but with a dark halo behind the glyphs -- the panel sits over live
    video, so a light color alone can wash out against a bright patch of frame behind it."""
    cv2.putText(frame, text, origin, _FONT, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, _FONT, font_scale, color, thickness, cv2.LINE_AA)


class FrameAnnotator:
    """Renders bounding boxes, per-track trails, a live summary panel (counts, traffic level,
    density sparkline), and an end-of-video summary card."""

    def __init__(self, trail_length: int = 20) -> None:
        self._trail_length = trail_length

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[TrackedDetection],
        trails: dict[int, list[tuple[float, float]]],
        counts_by_class: Counter[str],
        person_count: int,
        traffic_level: CongestionState,
        density_history: list[int],
        peak_vehicle_count: int,
    ) -> np.ndarray:
        scale = min(_MAX_SCALE, max(_MIN_SCALE, frame.shape[1] / _REFERENCE_WIDTH))
        annotated = frame.copy()
        for detection in detections:
            self._draw_detection(
                annotated,
                detection,
                trails.get(detection.track_id, []),
                scale,
                frame.shape[1],
            )
        self._draw_summary_panel(
            annotated, counts_by_class, person_count, traffic_level, density_history, peak_vehicle_count, scale
        )
        return annotated

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: TrackedDetection,
        trail: list[tuple[float, float]],
        scale: float,
        frame_width: int,
    ) -> None:
        color = _color_for_class(detection.class_name)
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        is_compact = frame_width > 0 and (x2 - x1) / frame_width < _COMPACT_LABEL_MAX_WIDTH_RATIO

        box_thickness = max(1, round(1.4 * scale)) if is_compact else max(2, round(2.4 * scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        if is_compact:
            self._draw_compact_label(frame, detection, x1, y1, color, scale)
        else:
            self._draw_full_label(frame, detection, x1, y1, color, scale)

        # A packed cluster of small/distant vehicles produces a tangle of full-length trails
        # that adds more visual noise than signal; a short, thin trail still shows which way
        # each one is moving without drowning the cluster in overlapping lines.
        trail_length = 6 if is_compact else self._trail_length
        recent_trail = trail[-trail_length:]
        if len(recent_trail) >= 2:
            points = np.array(recent_trail, dtype=np.int32)
            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=color,
                thickness=1 if is_compact else max(2, round(2.2 * scale)),
                lineType=cv2.LINE_AA,
            )

    def _draw_compact_label(
        self,
        frame: np.ndarray,
        detection: TrackedDetection,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        label = f"#{detection.track_id}"
        font_scale = max(0.32, 0.4 * scale)
        origin = (x1 + 1, max(12, y1 - 3))
        _put_label_outlined(frame, label, origin, color, font_scale, thickness=1)

    def _draw_full_label(
        self,
        frame: np.ndarray,
        detection: TrackedDetection,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        label = f"#{detection.track_id} {detection.class_name}"

        font_scale = 0.62 * scale
        text_thickness = 2 if scale >= 1.6 else 1
        pad = max(4, round(3.5 * scale))
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, text_thickness)
        text_color = _text_color_for_background(color)

        label_top = y1 - text_h - baseline - 2 * pad
        if label_top < 0:
            # Not enough room above the box (it starts near the top of the frame) -- draw the
            # label banner just inside the box instead of letting it clip off-screen.
            bg_top, bg_bottom = y1, y1 + text_h + baseline + 2 * pad
        else:
            bg_top, bg_bottom = label_top, y1
        cv2.rectangle(frame, (x1, bg_top), (x1 + text_w + 2 * pad, bg_bottom), color, thickness=-1)
        _put_label(frame, label, (x1 + pad, bg_bottom - pad - baseline), text_color, font_scale, text_thickness)

    def _draw_summary_panel(
        self,
        frame: np.ndarray,
        counts_by_class: Counter[str],
        person_count: int,
        traffic_level: CongestionState,
        density_history: list[int],
        peak_vehicle_count: int,
        scale: float,
    ) -> None:
        level_color = _CONGESTION_COLORS[traffic_level]
        vehicle_count = sum(counts_by_class.values())

        # Only list classes actually present this frame (in a fixed order, with any unknown
        # class appended) -- a dashboard row for "Buses  0" on every single frame is just
        # clutter on a clip that never has one.
        present_classes = [c for c in _CLASS_DISPLAY_ORDER if counts_by_class.get(c)]
        present_classes += [c for c in counts_by_class if c not in _CLASS_DISPLAY_ORDER and counts_by_class[c]]
        breakdown_rows = [(_CLASS_DISPLAY_LABEL.get(c, c.title()), counts_by_class[c]) for c in present_classes]

        margin = round(20 * scale)
        line_gap = round(40 * scale)
        header_rows = 4  # Vehicles total, People, Traffic level, Density label
        sparkline_height = round(34 * scale)
        width = round(380 * scale)
        height = round(88 * scale) + line_gap * (header_rows + len(breakdown_rows)) + sparkline_height
        x0, y0 = margin, margin
        x1, y1 = x0 + width, y0 + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _PANEL_BG, thickness=-1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), thickness=max(1, round(scale)))

        accent_width = max(6, round(8 * scale))
        cv2.rectangle(frame, (x0, y0), (x0 + accent_width, y1), level_color, thickness=-1)

        text_x = x0 + accent_width + round(16 * scale)
        line_scale = 0.9 * scale
        text_thickness = 2 if scale >= 1.6 else 1
        row_y = y0 + round(44 * scale)

        _put_label_outlined(
            frame, f"Vehicles  {vehicle_count}", (text_x, row_y), _PANEL_TEXT, line_scale * 1.1, text_thickness
        )
        for label, count in breakdown_rows:
            row_y += line_gap
            _put_label_outlined(
                frame, f"  {label}  {count}", (text_x, row_y), _PANEL_MUTED_TEXT, line_scale * 0.9, text_thickness
            )

        row_y += line_gap
        _put_label_outlined(
            frame, f"People    {person_count}", (text_x, row_y), _PANEL_TEXT, line_scale * 1.1, text_thickness
        )

        row_y += line_gap
        _put_label_outlined(
            frame,
            f"Traffic   {traffic_level.value}",
            (text_x, row_y),
            level_color,
            line_scale * 1.2,
            text_thickness,
        )

        row_y += line_gap
        _put_label_outlined(
            frame,
            f"Density (peak {peak_vehicle_count})",
            (text_x, row_y),
            _PANEL_MUTED_TEXT,
            line_scale * 0.85,
            text_thickness,
        )
        chart_top = row_y + round(8 * scale)
        chart_width = width - accent_width - round(32 * scale)
        self._draw_sparkline(
            frame, text_x, chart_top, chart_width, sparkline_height, density_history, peak_vehicle_count, scale
        )

    def _draw_sparkline(
        self,
        frame: np.ndarray,
        x0: int,
        y0: int,
        width: int,
        height: int,
        history: list[int],
        peak: int,
        scale: float,
    ) -> None:
        """Small trend line of recent vehicle density -- normalized to the whole-video peak
        (not just this window's max) so the line's height stays meaningful relative to how
        busy the video gets overall, instead of rescaling every frame."""
        if len(history) < 2 or width <= 0 or height <= 0:
            return
        max_value = max(peak, max(history), 1)
        n = len(history)
        points = [
            (
                x0 + round(i / (n - 1) * width),
                y0 + height - round((value / max_value) * height),
            )
            for i, value in enumerate(history)
        ]
        cv2.polylines(
            frame,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=_PANEL_TEXT,
            thickness=max(1, round(1.6 * scale)),
            lineType=cv2.LINE_AA,
        )

    def render_summary_card(
        self,
        frame: np.ndarray,
        metrics: TrafficMetrics,
        peak_vehicle_count: int,
    ) -> np.ndarray:
        """Renders a still, centered end-of-video stats card over a dimmed freeze-frame,
        meant to be written for the final few seconds of the annotated video as a clean,
        shareable closing shot -- distinct from the live corner panel so it reads as a
        deliberate outro, not a continuation of live tracking."""
        scale = min(_MAX_SCALE, max(_MIN_SCALE, frame.shape[1] / _REFERENCE_WIDTH))
        dimmed = frame.copy()
        black = np.zeros_like(frame)
        cv2.addWeighted(black, 0.55, dimmed, 0.45, 0, dimmed)

        present_classes = [c for c in _CLASS_DISPLAY_ORDER if metrics.vehicles_per_class.get(c)]
        present_classes += [
            c for c in metrics.vehicles_per_class if c not in _CLASS_DISPLAY_ORDER and metrics.vehicles_per_class[c]
        ]
        breakdown_rows = [
            (_CLASS_DISPLAY_LABEL.get(c, c.title()), metrics.vehicles_per_class[c]) for c in present_classes
        ]

        level_color = _CONGESTION_COLORS[metrics.traffic_level]
        line_gap = round(48 * scale)
        row_count = 5 + len(breakdown_rows)  # Vehicles, breakdown rows, People, Peak, Traffic, Duration
        width = round(560 * scale)
        height = round(90 * scale) + line_gap * row_count
        x0 = (frame.shape[1] - width) // 2
        y0 = (frame.shape[0] - height) // 2
        x1, y1 = x0 + width, y0 + height

        panel = dimmed.copy()
        cv2.rectangle(panel, (x0, y0), (x1, y1), _PANEL_BG, thickness=-1)
        cv2.addWeighted(panel, 0.92, dimmed, 0.08, 0, dimmed)
        cv2.rectangle(dimmed, (x0, y0), (x1, y1), (90, 90, 90), thickness=max(1, round(scale)))

        accent_width = max(6, round(8 * scale))
        cv2.rectangle(dimmed, (x0, y0), (x0 + accent_width, y1), level_color, thickness=-1)

        text_x = x0 + accent_width + round(20 * scale)
        text_thickness = 2 if scale >= 1.6 else 1
        line_scale = 0.85 * scale
        row_y = y0 + round(52 * scale)

        _put_label_outlined(dimmed, "Session Summary", (text_x, row_y), _PANEL_TEXT, 1.1 * scale, 2)

        row_y += line_gap
        total_vehicles = sum(metrics.vehicles_per_class.values())
        _put_label_outlined(
            dimmed, f"Vehicles  {total_vehicles}", (text_x, row_y), _PANEL_TEXT, line_scale * 1.1, text_thickness
        )
        for label, count in breakdown_rows:
            row_y += line_gap
            _put_label_outlined(
                dimmed, f"  {label}  {count}", (text_x, row_y), _PANEL_MUTED_TEXT, line_scale * 0.9, text_thickness
            )

        row_y += line_gap
        _put_label_outlined(
            dimmed,
            f"People    {metrics.total_pedestrians}",
            (text_x, row_y),
            _PANEL_TEXT,
            line_scale * 1.1,
            text_thickness,
        )

        row_y += line_gap
        _put_label_outlined(
            dimmed,
            f"Peak at once  {peak_vehicle_count}",
            (text_x, row_y),
            _PANEL_MUTED_TEXT,
            line_scale,
            text_thickness,
        )

        row_y += line_gap
        _put_label_outlined(
            dimmed,
            f"Traffic level  {metrics.traffic_level.value}",
            (text_x, row_y),
            level_color,
            line_scale * 1.15,
            text_thickness,
        )

        row_y += line_gap
        _put_label_outlined(
            dimmed,
            f"Duration  {metrics.video_duration_s:.0f}s",
            (text_x, row_y),
            _PANEL_MUTED_TEXT,
            line_scale,
            text_thickness,
        )

        return dimmed
