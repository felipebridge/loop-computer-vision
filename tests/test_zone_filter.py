from __future__ import annotations

from traffic_intelligence.analytics.zone_filter import exclude_zone_detections
from traffic_intelligence.schemas.detection import TrackedDetection


def _detection(track_id: int, bbox: tuple[float, float, float, float], class_name: str = "person") -> TrackedDetection:
    return TrackedDetection(
        frame_index=0,
        timestamp=0.0,
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
    )


def test_drops_detection_centered_inside_an_excluded_zone():
    statue = _detection(1, (400, 300, 500, 500))  # centroid at (450, 400) -> (0.45, 0.4) fraction
    zones = [(0.4, 0.35, 0.5, 0.45)]

    result = exclude_zone_detections([statue], zones, frame_width=1000, frame_height=1000)

    assert result == []


def test_keeps_detection_outside_any_excluded_zone():
    pedestrian = _detection(1, (0, 0, 20, 20))
    zones = [(0.4, 0.35, 0.5, 0.45)]

    result = exclude_zone_detections([pedestrian], zones, frame_width=1000, frame_height=1000)

    assert result == [pedestrian]


def test_applies_regardless_of_class():
    car = _detection(1, (400, 300, 500, 500), class_name="car")
    zones = [(0.4, 0.35, 0.5, 0.45)]

    result = exclude_zone_detections([car], zones, frame_width=1000, frame_height=1000)

    assert result == []


def test_no_zones_configured_keeps_everything():
    detections = [_detection(1, (400, 300, 500, 500))]

    result = exclude_zone_detections(detections, [], frame_width=1000, frame_height=1000)

    assert result == detections


def test_empty_detections_returns_empty():
    assert exclude_zone_detections([], [(0.0, 0.0, 1.0, 1.0)], frame_width=1000, frame_height=1000) == []
