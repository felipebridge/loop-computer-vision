from __future__ import annotations

from traffic_intelligence.schemas.detection import TrackedDetection


def exclude_zone_detections(
    detections: list[TrackedDetection],
    zones: list[tuple[float, float, float, float]],
    frame_width: float,
    frame_height: float,
) -> list[TrackedDetection]:
    """Drops any detection whose centroid falls inside a configured exclusion zone, regardless
    of class -- for a fixed static-camera shot with a known false-positive object (a statue, a
    mannequin, a poster) that a general-purpose detector can't tell apart from the real thing by
    appearance, motion, or confidence alone. See docs/architecture.md and configs/default.yaml's
    min_movement_ratio note: this position-based signal is the deliberate replacement for the
    motion-based approaches (per-track displacement, background-motion consistency) that were
    tried for this kind of static false positive and reverted for excluding real, legitimately
    still traffic/pedestrians along with it.

    Zones are fractions (0-1) of frame width/height, not raw pixels, so the same config works
    across differently-decoded resolutions of the same camera framing.
    """
    if not zones or frame_width <= 0 or frame_height <= 0:
        return detections

    def _in_excluded_zone(detection: TrackedDetection) -> bool:
        cx, cy = detection.centroid
        fx, fy = cx / frame_width, cy / frame_height
        return any(zx1 <= fx <= zx2 and zy1 <= fy <= zy2 for zx1, zy1, zx2, zy2 in zones)

    return [d for d in detections if not _in_excluded_zone(d)]
