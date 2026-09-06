from __future__ import annotations

import cv2
import numpy as np
import pytest

from traffic_intelligence.analytics.motion_compensation import CameraMotionEstimator


def _textured_frame(size: int = 240, seed: int = 7) -> np.ndarray:
    # A repeating pattern (e.g. a checkerboard) is ambiguous for optical flow -- neighboring
    # cells look identical, so a tracked point can lock onto the wrong cell. Blurred random
    # noise gives every patch a locally unique appearance, which is what real backgrounds
    # (roads, buildings, foliage) look like to a feature tracker.
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def _shifted(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(frame)
    height, width = frame.shape[:2]
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_y0 = max(0, dx), max(0, dy)
    shifted[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = frame[
        src_y0:src_y1, src_x0:src_x1
    ]
    return shifted


def test_identity_before_second_frame():
    estimator = CameraMotionEstimator()
    estimator.update(_textured_frame())

    assert estimator.to_reference_frame((37.0, 51.0)) == pytest.approx((37.0, 51.0))


def test_compensates_pure_camera_translation():
    base = _textured_frame()
    estimator = CameraMotionEstimator()
    estimator.update(base)
    estimator.update(_shifted(base, dx=15, dy=-8))

    # A point that's stationary in the *world* lands at (x+15, y-8) in the panned frame's raw
    # pixels; mapping it back through the estimated camera motion should recover ~its original
    # (frame-0) position.
    world_point = (120.0, 120.0)
    panned_point = (world_point[0] + 15.0, world_point[1] - 8.0)

    recovered = estimator.to_reference_frame(panned_point)
    assert recovered == pytest.approx(world_point, abs=2.0)


def test_exclude_boxes_prevents_foreground_contamination():
    # A large "vehicle" (a differently-textured patch) covers most of the frame and moves
    # every frame, while the small strip left uncovered is the only genuine background and
    # stays put. Without masking, goodFeaturesToTrack finds most of its corners on the moving
    # patch and RANSAC's "background" fit gets dragged along with it; masking that region out
    # should recover ~zero camera motion instead.
    base = _textured_frame(size=240)
    foreground = _textured_frame(size=240, seed=99)
    box = (0, 0, 240, 200)  # covers all but the bottom strip (the real background)

    def _composited(dx: int) -> np.ndarray:
        frame = base.copy()
        shifted_fg = _shifted(foreground, dx=dx, dy=0)
        frame[0:200, :] = shifted_fg[0:200, :]
        return frame

    estimator = CameraMotionEstimator()
    estimator.update(_composited(dx=0), exclude_boxes=[box])
    estimator.update(_composited(dx=40), exclude_boxes=[box])

    stationary_point = (120.0, 220.0)  # in the untouched background strip
    recovered = estimator.to_reference_frame(stationary_point)
    assert recovered == pytest.approx(stationary_point, abs=3.0)


def test_no_crash_on_blank_frames():
    estimator = CameraMotionEstimator()
    blank = np.zeros((240, 240, 3), dtype=np.uint8)
    estimator.update(blank)
    estimator.update(blank)

    # No trackable features -- falls back to identity rather than raising.
    assert estimator.to_reference_frame((10.0, 10.0)) == pytest.approx((10.0, 10.0))
