from __future__ import annotations

import cv2
import numpy as np

_MAX_CORNERS = 600
_QUALITY_LEVEL = 0.008
_MIN_DISTANCE = 8
_MIN_TRACKED_POINTS = 6
_LK_WIN_SIZE = (21, 21)
_LK_MAX_LEVEL = 3


class CameraMotionEstimator:
    """Estimates the camera's own frame-to-frame motion (pan/rotation/handshake) from sparse
    optical flow on background features, and exposes a running transform that maps a raw
    pixel coordinate in the current frame back into frame 0's coordinate system.

    Handheld video is never perfectly static: everything in frame -- including a parked car --
    drifts on screen as the camera moves. Without compensating for that, a stationary vehicle
    looks like it's moving and every vehicle's estimated speed is off by however much the
    camera itself moved that frame. This estimates that camera motion from the *background*
    (via RANSAC, plus an explicit mask over this frame's detected boxes -- see `update` -- so a
    scene where traffic covers most of the frame doesn't overwhelm the "background" fit with
    real vehicle motion) and lets callers undo it before computing displacement or speed.

    This is *relative* visual odometry, not absolute positioning: each frame's estimate carries
    a little error, and composing hundreds of frames of that error accumulates into a real,
    unbounded drift over a long shot (see docs/architecture.md).
    """

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._cumulative_transform = np.eye(2, 3, dtype=np.float64)  # current frame -> frame 0

    def update(
        self, frame: np.ndarray, exclude_boxes: list[tuple[float, float, float, float]] | None = None
    ) -> None:
        """`exclude_boxes` (typically this frame's detected vehicle/person boxes) are masked
        out before looking for corners to track. Without this, on a frame that's mostly
        vehicles -- a packed multi-lane avenue, say -- most of what `goodFeaturesToTrack` finds
        sits on vehicle bodies rather than genuine background, and RANSAC's "background" fit
        gets contaminated by real vehicle motion (see docs/architecture.md); masking those
        regions out first keeps the search confined to what's actually static (sky, buildings,
        road markings) regardless of how much of the frame the traffic itself covers.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._background_mask(gray.shape, exclude_boxes) if exclude_boxes else None

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_mask = mask
            return

        prev_points = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=_MAX_CORNERS,
            qualityLevel=_QUALITY_LEVEL,
            minDistance=_MIN_DISTANCE,
            mask=self._prev_mask,
        )
        if prev_points is None or len(prev_points) < _MIN_TRACKED_POINTS:
            self._prev_gray = gray
            self._prev_mask = mask
            return

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_points, None, winSize=_LK_WIN_SIZE, maxLevel=_LK_MAX_LEVEL
        )
        valid = status.reshape(-1) == 1
        prev_valid = prev_points[valid]
        curr_valid = curr_points[valid]

        if len(prev_valid) >= _MIN_TRACKED_POINTS:
            # Maps this frame's coordinates onto the previous frame's -- i.e. the camera's own
            # motion since the last frame. RANSAC keeps a minority of points sitting on moving
            # vehicles from skewing the background-motion estimate.
            frame_to_previous, _ = cv2.estimateAffinePartial2D(curr_valid, prev_valid, method=cv2.RANSAC)
            if frame_to_previous is not None:
                self._cumulative_transform = self._compose(self._cumulative_transform, frame_to_previous)

        self._prev_gray = gray
        self._prev_mask = mask

    @staticmethod
    def _background_mask(
        shape: tuple[int, int], exclude_boxes: list[tuple[float, float, float, float]]
    ) -> np.ndarray:
        mask = np.full(shape, 255, dtype=np.uint8)
        height, width = shape
        for x1, y1, x2, y2 in exclude_boxes:
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(width, int(x2) + 1), min(height, int(y2) + 1)
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 0
        return mask

    @staticmethod
    def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        outer_3x3 = np.vstack([outer, [0.0, 0.0, 1.0]])
        inner_3x3 = np.vstack([inner, [0.0, 0.0, 1.0]])
        return (outer_3x3 @ inner_3x3)[:2, :]

    def to_reference_frame(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        transformed = self._cumulative_transform @ np.array([x, y, 1.0])
        return float(transformed[0]), float(transformed[1])
