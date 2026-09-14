from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


class VideoSourceError(Exception):
    """Raised for missing, unreadable, or malformed input video files."""


class VideoSource:
    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise VideoSourceError(f"Video file not found: {self._path}")

        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise VideoSourceError(f"Could not open video file (unsupported or corrupt): {self._path}")

        # Phone-recorded video (e.g. a portrait clip) often carries a 90/180/270 rotation flag
        # in its container metadata rather than storing pixels upright. Without this, frames
        # come out sideways -- both what the model sees and what gets written to the annotated
        # output -- even though every normal video player honors the flag and displays it
        # upright. This asks OpenCV to apply that rotation while decoding.
        self._capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        detected_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps_override if fps_override else detected_fps
        if not self.fps or self.fps <= 0:
            self._capture.release()
            raise VideoSourceError(
                f"Video '{self._path}' reports an invalid FPS ({detected_fps}). "
                "Set video.fps_override in the config to work around this."
            )

        self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frame_width <= 0 or self.frame_height <= 0:
            self._capture.release()
            raise VideoSourceError(
                f"Video '{self._path}' reports an invalid resolution "
                f"({self.frame_width}x{self.frame_height})"
            )

        # Container-reported estimate, not authoritative: some codecs/containers report 0 or a
        # count that doesn't match what decoding actually yields. Callers use it as a progress-bar
        # hint, not a value to rely on for correctness.
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        frame_index = 0
        while True:
            has_frame, frame = self._capture.read()
            if not has_frame:
                break
            yield frame_index, frame_index / self.fps, frame
            frame_index += 1

        if frame_index == 0:
            raise VideoSourceError(f"Video '{self._path}' contains no readable frames")

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
