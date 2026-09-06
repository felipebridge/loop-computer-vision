from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from traffic_intelligence.analytics.congestion import CongestionClassifier
from traffic_intelligence.analytics.metrics import compute_traffic_metrics
from traffic_intelligence.analytics.motion_compensation import CameraMotionEstimator
from traffic_intelligence.analytics.occupant_filter import exclude_vehicle_occupants
from traffic_intelligence.analytics.speed import SpeedEstimator
from traffic_intelligence.config.settings import PipelineConfig
from traffic_intelligence.persistence.video_encoder import finalize_video
from traffic_intelligence.pipeline.hdr_preprocess import resolve_input_video
from traffic_intelligence.pipeline.video_source import VideoSource
from traffic_intelligence.schemas.metrics import CongestionState, TrafficMetrics
from traffic_intelligence.schemas.track import TrackSummary
from traffic_intelligence.tracking.track_accumulator import TrackAccumulator
from traffic_intelligence.tracking.track_stitcher import stitch_fragmented_tracks
from traffic_intelligence.tracking.ultralytics_tracker import UltralyticsTracker
from traffic_intelligence.utils.logging import get_logger
from traffic_intelligence.visualization.annotator import FrameAnnotator

logger = get_logger("pipeline.runner")

# Recent-trend window for the live density sparkline -- bounded like
# CongestionClassifier._history, so memory stays flat regardless of video length. 20s
# comfortably shows the whole trend on a short clip and a meaningful recent window on a long one.
_SPARKLINE_WINDOW_S = 20.0


@dataclass
class RunResult:
    track_summaries: list[TrackSummary]
    metrics: TrafficMetrics
    annotated_video_path: Path | None


class PipelineRunner:
    """Orchestrates the video-to-counts pipeline: tracking, per-frame vehicle
    and person counting, congestion classification, and annotated-video
    rendering."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._tracker = UltralyticsTracker(config.detection, config.tracking)
        self._congestion_classifier = CongestionClassifier(config.congestion)
        self._annotator = FrameAnnotator()
        self._traffic_level_history: list[CongestionState] = []
        self._peak_vehicle_count = 0

    def run(self, input_path: str | Path, output_dir: str | Path) -> RunResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        video_writer: cv2.VideoWriter | None = None
        raw_video_path: Path | None = None
        annotated_video_path: Path | None = None
        last_frame_index = -1
        last_timestamp = 0.0
        frame_diagonal = 0.0

        decode_path = resolve_input_video(
            input_path, output_dir / "videos", self._config.video.tone_map_hdr
        )

        with VideoSource(decode_path, self._config.video.fps_override) as source:
            fps = source.fps
            frame_diagonal = math.hypot(source.frame_width, source.frame_height)
            vehicle_count_window: deque[int] = deque(maxlen=max(1, round(fps * _SPARKLINE_WINDOW_S)))
            speed_estimator = (
                SpeedEstimator(self._config.speed.reference_widths_m, self._config.speed.min_calibration_samples)
                if self._config.speed.enabled
                else None
            )
            accumulator = TrackAccumulator(
                self._config.tracking.min_track_seconds,
                self._config.tracking.min_visibility_ratio,
                speed_estimator=speed_estimator,
                frame_diagonal=frame_diagonal,
                min_movement_ratio=self._config.tracking.min_movement_ratio,
                person_class=self._config.detection.person_class,
                frame_width=source.frame_width,
            )
            camera_motion = CameraMotionEstimator()

            if self._config.output.save_annotated_video:
                videos_dir = output_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)
                annotated_video_path = videos_dir / f"{input_path.stem}_annotated.mp4"
                raw_video_path = videos_dir / f"{input_path.stem}_raw.mp4"
                video_writer = cv2.VideoWriter(
                    str(raw_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source.fps,
                    (source.frame_width, source.frame_height),
                )

            for frame_index, timestamp, frame in source.frames():
                tracked = self._tracker.track(frame, frame_index, timestamp)
                tracked = exclude_vehicle_occupants(tracked, self._config.detection.person_class)
                camera_motion.update(frame, exclude_boxes=[d.bbox for d in tracked])
                for detection in tracked:
                    accumulator.add(detection, camera_motion.to_reference_frame(detection.centroid))

                person_class = self._config.detection.person_class
                confirmed = [d for d in tracked if accumulator.is_confirmed(d.track_id)]
                display_detections = []
                for d in confirmed:
                    class_id, class_name = accumulator.dominant_class(d.track_id)
                    display_detections.append(
                        d.model_copy(update={"class_id": class_id, "class_name": class_name})
                    )
                counts_by_class = Counter(d.class_name for d in display_detections)
                person_count = counts_by_class.pop(person_class, 0)
                vehicle_count = sum(counts_by_class.values())

                traffic_level = self._congestion_classifier.update(vehicle_count)
                self._traffic_level_history.append(traffic_level)
                vehicle_count_window.append(vehicle_count)
                self._peak_vehicle_count = max(self._peak_vehicle_count, vehicle_count)

                if video_writer is not None:
                    trails = {d.track_id: accumulator.trail(d.track_id) for d in confirmed}
                    annotated_frame = self._annotator.annotate(
                        frame,
                        display_detections,
                        trails,
                        counts_by_class,
                        person_count,
                        traffic_level,
                        list(vehicle_count_window),
                        self._peak_vehicle_count,
                    )
                    video_writer.write(annotated_frame)

                last_frame_index = frame_index
                last_timestamp = timestamp

            if video_writer is not None:
                video_writer.release()

            track_summaries = accumulator.finalize()

        if raw_video_path is not None and annotated_video_path is not None:
            finalize_video(raw_video_path, annotated_video_path)

        track_summaries = stitch_fragmented_tracks(
            track_summaries,
            frame_diagonal=frame_diagonal,
            max_gap_seconds=self._config.tracking.reid_max_gap_seconds,
            max_centroid_distance_ratio=self._config.tracking.reid_max_centroid_distance_ratio,
        )
        overall_level = self._overall_traffic_level()
        metrics = compute_traffic_metrics(
            track_summaries,
            overall_level,
            video_duration_s=last_timestamp,
            frames_processed=last_frame_index + 1,
        )

        logger.info(
            "Pipeline finished: %d vehicles, %d pedestrians, traffic level=%s",
            metrics.total_vehicles,
            metrics.total_pedestrians,
            overall_level.value,
        )

        return RunResult(
            track_summaries=track_summaries,
            metrics=metrics,
            annotated_video_path=annotated_video_path,
        )

    def _overall_traffic_level(self) -> CongestionState:
        if not self._traffic_level_history:
            return CongestionState.LOW
        most_common, _ = Counter(self._traffic_level_history).most_common(1)[0]
        return most_common
