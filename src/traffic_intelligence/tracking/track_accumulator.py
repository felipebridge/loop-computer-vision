from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

from traffic_intelligence.analytics.speed import SpeedEstimator
from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.schemas.track import TrackSummary

_TRAIL_LENGTH = 20
# A segment this short (a handful of frames) is dominated by single-frame detection/compensation
# noise rather than real motion -- a wider window lets that noise average out.
_MIN_SPEED_SEGMENT_S = 0.4
_INSTANT_SPEED_WINDOW_S = 0.7
_TOP_SPEED_PERCENTILE = 0.9
# Camera-motion compensation (see analytics.motion_compensation) is relative, frame-to-frame
# visual odometry -- its estimation error compounds over a long shot, and after just several
# seconds of continuous footage that residual drift can rival a parked vehicle's real (zero)
# displacement. A vehicle that's still in frame this long almost certainly isn't passing
# through, so it's held to a much stricter bar rather than the base one, which is tuned for the
# short few-second windows most real vehicles are visible for.
_LONG_TRACK_SECONDS = 10.0
_LONG_TRACK_MOVEMENT_MULTIPLIER = 4.0
# A single global pixels-per-meter scale (see analytics.speed.SpeedEstimator) is calibrated
# from vehicles across the whole frame, near and far alike -- applying it to a bounding box
# this small (a vehicle near the vanishing point, on a wide avenue) amplifies perspective
# error enough that the reported speed is closer to noise than an estimate. Below this width
# the vehicle keeps its box, ID, and track, it just stops getting a speed label.
_MIN_BBOX_WIDTH_RATIO_FOR_SPEED = 0.025
# Segment endpoints are individual raw samples, so one noisy frame corrupts every segment that
# touches it. Averaging each point with its neighbors first removes that without erasing real
# motion, which happens over many frames, not one.
_SMOOTHING_WINDOW = 5
# A pure "current majority of votes so far" flickers whenever two classes are close (e.g. a
# vehicle shape the model is genuinely unsure about between "car" and "truck"): one new vote for
# the trailing class can flip which one is "most common" for a frame or two before the leading
# class pulls back ahead. Requiring the challenger to lead by at least this fraction of votes
# seen so far before the displayed class switches turns that into a single, deliberate change
# once a track's classification genuinely shifts, instead of flip-flopping on a near-tie.
_CLASS_SWITCH_LEAD_RATIO = 0.15


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _smoothed(
    history: list[tuple[float, float, float]], window: int = _SMOOTHING_WINDOW
) -> list[tuple[float, float, float]]:
    if len(history) < window:
        return history
    half = window // 2
    smoothed: list[tuple[float, float, float]] = []
    for i in range(len(history)):
        chunk = history[max(0, i - half) : min(len(history), i + half + 1)]
        avg_x = sum(p[1] for p in chunk) / len(chunk)
        avg_y = sum(p[2] for p in chunk) / len(chunk)
        smoothed.append((history[i][0], avg_x, avg_y))
    return smoothed


class TrackAccumulator:
    """Accumulates per-track duration, visibility, dominant class, a recent-centroid trail,
    and (when a SpeedEstimator is supplied) speed."""

    def __init__(
        self,
        min_track_seconds: float,
        min_visibility_ratio: float = 0.6,
        speed_estimator: SpeedEstimator | None = None,
        frame_diagonal: float = 0.0,
        min_movement_ratio: float = 0.0,
        person_class: str = "person",
        frame_width: float = 0.0,
    ) -> None:
        self._min_track_seconds = min_track_seconds
        self._min_visibility_ratio = min_visibility_ratio
        self._speed_estimator = speed_estimator
        self._frame_diagonal = frame_diagonal
        self._min_movement_ratio = min_movement_ratio
        self._person_class = person_class
        self._frame_width = frame_width
        self._last_bbox_width: dict[int, float] = {}
        self._frame_counts: dict[int, int] = defaultdict(int)
        self._class_votes: dict[int, Counter[int]] = defaultdict(Counter)
        self._class_names: dict[int, dict[int, str]] = defaultdict(dict)
        self._displayed_class: dict[int, int] = {}
        self._confidences: dict[int, list[float]] = defaultdict(list)
        self._first_seen: dict[int, tuple[int, float]] = {}
        self._last_seen: dict[int, tuple[int, float]] = {}
        self._first_centroid: dict[int, tuple[float, float]] = {}
        self._last_centroid: dict[int, tuple[float, float]] = {}
        self._trails: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=_TRAIL_LENGTH)
        )
        # Full (unbounded) per-track position history, timestamped, kept separately from the
        # display trail above: speed needs the whole track's motion, not just the last
        # _TRAIL_LENGTH points drawn on screen. Positions here are in the *camera-motion-
        # compensated* reference frame when the caller supplies one (see
        # analytics.motion_compensation.CameraMotionEstimator) -- a raw pixel trail would make
        # a parked car look like it's moving whenever the camera itself pans or shakes.
        self._position_history: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        self._max_displacement: dict[int, float] = defaultdict(float)

    def add(self, detection: TrackedDetection, position: tuple[float, float] | None = None) -> None:
        track_id = detection.track_id
        resolved_position = position if position is not None else detection.centroid

        self._frame_counts[track_id] += 1
        self._class_votes[track_id][detection.class_id] += 1
        self._class_names[track_id][detection.class_id] = detection.class_name
        self._confidences[track_id].append(detection.confidence)
        self._trails[track_id].append(detection.centroid)
        x1, _, x2, _ = detection.bbox
        self._last_bbox_width[track_id] = x2 - x1
        self._position_history[track_id].append(
            (detection.timestamp, resolved_position[0], resolved_position[1])
        )

        first_x, first_y = self._position_history[track_id][0][1:]
        displacement = math.hypot(resolved_position[0] - first_x, resolved_position[1] - first_y)
        if displacement > self._max_displacement[track_id]:
            self._max_displacement[track_id] = displacement

        if self._speed_estimator is not None:
            x1, _, x2, _ = detection.bbox
            self._speed_estimator.observe(detection.class_name, x2 - x1)

        if track_id not in self._first_seen:
            self._first_seen[track_id] = (detection.frame_index, detection.timestamp)
            self._first_centroid[track_id] = detection.centroid
        self._last_seen[track_id] = (detection.frame_index, detection.timestamp)
        self._last_centroid[track_id] = detection.centroid

    def trail(self, track_id: int) -> list[tuple[float, float]]:
        return list(self._trails.get(track_id, ()))

    def dominant_class(self, track_id: int) -> tuple[int, str]:
        """Majority-voted class for a track using votes seen so far (not just at finalize),
        with hysteresis: the displayed class only switches once a challenger leads the
        currently-displayed class by at least _CLASS_SWITCH_LEAD_RATIO of votes seen so far,
        not on every new vote that nudges a near-tie the other way.

        Lets live rendering show the same flicker-resistant class (e.g. bicycle vs.
        motorcycle, or car vs. truck) that finalize() would report, instead of the raw
        per-frame prediction.
        """
        votes = self._class_votes[track_id]
        leading_class_id, leading_votes = votes.most_common(1)[0]

        displayed_class_id = self._displayed_class.get(track_id)
        if displayed_class_id is not None and displayed_class_id != leading_class_id:
            total_votes = sum(votes.values())
            required_lead = max(1, round(total_votes * _CLASS_SWITCH_LEAD_RATIO))
            displayed_votes = votes[displayed_class_id]
            if leading_votes - displayed_votes < required_lead:
                return displayed_class_id, self._class_names[track_id][displayed_class_id]

        self._displayed_class[track_id] = leading_class_id
        return leading_class_id, self._class_names[track_id][leading_class_id]

    def _too_far_for_speed(self, track_id: int) -> bool:
        if self._frame_width <= 0:
            return False
        return self._last_bbox_width.get(track_id, 0.0) < _MIN_BBOX_WIDTH_RATIO_FOR_SPEED * self._frame_width

    def current_speed_kmh(self, track_id: int) -> float | None:
        """Instantaneous speed for live on-screen display, from the most recent positions
        within a short rolling window. Only as accurate as calibration is so far into the
        video (see SpeedEstimator) -- it typically sharpens as more vehicles are seen."""
        if self._speed_estimator is None or self._too_far_for_speed(track_id):
            return None
        raw_history = self._position_history.get(track_id)
        if not raw_history or len(raw_history) < 2:
            return None
        history = _smoothed(raw_history)

        latest = history[-1]
        window_start = latest[0] - _INSTANT_SPEED_WINDOW_S
        anchor = history[-2]
        for point in reversed(history[:-1]):
            anchor = point
            if point[0] <= window_start:
                break

        duration = latest[0] - anchor[0]
        if duration < _MIN_SPEED_SEGMENT_S:
            return None
        distance_px = math.hypot(latest[1] - anchor[1], latest[2] - anchor[2])
        return self._speed_estimator.speed_kmh(distance_px, duration)

    def is_confirmed(self, track_id: int) -> bool:
        if track_id not in self._first_seen:
            return False
        first_frame, first_timestamp = self._first_seen[track_id]
        last_frame, last_timestamp = self._last_seen[track_id]
        if last_timestamp - first_timestamp < self._min_track_seconds:
            return False
        span_frames = last_frame - first_frame + 1
        if self._frame_counts[track_id] / span_frames < self._min_visibility_ratio:
            return False
        return self._has_moved_enough(track_id)

    def _has_moved_enough(self, track_id: int) -> bool:
        """Excludes parked/stationary vehicles: a track whose (motion-compensated) position
        never strays far from where it started is standing still, not part of moving traffic.
        Only applied to vehicles -- a pedestrian standing still is still a pedestrian."""
        if self._min_movement_ratio <= 0 or self._frame_diagonal <= 0:
            return True
        _, class_name = self.dominant_class(track_id)
        if class_name == self._person_class:
            return True

        required_ratio = self._min_movement_ratio
        _, first_timestamp = self._first_seen[track_id]
        _, last_timestamp = self._last_seen[track_id]
        if last_timestamp - first_timestamp >= _LONG_TRACK_SECONDS:
            required_ratio *= _LONG_TRACK_MOVEMENT_MULTIPLIER
        return self._max_displacement[track_id] >= required_ratio * self._frame_diagonal

    def _speed_summary(self, track_id: int) -> tuple[float | None, float | None]:
        """Average and top speed over the track's full path, from consecutive segments of
        the position history (not just first-to-last centroid, which understates speed on a
        curved or perspective-foreshortened path).

        "Top speed" is the 90th percentile of those segment speeds, not the literal max: a
        single noisy segment (a stray detection-box jitter or a frame where camera-motion
        compensation slipped a little) produces one wildly high segment speed among many
        normal ones, and reporting that raw spike as "max speed" reads as broken rather than
        as a real burst of speed. The percentile keeps a genuine sustained fast segment while
        discarding a one-frame outlier.
        """
        if self._speed_estimator is None or self._too_far_for_speed(track_id):
            return None, None

        history = _smoothed(self._position_history.get(track_id, []))
        speeds: list[float] = []
        if len(history) >= 2:
            start = history[0]
            for point in history[1:]:
                duration = point[0] - start[0]
                if duration < _MIN_SPEED_SEGMENT_S:
                    continue
                distance_px = math.hypot(point[1] - start[1], point[2] - start[2])
                speed = self._speed_estimator.speed_kmh(distance_px, duration)
                if speed is not None:
                    speeds.append(speed)
                start = point

        if not speeds:
            return None, None
        return sum(speeds) / len(speeds), _percentile(speeds, _TOP_SPEED_PERCENTILE)

    def finalize(self) -> list[TrackSummary]:
        summaries: list[TrackSummary] = []
        for track_id, frame_count in self._frame_counts.items():
            if not self.is_confirmed(track_id):
                continue

            first_frame, first_timestamp = self._first_seen[track_id]
            last_frame, last_timestamp = self._last_seen[track_id]
            dominant_class_id, dominant_class_name = self.dominant_class(track_id)
            confidences = self._confidences[track_id]
            avg_speed_kmh, max_speed_kmh = self._speed_summary(track_id)

            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    class_id=dominant_class_id,
                    class_name=dominant_class_name,
                    mean_confidence=sum(confidences) / len(confidences),
                    frame_count=frame_count,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp,
                    first_centroid=self._first_centroid[track_id],
                    last_centroid=self._last_centroid[track_id],
                    avg_speed_kmh=avg_speed_kmh,
                    max_speed_kmh=max_speed_kmh,
                    speed_estimated=avg_speed_kmh is not None,
                )
            )
        return summaries
