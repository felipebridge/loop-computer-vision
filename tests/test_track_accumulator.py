from __future__ import annotations

from traffic_intelligence.analytics.speed import SpeedEstimator
from traffic_intelligence.tracking import track_accumulator
from traffic_intelligence.tracking.track_accumulator import TrackAccumulator


def test_rejects_tracks_shorter_than_minimum_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3)
    for frame_index in range(3):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    assert accumulator.finalize() == []


def test_keeps_tracks_meeting_minimum_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3)
    for frame_index in range(5):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    summaries = accumulator.finalize()
    assert len(summaries) == 1
    assert summaries[0].track_id == 1
    assert summaries[0].frame_count == 5


def test_dominant_class_wins_over_flicker(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, class_name="car", class_id=2))
    accumulator.add(detection_factory(track_id=1, frame_index=1, class_name="car", class_id=2))
    accumulator.add(detection_factory(track_id=1, frame_index=2, class_name="truck", class_id=7))

    summaries = accumulator.finalize()
    assert summaries[0].class_name == "car"


def test_dominant_class_resists_bicycle_motorcycle_flicker(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, class_name="bicycle", class_id=1))
    accumulator.add(
        detection_factory(track_id=1, frame_index=1, class_name="motorcycle", class_id=3)
    )
    accumulator.add(detection_factory(track_id=1, frame_index=2, class_name="bicycle", class_id=1))

    assert accumulator.dominant_class(1) == (1, "bicycle")


def test_dominant_class_holds_displayed_class_until_challenger_leads_by_margin(detection_factory):
    """A challenger that merely edges ahead in raw vote count (a near-tie) shouldn't flip the
    displayed class frame-to-frame; it should only switch once it leads by enough votes."""
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    frame_index = 0

    for _ in range(6):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, class_name="truck", class_id=7))
        frame_index += 1
    assert accumulator.dominant_class(1) == (7, "truck")

    for _ in range(6):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, class_name="car", class_id=2))
        frame_index += 1
    # truck=6, car=6: still displaying truck (a tie doesn't dislodge the displayed class).
    assert accumulator.dominant_class(1) == (7, "truck")

    accumulator.add(detection_factory(track_id=1, frame_index=frame_index, class_name="car", class_id=2))
    frame_index += 1
    # truck=6, car=7: car is the raw majority by 1 vote, but that's not enough of a lead yet --
    # stays on the displayed class instead of flickering to car and possibly back.
    assert accumulator.dominant_class(1) == (7, "truck")

    accumulator.add(detection_factory(track_id=1, frame_index=frame_index, class_name="car", class_id=2))
    frame_index += 1
    # truck=6, car=8: now car's lead clears the required margin, so it switches once, on purpose.
    assert accumulator.dominant_class(1) == (2, "car")


def test_mean_confidence_is_averaged(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, confidence=0.6))
    accumulator.add(detection_factory(track_id=1, frame_index=1, confidence=0.8))

    summaries = accumulator.finalize()
    assert summaries[0].mean_confidence == 0.7


def test_trail_returns_recent_centroids(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, bbox=(0, 0, 10, 10)))
    accumulator.add(detection_factory(track_id=1, frame_index=1, bbox=(10, 10, 20, 20)))

    trail = accumulator.trail(1)
    assert trail == [(5.0, 5.0), (15.0, 15.0)]


def test_trail_empty_for_unknown_track():
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    assert accumulator.trail(999) == []


def test_single_frame_track_is_always_rejected(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, class_name="car"))
    accumulator.add(detection_factory(track_id=1, frame_index=1, class_name="car"))
    accumulator.add(detection_factory(track_id=2, frame_index=0, class_name="person", class_id=0))

    summaries = {s.track_id: s for s in accumulator.finalize()}
    assert set(summaries) == {1}


def test_rejects_flickering_track_despite_meeting_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3, min_visibility_ratio=0.6)
    for frame_index in (0, 5, 9):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    assert accumulator.finalize() == []


def test_keeps_track_meeting_visibility_ratio_despite_occlusion_gaps(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3, min_visibility_ratio=0.6)
    for frame_index in (0, 1, 2, 3, 5, 6, 7, 9):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    summaries = accumulator.finalize()
    assert len(summaries) == 1
    assert summaries[0].frame_count == 8


def test_is_confirmed_false_for_unknown_track():
    accumulator = TrackAccumulator(min_track_seconds=0.3)
    assert accumulator.is_confirmed(999) is False


def test_is_confirmed_false_before_minimum_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3)
    accumulator.add(detection_factory(track_id=1, frame_index=0, fps=10.0))
    assert accumulator.is_confirmed(1) is False


def test_is_confirmed_true_once_duration_and_visibility_are_met(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3, min_visibility_ratio=0.6)
    for frame_index in range(5):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))
    assert accumulator.is_confirmed(1) is True


def test_is_confirmed_matches_finalize_for_flickering_track(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3, min_visibility_ratio=0.6)
    for frame_index in (0, 5, 9):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    assert accumulator.is_confirmed(1) is False
    assert accumulator.finalize() == []


def test_finalize_speed_fields_default_without_estimator(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0))
    accumulator.add(detection_factory(track_id=1, frame_index=1))

    summaries = accumulator.finalize()
    assert summaries[0].avg_speed_kmh is None
    assert summaries[0].max_speed_kmh is None
    assert summaries[0].speed_estimated is False


def test_current_speed_kmh_none_without_estimator(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0))

    assert accumulator.current_speed_kmh(1) is None


def test_finalize_computes_speed_once_estimator_is_calibrated(detection_factory):
    estimator = SpeedEstimator(min_calibration_samples=2)
    accumulator = TrackAccumulator(min_track_seconds=0.01, speed_estimator=estimator)
    for frame_index in range(6):
        offset = frame_index * 20
        accumulator.add(
            detection_factory(
                track_id=1,
                frame_index=frame_index,
                class_name="car",
                bbox=(offset, 0, offset + 10, 10),
                fps=10.0,
            )
        )

    summaries = accumulator.finalize()
    assert summaries[0].speed_estimated is True
    assert summaries[0].avg_speed_kmh is not None
    assert summaries[0].avg_speed_kmh > 0
    assert summaries[0].max_speed_kmh is not None


def test_current_speed_kmh_none_before_min_segment_duration(detection_factory):
    estimator = SpeedEstimator(min_calibration_samples=1)
    accumulator = TrackAccumulator(min_track_seconds=0.01, speed_estimator=estimator)
    accumulator.add(
        detection_factory(track_id=1, frame_index=0, class_name="car", bbox=(0, 0, 10, 10), fps=100.0)
    )
    accumulator.add(
        detection_factory(track_id=1, frame_index=1, class_name="car", bbox=(1, 0, 11, 10), fps=100.0)
    )

    assert accumulator.current_speed_kmh(1) is None


def test_stationary_vehicle_is_not_confirmed(detection_factory):
    accumulator = TrackAccumulator(
        min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.05
    )
    for frame_index in range(5):
        accumulator.add(
            detection_factory(track_id=1, frame_index=frame_index, class_name="car", bbox=(0, 0, 10, 10))
        )

    assert accumulator.is_confirmed(1) is False
    assert accumulator.finalize() == []


def test_moving_vehicle_is_confirmed_despite_movement_filter(detection_factory):
    accumulator = TrackAccumulator(
        min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.05
    )
    for frame_index in range(5):
        offset = frame_index * 20
        accumulator.add(
            detection_factory(
                track_id=1,
                frame_index=frame_index,
                class_name="car",
                bbox=(offset, 0, offset + 10, 10),
            )
        )

    assert accumulator.is_confirmed(1) is True


def test_long_lived_track_needs_more_movement_to_stay_confirmed(detection_factory):
    # Camera-motion compensation drift compounds over a long shot -- a track that's been in
    # frame for 10s+ with only modest displacement (i.e. it would pass the base filter) reads
    # as parked/stationary once that residual drift is accounted for, not as real movement.
    accumulator = TrackAccumulator(
        min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.1
    )
    for frame_index in range(17):  # timestamps 0..16s at fps=1.0
        offset = frame_index * 10  # displacement reaches 160px, above the 100px base threshold
        accumulator.add(
            detection_factory(
                track_id=1,
                frame_index=frame_index,
                class_name="car",
                bbox=(offset, 0, offset + 10, 10),
                fps=1.0,
            )
        )

    assert accumulator.is_confirmed(1) is False


def test_long_lived_track_confirmed_with_clearly_real_movement(detection_factory):
    accumulator = TrackAccumulator(
        min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.1
    )
    for frame_index in range(17):  # timestamps 0..16s at fps=1.0
        offset = frame_index * 30  # displacement reaches 480px, above the 400px strict threshold
        accumulator.add(
            detection_factory(
                track_id=1,
                frame_index=frame_index,
                class_name="car",
                bbox=(offset, 0, offset + 10, 10),
                fps=1.0,
            )
        )

    assert accumulator.is_confirmed(1) is True


def test_movement_filter_does_not_apply_to_pedestrians(detection_factory):
    accumulator = TrackAccumulator(
        min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.05
    )
    for frame_index in range(5):
        accumulator.add(
            detection_factory(
                track_id=1, frame_index=frame_index, class_name="person", class_id=0, bbox=(0, 0, 10, 10)
            )
        )

    assert accumulator.is_confirmed(1) is True


def test_movement_filter_disabled_when_ratio_is_zero(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.0)
    for frame_index in range(5):
        accumulator.add(
            detection_factory(track_id=1, frame_index=frame_index, class_name="car", bbox=(0, 0, 10, 10))
        )

    assert accumulator.is_confirmed(1) is True


def test_percentile_helper_matches_known_values():
    assert track_accumulator._percentile([10.0], 0.9) == 10.0
    assert track_accumulator._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_percentile_helper_is_robust_to_a_single_outlier():
    # A one-off spike (e.g. one noisy segment among many normal ones) shouldn't dominate the
    # reported "top speed" -- see TrackAccumulator._speed_summary.
    values = [5.0] * 49 + [500.0]
    assert track_accumulator._percentile(values, 0.9) < 50.0


def test_add_uses_compensated_position_for_movement_and_speed(detection_factory):
    # A raw centroid that drifts (as it would if the camera itself pans) should not count as
    # real movement once a compensated (world-frame) position is supplied that stays put.
    accumulator = TrackAccumulator(
        min_track_seconds=0.01, frame_diagonal=1000.0, min_movement_ratio=0.05
    )
    for frame_index in range(5):
        offset = frame_index * 20
        accumulator.add(
            detection_factory(
                track_id=1,
                frame_index=frame_index,
                class_name="car",
                bbox=(offset, 0, offset + 10, 10),
            ),
            position=(5.0, 5.0),
        )

    assert accumulator.is_confirmed(1) is False
