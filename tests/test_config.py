from __future__ import annotations

import pytest
from pydantic import ValidationError

from traffic_intelligence.config.settings import (
    ConfigError,
    CongestionConfig,
    CongestionThresholds,
    DetectionConfig,
    PipelineConfig,
    TrackingConfig,
    load_config,
)


def test_load_default_config_succeeds():
    config = load_config("configs/default.yaml")
    assert config.detection.vehicle_classes
    assert config.tracking.tracker.value in {"bytetrack", "botsort"}


def test_load_default_config_includes_speed_section():
    config = load_config("configs/default.yaml")
    assert config.speed.enabled is True
    assert "car" in config.speed.reference_widths_m


def test_tracking_config_defaults_min_movement_ratio():
    config = TrackingConfig(tracker="bytetrack")
    assert config.min_movement_ratio == 0.08


def test_detection_config_defaults_imgsz():
    config = DetectionConfig()
    assert config.imgsz == 640


def test_load_default_config_uses_higher_imgsz():
    # Higher than the class default (640) so small/distant vehicles in a busy scene are still
    # detectable, while staying a middle ground usable on CPU-only hardware (see
    # configs/default.yaml) rather than the higher end this pipeline supports.
    config = load_config("configs/default.yaml")
    assert config.detection.imgsz == 960


def test_load_config_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("configs/does_not_exist.yaml")


def test_load_config_invalid_yaml_raises(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("congestion: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_file)


def test_detection_classes_includes_person():
    config = DetectionConfig(vehicle_classes=["car", "bus"], person_class="person")
    assert config.classes == ["car", "bus", "person"]


def test_detection_requires_at_least_one_vehicle_class():
    with pytest.raises(ValidationError):
        DetectionConfig(vehicle_classes=[])


def test_congestion_thresholds_require_high_above_moderate():
    with pytest.raises(ValidationError):
        CongestionThresholds(moderate=10, high=5)


def test_pipeline_config_requires_congestion_section():
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({})


def test_congestion_config_defaults_persistence():
    thresholds = CongestionThresholds(moderate=5, high=10)
    config = CongestionConfig(density_thresholds=thresholds)
    assert config.persistence_frames == 15


def test_appearance_reid_requires_botsort_tracker():
    with pytest.raises(ValidationError, match="appearance_reid_enabled"):
        TrackingConfig(tracker="bytetrack", appearance_reid_enabled=True)


def test_appearance_reid_allowed_with_botsort_tracker():
    config = TrackingConfig(tracker="botsort", appearance_reid_enabled=True)
    assert config.appearance_reid_enabled is True
