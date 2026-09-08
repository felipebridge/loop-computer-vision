from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


class TrackerType(StrEnum):
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"


class DeviceType(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class VideoConfig(BaseModel):
    fps_override: float | None = Field(default=None, gt=0)
    # iPhone-style HDR/Dolby Vision clips (HLG or PQ base layer) decode with muted contrast
    # and clipped highlights if the YUV samples are treated as ordinary gamma-encoded video --
    # see pipeline.hdr_preprocess. Has no effect (and costs nothing extra) on an already-SDR
    # source: detection is only enabled for actual HDR transfer functions.
    tone_map_hdr: bool = True


class DetectionConfig(BaseModel):
    model_path: str = "yolo11s.pt"
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Ultralytics' default (640) downscales a 4K frame enough that small/distant vehicles
    # shrink below what the detector can pick up, undercounting how much traffic is visible
    # at once. Raising this improves recall at roughly the square of the increase in runtime.
    imgsz: int = Field(default=640, gt=0)
    vehicle_classes: list[str] = Field(
        default_factory=lambda: ["car", "bus", "truck", "motorcycle", "bicycle"]
    )
    person_class: str = "person"

    device: DeviceType = DeviceType.AUTO

    @model_validator(mode="after")
    def _require_vehicle_classes(self) -> DetectionConfig:
        if not self.vehicle_classes:
            raise ValueError("detection.vehicle_classes must not be empty")
        return self

    @property
    def classes(self) -> list[str]:
        """Classes passed to the detector/tracker. Person is intentionally excluded here:
        this pipeline tracks vehicles only, so people should never get a detection box or
        tracking ID in the first place, not just be hidden from the rendered output."""
        return list(self.vehicle_classes)


class TrackingConfig(BaseModel):
    tracker: TrackerType = TrackerType.BYTETRACK
    min_track_seconds: float = Field(default=0.5, gt=0.0)
    min_visibility_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    reid_max_gap_seconds: float = Field(default=1.0, gt=0.0)
    reid_max_centroid_distance_ratio: float = Field(default=0.06, gt=0.0, le=1.0)
    appearance_reid_enabled: bool = Field(default=False)
    min_movement_ratio: float = Field(default=0.08, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _appearance_reid_requires_botsort(self) -> TrackingConfig:
        if self.appearance_reid_enabled and self.tracker != TrackerType.BOTSORT:
            raise ValueError("tracking.appearance_reid_enabled requires tracking.tracker: botsort")
        return self


class CongestionThresholds(BaseModel):
    moderate: float
    high: float

    @model_validator(mode="after")
    def _require_increasing_thresholds(self) -> CongestionThresholds:
        if self.high <= self.moderate:
            raise ValueError("congestion.density_thresholds.high must be greater than moderate")
        return self


class CongestionConfig(BaseModel):
    density_thresholds: CongestionThresholds
    persistence_frames: int = Field(default=15, ge=1)


class OutputConfig(BaseModel):
    output_dir: str = "outputs"
    save_annotated_video: bool = True


class SpeedConfig(BaseModel):
    """See analytics.speed.SpeedEstimator: speed is inferred from bounding-box size, not a
    real camera calibration, so it's an estimate."""

    enabled: bool = True
    reference_widths_m: dict[str, float] = Field(
        default_factory=lambda: {
            "car": 1.8,
            "bus": 2.5,
            "truck": 2.5,
            "motorcycle": 0.8,
            "bicycle": 0.6,
        }
    )
    min_calibration_samples: int = Field(default=30, ge=1)


class PipelineConfig(BaseModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    congestion: CongestionConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    speed: SpeedConfig = Field(default_factory=SpeedConfig)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root in {config_path} must be a mapping")

    try:
        return PipelineConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Invalid configuration in {config_path}: {exc}") from exc
