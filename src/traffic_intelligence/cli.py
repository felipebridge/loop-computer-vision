from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from traffic_intelligence.config.settings import ConfigError, load_config
from traffic_intelligence.pipeline.video_source import VideoSourceError
from traffic_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger("cli")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic_intelligence",
        description="Count vehicles and people in traffic video and classify the congestion level.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline on a video")
    run_parser.add_argument("--input", required=True, help="Path to the input video file")
    run_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to a pipeline YAML config"
    )
    run_parser.add_argument(
        "--output-dir", default=None, help="Override output.output_dir from the config"
    )
    run_parser.set_defaults(handler=_run_command)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Recompute counts, activity span and speed stats from a previously exported tracks.csv",
    )
    analyze_parser.add_argument("--input", required=True, help="Path to an exported tracks.csv")
    analyze_parser.set_defaults(handler=_analyze_command)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")
    dashboard_parser.set_defaults(handler=_dashboard_command)

    return parser


def _run_command(args: argparse.Namespace) -> int:
    """Runs the pipeline. UltralyticsTracker (torch) is imported before
    pandas-importing modules to avoid a Windows-specific DLL init conflict
    between torch's and pandas' bundled OpenMP/MKL runtimes; see
    docs/architecture.md."""
    from traffic_intelligence.pipeline.runner import PipelineRunner
    from traffic_intelligence.persistence.writers import write_json, write_tracks_csv
    from traffic_intelligence.tracking.ultralytics_tracker import ModelLoadError

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input video not found: %s", input_path)
        return 1

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output.output_dir)

    try:
        runner = PipelineRunner(config)
        result = runner.run(input_path, output_dir)
    except (VideoSourceError, ModelLoadError) as exc:
        logger.error("%s", exc)
        return 1

    analytics_dir = output_dir / "analytics"
    tracks_dir = output_dir / "tracks"

    write_tracks_csv(result.track_summaries, tracks_dir / "tracks.csv")
    write_json(result.metrics, analytics_dir / "metrics.json")

    summary = {
        "input_video": str(input_path),
        "total_vehicles": result.metrics.total_vehicles,
        "total_pedestrians": result.metrics.total_pedestrians,
        "vehicles_per_class": result.metrics.vehicles_per_class,
        "traffic_level": result.metrics.traffic_level.value,
        "average_vehicle_speed_kmh": result.metrics.average_vehicle_speed_kmh,
        "speed_estimated": result.metrics.speed_estimated,
        "annotated_video_path": str(result.annotated_video_path) if result.annotated_video_path else None,
    }
    write_json(summary, analytics_dir / "summary.json")

    logger.info(
        "Vehicles: %d | Pedestrians: %d | Traffic level: %s | Avg speed: %s",
        result.metrics.total_vehicles,
        result.metrics.total_pedestrians,
        result.metrics.traffic_level.value,
        f"{result.metrics.average_vehicle_speed_kmh:.1f} km/h (est.)"
        if result.metrics.average_vehicle_speed_kmh is not None
        else "n/a (not enough vehicles seen yet to calibrate)",
    )
    logger.info("Outputs written to %s", output_dir.resolve())
    return 0


def _analyze_command(args: argparse.Namespace) -> int:
    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("tracks.csv not found: %s", input_path)
        return 1

    frame = pd.read_csv(input_path)
    if frame.empty:
        logger.error("No track data found in %s", input_path)
        return 1

    vehicles = frame[frame["class_name"] != "person"]
    pedestrians = frame[frame["class_name"] == "person"]
    print(f"Vehicles:    {vehicles.shape[0]}")
    print(f"Pedestrians: {pedestrians.shape[0]}")
    print("Vehicles per class:")
    for class_name, count in vehicles["class_name"].value_counts().items():
        print(f"  {class_name}: {count}")

    # Each track row records when it was first and last seen, so the gap between the
    # earliest and latest of those timestamps is the window of the video that actually
    # contained tracked traffic. It is recomputable from the export alone, unlike the
    # congestion level, which needs the per-frame densities only `run` observes.
    activity_span_s = float(frame["last_timestamp"].max() - frame["first_timestamp"].min())
    print(f"Activity span: {activity_span_s:.1f} s (first to last detection)")

    estimated = vehicles[vehicles["speed_estimated"].astype(bool)]
    speeds = estimated["avg_speed_kmh"].dropna()
    if speeds.empty:
        print("Average vehicle speed: n/a (not enough vehicles seen yet to calibrate)")
    else:
        print(f"Average vehicle speed: {speeds.mean():.1f} km/h (est.)")
        fastest = estimated["max_speed_kmh"].dropna()
        if not fastest.empty:
            print(f"Fastest vehicle:       {fastest.max():.1f} km/h (est.)")

    return 0


def _dashboard_command(args: argparse.Namespace) -> int:
    app_path = _REPO_ROOT / "dashboard" / "app.py"
    if not app_path.exists():
        logger.error(
            "dashboard/app.py not found at %s. The dashboard must be launched from a cloned "
            "repository checkout.",
            app_path,
        )
        return 1

    return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)]).returncode


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
