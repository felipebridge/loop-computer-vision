from __future__ import annotations

from pathlib import Path

import pytest

from traffic_intelligence.cli import main
from traffic_intelligence.persistence.writers import write_tracks_csv
from traffic_intelligence.schemas.track import TrackSummary


def _summary(
    track_id: int,
    class_name: str,
    class_id: int,
    first_timestamp: float,
    last_timestamp: float,
    avg_speed_kmh: float | None = None,
    max_speed_kmh: float | None = None,
    speed_estimated: bool = False,
) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        mean_confidence=0.9,
        frame_count=10,
        first_frame=0,
        last_frame=9,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        avg_speed_kmh=avg_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        speed_estimated=speed_estimated,
    )


@pytest.fixture
def tracks_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tracks.csv"
    write_tracks_csv(
        [
            _summary(1, "car", 2, 0.0, 4.0, avg_speed_kmh=40.0, max_speed_kmh=50.0, speed_estimated=True),
            _summary(2, "bus", 5, 1.0, 6.0, avg_speed_kmh=30.0, max_speed_kmh=34.0, speed_estimated=True),
            _summary(3, "person", 0, 2.0, 8.0),
        ],
        path,
    )
    return path


def test_analyze_reports_counts_span_and_speeds(tracks_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main(["analyze", "--input", str(tracks_csv)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Vehicles:    2" in out
    assert "Pedestrians: 1" in out
    assert "car: 1" in out
    assert "bus: 1" in out
    # Span runs from the earliest first_timestamp (0.0) to the latest last_timestamp (8.0).
    assert "Activity span: 8.0 s" in out
    # Mean of the two estimated average speeds: (40 + 30) / 2.
    assert "Average vehicle speed: 35.0 km/h" in out
    assert "Fastest vehicle:       50.0 km/h" in out


def test_analyze_without_speed_estimation_reports_na(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "tracks.csv"
    write_tracks_csv([_summary(1, "car", 2, 0.0, 2.0)], path)

    exit_code = main(["analyze", "--input", str(path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Average vehicle speed: n/a" in out
    assert "Fastest vehicle" not in out


def test_analyze_rejects_csv_missing_expected_columns(tmp_path: Path):
    path = tmp_path / "not_tracks.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    assert main(["analyze", "--input", str(path)]) == 1


def test_analyze_rejects_header_only_csv(tmp_path: Path):
    path = tmp_path / "tracks.csv"
    write_tracks_csv([], path)

    assert main(["analyze", "--input", str(path)]) == 1


def test_analyze_rejects_zero_byte_file(tmp_path: Path):
    # A 0-byte file (failed export, accidental `touch`) makes read_csv raise
    # EmptyDataError, a sibling of ParserError rather than a subclass of it.
    path = tmp_path / "tracks.csv"
    path.write_bytes(b"")

    assert main(["analyze", "--input", str(path)]) == 1


def test_analyze_rejects_non_csv_file(tmp_path: Path):
    path = tmp_path / "binary.bin"
    path.write_bytes(bytes(range(256)))

    assert main(["analyze", "--input", str(path)]) == 1


def test_analyze_rejects_missing_file(tmp_path: Path):
    assert main(["analyze", "--input", str(tmp_path / "nope.csv")]) == 1
