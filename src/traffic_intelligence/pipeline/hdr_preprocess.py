from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("pipeline.hdr_preprocess")

# Transfer functions a naive YUV->BGR decode does not linearize correctly: OpenCV/FFmpeg's
# default swscale path treats these samples as if they were ordinary gamma-encoded (bt709)
# values, which is close enough for mid-tones but clips highlights and mutes contrast on
# iPhone-style HDR/Dolby Vision footage (HLG or PQ base layer). Re-encoding once through a
# real HDR->SDR tonemap before detection recovers that contrast for small/distant objects.
_HDR_TRANSFER_FUNCTIONS = {"arib-std-b67", "smpte2084", "smpte428"}

_TONE_MAP_FILTER = (
    "zscale=transfer=linear:npl=100,"
    "tonemap=hable:desat=0,"
    "zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=tv,"
    "format=yuv420p"
)


class HDRPreprocessError(Exception):
    """Raised when an HDR source is detected but ffmpeg/ffprobe can't process it."""


def _ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def is_hdr_source(path: str | Path) -> bool:
    """Detects an HDR transfer function (HLG/PQ, incl. a Dolby Vision base layer) via ffprobe.

    Returns False (rather than raising) when ffprobe is unavailable or the file can't be
    probed -- callers treat that as "nothing to convert" and fall back to decoding the
    original file directly, same as before this preprocessing step existed.
    """
    ffprobe = _ffprobe_path()
    if ffprobe is None:
        return False

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_transfer",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return False

    if not streams:
        return False
    return streams[0].get("color_transfer") in _HDR_TRANSFER_FUNCTIONS


def tone_map_to_sdr(input_path: str | Path, output_path: Path) -> Path:
    """Transcodes an HDR source to an upright SDR intermediate: HLG/PQ -> linear light ->
    Hable tonemap -> bt709, baking in the container's rotation (display matrix) at the same
    time since ffmpeg auto-rotates when it re-encodes video through a filter graph.

    This is a one-time cost paid before detection, not per-frame: ffmpeg re-encodes the whole
    clip once, then every downstream stage (detection, tracking, the annotated-video writer)
    reads the already-upright, already-tonemapped result.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HDRPreprocessError("ffmpeg not found on PATH; cannot tone-map HDR source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            _TONE_MAP_FILTER,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-an",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise HDRPreprocessError(f"ffmpeg HDR tone-map failed (exit {result.returncode}): {result.stderr[-800:]}")

    return output_path


def resolve_input_video(input_path: Path, cache_dir: Path, enabled: bool) -> Path:
    """Returns the path the pipeline should actually decode: the original file, unless it's
    an HDR source and tone-mapping is enabled and available, in which case a cached upright
    SDR intermediate is produced (once) and returned instead.

    Failures fall back to the original file with a warning rather than aborting the run --
    an un-tonemapped decode (today's behavior) is strictly better than no output at all.
    """
    if not enabled or not is_hdr_source(input_path):
        return input_path

    cached = cache_dir / f"{input_path.stem}_sdr_source.mp4"
    if cached.exists() and cached.stat().st_mtime >= input_path.stat().st_mtime:
        logger.info("Reusing cached HDR->SDR intermediate: %s", cached)
        return cached

    logger.info("HDR source detected (%s); tone-mapping to SDR before inference...", input_path.name)
    try:
        tone_map_to_sdr(input_path, cached)
    except HDRPreprocessError as exc:
        logger.warning("%s -- falling back to the original (HDR) file for decoding.", exc)
        return input_path

    logger.info("HDR->SDR tone-map written to %s", cached)
    return cached
