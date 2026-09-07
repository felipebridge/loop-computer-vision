<p align="center">
  <img src="docs/assets/logo.png" alt="Loop logo" width="120">
</p>

<h1 align="center">Loop Computer Vision</h1>

<p align="center">
Real-time vehicle & pedestrian tracking with live traffic congestion classification.
</p>

<p align="center">
  <a href="https://github.com/felipebridge/loop-computer-vision/actions/workflows/ci.yml"><img src="https://github.com/felipebridge/loop-computer-vision/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

<p align="center">
  <img src="docs/assets/demo.gif" alt="Annotated street video with live vehicle/pedestrian tracking boxes and IDs" width="300">
</p>

## What it does

Tracks vehicles and pedestrians in traffic video (each counted once, not once per frame),
classifies congestion as LOW / MODERATE / HIGH, and renders an annotated video plus a
Streamlit dashboard over the results.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"

python -m traffic_intelligence run --input data/raw/avenue.mp4
python -m traffic_intelligence dashboard
```

## Stack

Python 3.11+ · Ultralytics YOLO (ByteTrack/BoT-SORT) · OpenCV · Pydantic · Pandas ·
Streamlit/Altair · pytest

## Status

This project is actively being developed and improved. If you're interested in contributing, see [CONTRIBUTING.md](CONTRIBUTING.md) or reach out at felibridge49@gmail.com.

## License

MIT 

