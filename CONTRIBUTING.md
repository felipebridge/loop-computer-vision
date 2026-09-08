# Contributing

Thanks for your interest in Loop Computer Vision. The project is under active
development — issues, ideas, and pull requests are welcome.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Run the pipeline and dashboard locally:

```bash
python -m traffic_intelligence run --input data/raw/avenue.mp4
python -m traffic_intelligence dashboard
```

## Before opening a PR

```bash
ruff check src tests dashboard scripts
pytest -q
```

Both run in CI on every PR, so passing them locally first saves a round trip.

## Making changes

- Keep PRs focused on a single change; unrelated cleanups belong in their own PR.
- Match the existing code style (see `pyproject.toml` for `ruff` config).
- Add or update tests for behavior you change.
- Write commit messages and PR descriptions that explain *why*, not just *what*.

## Reporting bugs / proposing features

Open a GitHub issue with steps to reproduce (for bugs) or the motivation and
proposed approach (for features). For anything larger, feel free to open an
issue to discuss the approach before writing code.

## Questions

Reach out at felibridge49@gmail.com.
