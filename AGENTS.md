# Repository Guidelines

## Project Structure & Module Organization

- `scripts/` contains one executable Python module per pipeline stage (`stage0_proxy.py` through `stage11_video.py`), including optional Stage 10A coaching, plus the Stage 4 local review UI.
- `templates/` contains the local browser UI and example review JSON files; keep UI assets local and dependency-free where practical.
- `README.md` covers setup and review checkpoints; `command_pipeline.md` contains runnable stage commands.
- `runs/<match-name>/` holds per-match source video, generated JSON, contact sheets, review files, and rendered videos. Run contents are Git-ignored.
- `models/` holds local model weights and is Git-ignored. The package script includes ONNX and MediaPipe task weights when present.

Keep runtime outputs out of the repository root. Update the README and command examples when script behavior, arguments, or output formats change.

## Build, Test, and Development Commands

Create `.venv` with Python 3.11, install `requirements.txt`, and install FFmpeg with `ffprobe` as described in `README.md`. Run an individual stage with `.venv/bin/python scripts/stageN_name.py ...`; direct its generated files to that match's `runs/<match-name>/` directory. After Stage 4, launch `scripts/stage4_review_ui.py` with the proxy, player tracks, serve JSON, and review directory to edit candidate decisions and scoring in a local browser. Full commands are in `command_pipeline.md`. There is no automated test suite currently. Do not add or run tests unless requested.

## Coding Style & Naming Conventions

For Python modules, use four spaces for indentation, descriptive `snake_case` names, and small functions aligned with the pipeline stages. Keep time, court coordinates, and distances in explicit units. Preserve the documented court coordinate system: `x` spans 0–6.1 m and `y` spans 0–13.4 m. Use clear, consistent JSON field names and update examples when schemas change.

## Testing Guidelines

There is no test framework or test suite yet. If one is introduced, cover court-coordinate transforms, scoring rules, and missing or noisy detections with synthetic inputs that do not depend on private match footage. Document its command before relying on it.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects (for example, `Document calibration workflow`). Do not commit match videos, model weights, generated run data, or machine-specific paths. Pull requests should explain the change and its validation, link related issues when applicable, and include representative before/after images for changes to generated charts or video overlays.

## Security & Configuration

Do not commit match videos, credentials, or machine-specific paths. Keep optional model weights and generated analysis outputs outside version control unless a small, redistributable fixture is needed for development.
