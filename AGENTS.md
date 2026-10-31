# Repository Guidelines

## Project Structure & Module Organization

`README.md` describes the pipeline and commands. Runnable code lives in `scripts/`: `00_proxy.py` creates a per-video proxy and job manifest; `01`–`11` implement detection through overlay rendering; `common.py` holds shared helpers; `init_macos.sh` installs local prerequisites. Generated videos, stage JSON, previews, and reports belong under `runs/`, which is ignored by Git. Keep future modules grouped by pipeline stage and reusable samples or fixtures in clearly named directories.

## Build, Test, and Development Commands

Initialize the macOS environment with `bash scripts/init_macos.sh`, activate it with `source .venv/bin/activate`, then run the ordered per-video commands in `README.md`. The initializer detects Intel or Apple Silicon and selects native Homebrew; stages 1 and 8 use locally run Ultralytics models (MPS when available, otherwise CPU). No automated test or lint command is configured. Document setup and exact commands here and in the README as tools are added.

## Coding Style & Naming Conventions

Keep pipeline stages small and make input and output formats explicit. Use descriptive stage names consistent with the README (`players`, `serves`, `motion`, `pose_stats`). For Python code, use four-space indentation and `snake_case`; for JSON outputs, preserve the documented field meanings and use valid JSON numbers and strings. Document court coordinates in metres, with the camera behind the near baseline, so stages interpret positions consistently. Avoid silently guessing player identity when detections are uncertain; represent gaps explicitly.

## Testing Guidelines

No automated test framework or coverage target exists yet. For changes to analysis logic, add focused tests alongside the chosen framework. Validate results against sample frames and the README's manual checks: projected court lines align with the court, tracked points land on player feet, serve and side assignments agree with the service-court rule, rally ends are checked visually, and movement distances are plausible. Include the data or frame timestamps used for manual checks in the change description.

## Commit & Pull Request Guidelines

This checkout has no Git history to establish a commit convention. Use short, imperative commit subjects, such as `Document rally validation`. Pull requests should summarize the affected pipeline stages, explain any output schema changes, link related work when applicable, and include test results or representative frame screenshots for visual changes. Update `README.md` when behavior or setup instructions change.
