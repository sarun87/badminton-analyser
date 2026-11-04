#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="badminton_analyser"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [[ "$#" -gt 1 ]]; then
  printf 'Usage: %s [output.zip]\n' "$0" >&2
  exit 2
fi

if [[ "$#" -eq 1 ]]; then
  OUTPUT_PATH="$1"
  if [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT_PATH"
  fi
else
  OUTPUT_PATH="$PROJECT_ROOT/dist/${APP_NAME}-${STAMP}.zip"
fi

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$(cd "$OUTPUT_DIR" && pwd)/$(basename "$OUTPUT_PATH")"
if [[ -e "$OUTPUT_PATH" ]]; then
  printf 'Archive already exists: %s\nChoose another path or remove it first.\n' "$OUTPUT_PATH" >&2
  exit 1
fi

if ! command -v ditto >/dev/null 2>&1; then
  printf 'macOS ditto is required to create the ZIP archive.\n' >&2
  exit 1
fi

for required in README.md AGENTS.md command_pipeline.md requirements.txt .gitignore; do
  if [[ ! -f "$PROJECT_ROOT/$required" ]]; then
    printf 'Required project file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

PACKAGE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/badminton-analyser-package.XXXXXX")"
trap 'rm -rf "$PACKAGE_TMP"' EXIT
PACKAGE_DIR="$PACKAGE_TMP/$APP_NAME"
mkdir -p "$PACKAGE_DIR/scripts" "$PACKAGE_DIR/models"

cp -p \
  "$PROJECT_ROOT/README.md" \
  "$PROJECT_ROOT/AGENTS.md" \
  "$PROJECT_ROOT/command_pipeline.md" \
  "$PROJECT_ROOT/requirements.txt" \
  "$PROJECT_ROOT/.gitignore" \
  "$PROJECT_ROOT/compress_application.sh" \
  "$PACKAGE_DIR/"

for script in "$PROJECT_ROOT"/scripts/*.py; do
  [[ -f "$script" ]] || continue
  cp -p "$script" "$PACKAGE_DIR/scripts/"
done

touch "$PACKAGE_DIR/models/.gitkeep"
for model in "$PROJECT_ROOT"/models/*.onnx "$PROJECT_ROOT"/models/*.task; do
  [[ -f "$model" ]] || continue
  cp -L -p "$model" "$PACKAGE_DIR/models/"
done

# Older setups may keep this official model in the macOS cache directory.
POSE_CACHE_MODEL="${HOME}/Library/Caches/mediapipe/pose_landmarker_heavy.task"
if [[ ! -f "$PACKAGE_DIR/models/pose_landmarker_heavy.task" && -f "$POSE_CACHE_MODEL" ]]; then
  cp -p "$POSE_CACHE_MODEL" "$PACKAGE_DIR/models/pose_landmarker_heavy.task"
fi

ditto -c -k --sequesterRsrc --keepParent "$PACKAGE_DIR" "$OUTPUT_PATH"
printf 'Package created: %s\n' "$OUTPUT_PATH"
printf 'Included Python scripts and documentation; excluded runs, Git metadata, virtual environments, and caches.\n'
if [[ -f "$PACKAGE_DIR/models/yolox_s.onnx" ]]; then
  printf 'Included model: models/yolox_s.onnx\n'
else
  printf 'YOLOX-S ONNX model was not found; download it on the destination Mac as described in README.md.\n'
fi
if [[ -f "$PACKAGE_DIR/models/pose_landmarker_heavy.task" ]]; then
  printf 'Included model: models/pose_landmarker_heavy.task\n'
else
  printf 'MediaPipe pose model was not found; download it on the destination Mac as described in README.md.\n'
fi
