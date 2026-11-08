#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="badminton_analyser"
STAMP="$(date +%Y%m%d-%H%M%S)"

OUTPUT_PATH=""
TRACKNET_PATH=""
TRACKNET_CHECKPOINT_PATH=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tracknet-path)
      [[ "$#" -ge 2 ]] || { printf 'Missing path after --tracknet-path\n' >&2; exit 2; }
      TRACKNET_PATH="$2"
      shift 2
      ;;
    --tracknet-checkpoint)
      [[ "$#" -ge 2 ]] || { printf 'Missing path after --tracknet-checkpoint\n' >&2; exit 2; }
      TRACKNET_CHECKPOINT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      printf 'Usage: %s [output.zip] [--tracknet-path /path/to/TrackNetV3] [--tracknet-checkpoint /path/to/TrackNet_best.pt]\n' "$0"
      exit 0
      ;;
    -* )
      printf 'Unknown option: %s\nUsage: %s [output.zip] [--tracknet-path /path/to/TrackNetV3] [--tracknet-checkpoint /path/to/TrackNet_best.pt]\n' "$1" "$0" >&2
      exit 2
      ;;
    *)
      if [[ -n "$OUTPUT_PATH" ]]; then
        printf 'Only one output archive path may be provided.\n' >&2
        exit 2
      fi
      OUTPUT_PATH="$1"
      shift
      ;;
  esac
done

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$PROJECT_ROOT/dist/${APP_NAME}-${STAMP}.zip"
elif [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT_PATH"
fi

if [[ -n "$TRACKNET_PATH" ]]; then
  if [[ "$TRACKNET_PATH" != /* ]]; then
    TRACKNET_PATH="$PROJECT_ROOT/$TRACKNET_PATH"
  fi
  TRACKNET_PATH="$(cd "$TRACKNET_PATH" 2>/dev/null && pwd)" || {
    printf 'TrackNet path does not exist: %s\n' "$TRACKNET_PATH" >&2
    exit 1
  }
  if [[ ! -f "$TRACKNET_PATH/predict.py" ]]; then
    printf 'TrackNet path must contain predict.py: %s\n' "$TRACKNET_PATH" >&2
    exit 1
  fi
fi
if [[ -n "$TRACKNET_CHECKPOINT_PATH" ]]; then
  if [[ "$TRACKNET_CHECKPOINT_PATH" != /* ]]; then
    TRACKNET_CHECKPOINT_PATH="$PROJECT_ROOT/$TRACKNET_CHECKPOINT_PATH"
  fi
  if [[ ! -f "$TRACKNET_CHECKPOINT_PATH" ]]; then
    printf 'TrackNet checkpoint does not exist: %s\n' "$TRACKNET_CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ -z "$TRACKNET_PATH" ]]; then
    printf -- '--tracknet-checkpoint requires --tracknet-path.\n' >&2
    exit 2
  fi
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

for required in README.md LICENSE AGENTS.md docs/README.md docs/how-to-install.md docs/pipeline_architecture.md docs/command_pipeline.md docs/tracknet_pilot_plan.md docs/tracknet_event_analysis_plan.md requirements.txt requirements-tracknet.txt .gitignore templates/match_ui.html templates/serve_review.html templates/manual_score_review.json templates/coach_review_selection.json; do
  if [[ ! -f "$PROJECT_ROOT/$required" ]]; then
    printf 'Required project file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

PACKAGE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/badminton-analyser-package.XXXXXX")"
trap 'rm -rf "$PACKAGE_TMP"' EXIT
PACKAGE_DIR="$PACKAGE_TMP/$APP_NAME"
mkdir -p "$PACKAGE_DIR/scripts" "$PACKAGE_DIR/models" "$PACKAGE_DIR/templates"

cp -p \
  "$PROJECT_ROOT/README.md" \
  "$PROJECT_ROOT/LICENSE" \
  "$PROJECT_ROOT/AGENTS.md" \
  "$PROJECT_ROOT/requirements.txt" \
  "$PROJECT_ROOT/requirements-tracknet.txt" \
  "$PROJECT_ROOT/.gitignore" \
  "$PROJECT_ROOT/compress_application.sh" \
  "$PACKAGE_DIR/"

cp -R "$PROJECT_ROOT/docs" "$PACKAGE_DIR/docs"

for script in "$PROJECT_ROOT"/scripts/*.py; do
  [[ -f "$script" ]] || continue
  cp -p "$script" "$PACKAGE_DIR/scripts/"
done

for template in "$PROJECT_ROOT"/templates/*; do
  [[ -f "$template" ]] || continue
  cp -p "$template" "$PACKAGE_DIR/templates/"
done

if [[ -n "$TRACKNET_PATH" ]]; then
  mkdir -p "$PACKAGE_DIR/TrackNetV3"
  TRACKNET_CHECKPOINT="$TRACKNET_CHECKPOINT_PATH"
  if [[ -z "$TRACKNET_CHECKPOINT" ]]; then
    TRACKNET_CHECKPOINT="$(find "$TRACKNET_PATH" -type f -name 'TrackNet_best.pt' -print -quit)"
  fi
  cp -R "$TRACKNET_PATH"/. "$PACKAGE_DIR/TrackNetV3/"
  find "$PACKAGE_DIR/TrackNetV3" -type d \( \
    -name .git -o -name .venv -o -name venv -o -name __pycache__ -o \
    -name data -o -name corrected_test_label -o -name exp -o -name pred_result \
  \) -prune -exec rm -rf {} +
  find "$PACKAGE_DIR/TrackNetV3" -type f \( \
    -name '*.pyc' -o -name '*.mp4' -o -name '*.avi' -o -name '*.csv' -o \
    -name '*.pt' -o -name '*.pth' -o -name '.DS_Store' \
  \) -delete
  if [[ -n "$TRACKNET_CHECKPOINT" && -f "$TRACKNET_CHECKPOINT" ]]; then
    mkdir -p "$PACKAGE_DIR/TrackNetV3/models"
    cp -p "$TRACKNET_CHECKPOINT" "$PACKAGE_DIR/TrackNetV3/models/TrackNet_best.pt"
  fi
fi

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
if [[ -n "$TRACKNET_PATH" ]]; then
  printf 'Included optional TrackNetV3 source from: %s\n' "$TRACKNET_PATH"
  if [[ -f "$PACKAGE_DIR/TrackNetV3/models/TrackNet_best.pt" ]]; then
    printf 'Included TrackNet model: TrackNetV3/models/TrackNet_best.pt\n'
  else
    printf 'TrackNet_best.pt was not in that checkout; download it using the link in README.md.\n'
  fi
fi
if [[ -f "$PACKAGE_DIR/models/pose_landmarker_heavy.task" ]]; then
  printf 'Included model: models/pose_landmarker_heavy.task\n'
else
  printf 'MediaPipe pose model was not found; download it on the destination Mac as described in README.md.\n'
fi
