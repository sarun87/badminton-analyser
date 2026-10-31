#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This initializer supports macOS only." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SHELL_ARCH="$(uname -m)"
HAS_ARM64="$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)"
if [[ "$HAS_ARM64" == "1" || "$SHELL_ARCH" == "arm64" ]]; then
  MAC_ARCH=arm64
  BREW_PREFIX=/opt/homebrew
  PYTHON_ARCH=arm64

  # Use native Homebrew and Python even if Terminal was launched with Rosetta.
  if [[ "$SHELL_ARCH" != "arm64" && -x /usr/bin/arch ]]; then
    echo "Apple Silicon detected under Rosetta. Restarting setup in native arm64 mode..."
    exec /usr/bin/arch -arm64 /bin/bash "$0" "$@"
  fi
elif [[ "$SHELL_ARCH" == "x86_64" ]]; then
  MAC_ARCH=x86_64
  BREW_PREFIX=/usr/local
  PYTHON_ARCH=x86_64
else
  echo "Unsupported Mac architecture: $SHELL_ARCH" >&2
  exit 1
fi

EXPECTED_BREW="$BREW_PREFIX/bin/brew"
if [[ -x "$EXPECTED_BREW" ]]; then
  BREW="$EXPECTED_BREW"
elif command -v brew >/dev/null 2>&1; then
  BREW="$(command -v brew)"
else
  echo "Detected $MAC_ARCH hardware. Homebrew is required at $EXPECTED_BREW." >&2
  echo "Install Homebrew from https://brew.sh, then run this script again." >&2
  exit 1
fi
if ! FOUND_BREW_PREFIX="$("$BREW" --prefix 2>/dev/null)" || [[ "$FOUND_BREW_PREFIX" != "$BREW_PREFIX" ]]; then
  echo "Detected $MAC_ARCH hardware, but the selected Homebrew is not the native $BREW_PREFIX installation." >&2
  echo "Install native Homebrew from https://brew.sh, then run this script again." >&2
  exit 1
fi

echo "Installing FFmpeg and Python 3.12 with Homebrew..."
"$BREW" install ffmpeg python@3.12

PYTHON_PREFIX="$("$BREW" --prefix python@3.12)"
PYTHON="$PYTHON_PREFIX/bin/python3.12"
FFMPEG_PREFIX="$("$BREW" --prefix ffmpeg)"

if [[ ! -x "$PYTHON" || ! -x "$FFMPEG_PREFIX/bin/ffmpeg" || ! -x "$FFMPEG_PREFIX/bin/ffprobe" ]]; then
  echo "Homebrew finished, but Python, ffmpeg, or ffprobe was not found." >&2
  exit 1
fi

INSTALLED_ARCH="$("$PYTHON" -c 'import platform; print(platform.machine())')"
if [[ "$INSTALLED_ARCH" != "$PYTHON_ARCH" ]]; then
  echo "Homebrew Python has architecture $INSTALLED_ARCH; expected $PYTHON_ARCH for this Mac." >&2
  exit 1
fi

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ -e "$PROJECT_DIR/.venv" ]]; then
  if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "$PROJECT_DIR/.venv exists but is not a Python virtual environment; move it aside and rerun." >&2
    exit 1
  fi
  if ! VENV_INFO="$("$VENV_PYTHON" -c 'import platform, sys; print(f"{sys.version_info.major}.{sys.version_info.minor} {platform.machine()}")' 2>/dev/null)"; then
    echo "$PROJECT_DIR/.venv cannot run on this Mac; it may have been created for another CPU architecture." >&2
    echo "Move the existing .venv directory aside, then rerun this script." >&2
    exit 1
  fi
  VENV_VERSION="${VENV_INFO%% *}"
  VENV_ARCH="${VENV_INFO##* }"
  if [[ "$VENV_VERSION" != "3.12" || "$VENV_ARCH" != "$PYTHON_ARCH" ]]; then
    echo "$PROJECT_DIR/.venv uses Python $VENV_VERSION ($VENV_ARCH); this Mac needs Python 3.12 ($PYTHON_ARCH)." >&2
    echo "Move the existing .venv directory aside, then rerun this script." >&2
    exit 1
  fi
else
  "$PYTHON" -m venv "$PROJECT_DIR/.venv"
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"

cat <<EOF
Ready.
Detected Mac architecture: $MAC_ARCH
Python environment: $PROJECT_DIR/.venv
FFmpeg: $FFMPEG_PREFIX/bin/ffmpeg
ffprobe: $FFMPEG_PREFIX/bin/ffprobe

The pipeline packages are installed in this local environment. YOLO weights download on first use.
Inference runs locally; Apple Silicon uses MPS when available, while Intel Macs use CPU. Activate the environment with:
  source "$PROJECT_DIR/.venv/bin/activate"
EOF
