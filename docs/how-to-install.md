# Install the application

These instructions set up the standard pipeline on macOS. Python 3.11 is recommended; the scripts require Python 3.10 or newer. No NVIDIA GPU is needed. Keep the analyser and optional TrackNet environments separate.

## 1. Install system tools

Install Homebrew if needed, then install Python and FFmpeg. FFmpeg also provides `ffprobe`.

```sh
brew install python@3.11 ffmpeg
```

Confirm the commands are available:

```sh
python3.11 --version
ffmpeg -version
ffprobe -version
```

## 2. Create the analyser environment

Run these commands from the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the pipeline's Python packages:

| Package | Use |
|---|---|
| `mediapipe` | Pose Landmarker inference in Stages 8 and 11. |
| `matplotlib` | Movement heatmaps and report figures. |
| `numpy` | Tracking, coordinate transforms, and metrics. |
| `onnxruntime` | Stage 1 person detection. |
| `opencv-contrib-python` | Video processing, image review windows, and drawing. |

Use only one OpenCV wheel in this environment. Do not also install `opencv-python`; Stage 2 calibration and Stage 3 identity review need GUI support.

On Apple Silicon, Stage 1 tries ONNX Runtime's Core ML provider with CPU-and-GPU compute when available and otherwise falls back to CPU. Stages 0 and 11 can use FFmpeg's VideoToolbox H.264 encoder. Detection, scaling, pose inference, and drawing still have CPU paths. For an encoder fallback or warning, see the stage option tables in [Command Pipeline](command_pipeline.md).

Verify the Python packages:

```sh
.venv/bin/python -c 'import cv2, matplotlib, mediapipe, numpy, onnxruntime; print("Pipeline packages are ready")'
```

### Start the browser dashboard

From the project root, run:

```sh
.venv/bin/python scripts/match_ui.py
```

It opens `http://127.0.0.1:8765`. The web server uses Python's standard library; no additional web framework or pip packages are required. Add `--port 9000` if the default port is in use. To allow a trusted device on the same LAN to connect, use `--lan` and open `http://<this-mac-lan-ip>:8765` from that device. Keep the terminal open while the dashboard is in use. Runs and uploaded videos stay in `runs/` until you delete them in the dashboard.

## 3. Download model files

Model weights are not stored in the repository. Put them in the Git-ignored `models/` directory.

### Person detector

Stage 1 requires a YOLOX-S model exported to ONNX. Follow the [official YOLOX instructions](https://github.com/Megvii-BaseDetection/YOLOX), using a 640×640 input and the raw head output (do not enable `decode_in_inference` in the exporter). The COCO `person` class must be class 0. Save the export as:

```text
models/yolox_s.onnx
```

YOLOX is Apache-2.0 licensed. Check upstream terms for the specific checkpoint you use.

### Pose model

Stages 8 and 11 use Google's heavy Pose Landmarker task file:

```sh
mkdir -p models
curl -fL \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
  -o models/pose_landmarker_heavy.task
```

The package script includes `models/*.onnx` and `models/*.task` when present. It does not include match footage or generated run folders. A virtual environment cannot be moved between Macs; install it again on the destination system.

## Optional: TrackNet shuttle tracking

TrackNet adds shuttle observations, tentative serve-origin and rally-end cues, shot estimates, and a video trail. It is optional. Install it into a separate Python environment because it uses PyTorch and has different dependency constraints.

```sh
export TRACKNET_DIR="$HOME/TrackNetV3"
export TRACKNET_ENV="$HOME/.venvs/tracknetv3"

if [ ! -f "$TRACKNET_DIR/predict.py" ]; then
  git clone https://github.com/qaz812345/TrackNetV3.git "$TRACKNET_DIR"
fi

python3.11 -m venv "$TRACKNET_ENV"
"$TRACKNET_ENV/bin/python" -m pip install --upgrade pip
"$TRACKNET_ENV/bin/python" -m pip install -r requirements-tracknet.txt
```

Download the `TrackNetV3_ckpts.zip` checkpoint archive from the [TrackNetV3 checkpoint link](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing), extract `TrackNet_best.pt`, and keep it outside the repository, for example:

```text
~/TrackNetV3-models/TrackNet_best.pt
```

Check whether PyTorch can use Apple Silicon's MPS backend:

```sh
"$TRACKNET_ENV/bin/python" -c 'import torch; mps=getattr(torch.backends, "mps", None); print("PyTorch:", torch.__version__, "MPS available:", bool(mps and mps.is_available()))'
```

When running Stage 3A, pass the TrackNet checkout, its Python executable, and the checkpoint file. A short proxy clip is useful for checking setup before tracking a full match. See [the TrackNet pilot notes](tracknet_pilot_plan.md) and [Command Pipeline](command_pipeline.md). TrackNet source and checkpoint terms are separate from this application's license; check their upstream license information.

## Optional: local AI coaching

Stage 10A can export a prompt-and-evidence Markdown file without making a network request, or call Ollama on the same computer. Install [Ollama for macOS](https://ollama.com/download/mac), then download a model:

```sh
ollama pull qwen2.5:7b
```

The Ollama mode uses a local endpoint. The context-file mode shares data only if you choose to paste it into another service.

## Package the application for another Mac

Create an archive from the repository root:

```sh
./compress_application.sh
```

It writes a timestamped ZIP under `dist/`, including scripts, documentation, requirements, and any supported local models under `models/`. It excludes match runs, virtual environments, caches, and Git metadata. To include a TrackNet checkout and a checkpoint kept outside it, use:

```sh
./compress_application.sh /tmp/badminton-analyser.zip \
  --tracknet-path "$HOME/TrackNetV3" \
  --tracknet-checkpoint "$HOME/TrackNetV3-models/TrackNet_best.pt"
```

Extract the archive on the destination Mac and recreate both environments there. Python virtual environments are machine-specific and cannot be copied between Macs. The project is GPL-3.0-only; third-party source and model weights retain their own terms.

## Next step

Read [Pipeline Architecture](pipeline_architecture.md) to understand the review gates, then use [Command Pipeline](command_pipeline.md) for commands and UI instructions. For optional shuttle-tracking setup and pilot commands, see [TrackNet pilot notes](tracknet_pilot_plan.md).
