# Badminton Match Analysis

A local Python pipeline that turns one singles-match video into a reviewed score log, rally and movement metrics, pose summaries, coaching notes, and an optional narrated video. It runs locally without an NVIDIA GPU. Stage 1 can use Apple Core ML with CPU-and-GPU compute on supported Apple Silicon systems. Stages 0 and 11 use FFmpeg's VideoToolbox H.264 encoder on macOS when available. Stage 10A can use Ollama on the same computer or create a prompt file for a chat model.

## Filming requirements

- Place a steady camera behind one baseline, raised enough to see both players. Keep all visible court lines in frame, including the net, sidelines, baselines, and service lines.
- Keep the camera fixed. A moved camera needs a separate calibration for that position and time range.
- Record at normal speed with steady lighting. 1080p or 4K at 30 fps works well; Stage 0 makes a 720p proxy.
- Wear clothing that distinguishes the player you want analyzed. Stage 3 asks for a shirt or pants color, assigns that person as R wherever they are on court, and assigns the opponent as O.
- Keep spectators and adjacent courts out of the court area where possible. The detector can find them, so tracking still needs review.

The current tracking and score workflow assumes **singles** and a camera behind a baseline. Side-on footage and doubles need different tracking and calibration logic.

## Install on macOS

Install [Homebrew](https://brew.sh/) first if it is not already available. Then install Python and FFmpeg (which provides both `ffmpeg` and `ffprobe`):

```sh
brew install python@3.11 ffmpeg
```

Python 3.11 is the recommended setup; the scripts use Python 3.10 or newer syntax.

From the repository directory, create a virtual environment and install the runtime packages:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the packages used by the current scripts: MediaPipe, NumPy, ONNX Runtime, and OpenCV with its GUI modules. OpenCV's GUI is needed for Stage 2 calibration and Stage 3's identity review window. Use one OpenCV wheel only; do not add `opencv-python` alongside `opencv-contrib-python`. On Apple Silicon, Stage 1 automatically selects ONNX Runtime's Core ML provider when the installed build exposes it; otherwise it runs on CPU. Core ML is configured for CPU-and-GPU compute, with the CPU provider available for unsupported operations.

Stages 0 and 11 check FFmpeg's encoder list and automatically select `h264_videotoolbox` on macOS when available; otherwise they use `libx264`. VideoToolbox accelerates H.264 encoding. Stage 0 scaling/HDR tone mapping and Stage 11 frame overlays still run through their existing CPU paths. Use `--video-encoder libx264` to force software encoding or `--video-encoder h264_videotoolbox` to require VideoToolbox. VideoToolbox uses a target bitrate (`2500k` for Stage 0, `4M` for Stage 11 by default), while the `libx264` path keeps its CRF quality setting. Both stages accept `--videotoolbox-bitrate` to adjust the hardware encoder bitrate.

Check the installation:

```sh
ffmpeg -version
ffprobe -version
python -c 'import cv2, mediapipe, numpy, onnxruntime; print("Python packages are ready")'
```

### Model files

Stage 1 needs a YOLOX-S model exported to ONNX. The model is not versioned in the repository. Get the pretrained COCO model and export instructions from the [official YOLOX project](https://github.com/Megvii-BaseDetection/YOLOX). Stage 1 expects a 640×640 YOLOX-S model with the raw head output (do not enable `decode_in_inference` in the exporter); the COCO `person` class must be class 0. Save it as `models/yolox_s.onnx`; `models/` is ignored by Git so the large weights stay local. The package script includes that file if it is present. YOLOX is Apache-2.0 licensed.

Stages 8 and 11 need Google's official MediaPipe Pose Landmarker heavy model. If it is not already bundled under `models/`, download it there:

```sh
mkdir -p models
curl -fL \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
  -o models/pose_landmarker_heavy.task
```

The archive script includes `models/*.onnx` and `models/*.task` when present, but never includes match runs.

Stage 11 uses macOS `say` for optional local narration. If the default voice is unavailable, list installed voices with `say -v '?'`, choose one with `--voice`, or pass `--no-voice`. Stage 11 also needs FFmpeg. For AI coaching, Ollama is optional: install the [Ollama macOS app](https://ollama.com/download/mac), then use the local setup described below.

## Start a new match

Create a separate working directory for each match under `runs/`, such as `runs/match-1/`. Put the original video there, or refer to an input video elsewhere while directing all outputs into the run directory. The repository's `.gitignore` excludes run contents, model weights, `.venv`, and Python caches; scripts and documentation remain versionable.

Stage 0 requires the timestamp in the **original video** when the first serve at 0–0 begins. Use `0` if the match starts immediately:

```sh
mkdir -p runs/match-1
.venv/bin/python scripts/stage0_proxy.py \
  runs/match-1/match.mp4 \
  runs/match-1/match-720p.mp4 \
  --match-start-sec 123.4
```

Put the source video at `runs/match-1/match.mp4` or change the input path. Replace `123.4` with the first-serve timestamp. Stage 0 writes a 1280×720, 30 fps SDR video and a `.proxy.json` sidecar in the run directory. All later timestamps start at zero in the proxy; add `first_serve_at_0_0_source_sec` from the sidecar to map a timestamp back to the original video. Stage 0 tone-maps common HDR transfers. It stops if the source color transfer is missing or unrecognized; inspect that video before converting it manually.

The scripts and their argument order are listed below. Complete copy-ready examples are in [command_pipeline.md](command_pipeline.md); set `RUN_DIR` to a new folder for each match and update the sample video filename.

## Pipeline and review checkpoints

| Stage | What it produces | Review before continuing |
|---|---|---|
| 0 — Proxy | Trimmed 720p video and timestamp sidecar | Uses VideoToolbox H.264 encoding on macOS when available. Add `--quiet-ffmpeg` to hide FFmpeg warnings and progress while keeping errors visible. Check that the proxy begins at the first 0–0 serve, has both players and the whole court visible, and plays correctly. |
| 1 — Detection | `dets.json` and a detection contact sheet | Uses Core ML CPU-and-GPU compute when available, otherwise CPU. Check that player boxes are reasonable; adjacent-court players may also be detected. |
| 2 — Calibration | Court homography JSON and overlay image | Manually click at least four known court points. Confirm overlay lines sit on the actual lines; redo it for every camera position. |
| 3 — Tracking | `players.json` and a tracking contact sheet | Choose shirt/pants and its color to identify R. Review the opened image and confirm or swap the R/O assignment before continuing. R may be on either court half. |
| 4 — Serves | `serves.json`, `manual_score_review.json`, review sheets, and optional racket-orientation template | Review candidates, server, and court side. Stage 4 creates a score-review draft with one row per accepted candidate; its server/side and score guesses need review. |
| 5 — Score | `points.json` | Correct every row in Stage 4's `manual_score_review.json`, add any missed video serves or score-only dummy points, and enter the final result. Stage 5 requires every accepted Stage 4 candidate exactly once. |
| 6 — Rally ends | `rallies_review.json` and picture-only contact sheets | The optional `--estimate-rally-ends players.json` option finds the first sustained pause after active movement. Check estimated starts and ends against the frames and correct bad estimates. Do not use hall audio to mark rallies. |
| 7 — Movement | `motion.json` and heatmap images | Runs from the Stage 3 tracks and Stage 6 times. Check for plausible distances and speeds; incorrect rally windows affect these values. |
| 8 — Pose | `pose_stats.json` and `pose_review.jpg` | Confirm the skeleton follows R, as selected in Stage 3, and landmarks align with the body. Measurements are estimates, not proof of why a point was won or lost. |
| 9 — Pickups | `pickups.json` and `pickup_review.jpg` | Inspect candidate bends. For a clear correction, set that point's `landing_side` to `R` (analyzed player side), `O` (opponent side), or `null` (unknown), and set `status` to `manual_confirmed`. Pickup side is indirect evidence. |
| 10 — Report | `report_data.json` and `coach_report.md` | Enter the two player names. Review the selected losses, contact-sheet citations, and estimated patterns before treating them as coaching conclusions. |
| 10A — AI coach (optional) | `AI-coach-context.md` and optionally `AI-coach-narrative.md` | Use Ollama locally or paste the context into a chat model and save its response. This stage is optional; no chat agent is required to run the rest of the pipeline. |
| 11 — Video (optional) | Annotated full-match or short-analysis MP4 | Uses VideoToolbox H.264 encoding on macOS when available. Review the score box, server marker, tracking, court map, commentary, and audio. On the map, the camera-side near court should be below the net and the far court above it; R may appear on either half. Use `--short-mode` to omit the full-match portion, `--no-voice` to omit narration, or `--no-pose` to omit pose inference. |

The manual work is concentrated in selecting the Stage 0 start time, clicking and checking Stage 2 calibration, verifying Stage 5's score, and correcting visibly wrong Stage 6 estimates. Other stages run automatically but still need the checks above. You do not need to use an assistant or cloud service to run the pipeline.

## AI coaching and privacy

Stage 10 is deterministic: it summarizes the local JSON measurements and writes timestamped review notes. It does not call an LLM. Stage 10A can create a self-contained `AI-coach-context.md` file for a user-selected chat agent, or send the evidence to Ollama at `localhost`. Ollama mode is local; it does not send match data to a hosted endpoint. The context-file workflow only shares data if you choose to paste it elsewhere. Stage 11 uses `AI-coach-narrative.md` when present and otherwise uses Stage 10's notes.

For local Ollama coaching, install and start Ollama, then download a model once:

```sh
ollama pull qwen2.5:7b
.venv/bin/python scripts/stage10a_ai_coach.py runs/match-1/report_data.json --mode ollama --model qwen2.5:7b
```

To create a context file instead:

```sh
.venv/bin/python scripts/stage10a_ai_coach.py runs/match-1/report_data.json
```

When a stage offers `--overwrite`, use it to replace existing outputs. Stage 10 rewrites its two report files. Re-run downstream stages after changing an upstream review file so scores, rally metrics, and coaching outputs stay consistent. Do not reuse a score-review JSON from a different match without replacing its match-specific serve rows.

## Package the application

Create a ZIP archive for another Mac with:

```sh
./compress_application.sh
```

The archive is written under `dist/` with a timestamped name. It contains the scripts, documentation, requirements, and any local ONNX/Pose model files under `models/`. It excludes `runs/`, `.venv`, Git metadata, and caches. Move and extract the ZIP on the other Mac, then follow the installation and model setup sections above; virtual environments are not portable between Macs. You can pass an alternate archive path as the script's only argument; relative paths are resolved from the project directory, for example `./compress_application.sh /tmp/badminton-analyser.zip`.
