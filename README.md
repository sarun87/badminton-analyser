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

Stages 0 and 11 check FFmpeg's encoder list and probe a short VideoToolbox encode session before selecting it on macOS. If the encoder is listed but cannot open a session, `auto` falls back to `libx264`. VideoToolbox accelerates H.264 encoding. Stage 0 scaling/HDR tone mapping and Stage 11 frame overlays still run through their existing CPU paths. Use `--video-encoder libx264` to force software encoding or `--video-encoder h264_videotoolbox` to require VideoToolbox. VideoToolbox uses a target bitrate (`2500k` for Stage 0, `4M` for Stage 11 by default), while the `libx264` path keeps its CRF quality setting. Both stages accept `--videotoolbox-bitrate` to adjust the hardware encoder bitrate.

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

### Optional TrackNet shuttle tracking

TrackNet is an optional extra for shuttle trails, serve-origin clues, rally-end evidence, and tentative resting-shuttle locations. It is not required by the standard pipeline. Use a separate environment because TrackNet needs PyTorch; do not install these packages in the analyser's `.venv`.

Either clone TrackNetV3 or use the source bundled by `compress_application.sh`:

```sh
export TRACKNET_DIR="$HOME/TrackNetV3"
# For bundled source, set TRACKNET_DIR to the extracted TrackNetV3 folder above.
if [ ! -f "$TRACKNET_DIR/predict.py" ]; then git clone https://github.com/qaz812345/TrackNetV3.git "$TRACKNET_DIR"; fi
export TRACKNET_ENV="$HOME/.venvs/tracknetv3"
python3.11 -m venv "$TRACKNET_ENV"
"$TRACKNET_ENV/bin/python" -m pip install --upgrade pip
"$TRACKNET_ENV/bin/python" -m pip install -r requirements-tracknet.txt
"$TRACKNET_ENV/bin/python" -c 'import torch; mps=getattr(torch.backends, "mps", None); print("PyTorch:", torch.__version__, "MPS available:", bool(mps and mps.is_available()))'
```

Download `TrackNet_best.pt` from the [TrackNetV3 checkpoint archive on Google Drive](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing). Save it outside the repository, for example as `$HOME/TrackNetV3-models/TrackNet_best.pt`. TrackNet's source is MIT licensed; the separate model checkpoint is also distributed under the project's stated MIT terms. On Apple Silicon, the included TrackNet version selects PyTorch MPS when available; Intel Macs use CPU.

Run optional Stage 3A on the Stage 0 proxy after Stage 2 calibration. It writes normalized `shuttle_tracks.json`, the raw CSV, and a preview video into the match run directory. The video passed to Stage 3A can be a short clip for a quick pilot; full-match tracking takes longer. Downstream support is enabled with `--shuttle-tracks`: Stage 4 can apply a confident server/receiver cue after the center-line pass, Stage 6 can refine a movement-based rally end when a sustained main-court track is close to the pause, Stage 7 can estimate shot counts, Stage 9 adds a tentative landing candidate, and Stage 11 draws a filtered trail. Stage 10 includes the Stage 6, 7, and 9 evidence automatically. Review all cues; adjacent-court shuttle detections can still occur, and the exported CSV has only one selected point per frame.

Stage 11 uses macOS `say` for optional local narration and inserts a 600 ms pause between sentences by default; adjust it with `--sentence-pause-ms` (use `0` to disable). If the default voice is unavailable, list installed voices with `say -v '?'`, choose one with `--voice`, or pass `--no-voice`. Stage 11 also needs FFmpeg. For AI coaching, Ollama is optional: install the [Ollama macOS app](https://ollama.com/download/mac), then use the local setup described below.

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
| 3A — Shuttle tracking (optional) | Raw TrackNet tracks, CSV, and preview | Run only with a separate TrackNet environment and checkpoint. Inspect the preview and the main-court tags; adjacent-court activity and missed shuttle frames remain possible. |
| 4 — Serves | `serves.json`, review sheets, and racket-orientation template | Candidate generation and merging use the original stationary-window and net-distance flow. After candidates are fixed, the center-line pass runs, then an optional confident TrackNet cue can refine server/receiver labels. Review candidates in the local browser UI and correct false candidates, server/court, or missed serves. Tune movement limits with `--still-threshold` or separate near/far options. |
| 5 — Score | `points.json` | Stage 4 creates a full score-review draft. Use the browser UI to correct it, add missed video serves or score-only dummy points, and enter the final result. Stage 5 accepts untimed manual serves, but Stage 6 needs their timestamps to place rallies in the video. Stage 5 requires every accepted candidate exactly once. |
| 6 — Rally ends | `rallies_review.json` and picture-only contact sheets | The `--estimate-rally-ends players.json` option finds the first sustained pause after active movement. Optional `--rally-end-source shuttle` uses a sustained main-court TrackNet cluster near that pause when available; otherwise it keeps the movement estimate. Check estimated starts and ends against the frames. |
| 7 — Movement and shot estimates | `motion.json` and heatmap images | Runs from Stage 3 tracks and Stage 6 times. Includes beginning-to-tail player speed comparisons. TrackNet shot counts can use the original floor-line estimate or an optional clicked net-plane corridor; review candidate timestamps before relying on them. |
| 8 — Pose | `pose_stats.json` and `pose_review.jpg` for selected R or O | Confirm the skeleton follows the selected player's body. O uses a two-pose candidate check against O's tracked box to reduce identity swaps during occlusion. Measurements are estimates, not proof of why a point was won or lost. |
| 9 — Pickups | `pickups.json` and `pickup_review.jpg` | Inspect candidate bends. For a clear correction, set that point's `landing_side` to `R` (the Stage 3 selected identity), `O` (the other tracked identity), or `null` (unknown), and set `status` to `manual_confirmed`. Pickup side is indirect evidence. |
| 10 — Report | `report_data.json` and `coach_report.md` for selected R or O | Choose `--player-code R` or `O`; enter both names with `--r-name` and `--o-name`. Includes longest-rally, available shot-count, and beginning-to-tail speed summaries. Review these estimates against video. |
| 10A — AI coach (optional) | Subject-specific context and optional narrative | Use Ollama locally or paste the context into a chat model and save its response. O files use `-O` suffixes, for example `AI-coach-context-O.md`. |
| 11 — Video (optional) | Annotated full-match or short-analysis MP4 | Opens with a 2.5-second title card with player names, final score, winner trophy, and “AI Match Analysis,” then score and heatmap analysis. A 2.5-second “Coaching” card introduces the still-and-replay reviews. Full mode adds a “Full Match” card before match footage; short mode omits that card and footage. TrackNet mode draws the shuttle trail and shot counter; `--shot-count-only` shows the counter without the trail. Unavailable counts display as a dash, not zero. Narration has a 600 ms sentence pause by default. |

TrackNet remains optional at every downstream stage. Pass the Stage 3A JSON generated from the same full proxy to Stages 4, 6, 7, 9, and 11 with `--shuttle-tracks`; applicable stages check video alignment. Stage 4's `--use-shuttle-server-cue` applies qualifying launch-origin hints after the center-line pass (default confidence threshold `0.6`). Stage 6's `--rally-end-source shuttle` lets a sustained nearby shuttle cluster refine the movement pause; points without a suitable cluster retain their movement estimate. Stage 7 can use a clicked, extended image-space net-plane corridor to propose shuttle crossing timestamps without treating airborne positions as floor points; it does not estimate landing positions, and candidates need video review. See [command_pipeline.md](command_pipeline.md) for the calibration command. Set a reviewed count at `points[].shuttle_shot_analysis.manual_shot_count` in `motion.json`. Stage 11 shows the shot counter with the shuttle trail when `--shuttle-tracks` is supplied; use `--shot-count-only` to show only the count from the Stage 10 report, or `--no-shuttle-overlay` to hide the trail while retaining the counter. Stage 10 reports shot-count summaries and player speed near the start and tail of longer rallies. Counts can miss detections or include adjacent-court tracks, and a tail-speed decrease does not prove fatigue. Review the video before drawing conclusions.

## Analyze either player

Stages 0–7 keep the tracking labels R and O and produce shared score, court-position, movement, and pickup data. R is the player selected in Stage 3; O is the other player. Stages 8–11 let you choose which identity receives the pose and coaching analysis. R remains the default. To analyze O, run Stage 8 with `--player-code O`, then run Stage 10 with the same flag and use separate output names such as `pose_stats_O.json`, `report_data_O.json`, and `coach_report_O.md`. Stage 10A and Stage 11 must use those O-specific files; Stage 10A creates `AI-coach-context-O.md` and `AI-coach-narrative-O.md` by default. The scripts reject mismatched pose, report, or narrative subjects instead of silently mixing them. See [command_pipeline.md](command_pipeline.md) for the complete O command sequence. To analyze both players, run Stages 8–11 once per player with separate outputs.

The manual work is concentrated in selecting the Stage 0 start time, clicking and checking Stage 2 calibration, verifying Stage 5's score, and correcting visibly wrong Stage 6 estimates. Other stages run automatically but still need the checks above. You do not need to use an assistant or cloud service to run the pipeline.

Stage 4's review UI is a local Python HTTP server using the standard library and OpenCV already installed for the pipeline. It binds only to `127.0.0.1`; no additional package or internet access is needed.

## AI coaching and privacy

Stage 10 is deterministic: it summarizes the local JSON measurements and writes timestamped review notes. It does not call an LLM. Stage 10A can create a self-contained `AI-coach-context.md` file for R (or `AI-coach-context-O.md` for O), or send the evidence to Ollama at `localhost`. Ollama mode is local; it does not send match data to a hosted endpoint. The context-file workflow only shares data if you choose to paste it elsewhere. Stage 11 uses the matching `AI-coach-narrative.md` or `AI-coach-narrative-O.md` when present and otherwise uses Stage 10's notes.

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

The archive is written under `dist/` with a timestamped name. It contains the scripts, documentation, requirements, and any local ONNX/Pose model files under `models/`. It excludes `runs/`, `.venv`, Git metadata, and caches. Move and extract the ZIP on the other Mac, then follow the installation and model setup sections above; virtual environments are not portable between Macs. Pass an alternate archive path as the first argument, for example `./compress_application.sh /tmp/badminton-analyser.zip`. To bundle TrackNetV3 source, add `--tracknet-path /path/to/TrackNetV3`; source is included without Git history, training data, or virtual environments. If the checkpoint is stored outside that checkout, add `--tracknet-checkpoint /path/to/TrackNet_best.pt`. A checkpoint found inside the checkout is included automatically when the explicit option is omitted.

```sh
./compress_application.sh /tmp/badminton-analyser.zip \
  --tracknet-path "$HOME/TrackNetV3" \
  --tracknet-checkpoint "$HOME/TrackNetV3-models/TrackNet_best.pt"
```
