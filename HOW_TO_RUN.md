# How to Run the Badminton Analysis Pipeline

This guide takes one singles match video through Stage 11 and writes a rendered overlay video. Run commands from the repository root in one Terminal session so the environment and shell variables stay active. Use a new run directory for each match; Stage 0 will not overwrite an existing `job.json` or `proxy.mp4` there.

## 1. Prepare the Mac and select the video

Install Homebrew first if it is not already installed. Then install the local tools once and activate the project environment. The initializer detects Intel versus Apple Silicon, selects the matching native Homebrew installation, and creates an architecture-matched Python environment. `.venv` is machine-specific and ignored by Git; copy the source files, not an existing `.venv` directory.

```sh
bash scripts/init_macos.sh
source .venv/bin/activate
```

Set the full path to your video and a new folder name for this analysis. Keep the quotes, especially when a path contains spaces:

```sh
VIDEO="/Users/your-name/Movies/badminton-match.mov"
RUN_DIR="runs/match-01"
```

The pipeline currently supports singles. For usable calibration, all four court corners must be visible in each stable camera position. Side views are supported when those corners are visible. Moving-camera footage must include stable intervals; frames in gaps between calibrated intervals are skipped.

## 2. Run each stage in order

Run the following commands one at a time. The scripts use the default artifact filenames beside `job.json`, so the later commands need only `--job`.

```sh
# Stage 0: create the proxy and job settings.
python scripts/00_proxy.py --input "$VIDEO" --output-dir "$RUN_DIR"
MATCH_JOB="$RUN_DIR/job.json"

# Stage 1: detect people.
python scripts/01_detect.py --job "$MATCH_JOB"

# Stage 2: calibrate the court.
python scripts/02_calibrate.py --job "$MATCH_JOB"

# Stage 3: identify and track both players.
python scripts/03_track.py --job "$MATCH_JOB"

# Stage 4: find likely serve pauses.
python scripts/04_serve_candidates.py --job "$MATCH_JOB"

# Stage 5: review serves and replay the score.
python scripts/05_score.py --job "$MATCH_JOB"

# Stage 6: review the end time of each rally.
python scripts/06_rallies.py --job "$MATCH_JOB"

# Stage 7: calculate movement metrics and heatmaps.
python scripts/07_movement.py --job "$MATCH_JOB"

# Stage 8: estimate body position and save pose skeleton samples.
python scripts/08_pose.py --job "$MATCH_JOB"

# Stage 9: estimate pickup-side clues.
python scripts/09_pickups.py --job "$MATCH_JOB"

# Stage 10: create the local match report.
python scripts/10_report.py --job "$MATCH_JOB"

# Stage 11: render the overlay video with scoreboard labels.
python scripts/11_render_video.py --job "$MATCH_JOB" --you-name "Your Name" --opponent-name "Opponent"
```

Replace `Your Name` and `Opponent` with the labels you want shown in the video. Stage 11 writes `overlay.mp4` in the run directory at the proxy's 1280x720, 30 FPS settings, with source audio when present. If you render again, choose a new output filename with `--output`, for example `--output "$RUN_DIR/overlay-review-2.mp4"`.

Open the rendered video in the macOS default player with:

```sh
open "$RUN_DIR/overlay.mp4"
```

## 3. Answer the interactive review prompts

- **Stage 0:** Choose filming position (behind baseline or side) and camera movement. Confirm the detected source resolution and FPS by pressing Enter, or enter corrected values. Describe your clothing, such as `black shorts` or `red shirt`.
- **Stage 2:** For a fixed camera, the script uses the middle of the video. For a camera that moves or is repositioned, enter each stable interval's start and end time in seconds; press Enter at the interval prompt when finished. In each calibration window, click these corners in order: near-left baseline, near-right baseline, far-left baseline, far-right baseline. Press `r` to clear clicks. Inspect the saved court overlays before continuing. Use the same near baseline and left/right orientation for every interval.
- **Stage 3:** Accept or choose `shirt`/`shorts`, enter a reference timestamp (Enter accepts the suggested time), then click your player in the displayed frame. Review the annotated images under `review/tracking/` before trusting tracking-based results.
- **Stage 5:** Open each printed serve contact-sheet image. Enter the serve time or `x` to discard a false candidate. Select the server (`r` = you, `o` = opponent) and the server's court side. Add missed serves when asked. Enter the score before the first recorded point as `you-opponent` (for example `8-12`, or press Enter for `0-0`), mark new-game serves, and identify the winner of the final point. A missed serve can shift later score assignments, so review this carefully.
- **Stage 6:** Open each printed rally contact sheet and enter the timestamp in seconds of the last live-play frame before the pause or next serve.

Stages 1 and 8 download their free Ultralytics model weights the first time they run. Inference runs locally: Apple Silicon uses MPS when available and Intel Macs use CPU. Stages 7, 9, and 10 create analysis outputs used for the report; they can be skipped if you only want the overlay video. Stage 11 needs `players.json`, reviewed `points.json`, and (for the skeleton overlay) `pose_stats.json`.

## Outputs

All outputs are under `$RUN_DIR`: `proxy.mp4`, `job.json`, `detections.json`, `calibration.json`, `players.json`, `serve_candidates.json`, `score.json`, `points.json`, `motion.json`, `pose_stats.json`, `pickups.json`, `report.md`, and `overlay.mp4`. Review materials are saved under `calibration_previews/`, `review/`, and `heatmaps/`.
