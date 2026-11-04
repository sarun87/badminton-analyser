# How to Run the Badminton Analysis Pipeline

This guide takes one singles match video through Stage 11 and writes a rendered overlay video. Run commands from the repository root in one Terminal session so the environment and shell variables stay active. Use a new run directory for each match; Stage 0 will not overwrite an existing `job.json` or `proxy.mp4` there.

The commands below use the existing manual review by default. To opt into reduced review, use the assisted commands in section 2A instead of the Stage 5 and 6 manual commands. The assisted path assumes one game per video.

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

# Stage 11: render the default overlay and 30-second movement heatmap ending.
python scripts/11_render_video.py --job "$MATCH_JOB" --you-name "Your Name" --opponent-name "Opponent"
```

Replace `Your Name` and `Opponent` with the labels you want shown in the video; Stage 11 capitalizes them in the overlays. By default, the scorecard and court map are shown, and player foot rings/trails remain visible. The court map puts the camera-near player at the bottom. Skeleton and detail panels are off by default.

To render the optional skeleton and detail panel, pass `--show-skeleton --show-details`. To include available experimental shuttle marks, add `--show-shuttle`. For example, use a different output path when creating another version:

```sh
python scripts/11_render_video.py --job "$MATCH_JOB" --you-name "Your Name" --opponent-name "Opponent" --show-skeleton --show-details --show-shuttle --output "$RUN_DIR/overlay-full.mp4"
```

Stage 11 appends a 30-second full-match movement timelapse with player trails and accumulated court heat unless `--no-heatmap-clip` is passed. Change the ending length with `--heatmap-seconds 45`. The heatmap uses positions from the reviewed rally intervals. The source audio ends with the original match footage; the appended summary is silent. Output is 1280x720 at 30 FPS. If you render again, choose a new output filename.

To add the optional animated graphs after the heatmap, add `--show-stats` to Stage 11. This requires Stage 7's `motion.json` and adds two explained charts: point outcomes by rally duration and your average tracked movement on points won versus lost. The graph outro is 30 seconds by default; change it with `--stats-seconds 40`.

Use this command in place of the default Stage 11 command when rendering the graph outro:

```sh
python scripts/11_render_video.py --job "$MATCH_JOB" --you-name "Your Name" --opponent-name "Opponent" --show-stats
```

If you already created `overlay.mp4`, write a separate version with `--output "$RUN_DIR/overlay-stats.mp4"`; Stage 12 will use that latest render from `render_manifest.json`.

## 2B. Optional local coach voice-over

Run this after Stage 11. It uses the built-in macOS speech synthesizer, reads the score/movement summary and an available coaching note, and creates a narrated video copy. Speech starts over the heatmap outro, or over the graph outro when `--show-stats` was enabled.

```sh
python scripts/12_voiceover.py --job "$MATCH_JOB"
```

The command writes `coach_narration.txt`, `coach_voiceover.aiff`, and `overlay_with_voiceover.mp4`. To choose a system voice, list the installed options with `say -v '?'`, then pass one with `--voice`, for example `--voice Alex`. Adjust the speaking speed with `--rate 160`. If narration exceeds the ending clip, Stage 12 holds the final frame until the speech ends.

Open the rendered video in the macOS default player with:

```sh
open "$RUN_DIR/overlay.mp4"
```

### 2A. Optional reduced-review route

If you choose assisted review, run this block after Stage 4. Skip the manual Stage 5 and 6 commands in section 2, then continue with Stage 7.

```sh
# Optional, experimental shuttle path tracking. Uses local OpenCV; no additional weights or API.
python scripts/04a_shuttle_track.py --job "$MATCH_JOB"

# Associate early shuttle-like tracks with the player boxes at each serve candidate.
python scripts/04b_serve_analysis.py --job "$MATCH_JOB"

# Enter start score, final score, and first server; inspect only uncertain server suggestions.
python scripts/05_score.py --job "$MATCH_JOB" --review-mode assisted

# Estimate clear shared-player pauses; uncertain rally ends get manual contact sheets.
python scripts/06_rallies.py --job "$MATCH_JOB" --review-mode assisted
```

For the example 10-10 starting score and 21-18 final score, the tool expects `(21 + 18) - (10 + 10) = 19` recorded points. This verifies the total count only. A false serve candidate and a missed serve can cancel out, so matching totals do not prove the serve list is correct. If score replay does not reconcile, correct the uncertain server assignments or rerun Stage 5 in manual mode. The service-court side is calculated from the server's score (even = right, odd = left).

The shuttle pass analyzes every proxy frame and may take time on long videos. It tracks small bright moving regions, which can also be court lines, reflections, or compression artifacts. Its output can add visible path marks and possible direction changes to the overlay. It cannot reliably identify racket contact, clears, drops, smashes, or technique.

## 3. Answer the interactive review prompts

- **Stage 0:** Choose filming position (behind baseline or side) and camera movement. Confirm the detected source resolution and FPS by pressing Enter, or enter corrected values. Describe your clothing, such as `black shorts` or `red shirt`.
- **Stage 2:** For a fixed camera, the script uses the middle of the video. For a camera that moves or is repositioned, enter each stable interval's start and end time in seconds; press Enter at the interval prompt when finished. In each calibration window, click these corners in order: near-left baseline, near-right baseline, far-left baseline, far-right baseline. Press `r` to clear clicks. Inspect the saved court overlays before continuing. Use the same near baseline and left/right orientation for every interval.
- **Stage 3:** Accept or choose `shirt`/`shorts`, enter a reference timestamp (Enter accepts the suggested time), then click your player in the displayed frame. Review the annotated images under `review/tracking/` before trusting tracking-based results.
- **Stage 5, manual:** Open each serve contact sheet. Enter the serve time or `x` to discard a false candidate. Select server and service court, add missed serves, enter the starting score and first server, mark game breaks, and identify the last point's winner.
- **Stage 5, assisted:** Enter the starting and final score as `you-opponent` plus who served the first recorded point. For example, 10-10 to 21-18 expects 19 points in this clip. The candidate count must match or the script stops and directs you to manual review. High-confidence server suggestions are accepted unless `--confirm-high-confidence` is passed; uncertain cases show a contact sheet. The replay must finish at the entered final score.
- **Stage 6, manual:** Open each point contact sheet and enter the last live-play timestamp before the pause or next serve.
- **Stage 6, assisted:** High-confidence shared-player pauses are used as estimated rally ends. Uncertain points fall back to a contact sheet and manual timestamp. Inspect automatic estimates before trusting rally-length statistics.

Stages 1 and 8 download their free Ultralytics model weights the first time they run. Inference runs locally: Apple Silicon uses MPS when available and Intel Macs use CPU. Scripts 04a and 04b use OpenCV and existing player artifacts, without additional model downloads. Stage 11 needs `players.json` and `points.json`; add `motion.json` for `--show-stats`, `pose_stats.json` for `--show-skeleton`, and `shuttle_tracks.json` for `--show-shuttle`. Stage 12 uses only the built-in `say` command plus the project's FFmpeg installation; it does not call an online service.

## Outputs

All outputs are under `$RUN_DIR`: `proxy.mp4`, `job.json`, `detections.json`, `calibration.json`, `players.json`, `serve_candidates.json`, optional `shuttle_tracks.json` and `serving_analysis.json`, `score.json`, `points.json`, `motion.json`, `pose_stats.json`, `pickups.json`, `report.md`, `overlay.mp4`, `render_manifest.json`, and (when Stage 12 runs) `coach_narration.txt`, `coach_voiceover.aiff`, and `overlay_with_voiceover.mp4`. Review materials are saved under `calibration_previews/`, `review/`, and `heatmaps/`.
