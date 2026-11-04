# Command Pipeline

Run these commands from the repository root. Every input, output, review file, contact sheet, and rendered video goes in one match directory. The sample below uses `runs/match-1`; choose a new directory for each match.

```sh
RUN_DIR="runs/match-1"
mkdir -p "$RUN_DIR"
```

Put the original video at `$RUN_DIR/match.mp4`, or change that input path below. Put the YOLOX-S ONNX model at `models/yolox_s.onnx`. See [README.md](README.md) for system setup and model acquisition.

## Stage 0 — Proxy

Enter the original-video timestamp of the first serve at 0–0; use `0` if play starts immediately. Stage 0 writes the proxy and its `.proxy.json` sidecar beside it.

```sh
.venv/bin/python scripts/stage0_proxy.py \
  "$RUN_DIR/match.mp4" \
  "$RUN_DIR/match-720p.mp4" \
  --match-start-sec 123.4
```

On macOS, Stage 0 automatically uses FFmpeg's `h264_videotoolbox` encoder when available; otherwise it uses `libx264`. VideoToolbox accelerates encoding, while scaling and HDR tone mapping remain in the FFmpeg filter path. Use `--video-encoder libx264` to force software encoding, `--video-encoder h264_videotoolbox` to require VideoToolbox, or `--videotoolbox-bitrate 3M` to change its target bitrate. The default VideoToolbox bitrate is `2500k`. Add `--quiet-ffmpeg` to hide FFmpeg warnings and encoding progress while keeping errors visible; the script still prints its completion summary.

Use the trimmed proxy for every later stage. Their timestamps start at zero; add `first_serve_at_0_0_source_sec` from the sidecar to find a timestamp in the original recording. Add `--overwrite` when regenerating existing outputs.

## Stage 1 — Person detection

```sh
.venv/bin/python scripts/stage1_detect.py \
  "$RUN_DIR/match-720p.mp4" \
  models/yolox_s.onnx \
  "$RUN_DIR/dets.json" \
  "$RUN_DIR/detections_contact_sheet.jpg"
```

On Apple Silicon, the default `--execution-provider auto` uses ONNX Runtime's Core ML provider with CPU-and-GPU compute when available, falling back to CPU for unsupported graph operations. Use `--execution-provider cpu` for a CPU comparison, or `--execution-provider coreml` to require Core ML. The selected provider is printed and recorded in `dets.json`. Inspect the contact sheet after either run; player boxes should cover both players, and detections on adjacent courts can be expected.

## Stage 2 — Court calibration

```sh
.venv/bin/python scripts/stage2_calibrate.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/court_calibration.json" \
  "$RUN_DIR/court_overlay.png" \
  --time-sec 5
```

Click the named court landmarks in order. At least four points are required. Save, then check that the overlay follows the court lines. If a point is hidden, use `--skip LANDMARK` (repeatable), keeping at least four landmarks. Recalibrate each time the camera moves; keep calibrations and their time ranges distinct if the view changes during a match.

## Stage 3 — Player tracking

```sh
.venv/bin/python scripts/stage3_players.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/dets.json" \
  "$RUN_DIR/court_calibration.json" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/players_contact_sheet.jpg"
```

Stage 3 prompts for the garment (shirt or pants) and its color, then assigns R by that selected clothing cue on either court half and assigns O to the other player. You can instead provide `--identity-garment shirt` or `pants` and one supported `--identity-color`: black, white, gray, red, orange, yellow, green, cyan, blue, purple, pink, or brown. It opens the `players_contact_sheet.jpg` review image and asks you to confirm or swap the red R and blue O labels. Inspect every sampled tile for tracking jumps before proceeding; `--no-interactive-review` saves the image without asking, and `--swap-identities` applies a global swap.

## Stage 4 — Serve candidates

```sh
.venv/bin/python scripts/stage4_serves.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/serves.json" \
  "$RUN_DIR/serves_review" \
  --no-interactive-review
```

Inspect `$RUN_DIR/serves_review/serves_page_*.jpg` and `$RUN_DIR/serves.json` for missed or false candidates, server identity, and side. The position heuristic is the default. For a clear racket-direction correction, copy the template and fill each candidate's `R` and `O` values with `up`, `sideways`, `down`, or `unclear`:

```sh
cp "$RUN_DIR/serves_review/racket_orientation_review.template.json" "$RUN_DIR/racket_orientation_review.json"
```

Then rerun Stage 4 with that review file:

```sh
.venv/bin/python scripts/stage4_serves.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/serves.json" \
  "$RUN_DIR/serves_review" \
  --racket-review-json "$RUN_DIR/racket_orientation_review.json" \
  --no-interactive-review \
  --overwrite
```

Only a clear up-racket versus sideways/down-racket contrast overrides the position guess. Fill both `R` and `O` for a candidate where possible; a missing or `unclear` value cannot establish the contrast. Stage 4 reports how many rows were read and how many overrides applied. Review the final serve list; Stage 4 guesses can be wrong.

Stage 4 also creates `$RUN_DIR/manual_score_review.json` with one row for every accepted candidate. It fills server/side guesses and provisional `score_before` values using the next candidate's server as the prior rally winner. Review and correct these values, add any manual rows, and fill the final result before Stage 5. If a candidate is not a serve, rerun Stage 4 with interactive review and reject it with `x`; the manual score draft includes accepted candidates only. Rerunning Stage 4 with `--overwrite` regenerates this draft and replaces any edits.

## Stage 5 — Score review

```sh
.venv/bin/python scripts/stage5_score.py \
  "$RUN_DIR/serves.json" \
  "$RUN_DIR/points.json" \
  --manual-review "$RUN_DIR/manual_score_review.json"
```

Stage 4's draft includes every accepted candidate; Stage 5 requires each candidate exactly once, so a one-row review fails when Stage 4 found more than one candidate. Correct the draft's server (`R` or `O`), side (`left` or `right` from the server's perspective), and score before each serve. For a serve that happened in the video but Stage 4 missed, add a row with `candidate_n: null`, its nonzero proxy-relative `timestamp_sec`, and `dummy_point: false`. A score-only dummy point represents a point omitted from the players' score and absent from the video; use `candidate_n: null`, `timestamp_sec: 0`, and `dummy_point: true`. `n` is the chronological point number and must be renumbered sequentially after inserting or removing a row. `candidate_n` is the Stage 4 candidate ID: keep the existing IDs unchanged, and leave it `null` for manually added rows. Update the score-before values too. Fill the initial and final score and final-point winner. `templates/manual_score_review.json` remains an example showing two dummy points; it is not a replacement for the run-specific draft.

Minimal format example (replace the rows and final result with this match's complete serve list):

```json
{
  "score_order": "R-O",
  "initial_score": {"R": 0, "O": 0},
  "serves": [
    {"n": 1, "candidate_n": 1, "server": "R", "side": "right", "score_before": {"R": 0, "O": 0}},
    {"n": 2, "candidate_n": 2, "server": "O", "side": "left", "score_before": {"R": 0, "O": 1}}
  ],
  "final_point_winner": "R",
  "final_score": {"R": 1, "O": 1},
  "notes": []
}
```

If players forgot to record a point, include a score-only dummy row with `candidate_n: null`, `dummy_point: true`, and `timestamp_sec: 0`, along with the required server, side, and score-before values. This is an exceptional scoring correction, not a video rally; downstream video stages skip it.

## Stage 6 — Rally times

```sh
.venv/bin/python scripts/stage6_rallies.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/points.json" \
  "$RUN_DIR/rally_review" \
  "$RUN_DIR/rallies_review.json" \
  --estimate-rally-ends "$RUN_DIR/players.json"
```

Open `$RUN_DIR/rally_review/index.html`. The estimator uses movement to find the first sustained pause after active play and can infer a missing serve time between known serves. Review its estimates against the picture and edit `$RUN_DIR/rallies_review.json` where needed. Do not use racket sounds in a multi-court hall. Rally ends are approximate; a tail-end-of-rally time is sufficient for movement summaries.

## Stage 7 — Movement

```sh
.venv/bin/python scripts/stage7_movement.py \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/motion.json" \
  "$RUN_DIR/movement_heatmaps"
```

Check the heatmaps and whether speeds and distances look plausible. Stage 7 depends on Stage 3 tracking and Stage 6 times.

## Stage 8 — Pose

Download the MediaPipe Pose Landmarker heavy model once into the Git-ignored `models/` directory:

```sh
mkdir -p models
curl -fL \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
  -o models/pose_landmarker_heavy.task
```

Then run:

```sh
.venv/bin/python scripts/stage8_pose.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  models/pose_landmarker_heavy.task \
  "$RUN_DIR/pose_stats.json" \
  "$RUN_DIR/pose_review.jpg" \
  --racket-side auto
```

Check `$RUN_DIR/pose_review.jpg`: the skeleton should follow R, as selected in Stage 3, with shoulders, hips, ankles, and wrists aligned. Pass `--racket-side left` or `right` if the player's racket hand is known.

## Stage 9 — Pickup evidence

```sh
.venv/bin/python scripts/stage9_pickups.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/pickups.json" \
  "$RUN_DIR/pickup_review.jpg"
```

Inspect the bend candidates in `$RUN_DIR/pickup_review.jpg`. If a side is clear but wrong, edit that point's `landing_side` in `$RUN_DIR/pickups.json` to `R` (the analyzed player), `O` (the opponent), or `null` (unknown), and set `status` to `manual_confirmed`. Pickup side is indirect evidence; leave uncertain results unknown.

## Stage 10 — Report

```sh
.venv/bin/python scripts/stage10_report.py \
  "$RUN_DIR/points.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/motion.json" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/pose_stats.json" \
  "$RUN_DIR/pickups.json" \
  "$RUN_DIR/report_data.json" \
  "$RUN_DIR/coach_report.md" \
  --player-name "Arun" \
  --opponent-name "Noble"
```

Stage 10 is deterministic and uses local JSON files. It writes named score and movement summaries, timestamped loss examples, and heatmap analysis. Review the cited frames before treating patterns as causes.

## Stage 10A — Optional AI coach narrative

The default context mode writes `$RUN_DIR/AI-coach-context.md` beside `report_data.json`. Copy it into a chat agent if desired, then save the response as `$RUN_DIR/AI-coach-narrative.md` in the same run directory. Context mode makes no network request.

```sh
.venv/bin/python scripts/stage10a_ai_coach.py "$RUN_DIR/report_data.json"
```

To use Ollama locally instead, install Ollama and pull the model once:

```sh
ollama pull qwen2.5:7b
.venv/bin/python scripts/stage10a_ai_coach.py \
  "$RUN_DIR/report_data.json" \
  --mode ollama \
  --model qwen2.5:7b
```

Ollama mode writes the context and narrative beside `report_data.json`; it contacts only `localhost`. Add `--overwrite` when either output already exists.

## Stage 11 — Optional analysis video

```sh
.venv/bin/python scripts/stage11_video.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/pose_stats.json" \
  "$RUN_DIR/report_data.json" \
  "$RUN_DIR/match-analysis.mp4" \
  --pose-model models/pose_landmarker_heavy.task
```

Stage 11 reads `AI-coach-narrative.md` beside `report_data.json` by default, and uses Stage 10 notes if no narrative exists. It uses macOS `say` for local narration. On macOS, it automatically uses FFmpeg's `h264_videotoolbox` encoder when available; otherwise it uses `libx264`. Only video encoding is hardware accelerated; drawing tracking and pose overlays still uses the CPU. Add `--video-encoder libx264` to force software encoding, `--video-encoder h264_videotoolbox` to require VideoToolbox, or `--videotoolbox-bitrate 5M` to change its default `4M` target bitrate. Add `--short-mode` to omit the full-match segment, `--no-voice` to skip narration, `--no-pose` to skip pose inference, or `--overwrite` to replace the output video.

## Reuse for another match

Set `RUN_DIR` to a new folder, for example `runs/match-2`, copy the new source video there, and update its filename in the commands. Do not reuse match-specific JSON reviews. Re-run downstream stages whenever an upstream review changes. Generated run folders and model files are Git-ignored; the scripts and this workflow remain versionable.
