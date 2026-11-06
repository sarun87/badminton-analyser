# Command Pipeline

Run these commands from the repository root. Every input, output, review file, contact sheet, and rendered video goes in one match directory. The sample below uses `runs/match-1`; choose a new directory for each match.

Each stage's option table follows its command; required positional file arguments are shown in the command itself. Add `--help` to any stage script for its full current usage.

```sh
RUN_DIR="runs/match-1"
mkdir -p "$RUN_DIR"
```

Put the original video at `$RUN_DIR/match.mp4`, or change that input path below. Put the YOLOX-S ONNX model at `models/yolox_s.onnx`. See [README.md](README.md) for system setup and model acquisition.

TrackNet is optional and uses a separate environment; do not install PyTorch into the analyser's `.venv`. Set these variables from the repository root. Change the paths if you keep TrackNet elsewhere:

```sh
export TRACKNET_DIR="$HOME/TrackNetV3"                                      # folder containing predict.py
export TRACKNET_ENV="$HOME/.venvs/tracknetv3"                               # separate Python venv directory
export TRACKNET_CHECKPOINT="$HOME/TrackNetV3-models/TrackNet_best.pt"       # full path to the .pt checkpoint file
```

For a new setup, clone TrackNetV3 if needed, create its environment, and install the separate dependencies:

```sh
if [ ! -f "$TRACKNET_DIR/predict.py" ]; then
  git clone https://github.com/qaz812345/TrackNetV3.git "$TRACKNET_DIR"
fi
python3.11 -m venv "$TRACKNET_ENV"
"$TRACKNET_ENV/bin/python" -m pip install --upgrade pip
"$TRACKNET_ENV/bin/python" -m pip install -r requirements-tracknet.txt
```

| Option | Purpose and when to use it |
|---|---|
| `--upgrade` | Upgrade pip in the separate TrackNet environment before installing its requirements. |

Download the checkpoint archive from the [link in README.md](README.md#optional-tracknet-shuttle-tracking), extract `TrackNet_best.pt`, and save that file at the path in `TRACKNET_CHECKPOINT`. This variable names the checkpoint file itself, not its containing folder. If using a bundled TrackNet checkout with the model included, for example, set it to `"$TRACKNET_DIR/models/TrackNet_best.pt"`.

## Stage 0 — Proxy

Enter the original-video timestamp of the first serve at 0–0; use `0` if play starts immediately. Stage 0 writes the proxy and its `.proxy.json` sidecar beside it.

```sh
.venv/bin/python scripts/stage0_proxy.py \
  "$RUN_DIR/match.mp4" \
  "$RUN_DIR/match-720p.mp4" \
  --match-start-sec 123.4
```

| Option | Purpose and when to use it |
|---|---|
| `--match-start-sec SEC` | Required. Set the original-video time of the first 0–0 serve. |
| `--video-encoder` | Choose automatic encoding, or set it to `libx264` or `h264_videotoolbox` to force an encoder. |
| `--videotoolbox-bitrate RATE` | Set VideoToolbox output bitrate when using the hardware encoder. |
| `--quiet-ffmpeg` | Hide FFmpeg warnings and progress when the output is noisy. |
| `--overwrite` | Replace an existing proxy and sidecar. |

On macOS, Stage 0 probes a short FFmpeg `h264_videotoolbox` encode before using it; if FFmpeg lists the encoder but cannot open a session, `auto` falls back to `libx264`. VideoToolbox accelerates encoding, while scaling and HDR tone mapping remain in the FFmpeg filter path. Use `--video-encoder libx264` to force software encoding, `--video-encoder h264_videotoolbox` to require VideoToolbox, or `--videotoolbox-bitrate 3M` to change its target bitrate. The default VideoToolbox bitrate is `2500k`. Add `--quiet-ffmpeg` to hide FFmpeg warnings and encoding progress while keeping errors visible; the script still prints its completion summary.

Use the trimmed proxy for every later stage. Their timestamps start at zero; add `first_serve_at_0_0_source_sec` from the sidecar to find a timestamp in the original recording. Add `--overwrite` when regenerating existing outputs.

## Stage 1 — Person detection

```sh
.venv/bin/python scripts/stage1_detect.py \
  "$RUN_DIR/match-720p.mp4" \
  models/yolox_s.onnx \
  "$RUN_DIR/dets.json" \
  "$RUN_DIR/detections_contact_sheet.jpg"
```

| Option | Purpose and when to use it |
|---|---|
| `--every N` | Detect every Nth frame; use a smaller value for denser detections. Default: 3. |
| `--confidence VALUE` | Change the minimum person-detection confidence. Default: 0.3. |
| `--nms-iou VALUE` | Change non-maximum suppression overlap threshold. Default: 0.5. |
| `--threads N` | Set ONNX Runtime CPU thread count when tuning CPU performance. |
| `--execution-provider` | Select `auto`, `coreml`, or `cpu` for ONNX Runtime inference. |
| `--overwrite` | Replace existing detection JSON and contact sheet. |

On Apple Silicon, the default `--execution-provider auto` uses ONNX Runtime's Core ML provider with CPU-and-GPU compute when available, falling back to CPU for unsupported graph operations. Use `--execution-provider cpu` for a CPU comparison, or `--execution-provider coreml` to require Core ML. The selected provider is printed and recorded in `dets.json`. Inspect the contact sheet after either run; player boxes should cover both players, and detections on adjacent courts can be expected.

## Stage 2 — Court calibration

```sh
.venv/bin/python scripts/stage2_calibrate.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/court_calibration.json" \
  "$RUN_DIR/court_overlay.png" \
  --time-sec 5
```

| Option | Purpose and when to use it |
|---|---|
| `--time-sec SEC` | Choose the frame used for clicking court points. Use a clear frame with visible lines. Default: 5. |
| `--skip LANDMARK` | Omit a hidden landmark; repeat as needed while keeping at least four points. |
| `--overwrite` | Replace existing calibration JSON and overlay. |

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

| Option | Purpose and when to use it |
|---|---|
| `--identity-garment` | Specify `pant`, `pants`, or `shirt` as the clothing item identifying player R. |
| `--identity-color COLOR` | Specify R's garment color; use with `--identity-garment`. |
| `--x-margin-m METRES` | Change how far beyond the doubles sidelines to retain player detections. |
| `--swap-identities` | Swap R and O if the contact sheet shows the labels are reversed. |
| `--no-interactive-review` | Save the contact sheet without waiting for confirmation; review it yourself before proceeding. |
| `--overwrite` | Replace existing player tracks and contact sheet. |

Stage 3 prompts for the garment (shirt or pants) and its color, then assigns R by that selected clothing cue on either court half and assigns O to the other player. You can instead provide `--identity-garment shirt` or `pants` and one supported `--identity-color`: black, white, gray, red, orange, yellow, green, cyan, blue, purple, pink, or brown. It opens the `players_contact_sheet.jpg` review image and asks you to confirm or swap the red R and blue O labels. Inspect every sampled tile for tracking jumps before proceeding; `--no-interactive-review` saves the image without asking, and `--swap-identities` applies a global swap.

## Stage 3A — Optional TrackNet shuttle tracking

Run after Stage 2 calibration. This processes the video supplied to it; use a short clip first if you want a quick pilot. It writes `shuttle_tracks.json`, a raw CSV, and an annotated raw-track preview under the run directory. InpaintNet is disabled.

```sh
.venv/bin/python scripts/stage3a_tracknet.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/shuttle_tracks.json" \
  --tracknet-repo "$TRACKNET_DIR" \
  --tracknet-python "$TRACKNET_ENV/bin/python" \
  --checkpoint "$TRACKNET_CHECKPOINT" \
  --calibration-json "$RUN_DIR/court_calibration.json"
```

| Option | Purpose and when to use it |
|---|---|
| `--tracknet-repo PATH` | Required path to the TrackNetV3 checkout containing `predict.py`. |
| `--tracknet-python PATH` | Required Python executable from TrackNet's separate virtual environment. |
| `--checkpoint PATH` | Required path to the `TrackNet_best.pt` model file. |
| `--calibration-json PATH` | Supply Stage 2 calibration to tag likely main-court points; needed for downstream shot estimates. |
| `--preview-video PATH` | Choose a non-default path for the annotated preview. |
| `--no-preview-video` | Skip preview rendering to save time or disk space. |
| `--batch-size N` | Change inference batch size to tune memory use and speed. Default: 8. |
| `--court-margin-px PIXELS` | Widen or narrow the soft court-region margin for airborne detections. |
| `--overwrite` | Replace the tracks JSON, CSV, and preview. |

Review `$RUN_DIR/shuttle_tracking_preview.mp4` and `shuttle_tracks.json`. Court tags are only a soft image-space filter; the model exports one selected shuttle point per frame, so adjacent-court switches and missing detections can remain. Do not infer shuttle court metres while it is airborne.

## Stage 4 — Serve candidates

```sh
.venv/bin/python scripts/stage4_serves.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/serves.json" \
  "$RUN_DIR/serves_review" \
  --no-interactive-review
```

| Option | Purpose and when to use it |
|---|---|
| `--min-duration SEC` | Change the minimum still interval required for a serve candidate. Default: 0.8. |
| `--merge-gap SEC` | Change how close same-server, same-side candidates must be to merge. Default: 6.0. |
| `--still-threshold M/S` | Set one still-speed limit for both court halves. |
| `--near-still-threshold M/S`, `--far-still-threshold M/S` | Set separate still-speed limits when the two sides track differently. Do not combine with `--still-threshold`. |
| `--racket-review-json PATH` | Apply reviewed racket orientations from the template JSON. |
| `--shuttle-tracks PATH` | Attach TrackNet launch-origin evidence for review or optional server refinement. |
| `--use-shuttle-server-cue` | Apply the TrackNet cue after the center-line pass; requires `--shuttle-tracks`. |
| `--shuttle-server-min-confidence VALUE` | Raise or lower the heuristic score needed to apply the cue. Default: 0.6. |
| `--interactive-review`, `--no-interactive-review` | Prompt to correct candidates, or skip prompts and review the saved files later. |
| `--overwrite` | Replace Stage 4 outputs and regenerate the manual score draft. |

The stationary-candidate bounds are near `x=0.2–5.9 m`, `y=1.0–4.9 m` and far `x=0.2–5.9 m`, `y=8.5–12.4 m`; positions are smoothed over 0.5 s. The near-side player must move below 0.7 m/s, the far-side player below 0.9 m/s, and the window must last 0.8 s by default. Inspect `$RUN_DIR/serves_review/serves_page_*.jpg` and `$RUN_DIR/serves.json` for missed or false candidates, server identity, and side.

Candidate detection and merging retain the original stationary-window and net-distance flow. After that list is fixed, a separate center-line pass compares each player's distance from `x=3.05 m` across each merged candidate's stable windows. A player must be at least 0.25 m closer to the line to vote; if at least 60% of decisive votes agree, that pass refines the server label and the other player is treated as receiver. If its vote is inconclusive, the original net-distance guess is retained. The center-line pass does not add, reject, or merge candidates. Service side is still estimated from the selected server's position, accounting for the far player facing the camera. These are reviewable guesses. For a clear racket-direction correction, copy the template and fill each candidate's `R` and `O` values with `up`, `sideways`, `down`, or `unclear`:

When Stage 3A is enabled, append `--shuttle-tracks "$RUN_DIR/shuttle_tracks.json"` to retain the image-space launch-origin cue for review. To let a sufficiently strong cue refine the server and receiver after the center-line pass, also add `--use-shuttle-server-cue`. The default heuristic score threshold is 0.6; override it with `--shuttle-server-min-confidence 0.7`. This score measures spatial separation from the player boxes; it is not a calibrated probability. The cue does not change candidate detection, and the racket review or browser UI can still correct the result.

The default still-speed limits are 0.7 m/s near and 0.9 m/s far. Set one limit for both halves with `--still-threshold 0.8`, or override them separately with `--near-still-threshold 0.65 --far-still-threshold 0.85`. The selected values are recorded in `serves.json` as `still_speed_threshold_m_s`. The shared and separate options cannot be used together. `--min-duration` controls how long both players must remain within the candidate bounds; it is separate from the movement-speed limits.

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

| Option | Purpose and when to use it |
|---|---|
| `--racket-review-json PATH` | Read the edited racket-orientation review and refine server labels. |
| `--no-interactive-review` | Keep this rerun non-interactive; review the generated candidate files or UI separately. |
| `--overwrite` | Replace the existing Stage 4 JSON and review outputs. |

Only a clear up-racket versus sideways/down-racket contrast overrides the position guess. Fill both `R` and `O` for a candidate where possible; a missing or `unclear` value cannot establish the contrast. Stage 4 reports how many rows were read and how many overrides applied. Review the final serve list; Stage 4 guesses can be wrong.

Stage 4 also creates `$RUN_DIR/manual_score_review.json` with one row for every accepted candidate. It fills server/side guesses and provisional `score_before` values using the next candidate's server as the prior rally winner. Review and correct these values, add any manual rows, and fill the final result before Stage 5. If a candidate is not a serve, the local UI below can reject it; the manual score draft includes accepted candidates only. Rerunning Stage 4 with `--overwrite` regenerates this draft, removes any prior `serves_reviewed.json`, and replaces any edits. Run the UI again afterward.

For visual review, launch the local browser UI after Stage 4:

```sh
.venv/bin/python scripts/stage4_review_ui.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/serves.json" \
  "$RUN_DIR/serves_review"
```

| Option | Purpose and when to use it |
|---|---|
| `--manual-review-json PATH` | Use a score-review JSON at a different path. Defaults beside `serves.json`. |
| `--reviewed-serves-json PATH` | Save corrected serves to a different path. |
| `--port PORT` | Use another localhost port if 8765 is already occupied. |
| `--no-browser` | Print the local URL without opening a browser automatically. |

The UI runs on `127.0.0.1` and opens in your browser. Each candidate shows its annotated overview plus larger, unannotated R/O player crops from the same review moments for racket inspection. Accept/reject it, correct server and court, and optionally use the racket-orientation cue. Add missed video serves by proxy timestamp or score-only dummy points, then save. Existing manual serve rows without a known timestamp remain in the timeline; enter a proxy time in that row when available. You can save an untimed serve for Stage 5, but Stage 6 needs its timestamp to place the rally in the video. The UI writes `serves_reviewed.json` and updates `manual_score_review.json`; changes are stored only when you press **Save review**. Keep the terminal open while using the UI, then stop it with Ctrl+C. Use the reviewed serves file in Stage 5.

## Stage 5 — Score review

```sh
.venv/bin/python scripts/stage5_score.py \
  "$RUN_DIR/serves_reviewed.json" \
  "$RUN_DIR/points.json" \
  --manual-review "$RUN_DIR/manual_score_review.json"
```

| Option | Purpose and when to use it |
|---|---|
| `--manual-review PATH` | Apply the Stage 4 score draft, including corrected or added serves and dummy points. |
| `--overwrite` | Replace an existing `points.json`. |

Stage 4's draft includes every accepted candidate; Stage 5 requires each candidate exactly once, so a one-row review fails when Stage 4 found more than one candidate. Correct the draft's server (`R` or `O`), side (`left` or `right` from the server's perspective), and score before each serve. For a serve that happened in the video but Stage 4 missed, add a row with `candidate_n: null`, its nonzero proxy-relative `timestamp_sec`, and `dummy_point: false`. A score-only dummy point represents a point omitted from the players' score; use `candidate_n: null`, `timestamp_sec: 0`, and `dummy_point: true`. It is excluded from video-time analysis even if the rally appears in the footage. `n` is the chronological point number and must be renumbered sequentially after inserting or removing a row. `candidate_n` is the Stage 4 candidate ID: keep the existing IDs unchanged, and leave it `null` for manually added rows. Update the score-before values too. Fill the initial and final score and final-point winner. `templates/manual_score_review.json` remains an example showing two dummy points; it is not a replacement for the run-specific draft.

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

| Option | Purpose and when to use it |
|---|---|
| `--interval-sec SEC` | Change spacing between contact-sheet frames. Default: 0.5. |
| `--estimate-rally-ends PLAYERS_JSON` | Estimate missing serve times and rally ends from Stage 3 player movement. |
| `--pause-duration-sec SEC` | Change how long players must remain slow for a pause estimate. Default: 1.5. |
| `--pause-speed-m-s M/S` | Change the maximum player speed considered a pause. Default: 0.8. |
| `--active-speed-m-s M/S` | Change the speed threshold used to identify active play. Default: 1.0. |
| `--shuttle-tracks PATH` | Attach TrackNet coverage and rally-end comparison evidence. |
| `--rally-end-source` | Choose `players` (default) or `shuttle` to refine a pause with calibrated TrackNet clusters. |
| `--shuttle-rally-window-sec SEC` | Change how far before the player pause to search for a shuttle cluster. Default: 1.5. |
| `--shuttle-rally-after-sec SEC` | Change the allowed time after the pause for a shuttle observation. Default: 0.35. |
| `--overwrite` | Replace rally JSON and contact sheets. |

Open `$RUN_DIR/rally_review/index.html`. The estimator uses movement to find the first sustained pause after active play and can infer a missing serve time between known serves. Review its estimates against the picture and edit `$RUN_DIR/rallies_review.json` where needed. Do not use racket sounds in a multi-court hall. Rally ends are approximate; a tail-end-of-rally time is sufficient for movement summaries.

If TrackNet is enabled, append `--shuttle-tracks "$RUN_DIR/shuttle_tracks.json"` to attach raw-coverage and last-detection comparisons. In the default `players` source mode, these remain evidence and do not change the rally-end timestamp.

To let TrackNet refine rally ends, use the same option plus `--rally-end-source shuttle`:

```sh
.venv/bin/python scripts/stage6_rallies.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/points.json" \
  "$RUN_DIR/rally_review" \
  "$RUN_DIR/rallies_review.json" \
  --estimate-rally-ends "$RUN_DIR/players.json" \
  --shuttle-tracks "$RUN_DIR/shuttle_tracks.json" \
  --rally-end-source shuttle
```

| Option | Purpose and when to use it |
|---|---|
| `--estimate-rally-ends PLAYERS_JSON` | Provide movement-based rally timing for TrackNet to refine. Required for shuttle mode. |
| `--shuttle-tracks PATH` | Supply calibrated TrackNet observations. |
| `--rally-end-source shuttle` | Refine rally-end estimates using sustained nearby shuttle clusters. |
| `--overwrite` | Replace existing Stage 6 review files. |

This requires Stage 3A tracks made with Stage 2 calibration. A sustained main-court cluster must occur within 1.5 seconds before to 0.35 seconds after the player-pause estimate and before the next serve. The nearest point in a cluster of at least three observations refines the end time. If the cluster is absent, the movement estimate stays in place. Check each result against its contact sheet; TrackNet can miss the shuttle or select adjacent-court activity.

## Stage 7 — Movement and optional shot estimates

```sh
.venv/bin/python scripts/stage7_movement.py \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/motion.json" \
  "$RUN_DIR/movement_heatmaps"
```

| Option | Purpose and when to use it |
|---|---|
| `--shuttle-tracks PATH` | Add per-rally shot-count estimates and crossing timestamps using calibrated TrackNet data. |
| `--overwrite` | Replace `motion.json` and regenerate heatmaps. |

Stage 7 also compares each player's tracked speed near the beginning and tail of each rally (using windows up to 2 seconds). Rallies that are too short or have gaps in player tracks are omitted. A speed drop is descriptive, not proof of fatigue.

For optional TrackNet shot estimates, add `--shuttle-tracks` to the Stage 7 command:

```sh
.venv/bin/python scripts/stage7_movement.py \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/motion.json" \
  "$RUN_DIR/movement_heatmaps" \
  --shuttle-tracks "$RUN_DIR/shuttle_tracks.json" \
  --overwrite
```

| Option | Purpose and when to use it |
|---|---|
| `--shuttle-tracks PATH` | Estimate shuttle crossings per rally; use tracks from the same proxy with Stage 2 calibration. |
| `--net-plane-json PATH` | Use candidate passes through an extended, clicked image-space net-plane corridor instead of the floor-line estimator. Requires `--shuttle-tracks`. |
| `--overwrite` | Replace the existing movement metrics and heatmaps. |

To pilot the image-space net-plane estimator, first calibrate the vertical corridor in a clear frame from the same proxy:

```sh
.venv/bin/python scripts/stage7_net_plane_calibrate.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/court_calibration.json" \
  "$RUN_DIR/net_plane_calibration.json" \
  --frame-time 8
```

Click only the four physical net corners: floor at the left/right posts, then the white tape at those same posts. Press `s` to save. Do not click the wall, roof, or edge of the video frame. Review `net_plane_calibration.preview.jpg`: the green corridor is extrapolated above the tape to 4.5 m and can reach the top of the image; it should stay aligned with the selected net, not spread across adjacent courts. Use another clear `--frame-time` if a player hides a corner. Then add `--net-plane-json "$RUN_DIR/net_plane_calibration.json"` to Stage 7. The net-plane mode records candidate image-space passes; it does not reconstruct a ground position and may include projection overlap or adjacent-court activity. Short directional passes without full before/after observations are low-confidence candidates. If there is no countable candidate, Stage 7 stores `null` (unavailable), not zero. Review timestamps in the video before using estimates in Stage 10/11. Without `--net-plane-json`, Stage 7 retains the original floor-line estimator. If you visually review a rally and know its count, set `points[].shuttle_shot_analysis.manual_shot_count` in `motion.json`; Stage 10 uses that value in preference to the estimate. Re-run Stage 7 before making manual corrections, since rerunning replaces `motion.json`.

Check the heatmaps and whether speeds and distances look plausible. Stage 7 depends on Stage 3 tracking and Stage 6 times. Stage 10 includes the longest rally, shot-count estimates when available, and beginning-to-tail speed comparisons for rallies over 5 seconds. These speed comparisons are not a fatigue diagnosis.

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

| Option | Purpose and when to use it |
|---|---|
| `--player-code R`, `--player-code O` | Select which tracked player receives pose analysis. Default: R. |
| `--racket-side` | Choose `auto`, `left`, or `right` for the racket-holding side. |
| `--crop-scale VALUE` | Enlarge or reduce the crop around the player. |
| `--max-rally-crop-frames N` | Limit pose frames for a quick preview; 0 processes all selected frames. |
| `--overwrite` | Replace pose JSON and review image. |

By default this analyzes R, the player selected in Stage 3. Check `$RUN_DIR/pose_review.jpg`: the skeleton should follow the selected player's shoulders, hips, ankles, and wrists. For O, add `--player-code O`; Stage 8 considers up to two pose candidates and keeps the one whose torso aligns with O's Stage 3 box, which helps reject a nearby person during occlusion. O pose frames that cannot be matched confidently are skipped. Pass `--racket-side left` or `right` if the selected player's racket hand is known.

## Stage 9 — Pickup evidence

```sh
.venv/bin/python scripts/stage9_pickups.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/pickups.json" \
  "$RUN_DIR/pickup_review.jpg"
```

| Option | Purpose and when to use it |
|---|---|
| `--bend-ratio VALUE` | Adjust how much shorter a player box must be to count as a bend. Default: 0.75. |
| `--slow-speed M/S` | Adjust the maximum speed allowed for a bend candidate. Default: 0.8. |
| `--min-bend-duration SEC` | Require a bend to last at least this long. Default: 0.1. |
| `--max-search-sec SEC` | Change the post-rally search window for a pickup. Default: 12. |
| `--next-serve-margin-sec SEC` | Stop searching this long before the next serve. Default: 0.25. |
| `--shuttle-tracks PATH` | Add a tentative TrackNet resting-point candidate for visual review. |
| `--overwrite` | Replace pickup JSON and review image. |

Inspect the bend candidates in `$RUN_DIR/pickup_review.jpg`. If a side is clear but wrong, edit that point's `landing_side` in `$RUN_DIR/pickups.json` to `R` (the Stage 3 selected identity), `O` (the other tracked identity), or `null` (unknown), and set `status` to `manual_confirmed`. Pickup side is indirect evidence; leave uncertain results unknown.

If TrackNet is enabled, append `--shuttle-tracks "$RUN_DIR/shuttle_tracks.json"` to add a tentative stable-resting-point candidate. It remains separate from the pickup-based `landing_side` and needs visual confirmation.

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
  --r-name "Arun" \
  --o-name "Noble"
```

| Option | Purpose and when to use it |
|---|---|
| `--player-code R`, `--player-code O` | Choose which player's report and coaching analysis to generate. Default: R. |
| `--r-name NAME` (alias `--player-name`) | Set the name for identity R instead of answering the prompt. |
| `--o-name NAME` (alias `--opponent-name`) | Set the name for identity O instead of answering the prompt. |

Stage 10 is deterministic and uses local JSON files. It writes named score and movement summaries, the longest rally, TrackNet shot-count estimates when available, beginning-to-tail speed comparisons, timestamped loss examples, and heatmap analysis. Add `--player-code O` to focus the report on O; the pose JSON must also have been generated with `--player-code O`. Review the cited frames before treating patterns as causes.

## Stage 10A — Optional AI coach narrative

The default context mode writes `$RUN_DIR/AI-coach-context.md` beside an R report. Copy it into a chat agent if desired, then save the response as `$RUN_DIR/AI-coach-narrative.md` in the same run directory. For an O report, pass `--player-code O`; outputs default to `AI-coach-context-O.md` and `AI-coach-narrative-O.md`. Context mode makes no network request.

```sh
.venv/bin/python scripts/stage10a_ai_coach.py "$RUN_DIR/report_data.json"
```

| Option | Purpose and when to use it |
|---|---|
| `--player-code R`, `--player-code O` | Check that the context matches the selected Stage 10 report subject. |
| `--mode` | Choose `context` (default) or `ollama` for local narrative generation. |
| `--context-output PATH` | Choose where to write the self-contained coaching context. |
| `--narrative-output PATH` | Choose the Stage 11 coaching narrative output path. |
| `--ollama-url URL` | Change the local Ollama endpoint if it uses a non-default address. |
| `--model NAME` | Select an Ollama model already installed locally. |
| `--timeout SEC` | Change the maximum wait for Ollama. Default: 600. |
| `--overwrite` | Replace existing context and narrative files. |

To use Ollama locally instead, install Ollama and pull the model once:

```sh
ollama pull qwen2.5:7b
.venv/bin/python scripts/stage10a_ai_coach.py \
  "$RUN_DIR/report_data.json" \
  --mode ollama \
  --model qwen2.5:7b
```

| Option | Purpose and when to use it |
|---|---|
| `--mode ollama` | Generate the narrative with a local Ollama service instead of only writing a context file. |
| `--model NAME` | Choose the installed local model to use. |
| `--overwrite` | Replace existing context or narrative output. |

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

| Option | Purpose and when to use it |
|---|---|
| `--player-code R`, `--player-code O` | Choose the player; defaults to the subject in Stage 10. Pose/report files must match. |
| `--r-name NAME`, `--o-name NAME` | Set player names if they are missing or need overriding. |
| `--coach-narrative PATH` | Use a specific Stage 10A Markdown narrative. |
| `--shuttle-tracks PATH` | Draw the filtered shuttle trail and the per-rally shot counter. Requires Stage 7/10 shot data. |
| `--no-shuttle-overlay` | Hide the shuttle trail but keep the counter when TrackNet data is supplied. |
| `--shot-count-only` | Show the Stage 10 shot counter without loading or drawing the TrackNet trail. |
| `--pose-model PATH` | Select the local MediaPipe pose model file. |
| `--voice NAME` | Select an installed macOS `say` voice. |
| `--voice-rate WPM` | Change local narration speed. Default: 185. |
| `--sentence-pause-ms MS` | Change silence between spoken sentences; 0 disables it. Default: 600. |
| `--longest-rally-interval-sec SEC` | Change how often the longest-rally callout appears in full-match footage. Default: 48. |
| `--longest-rally-duration-sec SEC` | Change how long each callout remains visible. Default: 7. |
| `--freeze-duration SEC` | Set the minimum hold time for coaching review freeze frames. Default: 8. |
| `--video-encoder auto`, `libx264`, `h264_videotoolbox` | Choose automatic, software, or VideoToolbox H.264 encoding. |
| `--videotoolbox-bitrate RATE` | Set target bitrate when using VideoToolbox. |
| `--preset PRESET`, `--crf VALUE` | Tune software `libx264` speed and quality. |
| `--short-mode` | Omit the Full Match title card and full-match segment; keep the opening card, score/heatmap analysis, Coaching card, and review clips. |
| `--no-pose` | Skip pose inference and skeleton overlay. |
| `--no-voice` | Skip TTS narration and retain captions/source audio. |
| `--overwrite` | Replace an existing output video. |

Stage 11 opens with a 2.5-second title card showing both player names, final score, and a trophy beside the winner, followed by the score graph and heatmap. A 2.5-second “Coaching” card separates the heatmap discussion from the rally reviews. Each in-depth review presents its annotated still frame and spoken coaching note first, then plays the corresponding rally clip. Full-length mode adds a 2.5-second “Full Match” card before the annotated match footage; `--short-mode` omits that card and the match footage. Stage 11 reads the Stage 10 subject and requires matching Stage 8 pose data. It reads `AI-coach-narrative.md` for R or `AI-coach-narrative-O.md` for O beside `report_data.json`, and uses Stage 10 notes if no narrative exists. If TrackNet was used for Stage 7 shot counts, add `--shuttle-tracks "$RUN_DIR/shuttle_tracks.json"` to draw its short raw trail and show the live shot counter. To show only the count from Stage 10, use `--shot-count-only`; this does not draw a shuttle trail and does not need the TrackNet JSON at Stage 11. Both count modes need Stage 7 run with `--shuttle-tracks` and Stage 10 rerun afterward. `--no-shuttle-overlay` also hides the trail while keeping the count when `--shuttle-tracks` is supplied. The longest rally with a shot count is called out for 7 seconds every 48 seconds in the full-match section; adjust with `--longest-rally-interval-sec` and `--longest-rally-duration-sec`. Local narration uses macOS `say` with an added 600 ms silence between sentences by default; change it with `--sentence-pause-ms` or disable it with `0`. On macOS, Stage 11 probes `h264_videotoolbox` before using it; if FFmpeg lists the encoder but cannot open a session, `auto` falls back to `libx264`. Only video encoding is hardware accelerated; drawing tracking and pose overlays still uses the CPU. Add `--video-encoder libx264` to force software encoding, `--video-encoder h264_videotoolbox` to require VideoToolbox, or `--videotoolbox-bitrate 5M` to change its default `4M` target bitrate. Add `--no-voice` to skip narration, `--no-pose` to omit pose inference, or `--overwrite` to replace the output video. Its minimap always shows both tracked identities in camera perspective; the near court remains below the net and the far court above it.

For the trail and shot counter, add this option to the Stage 11 command:

```sh
--shuttle-tracks "$RUN_DIR/shuttle_tracks.json"
```

To display the counter without loading or drawing the shuttle trail, use:

```sh
--shot-count-only
```

This count-only option reads the per-rally shot counts from `report_data.json`; it still requires Stage 7 and Stage 10 to have been run with TrackNet shot analysis enabled.

### Optional: analyze O

Stages 0–7 produce shared canonical R/O tracking and match data. R means the Stage 3 clothing-selected player; O means the other tracked player. To create a separate O report, run Stage 8 for O and pass that output into Stage 10:

```sh
.venv/bin/python scripts/stage8_pose.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  models/pose_landmarker_heavy.task \
  "$RUN_DIR/pose_stats_O.json" \
  "$RUN_DIR/pose_review_O.jpg" \
  --player-code O
```

| Option | Purpose and when to use it |
|---|---|
| `--player-code O` | Analyze pose for the opponent identity instead of the default R. |

```sh
.venv/bin/python scripts/stage10_report.py \
  "$RUN_DIR/points.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/motion.json" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/pose_stats_O.json" \
  "$RUN_DIR/pickups.json" \
  "$RUN_DIR/report_data_O.json" \
  "$RUN_DIR/coach_report_O.md" \
  --player-code O \
  --r-name "Arun" \
  --o-name "Noble"
```

| Option | Purpose and when to use it |
|---|---|
| `--player-code O` | Generate the report for O; the pose JSON must also be from O. |
| `--r-name NAME`, `--o-name NAME` | Label both tracked identities in the report and coaching text. |

```sh
.venv/bin/python scripts/stage10a_ai_coach.py \
  "$RUN_DIR/report_data_O.json" \
  --player-code O
```

| Option | Purpose and when to use it |
|---|---|
| `--player-code O` | Match the AI-coach context to the O-specific Stage 10 report. |

```sh
.venv/bin/python scripts/stage11_video.py \
  "$RUN_DIR/match-720p.mp4" \
  "$RUN_DIR/players.json" \
  "$RUN_DIR/rallies_review.json" \
  "$RUN_DIR/pose_stats_O.json" \
  "$RUN_DIR/report_data_O.json" \
  "$RUN_DIR/match-O-analysis.mp4" \
  --player-code O \
  --pose-model models/pose_landmarker_heavy.task
```

| Option | Purpose and when to use it |
|---|---|
| `--player-code O` | Render the O-specific report and pose data. |
| `--pose-model PATH` | Use the local MediaPipe pose model. |

Stage 10A writes the O-specific context and narrative filenames automatically. For an AI narrative, either save the chat response as `$RUN_DIR/AI-coach-narrative-O.md` with the Stage 10A subject marker, or run Stage 10A with `--mode ollama`. To analyze R and O in the same match, retain the default R output files and repeat Stages 8–11 with the O-specific filenames above. Do not mix R and O pose/report files; Stage 10 and Stage 11 validate their subject codes.

## Reuse for another match

Set `RUN_DIR` to a new folder, for example `runs/match-2`, copy the new source video there, and update its filename in the commands. Do not reuse match-specific JSON reviews. Re-run downstream stages whenever an upstream review changes. Generated run folders and model files are Git-ignored; the scripts and this workflow remain versionable.
