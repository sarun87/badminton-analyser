For the full command-by-command setup and interactive review walkthrough, see [HOW_TO_RUN.md](HOW_TO_RUN.md).

## What you get at the end

From one phone video of a match:

| Output | What it tells you |
|---|---|
| Point-by-point score | Who served, from which side, who won each point, the running score |
| Rally lengths | How long every point lasted, and who wins short vs long rallies |
| Movement | Metres run, speed, how often each player is pulled out of position, court heatmaps |
| Body position | Crouch, stance width, racket up or down while waiting for the opponent's shot |
| Point-ending clues | Approximate pickup side after each point, for manual review |
| Coach summary | The 3 or 4 patterns that cost you the most points, with example timestamps |
| Overlay video | Tracking overlays, pose skeleton, court mini map, scoreboard; coach voice-over remains unimplemented |


---

## Run the Singles Pipeline (macOS)

This pipeline supports Intel and Apple Silicon Macs. Install Homebrew first if it is not already installed, then run the initializer; it detects the Mac architecture, selects native Homebrew, and installs FFmpeg, Python 3.12, and the local Python packages:

```sh
bash scripts/init_macos.sh
source .venv/bin/activate
```

The first run of detection or pose analysis downloads the free Ultralytics model weights. Video processing and all
analysis run locally; no hosted API is used. Apple Silicon uses MPS for inference when available; Intel Macs use CPU.

Create a new job for each video, then pass its `job.json` to every later stage:

```sh
python scripts/00_proxy.py --input "/path/to/match.mov" --output-dir "runs/match-01"
MATCH_JOB="runs/match-01/job.json"
python scripts/01_detect.py --job "$MATCH_JOB"
python scripts/02_calibrate.py --job "$MATCH_JOB"
python scripts/03_track.py --job "$MATCH_JOB"
python scripts/04_serve_candidates.py --job "$MATCH_JOB"
python scripts/05_score.py --job "$MATCH_JOB"
python scripts/06_rallies.py --job "$MATCH_JOB"
python scripts/07_movement.py --job "$MATCH_JOB"
python scripts/08_pose.py --job "$MATCH_JOB"
python scripts/09_pickups.py --job "$MATCH_JOB"
python scripts/10_report.py --job "$MATCH_JOB"
python scripts/11_render_video.py --job "$MATCH_JOB" --you-name "Your Name" --opponent-name "Opponent"
```

Stage 0 prompts for camera view and movement, confirms source resolution/FPS detected by `ffprobe`, and asks for a
description of your clothing. This pipeline currently analyzes singles only. It writes `proxy.mp4` and `job.json` in
the output directory. The proxy is H.264, 1280x720, 30 FPS, BT.709 SDR; it keeps the source aspect ratio with padding
and retains audio when present. HDR transfer tags trigger tone mapping. Source resolution/FPS are recorded as metadata;
the proxy always uses the standard dimensions and frame rate above.

### Stage outputs and review steps

| Stage | Script | Output and required review |
|---|---|---|
| 1. Detect people | `scripts/01_detect.py` | `detections.json`; person boxes sampled at about 10 FPS. |
| 2. Calibrate court | `scripts/02_calibrate.py` | `calibration.json` and `calibration_previews/`; click the four court corners and inspect every projected court overlay. |
| 3. Track players | `scripts/03_track.py` | `players.json` and `review/tracking/`; select yourself in a reference frame, then inspect annotated tracking samples. |
| 4. Find serve pauses | `scripts/04_serve_candidates.py` | `serve_candidates.json`; automated candidate periods only. |
| 5. Review serves and score | `scripts/05_score.py` | `score.json` and `review/serves/`; confirm or discard candidates, add missed serves, select server/service court, mark game starts, and confirm the final point winner. |
| 6. Review rally ends | `scripts/06_rallies.py` | `points.json` and `review/rallies/`; inspect each contact sheet and enter the last live-play timestamp. |
| 7. Measure movement | `scripts/07_movement.py` | `motion.json` and `heatmaps/`; distance, speed, court share, recovery trips, and heatmaps. |
| 8. Estimate body position | `scripts/08_pose.py` | `pose_stats.json`; approximate pose measures grouped by outcome and court context. |
| 9. Estimate pickup side | `scripts/09_pickups.py` | `pickups.json`; low-confidence bend-based shuttle-side clues. |
| 10. Build report | `scripts/10_report.py` | `report.md`; local statistics and timestamp-linked review suggestions. Optional `--ollama-model` writes `coach_draft.md` through a local Ollama server. |
| 11. Render overlay video | `scripts/11_render_video.py` | `overlay.mp4`; foot rings, trails, user skeleton, top-down court map, point number, server, winner, and running score. Pass `--you-name` and `--opponent-name` for scoreboard labels. |

Stages 1–11 accept `--job` and optional `--output` or input-artifact path arguments; Stage 0 accepts `--input` and `--output-dir`. Run `python scripts/NN_name.py --help` for a stage's options. Stages 2, 3, 5, and 6 require interactive review.

For an optional local AI draft, install Ollama and a small free model. Start `ollama serve` in a separate terminal,
then run `ollama pull llama3.2:3b` and `python scripts/10_report.py --job "$MATCH_JOB" --ollama-model llama3.2:3b`.
This writes `coach_draft.md` by calling Ollama on `127.0.0.1`; the default report works without Ollama.

The court calibration works with a baseline or side view when all four court corners are visible. For a moving camera,
Stage 2 asks for stable time intervals; uncalibrated gaps are skipped. A continuously moving camera requires enough
stable intervals to cover the useful footage. Singles tracking assumes one player on each side of the net.

---

## Architecture

```
phone video (4K / 1080p)
   |
   v
[0] PROXY        ffmpeg -> 1280x720, 30 fps, SDR (HDR iPhone video must be tone-mapped)
   |
   v
[1] DETECT       YOLO person boxes every 3rd frame (10 fps)                 -> detections.json
   |
[2] CALIBRATE    4+ court points clicked once per camera position -> homography H (court metres <-> pixels)
   |
   v
[3] PLAYERS      clothing appearance match + opponent, feet -> court metres, gap fill, despike         -> players.json
   |
   +--> [4] SERVES     stationary windows in opposite service areas                                  -> serve_candidates.json
   |         |
   |         v
   |    [5] SCORE      visual serve review + singles score replay                                    -> score.json
   |         |
   |         v
   |    [6] RALLIES    manually reviewed rally ends from contact sheets                               -> points.json
   |
   +--> [7] MOVEMENT   distance, speed, front/back court share, trips out of position, heatmaps        -> motion.json
   |
   +--> [8] POSE       body landmarks on a crop around the identified user                            -> pose_stats.json
   |
   +--> [9] PICKUPS    who bends to pick up the shuttle after the rally -> which side it landed         -> pickups.json
   |
   v
[10] REPORT      local metrics and evidence-linked coaching notes                                    -> report.md
   |
   v
[11] VIDEO       tracking overlays + pose skeleton + mini map + score panel                  -> overlay.mp4
```

### Court coordinate system (used everywhere)

Metres, oriented as if standing at the chosen near baseline and facing the far baseline. For a side view, choose one
baseline as near and keep that orientation for every calibration segment:

- `x` across the court: `0` = the left doubles sideline (camera view), `6.1` = the right doubles sideline.
- `y` along the court: `0` = near baseline, `6.7` = net, `13.4` = far baseline.
- Useful lines: centre line `x = 3.05`, singles sidelines `x = 0.46` and `x = 5.64`, near short service line
  `y = 4.72` (1.98 m from the net), far short service line `y = 8.68`, doubles long service lines 0.76 m inside each
  baseline.

---

## The stages in detail

### [0] Working copy

Convert the match to a 1280x720, 30 fps SDR file once. Everything below reads this copy. iPhone videos are often HDR
(HLG): Stage 0 tone-maps tagged HDR video to SDR so clothing colors remain useful for appearance matching.

### [1] Person detection

- Run the free Ultralytics YOLO11n person detector on every 3rd proxy frame (10 fps), confidence 0.3, NMS IoU 0.5.
- Store boxes as fractions of the frame so resolution does not matter:

```json
{"fps": 10.0, "frames": [{"t": 12.3, "p": [[x1, y1, x2, y2, score], ...]}, ...]}
```

The first run downloads the model weights; inference uses Apple Silicon MPS when available and CPU on Intel Macs.

### [2] Court calibration (homography)

- Take one clear frame per stable camera position and click the four baseline/sideline court corners in the order
  shown by the calibration UI. All four corners must be visible; side views work when they show the full court.
- `H = cv2.findHomography(court_points_metres, pixel_points)`. Map pixels to the court with `inv(H)`.
- **Check it:** draw the whole court from `H` back onto the frame. Every line must sit on the real line. If not,
  re-click. Everything downstream depends on this.
- If the camera moved, enter each stable time range. The tool stores one `H` per range and leaves movement gaps
  uncalibrated so downstream stages skip them.

### [3] Player identification and tracking

- Select yourself in a clear reference frame and choose whether the identifying clothing is your shirt or shorts.
  The tracker compares a color histogram for that clothing region across detections.
- A player's position is the **bottom centre of the box** (the feet), mapped to court metres with the segment's
  homography. The appearance-matched player is `r`; the remaining player in the opposite half is `o`.
- **Filter out** boxes taller than 55% of the frame (someone walking past the lens) and people outside the court
  margins (friends at the net post, people sitting on the side).
- **Gap fill:** fill a short identity gap only when a detection is within 0.8 m of the interpolated track and still
  resembles the selected clothing sample.
- **Despike:** a single frame more than 0.9 m away from both neighbours (which agree with each other) is a mix-up.
  Drop it.

```json
{"fps": 10.0, "frames": [{"t": 12.3, "r": [x, y], "rb": [box], "o": [x, y], "ob": [box]}, ...]}
```

`r` = you, `o` = opponent, positions in court metres. Annotated samples are written under `review/tracking/`.

### [4] Serve candidates

- Smooth positions over 0.5 s (box jitter), then compute speed.
- A candidate requires both players in opposite singles service areas, within the court's x margins, moving slowly
  for at least 0.8 s. This stage does not infer who served; the next stage visually confirms that.
- Stage 5 records the server and service court from contact sheets. The server's score must be even in his right
  service court and odd in his left; contradictions are flagged in `score.json`.

### [5] Score replay

- Singles rule: the **winner of a rally serves the next one**. So the server of serve N+1 is the winner of point N.
- The side rule is a consistency check: the server must stand in the right court when his score is even. Enter the
  initial score if the video starts mid-game and mark every game reset.
- **Check every serve by eye:** Stage 5 writes a contact sheet for each candidate, lets you discard false candidates,
  and lets you add missed serves manually. The winner of point N is inferred from the server of point N+1; the last
  point's winner is entered manually. Mark the first serve after each game break so the score and side parity reset.

### [6] Rally ends

- **Do not use racket sounds in a hall with more than one court.** My first version did, and other courts' hits made
  rallies look twice as long (wrong by more than 1 s on 52 of 64 points).
- Stage 6 writes a contact sheet per point (one frame every 0.5 s from serve to next serve) and asks you to enter
  the last live-play timestamp before the pause. The timing is manual and should be checked against the video.
- Automating this properly needs shuttle tracking (for example TrackNet-style models). I did not use one here.

```json
{"n": 21, "t": 236.1, "next_t": 247.0, "server": "R", "side": "right", "before": "8-12", "winner": "O",
 "after": "8-13", "r0": 236.1, "r1": 237.85, "rdur": 1.75}
```

### [7] Movement

Per rally (serve + 0.3 s to the rally end), for each player, in his own half (distance from his own baseline, and
across from his own left sideline, so both players compare directly):

- Smooth over 0.7 s before summing distance (jitter otherwise adds about 30%).
- Distance, average and peak speed, share of time in the back court (within 1.6 m of the baseline) and front court
  (past the short service line), median position.
- **Trips out of position:** moments at least 2.2 m from the base point (centre line, 3.4 m from the baseline), and
  how far away he still is 1 s later (recovery).
- Heatmap of each player's half (`numpy.histogram2d` drawn with matplotlib).

### [8] Body position (pose)

- Run the free YOLO11n-pose model on a square crop around the identified player's box (1.35x the box size), sampled
  at about 10 fps.
- 2D knee angles are viewpoint-dependent, so compare these pose measures within one video and one camera setup:
  - **crouch** = how much lower the hips are than when standing upright: hip-to-ankle height divided by torso
    length, compared with your own upright frames between points.
  - **stance width** = ankle-to-ankle distance divided by shoulder width.
  - **racket arm up** = a wrist above the shoulders (prepared).
- Contexts: **READY** = during a rally, moving slower than 0.8 m/s and within 1.6 m of base (waiting for the shot),
  FRONT (near the net), REAR (near the baseline). Compare points **won vs lost**.

### [9] Where the point ended (pickups)

- After a rally someone may bend to pick the shuttle up. The script estimates standing box height by court-depth
  bins, then treats a slow box below 72% of that estimate as a possible bend.
- The first slow post-rally bend is saved as a pickup-side clue. It is a rough estimate and should be visually
  confirmed alongside the winner and contact-sheet frames; the script does not label a specific shot error from
  this clue alone.

### [10] Report and coaching

- Put every number in one place, split by **points won vs lost** and by **rally length** (5 s or less, 5-10 s, over
  10 s). The biggest differences are the lessons.
- Stage 10 writes a deterministic local report with outcome comparisons and timestamp-linked review suggestions.
  For an optional local AI draft, install Ollama and a free model, start its local server, and run:

  ```sh
  brew install ollama
  # In a separate terminal:
  ollama serve
  # Back in this terminal:
  ollama pull llama3.2:3b
  python scripts/10_report.py --job "$MATCH_JOB" --ollama-model llama3.2:3b
  ```

  The draft is saved as `coach_draft.md`; review every claim against its cited footage.

### [11] The video (optional)

Stage 11 renders `overlay.mp4` at the proxy's frame rate and resolution, with source audio when present. It draws
foot rings and trails for both players, the user's pose skeleton from Stage 8, a top-down court map, and a scoreboard
with the point, server, winner, names, and running score. Supply display names with `--you-name` and
`--opponent-name`. Run Stage 8 first for the skeleton overlay; tracking, map, and score overlays can render without
pose samples.

---

## 5. Checks before you trust any number

1. The court drawn back from `H` sits on the real lines (every camera position).
2. A sample of tracked frames: the dot is on **your** feet, not on a friend at the net post.
3. Every serve on the contact sheets: right server, right side. The service-court rule has 0 contradictions.
4. Rally ends from the picture, not from sound.
5. Distances look human: around 1 m/s average during live play in singles.


