# TrackNetV3 Pilot Notes and Pipeline Integration

## Purpose

This document records the short TrackNetV3 pilot and the resulting optional pipeline integration. TrackNet inference runs from a separate Python environment. Match outputs belong in `runs/<match-name>/`; model weights and the environment stay outside the analyser repository unless deliberately bundled by `compress_application.sh --tracknet-path`.

TrackNetV3's repository documents its source and pretrained checkpoints. The inference script outputs per-frame `Frame`, `X`, `Y`, and binary `Visibility`; it can use TrackNet alone or TrackNet plus InpaintNet. Inpainted coordinates must be treated as estimates, not confirmed detections. Check the current upstream license terms before distributing its source or weights.

## Separate pilot environment

Use Python 3.11 and a new virtual environment. Do not install the clone's pinned `requirements.txt` into the analyser environment: it pins PyTorch 1.10, while this clone's Apple MPS selection needs a newer PyTorch. On Apple Silicon with a supported macOS version, PyTorch should report MPS available; on Intel Macs, inference falls back to CPU.

```sh
export TRACKNET_REPO="$HOME/TrackNetV3"
export TRACKNET_PILOT="$HOME/tracknetv3-pilot"
mkdir -p "$TRACKNET_PILOT"
python3.11 -m venv "$TRACKNET_PILOT/.venv"
"$TRACKNET_PILOT/.venv/bin/python" -m pip install --upgrade pip
"$TRACKNET_PILOT/.venv/bin/python" -m pip install torch "numpy<2" opencv-python pandas Pillow parse tqdm pycocotools
"$TRACKNET_PILOT/.venv/bin/python" -c 'import torch; mps=getattr(torch.backends, "mps", None); print("PyTorch:", torch.__version__); print("MPS available:", bool(mps and mps.is_available()))'
```

If `python3.11` is unavailable, install it with `brew install python@3.11`. Keep the pilot outputs under `tracknetv3-pilot/`.

## Checkpoint download

Download the checkpoint archive in a browser from:

<https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing>

The README calls the archive `TrackNetV3_ckpts.zip`. After downloading it to `~/Downloads`, extract the two expected files into the pilot directory:

```sh
mkdir -p "$TRACKNET_PILOT/models" "$TRACKNET_PILOT/checkpoints-unpacked"
unzip ~/Downloads/TrackNetV3_ckpts.zip -d "$TRACKNET_PILOT/checkpoints-unpacked"
find "$TRACKNET_PILOT/checkpoints-unpacked" -type f \( -name 'TrackNet_best.pt' -o -name 'InpaintNet_best.pt' \) -exec cp {} "$TRACKNET_PILOT/models/" \;
ls -lh "$TRACKNET_PILOT/models/TrackNet_best.pt" "$TRACKNET_PILOT/models/InpaintNet_best.pt"
```

If the archive has different filenames, inspect it with `unzip -l ~/Downloads/TrackNetV3_ckpts.zip` and use the actual names.

## First inference pilot

Choose a 20–30 second section of the Stage 0 proxy that includes a clear serve, live rally, and visible point ending. Use a second clip with adjacent-court activity or a shuttle occlusion if available. Set both start and end in seconds relative to the trimmed proxy. FFmpeg's `-t` is a duration, so the command calculates `end - start`.

```sh
export PROXY="/absolute/path/to/runs/match-name/match-720p.mp4"
mkdir -p "$TRACKNET_PILOT/clips" "$TRACKNET_PILOT/predictions-with-inpaint" "$TRACKNET_PILOT/predictions-raw"
START_SEC=120
END_SEC=145
CLIP_DURATION=$((END_SEC - START_SEC))
ffmpeg -ss "$START_SEC" -i "$PROXY" -t "$CLIP_DURATION" -an -c:v libx264 -preset fast -crf 18 "$TRACKNET_PILOT/clips/pilot-1.mp4"

cd "$TRACKNET_REPO"
"$TRACKNET_PILOT/.venv/bin/python" predict.py \
  --video_file "$TRACKNET_PILOT/clips/pilot-1.mp4" \
  --tracknet_file "$TRACKNET_PILOT/models/TrackNet_best.pt" \
  --inpaintnet_file "$TRACKNET_PILOT/models/InpaintNet_best.pt" \
  --save_dir "$TRACKNET_PILOT/predictions-with-inpaint" \
  --large_video --eval_mode weight --output_video
```

Run the same command again with `--save_dir "$TRACKNET_PILOT/predictions-raw"` and omit `--inpaintnet_file` to compare direct detections with repaired tracks. Expect a `pilot-1_ball.csv` and an annotated `pilot-1.mp4` in each output directory. The terminal should report `Using PyTorch device: mps` on a working Apple GPU setup or `cpu` otherwise. Review both videos in QuickTime and keep the CSVs; do not use the repaired path alone to judge a landing.

## What the pilot evaluates

1. **Serve candidates and server:** Locate shuttle-flight starts near service-ready moments and compare their image origins with the two tracked players. Use this only to rank Stage 4 candidates or suggest missing ones; retain the existing user review and score corrections.
2. **Rally and point endings:** Compare the last observed shuttle flight with Stage 6's player-pause estimate. Keep the pause estimate as fallback when the shuttle disappears or is occluded.
3. **Landing side and zone:** Use a likely floor-contact point plus court calibration and player pickup evidence. Apply the floor homography only to a shuttle visibly on or very near the floor; airborne image coordinates do not map directly to court metres. Return unknown when the landing is hidden or ambiguous.
4. **Coaching metrics:** After review, derive approximate flight/shot intervals, apparent direction, and landing zones with timestamps for Stage 10. Do not claim true shuttle speed, exact shot type, or a winner from TrackNet coordinates alone.

Record for each clip: whether the shuttle stays on the main court rather than an adjacent court, visible-track gaps, whether inpainting bridges only plausible short gaps, serve/server correctness, landing-side correctness, and rally-end difference from a visually reviewed timestamp. The pilot succeeds only if the overlays are useful and the added cues improve review without introducing confident false events. Share this plan, the raw/inpaint prediction videos and CSVs, plus notes on these checks before pipeline integration.

## Initial pilot observations

User review of the first clip reports that raw TrackNet output follows the main rally better, while InpaintNet adds implausible trails. Adjacent-court shuttle activity also appears in the output. Treat raw detections as the evidence baseline; do not use repaired points as proof of a shot, landing, or rally end.

The current prediction CSV contains one selected point per frame (`X`, `Y`, `Visibility`), not separate tracks or alternate shuttle candidates. Check whether the selected point switches to an adjacent court during the main rally. A court-region filter can discard off-court points, but it cannot recover a main-court shuttle candidate that the model did not export. If switching occurs, the next experiment should select from the model's heatmap using a soft main-court region and temporal continuity, with an expanded image-space margin for airborne shots.

Serve/server classification, point endings, and landing-side accuracy have not yet been reviewed in the pilot.

## Implemented optional pipeline support

- Stage 3A writes normalized raw observations and a CSV, with optional court-region tags from Stage 2 calibration. InpaintNet is disabled.
- Stage 4 attaches an image-space launch-origin cue without changing candidate detection. With `--use-shuttle-server-cue`, a confidence-qualified cue refines server/receiver labels after the center-line pass; racket and browser reviews can still override it.
- Stage 6 reports coverage by default. With `--rally-end-source shuttle`, a sustained calibrated main-court cluster near the player-pause estimate can set the rally end; a weak or missing cluster keeps the movement estimate.
- Stage 9 can add a tentative stationary shuttle location. The existing pickup-based `landing_side` remains authoritative.
- Stage 10 includes Stage 6 and 9 evidence in its report; Stage 10A warns against treating it as confirmed evidence.
- Stage 11 can draw a filtered raw shuttle trail. It appears when TrackNet JSON is supplied and can be hidden with `--no-shuttle-overlay`.

See the [installation guide](how-to-install.md) and [command pipeline](command_pipeline.md) for new-Mac setup and current run commands. Validate on short clips before choosing to run full-match inference. The model exports one selected coordinate per frame, so court-region filtering cannot recover an alternate main-court shuttle when TrackNet switches to adjacent-court activity.
