#!/usr/bin/env python3
"""Run raw TrackNetV3 inference in its separate environment and normalize outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def court_geometry(path: Path | None, width: int, length: int) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data.get("court_to_pixel"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("calibration JSON needs a finite 3x3 court_to_pixel matrix")
    size = data.get("court_size_m", {})
    court_width = float(size.get("width", 6.1))
    court_length = float(size.get("length", 13.4))
    corners_m = np.asarray(
        [[[0.0, 0.0], [court_width, 0.0], [court_width, court_length], [0.0, court_length]]],
        dtype=np.float32,
    )
    corners_px = cv2.perspectiveTransform(corners_m, matrix.astype(np.float32)).reshape(-1, 2)
    if not np.isfinite(corners_px).all():
        raise ValueError("calibration projected non-finite court corners")
    if "frame_size_px" in data:
        expected = data["frame_size_px"]
        if int(expected.get("width", width)) != width or int(expected.get("height", length)) != length:
            raise ValueError("calibration frame size differs from the input video")
    return corners_px.astype(np.float32), {
        "court_size_m": {"width": court_width, "length": court_length},
        "court_to_pixel": matrix.tolist(),
        "pixel_to_court": np.linalg.inv(matrix).tolist(),
        "court_polygon_px": np.round(corners_px, 2).tolist(),
        "source_calibration": path.name,
        "region_note": "Polygon is the court floor projection; airborne shuttle positions may fall outside it.",
    }


def read_track_csv(path: Path) -> dict[int, tuple[float | None, float | None, bool]]:
    rows: dict[int, tuple[float | None, float | None, bool]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Frame", "X", "Y", "Visibility"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("TrackNet CSV must contain Frame, X, Y, and Visibility columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_number = int(float(row["Frame"]))
                visible = int(float(row["Visibility"])) != 0
                x = float(row["X"]) if visible else None
                y = float(row["Y"]) if visible else None
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid TrackNet CSV row {line_number}") from error
            if frame_number < 0 or frame_number in rows:
                raise ValueError(f"invalid or duplicate TrackNet frame number {frame_number}")
            if visible and (x is None or y is None or not math.isfinite(x) or not math.isfinite(y)):
                visible, x, y = False, None, None
            rows[frame_number] = (x, y, visible)
    return rows


def build_payload(
    source_video: Path,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    raw_rows: dict[int, tuple[float | None, float | None, bool]],
    polygon: np.ndarray | None,
    court_metadata: dict[str, Any] | None,
    margin_px: float,
    checkpoint: Path,
    repo: Path,
) -> dict[str, Any]:
    frames = []
    for frame_index in range(frame_count):
        x, y, visible = raw_rows.get(frame_index, (None, None, False))
        region = "unknown"
        if visible and polygon is not None:
            distance = cv2.pointPolygonTest(
                polygon.reshape(-1, 1, 2), (float(x), float(y)), True
            )
            region = "inside" if distance >= 0 else "near_boundary" if distance >= -margin_px else "outside"
        frames.append({
            "frame": frame_index,
            "t": round(frame_index / fps, 6),
            "x_px": round(float(x), 2) if visible else None,
            "y_px": round(float(y), 2) if visible else None,
            "visible": bool(visible),
            "court_region": region,
        })
    visible_count = sum(row["visible"] for row in frames)
    return {
        "schema_version": 1,
        "source_video": source_video.name,
        "timestamp_convention": "seconds relative to the Stage 0 proxy",
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "frame_size_px": {"width": width, "height": height},
        "model": {
            "name": "TrackNetV3 TrackNet",
            "checkpoint": checkpoint.name,
            "repository": repo.name,
            "inpaintnet_used": False,
            "raw_visibility_count": visible_count,
            "court_margin_px": margin_px if polygon is not None else None,
        },
        "court_geometry": court_metadata,
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run optional raw TrackNetV3 shuttle tracking using a separate Python environment."
    )
    parser.add_argument("input_video", type=Path, help="Stage 0 proxy MP4")
    parser.add_argument("output_json", type=Path, help="output shuttle_tracks.json")
    parser.add_argument("--tracknet-repo", type=Path, required=True, help="local TrackNetV3 checkout")
    parser.add_argument("--tracknet-python", type=Path, required=True, help="Python executable in TrackNet's separate environment")
    parser.add_argument("--checkpoint", type=Path, required=True, help="TrackNet_best.pt checkpoint")
    parser.add_argument("--calibration-json", type=Path, help="Stage 2 calibration used to mark likely main-court points")
    parser.add_argument("--preview-video", type=Path, help="annotated raw-track video (default: shuttle_tracking_preview.mp4 beside JSON)")
    parser.add_argument("--no-preview-video", action="store_true", help="skip TrackNet's annotated preview video")
    parser.add_argument("--batch-size", type=int, default=8, help="TrackNet inference batch size (default: 8)")
    parser.add_argument("--court-margin-px", type=float, default=80.0, help="soft court polygon margin for airborne shuttle points")
    parser.add_argument("--overwrite", action="store_true", help="replace output JSON, CSV, and preview files")
    args = parser.parse_args()

    video = args.input_video.expanduser().resolve()
    output = args.output_json.expanduser().resolve()
    repo = args.tracknet_repo.expanduser().resolve()
    # Keep a venv's `bin/python` symlink intact; resolving it can bypass pyvenv.cfg
    # and launch the system interpreter without TrackNet's installed packages.
    python = Path(os.path.abspath(args.tracknet_python.expanduser()))
    checkpoint = args.checkpoint.expanduser().resolve()
    calibration = args.calibration_json.expanduser().resolve() if args.calibration_json else None
    raw_csv_output = output.with_name(output.stem + "_raw.csv")
    preview = args.preview_video.expanduser().resolve() if args.preview_video else output.with_name("shuttle_tracking_preview.mp4")

    if not video.is_file() or not (repo / "predict.py").is_file() or not python.is_file() or not checkpoint.is_file():
        parser.error("input video, TrackNet predict.py, TrackNet Python, and checkpoint must exist")
    if calibration is not None and not calibration.is_file():
        parser.error(f"calibration JSON does not exist: {calibration}")
    if args.batch_size < 1 or args.court_margin_px < 0:
        parser.error("batch size must be positive and court margin cannot be negative")
    outputs = [output, raw_csv_output] + ([] if args.no_preview_video else [preview])
    if len(set(outputs)) != len(outputs) or video in outputs or calibration in outputs:
        parser.error("input and output paths must be different")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error("output already exists: " + ", ".join(map(str, existing)) + "; pass --overwrite")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("input video has invalid frame properties")
    polygon, court_metadata = court_geometry(calibration, width, height)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tracknet_inference_", dir=output.parent) as temp_name:
        temp_dir = Path(temp_name)
        command = [
            str(python), str(repo / "predict.py"),
            "--video_file", str(video),
            "--tracknet_file", str(checkpoint),
            "--batch_size", str(args.batch_size),
            "--eval_mode", "weight",
            "--save_dir", str(temp_dir),
            "--large_video",
        ]
        if not args.no_preview_video:
            command.append("--output_video")
        print("Running raw TrackNetV3 inference (InpaintNet disabled).", flush=True)
        subprocess.run(command, cwd=repo, check=True)

        predicted_csv = temp_dir / f"{video.stem}_ball.csv"
        predicted_video = temp_dir / video.name
        if not predicted_csv.is_file():
            raise RuntimeError(f"TrackNet did not create the expected CSV: {predicted_csv}")
        raw_rows = read_track_csv(predicted_csv)
        if any(frame_index >= frame_count for frame_index in raw_rows):
            raise ValueError("TrackNet CSV contains frame indices beyond the input video")
        payload = build_payload(
            video, fps, frame_count, width, height, raw_rows,
            polygon, court_metadata, args.court_margin_px, checkpoint, repo,
        )
        raw_csv_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(predicted_csv, raw_csv_output)
        if not args.no_preview_video:
            if not predicted_video.is_file():
                raise RuntimeError(f"TrackNet did not create the expected preview video: {predicted_video}")
            preview.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(predicted_video, preview)

    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"TrackNet frames: {frame_count}; raw visible detections: {payload['model']['raw_visibility_count']}; "
        f"CSV: {raw_csv_output}; JSON: {output}"
    )
    if not args.no_preview_video:
        print(f"Raw TrackNet preview: {preview}")
    print("Court-region tags are a soft image-space cue; airborne points are not ground coordinates.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError, cv2.error, json.JSONDecodeError) as error:
        print(f"Stage 3A TrackNet failed: {error}", file=sys.stderr)
        raise SystemExit(1)
