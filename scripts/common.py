"""Shared file, video, and coordinate helpers for the analysis stages."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COURT_WIDTH = 6.1
COURT_LENGTH = 13.4
NET_Y = 6.7


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_job(path: Path) -> dict[str, Any]:
    job = read_json(path)
    if job.get("schema_version") != 1:
        raise ValueError(f"Unsupported job schema in {path}.")
    setup = job.get("analysis_setup", {})
    if setup.get("match_format") != "singles":
        raise ValueError("This pipeline currently supports singles matches only. Create a new singles job.")
    return job


def job_directory(job_path: Path) -> Path:
    return job_path.expanduser().resolve().parent


def artifact_path(job_path: Path, filename: str) -> Path:
    return job_directory(job_path) / filename


def proxy_path(job: dict[str, Any]) -> Path:
    value = job.get("stage_0_proxy", {}).get("path")
    if not value:
        raise ValueError("job.json has no Stage 0 proxy path. Run scripts/00_proxy.py first.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Proxy video not found: {path}")
    return path


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def video_properties(path: Path) -> dict[str, float | int]:
    capture = open_video(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read video frame rate: {path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if frame_count else 0.0,
    }


def frame_at(capture: cv2.VideoCapture, seconds: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def normalized_box_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    return (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(1, min(width, round(x2 * width))),
        max(1, min(height, round(y2 * height))),
    )


def calibration_for_time(calibration: dict[str, Any], seconds: float) -> dict[str, Any] | None:
    for segment in calibration.get("segments", []):
        if float(segment["start_time"]) <= seconds <= float(segment["end_time"]):
            return segment
    return None


def map_pixel_to_court(
    calibration_segment: dict[str, Any], point: tuple[float, float] | list[float]
) -> tuple[float, float] | None:
    matrix = np.asarray(calibration_segment["pixel_to_court"], dtype=np.float64)
    source = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(source, matrix)[0, 0]
    if not np.isfinite(mapped).all():
        return None
    return float(mapped[0]), float(mapped[1])


def timestamp_label(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def create_contact_sheet(
    video_path: Path,
    timestamps: list[float],
    output_path: Path,
    title: str,
    columns: int = 4,
) -> Path:
    """Write a timestamped frame grid for manual review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = open_video(video_path)
    tiles: list[np.ndarray] = []
    try:
        for seconds in timestamps:
            frame = frame_at(capture, seconds)
            tile = np.full((204, 320, 3), 245, dtype=np.uint8)
            if frame is not None:
                height, width = frame.shape[:2]
                scale = min(320 / width, 180 / height)
                resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                y = 22 + (180 - resized.shape[0]) // 2
                x = (320 - resized.shape[1]) // 2
                tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            cv2.putText(
                tile,
                f"{seconds:.2f}s  {timestamp_label(seconds)}",
                (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
    finally:
        capture.release()

    if not tiles:
        tiles.append(np.full((204, 320, 3), 245, dtype=np.uint8))
    rows = math.ceil(len(tiles) / columns)
    sheet = np.full((rows * 204 + 34, columns * 320, 3), 255, dtype=np.uint8)
    cv2.putText(sheet, title[:120], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y = 34 + row * 204
        x = column * 320
        sheet[y : y + 204, x : x + 320] = tile
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")
    return output_path


def prompt_float(
    prompt: str,
    minimum: float | None = None,
    maximum: float | None = None,
    default: float | None = None,
) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            value = default
        elif not raw:
            print("Enter a number.")
            continue
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Enter a number.")
                continue
        if not math.isfinite(value):
            print("Enter a finite number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Enter a value greater than or equal to {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Enter a value less than or equal to {maximum}.")
            continue
        return value


def prompt_choice(prompt: str, options: dict[str, str]) -> str:
    print(prompt)
    for key, label in options.items():
        print(f"  {key}. {label}")
    while True:
        answer = input("Choose: ").strip().lower()
        if answer in options:
            return answer
        print(f"Choose one of: {', '.join(options)}")
