#!/usr/bin/env python3
"""Click court corners and calibrate one or more stable camera intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import COURT_LENGTH, COURT_WIDTH, artifact_path, load_job, open_video, prompt_float, proxy_path, video_properties, write_json


COURT_POINTS = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (0.0, COURT_LENGTH),
    (COURT_WIDTH, COURT_LENGTH),
]
POINT_LABELS = ["near-left baseline corner", "near-right baseline corner", "far-left baseline corner", "far-right baseline corner"]


def prompt_segments(duration: float, movement: str) -> list[tuple[float, float, float]]:
    if movement == "fixed":
        return [(0.0, duration, duration / 2)]

    print(
        "Enter each interval where the camera is stable. Gaps between intervals are marked uncalibrated and will be skipped."
    )
    segments: list[tuple[float, float, float]] = []
    while True:
        raw_start = input("Stable interval start in seconds (blank when done): ").strip()
        if not raw_start:
            break
        try:
            start = float(raw_start)
        except ValueError:
            print("Enter a number.")
            continue
        if not 0 <= start < duration:
            print(f"Start must be between 0 and {duration:.2f}.")
            continue
        end = prompt_float("Stable interval end in seconds: ", minimum=start, maximum=duration)
        if end <= start:
            print("The interval end must be after its start.")
            continue
        reference = prompt_float(
            f"Timestamp of a clear frame in this interval [{(start + end) / 2:.2f}]: ",
            minimum=start,
            maximum=end,
            default=(start + end) / 2,
        )
        segments.append((start, end, reference))
    if not segments:
        raise ValueError("At least one stable camera interval is required.")
    segments.sort(key=lambda item: item[0])
    for previous, current in zip(segments, segments[1:]):
        if current[0] < previous[1]:
            raise ValueError("Stable camera intervals overlap. Enter non-overlapping intervals.")
    return segments


def click_corners(frame: np.ndarray, reference_time: float, segment_number: int) -> list[list[float]]:
    points: list[tuple[int, int]] = []
    window = f"Calibration segment {segment_number}"
    display = frame.copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    print("Use the same court orientation for every camera segment: x=0 is the left sideline when standing at the chosen near baseline facing the far baseline.")
    print("For a side view, choose one baseline as near and keep that choice for the whole video.")
    print(f"Click these four points in order on the {reference_time:.2f}s frame:")
    for index, label in enumerate(POINT_LABELS, start=1):
        print(f"  {index}. {label}")
    print("Press r to clear clicks, or q/Esc to cancel.")
    try:
        while True:
            display[:] = frame
            for index, (x, y) in enumerate(points):
                cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
                cv2.putText(display, str(index + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if len(points) == 4:
                break
            if key in (ord("r"), ord("R")):
                points.clear()
            elif key in (ord("q"), 27):
                raise RuntimeError("Calibration cancelled by user.")
    finally:
        cv2.destroyWindow(window)
    return [[float(x), float(y)] for x, y in points]


def make_overlay(frame: np.ndarray, pixel_to_court: np.ndarray) -> np.ndarray:
    court_to_pixel = np.linalg.inv(pixel_to_court)
    overlay = frame.copy()
    lines: list[list[tuple[float, float]]] = []
    for y in np.linspace(0, COURT_LENGTH, 9):
        lines.append([(0, y), (COURT_WIDTH, y)])
    for x in np.linspace(0, COURT_WIDTH, 7):
        lines.append([(x, 0), (x, COURT_LENGTH)])
    for line in lines:
        court_line = np.asarray([[line[0], line[1]]], dtype=np.float32)
        image_line = cv2.perspectiveTransform(court_line, court_to_pixel)[0]
        p1 = tuple(np.round(image_line[0]).astype(int))
        p2 = tuple(np.round(image_line[1]).astype(int))
        cv2.line(overlay, p1, p2, (40, 220, 40), 2, cv2.LINE_AA)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to calibration.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        duration = float(video_properties(video)["duration_seconds"])
        movement = job.get("analysis_setup", {}).get("camera_movement", "fixed")
        if job.get("analysis_setup", {}).get("camera_view") not in {"behind_baseline", "side"}:
            raise ValueError("Camera view must be behind_baseline or side.")
        intervals = prompt_segments(duration, movement)
        capture = open_video(video)
        segments = []
        previews = artifact_path(job_path, "calibration_previews")
        try:
            for index, (start, end, reference_time) in enumerate(intervals, start=1):
                frame = None
                capture.set(cv2.CAP_PROP_POS_MSEC, reference_time * 1000)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read frame at {reference_time:.2f}s.")
                image_points = click_corners(frame, reference_time, index)
                matrix, _inliers = cv2.findHomography(
                    np.asarray(image_points, dtype=np.float32),
                    np.asarray(COURT_POINTS, dtype=np.float32),
                    method=0,
                )
                if matrix is None or not np.isfinite(matrix).all():
                    raise RuntimeError("Could not calculate a court homography from those points.")
                preview_path = previews / f"segment_{index:02d}.jpg"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(preview_path), make_overlay(frame, matrix)):
                    raise RuntimeError(f"Could not write calibration preview: {preview_path}")
                segments.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "reference_time": reference_time,
                        "image_points": image_points,
                        "court_points": COURT_POINTS,
                        "pixel_to_court": matrix.tolist(),
                        "preview": str(preview_path),
                    }
                )
                print(f"Saved overlay for inspection: {preview_path}")
        finally:
            capture.release()

        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "calibration.json")
        write_json(
            output_path,
            {
                "schema_version": 1,
                "job": str(job_path),
                "camera_view": job["analysis_setup"]["camera_view"],
                "court_width_m": COURT_WIDTH,
                "court_length_m": COURT_LENGTH,
                "segments": segments,
            },
        )
        print(f"Calibration saved to {output_path}. Inspect every overlay before tracking players.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
