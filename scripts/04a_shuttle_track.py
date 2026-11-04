#!/usr/bin/env python3
"""Experimental local shuttle detector and short-trajectory tracker.

This OpenCV-only pass follows small bright moving image regions. It is a useful
visual aid, not a trained shuttlecock detector; gym lights, lines, and motion
blur can create false or missing tracks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import artifact_path, load_job, proxy_path, video_properties, write_json


def moving_bright_candidates(frame: np.ndarray, previous_gray: np.ndarray) -> list[dict[str, float]]:
    height, width = frame.shape[:2]
    scale = min(1.0, 960.0 / width)
    small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    diff = cv2.absdiff(gray, previous_gray)
    mask = cv2.inRange(gray, 170, 255) & cv2.inRange(hsv[:, :, 1], 0, 155) & cv2.inRange(diff, 20, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_back = 1.0 / scale
    candidates: list[dict[str, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, box_w, box_h = cv2.boundingRect(contour)
        pixels = box_w * box_h
        if pixels < 2 or pixels > 48 or area > 38:
            continue
        aspect = max(box_w, box_h) / max(1, min(box_w, box_h))
        if aspect > 7.0:
            continue
        region_gray = gray[y : y + box_h, x : x + box_w]
        region_diff = diff[y : y + box_h, x : x + box_w]
        bright_ratio = float(np.mean(region_gray >= 170)) if region_gray.size else 0.0
        moving_ratio = float(np.mean(region_diff >= 20)) if region_diff.size else 0.0
        shape_score = max(0.0, 1.0 - abs(aspect - 1.5) / 6.0)
        confidence = float(np.clip(0.48 * bright_ratio + 0.32 * moving_ratio + 0.20 * shape_score, 0.0, 1.0))
        if confidence < 0.38:
            continue
        candidates.append(
            {
                "x": (x + box_w / 2) * scale_back,
                "y": (y + box_h / 2) * scale_back,
                "area": area * scale_back * scale_back,
                "confidence": confidence,
            }
        )
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates[:24]


def append_tracks(
    active: list[dict[str, Any]],
    candidates: list[dict[str, float]],
    frame_index: int,
    seconds: float,
    next_id: int,
    frame_width: int,
) -> int:
    available = set(range(len(candidates)))
    for track in sorted(active, key=lambda value: value["last_frame"], reverse=True):
        gap = frame_index - int(track["last_frame"])
        if gap < 1 or gap > 3 or not available:
            continue
        last = track["points"][-1]
        if len(track["points"]) > 1:
            prior = track["points"][-2]
            dt = max(1e-6, float(last["t"]) - float(prior["t"]))
            vx = (float(last["x"]) - float(prior["x"])) / dt
            vy = (float(last["y"]) - float(prior["y"])) / dt
            predicted_x = float(last["x"]) + vx * (seconds - float(last["t"]))
            predicted_y = float(last["y"]) + vy * (seconds - float(last["t"]))
        else:
            predicted_x, predicted_y = float(last["x"]), float(last["y"])
        radius = max(45.0, frame_width * 0.20) * gap
        choices = [
            (float(np.hypot(candidates[index]["x"] - predicted_x, candidates[index]["y"] - predicted_y)), index)
            for index in available
        ]
        distance, index = min(choices)
        if distance <= radius:
            candidate = candidates[index]
            track["points"].append({"t": round(seconds, 4), **candidate})
            track["last_frame"] = frame_index
            available.remove(index)

    for index in sorted(available, key=lambda item: candidates[item]["confidence"], reverse=True):
        candidate = candidates[index]
        active.append(
            {
                "id": next_id,
                "last_frame": frame_index,
                "points": [{"t": round(seconds, 4), **candidate}],
            }
        )
        next_id += 1
    return next_id


def finish_tracklets(active: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tracklets = []
    for track in active:
        points = track["points"]
        if len(points) < 3:
            continue
        elapsed = float(points[-1]["t"]) - float(points[0]["t"])
        displacement = float(np.hypot(points[-1]["x"] - points[0]["x"], points[-1]["y"] - points[0]["y"]))
        if elapsed <= 0 or displacement < 8:
            continue
        speeds = []
        turns = []
        for before, after in zip(points, points[1:]):
            dt = float(after["t"]) - float(before["t"])
            if dt > 0:
                speeds.append(float(np.hypot(after["x"] - before["x"], after["y"] - before["y"]) / dt))
        for index in range(1, len(points) - 1):
            a, b, c = points[index - 1], points[index], points[index + 1]
            first = np.asarray([b["x"] - a["x"], b["y"] - a["y"]], dtype=float)
            second = np.asarray([c["x"] - b["x"], c["y"] - b["y"]], dtype=float)
            denom = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denom > 1:
                angle = float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denom, -1.0, 1.0))))
                if angle >= 55:
                    turns.append({"t": b["t"], "angle_degrees": round(angle, 1), "label": "possible_direction_change"})
        tracklets.append(
            {
                "id": int(track["id"]),
                "start": points[0]["t"],
                "end": points[-1]["t"],
                "duration_seconds": round(elapsed, 4),
                "displacement_px": round(displacement, 1),
                "median_speed_px_s": round(float(np.median(speeds)), 1) if speeds else None,
                "mean_detection_confidence": round(float(np.mean([point["confidence"] for point in points])), 3),
                "points": points,
                "possible_direction_changes": turns,
            }
        )
    return tracklets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to shuttle_tracks.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "shuttle_tracks.json")
        properties = video_properties(video)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        fps = float(properties["fps"])
        width, height = int(properties["width"]), int(properties["height"])
        frame_count = int(properties["frame_count"])
        active: list[dict[str, Any]] = []
        all_tracks: list[dict[str, Any]] = []
        previous_gray = None
        next_id = 1
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                scale = min(1.0, 960.0 / width)
                small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                seconds = frame_index / fps
                if previous_gray is not None:
                    candidates = moving_bright_candidates(frame, previous_gray)
                    next_id = append_tracks(active, candidates, frame_index, seconds, next_id, width)
                previous_gray = gray
                expired = [track for track in active if frame_index - int(track["last_frame"]) > 3]
                for track in expired:
                    all_tracks.append(track)
                    active.remove(track)
                frame_index += 1
                if frame_index % max(1, int(fps * 60)) == 0:
                    print(f"Scanned {frame_index / fps / 60:.0f} minutes of video...")
        finally:
            capture.release()
        all_tracks.extend(active)
        tracklets = finish_tracklets(all_tracks)
        tracklets.sort(key=lambda value: float(value["start"]))
        write_json(
            output_path,
            {
                "schema_version": 1,
                "method": "OpenCV bright-motion candidate tracking; experimental, no shuttle-specific model",
                "video_size": [width, height],
                "fps": fps,
                "frames_scanned": frame_index,
                "tracklets": tracklets,
                "possible_direction_change_count": sum(len(track["possible_direction_changes"]) for track in tracklets),
                "limitations": [
                    "Tracklets are pixel paths, not calibrated three-dimensional flight paths.",
                    "Bright motion from lights, court lines, and compression can be mistaken for the shuttle.",
                    "Direction changes are review cues; this does not classify clears, drops, smashes, or stroke technique.",
                ],
            },
        )
        print(f"Saved {len(tracklets)} experimental shuttle tracklets to {output_path}.")
        print("Inspect the rendered overlay before using any path or direction-change cue as evidence.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"Shuttle tracking failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
