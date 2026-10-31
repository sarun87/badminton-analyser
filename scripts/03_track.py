#!/usr/bin/env python3
"""Assign the user's appearance and opponent to detections, then map feet to court metres."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import (
    NET_Y,
    artifact_path,
    calibration_for_time,
    frame_at,
    load_job,
    map_pixel_to_court,
    normalized_box_to_pixels,
    open_video,
    prompt_float,
    proxy_path,
    read_json,
    write_json,
)


def appearance_histogram(frame: np.ndarray, box: list[float], part: str) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = normalized_box_to_pixels(box, width, height)
    box_height = max(1, y2 - y1)
    left = x1 + int((x2 - x1) * 0.20)
    right = x2 - int((x2 - x1) * 0.20)
    if part == "shorts":
        top, bottom = y1 + int(box_height * 0.47), y1 + int(box_height * 0.69)
    else:
        top, bottom = y1 + int(box_height * 0.16), y1 + int(box_height * 0.43)
    crop = frame[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram


def build_candidates(
    boxes: list[list[float]],
    width: int,
    height: int,
    segment: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []
    for box in boxes:
        if len(box) < 5:
            continue
        x1, y1, x2, y2, confidence = (float(value) for value in box[:5])
        if y2 - y1 > 0.55 or confidence < 0.25:
            continue
        pixel_box = normalized_box_to_pixels(box, width, height)
        center_bottom = ((pixel_box[0] + pixel_box[2]) / 2, pixel_box[3])
        court = map_pixel_to_court(segment, center_bottom)
        if court is None:
            continue
        court_x, court_y = court
        if not (-0.8 <= court_x <= 6.9 and -1.2 <= court_y <= 14.6):
            continue
        candidates.append({"box": box, "court": [court_x, court_y], "confidence": confidence})
    return candidates


def choose_reference_person(
    frame: np.ndarray,
    candidates: list[dict[str, Any]],
) -> int:
    display = frame.copy()
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = normalized_box_to_pixels(candidate["box"], frame.shape[1], frame.shape[0])
        cv2.rectangle(display, (x1, y1), (x2, y2), (50, 210, 60), 2)
        cv2.putText(display, str(index + 1), (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    print("Click inside the box of the player described in Stage 0.")
    selected: list[tuple[int, int]] = []
    window = "Select your player"

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while not selected:
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Player selection cancelled.")
    finally:
        cv2.destroyWindow(window)
    x, y = selected[0]
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = normalized_box_to_pixels(candidate["box"], frame.shape[1], frame.shape[0])
        if x1 <= x <= x2 and y1 <= y <= y2:
            return index
    raise ValueError("The click did not fall inside a detected player box. Rerun and click inside the box.")


def find_nearest_detection(
    candidates: list[dict[str, Any]],
    reference_histogram: np.ndarray,
    previous_position: list[float] | None,
    elapsed: float,
) -> tuple[int, float] | None:
    best: tuple[int, float, float] | None = None
    for index, candidate in enumerate(candidates):
        histogram = candidate.get("_appearance")
        if histogram is None:
            continue
        color_distance = float(cv2.compareHist(reference_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA))
        if color_distance > 0.62:
            continue
        continuity = 0.0
        if previous_position is not None:
            distance = float(np.linalg.norm(np.asarray(candidate["court"]) - np.asarray(previous_position)))
            continuity = min(1.0, distance / max(0.8, elapsed * 7.0))
        total = color_distance + 0.18 * continuity
        if best is None or total < best[1]:
            best = (index, total, color_distance)
    return (best[0], best[2]) if best is not None else None


def fix_short_gaps(
    rows: list[dict[str, Any]],
    all_candidates: list[list[dict[str, Any]]],
    reference_histogram: np.ndarray,
) -> None:
    for index, row in enumerate(rows):
        if row["r"] is not None:
            continue
        before = next((j for j in range(index - 1, max(-1, index - 6), -1) if rows[j]["r"] is not None), None)
        after = next((j for j in range(index + 1, min(len(rows), index + 6)) if rows[j]["r"] is not None), None)
        if before is None or after is None or after - before > 6:
            continue
        fraction = (index - before) / (after - before)
        expected = np.asarray(rows[before]["r"]) * (1 - fraction) + np.asarray(rows[after]["r"]) * fraction
        candidates = all_candidates[index]
        best = None
        for candidate in candidates:
            distance = float(np.linalg.norm(np.asarray(candidate["court"]) - expected))
            if distance > 0.8:
                continue
            appearance = candidate.get("_appearance")
            if appearance is None:
                continue
            color_distance = float(cv2.compareHist(reference_histogram, appearance, cv2.HISTCMP_BHATTACHARYYA))
            score = distance + color_distance
            if best is None or score < best[0]:
                best = (score, candidate, color_distance)
        if best is not None and best[2] <= 0.72:
            candidate = best[1]
            row["r"] = candidate["court"]
            row["rb"] = candidate["box"]
            row["r_color_distance"] = round(best[2], 4)
            row["r_gap_filled"] = True


def remove_spikes(rows: list[dict[str, Any]], key: str, box_key: str) -> None:
    for index in range(1, len(rows) - 1):
        previous, current, following = rows[index - 1][key], rows[index][key], rows[index + 1][key]
        if previous is None or current is None or following is None:
            continue
        neighbor_gap = float(np.linalg.norm(np.asarray(previous) - np.asarray(following)))
        previous_gap = float(np.linalg.norm(np.asarray(previous) - np.asarray(current)))
        following_gap = float(np.linalg.norm(np.asarray(following) - np.asarray(current)))
        if neighbor_gap < 0.35 and previous_gap > 0.9 and following_gap > 0.9:
            rows[index][key] = None
            rows[index][box_key] = None


def save_track_samples(video: Path, rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    sample_count = min(16, max(1, len(rows) // 50))
    indices = sorted(set(int(value) for value in np.linspace(0, len(rows) - 1, sample_count)))
    capture = open_video(video)
    saved = []
    try:
        for sample_number, index in enumerate(indices, start=1):
            row = rows[index]
            frame = frame_at(capture, float(row["t"]))
            if frame is None:
                continue
            for player, box_key, color, label in (
                ("r", "rb", (40, 40, 230), "YOU (R)"),
                ("o", "ob", (230, 160, 20), "OPPONENT (O)"),
            ):
                box = row.get(box_key)
                position = row.get(player)
                if box is None or position is None:
                    continue
                x1, y1, x2, y2 = normalized_box_to_pixels(box, frame.shape[1], frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, ((x1 + x2) // 2, y2), 6, color, -1)
                cv2.putText(
                    frame,
                    f"{label}  court {position[0]:.1f},{position[1]:.1f}m",
                    (x1, max(22, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            path = output_dir / f"track_{sample_number:02d}_{row['t']:.1f}s.jpg"
            if cv2.imwrite(str(path), frame):
                saved.append(str(path))
    finally:
        capture.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--detections", help="Detection JSON; defaults to detections.json beside job.json.")
    parser.add_argument("--calibration", help="Calibration JSON; defaults to calibration.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to players.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        detections_path = Path(args.detections).expanduser().resolve() if args.detections else artifact_path(job_path, "detections.json")
        calibration_path = Path(args.calibration).expanduser().resolve() if args.calibration else artifact_path(job_path, "calibration.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "players.json")
        detections = read_json(detections_path)
        calibration = read_json(calibration_path)
        setup = job.get("analysis_setup", {})

        sample_frames = detections.get("frames", [])
        description = str(setup.get("player_identity_description", "")).lower()
        suggested_part = "shorts" if "short" in description else "shirt"
        raw_part = input(f"Compare the clothing marker on shorts or shirt [{suggested_part}]: ").strip().lower()
        part = raw_part if raw_part in {"shorts", "shirt"} else suggested_part

        candidate_frames: list[list[dict[str, Any]]] = []
        capture = open_video(video)
        sample_every = max(1, int(detections.get("sample_every_frames", 3)))
        sample_index = 0
        frame_index = 0
        try:
            while sample_index < len(sample_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_every == 0:
                    item = sample_frames[sample_index]
                    segment = calibration_for_time(calibration, float(item["t"]))
                    if segment is None:
                        candidates = []
                    else:
                        candidates = build_candidates(item.get("p", []), frame.shape[1], frame.shape[0], segment)
                        for candidate in candidates:
                            candidate["_appearance"] = appearance_histogram(frame, candidate["box"], part)
                    candidate_frames.append(candidates)
                    sample_index += 1
                frame_index += 1
        finally:
            capture.release()
        while len(candidate_frames) < len(sample_frames):
            candidate_frames.append([])

        reference_index = next((i for i, people in enumerate(candidate_frames) if len(people) >= 2), None)
        if reference_index is None:
            raise RuntimeError("No sampled frame has two people inside the calibrated court. Check calibration and detections.")
        default_time = float(sample_frames[reference_index]["t"])
        reference_time = prompt_float(
            f"Reference timestamp with you clearly visible [{default_time:.2f}s]: ",
            minimum=0,
            default=default_time,
        )
        reference_index = min(range(len(sample_frames)), key=lambda i: abs(float(sample_frames[i]["t"]) - reference_time))
        if len(candidate_frames[reference_index]) < 2:
            raise RuntimeError("The selected reference frame has fewer than two detected court players.")
        capture = open_video(video)
        try:
            reference_frame = frame_at(capture, float(sample_frames[reference_index]["t"]))
        finally:
            capture.release()
        if reference_frame is None:
            raise RuntimeError("Could not read the selected reference frame.")
        user_index = choose_reference_person(reference_frame, candidate_frames[reference_index])

        reference_histogram = candidate_frames[reference_index][user_index].get("_appearance")
        if reference_histogram is None:
            raise RuntimeError("Could not sample the selected clothing region.")

        rows = []
        previous_position = None
        previous_time = None
        for index, item in enumerate(sample_frames):
            seconds = float(item["t"])
            candidates = candidate_frames[index]
            user_match = None
            if candidates:
                elapsed = 0.1 if previous_time is None else max(0.1, seconds - previous_time)
                user_match = find_nearest_detection(
                    candidates,
                    reference_histogram,
                    previous_position,
                    elapsed,
                )
            user = None
            user_box = None
            user_distance = None
            opponent = None
            opponent_box = None
            if user_match is not None:
                selected, user_distance = user_match
                user_candidate = candidates[selected]
                user = user_candidate["court"]
                user_box = user_candidate["box"]
                previous_position = user
                previous_time = seconds
                user_is_near = user[1] < NET_Y
                opponent_candidates = [
                    candidate
                    for candidate in candidates
                    if (candidate["court"][1] >= NET_Y) == user_is_near
                ]
                if opponent_candidates:
                    opponent_candidate = max(opponent_candidates, key=lambda candidate: candidate["confidence"])
                    opponent = opponent_candidate["court"]
                    opponent_box = opponent_candidate["box"]
            rows.append(
                {
                    "t": seconds,
                    "r": user,
                    "rb": user_box,
                    "r_color_distance": round(user_distance, 4) if user_distance is not None else None,
                    "o": opponent,
                    "ob": opponent_box,
                }
            )

        # Fill brief appearance gaps only when a nearby detection supports the interpolated position.
        if any(row["r"] is None for row in rows):
            fix_short_gaps(rows, candidate_frames, reference_histogram)
        remove_spikes(rows, "r", "rb")
        remove_spikes(rows, "o", "ob")

        write_json(
            output_path,
            {
                "schema_version": 1,
                "job": str(job_path),
                "fps": detections.get("fps", 10),
                "sample_every_frames": detections.get("sample_every_frames", 3),
                "coordinates": "court metres; r is the appearance-matched user, o is the opponent",
                "marker_description": setup.get("player_identity_description"),
                "marker_part_sampled": part,
                "reference_time": float(sample_frames[reference_index]["t"]),
                "frames": rows,
            },
        )
        review_images = save_track_samples(video, rows, artifact_path(job_path, "review/tracking"))
        identified = sum(row["r"] is not None for row in rows)
        print(f"Saved {output_path}; identified the user in {identified}/{len(rows)} sampled frames.")
        print(f"Review {len(review_images)} annotated frames under {artifact_path(job_path, 'review/tracking')} before using movement statistics.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Player tracking failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
