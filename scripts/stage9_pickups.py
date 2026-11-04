#!/usr/bin/env python3
"""Estimate shuttle landing side from slow post-rally player bends."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
HALF_LENGTH_M = 6.7
DEPTH_BIN_M = 0.75
HEIGHT_QUANTILE = 0.90
MIN_PROFILE_SAMPLES = 8


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2 or not all(finite_number(v) for v in value):
        return None
    return float(value[0]), float(value[1])


def box_height(value: Any) -> float | None:
    if not isinstance(value, list) or len(value) < 4 or not all(finite_number(v) for v in value[:4]):
        return None
    height = float(value[3]) - float(value[1])
    return height if 0.015 <= height <= 0.7 else None


def load_tracks(path: Path) -> tuple[list[float], dict[str, list[tuple[float, float] | None]], dict[str, list[float | None]], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    fps = float(data.get("fps", 0))
    if not isinstance(frames, list) or not frames or fps <= 0:
        raise ValueError("players JSON needs a non-empty frames array and positive fps")

    times: list[float] = []
    positions: dict[str, list[tuple[float, float] | None]] = {"R": [], "O": []}
    heights: dict[str, list[float | None]] = {"R": [], "O": []}
    for frame in frames:
        if not finite_number(frame.get("t")):
            raise ValueError("every players frame needs a finite timestamp")
        times.append(float(frame["t"]))
        for player, pos_key, box_key in (("R", "r", "rb"), ("O", "o", "ob")):
            point = point2(frame.get(pos_key))
            if point is None:
                positions[player].append(None)
                heights[player].append(None)
                continue
            if point[1] >= HALF_LENGTH_M:
                # Rotate whichever identity is on the far half to its own-court coordinates.
                point = (COURT_WIDTH_M - point[0], COURT_LENGTH_M - point[1])
            positions[player].append(point)
            heights[player].append(box_height(frame.get(box_key)))

    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("players frame timestamps must increase")
    return times, positions, heights, fps


def load_rallies(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("rallies review JSON needs a non-empty points array")

    included: list[dict[str, Any]] = []
    for point in points:
        if point.get("dummy_point") is True:
            continue
        number = point.get("point_n")
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError("each rally review point needs an integer point_n")
        for key in ("serve_t", "rally_end_sec"):
            if not finite_number(point.get(key)):
                raise ValueError(f"s{number} needs a finite {key}; complete Stage 6 first")
        serve_t = float(point["serve_t"])
        end_t = float(point["rally_end_sec"])
        if end_t < serve_t:
            raise ValueError(f"s{number} rally_end_sec precedes serve_t")
        next_t = point.get("next_serve_t")
        if next_t is not None and (not finite_number(next_t) or float(next_t) <= end_t):
            raise ValueError(f"s{number} next_serve_t must follow rally_end_sec")
        included.append(point)

    included.sort(key=lambda item: int(item["point_n"]))
    if len({int(point["point_n"]) for point in included}) != len(included):
        raise ValueError("rally review contains duplicate point_n values")
    return included


def smooth_positions(
    positions: list[tuple[float, float] | None], window_frames: int
) -> list[tuple[float, float] | None]:
    if window_frames % 2 == 0:
        window_frames += 1
    radius = window_frames // 2
    smoothed: list[tuple[float, float] | None] = []
    for index in range(len(positions)):
        values = [
            value for value in positions[max(0, index - radius):min(len(positions), index + radius + 1)]
            if value is not None
        ]
        if len(values) < max(1, math.ceil(window_frames / 2)):
            smoothed.append(None)
        else:
            median = np.median(np.asarray(values, dtype=np.float64), axis=0)
            smoothed.append((float(median[0]), float(median[1])))
    return smoothed


def estimate_speeds(
    positions: list[tuple[float, float] | None], times: list[float]
) -> list[float | None]:
    speeds: list[float | None] = [None] * len(positions)
    for index in range(len(positions)):
        previous = index - 1 if index > 0 else None
        following = index + 1 if index + 1 < len(positions) else None
        if previous is not None and following is not None:
            a, b = positions[previous], positions[following]
            dt = times[following] - times[previous]
        elif following is not None:
            a, b = positions[index], positions[following]
            dt = times[following] - times[index]
        elif previous is not None:
            a, b = positions[previous], positions[index]
            dt = times[index] - times[previous]
        else:
            continue
        if a is None or b is None or dt <= 0 or dt > 0.35:
            continue
        speed = math.dist(a, b) / dt
        if speed <= 8.0:
            speeds[index] = speed
    return speeds


def smooth_heights(values: list[float | None], window_frames: int = 3) -> list[float | None]:
    radius = window_frames // 2
    output: list[float | None] = []
    for index in range(len(values)):
        observed = [
            value for value in values[max(0, index - radius):min(len(values), index + radius + 1)]
            if value is not None
        ]
        output.append(float(np.median(observed)) if observed else None)
    return output


def fit_standing_height_profile(
    positions: list[tuple[float, float] | None], heights: list[float | None]
) -> tuple[list[float], list[float], list[int]]:
    edges = np.arange(0.0, HALF_LENGTH_M + DEPTH_BIN_M, DEPTH_BIN_M)
    if edges[-1] < HALF_LENGTH_M:
        edges = np.append(edges, HALF_LENGTH_M)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    quantiles: list[float | None] = []
    counts: list[int] = []
    for low, high in zip(edges[:-1], edges[1:]):
        values = [
            height for position, height in zip(positions, heights)
            if position is not None and height is not None and low <= position[1] < high
        ]
        counts.append(len(values))
        quantiles.append(float(np.quantile(values, HEIGHT_QUANTILE)) if len(values) >= MIN_PROFILE_SAMPLES else None)

    valid = [index for index, value in enumerate(quantiles) if value is not None]
    if not valid:
        raise ValueError("not enough valid player boxes to fit a standing-height perspective profile")
    valid_centers = [centers[index] for index in valid]
    valid_heights = [float(quantiles[index]) for index in valid]
    profile = np.interp(centers, valid_centers, valid_heights).tolist()
    if len(profile) >= 3:
        profile = [
            float(np.median(profile[max(0, i - 1):min(len(profile), i + 2)]))
            for i in range(len(profile))
        ]
    return centers, profile, counts


def first_bend_in_window(
    start_t: float,
    end_t: float,
    times: list[float],
    local_positions: list[tuple[float, float] | None],
    heights: list[float | None],
    expected_heights: list[float | None],
    speeds: list[float | None],
    bend_ratio: float,
    slow_speed: float,
    min_duration: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ratios: list[float | None] = [
        height / expected if height is not None and expected is not None and expected > 0 else None
        for height, expected in zip(heights, expected_heights)
    ]
    candidates: list[int] = []
    best_index: int | None = None
    for index, timestamp in enumerate(times):
        if not start_t <= timestamp <= end_t or local_positions[index] is None:
            continue
        ratio, speed = ratios[index], speeds[index]
        if ratio is None or speed is None:
            continue
        if speed <= max(1.2, slow_speed * 1.5) and (best_index is None or ratio < ratios[best_index]):
            best_index = index
        if ratio < bend_ratio and speed <= slow_speed:
            candidates.append(index)

    best = None
    if best_index is not None:
        best = {
            "frame_index": best_index,
            "t": round(times[best_index], 3),
            "height_ratio": round(float(ratios[best_index]), 3),
            "speed_m_s": round(float(speeds[best_index]), 3),
        }

    run: list[int] = []
    runs: list[list[int]] = []
    max_gap = max(0.25, 1.5 * (times[1] - times[0] if len(times) > 1 else 0.1))
    for index in candidates:
        if run and times[index] - times[run[-1]] > max_gap:
            runs.append(run)
            run = []
        run.append(index)
    if run:
        runs.append(run)

    sustained = [
        run for run in runs
        if times[run[-1]] - times[run[0]] + 1e-6 >= min_duration
    ]
    if not sustained:
        return None, best
    chosen = sustained[0]
    first_index = chosen[0]
    deepest_index = min(chosen, key=lambda index: ratios[index] if ratios[index] is not None else 1.0)
    bend = {
        "start_t": round(times[first_index], 3),
        "end_t": round(times[chosen[-1]], 3),
        "duration_sec": round(times[chosen[-1]] - times[first_index], 3),
        "height_ratio_min": round(float(ratios[deepest_index]), 3),
        "speed_m_s_at_start": round(float(speeds[first_index]), 3),
        "frame_index": deepest_index,
    }
    return bend, best


def choose_pickup(
    candidates: dict[str, dict[str, Any] | None],
    tie_tolerance: float = 0.3,
) -> tuple[str | None, str, str]:
    found = [(player, item) for player, item in candidates.items() if item is not None]
    if not found:
        return None, "not_found", "none"
    found.sort(key=lambda item: item[1]["start_t"])
    if len(found) > 1 and found[1][1]["start_t"] - found[0][1]["start_t"] <= tie_tolerance:
        return None, "ambiguous", found[0][0]
    player, bend = found[0]
    ratio = bend["height_ratio_min"]
    duration = bend["duration_sec"]
    confidence = "high" if ratio <= 0.60 and duration + 1e-6 >= 0.4 else "medium"
    return player, "detected", confidence


def normalized_box_crop(frame: np.ndarray, box: Any, side_scale: float = 2.5) -> np.ndarray | None:
    if not isinstance(box, list) or len(box) < 4 or not all(finite_number(v) for v in box[:4]):
        return None
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    height, width = frame.shape[:2]
    side = max(32, int(math.ceil(max((x2 - x1) * width, (y2 - y1) * height) * side_scale)))
    center_x, center_y = (x1 + x2) * width / 2, (y1 + y2) * height / 2
    left, top = round(center_x - side / 2), round(center_y - side / 2)
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(width, left + side), min(height, top + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop[sy0 - top:sy0 - top + sy1 - sy0, sx0 - left:sx0 - left + sx1 - sx0] = frame[sy0:sy1, sx0:sx1]
    return crop


def overlay_box(image: np.ndarray, box: Any, label: str, color: tuple[int, int, int]) -> None:
    if not isinstance(box, list) or len(box) < 4:
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    p1 = (round(x1 * width), round(y1 * height))
    p2 = (round(x2 * width), round(y2 * height))
    cv2.rectangle(image, p1, p2, color, 2, cv2.LINE_AA)
    cv2.putText(image, label, (max(2, p1[0]), max(16, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def write_review_sheet(
    video_path: Path,
    frames: list[dict[str, Any]],
    times: list[float],
    events: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video for pickup review: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        cap.release()
        raise RuntimeError("Could not read a valid frame rate for pickup review frames")
    cache: dict[int, np.ndarray | None] = {}

    def frame_at(index: int) -> np.ndarray | None:
        if index not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(times[index] * source_fps))
            ok, image = cap.read()
            cache[index] = image if ok else None
        return cache[index]

    columns = 4
    tile_width, tile_height = 500, 220
    full_width, full_height, crop_side = 320, 180, 160
    rows = math.ceil(len(events) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 32, dtype=np.uint8)
    try:
        for ordinal, event in enumerate(events):
            row, column = divmod(ordinal, columns)
            x0, y0 = column * tile_width, row * tile_height
            title = (
                f"s{event['point_n']:02d}  {event['status']}  pickup={event['pickup_player'] or '?'} "
                f"side={event['landing_side'] or '?'}  conf={event['confidence']}"
            )
            cv2.putText(sheet, title, (x0 + 5, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            index = event.get("review_frame_index")
            if not isinstance(index, int) or index < 0 or index >= len(frames):
                continue
            image = frame_at(index)
            if image is None:
                continue
            thumb = cv2.resize(image, (full_width, full_height), interpolation=cv2.INTER_AREA)
            source = frames[index]
            chosen = event.get("review_player")
            for player, box_key, color in (("R", "rb", (0, 0, 255)), ("O", "ob", (255, 100, 0))):
                box = source.get(box_key)
                overlay_box(thumb, box, player, (0, 255, 255) if player == chosen else color)
            sheet[y0 + 30:y0 + 30 + full_height, x0:x0 + full_width] = thumb

            player = event.get("review_player")
            box_key = "rb" if player == "R" else "ob" if player == "O" else None
            crop = normalized_box_crop(image, source.get(box_key) if box_key else None)
            if crop is not None:
                crop = cv2.resize(crop, (crop_side, crop_side), interpolation=cv2.INTER_AREA)
                sheet[y0 + 30:y0 + 30 + crop_side, x0 + full_width + 10:x0 + full_width + 10 + crop_side] = crop
                cv2.putText(sheet, f"{player} close-up", (x0 + full_width + 10, y0 + tile_height - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            ratio = event.get("review_height_ratio")
            time_text = "no sample" if event.get("review_time_sec") is None else f"t={event['review_time_sec']:.1f}s"
            ratio_text = "" if ratio is None else f"  height ratio={ratio:.2f}"
            cv2.putText(sheet, time_text + ratio_text, (x0 + 5, y0 + tile_height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    finally:
        cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write pickup review sheet: {output_path}")


def analyze(
    points: list[dict[str, Any]],
    times: list[float],
    positions: dict[str, list[tuple[float, float] | None]],
    raw_heights: dict[str, list[float | None]],
    fps: float,
    bend_ratio: float,
    slow_speed: float,
    min_bend_duration: float,
    max_search_sec: float,
    next_serve_margin_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    all_speeds: dict[str, list[float | None]] = {}
    all_smooth_heights: dict[str, list[float | None]] = {}
    all_expected_heights: dict[str, list[float | None]] = {}
    for player in ("R", "O"):
        centers, profile, counts = fit_standing_height_profile(positions[player], raw_heights[player])
        smooth_h = smooth_heights(raw_heights[player])
        local_y = [point[1] if point is not None else None for point in positions[player]]
        expected = [
            float(np.interp(y, centers, profile)) if y is not None else None
            for y in local_y
        ]
        smoothed_positions = smooth_positions(positions[player], max(3, round(0.7 * fps)))
        speeds = estimate_speeds(smoothed_positions, times)
        all_speeds[player] = speeds
        profiles[player] = {
            "height_fraction_quantile": HEIGHT_QUANTILE,
            "minimum_samples_per_bin": MIN_PROFILE_SAMPLES,
            "bins": [
                {
                    "local_depth_m": round(center, 3),
                    "expected_standing_box_height_fraction": round(height, 5),
                    "sample_count": count,
                }
                for center, height, count in zip(centers, profile, counts)
            ],
        }
        all_smooth_heights[player] = smooth_h
        all_expected_heights[player] = expected

    results: list[dict[str, Any]] = []
    for point in points:
        rally_end = float(point["rally_end_sec"])
        next_serve = point.get("next_serve_t")
        window_start = rally_end
        window_end = min(rally_end + max_search_sec, times[-1])
        if next_serve is not None:
            window_end = min(window_end, float(next_serve) - next_serve_margin_sec)

        player_bends: dict[str, dict[str, Any] | None] = {}
        best_samples: dict[str, dict[str, Any] | None] = {}
        for player in ("R", "O"):
            bend, best = first_bend_in_window(
                window_start,
                window_end,
                times,
                positions[player],
                all_smooth_heights[player],
                all_expected_heights[player],
                all_speeds[player],
                bend_ratio,
                slow_speed,
                min_bend_duration,
            )
            player_bends[player] = bend
            best_samples[player] = best

        pickup_player, status, confidence = choose_pickup(player_bends)
        if point.get("rally_end_confidence") == "low" and confidence == "high":
            confidence = "medium"
        review_player = pickup_player
        if review_player is None:
            usable = [(player, sample) for player, sample in best_samples.items() if sample is not None]
            if usable:
                review_player, _ = min(usable, key=lambda item: item[1]["height_ratio"])
        if pickup_player is not None:
            selected_bend = player_bends[pickup_player]
            review_index = selected_bend["frame_index"]
            review_time = times[review_index]
            review_ratio = selected_bend["height_ratio_min"]
            landing_side = pickup_player
        elif status == "ambiguous":
            earliest = min(
                ((player, bend) for player, bend in player_bends.items() if bend is not None),
                key=lambda item: item[1]["start_t"],
            )
            review_player = earliest[0]
            review_index = earliest[1]["frame_index"]
            review_time = times[review_index]
            review_ratio = earliest[1]["height_ratio_min"]
            landing_side = None
        else:
            candidate = best_samples.get(review_player) if review_player else None
            if candidate is not None:
                review_index = candidate["frame_index"]
                review_time = times[review_index]
                review_ratio = candidate["height_ratio"]
            else:
                review_index = min(range(len(times)), key=lambda index: abs(times[index] - rally_end))
                review_time = times[review_index]
                review_ratio = None
            landing_side = None

        winner = point.get("winner") if point.get("winner") in {"R", "O"} else None
        relation = None
        if landing_side is not None and winner is not None:
            relation = "winner_side" if landing_side == winner else "losing_player_side"
        result = {
            "point_n": int(point["point_n"]),
            "candidate_n": point.get("candidate_n"),
            "winner": winner,
            "score_before": point.get("score_before"),
            "score_after": point.get("score_after"),
            "serve_t": round(float(point["serve_t"]), 3),
            "rally_end_sec": round(rally_end, 3),
            "rally_end_confidence": point.get("rally_end_confidence"),
            "pickup_search_window_sec": [round(window_start, 3), round(max(window_start, window_end), 3)],
            "player_bends": player_bends,
            "status": status,
            "confidence": confidence,
            "pickup_time_sec": round(player_bends[pickup_player]["start_t"], 3) if pickup_player else None,
            "pickup_player": pickup_player,
            "landing_side": landing_side,
            "landing_side_label": {"R": "R analyzed player side", "O": "O opponent side"}.get(landing_side),
            "relation_to_winner_side": relation,
            "review_player": review_player,
            "review_time_sec": round(float(review_time), 3),
            "review_height_ratio": review_ratio,
            "review_frame_index": int(review_index),
        }
        results.append(result)

    method = {
        "description": (
            "For each player's local-court depth, fit the 90th percentile standing box height. "
            "A bend is a box-height ratio below the configured threshold while moving slowly for the minimum duration. "
            "The first unambiguous post-rally bend estimates the shuttle landing side."
        ),
        "bend_box_height_ratio_below": bend_ratio,
        "slow_speed_m_s_at_most": slow_speed,
        "minimum_bend_duration_sec": min_bend_duration,
        "maximum_search_after_rally_end_sec": max_search_sec,
        "next_serve_margin_sec": next_serve_margin_sec,
        "profiles_by_player": profiles,
    }
    return results, method


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate shuttle landing side from post-rally player bends and review frames."
    )
    parser.add_argument("input_video", type=Path, help="Stage 0 proxy video")
    parser.add_argument("players_json", type=Path, help="Stage 3 players.json with court positions and boxes")
    parser.add_argument("rallies_review_json", type=Path, help="Stage 6 rallies_review.json")
    parser.add_argument("output_json", type=Path, help="output pickups.json")
    parser.add_argument("review_image", type=Path, help="output pickup_review.jpg contact sheet")
    parser.add_argument("--bend-ratio", type=float, default=0.75, help="bend if box is below this share of standing height")
    parser.add_argument("--slow-speed", type=float, default=0.8, help="maximum movement speed for a bend (m/s)")
    parser.add_argument("--min-bend-duration", type=float, default=0.1, help="minimum sustained bend duration in seconds")
    parser.add_argument("--max-search-sec", type=float, default=12.0, help="maximum seconds after rally end to search")
    parser.add_argument("--next-serve-margin-sec", type=float, default=0.25, help="stop searching before next serve time")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args()

    video_path = args.input_video.expanduser().resolve()
    players_path = args.players_json.expanduser().resolve()
    rallies_path = args.rallies_review_json.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    review_path = args.review_image.expanduser().resolve()
    for path in (video_path, players_path, rallies_path):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if output_path == review_path or output_path in (video_path, players_path, rallies_path) or review_path in (video_path, players_path, rallies_path):
        parser.error("input and output paths must be different")
    if not 0 < args.bend_ratio < 1 or args.slow_speed <= 0 or args.min_bend_duration <= 0:
        parser.error("bend ratio must be between 0 and 1; speed and bend duration must be positive")
    if args.max_search_sec <= 0 or args.next_serve_margin_sec < 0:
        parser.error("search duration must be positive and next-serve margin must be zero or greater")
    if (output_path.exists() or review_path.exists()) and not args.overwrite:
        parser.error("an output already exists (pass --overwrite to replace it)")

    try:
        times, positions, raw_heights, fps = load_tracks(players_path)
        points = load_rallies(rallies_path)
        results, method = analyze(
            points,
            times,
            positions,
            raw_heights,
            fps,
            args.bend_ratio,
            args.slow_speed,
            args.min_bend_duration,
            args.max_search_sec,
            args.next_serve_margin_sec,
        )
        write_review_sheet(video_path, json.loads(players_path.read_text(encoding="utf-8"))["frames"], times, results, review_path)

        side_counts = {"R": 0, "O": 0, "unknown": 0}
        status_counts = {"detected": 0, "ambiguous": 0, "not_found": 0}
        for event in results:
            side_counts[event["landing_side"] or "unknown"] += 1
            status_counts[event["status"]] += 1
        payload = {
            "source_video": str(video_path),
            "source_players_json": str(players_path),
            "source_rallies_review_json": str(rallies_path),
            "player_labels": {"R": "Stage 3 selected player (you)", "O": "opponent"},
            "method": method,
            "point_count": len(results),
            "landing_side_counts": side_counts,
            "status_counts": status_counts,
            "review_image": review_path.name,
            "points": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Points analyzed: {len(results)} (dummy score rows are absent from Stage 6 review).")
        print(
            f"Pickup evidence: {status_counts['detected']} detected, {status_counts['ambiguous']} ambiguous, "
            f"{status_counts['not_found']} not found. Landing side R={side_counts['R']}, "
            f"O={side_counts['O']}, unknown={side_counts['unknown']}."
        )
        print(f"JSON: {output_path}")
        print(f"Visual review: {review_path}")
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error, json.JSONDecodeError) as error:
        print(f"Stage 9 failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
