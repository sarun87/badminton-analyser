#!/usr/bin/env python3
"""Create picture-only rally review sheets between successive recorded serves."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from html import escape
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shuttle_tracking import (
    load_shuttle_tracks,
    samples_between,
    summarize_window,
    validate_shuttle_video,
)


TILE_WIDTH = 320
TILE_HEIGHT = 180
COLUMNS = 4
ROWS_PER_PAGE = 4
FRAMES_PER_PAGE = COLUMNS * ROWS_PER_PAGE


def load_player_motion(path: Path) -> tuple[list[float], list[float | None], float]:
    """Build a smoothed two-player motion signal from Stage 3 frame samples."""
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    fps = float(data.get("fps", 0))
    if not isinstance(frames, list) or not frames or fps <= 0:
        raise ValueError("players JSON needs frames and a positive fps")

    times = [float(frame["t"]) for frame in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("players frame timestamps must increase")

    tracks: dict[str, list[tuple[float, float] | None]] = {"r": [], "o": []}
    for frame in frames:
        for key in ("r", "o"):
            point = frame.get(key)
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)
            ):
                tracks[key].append(None)
            else:
                tracks[key].append((float(point[0]), float(point[1])))

    smooth_window = max(1, round(0.7 * fps))
    if smooth_window % 2 == 0:
        smooth_window += 1
    radius = smooth_window // 2
    smoothed: dict[str, list[tuple[float, float] | None]] = {}
    for key, track in tracks.items():
        values = []
        for index in range(len(track)):
            nearby = [
                point for point in track[max(0, index - radius):min(len(track), index + radius + 1)]
                if point is not None
            ]
            if len(nearby) < max(1, math.ceil(smooth_window / 2)):
                values.append(None)
            else:
                median = np.median(np.asarray(nearby, dtype=np.float64), axis=0)
                values.append((float(median[0]), float(median[1])))
        smoothed[key] = values

    motion_window = max(1, round(0.5 * fps))
    motion = []
    for index, timestamp in enumerate(times):
        speeds = []
        earlier = index - motion_window
        if earlier >= 0:
            elapsed = timestamp - times[earlier]
            if 0 < elapsed <= 0.8:
                for key in ("r", "o"):
                    before, after = smoothed[key][earlier], smoothed[key][index]
                    if before is not None and after is not None:
                        speed = math.dist(before, after) / elapsed
                        if speed <= 8.0:  # Ignore obvious player-assignment jumps.
                            speeds.append(speed)
        motion.append(max(speeds) if speeds else None)

    # Median filtering suppresses isolated box jumps while preserving brief lunges.
    filtered = []
    for index in range(len(motion)):
        nearby = [
            value for value in motion[max(0, index - 1):min(len(motion), index + 2)]
            if value is not None
        ]
        filtered.append(float(np.median(nearby)) if nearby else None)
    return times, filtered, fps


def estimate_missing_serve_times(points: list[dict[str, Any]]) -> None:
    """Place undetected score rows evenly inside their bounded shared interval."""
    index = 0
    while index < len(points):
        if points[index].get("serve_t") is not None:
            index += 1
            continue
        first_missing = index
        while index < len(points) and points[index].get("serve_t") is None:
            index += 1
        left = float(points[first_missing - 1]["serve_t"]) if first_missing else None
        right = float(points[index]["serve_t"]) if index < len(points) else None
        if left is None or right is None or right <= left:
            labels = ", ".join(f"s{point['point_n']}" for point in points[first_missing:index])
            raise ValueError(
                f"cannot estimate serve time(s) for {labels}; "
                "each missing serve needs timed points on both sides"
            )
        count = index - first_missing
        interval = (right - left) / (count + 1)
        for offset, point in enumerate(points[first_missing:index], start=1):
            point["serve_t"] = round(left + interval * offset, 3)
            point["serve_time_estimated"] = True
            point["serve_time_method"] = "midpoint_of_bounded_shared_interval"


def estimate_rally_ends(
    points: list[dict[str, Any]],
    motion_times: list[float],
    motion: list[float | None],
    motion_fps: float,
    video_duration: float,
    pause_speed_m_s: float = 0.8,
    active_speed_m_s: float = 1.0,
    pause_duration_sec: float = 1.5,
) -> None:
    """Estimate each rally tail at the first sustained pause after tracked movement."""
    for index, point in enumerate(points):
        next_start = (
            float(points[index + 1]["serve_t"])
            if index + 1 < len(points)
            else video_duration
        )
        point["next_serve_t"] = round(next_start, 3) if index + 1 < len(points) else None
        start = float(point["serve_t"])
        last_allowed = min(next_start - 0.2, start + 60.0)
        if last_allowed <= start:
            raise ValueError(f"s{point['point_n']} has no room before the next serve")

        candidates = [
            i for i, timestamp in enumerate(motion_times)
            if start + 0.3 <= timestamp < last_allowed
        ]
        if not candidates:
            point["rally_end_sec"] = round(min(start + 1.5, last_allowed), 3)
            point["rally_end_method"] = "short_duration_fallback_no_track_frames"
            point["rally_end_confidence"] = "low"
            point["rally_end_estimated"] = True
            point["rally_duration_sec"] = round(point["rally_end_sec"] - start, 3)
            continue

        # Do not treat a brief still moment just after the serve as a rally end.
        earliest_pause = start + 2.0
        pause_frames = max(2, round(pause_duration_sec * motion_fps))
        required_known = max(2, math.ceil(pause_frames * 0.7))
        selected: tuple[float, str, str] | None = None

        for frame_index in candidates:
            pause_start = motion_times[frame_index]
            if pause_start < earliest_pause:
                continue
            history_start = pause_start - 1.5
            history_first = bisect_left(motion_times, history_start)
            history_last = bisect_left(motion_times, pause_start)
            recent_active = [
                value for value in motion[history_first:history_last] if value is not None
            ]
            if sum(value >= active_speed_m_s for value in recent_active) < 2:
                continue

            pause_end = pause_start + pause_duration_sec
            quiet_first = bisect_left(motion_times, pause_start)
            quiet_last = bisect_left(motion_times, pause_end)
            quiet_values = [
                value for value in motion[quiet_first:quiet_last] if value is not None
            ]
            if len(quiet_values) < required_known:
                continue
            quiet_share = sum(value <= pause_speed_m_s for value in quiet_values) / len(quiet_values)
            if quiet_share >= 0.85:
                duration = pause_start - start
                confidence = "low" if duration < 3.0 or duration > 30.0 else "medium"
                selected = (pause_start, "first_sustained_pause", confidence)
                break

        if selected is None:
            # If tracking never settles, use the last active frame in a bounded interval.
            active_times = [
                timestamp for timestamp, value in zip(motion_times, motion)
                if start + 1.0 <= timestamp < last_allowed
                and value is not None and value >= active_speed_m_s
            ]
            if active_times:
                end = active_times[-1]
                method, confidence = "last_active_frame_fallback", "low"
            else:
                end = min(start + 1.5, last_allowed)
                method, confidence = "short_duration_fallback_no_active_track", "low"
        else:
            end, method, confidence = selected

        end = min(max(start, end), last_allowed)
        point["rally_end_sec"] = round(end, 3)
        point["rally_end_method"] = method
        point["rally_end_confidence"] = confidence
        point["rally_end_estimated"] = True
        point["rally_duration_sec"] = round(end - start, 3)


def add_shuttle_evidence(points: list[dict[str, Any]], tracks: dict[str, Any]) -> None:
    """Attach raw-track coverage and a terminal-time comparison to each review row."""
    for point in points:
        start = point.get("serve_t")
        end = point.get("rally_end_sec")
        if not isinstance(start, (int, float)):
            point["tracknet_shuttle_evidence"] = {"status": "serve_time_unavailable"}
            continue
        interval_end = float(end) if isinstance(end, (int, float)) else float(point.get("review_end_t", start))
        summary = summarize_window(tracks, float(start), interval_end)
        last_t = summary.get("last_main_court_t")
        delta = round(float(last_t) - float(end), 3) if last_t is not None and isinstance(end, (int, float)) else None
        summary.update({
            "status": "raw_track_summary",
            "rally_end_delta_sec": delta,
            "supports_player_pause_estimate": abs(delta) <= 1.0 if delta is not None else None,
            "note": "Raw TrackNet observations are supporting evidence; gaps and adjacent-court selections can make the last point early or wrong.",
        })
        point["tracknet_shuttle_evidence"] = summary


def refine_rally_ends_with_shuttle(
    points: list[dict[str, Any]],
    tracks: dict[str, Any],
    search_before_sec: float = 1.5,
    search_after_sec: float = 0.35,
    min_samples: int = 3,
) -> tuple[int, int]:
    """Refine movement ends with a sustained main-court TrackNet cluster near the pause."""
    applied = 0
    fallback = 0
    fps = float(tracks["fps"])
    maximum_gap = max(0.20, 2.5 / fps)
    last_track_time = float(tracks["frames"][-1]["t"]) if tracks["frames"] else 0.0

    for point in points:
        movement_end = point.get("rally_end_sec")
        start = point.get("serve_t")
        if not isinstance(movement_end, (int, float)) or not isinstance(start, (int, float)):
            point["tracknet_rally_end_refinement"] = {"status": "movement_estimate_unavailable"}
            fallback += 1
            continue

        next_serve = point.get("next_serve_t")
        lower = max(float(start) + 0.8, float(movement_end) - search_before_sec)
        upper = min(
            float(movement_end) + search_after_sec,
            float(next_serve) - 0.2 if isinstance(next_serve, (int, float)) else last_track_time,
            last_track_time,
        )
        rows = [
            row for row in samples_between(tracks, lower, upper)
            if row.get("visible") and row.get("court_region") in ("inside", "near_boundary")
        ]

        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in rows:
            if current and float(row["t"]) - float(current[-1]["t"]) > maximum_gap:
                clusters.append(current)
                current = []
            current.append(row)
        if current:
            clusters.append(current)
        clusters = [
            cluster for cluster in clusters
            if len(cluster) >= min_samples
            and float(cluster[-1]["t"]) - float(cluster[0]["t"]) >= 0.05
        ]

        if not clusters:
            point["tracknet_rally_end_refinement"] = {
                "status": "insufficient_sustained_main_court_track_near_movement_end",
                "movement_end_sec": round(float(movement_end), 3),
                "search_window_sec": [round(lower, 3), round(upper, 3)],
                "fallback": "movement estimate retained",
            }
            fallback += 1
            continue

        cluster = min(
            clusters,
            key=lambda items: abs(float(np.median([float(row["t"]) for row in items])) - float(movement_end)),
        )
        selected = min(cluster, key=lambda row: abs(float(row["t"]) - float(movement_end)))
        tracknet_end = float(selected["t"])
        previous_end = float(movement_end)
        point["movement_rally_end_sec"] = round(previous_end, 3)
        point["rally_end_sec"] = round(tracknet_end, 3)
        point["rally_duration_sec"] = round(tracknet_end - float(start), 3)
        point["rally_end_method"] = "tracknet_cluster_near_movement_pause"
        point["rally_end_confidence"] = "medium" if len(cluster) >= 5 else "low"
        point["rally_end_estimated"] = True
        point["tracknet_rally_end_refinement"] = {
            "status": "applied",
            "movement_end_sec": round(previous_end, 3),
            "tracknet_end_sec": round(tracknet_end, 3),
            "delta_from_movement_sec": round(tracknet_end - previous_end, 3),
            "cluster_samples": len(cluster),
            "cluster_span_sec": round(float(cluster[-1]["t"]) - float(cluster[0]["t"]), 3),
            "method": "nearest raw TrackNet observation in a sustained calibrated main-court cluster near the movement pause",
        }
        applied += 1

    return applied, fallback


def load_video_points(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("points JSON must contain a non-empty points array")
    actual = [point for point in points if not point.get("dummy_point", False)]
    if not actual:
        raise ValueError("points JSON contains no recorded serves")
    timestamps = []
    for point in actual:
        t = point.get("t")
        if t is None:
            continue
        if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0:
            raise ValueError(f"recorded serve s{point.get('n')} has an invalid timestamp")
        timestamps.append(float(t))
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("known serve timestamps must increase; dummy points are excluded")
    expected = data.get("serve_count")
    if expected is not None and expected != len(actual):
        raise ValueError(f"points JSON says {expected} recorded serves but contains {len(actual)}")
    return actual


def sample_times(start: float, end: float, interval: float) -> list[float]:
    times = []
    index = 0
    while start + index * interval < end:
        times.append(start + index * interval)
        index += 1
    if not times:
        times.append(start)
    return times


def read_frame(cap: cv2.VideoCapture, timestamp: float, fps: float, frame_count: int) -> np.ndarray | None:
    frame_index = min(max(0, round(timestamp * fps)), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)


def put_text(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.48) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def make_point_pages(
    cap: cv2.VideoCapture,
    point: dict[str, Any],
    next_point: dict[str, Any] | None,
    fps: float,
    frame_count: int,
    video_duration: float,
    interval: float,
    review_dir: Path,
    relative_to: Path,
    label_override: str | None = None,
    stem_override: str | None = None,
) -> tuple[list[str], int]:
    start_t = float(point["t"])
    end_t = float(next_point["t"]) if next_point else video_duration
    if end_t <= start_t:
        raise ValueError(f"s{point['n']} review interval is empty or reversed")
    times = sample_times(start_t, end_t, interval)
    tiles: list[tuple[float, np.ndarray | None]] = [
        (timestamp, read_frame(cap, timestamp, fps, frame_count))
        for timestamp in times
    ]

    page_paths: list[str] = []
    page_count = math.ceil(len(tiles) / FRAMES_PER_PAGE)
    for page_index in range(page_count):
        page_tiles = tiles[page_index * FRAMES_PER_PAGE : (page_index + 1) * FRAMES_PER_PAGE]
        title_height = 48
        tile_label_height = 23
        page_rows = math.ceil(len(page_tiles) / COLUMNS)
        sheet = np.full(
            (title_height + page_rows * (TILE_HEIGHT + tile_label_height), COLUMNS * TILE_WIDTH, 3),
            36,
            dtype=np.uint8,
        )
        score = point.get("before", "?")
        server = point.get("server", "?")
        candidate = point.get("candidate_n", "?")
        winner = point.get("winner") or "?"
        next_label = f"{end_t:.1f}s" if next_point else f"video end {video_duration:.1f}s"
        title = label_override or (
            f"s{point['n']:02d}  candidate {candidate}  {server} serve  "
            f"score {score}  winner {winner}  interval to {next_label}"
        )
        put_text(sheet, title, 8, 20, 0.50)
        put_text(
            sheet,
            f"Page {page_index + 1}/{page_count}; inspect the last live-play frame before the next serve.",
            8,
            40,
            0.43,
        )

        for tile_index, (timestamp, frame) in enumerate(page_tiles):
            row, col = divmod(tile_index, COLUMNS)
            x = col * TILE_WIDTH
            y = title_height + row * (TILE_HEIGHT + tile_label_height)
            if frame is None:
                put_text(sheet, "Frame unavailable", x + 8, y + 24)
            else:
                sheet[y : y + TILE_HEIGHT, x : x + TILE_WIDTH] = frame
            put_text(sheet, f"t={timestamp:.2f}s", x + 6, y + TILE_HEIGHT + 17, 0.42)

        stem = stem_override or f"point_{int(point['n']):03d}"
        output = review_dir / f"{stem}_page_{page_index + 1:02d}.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise RuntimeError(f"could not write contact sheet {output}")
        page_paths.append(os.path.relpath(output.resolve(), relative_to.resolve()))
    return page_paths, len(tiles)


def write_index(review_dir: Path, points: list[dict[str, Any]]) -> Path:
    sections = []
    for point in points:
        thumbs = []
        for sheet in point["contact_sheets"]:
            filename = Path(sheet).name
            safe_name = escape(filename, quote=True)
            thumbs.append(
                f'<a href="{safe_name}" target="_blank">'
                f'<img src="{safe_name}" loading="lazy" alt="{safe_name}"></a>'
            )
        time_label = "serve time unresolved" if point["serve_t"] is None else f"serves at {point['serve_t']:.1f}s"
        if point.get("serve_time_estimated"):
            time_label += " (estimated)"
        end_label = (
            f" | rally ends ~{point['rally_end_sec']:.1f}s"
            if point.get("rally_end_estimated") else ""
        )
        if point.get("rally_end_method") == "tracknet_cluster_near_movement_pause":
            end_label += " (TrackNet refined)"
        shuttle = point.get("tracknet_shuttle_evidence") or {}
        shuttle_label = (
            f" | TrackNet {shuttle['main_court_candidate_count']} court samples; "
            f"last {shuttle['last_main_court_t']:.1f}s"
            if isinstance(shuttle.get("last_main_court_t"), (int, float)) else ""
        )
        title = (
            f"s{point['point_n']:02d} | candidate {point['candidate_n']} | "
            f"{point['server']} {time_label} | "
            f"{point['score_before']} → {point['score_after']}{end_label}{shuttle_label}"
        )
        sections.append(
            f"<details><summary>{escape(title)}</summary>"
            f"<div class=\"pages\">{''.join(thumbs)}</div></details>"
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Badminton rally contact sheets</title>
<style>
body{font:16px system-ui,sans-serif;margin:1.5rem;background:#f4f5f6;color:#222}
details{margin:0 0 .8rem;padding:.8rem;background:white;border:1px solid #ccc;border-radius:6px}
summary{cursor:pointer;font-weight:650}.pages{display:flex;flex-wrap:wrap;gap:.7rem;margin-top:.8rem}
img{width:min(480px,92vw);height:auto;border:1px solid #aaa}
</style><h1>Rally contact sheets</h1>
<p>End times marked with ~ are movement-based estimates. Check the pictures when a precise end time matters; do not use audio.</p>
""" + "\n".join(sections) + "\n</html>\n"
    index_path = review_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 0.5-second contact sheets from each serve to the next recorded serve."
    )
    parser.add_argument("input_video", type=Path, help="Stage 0 proxy video")
    parser.add_argument("points_json", type=Path, help="Stage 5 points.json")
    parser.add_argument("review_dir", type=Path, help="directory for rally contact sheets")
    parser.add_argument("output_json", type=Path, help="editable rally review JSON")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="seconds between sampled frames")
    parser.add_argument(
        "--estimate-rally-ends", type=Path, metavar="PLAYERS_JSON",
        help="estimate serve gaps and rally ends from Stage 3 sampled player tracks",
    )
    parser.add_argument(
        "--pause-duration-sec", type=float, default=1.5,
        help="low-motion duration that marks a sustained pause (default: 1.5)",
    )
    parser.add_argument(
        "--pause-speed-m-s", type=float, default=0.8,
        help="maximum tracked speed during a pause in m/s (default: 0.8)",
    )
    parser.add_argument(
        "--active-speed-m-s", type=float, default=1.0,
        help="speed that counts as an active moment in m/s (default: 1.0)",
    )
    parser.add_argument(
        "--shuttle-tracks", type=Path,
        help="optional Stage 3A raw TrackNet tracks; attach coverage and rally-end comparison",
    )
    parser.add_argument(
        "--rally-end-source", choices=("players", "shuttle"), default="players",
        help="use player pauses (default), or refine them with a sustained TrackNet cluster near the pause",
    )
    parser.add_argument(
        "--shuttle-rally-window-sec", type=float, default=1.5,
        help="seconds before the player pause to search for a TrackNet cluster (default: 1.5)",
    )
    parser.add_argument(
        "--shuttle-rally-after-sec", type=float, default=0.35,
        help="maximum seconds after the player pause to accept TrackNet points (default: 0.35)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing review outputs")
    args = parser.parse_args()

    video_path = args.input_video.expanduser().resolve()
    points_path = args.points_json.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    players_path = args.estimate_rally_ends.expanduser().resolve() if args.estimate_rally_ends else None
    shuttle_path = args.shuttle_tracks.expanduser().resolve() if args.shuttle_tracks else None
    if not video_path.is_file() or not points_path.is_file():
        parser.error("input video and points JSON must both exist")
    if players_path is not None and not players_path.is_file():
        parser.error(f"player tracks do not exist: {players_path}")
    if shuttle_path is not None and not shuttle_path.is_file():
        parser.error(f"TrackNet shuttle tracks do not exist: {shuttle_path}")
    if args.rally_end_source == "shuttle" and (shuttle_path is None or players_path is None):
        parser.error("--rally-end-source shuttle requires --shuttle-tracks and --estimate-rally-ends")
    if args.shuttle_rally_window_sec <= 0 or args.shuttle_rally_after_sec < 0:
        parser.error("TrackNet rally search window must be positive and the after-pause allowance cannot be negative")
    if args.interval_sec <= 0:
        parser.error("frame interval must be positive")
    if args.pause_duration_sec <= 0 or args.pause_speed_m_s < 0 or args.active_speed_m_s <= 0:
        parser.error("pause duration and active speed must be positive; pause speed cannot be negative")
    if output_path in (video_path, points_path) or review_dir in (video_path, points_path, output_path):
        parser.error("input and output paths must be different")
    if players_path is not None and (players_path in (output_path, review_dir) or players_path == video_path):
        parser.error("player-track input and review outputs must be different")
    if shuttle_path is not None and shuttle_path in (output_path, review_dir, video_path, points_path, players_path):
        parser.error("TrackNet input and Stage 6 outputs must be different")
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {output_path} (pass --overwrite)")
    if review_dir.exists() and not args.overwrite:
        parser.error(f"review directory already exists: {review_dir} (pass --overwrite)")

    points = load_video_points(points_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError("video has invalid frame rate or frame count")
    video_duration = frame_count / fps
    known_times = [float(point["t"]) for point in points if point["t"] is not None]
    if not known_times or known_times[-1] >= video_duration:
        cap.release()
        raise ValueError("last serve timestamp is at or beyond the end of the video")

    review_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for pattern in ("point_*_page_*.jpg", "gap_*_page_*.jpg"):
            for old_sheet in review_dir.glob(pattern):
                old_sheet.unlink()

    review_points = []
    unresolved_gaps = []
    total_frames = 0
    total_pages = 0
    index = 0
    while index < len(points):
        point = points[index]
        if point["t"] is None:
            raise ValueError(f"s{point['n']} has no earlier timed serve to bound its review interval")
        next_index = index + 1
        if next_index < len(points) and points[next_index]["t"] is None:
            boundary_index = next_index
            while boundary_index < len(points) and points[boundary_index]["t"] is None:
                boundary_index += 1
            if boundary_index >= len(points):
                raise ValueError("an untimed serve at the end of the match has no later timestamp bound")
            boundary_point = points[boundary_index]
            missing_points = points[next_index:boundary_index]
            first_n, last_n = int(point["n"]), int(missing_points[-1]["n"])
            stem = f"gap_s{first_n:03d}_s{last_n:03d}"
            if args.estimate_rally_ends:
                label = (
                    f"ESTIMATED SPLIT s{first_n:02d}-s{last_n:02d}: serve time(s) will be interpolated; "
                    f"{float(point['t']):.1f}s to next detected serve {float(boundary_point['t']):.1f}s"
                )
            else:
                label = (
                    f"UNRESOLVED s{first_n:02d}-s{last_n:02d}: locate the untimed serve and split this interval; "
                    f"{float(point['t']):.1f}s to next detected serve {float(boundary_point['t']):.1f}s"
                )
            pages, frame_total = make_point_pages(
                cap, point, boundary_point, fps, frame_count, video_duration,
                args.interval_sec, review_dir, output_path.parent,
                label_override=label, stem_override=stem,
            )
            total_frames += frame_total
            total_pages += len(pages)
            unresolved_gaps.append({
                "point_ns": [int(item["n"]) for item in [point, *missing_points]],
                "start_t": round(float(point["t"]), 3),
                "end_t": round(float(boundary_point["t"]), 3),
                "missing_serve_point_ns": [int(item["n"]) for item in missing_points],
                "contact_sheets": pages,
            })
            for group_index, item in enumerate([point, *missing_points]):
                item_t = None if item["t"] is None else round(float(item["t"]), 3)
                review_points.append({
                    "point_n": item["n"], "candidate_n": item.get("candidate_n"),
                    "serve_t": item_t,
                    "next_serve_t": None if item_t is None else (
                        round(float(points[index + 1]["t"]), 3)
                        if group_index == 0 and index + 1 < boundary_index and points[index + 1]["t"] is not None
                        else None
                    ),
                    "review_end_t": round(float(boundary_point["t"]), 3),
                    "score_before": item.get("before"), "server": item.get("server"),
                    "winner": item.get("winner"), "score_after": item.get("after"),
                    "rally_end_sec": None, "rally_duration_sec": None,
                    "review_scope": "shared unresolved interval; split by visual review",
                    "contact_sheets": pages,
                })
            interval_label = "Shared interval" if args.estimate_rally_ends else "Unresolved interval"
            print(
                f"{interval_label} s{first_n:02d}-s{last_n:02d}: {len(pages)} page(s), "
                f"{frame_total} frames, {float(point['t']):.1f}s to {float(boundary_point['t']):.1f}s."
            )
            print(f"Rallies processed: {boundary_index}/{len(points)}", flush=True)
            index = boundary_index
            continue

        next_point = points[next_index] if next_index < len(points) else None
        end_t = float(next_point["t"]) if next_point else video_duration
        pages, frame_total = make_point_pages(
            cap, point, next_point, fps, frame_count, video_duration,
            args.interval_sec, review_dir, output_path.parent,
        )
        total_frames += frame_total
        total_pages += len(pages)
        review_points.append({
            "point_n": point["n"], "candidate_n": point.get("candidate_n"),
            "serve_t": round(float(point["t"]), 3),
            "next_serve_t": None if next_point is None else round(float(next_point["t"]), 3),
            "review_end_t": round(end_t, 3), "score_before": point.get("before"),
            "server": point.get("server"), "winner": point.get("winner"),
            "score_after": point.get("after"), "rally_end_sec": None,
            "rally_duration_sec": None, "review_scope": "single point",
            "contact_sheets": pages,
        })
        print(
            f"Prepared s{point['n']:02d}: {len(pages)} page(s), "
            f"{frame_total} frames, {float(point['t']):.1f}s to {end_t:.1f}s."
        )
        index += 1
        print(f"Rallies processed: {index}/{len(points)}", flush=True)
    cap.release()

    end_estimation = None
    shuttle_data = load_shuttle_tracks(shuttle_path) if shuttle_path is not None else None
    if shuttle_data is not None:
        validate_shuttle_video(shuttle_data, video_path)
        if args.rally_end_source == "shuttle" and not shuttle_data.get("court_geometry"):
            parser.error("TrackNet rally-end selection requires Stage 3A --calibration-json court tags")
    estimated_serve_times = 0
    if players_path is not None:
        estimate_missing_serve_times(review_points)
        estimated_serve_times = sum(bool(point.get("serve_time_estimated")) for point in review_points)
        motion_times, motion, motion_fps = load_player_motion(players_path)
        estimate_rally_ends(
            review_points,
            motion_times,
            motion,
            motion_fps,
            video_duration,
            pause_speed_m_s=args.pause_speed_m_s,
            active_speed_m_s=args.active_speed_m_s,
            pause_duration_sec=args.pause_duration_sec,
        )
        shuttle_refined = shuttle_fallback = 0
        if args.rally_end_source == "shuttle" and shuttle_data is not None:
            shuttle_refined, shuttle_fallback = refine_rally_ends_with_shuttle(
                review_points,
                shuttle_data,
                search_before_sec=args.shuttle_rally_window_sec,
                search_after_sec=args.shuttle_rally_after_sec,
            )
        for gap in unresolved_gaps:
            gap["resolved_by_estimated_timestamps"] = all(
                point.get("serve_t") is not None
                for point in review_points
                if point["point_n"] in gap["point_ns"]
            )
        estimated_n = {
            point_n
            for gap in unresolved_gaps
            if gap.get("resolved_by_estimated_timestamps")
            for point_n in gap["point_ns"]
        }
        for point in review_points:
            if point["point_n"] in estimated_n:
                point["review_scope"] = "shared contact sheet; serve time interpolated; rally end estimated"
        end_estimation = {
            "method": (
                "Stage 3 player pause refined by a sustained raw TrackNet main-court cluster near that pause"
                if args.rally_end_source == "shuttle" else
                "smoothed Stage 3 player-track speed; first sustained low-motion window after active movement"
            ),
            "rally_end_source": args.rally_end_source,
            "missing_serve_method": "equal spacing inside the bounded shared interval",
            "source_players_json": players_path.name,
            "smoothing_window_sec": 0.7,
            "speed_window_sec": 0.5,
            "pause_duration_sec": args.pause_duration_sec,
            "pause_speed_m_s": args.pause_speed_m_s,
            "active_speed_m_s": args.active_speed_m_s,
            "estimates_are_approximate": True,
        }

    if shuttle_data is not None:
        add_shuttle_evidence(review_points, shuttle_data)
        if end_estimation is not None:
            end_estimation["tracknet_evidence"] = (
                "sustained calibrated main-court TrackNet clusters can refine the movement pause; "
                "points without enough nearby shuttle observations retain the player estimate"
                if args.rally_end_source == "shuttle" else
                "raw track coverage and proximity to the pause estimate are recorded per point; "
                "TrackNet does not change rally_end_sec in the default players mode"
            )
            if args.rally_end_source == "shuttle":
                end_estimation["tracknet_rally_end_refinements_applied"] = shuttle_refined
                end_estimation["tracknet_rally_end_fallbacks"] = shuttle_fallback

    index_path = write_index(review_dir, review_points)

    payload = {
        "source_video": video_path.name,
        "source_points_json": points_path.name,
        "video_duration_sec": round(video_duration, 3),
        "sample_interval_sec": args.interval_sec,
        "method": (
            "picture-only contact sheets plus player-track timing estimates with optional TrackNet refinement; no audio used"
            if end_estimation else "picture-only review; no audio used"
        ),
        "dummy_points_skipped": True,
        "review_instructions": (
            (
                "Rally end times are approximate when rally_end_estimated is true; inspect contact sheets for exact timing. "
                + (
                    "TrackNet-refined estimates use a sustained calibrated main-court cluster near the player pause; "
                    "points without one keep the player estimate. "
                    if args.rally_end_source == "shuttle" else ""
                )
                + "Serve times marked estimated were interpolated inside bounded shared intervals. No audio is used."
            )
            if end_estimation else
            "Set rally_end_sec to the last moment of live play; rally_duration_sec is rally_end_sec - serve_t. "
            "Shared unresolved intervals must be split visually first."
        ),
        "end_time_estimation": end_estimation,
        "source_shuttle_tracks_json": shuttle_path.name if shuttle_path else None,
        "contact_sheet_index": os.path.relpath(index_path.resolve(), output_path.parent.resolve()),
        "unresolved_intervals": unresolved_gaps,
        "points": review_points,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded serves: {len(points)}; contact sheets: {total_pages}; sampled frames: {total_frames}.")
    print(f"Review sheets: {review_dir}")
    print(f"Contact sheet index: {index_path}")
    print(f"Editable review JSON: {output_path}")
    if end_estimation:
        if args.rally_end_source == "shuttle":
            print(
                f"Estimated rally ends: {len(review_points)}; TrackNet refinements: {shuttle_refined}; "
                f"movement fallbacks: {shuttle_fallback}; estimated serve times: {estimated_serve_times}."
            )
        else:
            pause_count = sum(point.get("rally_end_method") == "first_sustained_pause" for point in review_points)
            fallback_count = len(review_points) - pause_count
            print(
                f"Estimated rally ends: {len(review_points)}; first-pause estimates: {pause_count}; "
                f"fallbacks: {fallback_count}; estimated serve times: {estimated_serve_times}."
            )
    if shuttle_data is not None and args.rally_end_source == "players":
        print(f"TrackNet evidence attached to {len(review_points)} review rows; rally ends were unchanged.")
    elif unresolved_gaps:
        print("Review the shared unresolved interval(s) to locate the untimed serve before assigning individual rally ends.")
    print("The 0-second dummy score point is excluded from video analysis.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, cv2.error, json.JSONDecodeError) as error:
        print(f"Stage 6 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
