#!/usr/bin/env python3
"""Calculate per-rally player movement and court heatmaps from reviewed rally times."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
HALF_LENGTH_M = 6.7
BASE_POINT = (3.05, 3.4)
OUT_OF_POSITION_M = 2.2
BACK_COURT_DEPTH_M = 1.6
FRONT_SERVICE_LINE_M = 4.72
MAX_TRACK_SPEED_M_S = 8.0


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def position(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2 or not all(number(item) for item in value):
        return None
    return float(value[0]), float(value[1])


def load_review(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("rally review JSON must contain a non-empty points array")
    missing_starts = []
    missing_ends = []
    for point in points:
        point_n = point.get("point_n", "?")
        if point.get("serve_t") is None:
            missing_starts.append(point_n)
        elif not number(point.get("serve_t")):
            raise ValueError(f"s{point_n} has an invalid serve_t")
        if point.get("rally_end_sec") is None:
            missing_ends.append(point_n)
        elif not number(point.get("rally_end_sec")):
            raise ValueError(f"s{point_n} has an invalid rally_end_sec")
    if missing_starts or missing_ends:
        details = []
        if missing_starts:
            details.append(f"serve_t missing for s{', s'.join(map(str, missing_starts))}")
        if missing_ends:
            details.append(f"rally_end_sec missing for s{', s'.join(map(str, missing_ends))}")
        raise ValueError("complete Stage 6 visual review first: " + "; ".join(details))
    ordered = sorted(points, key=lambda item: int(item["point_n"]))
    previous_end = None
    for point in ordered:
        start = float(point["serve_t"])
        end = float(point["rally_end_sec"])
        if end < start:
            raise ValueError(f"s{point['point_n']} rally_end_sec must not precede serve_t")
        next_t = point.get("next_serve_t")
        if next_t is not None and end > float(next_t):
            raise ValueError(f"s{point['point_n']} rally_end_sec must not pass the next serve")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"s{point['point_n']} starts before the prior reviewed rally ends")
        previous_end = end
    return ordered


def load_tracks(path: Path) -> tuple[list[float], dict[str, list[tuple[float, float] | None]], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    fps = float(data.get("fps", 0))
    if not isinstance(frames, list) or not frames or fps <= 0:
        raise ValueError("players JSON needs frames and a positive fps")
    times = [float(frame["t"]) for frame in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("players frame timestamps must increase")
    tracks: dict[str, list[tuple[float, float] | None]] = {"R": [], "O": []}
    for frame in frames:
        for player in ("R", "O"):
            point = position(frame.get("r" if player == "R" else "o"))
            if point is None:
                tracks[player].append(None)
            else:
                # Rotate whichever identity occupies the far half into their own-court view.
                if point[1] >= HALF_LENGTH_M:
                    point = (COURT_WIDTH_M - point[0], COURT_LENGTH_M - point[1])
                tracks[player].append(point)
    return times, tracks, fps


def smooth_track(
    track: list[tuple[float, float] | None], window_frames: int
) -> list[tuple[float, float] | None]:
    if window_frames % 2 == 0:
        window_frames += 1
    radius = window_frames // 2
    result = []
    for index in range(len(track)):
        values = [value for value in track[max(0, index-radius):min(len(track), index+radius+1)] if value is not None]
        if len(values) < max(1, math.ceil(window_frames / 2)):
            result.append(None)
        else:
            med = np.median(np.asarray(values, dtype=np.float64), axis=0)
            result.append((float(med[0]), float(med[1])))
    return result


def distance_to_base(point: tuple[float, float]) -> float:
    return math.dist(point, BASE_POINT)


def point_metrics(
    player: str,
    track: list[tuple[float, float] | None],
    times: list[float],
    start_t: float,
    end_t: float,
    fps: float,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    active_start = start_t + 0.3
    indexes = [i for i, t in enumerate(times) if active_start <= t <= end_t and track[i] is not None]
    samples = [(times[i], track[i]) for i in indexes]
    if not samples:
        return {
            "distance_m": 0.0, "observed_time_sec": 0.0, "average_speed_m_s": None,
            "peak_speed_m_s": None, "back_court_time_share": None,
            "front_court_time_share": None, "median_position_local_m": None,
            "out_of_position_trips": [],
        }, []

    distance = 0.0
    observed_time = 0.0
    speeds = []
    for (ta, a), (tb, b) in zip(samples, samples[1:]):
        dt = tb - ta
        if dt <= 0 or dt > 0.35:
            continue
        speed = math.dist(a, b) / dt
        if speed > MAX_TRACK_SPEED_M_S:
            # A larger one-frame jump is a tracking mix-up, not human court movement.
            continue
        distance += math.dist(a, b)
        observed_time += dt
        speeds.append(speed)

    us = [p[0] for _, p in samples]
    vs = [p[1] for _, p in samples]
    back_share = sum(v <= BACK_COURT_DEPTH_M for v in vs) / len(vs)
    front_share = sum(v >= FRONT_SERVICE_LINE_M for v in vs) / len(vs)

    trips = []
    trip_indexes = []
    last_index = None
    for sample_index, (frame_index, (t, p)) in enumerate(zip(indexes, samples)):
        away = distance_to_base(p) >= OUT_OF_POSITION_M
        gap_break = last_index is not None and times[frame_index] - times[last_index] > 0.35
        if gap_break and trip_indexes:
            trips.append(finish_trip(trip_indexes, samples, times, indexes))
            trip_indexes = []
        if away:
            trip_indexes.append(sample_index)
        elif trip_indexes:
            trips.append(finish_trip(trip_indexes, samples, times, indexes))
            trip_indexes = []
        last_index = frame_index
    if trip_indexes:
        trips.append(finish_trip(trip_indexes, samples, times, indexes))

    points = [point for _, point in samples]
    median = np.median(np.asarray(points, dtype=np.float64), axis=0)
    local_points = [(float(p[0]), float(p[1])) for p in points]
    metrics = {
        "distance_m": round(distance, 2),
        "observed_time_sec": round(observed_time, 2),
        "average_speed_m_s": round(distance / observed_time, 2) if observed_time > 0 else None,
        "peak_speed_m_s": round(max(speeds), 2) if speeds else None,
        "back_court_time_share": round(back_share, 3),
        "front_court_time_share": round(front_share, 3),
        "median_position_local_m": [round(float(median[0]), 2), round(float(median[1]), 2)],
        "out_of_position_trips": trips,
    }
    return metrics, local_points


def finish_trip(
    sample_indexes: list[int],
    samples: list[tuple[float, tuple[float, float]]],
    times: list[float],
    frame_indexes: list[int],
) -> dict[str, Any]:
    first = sample_indexes[0]
    last = sample_indexes[-1]
    max_away = max(distance_to_base(samples[i][1]) for i in sample_indexes)
    start_t = samples[first][0]
    target_t = start_t + 1.0
    possible = [
        (abs(t - target_t), p)
        for t, p in samples
        if abs(t - target_t) <= 0.2
    ]
    recovery = distance_to_base(min(possible, key=lambda item: item[0])[1]) if possible else None
    return {
        "start_t": round(start_t, 2),
        "end_t": round(samples[last][0], 2),
        "max_distance_from_base_m": round(max_away, 2),
        "distance_from_base_1s_later_m": None if recovery is None else round(recovery, 2),
    }


def make_heatmap(path: Path, player: str, points: list[tuple[float, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = [point[0] for point in points]
    v = [point[1] for point in points]
    histogram, _, _ = np.histogram2d(
        v, u, bins=(22, 20), range=((0, HALF_LENGTH_M), (0, COURT_WIDTH_M))
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(
        histogram, origin="lower", extent=(0, COURT_WIDTH_M, 0, HALF_LENGTH_M),
        aspect="equal", cmap="magma",
    )
    ax.set(xlim=(0, COURT_WIDTH_M), ylim=(0, HALF_LENGTH_M),
           xlabel="Across from player's own left sideline (m)",
           ylabel="Distance from player's own baseline (m)",
           title=f"{player} movement heatmap")
    ax.axhline(FRONT_SERVICE_LINE_M, color="cyan", linewidth=0.8, linestyle="--")
    ax.scatter([BASE_POINT[0]], [BASE_POINT[1]], marker="+", color="white", label="base point")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Sample count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate player movement per visually reviewed rally."
    )
    parser.add_argument("players_json", type=Path, help="Stage 3 players.json")
    parser.add_argument("rallies_review_json", type=Path, help="completed Stage 6 review JSON")
    parser.add_argument("output_json", type=Path, help="output movement metrics JSON")
    parser.add_argument("heatmap_dir", type=Path, help="directory for R/O heatmap PNGs")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    players_path = args.players_json.expanduser().resolve()
    review_path = args.rallies_review_json.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    heatmap_dir = args.heatmap_dir.expanduser().resolve()
    if not players_path.is_file() or not review_path.is_file():
        parser.error("players JSON and completed rally review JSON must both exist")
    if output_path in (players_path, review_path) or heatmap_dir in (players_path, review_path, output_path):
        parser.error("input and output paths must be different")
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {output_path} (pass --overwrite)")
    if heatmap_dir.exists() and not args.overwrite:
        parser.error(f"heatmap directory already exists: {heatmap_dir} (pass --overwrite)")

    rallies = load_review(review_path)
    times, raw_tracks, fps = load_tracks(players_path)
    window_frames = max(1, round(0.7 * fps))
    if window_frames % 2 == 0:
        window_frames += 1
    tracks = {player: smooth_track(raw_tracks[player], window_frames) for player in ("R", "O")}

    point_results = []
    heat_points: dict[str, list[tuple[float, float]]] = {"R": [], "O": []}
    total_distance = {"R": 0.0, "O": 0.0}
    total_time = {"R": 0.0, "O": 0.0}
    max_speed: dict[str, float | None] = {"R": None, "O": None}
    for rally in rallies:
        start_t = float(rally["serve_t"])
        end_t = float(rally["rally_end_sec"])
        players = {}
        for player in ("R", "O"):
            metrics, local_points = point_metrics(player, tracks[player], times, start_t, end_t, fps)
            players[player] = metrics
            heat_points[player].extend(local_points)
            total_distance[player] += metrics["distance_m"]
            total_time[player] += metrics["observed_time_sec"]
            speed = metrics["peak_speed_m_s"]
            if speed is not None and (max_speed[player] is None or speed > max_speed[player]):
                max_speed[player] = speed
        point_results.append({
            "point_n": rally["point_n"], "candidate_n": rally.get("candidate_n"),
            "serve_t": start_t, "rally_end_sec": end_t,
            "rally_end_estimated": bool(rally.get("rally_end_estimated", False)),
            "rally_end_method": rally.get("rally_end_method", "manual_review"),
            "rally_end_confidence": rally.get("rally_end_confidence", "reviewed"),
            "serve_time_estimated": bool(rally.get("serve_time_estimated", False)),
            "serve_time_method": rally.get("serve_time_method"),
            "rally_duration_sec": round(end_t - start_t, 3),
            "movement_window_start_sec": round(start_t + 0.3, 3),
            "players": players,
        })

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old_file in heatmap_dir.glob("*_heatmap.png"):
            old_file.unlink()
    heatmap_paths = {}
    for player in ("R", "O"):
        heatmap_path = heatmap_dir / f"{player}_heatmap.png"
        make_heatmap(heatmap_path, player, heat_points[player])
        heatmap_paths[player] = str(heatmap_path.relative_to(output_path.parent))

    summary = {}
    for player in ("R", "O"):
        summary[player] = {
            "distance_m": round(total_distance[player], 2),
            "observed_time_sec": round(total_time[player], 2),
            "average_speed_m_s": round(total_distance[player] / total_time[player], 2) if total_time[player] > 0 else None,
            "peak_speed_m_s": max_speed[player],
        }
    output = {
        "source_players_json": players_path.name,
        "source_rallies_review_json": review_path.name,
        "end_time_estimation": json.loads(review_path.read_text(encoding="utf-8")).get("end_time_estimation"),
        "smoothing_window_sec": 0.7,
        "movement_starts_sec_after_serve": 0.3,
        "court_coordinate_system": "local player view; u across from own left sideline, v from own baseline",
        "metrics_by_player": summary,
        "heatmaps": heatmap_paths,
        "points": point_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Rallies analyzed: {len(point_results)}; smoothing: {window_frames} frames ({window_frames/fps:.1f}s).")
    print(
        "Overall average speed during analyzed windows: "
        f"R {summary['R']['average_speed_m_s']} m/s, "
        f"O {summary['O']['average_speed_m_s']} m/s."
    )
    print(f"R heatmap: {heatmap_paths['R']}")
    print(f"O heatmap: {heatmap_paths['O']}")
    print(f"Motion JSON: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Stage 7 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
