#!/usr/bin/env python3
"""Calculate singles movement metrics and court heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import COURT_LENGTH, COURT_WIDTH, NET_Y, artifact_path, load_job, read_json, write_json


def own_coordinates(position: list[float]) -> tuple[float, float, bool]:
    x, y = map(float, position)
    is_near = y < NET_Y
    own_x = x if is_near else COURT_WIDTH - x
    depth = y if is_near else COURT_LENGTH - y
    return own_x, depth, is_near


def smoothed_positions(frames: list[dict[str, Any]], key: str, half_window: float = 0.35) -> list[list[float] | None]:
    times = [float(frame["t"]) for frame in frames]
    values = [frame.get(key) for frame in frames]
    smoothed = []
    left = 0
    right = 0
    for index, seconds in enumerate(times):
        while left < len(times) and times[left] < seconds - half_window:
            left += 1
        right = max(right, index)
        while right + 1 < len(times) and times[right + 1] <= seconds + half_window:
            right += 1
        neighbors = [values[j] for j in range(left, right + 1) if values[j] is not None]
        smoothed.append(np.mean(np.asarray(neighbors, dtype=float), axis=0).tolist() if neighbors else None)
    return smoothed


def base_position(position: list[float]) -> tuple[float, float]:
    return (3.05, 3.4) if position[1] < NET_Y else (3.05, COURT_LENGTH - 3.4)


def analyze_rally(
    frames: list[dict[str, Any]],
    smooth: list[list[float] | None],
    start: float,
    end: float,
) -> dict[str, Any]:
    indices = [i for i, frame in enumerate(frames) if start <= float(frame["t"]) <= end and smooth[i] is not None]
    if len(indices) < 2:
        return {"samples": len(indices), "distance_m": None, "average_speed_mps": None, "peak_speed_mps": None}

    points = [smooth[i] for i in indices]
    times = [float(frames[i]["t"]) for i in indices]
    steps = []
    for previous, current, t0, t1 in zip(points, points[1:], times, times[1:]):
        dt = t1 - t0
        if 0 < dt <= 0.3:
            distance = float(np.linalg.norm(np.asarray(current) - np.asarray(previous)))
            steps.append((t0, t1, dt, distance, distance / dt))
    distance = sum(step[3] for step in steps)
    valid_time = sum(step[2] for step in steps)
    speeds = [step[4] for step in steps]
    positions = [smooth[i] for i in indices]
    own = [own_coordinates(position) for position in positions]
    median = np.median(np.asarray([[item[0], item[1]] for item in own]), axis=0).tolist()

    front_time = 0.0
    back_time = 0.0
    for previous, current, t0, t1 in zip(own, own[1:], times, times[1:]):
        dt = t1 - t0
        if 0 < dt <= 0.3:
            depth = (previous[1] + current[1]) / 2
            if depth >= 4.72:
                front_time += dt
            if depth <= 1.6:
                back_time += dt

    trips = []
    last_trip_time = -10.0
    for index, position in zip(indices, positions):
        base = base_position(position)
        offset = float(np.linalg.norm(np.asarray(position) - np.asarray(base)))
        seconds = float(frames[index]["t"])
        if offset < 2.2 or seconds - last_trip_time < 2.0:
            continue
        recovery_index = min(range(len(times)), key=lambda j: abs(times[j] - (seconds + 1.0)))
        if abs(times[recovery_index] - (seconds + 1.0)) <= 0.25:
            recovery_position = positions[recovery_index]
            recovery_base = base_position(recovery_position)
            recovery_offset = float(np.linalg.norm(np.asarray(recovery_position) - np.asarray(recovery_base)))
        else:
            recovery_offset = None
        trips.append({"t": seconds, "offset_from_base_m": round(offset, 2), "offset_1s_later_m": round(recovery_offset, 2) if recovery_offset is not None else None})
        last_trip_time = seconds

    return {
        "samples": len(indices),
        "distance_m": round(distance, 2),
        "average_speed_mps": round(distance / valid_time, 2) if valid_time else None,
        "peak_speed_mps": round(max(speeds), 2) if speeds else None,
        "front_court_share": round(front_time / valid_time, 3) if valid_time else None,
        "back_court_share": round(back_time / valid_time, 3) if valid_time else None,
        "median_position_own_coordinates_m": [round(float(value), 2) for value in median],
        "trips_out_of_position": trips,
    }


def aggregate_by_outcome(per_point: list[dict[str, Any]], player_key: str) -> dict[str, Any]:
    summary = {}
    for outcome in ("won", "lost"):
        entries = [point[player_key] for point in per_point if point["outcome"] == outcome and point[player_key].get("distance_m") is not None]
        fields = ("distance_m", "average_speed_mps", "peak_speed_mps", "front_court_share", "back_court_share")
        summary[outcome] = {
            "points": len(entries),
            **{
                field: round(float(np.mean([entry[field] for entry in entries if entry.get(field) is not None])), 2)
                if any(entry.get(field) is not None for entry in entries)
                else None
                for field in fields
            },
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--points", help="Reviewed points JSON; defaults to points.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to motion.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        load_job(job_path)
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        points_path = Path(args.points).expanduser().resolve() if args.points else artifact_path(job_path, "points.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "motion.json")
        player_data = read_json(players_path)
        points_data = read_json(points_path)
        frames = player_data.get("frames", [])
        if not frames:
            raise ValueError("players.json contains no frames.")
        smooth_r = smoothed_positions(frames, "r")
        smooth_o = smoothed_positions(frames, "o")

        per_point = []
        heat_positions = {"r": [], "o": []}
        for point in points_data.get("points", []):
            start = float(point["r0"]) + 0.3
            end = float(point["r1"])
            r_metrics = analyze_rally(frames, smooth_r, start, end)
            o_metrics = analyze_rally(frames, smooth_o, start, end)
            outcome = "won" if point["winner"] == "R" else "lost"
            per_point.append({"n": point["n"], "t": point["t"], "rally_duration_s": point["rdur"], "outcome": outcome, "r": r_metrics, "o": o_metrics})
            for key, smooth in (("r", smooth_r), ("o", smooth_o)):
                for index, frame in enumerate(frames):
                    if start <= float(frame["t"]) <= end and smooth[index] is not None:
                        own_x, depth, _ = own_coordinates(smooth[index])
                        heat_positions[key].append([own_x, depth])

        heatmap_paths = {}
        heatmap_dir = artifact_path(job_path, "heatmaps")
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for key, positions in heat_positions.items():
            histogram, _, _ = np.histogram2d(
                [position[0] for position in positions],
                [position[1] for position in positions],
                bins=(31, 34),
                range=((0, COURT_WIDTH), (0, NET_Y)),
            )
            figure, axis = plt.subplots(figsize=(5, 6))
            axis.imshow(histogram.T, origin="lower", extent=(0, COURT_WIDTH, 0, NET_Y), aspect="auto", cmap="magma")
            axis.set(xlim=(0, COURT_WIDTH), ylim=(0, NET_Y), xlabel="Across court (m)", ylabel="Distance from own baseline (m)", title=f"Player {key.upper()} movement heatmap")
            path = heatmap_dir / f"{key}_heatmap.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            heatmap_paths[key] = str(path)

        write_json(
            output_path,
            {
                "schema_version": 1,
                "method": "0.7 second position smoothing; rally window starts 0.3 seconds after serve",
                "per_point": per_point,
                "user_by_outcome": aggregate_by_outcome(per_point, "r"),
                "opponent_by_outcome": aggregate_by_outcome(per_point, "o"),
                "heatmaps": heatmap_paths,
            },
        )
        print(f"Saved movement metrics to {output_path} and heatmaps to {heatmap_dir}.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Movement analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
