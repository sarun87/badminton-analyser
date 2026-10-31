#!/usr/bin/env python3
"""Estimate which half a shuttle landed in from post-rally player bends."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import COURT_LENGTH, NET_Y, artifact_path, load_job, read_json, write_json


def depth_for(position: list[float]) -> float:
    return float(position[1]) if float(position[1]) < NET_Y else COURT_LENGTH - float(position[1])


def build_height_reference(frames: list[dict[str, Any]], player: str) -> tuple[dict[int, float], float]:
    bins: dict[int, list[float]] = {}
    all_heights = []
    box_key = "rb" if player == "r" else "ob"
    for frame in frames:
        position, box = frame.get(player), frame.get(box_key)
        if position is None or box is None:
            continue
        height = float(box[3]) - float(box[1])
        if height <= 0:
            continue
        depth_bin = int(np.clip(depth_for(position), 0, 6.6))
        bins.setdefault(depth_bin, []).append(height)
        all_heights.append(height)
    fallback = float(np.percentile(all_heights, 80)) if all_heights else 0.0
    reference = {key: float(np.percentile(values, 80)) for key, values in bins.items() if values}
    return reference, fallback


def player_speeds(frames: list[dict[str, Any]], player: str) -> list[float | None]:
    result: list[float | None] = [None] * len(frames)
    for index in range(1, len(frames) - 1):
        before, after = frames[index - 1].get(player), frames[index + 1].get(player)
        dt = float(frames[index + 1]["t"]) - float(frames[index - 1]["t"])
        if before is not None and after is not None and 0 < dt <= 0.5:
            result[index] = float(np.linalg.norm(np.asarray(after) - np.asarray(before)) / dt)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--points", help="Reviewed points JSON; defaults to points.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to pickups.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        load_job(job_path)
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        points_path = Path(args.points).expanduser().resolve() if args.points else artifact_path(job_path, "points.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "pickups.json")
        frames = read_json(players_path).get("frames", [])
        points = read_json(points_path).get("points", [])
        references = {player: build_height_reference(frames, player) for player in ("r", "o")}
        speeds = {player: player_speeds(frames, player) for player in ("r", "o")}

        results = []
        for point in points:
            rally_end = float(point["r1"])
            next_serve = float(point.get("next_t", rally_end + 8.0))
            search_until = min(next_serve, rally_end + 8.0)
            possible = []
            for index, frame in enumerate(frames):
                seconds = float(frame["t"])
                if not rally_end <= seconds <= search_until:
                    continue
                for player in ("r", "o"):
                    position = frame.get(player)
                    box = frame.get("rb" if player == "r" else "ob")
                    speed = speeds[player][index]
                    if position is None or box is None or speed is None or speed >= 1.0:
                        continue
                    height = float(box[3]) - float(box[1])
                    if height <= 0:
                        continue
                    depth_bin = int(np.clip(depth_for(position), 0, 6.6))
                    by_depth, fallback = references[player]
                    expected = by_depth.get(depth_bin, fallback)
                    if expected <= 0:
                        continue
                    ratio = height / expected
                    if ratio < 0.72:
                        possible.append(
                            {
                                "t": seconds,
                                "player": player.upper(),
                                "ratio_to_standing_height": round(ratio, 3),
                                "position": position,
                                "side": "near" if float(position[1]) < NET_Y else "far",
                            }
                        )
            evidence = min(possible, key=lambda item: item["t"]) if possible else None
            if evidence is not None:
                evidence["confidence"] = "medium" if evidence["ratio_to_standing_height"] < 0.60 else "low"
            results.append(
                {
                    "n": point["n"],
                    "t": point["t"],
                    "winner": point["winner"],
                    "shuttle_side_estimate": evidence["side"] if evidence else None,
                    "evidence": evidence,
                }
            )

        write_json(
            output_path,
            {
                "schema_version": 1,
                "method": "first slow post-rally player box below 72 percent of depth-adjusted standing height",
                "caution": "A bend is only a pickup proxy; visually verify the shuttle side before drawing conclusions.",
                "points": results,
            },
        )
        print(f"Saved pickup-side estimates for {len(results)} points to {output_path}.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Pickup analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
