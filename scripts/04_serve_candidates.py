#!/usr/bin/env python3
"""Find likely pre-serve pauses from players' court positions and movement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import NET_Y, artifact_path, load_job, read_json, write_json


def smooth_positions(rows: list[dict[str, Any]], key: str, radius: int = 2) -> list[list[float] | None]:
    values = []
    for index in range(len(rows)):
        neighbors = [rows[j].get(key) for j in range(max(0, index - radius), min(len(rows), index + radius + 1))]
        valid = [value for value in neighbors if value is not None]
        values.append(np.mean(np.asarray(valid, dtype=float), axis=0).tolist() if valid else None)
    return values


def speeds(times: list[float], positions: list[list[float] | None]) -> list[float | None]:
    result: list[float | None] = [None] * len(times)
    for index in range(1, len(times) - 1):
        before, after = positions[index - 1], positions[index + 1]
        elapsed = times[index + 1] - times[index - 1]
        if before is not None and after is not None and 0 < elapsed <= 0.5:
            result[index] = float(np.linalg.norm(np.asarray(after) - np.asarray(before)) / elapsed)
    return result


def in_service_area(position: list[float] | None) -> bool:
    if position is None:
        return False
    x, y = position
    return 0.2 <= x <= 5.9 and (1.0 <= y <= 4.9 or 8.5 <= y <= 12.4)


def own_side(position: list[float] | None) -> str | None:
    if position is None:
        return None
    x, y = position
    player_right = x > 3.05 if y < NET_Y else x < 3.05
    return "right" if player_right else "left"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to serve_candidates.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        load_job(job_path)
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "serve_candidates.json")
        players = read_json(players_path)
        rows = players.get("frames", [])
        times = [float(row["t"]) for row in rows]
        if not rows:
            raise ValueError("players.json contains no frames.")
        r_smooth = smooth_positions(rows, "r")
        o_smooth = smooth_positions(rows, "o")
        r_speed = speeds(times, r_smooth)
        o_speed = speeds(times, o_smooth)
        ready = []
        for index, (r, o) in enumerate(zip(r_smooth, o_smooth)):
            if r is None or o is None or not in_service_area(r) or not in_service_area(o):
                ready.append(False)
                continue
            if (r[1] < NET_Y) == (o[1] < NET_Y):
                ready.append(False)
                continue
            r_limit = 0.7 if r[1] < NET_Y else 0.9
            o_limit = 0.7 if o[1] < NET_Y else 0.9
            ready.append(
                r_speed[index] is not None
                and o_speed[index] is not None
                and r_speed[index] < r_limit
                and o_speed[index] < o_limit
            )

        runs: list[tuple[int, int]] = []
        start = None
        for index, is_ready in enumerate(ready + [False]):
            if is_ready and start is None:
                start = index
            elif not is_ready and start is not None:
                end = index - 1
                if times[end] - times[start] >= 0.8:
                    runs.append((start, end))
                start = None

        candidates = []
        for start, end in runs:
            center = (start + end) // 2
            candidates.append(
                {
                    "start": times[start],
                    "end": times[end],
                    "t": times[center],
                    "r": r_smooth[center],
                    "o": o_smooth[center],
                    "r_own_side": own_side(r_smooth[center]),
                    "o_own_side": own_side(o_smooth[center]),
                    "samples": end - start + 1,
                }
            )

        for index, candidate in enumerate(candidates, start=1):
            candidate["n"] = index

        write_json(
            output_path,
            {
                "schema_version": 1,
                "fps": players.get("fps", 10),
                "method": "stationary players in opposite service areas; candidates require visual confirmation",
                "candidates": candidates,
            },
        )
        print(f"Found {len(candidates)} likely serve pauses; saved to {output_path}.")
        print("These are review candidates, not confirmed serves or server assignments.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Serve detection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
