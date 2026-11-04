#!/usr/bin/env python3
"""Suggest which tracked player served each pause using shuttle proximity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import artifact_path, load_job, read_json, video_properties, proxy_path, write_json


def box_distance(point: dict[str, Any], box: list[float] | None, width: int, height: int) -> float | None:
    if box is None:
        return None
    x1, y1, x2, y2 = (float(box[index]) for index in range(4))
    x, y = float(point["x"]), float(point["y"])
    left, right = x1 * width, x2 * width
    top, bottom = y1 * height, y2 * height
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return float(np.hypot(dx, dy))


def nearest_player_row(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["t"]) - seconds))


def analyze_candidate(
    candidate: dict[str, Any],
    tracks: list[dict[str, Any]],
    player_rows: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    start = float(candidate["start"])
    end = float(candidate["end"])
    window_start, window_end = start - 0.5, end + 1.5
    scores = {"R": 0.0, "O": 0.0}
    support = {"R": 0, "O": 0}
    useful_track_ids: set[int] = set()
    for track in tracks:
        track_start = float(track.get("start", 0))
        if track_start > window_end or float(track.get("end", 0)) < window_start:
            continue
        points = [point for point in track.get("points", []) if window_start <= float(point["t"]) <= window_end]
        if not points:
            continue
        # The first visible portion is most likely to still be close to the server.
        points = points[: min(len(points), 5)]
        track_used = False
        for point in points:
            row = nearest_player_row(player_rows, float(point["t"]))
            if row is None:
                continue
            distances = {
                "R": box_distance(point, row.get("rb"), width, height),
                "O": box_distance(point, row.get("ob"), width, height),
            }
            valid = {key: value for key, value in distances.items() if value is not None}
            if not valid:
                continue
            player = min(valid, key=valid.get)
            distance = valid[player]
            other = valid.get("O" if player == "R" else "R")
            if distance <= 115 and (other is None or other - distance >= 35):
                scores[player] += max(0.1, 1.0 - distance / 150.0) * float(point.get("confidence", 0.5))
                support[player] += 1
                track_used = True
        if track_used:
            useful_track_ids.add(int(track.get("id", -1)))

    total = scores["R"] + scores["O"]
    if total <= 0:
        suggestion, confidence = None, "low"
        margin = 0.0
    else:
        suggestion = max(scores, key=scores.get)
        share = scores[suggestion] / total
        other = "O" if suggestion == "R" else "R"
        margin = scores[suggestion] - scores[other]
        confidence = "high" if support[suggestion] >= 2 and share >= 0.8 and margin >= 0.8 else "low"
    return {
        "candidate_n": candidate.get("n"),
        "candidate_t": candidate.get("t"),
        "server_suggestion": suggestion,
        "confidence": confidence,
        "supporting_points": support,
        "proximity_scores": {key: round(value, 3) for key, value in scores.items()},
        "supporting_track_ids": sorted(useful_track_ids),
        "reason": "visible shuttle track points near one player's tracked box" if total else "no usable shuttle-to-player proximity evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--candidates", help="Serve candidates JSON; defaults to serve_candidates.json beside job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--shuttle", help="Shuttle JSON; defaults to shuttle_tracks.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to serving_analysis.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        candidates_path = Path(args.candidates).expanduser().resolve() if args.candidates else artifact_path(job_path, "serve_candidates.json")
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        shuttle_path = Path(args.shuttle).expanduser().resolve() if args.shuttle else artifact_path(job_path, "shuttle_tracks.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "serving_analysis.json")
        candidates = read_json(candidates_path).get("candidates", [])
        player_data = read_json(players_path)
        tracks = read_json(shuttle_path).get("tracklets", [])
        properties = video_properties(proxy_path(job))
        rows = player_data.get("frames", [])
        results = [
            analyze_candidate(candidate, tracks, rows, int(properties["width"]), int(properties["height"]))
            for candidate in candidates
        ]
        confident = sum(item["confidence"] == "high" for item in results)
        write_json(
            output_path,
            {
                "schema_version": 1,
                "method": "association of early shuttle detections with tracked player boxes",
                "suggestions": results,
                "high_confidence_count": confident,
                "limitations": [
                    "Suggestions depend on the experimental shuttle detector and do not confirm a racket contact.",
                    "Only high-confidence suggestions are eligible for automatic use; review low-confidence cases.",
                ],
            },
        )
        print(f"Saved serve-player suggestions to {output_path}; {confident}/{len(results)} are high confidence.")
        print("Stage 5 will ask you to identify the server where evidence is missing or uncertain.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Serve-player analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
