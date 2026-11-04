#!/usr/bin/env python3
"""Estimate rally ends from player pauses or collect manual visual reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import artifact_path, create_contact_sheet, load_job, prompt_float, proxy_path, read_json, video_properties, write_json


def player_speeds(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    result: list[float | None] = [None] * len(rows)
    for index in range(1, len(rows) - 1):
        before, after = rows[index - 1].get(key), rows[index + 1].get(key)
        elapsed = float(rows[index + 1]["t"]) - float(rows[index - 1]["t"])
        if before is not None and after is not None and 0 < elapsed <= 0.5:
            result[index] = float(np.linalg.norm(np.asarray(after, dtype=float) - np.asarray(before, dtype=float)) / elapsed)
    return result


def suggest_rally_end(
    point: dict[str, Any],
    rows: list[dict[str, Any]],
    r_speed: list[float | None],
    o_speed: list[float | None],
) -> tuple[float | None, str, str]:
    start = float(point["t"])
    next_serve = float(point.get("next_t", start))
    run_start: int | None = None
    for index, row in enumerate(rows):
        seconds = float(row["t"])
        if seconds < start + 0.45 or seconds >= next_serve - 0.65:
            continue
        quiet = (
            row.get("r") is not None
            and row.get("o") is not None
            and r_speed[index] is not None
            and o_speed[index] is not None
            and r_speed[index] < 0.35
            and o_speed[index] < 0.35
        )
        if quiet and run_start is None:
            run_start = index
        elif not quiet and run_start is not None:
            run_end = index - 1
            duration = float(rows[run_end]["t"]) - float(rows[run_start]["t"])
            if duration >= 0.8:
                confidence = "high" if duration >= 1.2 and next_serve - float(rows[run_start]["t"]) >= 2.0 else "low"
                return float(rows[run_start]["t"]), confidence, f"both tracked players stayed below 0.35 m/s for {duration:.1f}s"
            run_start = None
    if run_start is not None:
        run_end = len(rows) - 1
        while run_end >= run_start and float(rows[run_end]["t"]) >= next_serve - 0.65:
            run_end -= 1
        if run_end >= run_start:
            duration = float(rows[run_end]["t"]) - float(rows[run_start]["t"])
            if duration >= 0.8:
                confidence = "high" if duration >= 1.2 and next_serve - float(rows[run_start]["t"]) >= 2.0 else "low"
                return float(rows[run_start]["t"]), confidence, f"both tracked players stayed below 0.35 m/s for {duration:.1f}s"
    return None, "low", "no clear shared post-rally pause found"


def manual_end(job_path: Path, video: Path, point: dict[str, Any], duration: float, number: int) -> tuple[float, Path]:
    start = float(point["t"])
    end = float(point.get("next_t", duration))
    timestamps = np.arange(start, end + 0.001, 0.5).tolist()
    if not timestamps or timestamps[-1] < end:
        timestamps.append(end)
    sheet = create_contact_sheet(
        video,
        timestamps,
        artifact_path(job_path, f"review/rallies/point_{int(point['n']):03d}.jpg"),
        f"Point {point['n']}  |  serve {start:.2f}s  |  next serve {end:.2f}s",
    )
    print(f"\nInspect {sheet}; enter the last live-play timestamp before the pause/next serve.")
    rally_end = prompt_float(
        f"Point {point['n']} rally end in seconds [{start:.2f}..{end:.2f}]: ",
        minimum=start,
        maximum=end,
    )
    return rally_end, sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--score", help="Score replay JSON; defaults to score.json beside job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--review-mode", choices=("manual", "assisted"), default="manual", help="Assisted mode estimates clear stop pauses and reviews uncertain points.")
    parser.add_argument("--review-all", action="store_true", help="In assisted mode, request visual confirmation for every estimate.")
    parser.add_argument("--output", help="Output JSON; defaults to points.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        score_path = Path(args.score).expanduser().resolve() if args.score else artifact_path(job_path, "score.json")
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "points.json")
        score = read_json(score_path)
        points = score.get("points", [])
        if not points:
            raise ValueError("score.json contains no points.")
        duration = float(video_properties(video)["duration_seconds"])
        rows = read_json(players_path).get("frames", [])
        r_speed = player_speeds(rows, "r") if rows else []
        o_speed = player_speeds(rows, "o") if rows else []
        reviewed = []
        auto_count = 0
        manual_count = 0
        for index, point in enumerate(points):
            start = float(point["t"])
            end = float(point.get("next_t", duration))
            if end <= start:
                raise ValueError(f"Point {point['n']} has an invalid serve interval.")
            estimate, confidence, reason = (None, "low", "player tracks are unavailable")
            if args.review_mode == "assisted" and rows:
                estimate, confidence, reason = suggest_rally_end(point, rows, r_speed, o_speed)
            use_manual = args.review_mode == "manual" or estimate is None or confidence != "high" or args.review_all
            reviewed_point = dict(point)
            if use_manual:
                if args.review_mode == "assisted" and estimate is not None:
                    print(f"Point {point['n']} auto estimate {estimate:.2f}s is low confidence: {reason}.")
                rally_end, sheet = manual_end(job_path, video, point, duration, index + 1)
                reviewed_point["rally_review_sheet"] = str(sheet)
                reviewed_point["rally_end_method"] = "manual visual review"
                manual_count += 1
            else:
                rally_end = estimate
                reviewed_point["rally_end_method"] = "assisted player-pause estimate"
                reviewed_point["rally_end_confidence"] = confidence
                reviewed_point["rally_end_evidence"] = reason
                auto_count += 1
            rally_end = min(end, max(start, float(rally_end)))
            reviewed_point["r0"] = start
            reviewed_point["r1"] = rally_end
            reviewed_point["rdur"] = round(max(0.0, rally_end - start), 3)
            reviewed.append(reviewed_point)

        write_json(
            output_path,
            {
                "schema_version": 1,
                "score_order": score.get("score_order"),
                "review_mode": args.review_mode,
                "rally_end_method": "manual visual review or confidence-gated shared-player-pause estimate",
                "auto_estimated_points": auto_count,
                "manually_reviewed_points": manual_count,
                "limitations": [
                    "A pause during a rally can resemble a point ending; inspect automated estimates in the video.",
                    "Player-tracker gaps or incorrect identities can force a manual review or shift an estimate.",
                ],
                "points": reviewed,
            },
        )
        print(f"Saved rally intervals to {output_path}: {auto_count} estimated, {manual_count} manually reviewed.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Rally review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
