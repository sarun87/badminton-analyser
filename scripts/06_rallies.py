#!/usr/bin/env python3
"""Create timestamped point sheets and record visually reviewed rally ends."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from common import artifact_path, create_contact_sheet, load_job, prompt_float, proxy_path, read_json, video_properties, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--score", help="Score replay JSON; defaults to score.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to points.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        score_path = Path(args.score).expanduser().resolve() if args.score else artifact_path(job_path, "score.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "points.json")
        score = read_json(score_path)
        points = score.get("points", [])
        if not points:
            raise ValueError("score.json contains no points.")
        duration = float(video_properties(video)["duration_seconds"])
        review_dir = artifact_path(job_path, "review/rallies")
        reviewed = []
        for index, point in enumerate(points):
            start = float(point["t"])
            end = float(point.get("next_t", duration))
            if end <= start:
                raise ValueError(f"Point {point['n']} has an invalid serve interval.")
            timestamps = np.arange(start, end + 0.001, 0.5).tolist()
            if not timestamps or timestamps[-1] < end:
                timestamps.append(end)
            sheet = create_contact_sheet(
                video,
                timestamps,
                review_dir / f"point_{int(point['n']):03d}.jpg",
                f"Point {point['n']}  |  serve {start:.2f}s  |  next serve {end:.2f}s",
            )
            print(f"\nInspect {sheet}; enter the last live-play timestamp, before the pause/next serve.")
            rally_end = prompt_float(
                f"Point {point['n']} rally end in seconds [{start:.2f}..{end:.2f}]: ",
                minimum=start,
                maximum=end,
            )
            reviewed_point = dict(point)
            reviewed_point["r0"] = start
            reviewed_point["r1"] = rally_end
            reviewed_point["rdur"] = round(max(0.0, rally_end - start), 3)
            reviewed_point["rally_review_sheet"] = str(sheet)
            reviewed.append(reviewed_point)

        write_json(
            output_path,
            {
                "schema_version": 1,
                "score_order": score.get("score_order"),
                "rally_end_method": "manual visual review of 0.5 second contact sheets",
                "points": reviewed,
            },
        )
        print(f"Saved {len(reviewed)} visually reviewed rally intervals to {output_path}.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Rally review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
