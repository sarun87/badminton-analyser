#!/usr/bin/env python3
"""Review serve candidates and replay singles scoring from confirmed servers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from common import artifact_path, create_contact_sheet, load_job, prompt_choice, prompt_float, proxy_path, read_json, video_properties, write_json


def review_candidates(job_path: Path, video: Path, candidates: list[dict], duration: float) -> list[dict]:
    review_dir = artifact_path(job_path, "review/serves")
    events = []
    for index, candidate in enumerate(candidates, start=1):
        start = max(0.0, float(candidate["start"]) - 0.5)
        end = min(duration, float(candidate["end"]) + 1.0)
        timestamps = np.arange(start, end + 0.001, 0.25).tolist()
        if len(timestamps) > 45:
            stride = int(np.ceil(len(timestamps) / 45))
            timestamps = timestamps[::stride]
        sheet = create_contact_sheet(
            video,
            timestamps,
            review_dir / f"candidate_{index:03d}.jpg",
            f"Serve candidate {index}: {candidate['start']:.2f}-{candidate['end']:.2f}s",
        )
        print(f"\nReview {sheet} and confirm whether this is a serve.")
        raw = input(f"Serve timestamp in seconds [suggested {candidate['t']:.2f}; x to discard]: ").strip().lower()
        if raw == "x":
            continue
        try:
            serve_time = float(raw) if raw else float(candidate["t"])
        except ValueError:
            print("Invalid timestamp; candidate skipped.")
            continue
        if not max(0.0, float(candidate["start"]) - 1.0) <= serve_time <= min(duration, float(candidate["end"]) + 2.0):
            print("Timestamp is outside this candidate window; candidate skipped.")
            continue
        server_key = prompt_choice("Who served this point?", {"r": "You (R)", "o": "Opponent (O)"})
        side_key = prompt_choice(
            "Which service court was the server in, from the server's perspective?",
            {"r": "Right", "l": "Left", "u": "Unknown / cannot tell"},
        )
        events.append(
            {
                "t": serve_time,
                "server": "R" if server_key == "r" else "O",
                "server_side": {"r": "right", "l": "left", "u": None}[side_key],
                "review_sheet": str(sheet),
            }
        )
    events.sort(key=lambda event: event["t"])
    return events


def add_manual_events(job_path: Path, video: Path, events: list[dict], duration: float) -> None:
    review_dir = artifact_path(job_path, "review/serves")
    while input("Add a manually located serve missed by Stage 4? [y/N]: ").strip().lower() == "y":
        serve_time = prompt_float("Serve timestamp in seconds: ", minimum=0, maximum=duration)
        timestamps = [max(0.0, serve_time - 0.5), serve_time, min(duration, serve_time + 0.5)]
        sheet = create_contact_sheet(
            video,
            timestamps,
            review_dir / f"manual_{len(events) + 1:03d}.jpg",
            f"Manually located serve at {serve_time:.2f}s",
        )
        print(f"Review {sheet}.")
        server_key = prompt_choice("Who served this point?", {"r": "You (R)", "o": "Opponent (O)"})
        side_key = prompt_choice(
            "Which service court was the server in, from the server's perspective?",
            {"r": "Right", "l": "Left", "u": "Unknown / cannot tell"},
        )
        events.append(
            {
                "t": serve_time,
                "server": "R" if server_key == "r" else "O",
                "server_side": {"r": "right", "l": "left", "u": None}[side_key],
                "review_sheet": str(sheet),
                "manually_added": True,
            }
        )
    events.sort(key=lambda event: event["t"])


def initial_score() -> tuple[int, int]:
    while True:
        raw = input("Score before the first recorded point, user-opponent [0-0]: ").strip()
        if not raw:
            return 0, 0
        try:
            user, opponent = (int(value) for value in raw.split("-", maxsplit=1))
        except ValueError:
            print("Enter a score such as 0-0 or 8-12.")
            continue
        if user >= 0 and opponent >= 0:
            return user, opponent
        print("Scores cannot be negative.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--candidates", help="Serve candidate JSON; defaults to serve_candidates.json beside job.json.")
    parser.add_argument("--output", help="Output JSON; defaults to score.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        candidates_path = Path(args.candidates).expanduser().resolve() if args.candidates else artifact_path(job_path, "serve_candidates.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "score.json")
        candidates = read_json(candidates_path).get("candidates", [])
        duration = float(video_properties(video)["duration_seconds"])
        events = review_candidates(job_path, video, candidates, duration)
        add_manual_events(job_path, video, events, duration)
        for previous, current in zip(events, events[1:]):
            if current["t"] - previous["t"] < 1.0:
                raise ValueError("Two confirmed serves are less than one second apart. Review duplicate candidates.")
        if not events:
            raise ValueError("No serves were confirmed. Add them manually or rerun Stage 4.")

        initial_r, initial_o = initial_score()
        print("\nThe winner of each point is inferred from the server of the next point.")
        print("Check that Stage 4 found every serve; a missed serve can shift the replay.")
        events[0]["new_game"] = initial_r == 0 and initial_o == 0
        for event in events[1:]:
            event["new_game"] = input(f"Does the serve at {event['t']:.2f}s start a new game? [y/N]: ").strip().lower() == "y"
        winners = []
        for index, event in enumerate(events):
            if index + 1 < len(events):
                winner = events[index + 1]["server"]
            else:
                winner_key = prompt_choice("Who won the final recorded point?", {"r": "You (R)", "o": "Opponent (O)"})
                winner = "R" if winner_key == "r" else "O"
            winners.append(winner)

        score_r = initial_r
        score_o = initial_o
        game_number = 1
        points = []
        for index, (event, winner) in enumerate(zip(events, winners), start=1):
            if index > 1 and event.get("new_game"):
                score_r = 0
                score_o = 0
                game_number += 1
            before = f"{score_r}-{score_o}"
            server_score = score_r if event["server"] == "R" else score_o
            expected_side = "right" if server_score % 2 == 0 else "left"
            side_check = None if event["server_side"] is None else event["server_side"] == expected_side
            if winner == "R":
                score_r += 1
            else:
                score_o += 1
            points.append(
                {
                    "n": index,
                    "game": game_number,
                    "t": event["t"],
                    "next_t": events[index]["t"] if index < len(events) else duration,
                    "server": event["server"],
                    "side": event["server_side"],
                    "before": before,
                    "winner": winner,
                    "after": f"{score_r}-{score_o}",
                    "side_check": side_check,
                    "review_sheet": event["review_sheet"],
                }
            )

        write_json(
            output_path,
            {
                "schema_version": 1,
                "score_order": "R-O; R is the appearance-matched user",
                "serve_events": events,
                "points": points,
            },
        )
        checked = [point["side_check"] for point in points if point["side_check"] is not None]
        print(f"Saved score replay for {len(points)} points to {output_path}.")
        if checked:
            print(f"Service-court rule: {sum(checked)}/{len(checked)} reviewed serves match the score parity.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Score replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
