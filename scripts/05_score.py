#!/usr/bin/env python3
"""Review serve events and replay one game's singles score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    artifact_path,
    create_contact_sheet,
    load_job,
    prompt_choice,
    prompt_float,
    proxy_path,
    read_json,
    video_properties,
    write_json,
)


def review_candidates(job_path: Path, video: Path, candidates: list[dict], duration: float) -> list[dict]:
    review_dir = artifact_path(job_path, "review/serves")
    events = []
    for index, candidate in enumerate(candidates, start=1):
        start = max(0.0, float(candidate["start"]) - 0.5)
        end = min(duration, float(candidate["end"]) + 1.0)
        timestamps = np.arange(start, end + 0.001, 0.25).tolist()
        if len(timestamps) > 45:
            timestamps = timestamps[:: int(np.ceil(len(timestamps) / 45))]
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
        sheet = create_contact_sheet(
            video,
            [max(0.0, serve_time - 0.5), serve_time, min(duration, serve_time + 0.5)],
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


def score_input(prompt: str, default: str | None = None) -> tuple[int, int]:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default:
            raw = default
        try:
            user, opponent = (int(value) for value in raw.split("-", maxsplit=1))
        except ValueError:
            print("Enter a score such as 0-0 or 10-10.")
            continue
        if user >= 0 and opponent >= 0:
            return user, opponent
        print("Scores cannot be negative.")


def initial_score() -> tuple[int, int]:
    return score_input("Score before the first recorded point, user-opponent", "0-0")


def first_server() -> str:
    key = prompt_choice("Who served the first recorded point?", {"r": "You (R)", "o": "Opponent (O)"})
    return "R" if key == "r" else "O"


def assisted_events(
    job_path: Path,
    video: Path,
    candidates: list[dict[str, Any]],
    suggestions: dict[str, Any],
    duration: float,
    expected_count: int,
    declared_first_server: str,
    confirm_high: bool,
) -> list[dict[str, Any]]:
    if len(candidates) != expected_count:
        raise ValueError(
            f"Assisted review found {len(candidates)} serve candidates, but the starting and final scores imply "
            f"{expected_count} recorded points. Rerun with --review-mode manual to correct false or missed candidates."
        )
    print(
        f"\nStage 4 found {len(candidates)} candidate pauses; the score difference expects {expected_count} points. "
        "Matching counts are a useful check, but a missed serve and a false candidate can cancel each other out."
    )
    if input("Continue with assisted serve review? [Y/n]: ").strip().lower() == "n":
        raise ValueError("Assisted review cancelled. Rerun with --review-mode manual.")

    suggestions_by_n = {item.get("candidate_n"): item for item in suggestions.get("suggestions", [])}
    review_dir = artifact_path(job_path, "review/serves")
    events = []
    for index, candidate in enumerate(candidates):
        sheet = None
        candidate_n = candidate.get("n", index + 1)
        analysis = suggestions_by_n.get(candidate_n, {})
        suggested_server = analysis.get("server_suggestion")
        high = analysis.get("confidence") == "high" and suggested_server in {"R", "O"}
        if index == 0:
            server = declared_first_server
            source = "user-declared first server"
        elif high and not confirm_high:
            server = suggested_server
            source = "high-confidence shuttle proximity suggestion"
        else:
            start = max(0.0, float(candidate["start"]) - 0.5)
            end = min(duration, float(candidate["end"]) + 1.5)
            timestamps = np.arange(start, end + 0.001, 0.25).tolist()
            if len(timestamps) > 45:
                timestamps = timestamps[:: int(np.ceil(len(timestamps) / 45))]
            sheet = create_contact_sheet(
                video,
                timestamps,
                review_dir / f"assisted_{index + 1:03d}.jpg",
                f"Assisted serve review {index + 1}: {candidate['start']:.2f}-{candidate['end']:.2f}s",
            )
            why = "confirming high-confidence suggestions" if high else "server evidence is missing or uncertain"
            print(f"\nReview {sheet}; {why}.")
            default = "r" if suggested_server == "R" else "o" if suggested_server == "O" else None
            prompt = "Who served this point? [Enter accepts suggestion]" if default else "Who served this point?"
            key = prompt_choice(prompt, {"r": "You (R)", "o": "Opponent (O)"}) if not default else input(f"{prompt} (r/o): ").strip().lower()
            if not key:
                key = default
            while key not in {"r", "o"}:
                key = input("Choose r or o: ").strip().lower()
            server = "R" if key == "r" else "O"
            source = "manual confirmation"
        events.append(
            {
                "t": float(candidate.get("t", candidate["end"])),
                "server": server,
                "server_side": None,
                "review_sheet": str(sheet) if sheet else None,
                "candidate_n": candidate_n,
                "server_source": source,
                "server_confidence": analysis.get("confidence", "unknown"),
            }
        )
    events.sort(key=lambda event: event["t"])
    for previous, current in zip(events, events[1:]):
        if current["t"] - previous["t"] < 1.0:
            raise ValueError("Two serve candidate times are less than one second apart. Review Stage 4 candidates.")
    return events


def replay_manual(events: list[dict[str, Any]], initial_r: int, initial_o: int, duration: float) -> list[dict[str, Any]]:
    for index, event in enumerate(events):
        event["new_game"] = index == 0 and initial_r == 0 and initial_o == 0
        if index > 0:
            event["new_game"] = input(f"Does the serve at {event['t']:.2f}s start a new game? [y/N]: ").strip().lower() == "y"
    winners = []
    for index, event in enumerate(events):
        if index + 1 < len(events):
            winners.append(events[index + 1]["server"])
        else:
            key = prompt_choice("Who won the final recorded point?", {"r": "You (R)", "o": "Opponent (O)"})
            winners.append("R" if key == "r" else "O")

    score_r, score_o = initial_r, initial_o
    game_number = 1
    points = []
    for index, (event, winner) in enumerate(zip(events, winners), start=1):
        if index > 1 and event.get("new_game"):
            score_r, score_o = 0, 0
            game_number += 1
        before = f"{score_r}-{score_o}"
        server_score = score_r if event["server"] == "R" else score_o
        expected_side = "right" if server_score % 2 == 0 else "left"
        side_check = None if event.get("server_side") is None else event["server_side"] == expected_side
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
                "side": event.get("server_side"),
                "before": before,
                "winner": winner,
                "after": f"{score_r}-{score_o}",
                "side_check": side_check,
                "review_sheet": event.get("review_sheet"),
            }
        )
    return points


def replay_assisted(
    events: list[dict[str, Any]],
    initial_r: int,
    initial_o: int,
    final_r: int,
    final_o: int,
    duration: float,
) -> list[dict[str, Any]]:
    expected_count = (final_r + final_o) - (initial_r + initial_o)
    if expected_count != len(events):
        raise ValueError(f"Expected {expected_count} recorded points from the scores, but have {len(events)} serve events.")
    winners = [event["server"] for event in events[1:]]
    wins_before_last_r = sum(winner == "R" for winner in winners)
    wins_before_last_o = sum(winner == "O" for winner in winners)
    remaining_r = final_r - initial_r - wins_before_last_r
    remaining_o = final_o - initial_o - wins_before_last_o
    if (remaining_r, remaining_o) == (1, 0):
        winners.append("R")
    elif (remaining_r, remaining_o) == (0, 1):
        winners.append("O")
    else:
        raise ValueError(
            "The server sequence does not reconcile with the final score. Review the uncertain serve-player suggestions "
            "or rerun with --confirm-high-confidence or --review-mode manual."
        )

    score_r, score_o = initial_r, initial_o
    points = []
    for index, (event, winner) in enumerate(zip(events, winners), start=1):
        before = f"{score_r}-{score_o}"
        server_score = score_r if event["server"] == "R" else score_o
        side = "right" if server_score % 2 == 0 else "left"
        if winner == "R":
            score_r += 1
        else:
            score_o += 1
        points.append(
            {
                "n": index,
                "game": 1,
                "t": event["t"],
                "next_t": events[index]["t"] if index < len(events) else duration,
                "server": event["server"],
                "side": side,
                "before": before,
                "winner": winner,
                "after": f"{score_r}-{score_o}",
                "side_check": True,
                "review_sheet": event.get("review_sheet"),
                "server_source": event.get("server_source"),
                "server_confidence": event.get("server_confidence"),
            }
        )
    if (score_r, score_o) != (final_r, final_o):
        raise ValueError(f"Internal score replay ended at {score_r}-{score_o}, not the entered {final_r}-{final_o}.")
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--candidates", help="Serve candidate JSON; defaults to serve_candidates.json beside job.json.")
    parser.add_argument("--serving-analysis", help="Server suggestions JSON; defaults to serving_analysis.json beside job.json.")
    parser.add_argument("--review-mode", choices=("manual", "assisted"), default="manual", help="Assisted mode reviews uncertain servers and checks the score.")
    parser.add_argument("--confirm-high-confidence", action="store_true", help="Ask to confirm every high-confidence server suggestion.")
    parser.add_argument("--output", help="Output JSON; defaults to score.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        candidates_path = Path(args.candidates).expanduser().resolve() if args.candidates else artifact_path(job_path, "serve_candidates.json")
        analysis_path = Path(args.serving_analysis).expanduser().resolve() if args.serving_analysis else artifact_path(job_path, "serving_analysis.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "score.json")
        candidates = read_json(candidates_path).get("candidates", [])
        duration = float(video_properties(video)["duration_seconds"])
        if not candidates:
            raise ValueError("No serve candidates were found. Review Stage 4 or add serves in manual mode.")

        if args.review_mode == "manual":
            events = review_candidates(job_path, video, candidates, duration)
            add_manual_events(job_path, video, events, duration)
            if not events:
                raise ValueError("No serves were confirmed. Add them manually or rerun Stage 4.")
            for previous, current in zip(events, events[1:]):
                if current["t"] - previous["t"] < 1.0:
                    raise ValueError("Two confirmed serves are less than one second apart. Review duplicate candidates.")
            initial_r, initial_o = initial_score()
            declared = first_server()
            if events[0]["server"] != declared:
                print(f"Updating the first serve's server to {declared}, as entered.")
                events[0]["server"] = declared
            print("The winner of each point is inferred from the next point's server.")
            print("Mark game breaks and visually check the serve list; a missed event can shift later scores.")
            points = replay_manual(events, initial_r, initial_o, duration)
            score_summary = None
        else:
            initial_r, initial_o = initial_score()
            final_r, final_o = score_input("Final game score, user-opponent")
            delta_r, delta_o = final_r - initial_r, final_o - initial_o
            if delta_r < 0 or delta_o < 0 or delta_r + delta_o <= 0:
                raise ValueError("The final score must be at or above the starting score and include at least one recorded point.")
            expected_count = delta_r + delta_o
            declared = first_server()
            suggestions = read_json(analysis_path) if analysis_path.is_file() else {"suggestions": []}
            if not analysis_path.is_file():
                print("No serving_analysis.json found; Stage 5 will ask who served each point after the first.")
            events = assisted_events(
                job_path,
                video,
                candidates,
                suggestions,
                duration,
                expected_count,
                declared,
                args.confirm_high_confidence,
            )
            points = replay_assisted(events, initial_r, initial_o, final_r, final_o, duration)
            score_summary = {
                "starting_score": f"{initial_r}-{initial_o}",
                "final_score_entered": f"{final_r}-{final_o}",
                "expected_recorded_points": expected_count,
                "candidate_count": len(candidates),
                "count_matches": len(candidates) == expected_count,
            }

        write_json(
            output_path,
            {
                "schema_version": 1,
                "review_mode": args.review_mode,
                "score_order": "R-O; R is the appearance-matched user",
                "score_consistency": score_summary,
                "serve_events": events,
                "points": points,
            },
        )
        checked = [point["side_check"] for point in points if point["side_check"] is not None]
        print(f"Saved score replay for {len(points)} points to {output_path}.")
        if score_summary:
            print(f"Point count matches the entered start/final score: {score_summary['expected_recorded_points']} points.")
        if checked:
            print(f"Service-court parity: {sum(checked)}/{len(checked)} assigned points match the score rule.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Score replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
