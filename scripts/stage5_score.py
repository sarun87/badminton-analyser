#!/usr/bin/env python3
"""Replay singles badminton scoring and validate serve-court parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PLAYERS = ("R", "O")


def score_text(score: dict[str, int]) -> str:
    return f"{score['R']}-{score['O']}"


def parse_score(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain integer R and O scores")
    result = {}
    for player in PLAYERS:
        item = value.get(player)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label}.{player} must be a non-negative integer")
        result[player] = item
    return result


def load_events(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError("serves JSON must contain an events array")
    result = {}
    for event in events:
        candidate = event.get("candidate_n", event.get("n"))
        if not isinstance(candidate, int) or candidate <= 0 or candidate in result:
            raise ValueError(f"invalid or duplicate candidate_n: {candidate}")
        result[candidate] = event
    return result


def load_review(path: Path, events: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("score_order", "R-O") != "R-O":
        raise ValueError("score_order must be R-O")
    raw_rows = data.get("serves")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("manual review must contain a non-empty serves array")
    rows = []
    used = set()
    previous_t = None
    for n, raw in enumerate(raw_rows, 1):
        if not isinstance(raw, dict) or raw.get("n", n) != n:
            raise ValueError(f"serves[{n}] must be an object numbered {n}")
        server, side = raw.get("server"), raw.get("side")
        if server not in PLAYERS or side not in ("left", "right"):
            raise ValueError(f"serves[{n}] needs server R/O and side left/right")
        before = parse_score(raw.get("score_before"), f"serves[{n}].score_before")
        candidate = raw.get("candidate_n")
        dummy_point = raw.get("dummy_point", False)
        if not isinstance(dummy_point, bool):
            raise ValueError(f"serves[{n}].dummy_point must be true or false")
        if dummy_point and candidate is not None:
            raise ValueError(f"serves[{n}] cannot be both a detected serve and a dummy point")
        event = None
        if candidate is not None:
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate not in events:
                raise ValueError(f"serves[{n}] references unknown candidate_n {candidate}")
            if candidate in used:
                raise ValueError(f"candidate_n {candidate} is mapped more than once")
            used.add(candidate)
            event = events[candidate]
            timestamp = float(event["t"])
        else:
            timestamp = raw.get("timestamp_sec")
            timestamp = None if timestamp is None else float(timestamp)
        if dummy_point and timestamp != 0:
            raise ValueError(f"serves[{n}] dummy_point must use timestamp_sec 0")
        if timestamp == 0 and not dummy_point:
            raise ValueError(f"serves[{n}] timestamp_sec 0 is reserved for dummy points")
        if timestamp is not None and timestamp != 0 and previous_t is not None and timestamp <= previous_t:
            raise ValueError("known timestamps must increase in serve order")
        if timestamp is not None and timestamp != 0:
            previous_t = timestamp
        rows.append({
            "n": n, "candidate_n": candidate, "server": server, "side": side,
            "score_before": before, "t": timestamp, "event": event,
            "note": raw.get("note"), "dummy_point": dummy_point,
        })
    if used != set(events):
        raise ValueError(f"manual review must map all accepted candidates exactly once; missing={sorted(set(events)-used)}")
    meta = {
        "initial_score": parse_score(data.get("initial_score", {"R": 0, "O": 0}), "initial_score"),
        "final_point_winner": data.get("final_point_winner"),
        "final_score": None if data.get("final_score") is None else parse_score(data["final_score"], "final_score"),
        "notes": data.get("notes", []),
    }
    if meta["final_point_winner"] not in (None, *PLAYERS):
        raise ValueError("final_point_winner must be R, O, or null")
    return rows, meta


def automatic_rows(events: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(events.values(), key=lambda event: float(event["t"]))
    score = {"R": 0, "O": 0}
    rows = []
    for i, event in enumerate(ordered):
        rows.append({
            "n": i + 1, "candidate_n": event.get("candidate_n", event.get("n")),
            "server": event.get("server"), "side": event.get("side"),
            "score_before": dict(score), "t": float(event["t"]), "event": event, "note": None,
            "dummy_point": False,
        })
        if i + 1 < len(ordered) and ordered[i + 1].get("server") in PLAYERS:
            score[ordered[i + 1]["server"]] += 1
    return rows, {"initial_score": {"R": 0, "O": 0}, "final_point_winner": None, "final_score": None, "notes": []}


def replay(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    issues = []
    points = []
    contradictions = []
    side_passed = 0
    replay_score = dict(meta["initial_score"])
    for i, row in enumerate(rows):
        before = row["score_before"]
        dummy_point = row["dummy_point"]
        if before != replay_score:
            issues.append(f"s{row['n']}: listed {score_text(before)} but replay reached {score_text(replay_score)}")
        expected_side = None if dummy_point else ("right" if before[row["server"]] % 2 == 0 else "left")
        side_ok = None if dummy_point else row["side"] == expected_side
        if side_ok:
            side_passed += 1
        elif not dummy_point:
            contradictions.append({
                "serve_n": row["n"], "candidate_n": row["candidate_n"], "server": row["server"],
                "side": row["side"], "score_before": score_text(before), "expected_side": expected_side,
                "reason": "server's side does not match score parity",
            })
        winner = None
        next_server = None
        if i + 1 < len(rows):
            next_row = rows[i + 1]
            next_server = next_row["server"]
            delta = {p: next_row["score_before"][p] - before[p] for p in PLAYERS}
            if delta == {"R": 1, "O": 0}:
                winner = "R"
            elif delta == {"R": 0, "O": 1}:
                winner = "O"
            else:
                issues.append(f"s{row['n']} to s{next_row['n']}: score must rise by one point, got {score_text(before)} to {score_text(next_row['score_before'])}")
            if winner and next_server != winner:
                issues.append(f"s{next_row['n']}: next server {next_server} differs from prior winner {winner}")
        else:
            winner = meta["final_point_winner"]
        after = dict(before)
        if winner in PLAYERS:
            after[winner] += 1
        event = row["event"]
        points.append({
            "n": row["n"], "candidate_n": row["candidate_n"], "t": row["t"],
            "serve_window": {"start_t": event.get("start_t"), "end_t": event.get("end_t")} if event else None,
            "server": row["server"], "side": row["side"], "side_expected": expected_side,
            "side_rule_ok": side_ok, "before": score_text(before), "score_before": before,
            "winner": winner, "after": score_text(after), "score_after": after,
            "next_server": next_server,
            "serve_review_status": (
                "dummy_score_point_not_in_video" if dummy_point else
                event.get("review_status", "detected") if event else "manual_row_without_detected_candidate"
            ),
            "detected_guess": {
                "server": event.get("server"),
                "side": event.get("side"),
                "tracknet_server_cue": event.get("tracknet_serve_cue"),
            } if event else None,
            "manual_note": row["note"], "dummy_point": dummy_point,
        })
        replay_score = after
    final_ok = meta["final_score"] is None or replay_score == meta["final_score"]
    if not final_ok:
        issues.append(f"final score {score_text(replay_score)} differs from supplied {score_text(meta['final_score'])}")
    missing_times = [p["n"] for p in points if p["t"] is None and not p["dummy_point"]]
    recorded_serve_count = sum(not p["dummy_point"] for p in points)
    return {
        "players": {"R": "Stage 3 selected player", "O": "opponent"},
        "score_order": "R-O",
        "scoring_rule": "winner of a rally serves the next rally",
        "service_side_rule": "even server score = right; odd server score = left",
        "initial_score": meta["initial_score"], "serve_count": recorded_serve_count,
        "point_count": len(points), "dummy_point_count": len(points) - recorded_serve_count,
        "known_point_winners": sum(p["winner"] is not None for p in points),
        "final_point_winner": meta["final_point_winner"], "final_score": replay_score,
        "service_side_checks": {"passed": side_passed, "checked": recorded_serve_count, "contradictions": contradictions},
        "score_checks": {"transitions_ok": not issues, "final_score_ok": final_ok, "issues": issues, "timestamps_missing_for_serves": missing_times},
        "stage5_passed": not issues and not contradictions,
        "stage6_timeline_ready": not missing_times, "manual_review_notes": meta["notes"], "points": points,
    }


def print_report(data: dict[str, Any]) -> None:
    print("Stage 5 replay (score order R-O)")
    for point in data["points"]:
        t = "missing" if point["t"] is None else f"{point['t']:.1f}s"
        candidate = "-" if point["candidate_n"] is None else str(point["candidate_n"])
        check = "DUMMY POINT" if point["dummy_point"] else ("PASS" if point["side_rule_ok"] else f"CHECK expected {point['side_expected']}")
        print(f"s{point['n']:02d} t={t:>10} cand={candidate:>2} {point['server']}/{point['side']} @{point['before']} side={check}; winner={point['winner'] or '?'}; after={point['after']}")
    side = data["service_side_checks"]
    score = data["score_checks"]
    print(f"Side rule: {side['passed']}/{side['checked']} passed; {len(side['contradictions'])} contradiction(s).")
    print(f"Recorded serves: {data['serve_count']}; scored points: {data['point_count']}; dummy points: {data['dummy_point_count']}.")
    print(f"Score replay: {'PASS' if score['transitions_ok'] else 'CHECK'}; final score {score_text(data['final_score'])} ({'PASS' if score['final_score_ok'] else 'CHECK'}).")
    for issue in score["issues"]:
        print(f"  SCORE ISSUE: {issue}")
    for n in score["timestamps_missing_for_serves"]:
        before_t = next((p["t"] for p in reversed(data["points"][:n-1]) if p["t"] is not None), None)
        after_t = next((p["t"] for p in data["points"][n:] if p["t"] is not None), None)
        bracket = f" (between {before_t:.1f}s and {after_t:.1f}s)" if before_t is not None and after_t is not None else ""
        print(f"  TIMESTAMP NEEDED: s{n}{bracket}.")
    print(f"Stage 5: {'PASS' if data['stage5_passed'] else 'NEEDS REVIEW'}.")
    if not data["stage6_timeline_ready"]:
        print("Stage 6 is not ready until every serve has a timestamp.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay singles scores and check service-court parity.")
    parser.add_argument("serves_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--manual-review", type=Path, help="JSON corrections for serve order, side, and score")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.serves_json.expanduser().resolve()
    output = args.output_json.expanduser().resolve()
    review = args.manual_review.expanduser().resolve() if args.manual_review else None
    if not source.is_file():
        parser.error(f"serves JSON does not exist: {source}")
    if review is not None and not review.is_file():
        parser.error(f"manual review JSON does not exist: {review}")
    if output in (source, review):
        parser.error("output path must differ from input paths")
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output} (pass --overwrite)")
    events = load_events(source)
    if not events and review is None:
        raise ValueError("serves JSON has no detected candidates; provide a manual review with timestamped serve rows")
    rows, meta = load_review(review, events) if review else automatic_rows(events)
    result = replay(rows, meta)
    serve_data = json.loads(source.read_text(encoding="utf-8"))
    result["player_identity"] = serve_data.get("player_identity", {})
    if review:
        result["manual_review_source"] = str(review)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_report(result)
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Stage 5 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
