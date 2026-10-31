#!/usr/bin/env python3
"""Build a local, evidence-linked match report and rule-based coaching notes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from common import artifact_path, load_job, read_json, timestamp_label


def rally_bucket(seconds: float) -> str:
    if seconds <= 5:
        return "5 s or less"
    if seconds <= 10:
        return "5–10 s"
    return "over 10 s"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def number(value: float | None, unit: str = "") -> str:
    return "—" if value is None else f"{value:.2f}{unit}"


def build_report(
    points_data: dict[str, Any],
    motion: dict[str, Any],
    pose: dict[str, Any],
    pickups: dict[str, Any],
) -> tuple[str, list[str]]:
    points = points_data.get("points", [])
    metrics_by_point = {int(point["n"]): point for point in motion.get("per_point", [])}
    pickup_by_point = {int(point["n"]): point for point in pickups.get("points", [])}
    won = [point for point in points if point.get("winner") == "R"]
    lost = [point for point in points if point.get("winner") == "O"]
    total = len(points)

    lines = [
        "# Badminton Match Analysis",
        "",
        f"Points reviewed: **{total}** · user won **{len(won)}** · user lost **{len(lost)}**",
        "",
        "This report is generated locally from reviewed score and rally timestamps. Serve detection, rally ends, and pickup-side estimates need visual confirmation.",
        "",
        "## Results by rally length",
        "",
        "| Rally length | Points | User won | User lost | Loss rate |",
        "|---|---:|---:|---:|---:|",
    ]
    bucket_summary = {}
    for bucket in ("5 s or less", "5–10 s", "over 10 s"):
        group = [point for point in points if rally_bucket(float(point["rdur"])) == bucket]
        group_won = sum(point.get("winner") == "R" for point in group)
        group_lost = len(group) - group_won
        loss_rate = group_lost / len(group) if group else None
        bucket_summary[bucket] = {"points": len(group), "won": group_won, "lost": group_lost, "loss_rate": loss_rate}
        lines.append(f"| {bucket} | {len(group)} | {group_won} | {group_lost} | {pct(loss_rate)} |")

    lines.extend(
        [
            "",
            "## Movement",
            "",
            "| Outcome | Points with movement data | Distance per rally | Average speed | Peak speed | Front-court share | Back-court share |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for outcome in ("won", "lost"):
        summary = motion.get("user_by_outcome", {}).get(outcome, {})
        lines.append(
            f"| {outcome.title()} | {summary.get('points', 0)} | {number(summary.get('distance_m'), ' m')} | "
            f"{number(summary.get('average_speed_mps'), ' m/s')} | {number(summary.get('peak_speed_mps'), ' m/s')} | "
            f"{pct(summary.get('front_court_share'))} | {pct(summary.get('back_court_share'))} |"
        )
    heatmaps = motion.get("heatmaps", {})
    for player, path in heatmaps.items():
        lines.append(f"\nPlayer {player.upper()} heatmap: `{path}`")

    lines.extend(["", "## Body position", ""])
    pose_summary = pose.get("summary_by_outcome_and_context", {})
    lines.extend(
        [
            "| Outcome | Context | Samples | Crouch delta | Stance width / shoulder width | Racket-up rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for outcome in ("won", "lost"):
        for context in ("READY", "FRONT", "REAR"):
            summary = pose_summary.get(outcome, {}).get(context, {})
            lines.append(
                f"| {outcome.title()} | {context} | {summary.get('samples', 0)} | "
                f"{number(summary.get('crouch_delta'))} | {number(summary.get('stance_width_ratio'))} | "
                f"{pct(summary.get('racket_arm_up_rate'))} |"
            )

    lines.extend(["", "## Point-ending pickup clues", "", "| Point | Outcome | Shuttle-side estimate | Evidence |", "|---:|---|---|---|"])
    for point in points:
        pickup = pickup_by_point.get(int(point["n"]), {})
        evidence = pickup.get("evidence")
        clue = f"{timestamp_label(float(evidence['t']))}, {evidence['confidence']} confidence" if evidence else "No bend detected"
        lines.append(
            f"| {point['n']} | {'Won' if point['winner'] == 'R' else 'Lost'} | "
            f"{pickup.get('shuttle_side_estimate') or 'Unknown'} | {clue} |"
        )

    lost_examples = sorted(lost, key=lambda point: float(point.get("rdur", 0)), reverse=True)[:8]
    lines.extend(["", "## Lost-point review timestamps", ""])
    if lost_examples:
        for point in lost_examples:
            lines.append(
                f"- Point {point['n']} at **{timestamp_label(float(point['t']))}**; rally {float(point['rdur']):.1f}s; "
                f"score {point.get('before', '?')}. Review `{point.get('rally_review_sheet', 'contact sheet not found')}`."
            )
    else:
        lines.append("No lost points were recorded.")

    recommendations = []
    overall_loss_rate = len(lost) / total if total else 0.0
    long_bucket = bucket_summary["over 10 s"]
    if long_bucket["points"] >= 3 and long_bucket["loss_rate"] is not None and long_bucket["loss_rate"] > overall_loss_rate + 0.15:
        examples = [point for point in lost if float(point["rdur"]) > 10][:3]
        refs = ", ".join(f"point {point['n']} ({timestamp_label(float(point['t']))})" for point in examples)
        recommendations.append(
            f"Long rallies had a higher loss rate ({pct(long_bucket['loss_rate'])} vs {pct(overall_loss_rate)} overall). Review {refs}; check recovery between shots and whether you can keep a neutral base position."
        )

    won_motion = motion.get("user_by_outcome", {}).get("won", {})
    lost_motion = motion.get("user_by_outcome", {}).get("lost", {})
    won_distance, lost_distance = won_motion.get("distance_m"), lost_motion.get("distance_m")
    if won_distance and lost_distance and lost_distance > won_distance * 1.15:
        examples = [point for point in lost if metrics_by_point.get(int(point["n"]), {}).get("r", {}).get("distance_m") is not None]
        refs = ", ".join(f"{point['n']} ({timestamp_label(float(point['t']))})" for point in examples[:3])
        recommendations.append(
            f"Lost rallies involved more movement per point ({lost_distance:.1f}m vs {won_distance:.1f}m on wins). Review points {refs}; look for late recovery or repeated displacement before choosing a technical fix."
        )

    ready_won = pose_summary.get("won", {}).get("READY", {})
    ready_lost = pose_summary.get("lost", {}).get("READY", {})
    won_ready_rate = ready_won.get("racket_arm_up_rate")
    lost_ready_rate = ready_lost.get("racket_arm_up_rate")
    if won_ready_rate is not None and lost_ready_rate is not None and won_ready_rate > lost_ready_rate + 0.15:
        ready_losses = [sample for sample in pose.get("samples", []) if sample.get("outcome") == "lost" and sample.get("context") == "READY" and sample.get("racket_arm_up") is False]
        refs = ", ".join(f"point {sample['point']} ({timestamp_label(float(sample['t']))})" for sample in ready_losses[:3])
        recommendations.append(
            f"The racket-up rate in READY samples was lower on lost points ({pct(lost_ready_rate)}) than wins ({pct(won_ready_rate)}). Review {refs}; check preparation before the opponent's shot."
        )

    near_lost = [
        pickup_by_point.get(int(point["n"]), {}).get("shuttle_side_estimate")
        for point in lost
        if pickup_by_point.get(int(point["n"]), {}).get("shuttle_side_estimate") is not None
    ]
    if len(near_lost) >= 3 and near_lost.count("near") / len(near_lost) >= 0.67:
        examples = [point for point in lost if pickup_by_point.get(int(point["n"]), {}).get("shuttle_side_estimate") == "near"][:3]
        refs = ", ".join(f"point {point['n']} ({timestamp_label(float(point['t']))})" for point in examples)
        recommendations.append(
            f"Most classified lost points ended on the near half ({near_lost.count('near')}/{len(near_lost)} pickup clues). Treat this as a low-confidence clue and visually review {refs} before concluding that net or defensive coverage is the main issue."
        )

    lines.extend(["", "## Coaching notes", ""])
    if recommendations:
        lines.extend(f"- {recommendation}" for recommendation in recommendations)
    else:
        lines.append("The current metrics do not show a strong repeated pattern. Review the lost-point timestamps above and record manual observations before changing training priorities.")
    lines.extend(
        [
            "",
            "## Data limits",
            "",
            "Serve and player tracking are detector-based; visually review them. Rally end times were entered manually. Pose and pickup-side values are approximate and should be treated as prompts for video review, not definitive causes.",
            "",
        ]
    )
    return "\n".join(lines), recommendations


def local_model_draft(report: str, model: str) -> str:
    prompt = (
        "Write a concise badminton coaching summary using only the evidence in this report. "
        "Do not invent events or causes. Cite point numbers and timestamps for every observation. "
        "Separate measured patterns from hypotheses and give at most three practical training priorities.\n\n"
        + report
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}).encode()
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800) as response:
            result = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError("Could not reach local Ollama at 127.0.0.1:11434; start Ollama or omit --ollama-model.") from exc
    answer = result.get("response", "").strip()
    if not answer:
        raise RuntimeError("The local model returned an empty coaching draft.")
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--points", help="Reviewed points JSON; defaults to points.json beside job.json.")
    parser.add_argument("--motion", help="Movement JSON; defaults to motion.json beside job.json.")
    parser.add_argument("--pose", help="Pose JSON; defaults to pose_stats.json beside job.json.")
    parser.add_argument("--pickups", help="Pickup JSON; defaults to pickups.json beside job.json.")
    parser.add_argument("--output", help="Output Markdown; defaults to report.md beside job.json.")
    parser.add_argument("--ollama-model", help="Optional locally installed Ollama model, for example llama3.2:3b.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        load_job(job_path)
        points_path = Path(args.points).expanduser().resolve() if args.points else artifact_path(job_path, "points.json")
        motion_path = Path(args.motion).expanduser().resolve() if args.motion else artifact_path(job_path, "motion.json")
        pose_path = Path(args.pose).expanduser().resolve() if args.pose else artifact_path(job_path, "pose_stats.json")
        pickups_path = Path(args.pickups).expanduser().resolve() if args.pickups else artifact_path(job_path, "pickups.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "report.md")
        report, recommendations = build_report(
            read_json(points_path),
            read_json(motion_path),
            read_json(pose_path),
            read_json(pickups_path),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        if args.ollama_model:
            draft = local_model_draft(report, args.ollama_model)
            draft_path = output_path.with_name("coach_draft.md")
            draft_path.write_text(
                "# Local AI Coaching Draft\n\nReview every claim against the cited footage before using it.\n\n" + draft + "\n",
                encoding="utf-8",
            )
            print(f"Saved local model draft to {draft_path}.")
        print(f"Saved local report to {output_path} with {len(recommendations)} evidence-linked coaching notes.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
