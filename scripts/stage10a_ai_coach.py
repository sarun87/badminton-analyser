#!/usr/bin/env python3
"""Prepare a coaching prompt or ask a local Ollama model to write the narrative."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def coach_evidence(report: dict[str, Any], player_code: str | None = None) -> dict[str, Any]:
    names = report.get("player_names", {})
    report_player_code = report.get("analysis_subject_code", "R")
    player_code = player_code or report_player_code
    if player_code not in ("R", "O"):
        raise ValueError("analysis player code must be R or O")
    if player_code != report_player_code:
        raise ValueError(
            f"report_data.json is prepared for {report_player_code}; rerun Stage 10 with --player-code {player_code}"
        )
    opponent_code = "O" if player_code == "R" else "R"
    player_name = str(names.get(player_code) or f"Player {player_code}")
    opponent_name = str(names.get(opponent_code) or f"Player {opponent_code}")

    def name_for(code: Any) -> str | None:
        return player_name if code == "R" else opponent_name if code == "O" else None

    score = report.get("score", {})
    score_analysis = report.get("score_analysis", {})
    heatmap = report.get("heatmap_timelapse", {})
    identity = report.get("player_identity", {})
    player_role = identity.get(player_code, "player selected in Stage 3" if player_code == "R" else "other player")
    opponent_role = identity.get(opponent_code, "opponent")
    video_points = report.get("video_points", [])
    score_curve = []
    for row in score_analysis.get("curve", []):
        score_curve.append({
            "score_point": row.get("point_n"),
            "serve": row.get("video_serve_number"),
            "score": {player_name: row.get(player_code), opponent_name: row.get(opponent_code)},
            "winner": name_for(row.get("winner")),
            "score_only_point": bool(row.get("dummy_point")),
        })

    rally_length = {}
    for key, value in report.get("by_rally_length", {}).items():
        player_wins = value.get("player_wins", value.get(f"{player_code}_wins"))
        if player_wins is None and player_code == "O":
            player_wins = max(0, int(value.get("rally_count", 0)) - int(value.get("R_wins", 0)))
        player_losses = value.get("player_losses", value.get(f"{player_code}_losses"))
        if player_losses is None and player_code == "O":
            player_losses = max(0, int(value.get("rally_count", 0)) - int(value.get("O_wins", player_wins or 0)))
        player_rate = value.get("player_win_rate_pct", value.get(f"{player_code}_win_rate_pct"))
        if player_rate is None and player_code == "O":
            player_rate = value.get("O_win_rate_pct")
        rally_length[key] = {
            "duration_band": value.get("label"),
            "rallies": value.get("rally_count"),
            "player_wins": player_wins,
            "player_losses": player_losses,
            "player_win_rate_pct": player_rate,
            "mean_rally_duration_sec": value.get("mean_rally_duration_sec"),
            "player_movement": value.get("movement", {}).get(player_code, {}),
            "opponent_movement": value.get("movement", {}).get(opponent_code, {}),
        }

    phases = []
    for phase in heatmap.get("phases", []):
        phase_players = {}
        for code, role, name in ((player_code, "player", player_name), (opponent_code, "opponent", opponent_name)):
            summary = phase.get("players", {}).get(code, {})
            phase_players[role] = {
                "name": name,
                "samples": summary.get("samples", 0),
                "zone_percentages": summary.get("zone_percentages", {}),
            }
        phases.append({
            "name": phase.get("name"),
            "score_point_range": phase.get("point_range"),
            "players": phase_players,
        })

    reviews = []
    for row in report.get("example_lost_points", []):
        reviews.append({
            "serve": row.get("video_serve_number"),
            "score_point": row.get("point_n"),
            "review_type": row.get("reason_label", row.get("reason")),
            "score_before": row.get("score_before"),
            "score_after": row.get("score_after"),
            "serve_time_sec": row.get("serve_t"),
            "estimated_rally_duration_sec": row.get("rally_duration_sec"),
            "rally_end_confidence": row.get("rally_end_confidence"),
            "rally_end_method": row.get("rally_end_method"),
            "movement_rally_end_sec": row.get("movement_rally_end_sec"),
            "tracknet_rally_end_refinement": row.get("tracknet_rally_end_refinement"),
            "tracknet_shuttle_evidence": row.get("tracknet_shuttle_evidence"),
            "tracknet_landing_candidate": row.get("tracknet_landing_candidate"),
            "estimated_shot_count": row.get("estimated_shot_count"),
            "manual_shot_count": row.get("manual_shot_count"),
            "shot_count": row.get("shot_count"),
            "shot_count_source": row.get("shot_count_source"),
            "shot_count_quality": row.get("shot_count_quality"),
            "player_beginning_speed_m_s": row.get("movement_phase_speed", {}).get(player_code, {}).get("beginning_average_speed_m_s"),
            "player_tail_speed_m_s": row.get("movement_phase_speed", {}).get(player_code, {}).get("tail_average_speed_m_s"),
            "player_tail_speed_change_pct": row.get("movement_phase_speed", {}).get(player_code, {}).get("tail_change_pct"),
            "player_distance_m": row.get("player_distance_m", row.get(f"{player_code}_distance_m")),
            "stage10_observation": row.get("coach_note"),
            "contact_sheets": row.get("contact_sheets", []),
        })

    by_outcome = {}
    for outcome in ("won", "lost"):
        group = report.get("by_outcome", {}).get(outcome, {})
        by_outcome[outcome] = {
            "rallies": group.get("rally_count"),
            "mean_duration_sec": group.get("mean_rally_duration_sec"),
            "player_movement": group.get("players", {}).get(player_code, {}),
            "opponent_movement": group.get("players", {}).get(opponent_code, {}),
        }

    return {
        "analysis_subject_code": player_code,
        "participants": {
            "player": {"name": player_name, "role": player_role},
            "opponent": {"name": opponent_name, "role": opponent_role},
            "score_order": [player_name, opponent_name],
        },
        "match_result": {
            "final_score": {
                player_name: (score.get("final") or {}).get(player_code),
                opponent_name: (score.get("final") or {}).get(opponent_code),
            },
            "video_rallies": score.get("video_rally_count"),
            "player_video_wins": score.get("player_video_rally_wins", score.get(f"{player_code}_video_rally_wins")),
            "player_video_losses": score.get("player_video_rally_losses", score.get(f"{player_code}_video_rally_losses")),
            "score_only_points": score.get("score_only_dummy_point_numbers", []),
        },
        "score_progression": {
            "curve": score_curve,
            "stage10_observations": score_analysis.get("coaching_commentary", []),
        },
        "rally_length_comparison": rally_length,
        "movement_by_outcome": by_outcome,
        "rally_effort_analysis": report.get("rally_effort_analysis", {}),
        "court_heatmap_phases": phases,
        "heatmap_observations": heatmap.get("coaching_commentary", []),
        "heatmap_timing_note": heatmap.get("timing_note"),
        "ready_position_pose": report.get("pose_ready_by_outcome", {}),
        "pickup_evidence": {
            "landing_side_counts": {
                player_name: report.get("pickup_evidence", {}).get("landing_side_counts", {}).get(player_code, 0),
                opponent_name: report.get("pickup_evidence", {}).get("landing_side_counts", {}).get(opponent_code, 0),
                "unknown": report.get("pickup_evidence", {}).get("landing_side_counts", {}).get("unknown", 0),
            },
            "interpretation_note": report.get("pickup_evidence", {}).get("interpretation_note"),
        },
        "tracknet_shuttle_analysis": report.get("tracknet_shuttle_analysis", {"enabled": False}),
        "three_video_reviews": reviews,
        "data_quality_notes": report.get("quality_notes", {}),
    }


def build_context(report: dict[str, Any], player_code: str | None = None) -> str:
    evidence = coach_evidence(report, player_code)
    player_code = evidence["analysis_subject_code"]
    serves = [row.get("serve") for row in evidence["three_video_reviews"] if row.get("serve") is not None]
    headings = [
        "## Match opening",
        "## Score progression",
        "## Court movement",
        *(f"## Serve {serve}" for serve in serves),
        "## Practice focus",
    ]
    output_rules = "\n".join(f"- {heading}" for heading in headings)
    return "\n".join([
        "# Badminton coach narrative request",
        "",
        f"Act as an experienced badminton singles coach reviewing this match. The player is {evidence['participants']['player']['name']} ({evidence['participants']['player']['role']}); the opponent is {evidence['participants']['opponent']['name']} ({evidence['participants']['opponent']['role']}). Write a specific, natural coaching readout that sounds like a coach speaking to {evidence['participants']['player']['name']}, not a statistics report.",
        "",
        f"Use only the supplied evidence. Refer to the two players by name ({evidence['participants']['player']['name']} and {evidence['participants']['opponent']['name']}); do not use tracking codes such as R or O in the narrative. Treat movement, pose, pickup, rally-end, TrackNet shot-count, and shuttle estimates as clues rather than proof of cause. A TrackNet stationary-point candidate is not a confirmed landing or winner. Shot counts are estimates; image-space net-plane candidates can include projection overlap or adjacent-court tracks and should be checked against video. A lower tail speed is not proof of fatigue. Do not infer true shuttle speed or exact shot type from image coordinates. Do not invent shots or events that are not in the data. Use timestamps and serve numbers to point the player to video they can check. The score-only point is not a video serve or rally. Avoid explaining basic badminton scoring rules.",
        "",
        "Keep the opening and each analysis section to one or two concise spoken sentences. For each reviewed serve, explain what to look for in that clip and give one practical adjustment. Finish with two or three prioritized practice focuses. Do not repeat the JSON or add generic praise.",
        "",
        "## Input evidence (do not reproduce this section)",
        "",
        "```json",
        json.dumps(evidence, ensure_ascii=False, indent=2),
        "```",
        "",
        f"Return only the coaching narrative as Markdown. Start with this exact metadata line: `<!-- analysis_subject_code: {player_code} -->`. Use these exact headings, including one for each reviewed serve:",
        output_rules,
        "",
    ])


def ask_ollama(endpoint: str, model: str, prompt: str, timeout: float) -> str:
    parsed_endpoint = urllib.parse.urlsplit(endpoint)
    if parsed_endpoint.scheme not in ("http", "https") or parsed_endpoint.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Ollama endpoint must use localhost so match analysis stays on this computer")
    request_data = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a careful, evidence-led badminton singles coach."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.35},
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with local_opener.open(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    message = result.get("message", {}).get("content")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Ollama returned an empty response; check the model and local Ollama service")
    return message.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create AI coach context or generate a narrative with local Ollama.")
    parser.add_argument("report_json", type=Path, help="Stage 10 report_data.json")
    parser.add_argument("--player-code", choices=("R", "O"), help="verify the Stage 10 analysis subject")
    parser.add_argument("--mode", choices=("context", "ollama"), default="context", help="export a copy/paste prompt or call local Ollama")
    parser.add_argument("--context-output", type=Path, help="self-contained prompt and evidence file (defaults beside report_json)")
    parser.add_argument("--narrative-output", type=Path, help="Markdown narrative consumed by Stage 11 (defaults beside report_json)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat", help="local Ollama chat endpoint")
    parser.add_argument("--model", default="qwen2.5:7b", help="installed local Ollama model name")
    parser.add_argument("--timeout", type=float, default=600.0, help="Ollama request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    try:
        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than zero")
        report_path = args.report_json.expanduser().resolve()
        report = read_json(report_path)
        subject_code = args.player_code or report.get("analysis_subject_code", "R")
        if subject_code not in ("R", "O"):
            raise ValueError("report analysis_subject_code must be R or O")
        context = build_context(report, subject_code)
        suffix = "" if subject_code == "R" else "-O"
        context_output = (
            args.context_output or report_path.parent / f"AI-coach-context{suffix}.md"
        ).expanduser()
        narrative_output = (
            args.narrative_output or report_path.parent / f"AI-coach-narrative{suffix}.md"
        ).expanduser()
        context_output.parent.mkdir(parents=True, exist_ok=True)
        narrative_output.parent.mkdir(parents=True, exist_ok=True)
        outputs = [context_output]
        if args.mode == "ollama":
            outputs.append(narrative_output)
        existing = [path for path in outputs if path.exists()]
        if existing and not args.overwrite:
            names = ", ".join(str(path) for path in existing)
            raise ValueError(f"output already exists: {names}; add --overwrite to replace it")

        context_output.write_text(context, encoding="utf-8")
        print(f"Coach context: {context_output}")
        if args.mode == "ollama":
            try:
                narrative = ask_ollama(args.ollama_url, args.model, context, args.timeout)
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not get a narrative from local Ollama ({exc}). The context file was saved; "
                    "start Ollama with the requested model or use --mode context."
                ) from exc
            narrative_output.write_text(narrative, encoding="utf-8")
            print(f"AI coach narrative: {narrative_output}")
        else:
            print("Copy the context file into a chat agent, then save its Markdown response as "
                  f"{narrative_output} for Stage 11.")
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
