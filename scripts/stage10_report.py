from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from pathlib import Path
from typing import Any


LENGTH_BUCKETS = (
    ("short_0_5s", "0–5 s", 0.0, 5.0),
    ("medium_5_10s", ">5–10 s", 5.0, 10.0),
    ("long_over_10s", ">10 s", 10.0, math.inf),
)
COURT_WIDTH_M = 6.1
HALF_LENGTH_M = 6.7
HEATMAP_BINS_X = 10
HEATMAP_BINS_Y = 11
DEPTH_ZONES = ("back", "middle", "front")
SIDE_ZONES = ("left", "center", "right")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def mean(values: list[float], digits: int = 2) -> float | None:
    return round(statistics.mean(values), digits) if values else None


def bucket_for(duration: float | None) -> str | None:
    if duration is None:
        return None
    if duration <= 5.0:
        return "short_0_5s"
    if duration <= 10.0:
        return "medium_5_10s"
    return "long_over_10s"


def safe_number(value: Any) -> float | None:
    return float(value) if finite(value) else None


def point2(value: Any) -> tuple[float, float] | None:
    """Return a finite two-dimensional court position from players.json."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(finite(coordinate) for coordinate in value):
        return None
    return float(value[0]), float(value[1])


def summarize_movement(points: list[dict[str, Any]], player: str) -> dict[str, Any]:
    metrics = [point.get("players", {}).get(player, {}) for point in points]
    distances = [float(m["distance_m"]) for m in metrics if finite(m.get("distance_m"))]
    observed = [float(m["observed_time_sec"]) for m in metrics if finite(m.get("observed_time_sec"))]
    speeds = [float(m["average_speed_m_s"]) for m in metrics if finite(m.get("average_speed_m_s"))]
    peaks = [float(m["peak_speed_m_s"]) for m in metrics if finite(m.get("peak_speed_m_s"))]
    trips = [trip for metric in metrics for trip in (metric.get("out_of_position_trips") or [])]
    recovery = [float(t["distance_from_base_1s_later_m"]) for t in trips if finite(t.get("distance_from_base_1s_later_m"))]
    total_distance = sum(distances)
    total_observed = sum(observed)
    return {
        "rallies_with_distance": len(distances),
        "total_distance_m": round(total_distance, 2),
        "mean_distance_per_rally_m": mean(distances),
        "observed_time_sec": round(total_observed, 2),
        "weighted_average_speed_m_s": round(total_distance / total_observed, 2) if total_observed > 0 else None,
        "mean_peak_speed_m_s": mean(peaks),
        "out_of_position_trip_count": len(trips),
        "mean_trips_per_rally": round(len(trips) / len(points), 2) if points else None,
        "mean_distance_from_base_1s_after_trip_m": mean(recovery),
    }


def aggregate(points: list[dict[str, Any]], motion_by_n: dict[int, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for outcome in ("won", "lost"):
        group = [point for point in points if point["outcome"] == outcome]
        durations = [point["rally_duration_sec"] for point in group if point["rally_duration_sec"] is not None]
        output[outcome] = {
            "rally_count": len(group),
            "mean_rally_duration_sec": mean(durations),
            "players": {
                player: summarize_movement(
                    [motion_by_n[point["point_n"]] for point in group if point["point_n"] in motion_by_n], player
                )
                for player in ("R", "O")
            },
        }
    return output


def summarize_rally_phase_speed(points: list[dict[str, Any]], player: str) -> dict[str, Any]:
    selected = [
        point.get("movement_phase_speed", {}).get(player, {})
        for point in points
        if point.get("rally_duration_sec") is not None and point["rally_duration_sec"] > 5.0
    ]
    available = [row for row in selected if row.get("comparison_status") == "available"]
    beginning = [float(row["beginning_average_speed_m_s"]) for row in available if finite(row.get("beginning_average_speed_m_s"))]
    tail = [float(row["tail_average_speed_m_s"]) for row in available if finite(row.get("tail_average_speed_m_s"))]
    changes = [float(row["tail_minus_beginning_m_s"]) for row in available if finite(row.get("tail_minus_beginning_m_s"))]
    pct_changes = [float(row["tail_change_pct"]) for row in available if finite(row.get("tail_change_pct"))]
    return {
        "long_rallies_over_5_sec": len(selected),
        "rallies_with_comparable_tracking": len(available),
        "mean_beginning_speed_m_s": mean(beginning),
        "mean_tail_speed_m_s": mean(tail),
        "mean_tail_minus_beginning_m_s": mean(changes),
        "mean_tail_change_pct": mean(pct_changes),
    }


def build_rally_effort_analysis(points: list[dict[str, Any]]) -> dict[str, Any]:
    timed = [point for point in points if finite(point.get("rally_duration_sec"))]
    longest = max(timed, key=lambda point: point["rally_duration_sec"], default=None)
    shot_points = [point for point in timed if isinstance(point.get("shot_count"), int) and not isinstance(point.get("shot_count"), bool)]
    longest_by_shots = max(shot_points, key=lambda point: point["shot_count"], default=None)
    return {
        "longest_rally": ({
            "point_n": longest["point_n"],
            "video_serve_number": longest.get("video_serve_number"),
            "duration_sec": longest.get("rally_duration_sec"),
            "shot_count": longest.get("shot_count"),
            "shot_count_source": longest.get("shot_count_source"),
        } if longest else None),
        "shot_count": {
            "enabled": bool(shot_points),
            "rallies_with_count": len(shot_points),
            "mean_count": mean([float(point["shot_count"]) for point in shot_points]),
            "highest_count": max((point["shot_count"] for point in shot_points), default=None),
            "rally_with_highest_count": ({
                "point_n": longest_by_shots["point_n"],
                "video_serve_number": longest_by_shots.get("video_serve_number"),
                "duration_sec": longest_by_shots.get("rally_duration_sec"),
                "shot_count": longest_by_shots.get("shot_count"),
                "shot_count_source": longest_by_shots.get("shot_count_source"),
            } if longest_by_shots else None),
        },
        "within_rally_speed_over_5_sec": {
            player: summarize_rally_phase_speed(points, player) for player in ("R", "O")
        },
        "interpretation_note": (
            "TrackNet shot counts are estimates from observed shuttle transitions or image-space net-plane candidates. "
            "They can miss detections or include projection overlap and adjacent-court tracks; check timestamps against video. "
            "A lower tail speed is a descriptive within-rally change, not proof of fatigue; tactics, pauses, and tracking gaps also affect it."
        ),
    }


def score_progression(
    score_rows: list[dict[str, Any]], video_points: list[dict[str, Any]], names: dict[str, str]
) -> dict[str, Any]:
    video_by_n = {point["point_n"]: point for point in video_points}
    curve = []
    longest_runs = {player: {"count": 0, "start_point_n": None, "end_point_n": None} for player in ("R", "O")}
    active_winner: str | None = None
    active_start: int | None = None
    active_count = 0
    previous_leader: str | None = None
    lead_changes = []
    widest = {"R": {"margin": 0, "point_n": None}, "O": {"margin": 0, "point_n": None}}

    def finish_run() -> None:
        if active_winner in ("R", "O") and active_count > longest_runs[active_winner]["count"]:
            longest_runs[active_winner] = {
                "count": active_count,
                "start_point_n": active_start,
                "end_point_n": curve[-1]["point_n"],
            }

    for row in score_rows:
        n = int(row["n"])
        after = row.get("score_after", {})
        r_score, o_score = int(after.get("R", 0)), int(after.get("O", 0))
        margin = r_score - o_score
        winner = row.get("winner") if row.get("winner") in ("R", "O") else None
        if winner is None:
            finish_run()
            active_winner, active_start, active_count = None, None, 0
        elif winner == active_winner:
            active_count += 1
        else:
            finish_run()
            active_winner, active_start, active_count = winner, n, 1
        if margin > widest["R"]["margin"]:
            widest["R"] = {"margin": margin, "point_n": n}
        if -margin > widest["O"]["margin"]:
            widest["O"] = {"margin": -margin, "point_n": n}
        leader = "R" if margin > 0 else "O" if margin < 0 else None
        if leader is not None and previous_leader is not None and leader != previous_leader:
            lead_changes.append({"point_n": n, "from": previous_leader, "to": leader, "margin": margin})
        if leader is not None:
            previous_leader = leader
        video_point = video_by_n.get(n, {})
        curve.append({
            "point_n": n, "R": r_score, "O": o_score, "margin_R_minus_O": margin,
            "winner": winner, "dummy_point": bool(row.get("dummy_point", False)),
            "video_serve_number": video_point.get("video_serve_number"),
            "time_sec": safe_number(video_point.get("serve_t")),
        })
    finish_run()

    peak_r = max(curve, key=lambda row: row["margin_R_minus_O"], default=None)
    peak_o = min(curve, key=lambda row: row["margin_R_minus_O"], default=None)
    r_run, o_run = longest_runs["R"], longest_runs["O"]
    curve_by_n = {row["point_n"]: row for row in curve}

    def score_reference(point_n: int | None) -> str:
        row = curve_by_n.get(point_n, {})
        serve_number = row.get("video_serve_number")
        if isinstance(serve_number, int) and not isinstance(serve_number, bool):
            return f"serve {serve_number}"
        if row.get("dummy_point"):
            return f"score-only point {point_n}"
        return f"point {point_n}"

    r_run_text = (
        f"{r_run['count']} consecutive points from {score_reference(r_run['start_point_n'])} "
        f"through {score_reference(r_run['end_point_n'])}"
    ) if r_run["count"] else "no recorded scoring run"
    o_run_text = (
        f"{o_run['count']} consecutive points from {score_reference(o_run['start_point_n'])} "
        f"through {score_reference(o_run['end_point_n'])}"
    ) if o_run["count"] else "no recorded scoring run"
    commentary = [
        f"The longest scoring runs were {names['R']}: {r_run_text}, and {names['O']}: {o_run_text}.",
    ]
    if peak_r is not None and peak_o is not None:
        swing_note = (
            f"The margin swung from {names['R']} +{peak_r['margin_R_minus_O']} after {score_reference(peak_r['point_n'])} "
            f"to {names['O']} +{-peak_o['margin_R_minus_O']} after {score_reference(peak_o['point_n'])}."
        )
        if lead_changes:
            swing_note += (
                f" The lead changed {len(lead_changes)} time(s); check the rallies around those shifts for turning points."
            )
        commentary.append(swing_note)
    dummy_numbers = [f"point {row['point_n']}" for row in curve if row["dummy_point"]]
    if dummy_numbers:
        commentary.append(
            f"The graph marks score-only {', '.join(dummy_numbers)} as dashed point(s); they have no video rally or timestamp."
        )
    return {
        "curve": curve,
        "longest_scoring_runs": longest_runs,
        "lead_changes": lead_changes,
        "widest_lead": widest,
        "coaching_commentary": commentary,
    }


def empty_heatmap() -> dict[str, Any]:
    return {
        "samples": 0,
        "grid_counts": [[0 for _ in range(HEATMAP_BINS_X)] for _ in range(HEATMAP_BINS_Y)],
        "zone_counts": {f"{depth}_{side}": 0 for depth in DEPTH_ZONES for side in SIDE_ZONES},
    }


def add_heatmap_sample(summary: dict[str, Any], point: tuple[float, float]) -> None:
    x, y = point
    if not (0.0 <= x <= COURT_WIDTH_M and 0.0 <= y <= HALF_LENGTH_M):
        return
    x_bin = min(HEATMAP_BINS_X - 1, int(x / COURT_WIDTH_M * HEATMAP_BINS_X))
    y_bin = min(HEATMAP_BINS_Y - 1, int(y / HALF_LENGTH_M * HEATMAP_BINS_Y))
    summary["samples"] += 1
    summary["grid_counts"][y_bin][x_bin] += 1
    side = "left" if x < COURT_WIDTH_M / 3 else "right" if x >= 2 * COURT_WIDTH_M / 3 else "center"
    depth = "back" if y <= 1.6 else "front" if y >= 4.72 else "middle"
    summary["zone_counts"][f"{depth}_{side}"] += 1


def finish_heatmap(summary: dict[str, Any]) -> dict[str, Any]:
    samples = summary["samples"]
    return {
        **summary,
        "zone_percentages": {
            zone: round(count * 100 / samples, 1) if samples else None
            for zone, count in summary["zone_counts"].items()
        },
    }


def build_heatmap_timelapse(
    players: dict[str, Any], rally_rows: list[dict[str, Any]], video_points: list[dict[str, Any]],
    example_lost_points: list[dict[str, Any]], names: dict[str, str], player_code: str = "R",
) -> dict[str, Any]:
    if player_code not in ("R", "O"):
        raise ValueError("analysis player code must be R or O")
    frames = players.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("players JSON must contain a non-empty frames array for the heatmap analysis")
    times = [float(frame["t"]) for frame in frames]
    valid_rallies = sorted(
        (row for row in rally_rows if finite(row.get("serve_t")) and finite(row.get("rally_end_sec"))),
        key=lambda row: float(row["serve_t"]),
    )
    score_by_n = {point["point_n"]: point for point in video_points}
    phase_names = ("Opening third", "Middle third", "Closing third")
    phases = []
    outcome_summaries = {outcome: {player: empty_heatmap() for player in ("R", "O")} for outcome in ("won", "lost")}
    for phase_i, name in enumerate(phase_names):
        start_i = phase_i * len(valid_rallies) // 3
        end_i = (phase_i + 1) * len(valid_rallies) // 3
        phase_rallies = valid_rallies[start_i:end_i]
        phase_summaries = {player: empty_heatmap() for player in ("R", "O")}
        for rally in phase_rallies:
            point_n = int(rally["point_n"])
            outcome = score_by_n.get(point_n, {}).get("outcome")
            first = bisect.bisect_left(times, float(rally["serve_t"]) + 0.3)
            last = bisect.bisect_right(times, float(rally["rally_end_sec"]))
            for frame in frames[first:last]:
                for player in ("R", "O"):
                    point = point2(frame.get("r" if player == "R" else "o"))
                    if point is None:
                        continue
                    if point[1] >= 6.7:
                        point = (COURT_WIDTH_M - point[0], 13.4 - point[1])
                    add_heatmap_sample(phase_summaries[player], point)
                    if outcome in outcome_summaries:
                        add_heatmap_sample(outcome_summaries[outcome][player], point)
        first_n = phase_rallies[0].get("point_n") if phase_rallies else None
        last_n = phase_rallies[-1].get("point_n") if phase_rallies else None
        phases.append({
            "name": name,
            "point_numbers": [int(row["point_n"]) for row in phase_rallies],
            "point_range": [first_n, last_n],
            "players": {player: finish_heatmap(value) for player, value in phase_summaries.items()},
        })
    by_outcome = {
        outcome: {player: finish_heatmap(value) for player, value in summaries.items()}
        for outcome, summaries in outcome_summaries.items()
    }

    def strongest_shift(before: dict[str, Any], after: dict[str, Any], player: str) -> dict[str, Any] | None:
        early = before["players"][player]["zone_percentages"]
        late = after["players"][player]["zone_percentages"]
        changes = [
            (abs((late[zone] or 0) - (early[zone] or 0)), (late[zone] or 0) - (early[zone] or 0), zone)
            for zone in early
            if early[zone] is not None and late[zone] is not None
        ]
        if not changes:
            return None
        magnitude, delta, zone = max(changes)
        return {"zone": zone, "delta_percentage_points": round(delta, 1), "absolute_change_percentage_points": round(magnitude, 1)}

    won_player, lost_player = by_outcome["won"][player_code], by_outcome["lost"][player_code]
    outcome_changes = []
    if won_player["samples"] and lost_player["samples"]:
        for zone, loss_pct in lost_player["zone_percentages"].items():
            win_pct = won_player["zone_percentages"][zone]
            outcome_changes.append((abs(loss_pct - win_pct), loss_pct - win_pct, zone, win_pct, loss_pct))
    outcome_change = max(outcome_changes, default=None)
    phase_change = strongest_shift(phases[0], phases[-1], player_code) if len(phases) >= 2 else None
    reference = next((row for row in example_lost_points if row.get("player_distance_m") is not None), None)
    reference_text = (
        f" Review serve {reference['video_serve_number']} at {format_timestamp(reference.get('serve_t'))}."
        if reference else ""
    )
    commentary = []
    if outcome_change:
        _, delta, zone, win_pct, loss_pct = outcome_change
        commentary.append(
            f"{names[player_code]}'s tracked positions in {zone.replace('_', ' ')} accounted for {loss_pct:.1f}% of lost-point samples vs {win_pct:.1f}% of won-point samples (largest zone difference)."
            f" Treat this as a location clue, then review the video before changing recovery choices.{reference_text}"
        )
    if phase_change:
        direction = "more" if phase_change["delta_percentage_points"] > 0 else "less"
        commentary.append(
            f"From the opening to closing third, {names[player_code]} spent {abs(phase_change['delta_percentage_points']):.1f} percentage points {direction} of tracked time in {phase_change['zone'].replace('_', ' ')}."
            " Use the heatmap sequence to check whether late-rally positions changed as the match progressed."
        )
    if not commentary:
        commentary.append("There are too few valid player positions by outcome to compare a reliable spatial pattern.")
    return {
        "local_court_coordinates": "Each player's own half; u across from own left sideline and v from own baseline.",
        "grid_bins": {"x": HEATMAP_BINS_X, "y": HEATMAP_BINS_Y},
        "phases": phases,
        "by_outcome": by_outcome,
        "largest_win_loss_zone_difference": None if not outcome_change else {
            "zone": outcome_change[2], "won_pct": outcome_change[3], "lost_pct": outcome_change[4],
            "delta_percentage_points": round(outcome_change[1], 1),
        },
        "largest_opening_closing_shift": phase_change,
        "coaching_commentary": commentary,
        "timing_note": "Only samples inside reviewed rally windows are included; player tracking and rally boundaries are approximate.",
    }


def format_number(value: Any, suffix: str = "") -> str:
    return "not available" if value is None else f"{value}{suffix}"


def format_timestamp(value: Any) -> str:
    return "time unavailable" if not finite(value) else f"{float(value):.1f}s"


def build_report(
    score: dict[str, Any],
    rallies: dict[str, Any],
    motion: dict[str, Any],
    players: dict[str, Any],
    pose: dict[str, Any],
    pickups: dict[str, Any],
    player_name: str = "You",
    opponent_name: str = "Opponent",
    player_code: str = "R",
) -> dict[str, Any]:
    if player_code not in ("R", "O"):
        raise ValueError("analysis player code must be R or O")
    opponent_code = "O" if player_code == "R" else "R"
    names = {
        "R": player_name.strip() or "You",
        "O": opponent_name.strip() or "Opponent",
    }
    score_rows = score.get("points")
    rally_rows = rallies.get("points")
    motion_rows = motion.get("points")
    pose_rows = pose.get("points")
    pickup_rows = pickups.get("points")
    if not all(isinstance(rows, list) for rows in (score_rows, rally_rows, motion_rows, pose_rows, pickup_rows)):
        raise ValueError("each input JSON must contain its expected points array")
    pose_player_code = pose.get("player_code", "R")
    if pose_player_code != player_code:
        raise ValueError(
            f"pose_stats.json was generated for {pose_player_code}; run Stage 8 with --player-code {player_code} first"
        )

    score_by_n: dict[int, dict[str, Any]] = {}
    for row in score_rows:
        n = row.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n in score_by_n:
            raise ValueError("Stage 5 points must have unique integer n values")
        score_by_n[n] = row

    motion_by_n = {row["point_n"]: row for row in motion_rows if isinstance(row.get("point_n"), int)}
    pose_by_n = {row["point_n"]: row for row in pose_rows if isinstance(row.get("point_n"), int)}
    pickup_by_n = {row["point_n"]: row for row in pickup_rows if isinstance(row.get("point_n"), int)}

    video_points: list[dict[str, Any]] = []
    for rally in rally_rows:
        n = rally.get("point_n")
        if not isinstance(n, int) or isinstance(n, bool):
            raise ValueError("Stage 6 rally points must have integer point_n values")
        score_row = score_by_n.get(n)
        if score_row is None or score_row.get("dummy_point") is True:
            raise ValueError(f"Stage 6 s{n} has no matching video point in Stage 5")
        winner = rally.get("winner") or score_row.get("winner")
        if winner not in ("R", "O") or score_row.get("winner") != winner:
            raise ValueError(f"winner mismatch or invalid winner for s{n}")
        start = safe_number(rally.get("serve_t"))
        end = safe_number(rally.get("rally_end_sec"))
        duration = safe_number(rally.get("rally_duration_sec"))
        if duration is None and start is not None and end is not None:
            duration = max(0.0, end - start)
        motion_row = motion_by_n.get(n, {})
        if duration is None:
            duration = safe_number(motion_row.get("rally_duration_sec"))
        motion_players = motion_row.get("players", {})
        point_pose = pose_by_n.get(n, {})
        shot_analysis = motion_row.get("shuttle_shot_analysis", {})
        estimated_shots = shot_analysis.get("estimated_shot_count") if isinstance(shot_analysis, dict) else None
        manual_shots = shot_analysis.get("manual_shot_count") if isinstance(shot_analysis, dict) else None
        shot_count = (
            manual_shots if isinstance(manual_shots, int) and not isinstance(manual_shots, bool) and manual_shots >= 0
            else estimated_shots if isinstance(estimated_shots, int) and not isinstance(estimated_shots, bool) and estimated_shots >= 0
            else None
        )
        movement_phase_speed = {
            player: motion_players.get(player, {}).get("rally_phase_speed", {})
            for player in ("R", "O")
        }
        sheets = rally.get("contact_sheets") if isinstance(rally.get("contact_sheets"), list) else []
        video_points.append({
            "point_n": n,
            "winner": winner,
            "outcome": "won" if winner == player_code else "lost",
            "score_before": score_row.get("before"),
            "score_after": score_row.get("after"),
            "serve_t": start,
            "rally_end_sec": end,
            "rally_duration_sec": round(duration, 2) if duration is not None else None,
            "length_bucket": bucket_for(duration),
            "rally_end_estimated": bool(rally.get("rally_end_estimated", False)),
            "rally_end_confidence": rally.get("rally_end_confidence"),
            "rally_end_method": rally.get("rally_end_method"),
            "movement_rally_end_sec": safe_number(rally.get("movement_rally_end_sec")),
            "tracknet_rally_end_refinement": rally.get("tracknet_rally_end_refinement"),
            "serve_time_estimated": bool(rally.get("serve_time_estimated", False)),
            "contact_sheets": sheets,
            "R_distance_m": safe_number(motion_row.get("players", {}).get("R", {}).get("distance_m")),
            "O_distance_m": safe_number(motion_row.get("players", {}).get("O", {}).get("distance_m")),
            "player_distance_m": safe_number(motion_row.get("players", {}).get(player_code, {}).get("distance_m")),
            "opponent_distance_m": safe_number(motion_row.get("players", {}).get(opponent_code, {}).get("distance_m")),
            "movement_by_player": motion_players,
            "pose_by_context": point_pose.get("pose_by_context", {}),
            "tracknet_shuttle_evidence": rally.get("tracknet_shuttle_evidence"),
            "tracknet_landing_candidate": pickup_by_n.get(n, {}).get("tracknet_landing_candidate"),
            "estimated_shot_count": estimated_shots,
            "manual_shot_count": manual_shots,
            "shot_count": shot_count,
            "shot_count_source": (
                "manual_review" if isinstance(manual_shots, int) and not isinstance(manual_shots, bool) and manual_shots >= 0
                else "tracknet_estimate" if isinstance(estimated_shots, int) and not isinstance(estimated_shots, bool) and estimated_shots >= 0
                else None
            ),
            "shot_count_quality": shot_analysis.get("count_quality") if isinstance(shot_analysis, dict) else None,
            "shot_count_candidate_crossings": shot_analysis.get("candidate_crossings", []) if isinstance(shot_analysis, dict) else [],
            "movement_phase_speed": movement_phase_speed,
        })
    video_points.sort(key=lambda point: point["point_n"])
    for serve_number, point in enumerate(video_points, start=1):
        point["video_serve_number"] = serve_number

    duration_groups: dict[str, Any] = {}
    for key, label, low, high in LENGTH_BUCKETS:
        subset = [point for point in video_points if point["length_bucket"] == key]
        r_wins = sum(point["winner"] == "R" for point in subset)
        o_wins = sum(point["winner"] == "O" for point in subset)
        player_wins = r_wins if player_code == "R" else o_wins
        opponent_wins = o_wins if player_code == "R" else r_wins
        by_outcome = {}
        for outcome in ("won", "lost"):
            outcome_points = [point for point in subset if point["outcome"] == outcome]
            outcome_motion = [motion_by_n[p["point_n"]] for p in outcome_points if p["point_n"] in motion_by_n]
            by_outcome[outcome] = {
                "rally_count": len(outcome_points),
                "players": {player: summarize_movement(outcome_motion, player) for player in ("R", "O")},
            }
        duration_groups[key] = {
            "label": label,
            "rally_count": len(subset),
            "R_wins": r_wins,
            "R_losses": len(subset) - r_wins,
            "R_win_rate_pct": round(100 * r_wins / len(subset), 1) if subset else None,
            "O_wins": o_wins,
            "O_losses": len(subset) - o_wins,
            "O_win_rate_pct": round(100 * o_wins / len(subset), 1) if subset else None,
            "player_wins": player_wins,
            "player_losses": len(subset) - player_wins,
            "player_win_rate_pct": round(100 * player_wins / len(subset), 1) if subset else None,
            "opponent_wins": opponent_wins,
            "opponent_losses": len(subset) - opponent_wins,
            "opponent_win_rate_pct": round(100 * opponent_wins / len(subset), 1) if subset else None,
            "mean_rally_duration_sec": mean([p["rally_duration_sec"] for p in subset if p["rally_duration_sec"] is not None]),
            "movement": {player: summarize_movement([motion_by_n[p["point_n"]] for p in subset if p["point_n"] in motion_by_n], player) for player in ("R", "O")},
            "by_outcome": by_outcome,
        }

    outcomes = aggregate(video_points, motion_by_n)
    pose_summary = pose.get("summary_by_outcome_and_context", {})
    pose_frames = pose.get("frames") if isinstance(pose.get("frames"), list) else []
    pose_identity_filter = pose.get("pose_identity_filter")
    if not isinstance(pose_identity_filter, dict):
        pose_identity_filter = {}
    pickup_counts: dict[str, int] = {"R": 0, "O": 0, "unknown": 0}
    pickup_by_outcome = {"won": {"R": 0, "O": 0, "unknown": 0}, "lost": {"R": 0, "O": 0, "unknown": 0}}
    manual_count = 0
    for point in video_points:
        record = pickup_by_n.get(point["point_n"], {})
        side = record.get("landing_side") if record.get("landing_side") in ("R", "O") else "unknown"
        pickup_counts[side] += 1
        pickup_by_outcome[point["outcome"]][side] += 1
        if record.get("status") == "manual_confirmed":
            manual_count += 1

    dummy_rows = [row for row in score_rows if row.get("dummy_point") is True]
    estimates = sum(point["rally_end_estimated"] for point in video_points)
    estimated_serves = sum(point["serve_time_estimated"] for point in video_points)
    confidence_counts = {name: sum(point["rally_end_confidence"] == name for point in video_points) for name in ("high", "medium", "low")}
    confidence_counts["unspecified"] = sum(point["rally_end_confidence"] not in ("high", "medium", "low") for point in video_points)
    selected_losses = select_lost_examples(video_points, duration_groups, outcomes, names, player_code)
    progression = score_progression(score_rows, video_points, names)
    heatmap_timelapse = build_heatmap_timelapse(
        players, rally_rows, video_points, selected_losses, names, player_code
    )
    shuttle_rows = [
        point.get("tracknet_shuttle_evidence") for point in video_points
        if isinstance(point.get("tracknet_shuttle_evidence"), dict)
    ]
    landing_candidates = [
        point.get("tracknet_landing_candidate") for point in video_points
        if isinstance(point.get("tracknet_landing_candidate"), dict)
    ]
    tracknet_analysis = {
        "enabled": bool(shuttle_rows or landing_candidates or any(
            isinstance(row.get("shuttle_shot_analysis"), dict) for row in motion_rows
        )),
        "rally_end_source": (rallies.get("end_time_estimation") or {}).get("rally_end_source", "players"),
        "rallies_with_track_summaries": len(shuttle_rows),
        "rallies_with_tracknet_refined_ends": sum(
            point.get("rally_end_method") == "tracknet_cluster_near_movement_pause"
            for point in video_points
        ),
        "rallies_falling_back_to_movement_end": sum(
            isinstance(point.get("tracknet_rally_end_refinement"), dict)
            and point["tracknet_rally_end_refinement"].get("status")
            != "applied"
            for point in video_points
        ),
        "rallies_with_main_court_observations": sum(
            int(row.get("main_court_candidate_count", 0) or 0) > 0 for row in shuttle_rows
        ),
        "rallies_with_stationary_landing_candidates": sum(
            row.get("status") == "tentative_resting_shuttle_candidate" for row in landing_candidates
        ),
        "interpretation_note": (
            "TrackNet shuttle points are image-space observations. A rally end may be refined by a sustained "
            "calibrated main-court cluster near the movement pause; resting-point candidates require visual "
            "confirmation, and airborne pixels do not directly give court positions or winners."
        ),
    }
    rally_effort_analysis = build_rally_effort_analysis(video_points)
    tracknet_analysis["shot_counts_available"] = rally_effort_analysis["shot_count"]["enabled"]
    tracknet_analysis["rallies_with_shot_count"] = rally_effort_analysis["shot_count"]["rallies_with_count"]
    player_identity = players.get("player_identity", {})
    pickup_by_r_outcome = (
        pickup_by_outcome if player_code == "R"
        else {"won": pickup_by_outcome["lost"], "lost": pickup_by_outcome["won"]}
    )
    player_description = (
        f"You ({player_identity.get(player_code, 'player selected in Stage 3')})"
    )
    opponent_description = f"Opponent ({player_identity.get(opponent_code, 'other player')})"
    return {
        "title": "Badminton match analysis report",
        "review_evidence_version": 1,
        "analysis_subject_code": player_code,
        "analysis_subject": {"code": player_code, "name": names[player_code]},
        "players": {
            player_code: player_description,
            opponent_code: opponent_description,
        },
        "player_identity": player_identity,
        "player_names": names,
        "score": {
            "final": score.get("final_score"),
            "score_order": score.get("score_order", "R-O"),
            "score_log_point_count": score.get("point_count", len(score_rows)),
            "video_rally_count": len(video_points),
            "score_only_dummy_point_count": len(dummy_rows),
            "score_only_dummy_point_numbers": [row["n"] for row in dummy_rows],
            "R_video_rally_wins": sum(point["winner"] == "R" for point in video_points),
            "R_video_rally_losses": sum(point["winner"] != "R" for point in video_points),
            "O_video_rally_wins": sum(point["winner"] == "O" for point in video_points),
            "O_video_rally_losses": sum(point["winner"] != "O" for point in video_points),
            "player_video_rally_wins": sum(point["winner"] == player_code for point in video_points),
            "player_video_rally_losses": sum(point["winner"] != player_code for point in video_points),
        },
        "quality_notes": {
            "rally_end_estimated_count": estimates,
            "serve_time_estimated_count": estimated_serves,
            "rally_end_confidence_counts": confidence_counts,
            "rally_end_times_are_approximate": bool(rallies.get("end_time_estimation", {}).get("estimates_are_approximate", False)),
            "timeline_note": "Times use the analysis video timeline. Add the Stage 0 first-serve source offset to locate them in the original recording.",
            "dummy_points_excluded_from_video_metrics": True,
            "pose_frame_count": len(pose_frames),
            "pose_identity_filter": pose_identity_filter,
        },
        "by_outcome": outcomes,
        "by_rally_length": duration_groups,
        "score_analysis": progression,
        "heatmap_timelapse": heatmap_timelapse,
        "pose_player_code": pose_player_code,
        "pose_ready_by_outcome": {
            outcome: pose_summary.get(outcome, {}).get("READY", {}) for outcome in ("won", "lost")
        },
        "pickup_evidence": {
            "landing_side_counts": pickup_counts,
            "by_R_outcome": pickup_by_r_outcome,
            "by_player_outcome": pickup_by_outcome,
            "manual_confirmed_count": manual_count,
            "pickup_total": len(video_points),
            "interpretation_note": "Pickup side is indirect evidence of where the shuttle was collected; unknown results are not inferred.",
        },
        "tracknet_shuttle_analysis": tracknet_analysis,
        "rally_effort_analysis": rally_effort_analysis,
        "example_lost_points": selected_losses,
        "video_points": video_points,
    }


def select_lost_examples(
    points: list[dict[str, Any]], duration_groups: dict[str, Any], outcomes: dict[str, Any],
    names: dict[str, str], player_code: str = "R",
) -> list[dict[str, Any]]:
    if player_code not in ("R", "O"):
        raise ValueError("analysis player code must be R or O")
    lost = [point for point in points if point["outcome"] == "lost"]
    mid_reliable = sorted(
        (
            point for point in lost
            if point["rally_duration_sec"] is not None
            and 5.0 < point["rally_duration_sec"] <= 10.0
            and point["rally_end_confidence"] in ("high", "medium")
        ),
        key=lambda point: point["rally_duration_sec"],
    )
    representative_mid = mid_reliable[len(mid_reliable) // 2] if mid_reliable else None
    mid_candidates = [representative_mid] if representative_mid else []
    mid_candidates.extend(sorted(
        (p for p in lost if p["rally_duration_sec"] is not None and 5.0 < p["rally_duration_sec"] <= 10.0),
        key=lambda p: abs(p["rally_duration_sec"] - 7.5),
    ))
    long_candidates = sorted(
        (p for p in lost if p["rally_duration_sec"] is not None),
        key=lambda p: p["rally_duration_sec"], reverse=True,
    )
    movement_candidates = sorted(
        (p for p in lost if p.get("player_distance_m") is not None),
        key=lambda p: p["player_distance_m"], reverse=True,
    )
    mid_reason = f"representative mid-length {player_code} loss"
    long_reason = f"longest {player_code} loss"
    movement_reason = f"highest {player_code} movement among losses"
    candidates = (
        (mid_reason, mid_candidates),
        (long_reason, long_candidates),
        (movement_reason, movement_candidates),
    )
    reason_labels = {
        mid_reason: f"representative mid-length loss by {names[player_code]}",
        long_reason: f"longest loss by {names[player_code]}",
        movement_reason: f"highest tracked movement by {names[player_code]} in a loss",
    }
    seen: set[int] = set()
    result = []
    mid_band = duration_groups["medium_5_10s"]
    long_losses = duration_groups["long_over_10s"]["by_outcome"]["lost"]
    loss_movement = outcomes["lost"]["players"][player_code]
    for reason, choices in candidates:
        point = next((choice for choice in choices if choice and choice["point_n"] not in seen), None)
        if point is None:
            continue
        seen.add(point["point_n"])
        if reason == mid_reason:
            rate = mid_band.get("player_win_rate_pct")
            rate_text = f"{names[player_code]} won {mid_band['player_wins']} of {mid_band['rally_count']} rallies in the 5–10 s band ({rate}%). " if rate is not None else "The mid-length-rally sample is limited. "
            duration = point.get("rally_duration_sec")
            duration_text = f"about {duration:.1f} seconds" if finite(duration) else "an estimated mid-length duration"
            coach_note = (
                f"{rate_text}This loss lasted {duration_text}. Review the positioning changes and recovery through the exchange; shot type is not identified reliably."
            )
        elif reason == long_reason:
            trips = long_losses["players"][player_code].get("mean_trips_per_rally")
            trips_text = f"Long-rally losses averaged {trips} tracked trips at least 2.2 m from base. " if trips is not None else "The long-rally sample is limited. "
            duration = point.get("rally_duration_sec")
            duration_text = f"about {duration:.1f} seconds" if finite(duration) else "an estimated long duration"
            coach_note = (
                f"This was the longest loss at {duration_text}. {trips_text}Review whether the final displacement was followed by a quick recovery toward base."
            )
        else:
            distance = point.get("player_distance_m")
            mean_loss = loss_movement.get("mean_distance_per_rally_m")
            distance_text = f"{distance:.1f} m" if finite(distance) else "unavailable"
            mean_text = f"{mean_loss:.1f} m" if finite(mean_loss) else "unavailable"
            coach_note = (
                f"{names[player_code]} covered {distance_text} in this loss; lost rallies averaged {mean_text}. Review the farthest movement and whether you recovered before the next shot."
            )
        result.append({"reason": reason, "reason_label": reason_labels[reason], "coach_note": coach_note, **point})
    return result


def markdown_report(data: dict[str, Any]) -> str:
    score = data["score"]
    final = score["final"] or {}
    names = data.get("player_names", {"R": "You", "O": "Opponent"})
    player_code = data.get("analysis_subject_code", "R")
    opponent_code = "O" if player_code == "R" else "R"
    player_name, opponent_name = names[player_code], names[opponent_code]
    player_wins = score.get("player_video_rally_wins", score.get(f"{player_code}_video_rally_wins", 0))
    player_losses = score.get("player_video_rally_losses", score.get(f"{player_code}_video_rally_losses", 0))
    lines = [
        "# Badminton Match Report",
        "",
        "## Result and coverage",
        "",
        f"- Final score ({player_name}–{opponent_name}): **{final.get(player_code, '?')}–{final.get(opponent_code, '?')}**.",
        f"- Video analysis: **{score['video_rally_count']} rallies** ({player_wins} won, {player_losses} lost for {player_name}).",
        f"- Score log: {score['score_log_point_count']} points, including {score['score_only_dummy_point_count']} score-only "
        f"{'point' if score['score_only_dummy_point_count'] == 1 else 'points'} excluded from video metrics.",
        f"- Estimated rally-end timestamps: {data['quality_notes']['rally_end_estimated_count']} of {score['video_rally_count']} "
        f"(confidence high/medium/low: {data['quality_notes']['rally_end_confidence_counts']['high']}/"
        f"{data['quality_notes']['rally_end_confidence_counts']['medium']}/"
        f"{data['quality_notes']['rally_end_confidence_counts']['low']}); treat durations as approximate.",
        f"- {data['quality_notes']['timeline_note']}",
        *(
            [
                f"- Pose tracking kept {data['quality_notes'].get('pose_frame_count', 0)} O samples; "
                f"skipped {data['quality_notes']['pose_identity_filter'].get('rejected_frame_count', 0)} pose candidates "
                "that did not align with O's tracked box."
            ]
            if data['quality_notes'].get('pose_identity_filter', {}).get('enabled') else []
        ),
        "",
        "## Rally length and results",
        "",
        f"| Rally duration | {player_name} wins | {player_name} losses | {player_name} win rate | Mean duration |",
        "|---|---:|---:|---:|---:|",
    ]
    shuttle_analysis = data.get("tracknet_shuttle_analysis", {})
    if shuttle_analysis.get("enabled"):
        insert_at = lines.index("## Rally length and results") - 1
        tracknet_lines = []
        if shuttle_analysis.get("rallies_with_track_summaries", 0):
            tracknet_lines.append(
                f"- Raw TrackNet summaries are available for {shuttle_analysis['rallies_with_track_summaries']} rallies; "
                f"{shuttle_analysis.get('rallies_with_main_court_observations', 0)} include main-court candidate points."
            )
        if shuttle_analysis.get("rallies_with_track_summaries", 0) and shuttle_analysis.get("rally_end_source") == "shuttle":
            tracknet_lines.append(
                f"- TrackNet refined {shuttle_analysis.get('rallies_with_tracknet_refined_ends', 0)} rally ends; "
                f"{shuttle_analysis.get('rallies_falling_back_to_movement_end', 0)} kept the movement estimate because no suitable cluster was found."
            )
        elif shuttle_analysis.get("rallies_with_track_summaries", 0):
            tracknet_lines.append("- Rally ends remained movement-based; TrackNet rally-end refinement was not enabled.")
        if shuttle_analysis.get("rallies_with_stationary_landing_candidates", 0):
            tracknet_lines.append(
                f"- Tentative stationary-shuttle landing candidates: {shuttle_analysis['rallies_with_stationary_landing_candidates']}. "
                "These require visual confirmation and do not determine the point winner."
            )
        if tracknet_lines:
            lines[insert_at:insert_at] = tracknet_lines + [""]
    for key, _, _, _ in LENGTH_BUCKETS:
        group = data["by_rally_length"][key]
        lines.append(
            f"| {group['label']} ({group['rally_count']}) | {group.get('player_wins', group['R_wins'])} | "
            f"{group.get('player_losses', group['R_losses'])} | "
            f"{format_number(group.get('player_win_rate_pct', group['R_win_rate_pct']), '%')} | "
            f"{format_number(group['mean_rally_duration_sec'], ' s')} |"
        )
    score_analysis = data["score_analysis"]
    lines += [
        "",
        "## Score progression and momentum",
        "",
    ]
    lines.extend(f"- {sentence}" for sentence in score_analysis["coaching_commentary"])
    lines += [
        "",
        "## Movement by point outcome",
        "",
        "Distance is the mean per rally; speed is total tracked distance divided by observed tracking time.",
        "",
        f"| {player_name} outcome | {player_name} distance/rally | {player_name} speed | {opponent_name} distance/rally | {opponent_name} speed | {player_name} trips out of position/rally |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome, label in (("won", "Won"), ("lost", "Lost")):
        group = data["by_outcome"][outcome]
        player, opponent = group["players"][player_code], group["players"][opponent_code]
        lines.append(
            f"| {label} ({group['rally_count']}) | {format_number(player['mean_distance_per_rally_m'], ' m')} | "
            f"{format_number(player['weighted_average_speed_m_s'], ' m/s')} | {format_number(opponent['mean_distance_per_rally_m'], ' m')} | "
            f"{format_number(opponent['weighted_average_speed_m_s'], ' m/s')} | {format_number(player['mean_trips_per_rally'])} |"
        )
    effort = data.get("rally_effort_analysis", {})
    longest = effort.get("longest_rally")
    shot_summary = effort.get("shot_count", {})
    lines += ["", "## Shuttle shot estimates and within-rally speed", ""]
    if longest:
        if finite(longest.get("shot_count")):
            longest_shots = (
                f", manually reviewed shot count {longest['shot_count']}"
                if longest.get("shot_count_source") == "manual_review"
                else f", about {longest['shot_count']} shuttle crossings"
            )
        else:
            longest_shots = ""
        lines.append(
            f"- Longest video rally: serve {longest.get('video_serve_number') or '?'} "
            f"(score point {longest['point_n']}), {format_number(longest.get('duration_sec'), ' s')}{longest_shots}."
        )
    if shot_summary.get("enabled"):
        lines.append(
            f"- TrackNet/visual review produced a count for {shot_summary['rallies_with_count']} rallies; "
            f"the mean reported shot count was {format_number(shot_summary.get('mean_count'))} per rally."
        )
        highest = shot_summary.get("rally_with_highest_count")
        if highest:
            lines.append(
                f"- Highest count: serve {highest.get('video_serve_number') or '?'} "
                f"(score point {highest['point_n']}), {highest['shot_count']} shots by estimate or visual correction."
            )
        lines.append(f"- {effort.get('interpretation_note', '')}")
        shot_rows = sorted(
            (point for point in data["video_points"] if finite(point.get("shot_count"))),
            key=lambda point: point.get("rally_duration_sec") or 0.0,
            reverse=True,
        )[:10]
        if shot_rows:
            lines += [
                "",
                f"| Serve | Duration | {player_name} estimated/reviewed shots | {player_name} beginning → tail speed | {opponent_name} beginning → tail speed |",
                "|---:|---:|---:|---:|---:|",
            ]
            for point in shot_rows:
                player_phase = point.get("movement_phase_speed", {}).get(player_code, {})
                opponent_phase = point.get("movement_phase_speed", {}).get(opponent_code, {})
                shot_text = (
                    f"{point['shot_count']} manually reviewed"
                    if point.get("manual_shot_count") is not None else
                    f"~{point['shot_count']} ({point.get('shot_count_quality') or 'unrated'})"
                )
                player_speeds = (
                    f"{format_number(player_phase.get('beginning_average_speed_m_s'), ' m/s')} → "
                    f"{format_number(player_phase.get('tail_average_speed_m_s'), ' m/s')}"
                )
                opponent_speeds = (
                    f"{format_number(opponent_phase.get('beginning_average_speed_m_s'), ' m/s')} → "
                    f"{format_number(opponent_phase.get('tail_average_speed_m_s'), ' m/s')}"
                )
                lines.append(
                    f"| {point.get('video_serve_number') or '?'} | {format_number(point.get('rally_duration_sec'), ' s')} | "
                    f"{shot_text} | {player_speeds} | {opponent_speeds} |"
                )
    else:
        lines.append("- Shot estimates are unavailable because Stage 7 was run without calibrated TrackNet tracks.")
    for code, name in ((player_code, player_name), (opponent_code, opponent_name)):
        speed_summary = effort.get("within_rally_speed_over_5_sec", {}).get(code, {})
        if speed_summary.get("rallies_with_comparable_tracking"):
            lines.append(
                f"- In rallies over 5 seconds, {name}'s average tracked speed changed from "
                f"{format_number(speed_summary.get('mean_beginning_speed_m_s'), ' m/s')} near the beginning to "
                f"{format_number(speed_summary.get('mean_tail_speed_m_s'), ' m/s')} near the end "
                f"across {speed_summary['rallies_with_comparable_tracking']} rallies."
            )
    lines += ["", "A lower tail speed can reflect tactics or pauses as well as fatigue; it is not a fatigue diagnosis."]
    heatmap = data["heatmap_timelapse"]
    lines += ["", "## Court heatmap and spatial coaching", ""]
    lines.extend(f"- {sentence}" for sentence in heatmap["coaching_commentary"])
    lines.append(f"- Timelapse phases: {', '.join(phase['name'] for phase in heatmap['phases'])}. {heatmap['timing_note']}")
    lines += ["", "## Ready position (pose estimate)", ""]
    ready = data["pose_ready_by_outcome"]
    for outcome, label in (("won", "Won"), ("lost", "Lost")):
        stats = ready[outcome]
        lines.append(
            f"- **{label}:** {stats.get('pose_frame_count', 0)} frames across {stats.get('point_count', 0)} points; "
            f"median crouch {format_number(stats.get('median_crouch_pct_vs_upright'), '%')}; "
            f"median stance-width ratio {format_number(stats.get('median_stance_width_ratio'))}."
        )
    lines += [
        "",
        "The racket-arm-up measure is unavailable for this match. These 2D pose comparisons are descriptive, not proof that posture caused a result.",
        "",
        "## Pickup evidence",
        "",
        f"- Estimated/confirmed landing side: {player_name} **{data['pickup_evidence']['landing_side_counts'][player_code]}**, {opponent_name} **{data['pickup_evidence']['landing_side_counts'][opponent_code]}**, unknown **{data['pickup_evidence']['landing_side_counts']['unknown']}** of {data['pickup_evidence']['pickup_total']} video rallies.",
        f"- {data['pickup_evidence']['manual_confirmed_count']} sides were confirmed manually. The rest are sparse, indirect evidence; do not use them to label a specific shot as an error or winner.",
        "",
        "## Losses to review",
        "",
    ]
    for point in data["example_lost_points"]:
        when = format_timestamp(point.get("serve_t"))
        duration = format_number(point.get("rally_duration_sec"), " s")
        serve_label = f"serve {point['video_serve_number']}" if point.get("video_serve_number") else f"score point {point['point_n']}"
        score_text = f"score {point.get('score_before', '?')} → {point.get('score_after', '?')}"
        sheets = ", ".join(f"`{path}`" for path in point.get("contact_sheets", [])) or "open the rally review index"
        lines.append(
            f"- **{serve_label}** (score point {point['point_n']}, {point.get('reason_label', point['reason'])}): start {when}, duration {duration}, {score_text}; "
            f"rally end {format_timestamp(point.get('rally_end_sec'))} ({point.get('rally_end_confidence') or 'unrated'} estimate); "
            f"coach review: {point['coach_note']} See {sheets}."
        )
    if not data["example_lost_points"]:
        lines.append("- No lost rallies with usable timestamps were available.")
    lines += [
        "",
        "## Coaching readout",
        "",
    ]
    short = data["by_rally_length"]["short_0_5s"]
    medium = data["by_rally_length"]["medium_5_10s"]
    long = data["by_rally_length"]["long_over_10s"]
    if short["rally_count"] and medium["rally_count"] and long["rally_count"]:
        mid_example = next(
            (point for point in data["example_lost_points"] if point["reason"].startswith("representative mid-length")),
            None,
        )
        mid_cite = (
            f" Review mid-length loss at serve {mid_example['video_serve_number']} ({format_timestamp(mid_example['serve_t'])})."
            if mid_example else " Review the mid-length-loss contact sheets above."
        )
        lines.append(
            f"- **Rally-length comparison:** {player_name} won {short.get('player_wins', short['R_wins'])}/{short['rally_count']} rallies lasting 5 seconds or less "
            f"({short.get('player_win_rate_pct', short['R_win_rate_pct'])}%), compared with {medium.get('player_wins', medium['R_wins'])}/{medium['rally_count']} in the 5–10 second band "
            f"({medium.get('player_win_rate_pct', medium['R_win_rate_pct'])}%). Use this as a broad outcome pattern, not a diagnosis of serve or return quality.{mid_cite}"
        )
    won_ready = ready.get("won", {})
    lost_ready = ready.get("lost", {})
    won_width, lost_width = won_ready.get("median_stance_width_ratio"), lost_ready.get("median_stance_width_ratio")
    if finite(won_width) and finite(lost_width):
        lines.append(
            f"- **Ready base:** stance-width ratio was {won_width} in READY frames during {player_name}'s wins and {lost_width} during losses. "
            "Try setting a stable split-step base before the opponent strikes; compare a few clips to see if the pose estimate matches what you see."
        )
    long_lost = long["by_outcome"]["lost"]["players"][player_code]
    if long["rally_count"] and long_lost["mean_trips_per_rally"] is not None:
        cited = next((p for p in data["example_lost_points"] if p["reason"] == f"highest {player_code} movement among losses"), None)
        citation = f" serve {cited['video_serve_number']} at {format_timestamp(cited['serve_t'])}" if cited else ""
        lines.append(
            f"- **Long-rally recovery:** {player_name} lost {long.get('player_losses', long['R_losses'])}/{long['rally_count']} rallies over 10 seconds; the lost long rallies averaged "
            f"{long_lost['mean_trips_per_rally']} tracked trips at least 2.2 m from the base point. The high-movement example is{citation}. "
            "Practice returning toward base after a deep-court shot, then compare the next split step in the review footage."
        )
    lines += [
        "",
        "These are practice hypotheses, not causal findings. Durations use approximate pauses; tracking trips and 2D pose can be noisy. The racket-arm-up measure is unavailable, and shuttle pickup evidence is incomplete.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Stages 5–9 into a checkable match report.")
    parser.add_argument("points_json", type=Path, help="Stage 5 points.json")
    parser.add_argument("rallies_review_json", type=Path, help="completed Stage 6 rallies_review.json")
    parser.add_argument("motion_json", type=Path, help="Stage 7 motion.json")
    parser.add_argument("players_json", type=Path, help="Stage 3 players.json for court-position time-lapse")
    parser.add_argument("pose_stats_json", type=Path, help="Stage 8 pose_stats.json")
    parser.add_argument("pickups_json", type=Path, help="Stage 9 pickups.json, including any manual corrections")
    parser.add_argument("output_json", type=Path, help="structured report data output")
    parser.add_argument("output_markdown", type=Path, help="human-readable report output")
    parser.add_argument("--player-code", choices=("R", "O"), default="R",
                        help="player to coach and report on (default: R)")
    parser.add_argument("--r-name", "--player-name", dest="player_name",
                        help="name assigned to Stage 3 identity R; prompted if omitted")
    parser.add_argument("--o-name", "--opponent-name", dest="opponent_name",
                        help="name assigned to Stage 3 identity O; prompted if omitted")
    args = parser.parse_args()
    try:
        player_name = (args.player_name or input("Name for Stage 3 identity R: ")).strip() or "You"
        opponent_name = (args.opponent_name or input("Name for Stage 3 identity O: ")).strip() or "Opponent"
        report = build_report(
            read_json(args.points_json), read_json(args.rallies_review_json), read_json(args.motion_json),
            read_json(args.players_json), read_json(args.pose_stats_json), read_json(args.pickups_json),
            player_name, opponent_name, args.player_code,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    player_code = report["analysis_subject_code"]
    opponent_code = "O" if player_code == "R" else "R"
    final_score = report["score"]["final"] or {}
    print(f"Stage 10 report complete for {report['player_names'][player_code]} ({player_code}): "
          f"{report['score']['video_rally_count']} video rallies, final score "
          f"{report['player_names'][player_code]} {final_score.get(player_code, '?')} - "
          f"{report['player_names'][opponent_code]} {final_score.get(opponent_code, '?')}.")
    print(f"Structured data: {args.output_json}")
    print(f"Report: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
