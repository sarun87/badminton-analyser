"""Shared match statistics and narration text for the optional video outro."""

from __future__ import annotations

from typing import Any


def build_match_stats(points: list[dict[str, Any]], motion: dict[str, Any]) -> dict[str, Any]:
    buckets = []
    for label, predicate in (
        ("5 seconds or less", lambda seconds: seconds <= 5),
        ("5 to 10 seconds", lambda seconds: 5 < seconds <= 10),
        ("Over 10 seconds", lambda seconds: seconds > 10),
    ):
        group = [point for point in points if predicate(float(point.get("rdur", 0)))]
        won = sum(point.get("winner") == "R" for point in group)
        lost = sum(point.get("winner") == "O" for point in group)
        buckets.append({"label": label, "won": won, "lost": lost, "total": len(group)})

    movement = motion.get("user_by_outcome", {})
    won = movement.get("won", {})
    lost = movement.get("lost", {})
    return {
        "total_points": len(points),
        "points_won": sum(point.get("winner") == "R" for point in points),
        "points_lost": sum(point.get("winner") == "O" for point in points),
        "rally_buckets": buckets,
        "movement": {
            "distance_won_m": won.get("distance_m"),
            "distance_lost_m": lost.get("distance_m"),
            "speed_won_mps": won.get("average_speed_mps"),
            "speed_lost_mps": lost.get("average_speed_mps"),
        },
    }


def narration_text(stats: dict[str, Any], include_graphs: bool = True) -> str:
    total = int(stats.get("total_points", 0))
    won = int(stats.get("points_won", 0))
    lost = int(stats.get("points_lost", 0))
    text = f"Match review. You won {won} of {total} recorded points, and lost {lost}. "
    if include_graphs:
        text += (
            "The first graph groups point outcomes by rally length: five seconds or less, five to ten seconds, and over ten seconds. "
            "Small groups have less evidence, so compare the bars as patterns rather than conclusions. "
        )
    else:
        text += "Recorded rally lengths are grouped into five seconds or less, five to ten seconds, and over ten seconds. "
    movement = stats.get("movement", {})
    distance_won = movement.get("distance_won_m")
    distance_lost = movement.get("distance_lost_m")
    if distance_won is not None and distance_lost is not None:
        if include_graphs:
            text += (
                f"The second graph shows your average tracked distance per rally: {float(distance_won):.1f} metres on points won "
                f"and {float(distance_lost):.1f} metres on points lost. Longer or more demanding rallies can require more movement; "
                "distance alone does not identify a technique problem."
            )
        else:
            text += (
                f"Your average tracked distance was {float(distance_won):.1f} metres on points won and "
                f"{float(distance_lost):.1f} metres on points lost. Longer or more demanding rallies can require more movement; "
                "distance alone does not identify a technique problem."
            )
    else:
        if include_graphs:
            text += "The second graph summarizes movement distance by point outcome. Movement data was incomplete, so treat it as a visual guide."
        else:
            text += "Movement data was incomplete, so a movement comparison is not available."
    return text
