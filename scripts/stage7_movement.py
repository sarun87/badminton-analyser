#!/usr/bin/env python3
"""Calculate per-rally movement, heatmaps, and optional TrackNet shot estimates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from shuttle_tracking import load_shuttle_tracks


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
HALF_LENGTH_M = 6.7
BASE_POINT = (3.05, 3.4)
OUT_OF_POSITION_M = 2.2
BACK_COURT_DEPTH_M = 1.6
FRONT_SERVICE_LINE_M = 4.72
MAX_TRACK_SPEED_M_S = 8.0
MAX_SHUTTLE_OBSERVATION_GAP_SEC = 0.6
NET_SIDE_MARGIN_FRACTION = 0.03
NET_PLANE_MAX_SEGMENT_GAP_SEC = 0.2
NET_PLANE_MAX_IMAGE_SPEED_PX_S = 1800.0
NET_PLANE_MAX_DIRECTIONAL_PASS_SEC = 1.25
NET_PLANE_MIN_DIRECTIONAL_TRAVEL_PX = 20.0


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def position(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2 or not all(number(item) for item in value):
        return None
    return float(value[0]), float(value[1])


def load_review(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("rally review JSON must contain a non-empty points array")
    missing_starts = []
    missing_ends = []
    for point in points:
        point_n = point.get("point_n", "?")
        if point.get("serve_t") is None:
            missing_starts.append(point_n)
        elif not number(point.get("serve_t")):
            raise ValueError(f"s{point_n} has an invalid serve_t")
        if point.get("rally_end_sec") is None:
            missing_ends.append(point_n)
        elif not number(point.get("rally_end_sec")):
            raise ValueError(f"s{point_n} has an invalid rally_end_sec")
    if missing_starts or missing_ends:
        details = []
        if missing_starts:
            details.append(f"serve_t missing for s{', s'.join(map(str, missing_starts))}")
        if missing_ends:
            details.append(f"rally_end_sec missing for s{', s'.join(map(str, missing_ends))}")
        raise ValueError("complete Stage 6 visual review first: " + "; ".join(details))
    ordered = sorted(points, key=lambda item: int(item["point_n"]))
    previous_end = None
    for point in ordered:
        start = float(point["serve_t"])
        end = float(point["rally_end_sec"])
        if end < start:
            raise ValueError(f"s{point['point_n']} rally_end_sec must not precede serve_t")
        next_t = point.get("next_serve_t")
        if next_t is not None and end > float(next_t):
            raise ValueError(f"s{point['point_n']} rally_end_sec must not pass the next serve")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"s{point['point_n']} starts before the prior reviewed rally ends")
        previous_end = end
    return ordered


def load_tracks(path: Path) -> tuple[list[float], dict[str, list[tuple[float, float] | None]], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    fps = float(data.get("fps", 0))
    if not isinstance(frames, list) or not frames or fps <= 0:
        raise ValueError("players JSON needs frames and a positive fps")
    times = [float(frame["t"]) for frame in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("players frame timestamps must increase")
    tracks: dict[str, list[tuple[float, float] | None]] = {"R": [], "O": []}
    for frame in frames:
        for player in ("R", "O"):
            point = position(frame.get("r" if player == "R" else "o"))
            if point is None:
                tracks[player].append(None)
            else:
                # Rotate whichever identity occupies the far half into their own-court view.
                if point[1] >= HALF_LENGTH_M:
                    point = (COURT_WIDTH_M - point[0], COURT_LENGTH_M - point[1])
                tracks[player].append(point)
    return times, tracks, fps


def smooth_track(
    track: list[tuple[float, float] | None], window_frames: int
) -> list[tuple[float, float] | None]:
    if window_frames % 2 == 0:
        window_frames += 1
    radius = window_frames // 2
    result = []
    for index in range(len(track)):
        values = [value for value in track[max(0, index-radius):min(len(track), index+radius+1)] if value is not None]
        if len(values) < max(1, math.ceil(window_frames / 2)):
            result.append(None)
        else:
            med = np.median(np.asarray(values, dtype=np.float64), axis=0)
            result.append((float(med[0]), float(med[1])))
    return result


def distance_to_base(point: tuple[float, float]) -> float:
    return math.dist(point, BASE_POINT)


def project_pixel(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    value = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if abs(float(value[2])) < 1e-9:
        raise ValueError("calibration maps a court landmark to infinity")
    return float(value[0] / value[2]), float(value[1] / value[2])


def calibrated_net_line(shuttle_data: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float], int, int]:
    geometry = shuttle_data.get("court_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("TrackNet shot estimates require Stage 3A tracks made with --calibration-json")
    matrix = np.asarray(geometry.get("court_to_pixel"), dtype=np.float64)
    size = geometry.get("court_size_m", {})
    if not isinstance(size, dict):
        size = {}
    width = float(size.get("width", COURT_WIDTH_M))
    length = float(size.get("length", COURT_LENGTH_M))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or width <= 0 or length <= 0:
        raise ValueError("TrackNet court geometry must contain a finite homography and court dimensions")
    net_y = length / 2.0
    left = project_pixel(matrix, (0.0, net_y))
    right = project_pixel(matrix, (width, net_y))

    def signed_distance(point: tuple[float, float]) -> float:
        dx, dy = right[0] - left[0], right[1] - left[1]
        line_length = math.hypot(dx, dy)
        if line_length <= 1e-6:
            raise ValueError("calibrated net line has zero image length")
        return (dx * (point[1] - left[1]) - dy * (point[0] - left[0])) / line_length

    near_reference = signed_distance(project_pixel(matrix, (width / 2.0, 0.0)))
    far_reference = signed_distance(project_pixel(matrix, (width / 2.0, length)))
    if near_reference * far_reference >= 0:
        raise ValueError("calibrated net line does not separate the near and far court halves")
    return left, right, 1 if near_reference > 0 else -1, 1 if far_reference > 0 else -1


def estimate_shots_for_rally(
    shuttle_data: dict[str, Any],
    rally: dict[str, Any],
    net_line: tuple[tuple[float, float], tuple[float, float], int, int],
) -> dict[str, Any]:
    start_t = float(rally["serve_t"])
    end_t = float(rally["rally_end_sec"])
    frames = shuttle_data["frames"]
    fps = float(shuttle_data["fps"])
    first = max(0, int(math.ceil(start_t * fps - 1e-6)))
    last = min(len(frames), int(math.floor(end_t * fps + 1e-6)) + 1)
    samples = frames[first:last]
    visible = [row for row in samples if row.get("visible") is True]
    on_court = [row for row in visible if row.get("court_region", "unknown") != "outside"]
    left, right, near_sign, far_sign = net_line
    dx, dy = right[0] - left[0], right[1] - left[1]
    line_length = math.hypot(dx, dy)
    threshold_px = line_length * NET_SIDE_MARGIN_FRACTION

    # Ignore points tagged outside the calibrated court polygon, plus points too close
    # to the projected net line to assign reliably to either half.
    classified: list[tuple[float, str | None]] = []
    for row in on_court:
        x, y = float(row["x_px"]), float(row["y_px"])
        signed = (dx * (y - left[1]) - dy * (x - left[0])) / line_length
        if abs(signed) < threshold_px:
            side = None
        elif signed * near_sign > 0:
            side = "near"
        elif signed * far_sign > 0:
            side = "far"
        else:
            side = None
        classified.append((float(row["t"]), side))

    transitions: list[dict[str, Any]] = []
    stable_side: str | None = None
    last_stable_t: float | None = None
    pending_side: str | None = None
    pending_first_t: float | None = None
    pending_count = 0
    last_classified_t: float | None = None
    for timestamp, side in classified:
        if side is None:
            continue
        if last_classified_t is not None and timestamp - last_classified_t > MAX_SHUTTLE_OBSERVATION_GAP_SEC:
            stable_side = None
            last_stable_t = None
            pending_side = None
            pending_first_t = None
            pending_count = 0
        last_classified_t = timestamp
        if stable_side is None:
            if pending_side == side and pending_first_t is not None and timestamp - pending_first_t <= 0.45:
                pending_count += 1
            else:
                pending_side, pending_first_t, pending_count = side, timestamp, 1
            if pending_count >= 2:
                stable_side = side
                last_stable_t = timestamp
                pending_side = None
                pending_first_t = None
                pending_count = 0
            continue
        if side == stable_side:
            last_stable_t = timestamp
            pending_side = None
            pending_first_t = None
            pending_count = 0
            continue
        if pending_side == side and pending_first_t is not None and timestamp - pending_first_t <= 0.45:
            pending_count += 1
        else:
            pending_side, pending_first_t, pending_count = side, timestamp, 1
        if pending_count >= 2:
            transition_t = (
                (last_stable_t + pending_first_t) / 2.0
                if last_stable_t is not None and pending_first_t is not None
                else pending_first_t
            )
            transitions.append({
                "t_sec": round(float(transition_t), 3),
                "from_half": stable_side,
                "to_half": side,
            })
            stable_side = side
            last_stable_t = timestamp
            pending_side = None
            pending_first_t = None
            pending_count = 0

    total_expected = max(1, int(round((end_t - start_t) * fps)) + 1)
    coverage_pct = 100.0 * len(on_court) / total_expected
    enough_data = len(on_court) >= 4 and coverage_pct >= 5.0
    estimate = len(transitions) if enough_data else None
    if coverage_pct >= 35.0:
        quality = "good_track_coverage"
    elif coverage_pct >= 15.0:
        quality = "limited_track_coverage"
    else:
        quality = "poor_track_coverage"
    return {
        "estimated_shot_count": estimate,
        "manual_shot_count": None,
        "observed_net_crossing_count": len(transitions),
        "count_quality": quality if enough_data else "insufficient_track_data",
        "track_sample_count": len(samples),
        "visible_sample_count": len(visible),
        "main_court_sample_count": len(on_court),
        "main_court_track_coverage_pct": round(coverage_pct, 1),
        "candidate_crossings": transitions,
        "method": (
            "Two consistent TrackNet observations establish a side; a stable transition across the calibrated "
            "image-space net line counts as one estimated shot. This is a crossing-based estimate and can miss "
            "net errors, gaps, or adjacent-court contamination."
        ),
    }


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def segment_polygon_intersection_fraction(
    start: tuple[float, float], end: tuple[float, float], polygon: list[tuple[float, float]]
) -> float | None:
    """Return the first segment fraction that touches the polygon, if any."""
    if point_in_polygon(start, polygon):
        return 0.0
    sx, sy = start
    rx, ry = end[0] - sx, end[1] - sy
    first_hit: float | None = None
    previous = polygon[-1]
    for current in polygon:
        qx, qy = previous
        ex, ey = current[0] - qx, current[1] - qy
        denominator = rx * ey - ry * ex
        if abs(denominator) > 1e-9:
            qpx, qpy = qx - sx, qy - sy
            t = (qpx * ey - qpy * ex) / denominator
            u = (qpx * ry - qpy * rx) / denominator
            if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
                t = min(1.0, max(0.0, t))
                first_hit = t if first_hit is None else min(first_hit, t)
        previous = current
    if first_hit is not None:
        return first_hit
    return 1.0 if point_in_polygon(end, polygon) else None


def estimate_net_plane_candidates(
    shuttle_data: dict[str, Any],
    rally: dict[str, Any],
    net_plane: dict[str, Any],
) -> dict[str, Any]:
    """Find TrackNet segments that pass through an extended image of the net plane."""
    start_t = float(rally["serve_t"])
    end_t = float(rally["rally_end_sec"])
    frames = shuttle_data["frames"]
    fps = float(shuttle_data["fps"])
    first = max(0, int(math.ceil(start_t * fps - 1e-6)))
    last = min(len(frames), int(math.floor(end_t * fps + 1e-6)) + 1)
    samples = frames[first:last]
    visible = [row for row in samples if row.get("visible") is True]
    polygon = [tuple(map(float, point)) for point in net_plane["extended_plane_polygon_px"]]
    bottom_center = (
        (polygon[0][0] + polygon[1][0]) / 2.0,
        (polygon[0][1] + polygon[1][1]) / 2.0,
    )
    top_center = (
        (polygon[2][0] + polygon[3][0]) / 2.0,
        (polygon[2][1] + polygon[3][1]) / 2.0,
    )
    plane_axis = (top_center[0] - bottom_center[0], top_center[1] - bottom_center[1])
    axis_length = math.hypot(*plane_axis)
    if axis_length <= 1e-6:
        raise ValueError("net-plane corridor has no vertical image-space extent")
    plane_axis = (plane_axis[0] / axis_length, plane_axis[1] / axis_length)
    inside_count = sum(
        point_in_polygon((float(row["x_px"]), float(row["y_px"])), polygon)
        for row in visible
    )

    intersections: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in samples:
        if row.get("visible") is not True:
            continue
        current = row
        if previous is not None:
            elapsed = float(current["t"]) - float(previous["t"])
            if 0 < elapsed <= NET_PLANE_MAX_SEGMENT_GAP_SEC:
                a = (float(previous["x_px"]), float(previous["y_px"]))
                b = (float(current["x_px"]), float(current["y_px"]))
                image_speed = math.dist(a, b) / elapsed
                if image_speed <= NET_PLANE_MAX_IMAGE_SPEED_PX_S:
                    fraction = segment_polygon_intersection_fraction(a, b, polygon)
                    if fraction is not None:
                        intersections.append({
                            "t_sec": float(previous["t"]) + elapsed * fraction,
                            "from_frame": int(previous.get("frame", first)),
                            "to_frame": int(current.get("frame", first)),
                            "from_px": [round(a[0], 1), round(a[1], 1)],
                            "to_px": [round(b[0], 1), round(b[1], 1)],
                            "image_speed_px_s": round(image_speed, 1),
                            "plane_axis_step_px": abs((b[0] - a[0]) * plane_axis[0] + (b[1] - a[1]) * plane_axis[1]),
                        })
                else:
                    # A court-to-court jump is not a continuous shuttle trajectory.
                    previous = current
                    continue
        previous = current

    # Several adjacent frames can intersect the same projected net corridor.
    # Collapse those samples into one crossing candidate per continuous pass.
    candidate_groups: list[list[dict[str, Any]]] = []
    for hit in intersections:
        if (
            not candidate_groups
            or hit["t_sec"] - candidate_groups[-1][-1]["t_sec"] > NET_PLANE_MAX_SEGMENT_GAP_SEC
            or hit["from_frame"] - candidate_groups[-1][-1]["to_frame"] > round(fps * NET_PLANE_MAX_SEGMENT_GAP_SEC)
        ):
            candidate_groups.append([hit])
        else:
            candidate_groups[-1].append(hit)

    crossings = []
    possible_crossings = []
    for group in candidate_groups:
        times = [hit["t_sec"] for hit in group]
        pass_duration = max(0.0, (group[-1]["to_frame"] - group[0]["from_frame"]) / fps)
        plane_axis_travel = sum(hit["plane_axis_step_px"] for hit in group)
        start_point = tuple(group[0]["from_px"])
        end_point = tuple(group[-1]["to_px"])
        bracketed = not point_in_polygon(start_point, polygon) and not point_in_polygon(end_point, polygon)
        directional = (
            pass_duration <= NET_PLANE_MAX_DIRECTIONAL_PASS_SEC
            and plane_axis_travel >= NET_PLANE_MIN_DIRECTIONAL_TRAVEL_PX
        )
        candidate = {
            "t_sec": round(float(np.median(times)), 3),
            "candidate_type": "image_space_net_plane_pass",
            "segment_count": len(group),
            "first_frame": group[0]["from_frame"],
            "last_frame": group[-1]["to_frame"],
            "corridor_duration_sec": round(pass_duration, 3),
            "plane_axis_travel_px": round(plane_axis_travel, 1),
            "image_path_start_px": group[0]["from_px"],
            "image_path_end_px": group[-1]["to_px"],
            "max_segment_speed_px_s": max(hit["image_speed_px_s"] for hit in group),
            "bracketed_by_observations_outside_corridor": bracketed,
        }
        if bracketed:
            candidate["evidence"] = "observed_outside_before_and_after"
            crossings.append(candidate)
        elif directional:
            # TrackNet sometimes first sees the shuttle after it has entered the
            # projected plane corridor. Keep a short, directed traversal as a
            # lower-confidence candidate instead of silently forcing the count to 0.
            candidate["evidence"] = "short_directional_pass_without_full_bracketing"
            candidate["confidence"] = "low"
            crossings.append(candidate)
        else:
            candidate["evidence"] = "ambiguous_corridor_overlap"
            possible_crossings.append(candidate)

    total_expected = max(1, int(round((end_t - start_t) * fps)) + 1)
    track_visibility_pct = 100.0 * len(visible) / total_expected
    plane_coverage_pct = 100.0 * inside_count / total_expected
    enough_data = len(visible) >= 4 and track_visibility_pct >= 5.0
    estimate = len(crossings) if enough_data and crossings else None
    return {
        "estimated_shot_count": estimate,
        "manual_shot_count": None,
        "observed_net_crossing_count": len(crossings),
        "count_quality": (
            "image_space_net_plane_candidates" if crossings else
            "no_countable_net_plane_candidate" if enough_data else
            "insufficient_track_data"
        ),
        "track_sample_count": len(samples),
        "visible_sample_count": len(visible),
        "main_court_sample_count": inside_count,
        "main_court_track_coverage_pct": round(plane_coverage_pct, 1),
        "track_visibility_pct": round(track_visibility_pct, 1),
        "net_plane_sample_count": inside_count,
        "candidate_crossings": crossings,
        "possible_unbracketed_plane_encounters": possible_crossings,
        "method": (
            "Candidate passes through a manually calibrated, extended image-space net-plane corridor. "
            "Counts include fully bracketed passes and short directional passes when TrackNet first sees the shuttle inside the corridor; "
            "the latter are low-confidence. Long or stationary corridor overlaps are listed separately and not counted. "
            "This does not estimate shuttle ground coordinates. Review timestamps for adjacent-court or projection-overlap errors."
        ),
    }


def load_net_plane(path: Path, shuttle_data: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("net-plane calibration must be a JSON object")
    source_video = data.get("source_video")
    if source_video and Path(str(source_video)).name != Path(str(shuttle_data.get("source_video", ""))).name:
        raise ValueError("net-plane calibration and TrackNet tracks refer to different videos")
    size = data.get("frame_size_px", {})
    track_size = shuttle_data.get("frame_size_px", {})
    if (int(size.get("width", -1)), int(size.get("height", -1))) != (
        int(track_size.get("width", -2)), int(track_size.get("height", -2))
    ):
        raise ValueError("net-plane calibration frame size differs from TrackNet tracks")
    polygon = data.get("extended_plane_polygon_px")
    if (
        not isinstance(polygon, list)
        or len(polygon) != 4
        or any(
            not isinstance(point, list) or len(point) != 2
            or not all(number(value) for value in point)
            for point in polygon
        )
    ):
        raise ValueError("net-plane calibration needs four finite extended_plane_polygon_px points")
    data["extended_plane_polygon_px"] = [[float(value) for value in point] for point in polygon]
    return data


def mean_speed_in_window(
    track: list[tuple[float, float] | None], times: list[float], start_t: float, end_t: float
) -> tuple[float | None, float]:
    samples = [
        (times[index], track[index])
        for index in range(len(times))
        if start_t <= times[index] <= end_t and track[index] is not None
    ]
    distance = 0.0
    observed_time = 0.0
    for (ta, a), (tb, b) in zip(samples, samples[1:]):
        dt = tb - ta
        step = math.dist(a, b)
        if 0 < dt <= 0.35 and step / dt <= MAX_TRACK_SPEED_M_S:
            distance += step
            observed_time += dt
    speed = distance / observed_time if observed_time >= 0.3 else None
    return (round(speed, 2) if speed is not None else None), round(observed_time, 2)


def rally_phase_speeds(
    track: list[tuple[float, float] | None], times: list[float], start_t: float, end_t: float
) -> dict[str, Any]:
    active_start = start_t + 0.3
    active_duration = end_t - active_start
    window = min(2.0, 0.4 * active_duration)
    if window < 1.0:
        return {
            "window_sec": round(max(0.0, window), 2),
            "beginning_average_speed_m_s": None,
            "tail_average_speed_m_s": None,
            "tail_minus_beginning_m_s": None,
            "tail_change_pct": None,
            "comparison_status": "rally_too_short_for_reliable_comparison",
        }
    beginning, beginning_observed = mean_speed_in_window(
        track, times, active_start, active_start + window
    )
    tail, tail_observed = mean_speed_in_window(
        track, times, end_t - window, end_t
    )
    change = round(tail - beginning, 2) if beginning is not None and tail is not None else None
    change_pct = (
        round(100.0 * (tail - beginning) / beginning, 1)
        if beginning is not None and tail is not None and beginning >= 0.05 else None
    )
    return {
        "window_sec": round(window, 2),
        "beginning_average_speed_m_s": beginning,
        "tail_average_speed_m_s": tail,
        "tail_minus_beginning_m_s": change,
        "tail_change_pct": change_pct,
        "beginning_observed_time_sec": beginning_observed,
        "tail_observed_time_sec": tail_observed,
        "comparison_status": "available" if beginning is not None and tail is not None else "insufficient_player_track_data",
    }


def point_metrics(
    player: str,
    track: list[tuple[float, float] | None],
    times: list[float],
    start_t: float,
    end_t: float,
    fps: float,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    active_start = start_t + 0.3
    indexes = [i for i, t in enumerate(times) if active_start <= t <= end_t and track[i] is not None]
    samples = [(times[i], track[i]) for i in indexes]
    if not samples:
        return {
            "distance_m": 0.0, "observed_time_sec": 0.0, "average_speed_m_s": None,
            "peak_speed_m_s": None, "back_court_time_share": None,
            "front_court_time_share": None, "median_position_local_m": None,
            "out_of_position_trips": [],
        }, []

    distance = 0.0
    observed_time = 0.0
    speeds = []
    for (ta, a), (tb, b) in zip(samples, samples[1:]):
        dt = tb - ta
        if dt <= 0 or dt > 0.35:
            continue
        speed = math.dist(a, b) / dt
        if speed > MAX_TRACK_SPEED_M_S:
            # A larger one-frame jump is a tracking mix-up, not human court movement.
            continue
        distance += math.dist(a, b)
        observed_time += dt
        speeds.append(speed)

    us = [p[0] for _, p in samples]
    vs = [p[1] for _, p in samples]
    back_share = sum(v <= BACK_COURT_DEPTH_M for v in vs) / len(vs)
    front_share = sum(v >= FRONT_SERVICE_LINE_M for v in vs) / len(vs)

    trips = []
    trip_indexes = []
    last_index = None
    for sample_index, (frame_index, (t, p)) in enumerate(zip(indexes, samples)):
        away = distance_to_base(p) >= OUT_OF_POSITION_M
        gap_break = last_index is not None and times[frame_index] - times[last_index] > 0.35
        if gap_break and trip_indexes:
            trips.append(finish_trip(trip_indexes, samples, times, indexes))
            trip_indexes = []
        if away:
            trip_indexes.append(sample_index)
        elif trip_indexes:
            trips.append(finish_trip(trip_indexes, samples, times, indexes))
            trip_indexes = []
        last_index = frame_index
    if trip_indexes:
        trips.append(finish_trip(trip_indexes, samples, times, indexes))

    points = [point for _, point in samples]
    median = np.median(np.asarray(points, dtype=np.float64), axis=0)
    local_points = [(float(p[0]), float(p[1])) for p in points]
    metrics = {
        "distance_m": round(distance, 2),
        "observed_time_sec": round(observed_time, 2),
        "average_speed_m_s": round(distance / observed_time, 2) if observed_time > 0 else None,
        "peak_speed_m_s": round(max(speeds), 2) if speeds else None,
        "back_court_time_share": round(back_share, 3),
        "front_court_time_share": round(front_share, 3),
        "median_position_local_m": [round(float(median[0]), 2), round(float(median[1]), 2)],
        "out_of_position_trips": trips,
    }
    return metrics, local_points


def finish_trip(
    sample_indexes: list[int],
    samples: list[tuple[float, tuple[float, float]]],
    times: list[float],
    frame_indexes: list[int],
) -> dict[str, Any]:
    first = sample_indexes[0]
    last = sample_indexes[-1]
    max_away = max(distance_to_base(samples[i][1]) for i in sample_indexes)
    start_t = samples[first][0]
    target_t = start_t + 1.0
    possible = [
        (abs(t - target_t), p)
        for t, p in samples
        if abs(t - target_t) <= 0.2
    ]
    recovery = distance_to_base(min(possible, key=lambda item: item[0])[1]) if possible else None
    return {
        "start_t": round(start_t, 2),
        "end_t": round(samples[last][0], 2),
        "max_distance_from_base_m": round(max_away, 2),
        "distance_from_base_1s_later_m": None if recovery is None else round(recovery, 2),
    }


def make_heatmap(path: Path, player: str, points: list[tuple[float, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = [point[0] for point in points]
    v = [point[1] for point in points]
    histogram, _, _ = np.histogram2d(
        v, u, bins=(22, 20), range=((0, HALF_LENGTH_M), (0, COURT_WIDTH_M))
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(
        histogram, origin="lower", extent=(0, COURT_WIDTH_M, 0, HALF_LENGTH_M),
        aspect="equal", cmap="magma",
    )
    ax.set(xlim=(0, COURT_WIDTH_M), ylim=(0, HALF_LENGTH_M),
           xlabel="Across from player's own left sideline (m)",
           ylabel="Distance from player's own baseline (m)",
           title=f"{player} movement heatmap")
    ax.axhline(FRONT_SERVICE_LINE_M, color="cyan", linewidth=0.8, linestyle="--")
    ax.scatter([BASE_POINT[0]], [BASE_POINT[1]], marker="+", color="white", label="base point")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Sample count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate player movement per reviewed rally, with optional TrackNet shot estimates."
    )
    parser.add_argument("players_json", type=Path, help="Stage 3 players.json")
    parser.add_argument("rallies_review_json", type=Path, help="completed Stage 6 review JSON")
    parser.add_argument("output_json", type=Path, help="output movement metrics JSON")
    parser.add_argument("heatmap_dir", type=Path, help="directory for R/O heatmap PNGs")
    parser.add_argument(
        "--shuttle-tracks", type=Path,
        help="optional Stage 3A TrackNet JSON for estimated per-rally shot counts",
    )
    parser.add_argument(
        "--net-plane-json", type=Path,
        help="optional clicked image-space net-plane calibration; enables candidate-pass counting instead of floor-line counting",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    players_path = args.players_json.expanduser().resolve()
    review_path = args.rallies_review_json.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    heatmap_dir = args.heatmap_dir.expanduser().resolve()
    shuttle_path = args.shuttle_tracks.expanduser().resolve() if args.shuttle_tracks else None
    net_plane_path = args.net_plane_json.expanduser().resolve() if args.net_plane_json else None
    if not players_path.is_file() or not review_path.is_file():
        parser.error("players JSON and completed rally review JSON must both exist")
    if shuttle_path is not None and not shuttle_path.is_file():
        parser.error(f"TrackNet shuttle tracks do not exist: {shuttle_path}")
    if net_plane_path is not None and not net_plane_path.is_file():
        parser.error(f"net-plane calibration does not exist: {net_plane_path}")
    if net_plane_path is not None and shuttle_path is None:
        parser.error("--net-plane-json requires --shuttle-tracks")
    if output_path in (players_path, review_path, shuttle_path, net_plane_path) or heatmap_dir in (players_path, review_path, output_path, shuttle_path, net_plane_path):
        parser.error("input and output paths must be different")
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {output_path} (pass --overwrite)")
    if heatmap_dir.exists() and not args.overwrite:
        parser.error(f"heatmap directory already exists: {heatmap_dir} (pass --overwrite)")

    rallies = load_review(review_path)
    times, raw_tracks, fps = load_tracks(players_path)
    try:
        shuttle_data = load_shuttle_tracks(shuttle_path) if shuttle_path is not None else None
        net_plane = load_net_plane(net_plane_path, shuttle_data) if net_plane_path is not None and shuttle_data is not None else None
        net_line = (
            calibrated_net_line(shuttle_data)
            if shuttle_data is not None and net_plane is None else None
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if shuttle_data is not None:
        if not shuttle_data["frames"]:
            parser.error("TrackNet shuttle tracks contain no frames")
        player_data = json.loads(players_path.read_text(encoding="utf-8"))
        source_fps = player_data.get("source_fps")
        if number(source_fps) and abs(float(source_fps) - float(shuttle_data["fps"])) > 0.01:
            parser.error("TrackNet frame rate differs from the Stage 3 source video frame rate")
        track_end = float(shuttle_data["frames"][-1]["t"])
        latest_rally_end = max(float(row["rally_end_sec"]) for row in rallies)
        if latest_rally_end > track_end + 1.0 / float(shuttle_data["fps"]):
            parser.error("TrackNet tracks do not cover the full Stage 6 rally timeline; use tracks from the same proxy")
    window_frames = max(1, round(0.7 * fps))
    if window_frames % 2 == 0:
        window_frames += 1
    tracks = {player: smooth_track(raw_tracks[player], window_frames) for player in ("R", "O")}

    point_results = []
    heat_points: dict[str, list[tuple[float, float]]] = {"R": [], "O": []}
    total_distance = {"R": 0.0, "O": 0.0}
    total_time = {"R": 0.0, "O": 0.0}
    max_speed: dict[str, float | None] = {"R": None, "O": None}
    for rally in rallies:
        start_t = float(rally["serve_t"])
        end_t = float(rally["rally_end_sec"])
        players = {}
        for player in ("R", "O"):
            metrics, local_points = point_metrics(player, tracks[player], times, start_t, end_t, fps)
            metrics["rally_phase_speed"] = rally_phase_speeds(
                tracks[player], times, start_t, end_t
            )
            players[player] = metrics
            heat_points[player].extend(local_points)
            total_distance[player] += metrics["distance_m"]
            total_time[player] += metrics["observed_time_sec"]
            speed = metrics["peak_speed_m_s"]
            if speed is not None and (max_speed[player] is None or speed > max_speed[player]):
                max_speed[player] = speed
        point_result = {
            "point_n": rally["point_n"], "candidate_n": rally.get("candidate_n"),
            "serve_t": start_t, "rally_end_sec": end_t,
            "rally_end_estimated": bool(rally.get("rally_end_estimated", False)),
            "rally_end_method": rally.get("rally_end_method", "manual_review"),
            "rally_end_confidence": rally.get("rally_end_confidence", "reviewed"),
            "serve_time_estimated": bool(rally.get("serve_time_estimated", False)),
            "serve_time_method": rally.get("serve_time_method"),
            "rally_duration_sec": round(end_t - start_t, 3),
            "movement_window_start_sec": round(start_t + 0.3, 3),
            "players": players,
        }
        if shuttle_data is not None:
            if net_plane is not None:
                point_result["shuttle_shot_analysis"] = estimate_net_plane_candidates(
                    shuttle_data, rally, net_plane
                )
            elif net_line is not None:
                point_result["shuttle_shot_analysis"] = estimate_shots_for_rally(
                    shuttle_data, rally, net_line
                )
        point_results.append(point_result)

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old_file in heatmap_dir.glob("*_heatmap.png"):
            old_file.unlink()
    heatmap_paths = {}
    for player in ("R", "O"):
        heatmap_path = heatmap_dir / f"{player}_heatmap.png"
        make_heatmap(heatmap_path, player, heat_points[player])
        heatmap_paths[player] = str(heatmap_path.relative_to(output_path.parent))

    summary = {}
    for player in ("R", "O"):
        summary[player] = {
            "distance_m": round(total_distance[player], 2),
            "observed_time_sec": round(total_time[player], 2),
            "average_speed_m_s": round(total_distance[player] / total_time[player], 2) if total_time[player] > 0 else None,
            "peak_speed_m_s": max_speed[player],
        }
    output = {
        "source_players_json": players_path.name,
        "source_rallies_review_json": review_path.name,
        "source_shuttle_tracks_json": shuttle_path.name if shuttle_path else None,
        "end_time_estimation": json.loads(review_path.read_text(encoding="utf-8")).get("end_time_estimation"),
        "smoothing_window_sec": 0.7,
        "movement_starts_sec_after_serve": 0.3,
        "rally_phase_speed_method": (
            "Compare average tracked speed in the first and last portions of live play, excluding the initial 0.3 s after serve. "
            "Windows are at most 2 s and use 40% of live duration for shorter rallies; missing or short player tracks are omitted. "
            "A change is descriptive and is not proof of fatigue."
        ),
        "shuttle_shot_count_method": (
            (
                "Optional candidate passes through a manually calibrated, extended image-space net-plane corridor. "
                "This does not estimate ground coordinates and can be affected by adjacent-court detections or image-projection overlap; review candidate timestamps."
                if net_plane is not None else
                "Optional crossing-based estimate from stable TrackNet side transitions across the calibrated image-space floor net line. "
                "It can miss airborne shuttle paths, strokes that do not cross the net, missed detections, and adjacent-court selections; review candidate timestamps."
            ) if shuttle_data is not None else None
        ),
        "source_net_plane_calibration_json": net_plane_path.name if net_plane_path is not None else None,
        "court_coordinate_system": "local player view; u across from own left sideline, v from own baseline",
        "metrics_by_player": summary,
        "heatmaps": heatmap_paths,
        "points": point_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Rallies analyzed: {len(point_results)}; smoothing: {window_frames} frames ({window_frames/fps:.1f}s).")
    print(
        "Overall average speed during analyzed windows: "
        f"R {summary['R']['average_speed_m_s']} m/s, "
        f"O {summary['O']['average_speed_m_s']} m/s."
    )
    if shuttle_data is not None:
        estimated = [
            row["shuttle_shot_analysis"]["estimated_shot_count"]
            for row in point_results
            if isinstance(row.get("shuttle_shot_analysis"), dict)
            and isinstance(row["shuttle_shot_analysis"].get("estimated_shot_count"), int)
        ]
        print(
            f"TrackNet shot estimates: {len(estimated)}/{len(point_results)} rallies had usable track coverage; "
            f"mean observed crossings per estimated rally: "
            f"{round(sum(estimated) / len(estimated), 1) if estimated else 'unavailable'}."
        )
    print(f"R heatmap: {heatmap_paths['R']}")
    print(f"O heatmap: {heatmap_paths['O']}")
    print(f"Motion JSON: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Stage 7 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
