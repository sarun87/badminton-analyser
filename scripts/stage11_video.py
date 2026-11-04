#!/usr/bin/env python3
"""Render a local match-analysis video with tracking, court map, pose, and coach notes."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from stage8_pose import POSE_CONNECTIONS, player_crop, posture_metrics
from video_encoding import h264_encoder_args, select_h264_encoder


COURT_W = 6.1
COURT_L = 13.4
COLORS = {"R": (60, 240, 210), "O": (50, 160, 255)}  # BGR: teal and orange
FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_TRANSLATION = str.maketrans({"·": "|", "–": "-", "—": "-", "→": "->", "≤": "<=", "≥": ">=", "’": "'"})


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def point2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2 or not all(finite(v) for v in value):
        return None
    return float(value[0]), float(value[1])


def interpolate_pair(a: Any, b: Any, alpha: float) -> Any:
    if (
        isinstance(a, list) and isinstance(b, list) and len(a) >= 4 and len(b) >= 4
        and all(finite(v) for v in a[:4]) and all(finite(v) for v in b[:4])
    ):
        return [float(a[i]) * (1 - alpha) + float(b[i]) * alpha for i in range(4)] + [
            float(a[4]) * (1 - alpha) + float(b[4]) * alpha if len(a) > 4 and len(b) > 4 else 1.0
        ]
    return a


def frame_track(frames: list[dict[str, Any]], times: list[float], timestamp: float, cursor: int) -> tuple[dict[str, Any], int]:
    while cursor + 1 < len(times) and times[cursor + 1] <= timestamp:
        cursor += 1
    left = frames[cursor]
    if cursor + 1 >= len(times) or times[cursor + 1] - times[cursor] > 0.25:
        return left, cursor
    right = frames[cursor + 1]
    alpha = (timestamp - times[cursor]) / max(1e-6, times[cursor + 1] - times[cursor])
    result = dict(left)
    for key in ("r", "o"):
        pa, pb = point2(left.get(key)), point2(right.get(key))
        if pa is not None and pb is not None:
            result[key] = [pa[0] * (1 - alpha) + pb[0] * alpha, pa[1] * (1 - alpha) + pb[1] * alpha]
        elif pa is None:
            result[key] = None
    for key in ("rb", "ob"):
        result[key] = interpolate_pair(left.get(key), right.get(key), alpha)
    return result, cursor


def compute_speeds(frames: list[dict[str, Any]], times: list[float], player: str) -> list[float | None]:
    key = "r" if player == "R" else "o"
    points = [point2(frame.get(key)) for frame in frames]
    speeds: list[float | None] = [None] * len(frames)
    for i in range(1, len(frames)):
        dt = times[i] - times[i - 1]
        a, b = points[i - 1], points[i]
        if a is None or b is None or dt <= 0 or dt > 0.35:
            continue
        value = math.dist(a, b) / dt
        if value <= 8.0:
            speeds[i] = value
    return speeds


def nearest_index(times: list[float], timestamp: float) -> int:
    index = bisect.bisect_left(times, timestamp)
    if index <= 0:
        return 0
    if index >= len(times):
        return len(times) - 1
    return index if times[index] - timestamp < timestamp - times[index - 1] else index - 1


def precompute_pose(
    video: Path,
    model_path: Path,
    track_frames: list[dict[str, Any]],
    track_times: list[float],
    pose_frames: list[dict[str, Any]],
    fps: float,
) -> list[dict[str, Any]]:
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("MediaPipe Tasks is unavailable; activate the project .venv") from exc
    if not model_path.is_file():
        raise ValueError(f"pose model not found: {model_path}")

    active = [row for row in pose_frames if row.get("point_n") is not None]
    requests = [(round(float(row["t"]) * fps), float(row["t"])) for row in active if finite(row.get("t"))]
    if not requests:
        return []

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    current_frame = -1
    track_cursor = 0
    output: list[dict[str, Any]] = []
    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        for request_n, (target_frame, timestamp) in enumerate(requests, start=1):
            while current_frame + 1 < target_frame:
                if not capture.grab():
                    break
                current_frame += 1
            ok, image = capture.read()
            current_frame += 1
            if not ok:
                continue
            track_cursor = nearest_index(track_times, timestamp)
            crop_data = player_crop(image, track_frames[track_cursor].get("rb"))
            if crop_data is None:
                continue
            crop, origin_x, origin_y, side = crop_data
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, round(timestamp * 1000))
            if not result.pose_landmarks:
                continue
            metrics = posture_metrics(result.pose_landmarks[0], side)
            points = metrics.get("landmarks", [])
            visibility = metrics.get("visibility", {})
            if not points:
                continue
            output.append({
                "t": timestamp,
                "points": [(float(x) + origin_x, float(y) + origin_y) for x, y in points],
                "visibility": [float(visibility.get(i, 1.0)) for i in range(len(points))],
            })
            if request_n % 300 == 0:
                print(f"Pose overlay samples: {request_n}/{len(requests)}", flush=True)
    finally:
        landmarker.close()
        capture.release()
    print(f"Pose overlay samples ready: {len(output)}", flush=True)
    return output


def pose_at(samples: list[dict[str, Any]], times: list[float], timestamp: float) -> tuple[list[tuple[float, float]], list[float]] | None:
    if not samples:
        return None
    right = bisect.bisect_left(times, timestamp)
    if right == 0:
        if times[0] - timestamp > 0.15:
            return None
        row = samples[0]
        return row["points"], row["visibility"]
    if right >= len(samples):
        if timestamp - times[-1] > 0.15:
            return None
        row = samples[-1]
        return row["points"], row["visibility"]
    left_row, right_row = samples[right - 1], samples[right]
    gap = times[right] - times[right - 1]
    if gap > 0.25:
        return None
    alpha = (timestamp - times[right - 1]) / max(gap, 1e-6)
    points = [
        (a[0] * (1 - alpha) + b[0] * alpha, a[1] * (1 - alpha) + b[1] * alpha)
        for a, b in zip(left_row["points"], right_row["points"])
    ]
    visibility = [min(a, b) for a, b in zip(left_row["visibility"], right_row["visibility"])]
    return points, visibility


def draw_pose(frame: np.ndarray, pose: tuple[list[tuple[float, float]], list[float]] | None) -> None:
    if pose is None:
        return
    points, visible = pose
    for start, end in POSE_CONNECTIONS:
        if end >= len(points) or visible[start] < 0.35 or visible[end] < 0.35:
            continue
        a = tuple(round(v) for v in points[start])
        b = tuple(round(v) for v in points[end])
        cv2.line(frame, a, b, (80, 255, 110), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        if visible[index] >= 0.4:
            cv2.circle(frame, tuple(round(v) for v in point), 3, (50, 60, 255), -1, cv2.LINE_AA)


def overlay_panel(frame: np.ndarray, rect: tuple[int, int, int, int], alpha: float = 0.62) -> None:
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, roi)


def text(frame: np.ndarray, label: str, origin: tuple[int, int], scale: float = 0.55,
         color: tuple[int, int, int] = (245, 245, 245), thickness: int = 1) -> None:
    label = label.translate(DISPLAY_TRANSLATION)
    cv2.putText(frame, label, origin, FONT, scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def foot_pixel(box: Any, width: int, height: int) -> tuple[int, int] | None:
    if not isinstance(box, list) or len(box) < 4 or not all(finite(v) for v in box[:4]):
        return None
    x1, y1, x2, y2 = map(float, box[:4])
    return round((x1 + x2) * width * 0.5), round(y2 * height)


def draw_minimap(frame: np.ndarray, positions: dict[str, tuple[float, float] | None], speeds: dict[str, float | None]) -> None:
    x, y, w, h = frame.shape[1] - 172, 64, 158, 338
    overlay_panel(frame, (x, y, x + w, y + h), 0.72)
    text(frame, "COURT MAP", (x + 10, y + 20), 0.45, thickness=1)
    left, top, court_w, court_h = x + 15, y + 30, w - 30, h - 70

    def px(mx: float) -> int:
        return round(left + mx / COURT_W * court_w)

    def py(my: float) -> int:
        # Court y=0 is the camera-side near baseline; place it at the bottom.
        return round(top + (COURT_L - my) / COURT_L * court_h)

    white = (215, 215, 215)
    cv2.rectangle(frame, (left, top), (left + court_w, top + court_h), white, 1)
    cv2.line(frame, (px(3.05), top), (px(3.05), top + court_h), white, 1)
    for yy in (0.76, 4.72, 6.7, 8.68, 12.64):
        cv2.line(frame, (left, py(yy)), (left + court_w, py(yy)), (145, 145, 145), 1)
    cv2.line(frame, (px(0.46), top), (px(0.46), top + court_h), (120, 120, 120), 1)
    cv2.line(frame, (px(5.64), top), (px(5.64), top + court_h), (120, 120, 120), 1)
    for player in ("R", "O"):
        point = positions.get(player)
        if point is not None:
            cv2.circle(frame, (px(point[0]), py(point[1])), 6, COLORS[player], -1, cv2.LINE_AA)
            cv2.circle(frame, (px(point[0]), py(point[1])), 8, (255, 255, 255), 1, cv2.LINE_AA)
    speed_text = "R --" if speeds.get("R") is None else f"R {speeds['R']:.1f} m/s"
    speed_text += "   O --" if speeds.get("O") is None else f"   O {speeds['O']:.1f} m/s"
    text(frame, speed_text, (x + 9, y + h - 14), 0.38, thickness=1)


def current_point(points: list[dict[str, Any]], timestamp: float, cursor: int) -> tuple[dict[str, Any] | None, int]:
    while cursor + 1 < len(points) and float(points[cursor + 1]["serve_t"]) <= timestamp:
        cursor += 1
    if cursor < 0 or not points or timestamp < float(points[0]["serve_t"]):
        return None, cursor
    return points[cursor], cursor


def parse_score(value: Any, fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    if isinstance(value, str) and "-" in value:
        try:
            left, right = value.split("-", 1)
            return int(left), int(right)
        except ValueError:
            return fallback
    return fallback


def draw_score_box(
    frame: np.ndarray,
    names: dict[str, str],
    score: tuple[int, int],
    serving_player: str | None,
    rect: tuple[int, int, int, int],
    label: str = "CURRENT SCORE",
) -> None:
    x, y, w, h = rect
    overlay_panel(frame, (x, y, x + w, y + h), 0.72)
    text(frame, label, (x + 12, y + 21), 0.39, (190, 210, 220), 1)
    separator = x + w - 78
    text(frame, "SCORE", (separator + 12, y + 21), 0.31, (190, 210, 220), 1)
    cv2.line(frame, (separator, y + 28), (separator, y + h - 9), (120, 140, 150), 1)
    for row_i, player in enumerate(("R", "O")):
        row_y = y + 50 + row_i * 30
        if serving_player == player:
            cv2.circle(frame, (x + 18, row_y - 5), 6, (80, 245, 235), -1, cv2.LINE_AA)
            cv2.circle(frame, (x + 18, row_y - 5), 8, (245, 245, 245), 1, cv2.LINE_AA)
        name_x = x + 34
        name = names[player]
        scale = 0.58
        max_width = separator - name_x - 12
        while scale > 0.36 and cv2.getTextSize(name, FONT, scale, 1)[0][0] > max_width:
            scale -= 0.03
        if cv2.getTextSize(name, FONT, scale, 1)[0][0] > max_width:
            while name and cv2.getTextSize(name + "...", FONT, scale, 1)[0][0] > max_width:
                name = name[:-1]
            name = name.rstrip() + "..."
        text(frame, name, (name_x, row_y), scale, COLORS[player], 1)
        value = str(score[0] if player == "R" else score[1])
        text(frame, value, (separator + 24, row_y), 0.63, (250, 250, 250), 2)
    cv2.line(frame, (x + 11, y + 64), (x + w - 10, y + 64), (85, 100, 110), 1)


def draw_hud(
    frame: np.ndarray,
    point: dict[str, Any] | None,
    next_point: dict[str, Any] | None,
    timestamp: float,
    names: dict[str, str],
) -> None:
    x, y = 18, 16
    w = min(480, frame.shape[1] - 36)
    if point is None:
        upcoming = next_point
        score = parse_score(upcoming.get("score_before")) if upcoming else (0, 0)
        serving_player = upcoming.get("server") if upcoming and upcoming.get("server") in ("R", "O") else None
        serve_number = upcoming.get("video_serve_number", upcoming.get("point_n")) if upcoming else None
        status = f"FIRST  ·  SERVE {serve_number}" if upcoming else "MATCH SCORE"
    else:
        end_t = float(point.get("rally_end_sec") or timestamp)
        before = parse_score(point.get("score_before"))
        after = parse_score(point.get("score_after"), before)
        serve_number = point.get("video_serve_number", point.get("point_n"))
        if timestamp <= end_t:
            score = before
            serving_player = point.get("server") if point.get("server") in ("R", "O") else None
            status = f"LIVE RALLY  ·  SERVE {serve_number}"
        elif next_point is not None:
            # Stage 5 may include a score-only point with no video timestamp.
            score = parse_score(next_point.get("score_before"), after)
            serving_player = next_point.get("server") if next_point.get("server") in ("R", "O") else None
            next_serve_number = next_point.get("video_serve_number", next_point.get("point_n"))
            status = f"NEXT  ·  SERVE {next_serve_number}"
        else:
            score = after
            serving_player = None
            status = "FINAL SCORE"
    draw_score_box(frame, names, score, serving_player, (x, y, w, 98), status)


def draw_tracking(
    frame: np.ndarray, boxes: dict[str, Any], history: dict[str, deque], timestamp: float,
    names: dict[str, str],
) -> None:
    height, width = frame.shape[:2]
    for player in ("R", "O"):
        color = COLORS[player]
        foot = foot_pixel(boxes.get("rb" if player == "R" else "ob"), width, height)
        if foot is None:
            if history[player] and timestamp - history[player][-1][0] > 0.35:
                history[player].clear()
            continue
        history[player].append((timestamp, foot))
        track = list(history[player])
        for i in range(1, len(track)):
            age = timestamp - track[i][0]
            if age > 2.0:
                continue
            strength = max(0.2, 1.0 - age / 2.0)
            shade = tuple(round(c * strength) for c in color)
            cv2.line(frame, track[i - 1][1], track[i][1], shade, max(1, round(4 * strength)), cv2.LINE_AA)
        radius_x = max(12, round((foot[1] / max(1, height)) * 17))
        cv2.ellipse(frame, foot, (radius_x, 6), 0, 0, 360, color, 2, cv2.LINE_AA)
        cv2.circle(frame, foot, 4, color, -1, cv2.LINE_AA)
        label_y = max(24, foot[1] - 22)
        label = names[player]
        text(frame, label, (foot[0] - 33, label_y), 0.39, color, 1)


def wrap_text(value: str, max_width: int, scale: float, thickness: int = 1) -> list[str]:
    value = value.translate(DISPLAY_TRANSLATION)
    output: list[str] = []
    current = ""
    for word in value.split():
        candidate = word if not current else current + " " + word
        if current and cv2.getTextSize(candidate, FONT, scale, thickness)[0][0] > max_width:
            output.append(current)
            current = word
        else:
            current = candidate
    if current:
        output.append(current)
    return output


def draw_freeze_card(frame: np.ndarray, headline: str, body: str, subline: str) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    overlay_panel(result, (0, height - 152, width, height), 0.76)
    text(result, headline, (26, height - 112), 0.7, (80, 245, 235), 2)
    lines = wrap_text(body, width - 55, 0.55)
    for line_i, line in enumerate(lines[:2]):
        text(result, line, (28, height - 78 + line_i * 25), 0.55, (250, 250, 250), 1)
    sublines = wrap_text(subline, width - 55, 0.38)
    first_sub_y = height - 16 - 18 * max(0, len(sublines) - 1)
    for line_i, line in enumerate(sublines[:2]):
        text(result, line, (28, first_sub_y + line_i * 18), 0.38, (190, 220, 220), 1)
    return result


def analysis_slide(width: int, height: int, eyebrow: str, title: str) -> np.ndarray:
    frame = np.full((height, width, 3), (27, 25, 19), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (12, height), (210, 190, 65), -1)
    text(frame, eyebrow.upper(), (42, 42), 0.43, (190, 210, 220), 1)
    text(frame, title, (42, 88), 0.92, (248, 248, 245), 2)
    return frame


def draw_wrapped_lines(
    frame: np.ndarray, value: str, origin: tuple[int, int], max_width: int,
    scale: float, color: tuple[int, int, int], max_lines: int, line_gap: int = 24,
) -> int:
    x, y = origin
    lines = wrap_text(value, max_width, scale)
    if len(lines) > max_lines and max_lines:
        lines = lines[:max_lines]
        if lines[-1] and not lines[-1].endswith("..."):
            lines[-1] = lines[-1].rstrip(" .") + "..."
    for line in lines:
        text(frame, line, (x, y), scale, color, 1)
        y += line_gap
    return y


def draw_dashed_line(
    frame: np.ndarray, start: tuple[int, int], end: tuple[int, int],
    color: tuple[int, int, int], thickness: int = 2,
) -> None:
    length = math.dist(start, end)
    if length < 1:
        return
    dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    for offset in range(0, int(length), 12):
        stop = min(length, offset + 6)
        a = (round(start[0] + dx * offset), round(start[1] + dy * offset))
        b = (round(start[0] + dx * stop), round(start[1] + dy * stop))
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def draw_score_intro(
    frame: np.ndarray, curve: list[dict[str, Any]], visible_count: int,
    names: dict[str, str], commentary: list[str],
) -> None:
    height, width = frame.shape[:2]
    text(frame, f"{names['R']}  ·  TEAL", (46, 126), 0.46, COLORS["R"], 1)
    text(frame, f"{names['O']}  ·  ORANGE", (width - 310, 126), 0.46, COLORS["O"], 1)
    note_y = 166
    for sentence in commentary[:2]:
        note_y = draw_wrapped_lines(frame, sentence, (46, note_y), width - 92, 0.40, (178, 210, 216), 2, 20) + 3

    chart_left, chart_top = 86, max(302, note_y + 20)
    chart_right, chart_bottom = width - 72, height - 78
    if chart_top >= chart_bottom - 100:
        chart_top = height // 2 - 28
    cv2.rectangle(frame, (chart_left, chart_top), (chart_right, chart_bottom), (130, 145, 150), 1)
    text(frame, "RUNNING POINTS", (chart_left, chart_top - 15), 0.40, (210, 220, 225), 1)
    plot_top = chart_top + 50
    legend_y = chart_top + 25
    legend_columns = (chart_left + 18, chart_left + (chart_right - chart_left) // 2)
    for player, legend_x in zip(("R", "O"), legend_columns):
        cv2.line(frame, (legend_x, legend_y), (legend_x + 34, legend_y), COLORS[player], 3, cv2.LINE_AA)
        text(frame, names[player], (legend_x + 43, legend_y + 5), 0.38, COLORS[player], 1)
    shown = curve[:max(0, min(visible_count, len(curve)))]
    total_rows = max(1, len(curve))
    final_peak = max((max(int(row.get("R", 0)), int(row.get("O", 0))) for row in curve), default=1)
    score_max = max(5, final_peak + 2)
    y_ticks = sorted({0, score_max // 2, score_max})
    for score_tick in y_ticks:
        yy = chart_bottom - round(score_tick / score_max * (chart_bottom - plot_top))
        cv2.line(frame, (chart_left, yy), (chart_right, yy), (65, 77, 82), 1)
        text(frame, str(score_tick), (chart_left - 35, yy + 5), 0.34, (165, 180, 184), 1)
    text(frame, "SCORE ENTRY", (chart_right - 95, chart_bottom + 32), 0.34, (165, 180, 184), 1)

    def chart_point(row: dict[str, Any], player: str) -> tuple[int, int]:
        n = int(row.get("point_n", 0))
        x = chart_left + round(n / total_rows * (chart_right - chart_left))
        score_value = int(row.get(player, 0))
        y = chart_bottom - round(score_value / score_max * (chart_bottom - plot_top))
        return x, y

    plot_rows = [{"point_n": 0, "R": 0, "O": 0}] + shown
    for player in ("R", "O"):
        for previous, current in zip(plot_rows, plot_rows[1:]):
            a, b = chart_point(previous, player), chart_point(current, player)
            if current.get("dummy_point"):
                draw_dashed_line(frame, a, b, COLORS[player], 2)
            else:
                cv2.line(frame, a, b, COLORS[player], 2, cv2.LINE_AA)
        for row in plot_rows:
            p = chart_point(row, player)
            cv2.circle(frame, p, 3, COLORS[player], -1, cv2.LINE_AA)
    for row in shown:
        if row.get("dummy_point"):
            p = chart_point(row, "R")
            cv2.circle(frame, p, 8, (60, 225, 255), 2, cv2.LINE_AA)
            text(frame, f"SCORE ONLY POINT {row['point_n']} - NO VIDEO SERVE", (p[0] - 110, plot_top - 8), 0.32, (60, 225, 255), 1)
    text(frame, f"{len(shown)} / {len(curve)} score rows", (chart_left, height - 26), 0.36, (175, 190, 195), 1)


def draw_heatmap_court(
    frame: np.ndarray, x: int, y: int, court_w: int, court_h: int,
    grid: list[list[int]], peak: int, player_name: str, color: tuple[int, int, int],
) -> None:
    bins_y = max(1, len(grid))
    bins_x = max(1, len(grid[0]) if grid else 1)
    for row_i, row in enumerate(grid):
        for col_i, count in enumerate(row):
            intensity = min(1.0, count / peak) if peak > 0 else 0.0
            left = x + round(col_i / bins_x * court_w)
            right = x + round((col_i + 1) / bins_x * court_w)
            top = y + round(row_i / bins_y * court_h)
            bottom = y + round((row_i + 1) / bins_y * court_h)
            cool = (43, 48, 53)
            hot = (64, 194, 255)
            shade = tuple(round(cool[i] * (1 - intensity) + hot[i] * intensity) for i in range(3))
            cv2.rectangle(frame, (left, top), (right, bottom), shade, -1)
    cv2.rectangle(frame, (x, y), (x + court_w, y + court_h), (230, 230, 225), 2)

    def px(metres: float) -> int:
        return x + round(metres / COURT_W * court_w)

    def py(metres: float) -> int:
        return y + round(metres / 6.7 * court_h)

    for line_x in (0.46, 3.05, 5.64):
        cv2.line(frame, (px(line_x), y), (px(line_x), y + court_h), (210, 215, 215), 1)
    for line_y in (0.76, 4.72, 6.7):
        cv2.line(frame, (x, py(line_y)), (x + court_w, py(line_y)), (210, 215, 215), 1)
    text(frame, player_name, (x, y - 15), 0.49, color, 1)


def draw_heatmap_slide(
    frame: np.ndarray, phase: dict[str, Any], phase_index: int, phase_count: int,
    names: dict[str, str], peaks: dict[str, int], commentary: list[str],
) -> None:
    height, width = frame.shape[:2]
    phase_name = str(phase.get("name", f"Phase {phase_index + 1}"))
    text(frame, f"COURT MOVEMENT  ·  {phase_index + 1}/{phase_count}  ·  {phase_name.upper()}", (42, 49), 0.48, (235, 240, 238), 1)
    court_h = min(390, round(height * 0.55))
    court_w = round(court_h * COURT_W / 6.7)
    court_top = 141
    left_x = max(48, round(width * 0.14) - court_w // 2)
    right_x = width - left_x - court_w
    player_data = phase.get("players", {})
    for player, x in (("R", left_x), ("O", right_x)):
        grid = player_data.get(player, {}).get("grid_counts", [])
        if not isinstance(grid, list) or not grid:
            grid = [[0 for _ in range(10)] for _ in range(11)]
        draw_heatmap_court(frame, x, court_top, court_w, court_h, grid, peaks.get(player, 0), names[player], COLORS[player])
        count = player_data.get(player, {}).get("samples", 0)
        text(frame, f"{count:,} tracked samples", (x, court_top + court_h + 24), 0.36, (180, 190, 196), 1)
    caption_y = court_top + court_h + 62
    for sentence in commentary[:2]:
        caption_y = draw_wrapped_lines(
            frame, sentence, (43, caption_y), width - 86, 0.38, (218, 225, 225), 2, 20,
        ) + 2


def draw_review_banner(frame: np.ndarray, example: dict[str, Any], index: int, total: int) -> None:
    height, width = frame.shape[:2]
    overlay_panel(frame, (0, height - 43, width, height), 0.78)
    serve_number = example.get("video_serve_number", example.get("point_n"))
    text(
        frame, f"IN-DEPTH REVIEW {index}/{total}  ·  SERVE {serve_number}  ·  {example['title']}",
        (22, height - 14), 0.48, (245, 245, 242), 1,
    )


def highlight_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    examples = report.get("example_lost_points", [])
    wanted = (
        ("representative short", "SHORT RALLY LOSS"),
        ("longest R loss", "LONG RALLY LOSS"),
        ("highest R movement", "HIGHEST TRACKED MOVEMENT"),
    )
    selected = []
    for prefix, title in wanted:
        match = next((row for row in examples if str(row.get("reason", "")).startswith(prefix)), None)
        if match:
            selected.append({**match, "title": title})
    return selected


def highlight_frame_time(
    example: dict[str, Any], track_frames: list[dict[str, Any]], track_times: list[float]
) -> float:
    if example["title"] != "HIGHEST TRACKED MOVEMENT":
        return float(example["rally_end_sec"])
    start, end = float(example["serve_t"]), float(example["rally_end_sec"])
    lo, hi = bisect.bisect_left(track_times, start), bisect.bisect_right(track_times, end)
    candidates = []
    for index in range(lo, hi):
        position = point2(track_frames[index].get("r"))
        if position is not None:
            local_position = (
                (COURT_W - position[0], COURT_L - position[1])
                if position[1] >= COURT_L / 2 else position
            )
            candidates.append((math.dist(local_position, (3.05, 3.4)), track_times[index]))
    return max(candidates)[1] if candidates else float(example["rally_end_sec"])


def read_coach_narrative(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    content = path.read_text(encoding="utf-8")
    sections: dict[str, Any] = {"serves": {}}
    buffers: dict[tuple[str, int | None], list[str]] = {}
    current: tuple[str, int | None] | None = None
    recognized_heading = False
    for line in content.splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            title = heading.group(1).strip().lower()
            serve_match = re.search(r"\bserve\s+(\d+)\b", title)
            if serve_match:
                current = ("serve", int(serve_match.group(1)))
            elif "opening" in title or "overview" in title:
                current = ("opening", None)
            elif "score" in title or "momentum" in title:
                current = ("score", None)
            elif "movement" in title or "heatmap" in title or "court" in title:
                current = ("movement", None)
            elif "practice" in title or "closing" in title or "priority" in title or "takeaway" in title:
                current = ("closing", None)
            else:
                current = None
            if current is not None:
                recognized_heading = True
                buffers.setdefault(current, [])
            continue
        if current is not None:
            buffers[current].append(line)

    def plain_text(lines: list[str]) -> str:
        output = []
        for line in lines:
            line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line.strip())
            line = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda match: match.group(1) or match.group(2), line)
            line = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", line)
            line = re.sub(r"`([^`]+)`", r"\1", line)
            if line:
                output.append(line)
        return " ".join(output).strip()

    for (section, serve_number), lines in buffers.items():
        value = plain_text(lines)
        if not value:
            continue
        if section == "serve" and serve_number is not None:
            sections["serves"][serve_number] = value
        else:
            sections[section] = value
    if not recognized_heading:
        sections["opening"] = plain_text(content.splitlines())
    return sections


def review_coach_note(event: dict[str, Any], coach_sections: dict[str, Any]) -> str:
    serve_number = event.get("video_serve_number", event.get("point_n"))
    narrative_reviews = coach_sections.get("serves", {})
    ai_note = narrative_reviews.get(serve_number)
    if isinstance(ai_note, str) and ai_note.strip():
        return ai_note.strip()
    return str(event.get("coach_note") or "Review the exchange and check the final recovery before drawing a conclusion.")


def narration_text(event: dict[str, Any], coach_sections: dict[str, Any]) -> str:
    serve_number = event.get("video_serve_number", event["point_n"])
    timestamp = float(event.get("serve_t") or 0)
    note = review_coach_note(event, coach_sections)
    return f"Serve {serve_number}, beginning at about {timestamp:.0f} seconds. {note}"


def create_voice(path: Path, message: str, voice: str, rate: int) -> None:
    say = shutil.which("say")
    if not say:
        raise RuntimeError("macOS 'say' was not found; use --no-voice or install a local TTS command")
    subprocess.run([say, "-v", voice, "-r", str(rate), "-o", str(path), message], check=True)


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def media_duration(path: Path) -> float:
    result = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=duration", "-of", "json", str(path)],
        text=True,
    )
    data = json.loads(result)
    durations = [data.get("format", {}).get("duration")]
    durations.extend(stream.get("duration") for stream in data.get("streams", []))
    for value in durations:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(seconds) and seconds > 0:
            return seconds
    return 0.0


def attach_audio(
    silent_video: Path,
    source_video: Path,
    voice_files: list[Path],
    voice_starts: list[float],
    source_segments: list[tuple[float, float, float]],
    output: Path,
    include_voice: bool,
) -> None:
    command = ["ffmpeg", "-hide_banner", "-y", "-i", str(silent_video), "-i", str(source_video)]
    if include_voice:
        for voice_file in voice_files:
            command += ["-i", str(voice_file)]
    source_has_audio = has_audio(source_video)
    video_duration = media_duration(silent_video)
    filters: list[str] = []
    valid_source_segments = [
        (timeline_start, source_start, source_end)
        for timeline_start, source_start, source_end in source_segments
        if source_end > source_start and timeline_start < video_duration
    ] if source_has_audio else []
    source_labels: list[str] = []
    source_inputs = [f"srcin{i}" for i in range(len(valid_source_segments))]
    if len(source_inputs) > 1:
        filters.append("[1:a]asplit=" + str(len(source_inputs)) + "".join(f"[{label}]" for label in source_inputs))
    elif source_inputs:
        source_inputs = ["1:a"]
    for i, (timeline_start, source_start, source_end) in enumerate(valid_source_segments):
        label = f"src{i}"
        delay_ms = max(0, round(timeline_start * 1000))
        filters.append(
            f"[{source_inputs[i]}]atrim=start={source_start:.3f}:end={source_end:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,volume=0.38,adelay=delays={delay_ms}:all=1,asetpts=N/SR/TB[{label}]"
        )
        source_labels.append(f"[{label}]")
    if source_labels:
        filters.append(
            "".join(source_labels)
            + f"amix=inputs={len(source_labels)}:duration=longest:dropout_transition=0:normalize=0,asetpts=N/SR/TB[srcmix]"
        )

    voice_labels: list[str] = []
    voice_intervals: list[tuple[float, float]] = []
    if include_voice and voice_files:
        for i, start in enumerate(voice_starts):
            source_index = i + 2
            label = f"voice{i}"
            delay_ms = max(0, round(start * 1000))
            filters.append(
                f"[{source_index}:a]aresample=48000,adelay=delays={delay_ms}:all=1,asetpts=N/SR/TB[{label}]"
            )
            voice_labels.append(f"[{label}]")
            voice_intervals.append((start, start + media_duration(voice_files[i])))
        filters.append(
            "".join(voice_labels)
            + f"amix=inputs={len(voice_labels)}:duration=longest:dropout_transition=0:normalize=0,asetpts=N/SR/TB[voicemix]"
        )

    if source_labels and voice_labels:
        volume_expression = "0.38"
        for start, end in reversed(voice_intervals):
            volume_expression = f"if(between(t,{start:.3f},{end:.3f}),0.07,{volume_expression})"
        filters.append(
            f"[srcmix]volume='{volume_expression}':eval=frame,apad=whole_dur={video_duration:.3f},asetpts=N/SR/TB[sourcepad]"
        )
        filters.append(f"[voicemix]apad=whole_dur={video_duration:.3f},asetpts=N/SR/TB[voicepad]")
        filters.append(
            f"[sourcepad][voicepad]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"asetpts=N/SR/TB,atrim=duration={video_duration:.3f}[aout]"
        )
    elif source_labels:
        filters.append(f"[srcmix]apad=whole_dur={video_duration:.3f},asetpts=N/SR/TB,atrim=duration={video_duration:.3f}[aout]")
    elif voice_labels:
        filters.append(f"[voicemix]apad=whole_dur={video_duration:.3f},asetpts=N/SR/TB,atrim=duration={video_duration:.3f}[aout]")
    else:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-i", str(silent_video), "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(output)],
            check=True,
        )
        return

    command += [
        "-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{video_duration:.3f}",
        "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)


def render(args: argparse.Namespace) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    video_encoder = select_h264_encoder(ffmpeg_path, args.video_encoder)
    print(f"Stage 11 video encoder: {video_encoder}", flush=True)
    if video_encoder == "h264_videotoolbox":
        print(f"VideoToolbox target bitrate: {args.videotoolbox_bitrate}", flush=True)
    if args.output_video.exists() and not args.overwrite:
        raise ValueError(f"output already exists: {args.output_video}; add --overwrite to replace it")
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    players = read_json(args.players_json)
    rallies_data = read_json(args.rallies_json)
    pose_data = read_json(args.pose_json)
    report = read_json(args.report_json)
    coach_narrative_path = (args.coach_narrative or args.report_json.parent / "AI-coach-narrative.md").expanduser()
    coach_sections = read_coach_narrative(coach_narrative_path)
    if coach_sections:
        print(f"AI coach narrative loaded: {coach_narrative_path}", flush=True)
    else:
        print("No AI coach narrative found; using Stage 10 coaching notes.", flush=True)
    report_names = report.get("player_names", {})
    player_name = (args.player_name or report_names.get("R") or "").strip()
    opponent_name = (args.opponent_name or report_names.get("O") or "").strip()
    if not player_name:
        player_name = input("Your name (Stage 3 selected R player): ").strip() or "Player R"
    if not opponent_name:
        opponent_name = input("Opponent's name (O player): ").strip() or "Player O"
    names = {"R": player_name, "O": opponent_name}
    track_frames = players.get("frames")
    points = rallies_data.get("points")
    if not isinstance(track_frames, list) or not track_frames or not isinstance(points, list):
        raise ValueError("players and rally JSON files need non-empty frames and points arrays")
    track_times = [float(row["t"]) for row in track_frames]
    track_fps = float(players.get("fps", 0))
    if track_fps <= 0 or any(b <= a for a, b in zip(track_times, track_times[1:])):
        raise ValueError("players JSON needs increasing timestamps and a positive fps")
    report_points = {int(row["point_n"]): row for row in report.get("video_points", [])}
    for rally in points:
        if int(rally["point_n"]) not in report_points:
            continue
        rally.update({key: report_points[int(rally["point_n"])].get(key) for key in (
            "outcome", "R_distance_m", "length_bucket", "score_before", "score_after", "video_serve_number"
        )})
    points = sorted((row for row in points if finite(row.get("serve_t")) and finite(row.get("rally_end_sec"))), key=lambda row: float(row["serve_t"]))
    examples = highlight_examples(report)
    if not examples:
        raise ValueError("report data has no timestamped lost-point examples; regenerate it with Stage 10")
    for example in examples:
        if not finite(example.get("serve_t")) or not finite(example.get("rally_end_sec")):
            raise ValueError(f"Stage 10 review at score point {example.get('point_n', '?')} needs serve and rally-end timestamps")
    score_analysis = report.get("score_analysis", {})
    score_curve = score_analysis.get("curve", [])
    heatmap_data = report.get("heatmap_timelapse", {})
    heatmap_phases = heatmap_data.get("phases", [])
    if not score_curve or not heatmap_phases:
        raise ValueError("report_data.json is missing score/heatmap video analysis; rerun the updated Stage 10 first")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError("input video has invalid frame properties")
    duration = frame_count / fps
    heatmap_commentary = heatmap_data.get("coaching_commentary", [])
    if not heatmap_commentary:
        heatmap_commentary = ["Compare opening, middle, and closing court coverage before changing movement choices."]
    score_commentary = score_analysis.get("coaching_commentary", [])
    if not score_commentary:
        score_commentary = ["Use the graph to locate score swings and review the nearby rallies."]
    ai_score_comments = [
        str(coach_sections.get(section, "")).strip()
        for section in ("opening", "score")
        if str(coach_sections.get(section, "")).strip()
    ]
    score_visual_commentary = ai_score_comments or list(map(str, score_commentary[:2]))
    ai_movement_comments = [
        str(coach_sections.get(section, "")).strip()
        for section in ("movement", "closing")
        if str(coach_sections.get(section, "")).strip()
    ]
    heatmap_visual_commentary = ai_movement_comments or list(map(str, heatmap_commentary[:2]))
    score_voice = " ".join(score_visual_commentary).strip()
    heatmap_voice = " ".join(heatmap_visual_commentary).strip()
    if not ai_movement_comments:
        heatmap_voice = "Court movement time-lapse. " + heatmap_voice
    review_voice_texts = [narration_text(example, coach_sections) for example in examples]
    voice_texts = [score_voice, heatmap_voice, *review_voice_texts]

    with tempfile.TemporaryDirectory(prefix="stage11_") as temporary:
        tmp = Path(temporary)
        silent_video = tmp / "overlays_silent.mp4"
        pose_model = args.pose_model.expanduser()
        if not pose_model.is_file() and pose_model == Path("models/pose_landmarker_heavy.task"):
            cached_pose_model = Path.home() / "Library/Caches/mediapipe/pose_landmarker_heavy.task"
            if cached_pose_model.is_file():
                pose_model = cached_pose_model
        pose_samples = []
        if not args.no_pose:
            pose_samples = precompute_pose(
                args.video, pose_model, track_frames, track_times,
                pose_data.get("frames", []), fps,
            )
        pose_times = [row["t"] for row in pose_samples]

        voice_files: list[Path] = []
        voice_enabled = not args.no_voice
        if voice_enabled:
            try:
                for i, message in enumerate(voice_texts):
                    path = tmp / f"coach_voice_{i:02d}.aiff"
                    create_voice(path, message, args.voice, args.voice_rate)
                    if media_duration(path) < 0.1:
                        raise RuntimeError(f"the local voice '{args.voice}' produced an empty audio file")
                    voice_files.append(path)
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                print(f"Local voice-over unavailable ({exc}); using on-screen coach captions and match audio.", flush=True)
                voice_files = []
                voice_enabled = False
        voice_lengths = [media_duration(path) for path in voice_files]
        intro_duration = max(9.0, voice_lengths[0] + 0.8) if voice_enabled else 9.0
        heatmap_duration = max(12.0, voice_lengths[1] + 0.8) if voice_enabled else 12.0
        freeze_duration = max(4.0, float(args.freeze_duration))
        if voice_enabled and len(voice_lengths) > 2:
            freeze_duration = max(freeze_duration, max(voice_lengths[2:]) + 0.8)
        voice_starts: list[float] = [0.0, intro_duration] if voice_enabled else []
        source_audio_segments: list[tuple[float, float, float]] = []
        written_frames = 0

        ffmpeg = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", f"{fps:.6f}", "-i", "pipe:0", "-an",
            *h264_encoder_args(
                video_encoder,
                bitrate=args.videotoolbox_bitrate,
                preset=args.preset,
                crf=args.crf,
            ),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(silent_video),
        ]
        encoder = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if encoder.stdin is None:
            raise RuntimeError("could not open ffmpeg video input")
        speeds = {player: compute_speeds(track_frames, track_times, player) for player in ("R", "O")}
        max_heat_counts = {player: 0 for player in ("R", "O")}
        for phase in heatmap_phases:
            for player in ("R", "O"):
                grid = phase.get("players", {}).get(player, {}).get("grid_counts", [])
                max_heat_counts[player] = max(max_heat_counts[player], max((max(row) for row in grid if row), default=0))

        def write_frame(image: np.ndarray) -> None:
            nonlocal written_frames
            encoder.stdin.write(image.tobytes())
            written_frames += 1

        def draw_annotated(
            image: np.ndarray, timestamp: float, histories: dict[str, deque],
            track_cursor: int, point_cursor: int,
        ) -> tuple[np.ndarray, int, int, dict[str, Any] | None]:
            tracked, track_cursor = frame_track(track_frames, track_times, timestamp, track_cursor)
            location = {player: point2(tracked.get(player.lower())) for player in ("R", "O")}
            speed_index = nearest_index(track_times, timestamp)
            speeds_now = {player: speeds[player][speed_index] for player in ("R", "O")}
            draw_tracking(image, tracked, histories, timestamp, names)
            this_point, point_cursor = current_point(points, timestamp, point_cursor)
            next_point = points[point_cursor + 1] if point_cursor + 1 < len(points) else None
            draw_hud(image, this_point, next_point, timestamp, names)
            draw_minimap(image, location, speeds_now)
            draw_pose(image, pose_at(pose_samples, pose_times, timestamp))
            return image, track_cursor, point_cursor, this_point

        def timeline_sec() -> float:
            return written_frames / fps

        try:
            for frame_index in range(max(1, round(intro_duration * fps))):
                image = analysis_slide(width, height, "Match analysis", "Score progression")
                visible = max(1, math.ceil((frame_index + 1) / max(1, round(intro_duration * fps)) * len(score_curve)))
                draw_score_intro(
                    image, score_curve, visible, names,
                    score_visual_commentary,
                )
                write_frame(image)

            heat_phases = list(heatmap_phases)
            phase_frames = max(1, round(heatmap_duration * fps / max(1, len(heat_phases))))
            for phase_index, phase in enumerate(heat_phases):
                for _ in range(phase_frames):
                    image = np.full((height, width, 3), (27, 25, 19), dtype=np.uint8)
                    cv2.rectangle(image, (0, 0), (12, height), (210, 190, 65), -1)
                    draw_heatmap_slide(
                        image, phase, phase_index, len(heat_phases), names,
                        max_heat_counts, heatmap_visual_commentary,
                    )
                    write_frame(image)

            for example_index, example in enumerate(examples, start=1):
                source_start = max(0.0, float(example["serve_t"]) - 0.4)
                source_end = min(duration, float(example["rally_end_sec"]) + 0.35)
                if source_end <= source_start:
                    raise ValueError(f"review serve {example.get('video_serve_number', example['point_n'])} has no playable video interval")
                start_frame = min(frame_count - 1, max(0, round(source_start * fps)))
                end_frame = min(frame_count - 1, max(start_frame, round(source_end * fps)))
                source_start_actual = start_frame / fps
                review_timeline_start = timeline_sec()
                review_capture = cv2.VideoCapture(str(args.video))
                if not review_capture.isOpened():
                    raise RuntimeError(f"could not reopen video for review clip: {args.video}")
                review_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                histories = {"R": deque(maxlen=75), "O": deque(maxlen=75)}
                track_cursor = max(0, bisect.bisect_right(track_times, source_start_actual) - 1)
                point_cursor = -1
                last_image: np.ndarray | None = None
                source_index = start_frame
                try:
                    while source_index <= end_frame:
                        ok, image = review_capture.read()
                        if not ok:
                            break
                        timestamp = source_index / fps
                        image, track_cursor, point_cursor, _ = draw_annotated(
                            image, timestamp, histories, track_cursor, point_cursor,
                        )
                        draw_review_banner(image, example, example_index, len(examples))
                        write_frame(image)
                        last_image = image.copy()
                        source_index += 1
                finally:
                    review_capture.release()
                if last_image is None:
                    raise RuntimeError(f"no frames could be read for review serve {example.get('video_serve_number', example['point_n'])}")
                source_end_actual = source_index / fps
                source_audio_segments.append((review_timeline_start, source_start_actual, source_end_actual))
                freeze_start = timeline_sec()
                if voice_enabled and example_index - 1 < len(review_voice_texts):
                    voice_starts.append(freeze_start)
                duration_text = example.get("rally_duration_sec")
                duration_label = f"estimated rally: {duration_text:.1f} s" if finite(duration_text) else "rally duration estimate unavailable"
                confidence = str(example.get("rally_end_confidence") or "unrated")
                coach_note = review_coach_note(example, coach_sections)
                card = draw_freeze_card(
                    last_image,
                    f"REVIEW  ·  SERVE {example.get('video_serve_number', example['point_n'])}  ·  {example['title']}",
                    coach_note,
                    f"Serve time {float(example['serve_t']):.1f}s  |  end estimate confidence: {confidence}  |  {duration_label}",
                )
                for _ in range(round(freeze_duration * fps)):
                    write_frame(card)

            full_match_start = timeline_sec()
            if not args.short_mode:
                source_audio_segments.append((full_match_start, 0.0, duration))
                video = cv2.VideoCapture(str(args.video))
                if not video.isOpened():
                    raise RuntimeError(f"could not reopen video: {args.video}")
                histories = {"R": deque(maxlen=75), "O": deque(maxlen=75)}
                track_cursor = 0
                point_cursor = -1
                frame_index = 0
                try:
                    while True:
                        ok, image = video.read()
                        if not ok:
                            break
                        timestamp = frame_index / fps
                        image, track_cursor, point_cursor, _ = draw_annotated(
                            image, timestamp, histories, track_cursor, point_cursor,
                        )
                        write_frame(image)
                        frame_index += 1
                        if frame_index % 3000 == 0:
                            print(f"Full match frames rendered: {frame_index}/{frame_count}", flush=True)
                finally:
                    video.release()
                if frame_index == 0:
                    raise RuntimeError("input video produced no frames")

            encoder.stdin.close()
            return_code = encoder.wait()
            if return_code != 0:
                details = encoder.stderr.read().decode("utf-8", "replace") if encoder.stderr else ""
                raise RuntimeError(f"ffmpeg video encoding failed ({return_code}): {details[-2000:]}")
        except (BrokenPipeError, OSError):
            encoder.kill()
            raise
        finally:
            if encoder.poll() is None:
                encoder.kill()

        attach_audio(
            silent_video, args.video, voice_files,
            voice_starts if voice_enabled else [], source_audio_segments,
            args.output_video, voice_enabled,
        )

    match_text = "analysis-only short mode" if args.short_mode else f"{frame_count} full-match frames"
    print(f"Stage 11 video complete: score intro, heatmap time-lapse, {len(examples)} replay/freeze reviews, {match_text}.")
    print(f"Voice-over: {'local narration' if voice_enabled else 'on-screen captions only'}.")
    print(f"Output: {args.output_video}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Stage 11 tracking overlays and narrated match highlights.")
    parser.add_argument("video", type=Path, help="Stage 0 analysis proxy video")
    parser.add_argument("players_json", type=Path, help="Stage 3 players.json")
    parser.add_argument("rallies_json", type=Path, help="Stage 6 rallies_review.json")
    parser.add_argument("pose_json", type=Path, help="Stage 8 pose_stats.json")
    parser.add_argument("report_json", type=Path, help="Stage 10 report_data.json")
    parser.add_argument("output_video", type=Path, help="annotated MP4 output")
    parser.add_argument("--player-name", help="name shown for the Stage 3 selected R player; prompted if omitted")
    parser.add_argument("--opponent-name", help="name shown for the O opponent; prompted if omitted")
    parser.add_argument("--coach-narrative", type=Path, help="optional Stage 10A Markdown narrative (defaults beside report_json)")
    parser.add_argument("--pose-model", type=Path, default=Path("models/pose_landmarker_heavy.task"), help="local MediaPipe pose model")
    parser.add_argument("--voice", default="Aman", help="local macOS say voice; default Aman (en_IN)")
    parser.add_argument("--voice-rate", type=int, default=185, help="local speech rate in words per minute")
    parser.add_argument("--freeze-duration", type=float, default=8.0, help="minimum seconds to hold each review frame")
    parser.add_argument(
        "--video-encoder",
        choices=("auto", "libx264", "h264_videotoolbox"),
        default="auto",
        help="auto uses VideoToolbox on macOS when available, otherwise libx264 (default: auto)",
    )
    parser.add_argument(
        "--videotoolbox-bitrate",
        default="4M",
        help="target H.264 bitrate for VideoToolbox encoding (default: 4M)",
    )
    parser.add_argument("--preset", default="veryfast", help="libx264 speed/quality preset")
    parser.add_argument("--crf", type=int, default=21, help="libx264 quality value; lower is higher quality")
    parser.add_argument("--short-mode", action="store_true", help="render score analysis, heatmaps, and review clips without the full match")
    parser.add_argument("--no-pose", action="store_true", help="skip MediaPipe skeleton overlay")
    parser.add_argument("--no-voice", action="store_true", help="skip local voice-over")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args()
    try:
        render(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
