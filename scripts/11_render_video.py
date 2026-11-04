#!/usr/bin/env python3
"""Render player tracking, score and map overlays with optional pose, shuttle, and stats clips."""

from __future__ import annotations

import argparse
import bisect
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import (
    COURT_LENGTH,
    COURT_WIDTH,
    NET_Y,
    artifact_path,
    load_job,
    open_video,
    proxy_path,
    read_json,
    video_properties,
    write_json,
)
from visual_stats import build_match_stats


PLAYER_COLORS = {"r": (235, 125, 35), "o": (40, 175, 245)}  # BGR
SKELETON_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
)


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for path in (Path("/opt/homebrew/bin/ffmpeg"), Path("/usr/local/bin/ffmpeg")):
        if path.is_file():
            return str(path)
    raise RuntimeError("Could not find ffmpeg. Run scripts/init_macos.sh first.")


def interpolate_field(
    rows: list[dict[str, Any]],
    times: list[float],
    seconds: float,
    field: str,
    max_gap: float = 0.22,
) -> list[float] | None:
    if not rows:
        return None
    after = bisect.bisect_right(times, seconds)
    before_index, after_index = after - 1, after
    before = rows[before_index].get(field) if before_index >= 0 else None
    later = rows[after_index].get(field) if after_index < len(rows) else None
    if before is not None and later is not None:
        t0, t1 = times[before_index], times[after_index]
        if t1 > t0 and t1 - t0 <= 0.5:
            amount = min(1.0, max(0.0, (seconds - t0) / (t1 - t0)))
            return [float(a) * (1 - amount) + float(b) * amount for a, b in zip(before, later)]
    options = []
    if before is not None and before_index >= 0 and seconds - times[before_index] <= max_gap:
        options.append((abs(seconds - times[before_index]), before))
    if later is not None and after_index < len(rows) and times[after_index] - seconds <= max_gap:
        options.append((abs(times[after_index] - seconds), later))
    return list(min(options, key=lambda item: item[0])[1]) if options else None


def interpolate_skeleton(samples: list[dict[str, Any]], times: list[float], seconds: float) -> list[list[float]] | None:
    if not samples:
        return None
    after = bisect.bisect_right(times, seconds)
    before_index, after_index = after - 1, after
    before = samples[before_index].get("skeleton") if before_index >= 0 else None
    later = samples[after_index].get("skeleton") if after_index < len(samples) else None
    if before is not None and later is not None:
        t0, t1 = times[before_index], times[after_index]
        if t1 > t0 and t1 - t0 <= 0.5:
            amount = min(1.0, max(0.0, (seconds - t0) / (t1 - t0)))
            blended = []
            for first, second in zip(before, later):
                if float(first[2]) >= 0.25 and float(second[2]) >= 0.25:
                    blended.append(
                        [
                            float(first[0]) * (1 - amount) + float(second[0]) * amount,
                            float(first[1]) * (1 - amount) + float(second[1]) * amount,
                            min(float(first[2]), float(second[2])),
                        ]
                    )
                elif float(first[2]) >= 0.25:
                    blended.append([float(value) for value in first])
                else:
                    blended.append([float(value) for value in second])
            return blended
    options = []
    if before is not None and before_index >= 0 and seconds - times[before_index] <= 0.20:
        options.append((abs(seconds - times[before_index]), before))
    if later is not None and after_index < len(samples) and times[after_index] - seconds <= 0.20:
        options.append((abs(times[after_index] - seconds), later))
    selected = min(options, key=lambda item: item[0])[1] if options else None
    return [[float(value) for value in point] for point in selected] if selected is not None else None


def blend_panel(frame: np.ndarray, x: int, y: int, width: int, height: int, alpha: float = 0.78) -> None:
    panel = frame[y : y + height + 1, x : x + width + 1]
    layer = panel.copy()
    cv2.rectangle(layer, (0, 0), (width, height), (18, 22, 29), -1)
    cv2.addWeighted(layer, alpha, panel, 1 - alpha, 0, panel)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (205, 210, 220), 1)


def put_text(
    frame: np.ndarray,
    text: str,
    point: tuple[int, int],
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.58,
    thickness: int = 1,
) -> None:
    safe_text = text.encode("ascii", "replace").decode("ascii")
    cv2.putText(frame, safe_text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_scoreboard(
    frame: np.ndarray,
    seconds: float,
    points: list[dict[str, Any]],
    starts: list[float],
    you_name: str,
    opponent_name: str,
    show_details: bool,
) -> None:
    if not points or seconds < starts[0]:
        return
    point_index = bisect.bisect_right(starts, seconds) - 1
    point_index = min(max(point_index, 0), len(points) - 1)
    point = points[point_index]
    after = seconds >= float(point.get("r1", point.get("next_t", seconds + 1)))
    score = point.get("after" if after else "before", "0-0")
    score_parts = str(score).split("-", maxsplit=1)
    user_score = score_parts[0] if score_parts else "0"
    opponent_score = score_parts[1] if len(score_parts) == 2 else "0"
    panel_x, panel_y, panel_w, panel_h = 22, 22, 390, 128
    blend_panel(frame, panel_x, panel_y, panel_w, panel_h)
    put_text(frame, "MATCH SCORE", (panel_x + 16, panel_y + 26), color=(205, 210, 220), scale=0.54, thickness=2)
    table_x, table_y = panel_x + 12, panel_y + 38
    table_w, table_h = panel_w - 24, 78
    split_x = table_x + 265
    cv2.rectangle(frame, (table_x, table_y), (table_x + table_w, table_y + table_h), (105, 115, 130), 1)
    cv2.line(frame, (split_x, table_y), (split_x, table_y + table_h), (105, 115, 130), 1)
    cv2.line(frame, (table_x, table_y + table_h // 2), (table_x + table_w, table_y + table_h // 2), (105, 115, 130), 1)
    scores = (user_score, opponent_score)
    names = (you_name.upper(), opponent_name.upper())
    player_keys = ("R", "O")
    for row_index, (name, player_key, player_score) in enumerate(zip(names, player_keys, scores)):
        row_top = table_y + row_index * (table_h // 2)
        baseline = row_top + 26
        if point.get("server") == player_key:
            cv2.circle(frame, (table_x + 14, baseline - 6), 5, PLAYER_COLORS[player_key.lower()], -1, cv2.LINE_AA)
        put_text(frame, name, (table_x + 27, baseline), color=PLAYER_COLORS[player_key.lower()], scale=0.62, thickness=2)
        put_text(frame, str(player_score), (split_x + 31, baseline + 1), color=(250, 250, 250), scale=0.82, thickness=2)

    server = you_name.upper() if point.get("server") == "R" else opponent_name.upper()
    side = point.get("side")
    if after:
        winner = you_name.upper() if point.get("winner") == "R" else opponent_name.upper()
        phase = f"POINT COMPLETE - {winner} WON"
    else:
        phase = "RALLY IN PLAY"
    if show_details:
        details_y = panel_y + panel_h + 10
        blend_panel(frame, panel_x, details_y, panel_w, 108)
        put_text(frame, f"GAME {point.get('game', 1)}  |  POINT {point.get('n', point_index + 1)}", (panel_x + 14, details_y + 25), scale=0.55, thickness=2)
        server_text = f"SERVER: {server}" + (f"  ({str(side).upper()} COURT)" if side else "")
        put_text(frame, server_text, (panel_x + 14, details_y + 53), color=(215, 225, 240), scale=0.52)
        put_text(frame, phase, (panel_x + 14, details_y + 81), color=(125, 235, 155), scale=0.48, thickness=2)
        tenths = int(round(seconds * 10))
        minutes, remainder = divmod(tenths, 600)
        whole_seconds, decimal = divmod(remainder, 10)
        put_text(frame, f"VIDEO {minutes:02d}:{whole_seconds:02d}.{decimal}", (panel_x + 14, details_y + 101), color=(205, 210, 220), scale=0.42)


def map_point(position: list[float], court_box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, width, height = court_box
    px = x + int(np.clip(float(position[0]) / COURT_WIDTH, 0, 1) * width)
    # The map is viewed from the camera behind the near baseline, so near y=0 is at the bottom.
    py = y + int(np.clip((COURT_LENGTH - float(position[1])) / COURT_LENGTH, 0, 1) * height)
    return px, py


def draw_minimap(
    frame: np.ndarray,
    positions: dict[str, list[float] | None],
    histories: dict[str, deque[tuple[float, list[float]]]],
    seconds: float,
    you_name: str,
    opponent_name: str,
) -> None:
    panel_x, panel_y, panel_w, panel_h = frame.shape[1] - 270, 22, 246, 360
    blend_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.82)
    put_text(frame, "COURT MAP", (panel_x + 82, panel_y + 25), scale=0.56, thickness=2)
    court_w, court_h = 140, 292
    court_x, court_y = panel_x + 53, panel_y + 42
    cv2.rectangle(frame, (court_x, court_y), (court_x + court_w, court_y + court_h), (230, 230, 230), 1)

    def court_x_pixel(metres: float) -> int:
        return court_x + int(metres / COURT_WIDTH * court_w)

    def court_y_pixel(metres: float) -> int:
        return court_y + int((COURT_LENGTH - metres) / COURT_LENGTH * court_h)

    # Singles sidelines, net, short service lines, and center line.
    for x_m in (0.46, 3.05, 5.64):
        cv2.line(frame, (court_x_pixel(x_m), court_y), (court_x_pixel(x_m), court_y + court_h), (110, 125, 140), 1)
    for y_m in (4.72, NET_Y, 8.68):
        color = (210, 210, 210) if y_m == NET_Y else (110, 125, 140)
        cv2.line(frame, (court_x, court_y_pixel(y_m)), (court_x + court_w, court_y_pixel(y_m)), color, 1)

    for key, name in (("r", you_name), ("o", opponent_name)):
        history = histories[key]
        while history and seconds - history[0][0] > 2.0:
            history.popleft()
        position = positions.get(key)
        if position is None and history and seconds - history[-1][0] > 0.25:
            history.clear()
        elif position is not None and history and seconds - history[-1][0] > 0.25:
            history.clear()
        trail_points = [map_point(position, (court_x, court_y, court_w, court_h)) for _, position in history]
        for index in range(1, len(trail_points)):
            strength = index / max(1, len(trail_points) - 1)
            color = tuple(int(channel * strength) for channel in PLAYER_COLORS[key])
            cv2.line(frame, trail_points[index - 1], trail_points[index], color, 2, cv2.LINE_AA)
        if position is not None:
            point = map_point(position, (court_x, court_y, court_w, court_h))
            cv2.circle(frame, point, 7, PLAYER_COLORS[key], -1, cv2.LINE_AA)
            cv2.circle(frame, point, 9, (255, 255, 255), 1, cv2.LINE_AA)

    put_text(frame, f"{you_name.upper()}  /  {opponent_name.upper()}", (panel_x + 36, panel_y + panel_h - 12), color=(215, 225, 240), scale=0.45)


def rally_motion_rows(players: list[dict[str, Any]], points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = sorted(
        (
            float(point.get("r0", point.get("t", 0))),
            float(point.get("r1", point.get("next_t", point.get("t", 0)))),
        )
        for point in points
    )
    starts = [interval[0] for interval in intervals]
    selected = []
    for row in players:
        seconds = float(row["t"])
        index = bisect.bisect_right(starts, seconds) - 1
        if index >= 0 and seconds <= intervals[index][1] and (row.get("r") is not None or row.get("o") is not None):
            selected.append(row)
    return selected or players


def draw_court_lines(frame: np.ndarray, court_box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = court_box

    def px(metres: float) -> int:
        return x + int(metres / COURT_WIDTH * width)

    def py(metres: float) -> int:
        return y + int((COURT_LENGTH - metres) / COURT_LENGTH * height)

    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    for x_m in (0.46, 3.05, 5.64):
        cv2.line(frame, (px(x_m), y), (px(x_m), y + height), color, 1, cv2.LINE_AA)
    for y_m in (0.0, 4.72, NET_Y, 8.68, COURT_LENGTH):
        cv2.line(frame, (x, py(y_m)), (x + width, py(y_m)), color, 1 if y_m != NET_Y else 2, cv2.LINE_AA)


def heatmap_timelapse_frame(
    width: int,
    height: int,
    match_seconds: float,
    match_duration: float,
    you_name: str,
    opponent_name: str,
    players: list[dict[str, Any]],
    player_times: list[float],
    heat_rows: list[dict[str, Any]],
    heat_grid: np.ndarray,
    heat_cursor: int,
    histories: dict[str, deque[tuple[int, int]]],
) -> tuple[np.ndarray, int]:
    frame = np.full((height, width, 3), (15, 19, 27), dtype=np.uint8)
    put_text(frame, "FULL MATCH MOVEMENT TIMELAPSE", (38, 42), color=(248, 249, 250), scale=0.88, thickness=2)
    put_text(frame, "PLAYER POSITIONS AND ACCUMULATED RALLY HEAT", (40, 69), color=(170, 185, 200), scale=0.48)

    court_h = min(height - 110, 600)
    court_w = int(court_h * COURT_WIDTH / COURT_LENGTH)
    court_x = (width - court_w) // 2
    court_y = 82
    court_box = (court_x, court_y, court_w, court_h)
    grid_h, grid_w = heat_grid.shape

    while heat_cursor < len(heat_rows) and float(heat_rows[heat_cursor]["t"]) <= match_seconds:
        row = heat_rows[heat_cursor]
        for key in ("r", "o"):
            position = row.get(key)
            if position is None:
                continue
            grid_x = int(np.clip(float(position[0]) / COURT_WIDTH * (grid_w - 1), 0, grid_w - 1))
            grid_y = int(np.clip((COURT_LENGTH - float(position[1])) / COURT_LENGTH * (grid_h - 1), 0, grid_h - 1))
            cv2.circle(heat_grid, (grid_x, grid_y), 4, 1.0, -1, cv2.LINE_AA)
        heat_cursor += 1

    base = np.full((court_h, court_w, 3), (35, 49, 44), dtype=np.uint8)
    blurred = cv2.GaussianBlur(heat_grid, (0, 0), sigmaX=5, sigmaY=5)
    density = np.log1p(blurred)
    maximum = float(density.max())
    if maximum > 0:
        normalized = np.clip(density / maximum * 255.0, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colors = cv2.resize(colors, (court_w, court_h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(normalized, (court_w, court_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)[:, :, None] / 255.0
        alpha *= 0.78
        base = np.clip(base.astype(np.float32) * (1.0 - alpha) + colors.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    frame[court_y : court_y + court_h, court_x : court_x + court_w] = base
    draw_court_lines(frame, court_box, (205, 216, 221))

    for key, name in (("r", you_name), ("o", opponent_name)):
        position = interpolate_field(players, player_times, match_seconds, key)
        if position is None:
            continue
        point = map_point(position, court_box)
        histories[key].append(point)
        trail = list(histories[key])
        for index in range(1, len(trail)):
            strength = index / max(1, len(trail) - 1)
            color = tuple(int(channel * (0.30 + 0.70 * strength)) for channel in PLAYER_COLORS[key])
            cv2.line(frame, trail[index - 1], trail[index], color, 3, cv2.LINE_AA)
        cv2.circle(frame, point, 11, PLAYER_COLORS[key], -1, cv2.LINE_AA)
        cv2.circle(frame, point, 14, (250, 250, 250), 2, cv2.LINE_AA)
        put_text(frame, key.upper(), (point[0] + 12, point[1] - 10), color=(250, 250, 250), scale=0.55, thickness=2)

    put_text(frame, you_name.upper(), (40, height - 74), color=PLAYER_COLORS["r"], scale=0.65, thickness=2)
    put_text(frame, opponent_name.upper(), (40, height - 45), color=PLAYER_COLORS["o"], scale=0.65, thickness=2)
    cv2.circle(frame, (24, height - 78), 6, PLAYER_COLORS["r"], -1, cv2.LINE_AA)
    cv2.circle(frame, (24, height - 49), 6, PLAYER_COLORS["o"], -1, cv2.LINE_AA)

    elapsed = max(0, int(match_seconds))
    total = max(0, int(match_duration))
    put_text(frame, f"MATCH {elapsed // 60:02d}:{elapsed % 60:02d} / {total // 60:02d}:{total % 60:02d}", (width - 255, height - 45), color=(220, 225, 232), scale=0.48)
    bar_x, bar_y, bar_w = width - 258, height - 75, 210
    gradient = np.tile(np.linspace(0, 255, bar_w, dtype=np.uint8), (10, 1))
    frame[bar_y : bar_y + 10, bar_x : bar_x + bar_w] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    put_text(frame, "LOW USE", (bar_x, bar_y - 5), color=(210, 215, 225), scale=0.35)
    put_text(frame, "HIGH USE", (bar_x + bar_w - 62, bar_y - 5), color=(210, 215, 225), scale=0.35)
    return frame, heat_cursor


def draw_stats_card(
    width: int,
    height: int,
    stats: dict[str, Any],
    card_index: int,
    progress: float,
    you_name: str,
    fps_time: str,
) -> np.ndarray:
    frame = np.full((height, width, 3), (15, 19, 27), dtype=np.uint8)
    won_color, lost_color = PLAYER_COLORS["r"], PLAYER_COLORS["o"]
    put_text(frame, "MATCH STATS", (38, 42), color=(248, 249, 250), scale=0.9, thickness=2)
    put_text(frame, f"GRAPH {card_index + 1} OF 2", (width - 215, 42), color=(170, 185, 200), scale=0.48)
    animation = float(np.clip((progress - 0.08) / 0.55, 0.0, 1.0))
    animation = 1.0 - (1.0 - animation) ** 3

    chart_left, chart_top, chart_width, chart_height = 185, 208, 910, 330
    baseline = chart_top + chart_height
    cv2.line(frame, (chart_left, baseline), (chart_left + chart_width, baseline), (125, 140, 155), 2, cv2.LINE_AA)
    if card_index == 0:
        put_text(frame, "POINT OUTCOMES BY RALLY LENGTH", (38, 102), color=(248, 249, 250), scale=0.72, thickness=2)
        put_text(frame, "Bar height is the number of points. Orange = won; blue = lost.", (40, 139), color=(190, 200, 210), scale=0.52)
        buckets = stats.get("rally_buckets", [])
        max_value = max([int(item.get("won", 0)) for item in buckets] + [int(item.get("lost", 0)) for item in buckets] + [1])
        group_width = chart_width / max(1, len(buckets))
        bar_width = 74
        max_bar_height = chart_height - 18
        for index, bucket in enumerate(buckets):
            center = int(chart_left + group_width * (index + 0.5))
            for offset, key, color in ((-bar_width - 5, "won", won_color), (5, "lost", lost_color)):
                value = int(bucket.get(key, 0))
                bar_h = int(max_bar_height * value / max_value * animation)
                x1 = center + offset
                cv2.rectangle(frame, (x1, baseline - bar_h), (x1 + bar_width, baseline), color, -1)
                cv2.rectangle(frame, (x1, baseline - bar_h), (x1 + bar_width, baseline), (230, 235, 240), 1)
                put_text(frame, str(value), (x1 + bar_width // 2 - 8, baseline - bar_h - 12), color=(248, 249, 250), scale=0.48, thickness=2)
            label = str(bucket.get("label", ""))
            put_text(frame, label.upper(), (center - 82, baseline + 30), color=(220, 225, 232), scale=0.43)
        cv2.rectangle(frame, (chart_left + 240, 174), (chart_left + 258, 188), won_color, -1)
        put_text(frame, f"{you_name.upper()} WON", (chart_left + 265, 187), color=(220, 225, 232), scale=0.42)
        cv2.rectangle(frame, (chart_left + 440, 174), (chart_left + 458, 188), lost_color, -1)
        put_text(frame, "LOST", (chart_left + 465, 187), color=(220, 225, 232), scale=0.42)
        explanation = "Each group compares point outcomes by duration. Use the counts as a pattern, especially when a group has several points."
    else:
        put_text(frame, "YOUR AVERAGE MOVEMENT PER RALLY", (38, 102), color=(248, 249, 250), scale=0.72, thickness=2)
        put_text(frame, "Tracked distance on points won compared with points lost.", (40, 139), color=(190, 200, 210), scale=0.52)
        movement = stats.get("movement", {})
        values = [movement.get("distance_won_m"), movement.get("distance_lost_m")]
        finite_values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
        maximum = max(finite_values + [1.0])
        max_bar_height = chart_height - 25
        for index, (label, value, color) in enumerate((
            ("POINTS WON", values[0], won_color),
            ("POINTS LOST", values[1], lost_color),
        )):
            center = chart_left + int(chart_width * (0.28 + 0.44 * index))
            numeric = float(value) if value is not None and np.isfinite(float(value)) else 0.0
            bar_h = int(max_bar_height * numeric / maximum * animation)
            x1 = center - 70
            cv2.rectangle(frame, (x1, baseline - bar_h), (x1 + 140, baseline), color, -1)
            cv2.rectangle(frame, (x1, baseline - bar_h), (x1 + 140, baseline), (230, 235, 240), 1)
            value_label = f"{numeric:.1f} m" if value is not None else "N/A"
            put_text(frame, value_label, (center - 38, baseline - bar_h - 14), color=(248, 249, 250), scale=0.55, thickness=2)
            put_text(frame, label, (center - 62, baseline + 30), color=(220, 225, 232), scale=0.48)
        speed_won = movement.get("speed_won_mps")
        speed_lost = movement.get("speed_lost_mps")
        if speed_won is not None and speed_lost is not None:
            put_text(frame, f"AVERAGE SPEED: WON {float(speed_won):.2f} m/s  |  LOST {float(speed_lost):.2f} m/s", (chart_left + 135, baseline + 72), color=(190, 200, 210), scale=0.48)
        explanation = "Distance reflects tracked movement during each rally. A higher value can reflect rally length or difficulty; it is not automatically a problem."

    put_text(frame, explanation, (38, height - 72), color=(220, 225, 232), scale=0.49)
    put_text(frame, "DESCRIPTIVE MATCH STATISTICS - REVIEW THE FOOTAGE FOR CONTEXT", (38, height - 38), color=(145, 160, 175), scale=0.40)
    put_text(frame, fps_time, (width - 180, height - 38), color=(210, 218, 226), scale=0.42)
    return frame


def draw_skeleton(frame: np.ndarray, skeleton: list[list[float]] | None) -> None:
    if skeleton is None or len(skeleton) < 17:
        return
    height, width = frame.shape[:2]
    pixels = []
    for x, y, confidence in skeleton:
        if confidence < 0.25 or not (0 <= x <= 1 and 0 <= y <= 1):
            pixels.append(None)
        else:
            pixels.append((int(x * width), int(y * height)))
    for start, end in SKELETON_EDGES:
        if pixels[start] is not None and pixels[end] is not None:
            cv2.line(frame, pixels[start], pixels[end], (70, 245, 115), 3, cv2.LINE_AA)
    for point in pixels:
        if point is not None:
            cv2.circle(frame, point, 4, (40, 255, 255), -1, cv2.LINE_AA)


def draw_shuttle_tracks(
    frame: np.ndarray,
    points: list[dict[str, Any]],
    point_times: list[float],
    next_index: int,
    histories: dict[int, deque[tuple[float, tuple[int, int]]]],
    seconds: float,
) -> int:
    while next_index < len(points) and point_times[next_index] <= seconds:
        item = points[next_index]
        confidence = float(item.get("confidence", 0.0))
        if confidence >= 0.45:
            track_id = int(item.get("track_id", -1))
            history = histories.setdefault(track_id, deque())
            history.append((point_times[next_index], (int(round(float(item["x"]))), int(round(float(item["y"]))))))
        next_index += 1
    for track_id, history in list(histories.items()):
        while history and seconds - history[0][0] > 0.3:
            history.popleft()
        if not history or seconds - history[-1][0] > 0.15:
            histories.pop(track_id, None)
            continue
        trail = [point for _, point in history]
        for index in range(1, len(trail)):
            cv2.line(frame, trail[index - 1], trail[index], (0, 210, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, trail[-1], 7, (0, 210, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, trail[-1], 2, (255, 255, 255), -1, cv2.LINE_AA)
    return next_index


def draw_player_overlay(
    frame: np.ndarray,
    key: str,
    box: list[float] | None,
    position: list[float] | None,
    name: str,
    history: deque[tuple[float, tuple[int, int]]],
    seconds: float,
) -> None:
    while history and seconds - history[0][0] > 2.0:
        history.popleft()
    if history and seconds - history[-1][0] > 0.25:
        history.clear()
    if box is None or position is None:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    left = int(np.clip(x1 * width, 0, width - 1))
    top = int(np.clip(y1 * height, 0, height - 1))
    right = int(np.clip(x2 * width, 1, width))
    bottom = int(np.clip(y2 * height, 1, height))
    foot = ((left + right) // 2, bottom)
    history.append((seconds, foot))
    trail = [point for _, point in history]
    for index in range(1, len(trail)):
        strength = index / max(1, len(trail) - 1)
        color = tuple(int(channel * strength) for channel in PLAYER_COLORS[key])
        cv2.line(frame, trail[index - 1], trail[index], color, 3, cv2.LINE_AA)
    radius_x = max(18, min(38, (right - left) // 4))
    cv2.ellipse(frame, foot, (radius_x, 12), 0, 0, 360, PLAYER_COLORS[key], 4, cv2.LINE_AA)
    cv2.circle(frame, foot, 6, (255, 255, 255), -1, cv2.LINE_AA)
    put_text(frame, name, (left, max(20, top - 7)), color=PLAYER_COLORS[key], scale=0.55, thickness=2)


def ffmpeg_command(ffmpeg: str, proxy: Path, partial: Path, width: int, height: int, fps: float, duration: float, audio: bool) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio:
        command.extend(["-i", str(proxy)])
    command.extend(["-map", "0:v:0"])
    if audio:
        command.extend(["-map", "1:a?"])
    command.extend(
        [
            "-frames:v",
            str(int(round(duration * fps))),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-color_range",
            "tv",
        ]
    )
    if audio:
        command.extend(["-c:a", "copy"])
    command.extend(["-map_metadata", "-1", "-movflags", "+faststart", str(partial)])
    return command


def render(
    proxy: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    duration: float,
    audio: bool,
    players: list[dict[str, Any]],
    points: list[dict[str, Any]],
    pose_samples: list[dict[str, Any]],
    shuttle_tracklets: list[dict[str, Any]],
    show_details: bool,
    heatmap_seconds: float,
    stats_data: dict[str, Any] | None,
    stats_seconds: float,
    you_name: str,
    opponent_name: str,
) -> None:
    ffmpeg = find_ffmpeg()
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    if partial.exists():
        raise RuntimeError(f"Temporary output already exists: {partial}. Remove it or choose another output path.")
    player_times = [float(row["t"]) for row in players]
    pose_times = [float(row["t"]) for row in pose_samples]
    point_starts = [float(point["t"]) for point in points]
    player_histories: dict[str, deque[tuple[float, tuple[int, int]]]] = {
        "r": deque(),
        "o": deque(),
    }
    map_histories: dict[str, deque[tuple[float, list[float]]]] = {"r": deque(), "o": deque()}
    shuttle_points = [
        {**sample, "track_id": track.get("id", -1)}
        for track in shuttle_tracklets
        for sample in track.get("points", [])
    ]
    shuttle_points.sort(key=lambda sample: float(sample["t"]))
    shuttle_times = [float(sample["t"]) for sample in shuttle_points]
    shuttle_histories: dict[int, deque[tuple[float, tuple[int, int]]]] = {}
    shuttle_next = 0
    heatmap_frame_count = max(0, int(round(heatmap_seconds * fps)))
    stats_frame_count = max(0, int(round(stats_seconds * fps))) if stats_data is not None else 0
    output_duration = duration + ((heatmap_frame_count + stats_frame_count) / fps)
    command = ffmpeg_command(ffmpeg, proxy, partial, width, height, fps, output_duration, audio)
    capture = open_video(proxy)
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except BaseException:
        capture.release()
        raise
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended after {frame_index} of {frame_count} expected frames.")
            seconds = frame_index / fps
            positions = {}
            for key in ("r", "o"):
                position = interpolate_field(players, player_times, seconds, key)
                box = interpolate_field(players, player_times, seconds, "rb" if key == "r" else "ob")
                positions[key] = position
                if position is not None:
                    map_histories[key].append((seconds, position))
                draw_player_overlay(
                    frame,
                    key,
                    box,
                    position,
                    you_name if key == "r" else opponent_name,
                    player_histories[key],
                    seconds,
                )
            skeleton = interpolate_skeleton(pose_samples, pose_times, seconds) if pose_samples else None
            draw_skeleton(frame, skeleton)
            shuttle_next = draw_shuttle_tracks(frame, shuttle_points, shuttle_times, shuttle_next, shuttle_histories, seconds)
            draw_scoreboard(frame, seconds, points, point_starts, you_name, opponent_name, show_details)
            draw_minimap(frame, positions, map_histories, seconds, you_name, opponent_name)
            if process.stdin is None:
                raise RuntimeError("Could not open FFmpeg video input pipe.")
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("FFmpeg stopped accepting video frames.") from exc
            if (frame_index + 1) % int(max(1, fps * 30)) == 0:
                print(f"Rendered {seconds / 60:.1f} minutes of video.")
        if heatmap_frame_count:
            print(f"Rendering {heatmap_frame_count / fps:.0f}-second full-match movement timelapse...")
            heat_rows = rally_motion_rows(players, points)
            heat_grid = np.zeros((420, 190), dtype=np.float32)
            heat_histories = {
                "r": deque(maxlen=max(12, int(fps * 1.5))),
                "o": deque(maxlen=max(12, int(fps * 1.5))),
            }
            heat_cursor = 0
            for clip_index in range(heatmap_frame_count):
                match_seconds = duration * clip_index / max(1, heatmap_frame_count - 1)
                heat_frame, heat_cursor = heatmap_timelapse_frame(
                    width,
                    height,
                    match_seconds,
                    duration,
                    you_name,
                    opponent_name,
                    players,
                    player_times,
                    heat_rows,
                    heat_grid,
                    heat_cursor,
                    heat_histories,
                )
                if process.stdin is None:
                    raise RuntimeError("Could not open FFmpeg video input pipe.")
                try:
                    process.stdin.write(heat_frame.tobytes())
                except BrokenPipeError as exc:
                    raise RuntimeError("FFmpeg stopped accepting heatmap frames.") from exc
        if stats_frame_count and stats_data is not None:
            print(f"Rendering {stats_frame_count / fps:.0f}-second animated statistics outro...")
            card_frame_count = max(1, stats_frame_count // 2)
            for clip_index in range(stats_frame_count):
                card_index = min(1, clip_index // card_frame_count)
                card_progress = (clip_index % card_frame_count) / card_frame_count
                stats_frame = draw_stats_card(
                    width,
                    height,
                    stats_data,
                    card_index,
                    card_progress,
                    you_name,
                    f"GRAPH {card_index + 1}  {int(card_progress * stats_seconds / 2):02d}s",
                )
                if process.stdin is None:
                    raise RuntimeError("Could not open FFmpeg video input pipe.")
                try:
                    process.stdin.write(stats_frame.tobytes())
                except BrokenPipeError as exc:
                    raise RuntimeError("FFmpeg stopped accepting statistics frames.") from exc
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg render failed: {stderr.strip() or f'exit code {return_code}'}")
        partial.replace(output)
    except BaseException:
        if process.poll() is None:
            process.kill()
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        process.wait()
        partial.unlink(missing_ok=True)
        raise
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--points", help="Reviewed points JSON; defaults to points.json beside job.json.")
    parser.add_argument("--pose", help="Pose JSON; defaults to pose_stats.json beside job.json when present.")
    parser.add_argument("--show-skeleton", action="store_true", help="Draw the Stage 8 skeleton samples over the match video.")
    parser.add_argument("--show-details", action="store_true", help="Add a detail box under the scorecard with game, point, server, status, and time.")
    parser.add_argument("--show-shuttle", action="store_true", help="Draw experimental shuttle tracks when shuttle_tracks.json exists.")
    parser.add_argument("--no-heatmap-clip", action="store_true", help="Do not append the movement heatmap timelapse.")
    parser.add_argument("--heatmap-seconds", type=float, default=30.0, help="Duration of the appended movement timelapse (default: 30 seconds).")
    parser.add_argument("--show-stats", action="store_true", help="Append two animated match-stat graphs with explanations.")
    parser.add_argument("--stats-seconds", type=float, default=30.0, help="Duration of the optional statistics outro (default: 30 seconds).")
    parser.add_argument("--motion", help="Movement JSON; required with --show-stats, defaults to motion.json beside job.json.")
    parser.add_argument("--shuttle", help="Optional shuttle_tracks.json; used with --show-shuttle.")
    parser.add_argument("--output", help="Output video path; defaults to overlay.mp4 beside job.json.")
    parser.add_argument("--you-name", default="YOU", help="Scoreboard label for the appearance-matched player.")
    parser.add_argument("--opponent-name", default="OPPONENT", help="Scoreboard label for the opponent.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        proxy = proxy_path(job)
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        points_path = Path(args.points).expanduser().resolve() if args.points else artifact_path(job_path, "points.json")
        pose_path = Path(args.pose).expanduser().resolve() if args.pose else artifact_path(job_path, "pose_stats.json")
        shuttle_path = Path(args.shuttle).expanduser().resolve() if args.shuttle else artifact_path(job_path, "shuttle_tracks.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "overlay.mp4")
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}. Choose another --output path.")
        if not np.isfinite(args.heatmap_seconds) or args.heatmap_seconds < 0 or not np.isfinite(args.stats_seconds) or args.stats_seconds < 0:
            raise ValueError("--heatmap-seconds and --stats-seconds must be non-negative numbers.")
        player_data = read_json(players_path)
        point_data = read_json(points_path)
        pose_data = read_json(pose_path) if pose_path.is_file() else {"samples": []}
        shuttle_data = read_json(shuttle_path) if shuttle_path.is_file() else {"tracklets": []}
        properties = video_properties(proxy)
        width, height = int(properties["width"]), int(properties["height"])
        fps = float(properties["fps"])
        frame_count = int(properties["frame_count"])
        duration = frame_count / fps
        points = point_data.get("points", [])
        if not points:
            raise ValueError("points.json contains no scored points. Run Stage 6 first.")
        stats_data = None
        if args.show_stats:
            motion_path = Path(args.motion).expanduser().resolve() if args.motion else artifact_path(job_path, "motion.json")
            stats_data = build_match_stats(points, read_json(motion_path))
            if args.stats_seconds <= 0:
                raise ValueError("--stats-seconds must be greater than zero when --show-stats is enabled.")
        if args.show_skeleton and not pose_data.get("samples"):
            print("Pose samples were not found; rendering without the requested skeleton.")
        if args.show_shuttle and not shuttle_data.get("tracklets"):
            print("Shuttle tracklets were not found; rendering without shuttle marks.")
        display_you = args.you_name.strip().upper() or "YOU"
        display_opponent = args.opponent_name.strip().upper() or "OPPONENT"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render(
            proxy,
            output_path,
            width,
            height,
            fps,
            frame_count,
            duration,
            bool(job.get("input_video", {}).get("ffprobe", {}).get("has_audio")),
            player_data.get("frames", []),
            points,
            pose_data.get("samples", []) if args.show_skeleton else [],
            shuttle_data.get("tracklets", []) if args.show_shuttle else [],
            args.show_details,
            0.0 if args.no_heatmap_clip else args.heatmap_seconds,
            stats_data,
            args.stats_seconds,
            display_you,
            display_opponent,
        )
        heatmap_seconds = 0.0 if args.no_heatmap_clip else args.heatmap_seconds
        stats_seconds = args.stats_seconds if stats_data is not None else 0.0
        write_json(
            artifact_path(job_path, "render_manifest.json"),
            {
                "schema_version": 1,
                "video_path": str(output_path),
                "match_seconds": duration,
                "heatmap_seconds": heatmap_seconds,
                "stats_seconds": stats_seconds,
                "stats_enabled": stats_data is not None,
                "rendered_seconds": duration + heatmap_seconds + stats_seconds,
            },
        )
        print(f"Rendered overlay video to {output_path}.")
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as exc:
        print(f"Video rendering failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
