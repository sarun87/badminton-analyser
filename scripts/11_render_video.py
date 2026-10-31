#!/usr/bin/env python3
"""Render player tracking, pose, court-map, and score overlays into an MP4."""

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
)


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
    panel_x, panel_y, panel_w, panel_h = 22, 22, 475, 150
    blend_panel(frame, panel_x, panel_y, panel_w, panel_h)
    game = point.get("game", 1)
    put_text(frame, f"GAME {game}   POINT {point.get('n', point_index + 1)}", (panel_x + 16, panel_y + 27), scale=0.62, thickness=2)
    put_text(frame, f"{you_name}: {user_score}    -    {opponent_name}: {opponent_score}", (panel_x + 16, panel_y + 61), scale=0.73, thickness=2)
    server = you_name if point.get("server") == "R" else opponent_name
    side = point.get("side")
    server_text = f"Server: {server}" + (f"  ({side} court)" if side else "")
    put_text(frame, server_text, (panel_x + 16, panel_y + 94), color=(215, 225, 240), scale=0.54)
    if after:
        winner = you_name if point.get("winner") == "R" else opponent_name
        phase = f"POINT COMPLETE - {winner} won"
    else:
        phase = "RALLY IN PLAY"
    put_text(frame, phase, (panel_x + 16, panel_y + 125), color=(125, 235, 155), scale=0.56, thickness=2)
    tenths = int(round(seconds * 10))
    minutes, remainder = divmod(tenths, 600)
    whole_seconds, decimal = divmod(remainder, 10)
    put_text(frame, f"{minutes:02d}:{whole_seconds:02d}.{decimal}", (panel_x + panel_w - 86, panel_y + 27), color=(205, 210, 220), scale=0.48)


def map_point(position: list[float], court_box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, width, height = court_box
    px = x + int(np.clip(float(position[0]) / COURT_WIDTH, 0, 1) * width)
    py = y + int(np.clip(float(position[1]) / COURT_LENGTH, 0, 1) * height)
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
        return court_y + int(metres / COURT_LENGTH * court_h)

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

    put_text(frame, f"{you_name}  /  {opponent_name}", (panel_x + 36, panel_y + panel_h - 12), color=(215, 225, 240), scale=0.45)


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
    radius_x = max(14, min(30, (right - left) // 5))
    cv2.ellipse(frame, foot, (radius_x, 8), 0, 0, 360, PLAYER_COLORS[key], 3, cv2.LINE_AA)
    cv2.circle(frame, foot, 4, (255, 255, 255), -1, cv2.LINE_AA)
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
    command = ffmpeg_command(ffmpeg, proxy, partial, width, height, fps, duration, audio)
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
            draw_scoreboard(frame, seconds, points, point_starts, you_name, opponent_name)
            draw_minimap(frame, positions, map_histories, seconds, you_name, opponent_name)
            if process.stdin is None:
                raise RuntimeError("Could not open FFmpeg video input pipe.")
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("FFmpeg stopped accepting video frames.") from exc
            if (frame_index + 1) % int(max(1, fps * 30)) == 0:
                print(f"Rendered {seconds / 60:.1f} minutes of video.")
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
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "overlay.mp4")
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}. Choose another --output path.")
        player_data = read_json(players_path)
        point_data = read_json(points_path)
        pose_data = read_json(pose_path) if pose_path.is_file() else {"samples": []}
        properties = video_properties(proxy)
        width, height = int(properties["width"]), int(properties["height"])
        fps = float(properties["fps"])
        frame_count = int(properties["frame_count"])
        duration = frame_count / fps
        points = point_data.get("points", [])
        if not points:
            raise ValueError("points.json contains no scored points. Run Stage 6 first.")
        if not pose_data.get("samples"):
            print("Pose samples were not found; rendering tracking and scoreboard overlays without the skeleton.")
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
            pose_data.get("samples", []),
            args.you_name,
            args.opponent_name,
        )
        print(f"Rendered overlay video to {output_path}.")
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as exc:
        print(f"Video rendering failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
