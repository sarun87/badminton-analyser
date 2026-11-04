#!/usr/bin/env python3
"""Estimate the Stage 3 selected player's posture from MediaPipe Pose Landmarker crops."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_POINT = (3.05, 3.4)
READY_RADIUS_M = 1.6
READY_SPEED_M_S = 0.8
BASELINE_SPEED_M_S = 0.5
FRONT_COURT_START_M = 4.72
REAR_COURT_END_M = 1.6
CROP_SCALE = 1.35
POSE_VISIBILITY = 0.35
POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point2(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(finite_number(item) for item in value)
    ):
        return None
    return float(value[0]), float(value[1])


def load_inputs(
    players_path: Path, review_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    players_data = json.loads(players_path.read_text(encoding="utf-8"))
    frames = players_data.get("frames")
    fps = float(players_data.get("fps", 0))
    if not isinstance(frames, list) or not frames or fps <= 0:
        raise ValueError("players JSON needs frames and a positive fps")
    if any(float(b["t"]) <= float(a["t"]) for a, b in zip(frames, frames[1:])):
        raise ValueError("players frame timestamps must increase")

    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    points = review_data.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("rally review JSON must contain a non-empty points array")
    points = sorted(points, key=lambda item: int(item["point_n"]))
    missing = [item["point_n"] for item in points if item.get("serve_t") is None or item.get("rally_end_sec") is None]
    if missing:
        raise ValueError(f"rally review is missing estimated or reviewed times for points {missing}")
    for point in points:
        start = float(point["serve_t"]) + 0.3
        end = float(point["rally_end_sec"])
        if end < start:
            raise ValueError(f"s{point['point_n']} has no usable in-play interval")
    return frames, points, fps


def smooth_positions(
    frames: list[dict[str, Any]], fps: float
) -> tuple[list[float], list[tuple[float, float] | None], list[float | None]]:
    times = [float(frame["t"]) for frame in frames]
    raw = [point2(frame.get("r")) for frame in frames]
    raw = [
        (6.1 - point[0], 13.4 - point[1]) if point is not None and point[1] >= 6.7 else point
        for point in raw
    ]
    window = max(1, round(0.7 * fps))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    smooth: list[tuple[float, float] | None] = []
    for index in range(len(raw)):
        values = [value for value in raw[max(0, index-radius):min(len(raw), index+radius+1)] if value is not None]
        if len(values) < max(1, math.ceil(window / 2)):
            smooth.append(None)
        else:
            median = np.median(np.asarray(values, dtype=np.float64), axis=0)
            smooth.append((float(median[0]), float(median[1])))

    speeds: list[float | None] = [None] * len(times)
    for index in range(1, len(times)):
        before, after = smooth[index - 1], smooth[index]
        elapsed = times[index] - times[index - 1]
        if before is None or after is None or elapsed <= 0 or elapsed > 0.35:
            continue
        speed = math.dist(before, after) / elapsed
        if speed <= 8.0:
            speeds[index] = speed
    return times, smooth, speeds


def frame_context(
    timestamp: float,
    position: tuple[float, float] | None,
    speed: float | None,
    points: list[dict[str, Any]],
    point_cursor: int,
) -> tuple[dict[str, Any] | None, int, str, bool]:
    while point_cursor < len(points) and timestamp > float(points[point_cursor]["rally_end_sec"]):
        point_cursor += 1
    point = None
    if point_cursor < len(points):
        candidate = points[point_cursor]
        if float(candidate["serve_t"]) + 0.3 <= timestamp <= float(candidate["rally_end_sec"]):
            point = candidate

    if point is None:
        baseline = (
            position is not None
            and speed is not None
            and speed <= BASELINE_SPEED_M_S
            and math.dist(position, BASE_POINT) <= READY_RADIUS_M
        )
        return None, point_cursor, "BETWEEN_POINTS", baseline

    if position is not None and speed is not None and speed <= READY_SPEED_M_S and math.dist(position, BASE_POINT) <= READY_RADIUS_M:
        context = "READY"
    elif position is not None and position[1] >= FRONT_COURT_START_M:
        context = "FRONT"
    elif position is not None and position[1] <= REAR_COURT_END_M:
        context = "REAR"
    else:
        context = "MID"
    return point, point_cursor, context, False


def player_crop(
    frame: np.ndarray, normalized_box: Any, scale: float = CROP_SCALE
) -> tuple[np.ndarray, int, int, int] | None:
    if not isinstance(normalized_box, list) or len(normalized_box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in normalized_box[:4]]
    height, width = frame.shape[:2]
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    box_width = (x2 - x1) * width
    box_height = (y2 - y1) * height
    side = max(32, int(math.ceil(max(box_width, box_height) * scale)))
    center_x = (x1 + x2) * 0.5 * width
    center_y = (y1 + y2) * 0.5 * height
    origin_x = int(round(center_x - side / 2))
    origin_y = int(round(center_y - side / 2))

    crop = np.zeros((side, side, 3), dtype=np.uint8)
    source_x0, source_y0 = max(0, origin_x), max(0, origin_y)
    source_x1, source_y1 = min(width, origin_x + side), min(height, origin_y + side)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return None
    target_x0, target_y0 = source_x0 - origin_x, source_y0 - origin_y
    crop[
        target_y0:target_y0 + source_y1 - source_y0,
        target_x0:target_x0 + source_x1 - source_x0,
    ] = frame[source_y0:source_y1, source_x0:source_x1]
    return crop, origin_x, origin_y, side


def landmark_visibility(landmark: Any) -> float:
    value = getattr(landmark, "visibility", 1.0)
    return float(value) if finite_number(value) else 1.0


def posture_metrics(landmarks: list[Any], side: int) -> dict[str, Any]:
    if len(landmarks) < 29:
        return {"pose_quality": None, "hip_ankle_to_torso_ratio": None, "stance_width_ratio": None,
                "left_wrist_above_shoulder": None, "right_wrist_above_shoulder": None,
                "landmarks": []}

    points = [(float(item.x) * side, float(item.y) * side) for item in landmarks]
    required = (11, 12, 23, 24, 27, 28)
    visibility = {index: landmark_visibility(landmarks[index]) for index in range(len(landmarks))}
    body_ok = all(visibility[index] >= POSE_VISIBILITY for index in required)
    quality = float(np.mean([visibility[index] for index in required]))

    shoulder_center = (
        (points[11][0] + points[12][0]) / 2,
        (points[11][1] + points[12][1]) / 2,
    )
    hip_center = (
        (points[23][0] + points[24][0]) / 2,
        (points[23][1] + points[24][1]) / 2,
    )
    ankle_center = (
        (points[27][0] + points[28][0]) / 2,
        (points[27][1] + points[28][1]) / 2,
    )
    torso_length = math.dist(shoulder_center, hip_center)
    shoulder_width = math.dist(points[11], points[12])
    ankle_width = math.dist(points[27], points[28])
    hip_ankle_ratio = (
        abs(ankle_center[1] - hip_center[1]) / torso_length
        if body_ok and torso_length > 1.0 else None
    )
    stance_width = (
        ankle_width / shoulder_width
        if body_ok and shoulder_width > 1.0 else None
    )
    left_up = (
        points[15][1] < points[11][1]
        if visibility[15] >= POSE_VISIBILITY and visibility[11] >= POSE_VISIBILITY else None
    )
    right_up = (
        points[16][1] < points[12][1]
        if visibility[16] >= POSE_VISIBILITY and visibility[12] >= POSE_VISIBILITY else None
    )
    return {
        "pose_quality": round(quality, 3),
        "hip_ankle_to_torso_ratio": None if hip_ankle_ratio is None else round(hip_ankle_ratio, 4),
        "stance_width_ratio": None if stance_width is None else round(stance_width, 4),
        "left_wrist_above_shoulder": left_up,
        "right_wrist_above_shoulder": right_up,
        "landmarks": points,
        "visibility": visibility,
    }


def infer_racket_side(records: list[dict[str, Any]]) -> dict[str, Any]:
    rates = {}
    for side in ("left", "right"):
        values = [
            record[f"{side}_wrist_above_shoulder"]
            for record in records
            if record["context"] != "BETWEEN_POINTS"
            and record.get(f"{side}_wrist_above_shoulder") is not None
        ]
        rates[side] = sum(values) / len(values) if values else None
    if rates["left"] is None or rates["right"] is None or abs(rates["left"] - rates["right"]) < 0.05:
        side = "unknown"
    else:
        side = max(rates, key=lambda name: rates[name])
    return {"side": side, "wrist_above_shoulder_rate": rates, "method": "higher wrist-above-shoulder rate during rallies"}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def median_for(name: str) -> float | None:
        values = [record[name] for record in records if finite_number(record.get(name))]
        return None if not values else round(float(np.median(values)), 2)

    arm_values = [record["racket_arm_up"] for record in records if record.get("racket_arm_up") is not None]
    point_ns = {record["point_n"] for record in records if record.get("point_n") is not None}
    return {
        "pose_frame_count": len(records),
        "point_count": len(point_ns),
        "median_crouch_pct_vs_upright": median_for("crouch_pct_vs_upright"),
        "median_hip_ankle_to_torso_ratio": median_for("hip_ankle_to_torso_ratio"),
        "median_stance_width_ratio": median_for("stance_width_ratio"),
        "racket_arm_up_share": None if not arm_values else round(sum(arm_values) / len(arm_values), 3),
        "left_wrist_up_share": share(records, "left_wrist_above_shoulder"),
        "right_wrist_up_share": share(records, "right_wrist_above_shoulder"),
    }


def share(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return None if not values else round(sum(values) / len(values), 3)


def write_preview(path: Path, previews: list[dict[str, Any]], racket_side: str) -> None:
    tile_size = 320
    columns = 4
    rows = max(1, math.ceil(len(previews) / columns))
    sheet = np.full((rows * (tile_size + 38), columns * tile_size, 3), 32, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, column = divmod(index, columns)
        x0, y0 = column * tile_size, row * (tile_size + 38)
        crop = cv2.resize(preview["crop"], (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        landmarks = preview.get("landmarks", [])
        visibility = preview.get("visibility", {})
        for start, end in POSE_CONNECTIONS:
            if end >= len(landmarks):
                continue
            if visibility.get(start, 0.0) < 0.25 or visibility.get(end, 0.0) < 0.25:
                continue
            a = tuple(round(v * tile_size / preview["side"]) for v in landmarks[start])
            b = tuple(round(v * tile_size / preview["side"]) for v in landmarks[end])
            cv2.line(crop, a, b, (30, 230, 80), 2, cv2.LINE_AA)
        for landmark_index, (px, py) in enumerate(landmarks):
            if visibility.get(landmark_index, 0.0) < 0.25:
                continue
            center = (round(px * tile_size / preview["side"]), round(py * tile_size / preview["side"]))
            cv2.circle(crop, center, 3, (30, 50, 240), -1, cv2.LINE_AA)
        sheet[y0:y0 + tile_size, x0:x0 + tile_size] = crop
        label = (
            f"t={preview['t']:.1f} {preview.get('context','')} "
            f"{preview.get('outcome','')} crouch={preview.get('crouch_pct_vs_upright')}%"
        )
        cv2.putText(sheet, label[:52], (x0 + 5, y0 + tile_size + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
    title = f"Pose preview | racket side: {racket_side}"
    sheet = cv2.copyMakeBorder(sheet, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    cv2.putText(sheet, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write pose preview: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the Stage 3 selected R player's pose from track crops.")
    parser.add_argument("input_video", type=Path, help="Stage 0 720p proxy video")
    parser.add_argument("players_json", type=Path, help="Stage 3 players.json")
    parser.add_argument("rallies_review_json", type=Path, help="Stage 6 review JSON with rally times")
    parser.add_argument("model_task", type=Path, help="MediaPipe pose_landmarker_heavy.task")
    parser.add_argument("output_json", type=Path, help="pose_stats.json output")
    parser.add_argument("preview_image", type=Path, help="pose_review.jpg output")
    parser.add_argument("--racket-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--crop-scale", type=float, default=CROP_SCALE)
    parser.add_argument("--overwrite", action="store_true", help="replace existing pose outputs")
    parser.add_argument("--max-rally-crop-frames", type=int, default=0,
                        help="optional cap for quick previews; 0 processes all selected frames")
    args = parser.parse_args()

    video_path = args.input_video.expanduser().resolve()
    players_path = args.players_json.expanduser().resolve()
    review_path = args.rallies_review_json.expanduser().resolve()
    model_path = args.model_task.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    preview_path = args.preview_image.expanduser().resolve()
    for path in (video_path, players_path, review_path, model_path):
        if not path.is_file():
            if path == model_path:
                parser.error(
                    f"pose model not found: {model_path}. Download the official free model to this path: "
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
                )
            parser.error(f"input file does not exist: {path}")
    if args.crop_scale < 1.0 or args.max_rally_crop_frames < 0:
        parser.error("crop scale must be at least 1.0 and frame cap cannot be negative")
    if output_path in (video_path, players_path, review_path, model_path) or preview_path in (
        video_path, players_path, review_path, model_path, output_path
    ):
        parser.error("input and output paths must be different")
    if not args.overwrite and (output_path.exists() or preview_path.exists()):
        parser.error("an output exists already; move it or choose different output paths")

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise RuntimeError("MediaPipe Tasks is unavailable; install mediapipe in the project .venv") from error

    frames, rallies, track_fps = load_inputs(players_path, review_path)
    times, smooth, speeds = smooth_positions(frames, track_fps)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        capture.release()
        raise RuntimeError("proxy video has an invalid frame rate")

    point_contexts = []
    cursor = 0
    baseline_flags = []
    sample_flags = []
    for index, timestamp in enumerate(times):
        point, cursor, context, baseline = frame_context(
            timestamp, smooth[index], speeds[index], rallies, cursor
        )
        point_contexts.append((point, context))
        baseline_flags.append(baseline)
        live = point is not None
        sample_flags.append(live or (baseline and index % max(1, round(track_fps / 2)) == 0))

    selected_count = sum(sample_flags)
    if args.max_rally_crop_frames:
        rally_indexes = [i for i, (point, _) in enumerate(point_contexts) if point is not None]
        allow = set(rally_indexes[:args.max_rally_crop_frames])
        sample_flags = [flag and (point_contexts[i][0] is None or i in allow) for i, flag in enumerate(sample_flags)]
        selected_count = sum(sample_flags)
    print(f"Pose samples selected: {selected_count} (10 fps during rallies; up to 2 fps for upright reference).")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    preview_counts: dict[str, int] = {}
    current_frame = -1
    try:
        for index, track_frame in enumerate(frames):
            timestamp = times[index]
            target_frame = round(timestamp * video_fps)
            while current_frame + 1 < target_frame:
                if not capture.grab():
                    break
                current_frame += 1
            ok, frame = capture.read()
            current_frame += 1
            if not ok:
                continue
            if not sample_flags[index]:
                continue
            crop_data = player_crop(frame, track_frame.get("rb"), args.crop_scale)
            if crop_data is None:
                continue
            crop, origin_x, origin_y, side = crop_data
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, round(timestamp * 1000))
            if not result.pose_landmarks:
                continue

            pose = posture_metrics(result.pose_landmarks[0], side)
            point, context = point_contexts[index]
            record: dict[str, Any] = {
                "t": round(timestamp, 3),
                "point_n": None if point is None else point["point_n"],
                "outcome": None if point is None else ("won" if point.get("winner") == "R" else "lost"),
                "context": context,
                "between_points_upright_candidate": baseline_flags[index],
                "speed_m_s": None if speeds[index] is None else round(speeds[index], 3),
                "distance_from_base_m": None if smooth[index] is None else round(math.dist(smooth[index], BASE_POINT), 3),
                "pose_quality": pose["pose_quality"],
                "hip_ankle_to_torso_ratio": pose["hip_ankle_to_torso_ratio"],
                "stance_width_ratio": pose["stance_width_ratio"],
                "left_wrist_above_shoulder": pose["left_wrist_above_shoulder"],
                "right_wrist_above_shoulder": pose["right_wrist_above_shoulder"],
            }
            records.append(record)
            preview_key = "BETWEEN_POINTS" if point is None else f"{context}_{record['outcome']}"
            if len(previews) < 24 and preview_counts.get(preview_key, 0) < 2:
                previews.append({
                    "t": timestamp, "crop": crop, "side": side,
                    "landmarks": pose["landmarks"], "visibility": pose.get("visibility", {}),
                    "context": context,
                    "outcome": "baseline" if point is None else record["outcome"],
                    "record": record,
                })
                preview_counts[preview_key] = preview_counts.get(preview_key, 0) + 1
            if len(records) % 100 == 0:
                print(f"Pose frames processed: {len(records)} / {selected_count}")
    finally:
        landmarker.close()
        capture.release()

    if not records:
        raise ValueError("MediaPipe found no usable poses; check the model, crop, and player boxes")

    upright_ratios = [
        record["hip_ankle_to_torso_ratio"]
        for record in records
        if record["between_points_upright_candidate"]
        and finite_number(record.get("hip_ankle_to_torso_ratio"))
    ]
    valid_ratios = [record["hip_ankle_to_torso_ratio"] for record in records if finite_number(record.get("hip_ankle_to_torso_ratio"))]
    if len(upright_ratios) >= 5:
        upright_reference = float(np.percentile(upright_ratios, 75))
        upright_reference_method = "75th percentile of slow near-base poses between points"
    elif valid_ratios:
        upright_reference = float(np.percentile(valid_ratios, 90))
        upright_reference_method = "90th percentile fallback from visible poses"
    else:
        upright_reference = None
        upright_reference_method = "unavailable; key body landmarks were not visible"

    for record in records:
        ratio = record.get("hip_ankle_to_torso_ratio")
        record["crouch_pct_vs_upright"] = (
            None if upright_reference is None or not finite_number(ratio) or upright_reference <= 0
            else round(max(0.0, min(100.0, 100.0 * (upright_reference - ratio) / upright_reference)), 1)
        )

    racket_inference = infer_racket_side(records)
    racket_side = args.racket_side if args.racket_side != "auto" else racket_inference["side"]
    for record in records:
        record["racket_arm_up"] = (
            record.get(f"{racket_side}_wrist_above_shoulder") if racket_side in ("left", "right") else None
        )

    grouped: dict[str, Any] = {}
    for outcome in ("won", "lost"):
        grouped[outcome] = {}
        for context in ("READY", "FRONT", "REAR"):
            subset = [record for record in records if record.get("outcome") == outcome and record["context"] == context]
            grouped[outcome][context] = summarize(subset)

    per_point = []
    for rally in rallies:
        point_records = [record for record in records if record.get("point_n") == rally["point_n"]]
        per_point.append({
            "point_n": rally["point_n"],
            "winner": rally.get("winner"),
            "rally_end_estimated": rally.get("rally_end_estimated", False),
            "pose_by_context": {
                context: summarize([record for record in point_records if record["context"] == context])
                for context in ("READY", "FRONT", "REAR", "MID")
            },
        })

    preview_metrics = {round(item["t"], 3): item["record"] for item in previews}
    for item in previews:
        record = preview_metrics[round(item["t"], 3)]
        item["crouch_pct_vs_upright"] = record.get("crouch_pct_vs_upright")
    write_preview(preview_path, previews, racket_side)

    output = {
        "source_video": video_path.name,
        "source_players_json": players_path.name,
        "source_rallies_review_json": review_path.name,
        "model": model_path.name,
        "player": "R (selected in Stage 3 by clothing color)",
        "analysis_fps": track_fps,
        "crop_scale": args.crop_scale,
        "contexts": {
            "READY": "in a rally, speed <= 0.8 m/s and within 1.6 m of base point",
            "FRONT": "in a rally, near the net beyond the short service line",
            "REAR": "in a rally, within 1.6 m of own baseline",
        },
        "upright_reference_ratio": None if upright_reference is None else round(upright_reference, 4),
        "upright_reference_method": upright_reference_method,
        "racket_side": racket_side,
        "racket_side_inference": racket_inference,
        "summary_by_outcome_and_context": grouped,
        "points": per_point,
        "frames": records,
        "preview_image": preview_path.name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Pose samples with landmarks: {len(records)}; upright reference frames: {len(upright_ratios)}.")
    print(f"Inferred racket side: {racket_side} ({racket_inference['wrist_above_shoulder_rate']}).")
    print(f"Pose summary JSON: {output_path}")
    print(f"Pose preview: {preview_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, cv2.error, json.JSONDecodeError) as error:
        print(f"Stage 8 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
