#!/usr/bin/env python3
"""Estimate the user's ready stance and body-position features with a local YOLO pose model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import COURT_LENGTH, NET_Y, artifact_path, load_job, open_video, proxy_path, read_json, write_json


def choose_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def point_for_time(points: list[dict[str, Any]], seconds: float) -> dict[str, Any] | None:
    return next((point for point in points if float(point["r0"]) <= seconds <= float(point["r1"])), None)


def keypoint_metrics(keypoints: np.ndarray, confidences: np.ndarray | None) -> dict[str, float | bool] | None:
    def point(index: int) -> np.ndarray | None:
        if confidences is not None:
            confidence = float(confidences[index])
            if not np.isfinite(confidence) or confidence < 0.25:
                return None
        value = keypoints[index]
        return value if np.isfinite(value).all() else None

    left_shoulder, right_shoulder = point(5), point(6)
    left_hip, right_hip = point(11), point(12)
    left_ankle, right_ankle = point(15), point(16)
    left_wrist, right_wrist = point(9), point(10)
    if any(value is None for value in (left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle)):
        return None
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    ankle_mid = (left_ankle + right_ankle) / 2
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if torso < 1 or shoulder_width < 1:
        return None
    ratio = abs(float(hip_mid[1] - ankle_mid[1])) / torso
    stance = float(np.linalg.norm(left_ankle - right_ankle)) / shoulder_width
    racket_up = any(wrist is not None and wrist[1] < shoulder_mid[1] for wrist in (left_wrist, right_wrist))
    return {"hip_ankle_to_torso_ratio": ratio, "stance_width_ratio": stance, "racket_arm_up": racket_up}


def summarize(samples: list[dict[str, Any]], outcome: str, context: str) -> dict[str, Any]:
    selected = [
        sample
        for sample in samples
        if sample.get("outcome") == outcome
        and sample.get("context") == context
        and sample.get("hip_ankle_to_torso_ratio") is not None
    ]
    result: dict[str, Any] = {"samples": len(selected)}
    for key in ("crouch_delta", "stance_width_ratio"):
        values = [float(sample[key]) for sample in selected if sample.get(key) is not None]
        result[key] = round(float(np.mean(values)), 3) if values else None
    racket = [sample["racket_arm_up"] for sample in selected if sample.get("racket_arm_up") is not None]
    result["racket_arm_up_rate"] = round(sum(racket) / len(racket), 3) if racket else None
    return result


def synchronized_samples(capture: Any, frames: list[dict[str, Any]], every: int):
    """Read the proxy once and pair its every-Nth frame with Stage 1/3 samples."""
    sample_index = 0
    frame_index = 0
    while sample_index < len(frames):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every == 0:
            yield sample_index, frames[sample_index], frame
            sample_index += 1
        frame_index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--players", help="Player JSON; defaults to players.json beside job.json.")
    parser.add_argument("--points", help="Reviewed points JSON; defaults to points.json beside job.json.")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose weights.")
    parser.add_argument("--output", help="Output JSON; defaults to pose_stats.json beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video = proxy_path(job)
        players_path = Path(args.players).expanduser().resolve() if args.players else artifact_path(job_path, "players.json")
        points_path = Path(args.points).expanduser().resolve() if args.points else artifact_path(job_path, "points.json")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "pose_stats.json")
        player_data = read_json(players_path)
        points = read_json(points_path).get("points", [])

        from ultralytics import YOLO

        device = choose_device()
        print(f"Loading {args.model} on {device}. The weights may download the first time.")
        model = YOLO(args.model)
        capture = open_video(video)
        samples = []
        frames = player_data.get("frames", [])
        sample_every = max(1, int(player_data.get("sample_every_frames", 3)))
        try:
            for index, item, frame in synchronized_samples(capture, frames, sample_every):
                target_box = item.get("rb")
                position = item.get("r")
                if target_box is None or position is None:
                    continue
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = (float(value) for value in target_box[:4])
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
                side = max(x2 - x1, y2 - y1) * 1.35
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                crop_x1 = max(0, int(center_x - side / 2))
                crop_y1 = max(0, int(center_y - side / 2))
                crop_x2 = min(width, int(center_x + side / 2))
                crop_y2 = min(height, int(center_y + side / 2))
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue
                result = model.predict(crop, conf=0.25, imgsz=320, device=device, verbose=False)[0]
                if result.keypoints is None or len(result.keypoints) == 0:
                    continue
                keypoints_all = result.keypoints.xy.cpu().numpy()
                confidence_all = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                target_local = np.asarray([x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1])
                best_index = 0
                best_iou = -1.0
                if result.boxes is not None:
                    pose_boxes = result.boxes.xyxy.cpu().numpy()
                    for pose_index, pose_box in enumerate(pose_boxes):
                        ix1, iy1 = max(target_local[0], pose_box[0]), max(target_local[1], pose_box[1])
                        ix2, iy2 = min(target_local[2], pose_box[2]), min(target_local[3], pose_box[3])
                        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        area_a = max(0, target_local[2] - target_local[0]) * max(0, target_local[3] - target_local[1])
                        area_b = max(0, pose_box[2] - pose_box[0]) * max(0, pose_box[3] - pose_box[1])
                        overlap = intersection / max(1.0, area_a + area_b - intersection)
                        if overlap > best_iou:
                            best_iou, best_index = overlap, pose_index
                keypoints = keypoints_all[best_index].copy()
                keypoints[:, 0] += crop_x1
                keypoints[:, 1] += crop_y1
                confidences = confidence_all[best_index] if confidence_all is not None else None
                metrics = keypoint_metrics(keypoints, confidences)
                skeleton = []
                for keypoint_index, keypoint in enumerate(keypoints):
                    confidence = float(confidences[keypoint_index]) if confidences is not None else 1.0
                    normalized = np.asarray([keypoint[0] / width, keypoint[1] / height, confidence])
                    if not np.isfinite(normalized).all():
                        normalized = np.asarray([0.0, 0.0, 0.0])
                    skeleton.append(
                        [
                            round(float(normalized[0]), 6),
                            round(float(normalized[1]), 6),
                            round(float(normalized[2]), 4),
                        ]
                    )

                seconds = float(item["t"])
                point = point_for_time(points, seconds)
                is_near = float(position[1]) < NET_Y
                depth = float(position[1]) if is_near else COURT_LENGTH - float(position[1])
                speed = None
                if index > 0 and index + 1 < len(frames):
                    before, after = frames[index - 1], frames[index + 1]
                    if before.get("r") is not None and after.get("r") is not None:
                        dt = float(after["t"]) - float(before["t"])
                        if 0 < dt <= 0.5:
                            speed = float(np.linalg.norm(np.asarray(after["r"]) - np.asarray(before["r"])) / dt)
                base = [3.05, 3.4 if is_near else COURT_LENGTH - 3.4]
                base_distance = float(np.linalg.norm(np.asarray(position) - np.asarray(base)))
                if point is None:
                    context = "BETWEEN_POINTS"
                elif speed is not None and speed < 0.8 and base_distance < 1.6:
                    context = "READY"
                elif depth >= 4.72:
                    context = "FRONT"
                elif depth <= 3.1:
                    context = "REAR"
                else:
                    context = "RALLY"
                sample = {
                    "t": seconds,
                    "point": point["n"] if point else None,
                    "outcome": ("won" if point["winner"] == "R" else "lost") if point else None,
                    "context": context,
                    "skeleton": skeleton,
                }
                if metrics is not None:
                    sample.update(metrics)
                samples.append(sample)
                if len(samples) % 500 == 0:
                    print(f"Processed pose for {len(samples)} samples.")
        finally:
            capture.release()

        rest_ratios = [
            float(sample["hip_ankle_to_torso_ratio"])
            for sample in samples
            if sample["point"] is None and sample.get("hip_ankle_to_torso_ratio") is not None
        ]
        if not rest_ratios:
            rest_ratios = [
                float(sample["hip_ankle_to_torso_ratio"])
                for sample in samples
                if sample.get("hip_ankle_to_torso_ratio") is not None
            ]
        upright_reference = float(np.percentile(rest_ratios, 75)) if rest_ratios else None
        for sample in samples:
            sample["upright_reference_ratio"] = round(upright_reference, 3) if upright_reference is not None else None
            ratio = sample.get("hip_ankle_to_torso_ratio")
            sample["crouch_delta"] = (
                round(max(0.0, upright_reference - float(ratio)), 3)
                if upright_reference is not None and ratio is not None
                else None
            )

        summaries = {
            outcome: {context: summarize(samples, outcome, context) for context in ("READY", "FRONT", "REAR", "RALLY")}
            for outcome in ("won", "lost")
        }
        write_json(
            output_path,
            {
                "schema_version": 1,
                "model": args.model,
                "device": device,
                "upright_reference_ratio": round(upright_reference, 3) if upright_reference is not None else None,
                "samples": samples,
                "summary_by_outcome_and_context": summaries,
            },
        )
        print(f"Saved {len(samples)} pose samples and win/loss summaries to {output_path}.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Pose analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
