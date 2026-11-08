#!/usr/bin/env python3
"""Run YOLOX-S person detection on every third frame of a proxy video."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


INPUT_HEIGHT = 640
INPUT_WIDTH = 640
YOLOX_STRIDES = (8, 16, 32)
PERSON_CLASS_ID = 0  # COCO class order: person is class 0.
COREML_PROVIDER = "CoreMLExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
COREML_COMPUTE_UNITS = "CPUAndGPU"


def create_inference_session(
    model_path: Path,
    session_options: ort.SessionOptions,
    requested_provider: str,
) -> tuple[ort.InferenceSession, str, list[str]]:
    """Create a Core ML session when requested or available, with CPU fallback."""
    available_providers = ort.get_available_providers()
    if requested_provider == "cpu":
        selected_provider = "cpu"
    elif requested_provider == "coreml":
        if COREML_PROVIDER not in available_providers:
            raise RuntimeError(
                f"{COREML_PROVIDER} was requested but is not available in this ONNX Runtime build; "
                f"available providers: {', '.join(available_providers)}"
            )
        selected_provider = "coreml"
    else:
        selected_provider = "coreml" if COREML_PROVIDER in available_providers else "cpu"

    if selected_provider == "coreml":
        providers = [
            (COREML_PROVIDER, {"MLComputeUnits": COREML_COMPUTE_UNITS}),
            CPU_PROVIDER,
        ]
        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception as error:
            if requested_provider != "auto":
                raise RuntimeError(f"Could not create a Core ML inference session: {error}") from error
            print(
                f"Warning: Core ML session setup failed ({error}); retrying with CPU only.",
                file=sys.stderr,
            )
            selected_provider = "cpu"
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=[CPU_PROVIDER],
            )
    else:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[CPU_PROVIDER],
        )

    return session, selected_provider, available_providers


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize with YOLOX's top-left, 114-gray padding convention."""
    height, width = frame.shape[:2]
    ratio = min(INPUT_HEIGHT / height, INPUT_WIDTH / width)
    resized_width = int(width * ratio)
    resized_height = int(height * ratio)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    padded = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    tensor = padded.transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(tensor), ratio


def decode_yolox(raw: np.ndarray) -> np.ndarray:
    """Decode the grid-relative YOLOX output into input-image pixel coordinates."""
    decoded = raw.copy()
    offset = 0
    for stride in YOLOX_STRIDES:
        grid_height = INPUT_HEIGHT // stride
        grid_width = INPUT_WIDTH // stride
        grid_y, grid_x = np.meshgrid(
            np.arange(grid_height), np.arange(grid_width), indexing="ij"
        )
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2).astype(np.float32)
        count = len(grid)
        end = offset + count
        decoded[offset:end, :2] = (decoded[offset:end, :2] + grid) * stride
        decoded[offset:end, 2:4] = (
            np.exp(np.clip(decoded[offset:end, 2:4], -20.0, 20.0)) * stride
        )
        offset = end

    if offset != len(decoded):
        raise RuntimeError(
            f"YOLOX output has {len(decoded)} rows; expected {offset} for a 640x640 input"
        )
    return decoded


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression for the person class."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    kept: list[int] = []

    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]

    return kept


def detect_people(
    raw_output: np.ndarray,
    ratio: float,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped source-frame XYXY boxes and person confidence scores."""
    predictions = decode_yolox(raw_output)
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = predictions[:, 4] * class_scores[np.arange(len(predictions)), class_ids]
    selected = (class_ids == PERSON_CLASS_ID) & (scores >= confidence_threshold)
    if not np.any(selected):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    selected_predictions = predictions[selected]
    selected_scores = scores[selected].astype(np.float32)
    centers_x = selected_predictions[:, 0]
    centers_y = selected_predictions[:, 1]
    widths = selected_predictions[:, 2]
    heights = selected_predictions[:, 3]
    boxes = np.column_stack(
        (
            centers_x - widths / 2,
            centers_y - heights / 2,
            centers_x + widths / 2,
            centers_y + heights / 2,
        )
    ).astype(np.float32)

    # Preprocessing pads only on the right and bottom, so no offset is removed.
    boxes /= ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    selected_scores = selected_scores[valid]

    keep = nms_xyxy(boxes, selected_scores, nms_iou)
    return boxes[keep], selected_scores[keep]


def make_contact_sheet(
    samples: list[tuple[float, np.ndarray, list[list[float]]]], output: Path
) -> None:
    """Write a 4-column visual sample with person boxes overlaid."""
    if not samples:
        raise RuntimeError("No preview frames were collected")

    columns = 4
    tile_width = 480
    image_height = 270
    label_height = 30
    tile_height = image_height + label_height
    rows = math.ceil(len(samples) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 32, dtype=np.uint8)

    for index, (timestamp, frame, detections) in enumerate(samples):
        row, column = divmod(index, columns)
        x0 = column * tile_width
        y0 = row * tile_height
        thumb = cv2.resize(frame, (tile_width, image_height), interpolation=cv2.INTER_AREA)
        sheet[y0 + label_height : y0 + tile_height, x0 : x0 + tile_width] = thumb
        cv2.putText(
            sheet,
            f"t={timestamp:.1f}s  detections={len(detections)}",
            (x0 + 10, y0 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for x1, y1, x2, y2, score in detections:
            left = x0 + round(x1 * tile_width)
            right = x0 + round(x2 * tile_width)
            top = y0 + label_height + round(y1 * image_height)
            bottom = y0 + label_height + round(y2 * image_height)
            cv2.rectangle(sheet, (left, top), (right, bottom), (0, 255, 0), 2)
            label_y = max(y0 + label_height + 16, top - 5)
            cv2.putText(
                sheet,
                f"person {score:.2f}",
                (left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect people in a 720p proxy using local YOLOX-S on Apple Core ML or the CPU."
    )
    parser.add_argument("input_video", type=Path, help="proxy video from Stage 0")
    parser.add_argument("model", type=Path, help="YOLOX-S ONNX model file")
    parser.add_argument("output_json", type=Path, help="path for dets.json")
    parser.add_argument("contact_sheet", type=Path, help="path for detection contact sheet JPG")
    parser.add_argument("--every", type=int, default=3, help="detect every Nth frame (default: 3)")
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="person confidence threshold (default: 0.3)"
    )
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--execution-provider",
        choices=("auto", "coreml", "cpu"),
        default="auto",
        help="auto uses Core ML with CPU-and-GPU compute when available; otherwise CPU (default: auto)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    source = args.input_video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    contact_sheet = args.contact_sheet.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input video does not exist: {source}")
    if not model_path.is_file():
        parser.error(f"ONNX model does not exist: {model_path}")
    if args.every < 1 or args.threads < 1:
        parser.error("--every and --threads must be positive integers")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.nms_iou <= 1.0:
        parser.error("--nms-iou must be between 0 and 1")
    if output_json == contact_sheet or source in (output_json, contact_sheet):
        parser.error("input and output paths must be different")
    existing = [path for path in (output_json, contact_sheet) if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output already exists: " + ", ".join(str(path) for path in existing) + "; pass --overwrite"
        )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        parser.error(f"OpenCV could not open input video: {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        cap.release()
        parser.error("Could not read a valid frame rate from the input video")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = args.threads
    try:
        session, selected_provider, available_providers = create_inference_session(
            model_path, session_options, args.execution_provider
        )
    except RuntimeError as error:
        cap.release()
        parser.error(str(error))
    model_input = session.get_inputs()[0]
    input_shape = model_input.shape
    if input_shape != [1, 3, INPUT_HEIGHT, INPUT_WIDTH]:
        cap.release()
        parser.error(f"Expected YOLOX input shape [1, 3, 640, 640], got {input_shape}")

    output_fps = source_fps / args.every
    sample_count = 0
    source_frame_index = 0
    total_detections = 0
    records: list[dict[str, object]] = []
    preview_samples: list[tuple[float, np.ndarray, list[list[float]]]] = []
    estimated_sampled_frames = math.ceil(total_source_frames / args.every) if total_source_frames else 0
    preview_count = min(12, estimated_sampled_frames) if estimated_sampled_frames else 12
    if preview_count <= 1:
        preview_targets = {0}
    else:
        preview_targets = {
            round(i * (estimated_sampled_frames - 1) / (preview_count - 1))
            for i in range(preview_count)
        }

    print(f"Input: {source}")
    print(f"Model: {model_path}")
    print(
        f"Execution provider: {selected_provider}; session providers: "
        f"{', '.join(session.get_providers())}."
    )
    if selected_provider == "coreml":
        print(f"Core ML compute units: {COREML_COMPUTE_UNITS} (CPU and Apple GPU).")
    elif args.execution_provider == "auto" and COREML_PROVIDER not in available_providers:
        print(f"Core ML unavailable; using {CPU_PROVIDER}.")
    print(
        f"Sampling every {args.every} frames ({output_fps:.3f} detections/s); "
        f"confidence={args.confidence:.2f}, NMS IoU={args.nms_iou:.2f}, CPU threads={args.threads}."
    )
    start_time = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = source_frame_index
            source_frame_index += 1
            if frame_index % args.every != 0:
                continue

            frame_height, frame_width = frame.shape[:2]
            tensor, ratio = letterbox(frame)
            raw = session.run(None, {model_input.name: tensor})[0]
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 85:
                raise RuntimeError(f"Unexpected YOLOX output shape: {raw.shape}")
            boxes, scores = detect_people(
                raw[0],
                ratio,
                frame_width,
                frame_height,
                args.confidence,
                args.nms_iou,
            )

            detections: list[list[float]] = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                detections.append(
                    [
                        round(float(x1 / frame_width), 6),
                        round(float(y1 / frame_height), 6),
                        round(float(x2 / frame_width), 6),
                        round(float(y2 / frame_height), 6),
                        round(float(score), 6),
                    ]
                )

            timestamp = frame_index / source_fps
            records.append({"t": round(timestamp, 3), "p": detections})
            total_detections += len(detections)
            if sample_count in preview_targets:
                preview_samples.append((timestamp, frame.copy(), detections))
            sample_count += 1
            if sample_count % 1000 == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"Processed {sample_count}/{estimated_sampled_frames} sampled frames; "
                    f"{total_detections} person boxes so far; elapsed {elapsed:.0f}s.",
                    flush=True,
                )
    finally:
        cap.release()

    if not records:
        raise RuntimeError("No frames were processed")

    payload = {
        "fps": round(output_fps, 6),
        "source_fps": round(source_fps, 6),
        "sample_every": args.every,
        "confidence_threshold": args.confidence,
        "nms_iou": args.nms_iou,
        "model": model_path.name,
        "execution": {
            "requested_provider": args.execution_provider,
            "selected_provider": selected_provider,
            "session_providers": session.get_providers(),
            "coreml_compute_units": COREML_COMPUTE_UNITS if selected_provider == "coreml" else None,
        },
        "frames": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
        file.write("\n")
    make_contact_sheet(preview_samples, contact_sheet)

    elapsed = time.monotonic() - start_time
    print(f"Detection complete: {len(records)} sampled frames, {total_detections} person boxes.")
    print(f"JSON: {output_json}")
    print(f"Contact sheet: {contact_sheet}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, cv2.error) as error:
        print(f"Stage 1 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
