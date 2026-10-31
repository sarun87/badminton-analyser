#!/usr/bin/env python3
"""Detect people on the proxy video at 10 FPS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import artifact_path, load_job, open_video, proxy_path, write_json


def get_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to the Stage 0 job.json.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics person detector weights.")
    parser.add_argument("--output", help="Output JSON path; defaults to detections.json beside job.json.")
    parser.add_argument("--every", type=int, default=3, help="Analyze every Nth proxy frame (default: 3 = 10 FPS).")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    job_path = Path(args.job).expanduser().resolve()
    try:
        job = load_job(job_path)
        video_path = proxy_path(job)
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "detections.json")
        if args.every < 1:
            raise ValueError("--every must be at least 1.")

        from ultralytics import YOLO

        device = get_device()
        print(f"Loading {args.model} on {device}. The weights may download the first time.")
        model = YOLO(args.model)
        capture = open_video(video_path)
        fps = float(capture.get(5))
        frames = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % args.every == 0:
                    height, width = frame.shape[:2]
                    result = model.predict(
                        frame,
                        classes=[0],
                        conf=args.confidence,
                        iou=args.iou,
                        imgsz=960,
                        device=device,
                        verbose=False,
                    )[0]
                    people = []
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box
                            people.append(
                                [
                                    round(float(x1 / width), 6),
                                    round(float(y1 / height), 6),
                                    round(float(x2 / width), 6),
                                    round(float(y2 / height), 6),
                                    round(float(score), 4),
                                ]
                            )
                    frames.append({"t": round(frame_index / fps, 3), "p": people})
                    if len(frames) % 300 == 0:
                        print(f"Processed {frames[-1]['t']:.1f}s ({len(frames)} samples).")
                frame_index += 1
        finally:
            capture.release()

        write_json(
            output_path,
            {
                "schema_version": 1,
                "video": str(video_path),
                "fps": round(fps / args.every, 4),
                "sample_every_frames": args.every,
                "confidence": args.confidence,
                "iou": args.iou,
                "model": args.model,
                "device": device,
                "frames": frames,
            },
        )
        print(f"Saved {len(frames)} sampled frames to {output_path}")
        print("Detections are normalized xyxy boxes with confidence, sampled about 10 times per second.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Detection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
