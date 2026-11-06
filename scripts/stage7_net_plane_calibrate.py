#!/usr/bin/env python3
"""Calibrate an image-space vertical model of the badminton net plane."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POINT_LABELS = (
    "net-floor left",
    "net-floor right",
    "net-tape left",
    "net-tape right",
)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def project(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    result = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if abs(float(result[2])) < 1e-9:
        raise ValueError("net-plane homography projects a point to infinity")
    return float(result[0] / result[2]), float(result[1] / result[2])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Click the four net-plane corners so Stage 7 can find image-space shuttle crossing candidates."
    )
    parser.add_argument("video", type=Path, help="Stage 0 proxy used for Stage 3A tracks")
    parser.add_argument("court_calibration_json", type=Path, help="Stage 2 court_calibration.json")
    parser.add_argument("output_json", type=Path, help="output net_plane_calibration.json")
    parser.add_argument("--frame-time", type=float, help="clear frame time in proxy seconds (defaults to Stage 2 frame)")
    parser.add_argument("--net-top-height-m", type=float, default=1.55, help="net top height; default 1.55 m")
    parser.add_argument(
        "--crossing-height-m", type=float, default=4.5,
        help="extend the projected net plane to this height for shuttle paths above the tape; default 4.5 m",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    court_path = args.court_calibration_json.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    if not video_path.is_file() or not court_path.is_file():
        parser.error("video and court calibration JSON must exist")
    if output_path in (video_path, court_path):
        parser.error("output path must differ from both inputs")
    if args.net_top_height_m <= 0 or args.crossing_height_m <= args.net_top_height_m:
        parser.error("crossing height must be greater than the positive net-top height")

    try:
        court = load_json(court_path)
        court_source = court.get("source_video")
        if court_source and Path(str(court_source)).name != video_path.name:
            raise ValueError("court calibration refers to a different proxy video")
        court_matrix = np.asarray(court.get("court_to_pixel"), dtype=np.float64)
        court_size = court.get("court_size_m", {})
        court_width = float(court_size.get("width", 6.1))
        court_length = float(court_size.get("length", 13.4))
        if court_matrix.shape != (3, 3) or not np.isfinite(court_matrix).all():
            raise ValueError("court calibration needs a finite 3x3 court_to_pixel matrix")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_time = float(args.frame_time if args.frame_time is not None else court.get("frame_time_sec", 0.0))
        if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            capture.release()
            raise ValueError("video has invalid frame properties")
        if frame_time < 0 or frame_time >= frame_count / fps:
            capture.release()
            raise ValueError("frame time is outside the video")
        frame_index = round(frame_time * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, source = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"could not read calibration frame at {frame_time:.2f}s")

        court_frame_size = court.get("frame_size_px", {})
        if court_frame_size and (
            int(court_frame_size.get("width", width)) != width
            or int(court_frame_size.get("height", height)) != height
        ):
            raise ValueError("court calibration frame size differs from the proxy")

        # The Stage 2 homography supplies a visual reference for the floor line.
        bottom_left = project(court_matrix, (0.0, court_length / 2.0))
        bottom_right = project(court_matrix, (court_width, court_length / 2.0))
        clicked: list[tuple[int, int]] = []
        window_name = "Stage 7 net-plane calibration"

        def paint() -> np.ndarray:
            view = source.copy()
            cv2.line(
                view,
                tuple(round(v) for v in bottom_left),
                tuple(round(v) for v in bottom_right),
                (0, 210, 255), 2, cv2.LINE_AA,
            )
            instructions = [
                "Click only the four physical net corners: floor-left/right, then WHITE TAPE at those posts.",
                "The green flight corridor is calculated above the tape; do not click the wall, roof, or image edge.",
                "Yellow: Stage 2 floor line. U undo | R reset | S save | Esc cancel.",
            ]
            for index, line in enumerate(instructions):
                cv2.putText(view, line, (16, 25 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (10, 10, 10), 4, cv2.LINE_AA)
                cv2.putText(view, line, (16, 25 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (255, 255, 255), 1, cv2.LINE_AA)
            for index, point in enumerate(clicked):
                cv2.circle(view, point, 7, (0, 80, 255), -1, cv2.LINE_AA)
                cv2.putText(view, f"{index + 1}: {POINT_LABELS[index]}",
                            (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2, cv2.LINE_AA)
            if len(clicked) == 4:
                top_left, top_right = clicked[2], clicked[3]
                cv2.line(view, clicked[0], clicked[1], (80, 255, 100), 2, cv2.LINE_AA)
                cv2.line(view, top_left, top_right, (80, 255, 100), 2, cv2.LINE_AA)
                cv2.line(view, clicked[0], top_left, (80, 255, 100), 2, cv2.LINE_AA)
                cv2.line(view, clicked[1], top_right, (80, 255, 100), 2, cv2.LINE_AA)
            return view

        def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
                clicked.append((int(x), int(y)))

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)
        save_requested = False
        while True:
            cv2.imshow(window_name, paint())
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("u"), 8, 127) and clicked:
                clicked.pop()
            elif key == ord("r"):
                clicked.clear()
            elif key == ord("s"):
                if len(clicked) == 4:
                    save_requested = True
                    break
                print(f"Click all four points first ({len(clicked)}/4).", flush=True)
        cv2.destroyWindow(window_name)
        if not save_requested:
            print("Calibration cancelled; no output written.")
            return 0

        bottom_left, bottom_right, top_left, top_right = clicked
        if bottom_left[0] >= bottom_right[0] or top_left[0] >= top_right[0]:
            raise ValueError("click left and right net corners in consistent camera-view order")
        for bottom, top, label in (
            (bottom_left, top_left, "left"),
            (bottom_right, top_right, "right"),
        ):
            vertical_gap = bottom[1] - top[1]
            if top[1] <= 10 or vertical_gap < 15 or vertical_gap > 0.45 * height:
                raise ValueError(
                    f"the {label} net-top click must be on the white tape at the post, "
                    "not at the top edge of the frame; reset and click the actual net"
                )
            if abs(top[0] - bottom[0]) > 0.25 * width:
                raise ValueError(
                    f"the {label} net-top click is too far sideways from its floor point; "
                    "click the tape at the same post"
                )

        world_points = np.asarray(
            [
                [0.0, 0.0],
                [court_width, 0.0],
                [0.0, args.net_top_height_m],
                [court_width, args.net_top_height_m],
            ],
            dtype=np.float64,
        )
        image_points = np.asarray(clicked, dtype=np.float64)
        plane_matrix, _ = cv2.findHomography(world_points, image_points, method=0)
        if plane_matrix is None or not np.isfinite(plane_matrix).all():
            raise ValueError("could not calculate the net-plane homography")
        extended_left = project(plane_matrix, (0.0, args.crossing_height_m))
        extended_right = project(plane_matrix, (court_width, args.crossing_height_m))
        polygon = [
            list(map(float, clicked[0])),
            list(map(float, clicked[1])),
            list(extended_right),
            list(extended_left),
        ]
        if min(point[1] for point in polygon) < -0.45 * height:
            raise ValueError(
                "the extended net corridor projects far above the frame; check that both upper clicks are on the net tape"
            )
        area = abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
        if area < 100.0:
            raise ValueError("net-plane projection is too small; check the four clicked corners")
        if any(not math.isfinite(value) for point in polygon for value in point):
            raise ValueError("net-plane extension produced non-finite image points")

        output = {
            "schema_version": 1,
            "source_video": video_path.name,
            "source_court_calibration": court_path.name,
            "frame_time_sec": round(frame_index / fps, 3),
            "frame_size_px": {"width": width, "height": height},
            "court_width_m": court_width,
            "net_top_height_m": args.net_top_height_m,
            "crossing_height_m": args.crossing_height_m,
            "point_order": list(POINT_LABELS),
            "net_plane_corner_px": [list(point) for point in clicked],
            "net_plane_homography_xz_to_pixel": plane_matrix.tolist(),
            "extended_plane_polygon_px": polygon,
            "calibration_note": (
                "The polygon extends the visible net plane above the tape as an image-space crossing corridor. "
                "Its detections are candidate crossings, not metric shuttle positions."
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        preview = source.copy()
        clipped = np.asarray(
            [[round(x), round(y)] for x, y in polygon], dtype=np.int32
        )
        cv2.polylines(preview, [clipped], True, (60, 240, 90), 2, cv2.LINE_AA)
        cv2.line(preview, tuple(clicked[0]), tuple(clicked[1]), (0, 210, 255), 2, cv2.LINE_AA)
        cv2.line(preview, tuple(clicked[2]), tuple(clicked[3]), (255, 180, 30), 2, cv2.LINE_AA)
        preview_path = output_path.with_name(f"{output_path.stem}.preview.jpg")
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"could not write calibration preview: {preview_path}")
        print(f"Net-plane calibration: {output_path}")
        print(f"Extended image-space corridor preview: {preview_path}")
        print(
            f"The four clicks mark the physical net rectangle. Review the green corridor extrapolated to "
            f"{args.crossing_height_m:g} m above the court; it may reach the top of the image, but should stay aligned with the selected net."
        )
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error, json.JSONDecodeError) as error:
        print(f"Net-plane calibration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
