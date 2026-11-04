#!/usr/bin/env python3
"""Click court-line landmarks, solve a court homography, and draw its overlay."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
LANDMARKS = (
    ("near_left_corner", 0.0, 0.0),
    ("near_right_corner", 6.1, 0.0),
    ("far_right_corner", 6.1, 13.4),
    ("far_left_corner", 0.0, 13.4),
    ("near_short_service_center", 3.05, 4.72),
    ("far_short_service_center", 3.05, 8.68),
    ("near_doubles_long_service_center", 3.05, 0.76),
    ("far_doubles_long_service_center", 3.05, 12.64),
    ("near_left_doubles_long_service_intersection", 0.0, 0.76),
    ("near_right_doubles_long_service_intersection", 6.1, 0.76),
    ("far_left_doubles_long_service_intersection", 0.0, 12.64),
    ("far_right_doubles_long_service_intersection", 6.1, 12.64),
)


@dataclass
class CalibrationUI:
    frame: np.ndarray
    display_width: int
    display_height: int
    landmarks: list[tuple[str, float, float]]
    clicks: list[tuple[float, float]]
    saved_overlay: np.ndarray | None = None

    def mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if self.saved_overlay is not None or y >= self.display_height:
            return

        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < len(self.landmarks):
            frame_height, frame_width = self.frame.shape[:2]
            source_x = x * frame_width / self.display_width
            source_y = y * frame_height / self.display_height
            self.clicks.append((source_x, source_y))
            self.print_next()
        elif event == cv2.EVENT_RBUTTONDOWN and self.clicks:
            removed = self.clicks.pop()
            print(f"Removed last point at ({removed[0]:.1f}, {removed[1]:.1f}) px.")
            self.print_next()

    def print_next(self) -> None:
        if len(self.clicks) == len(self.landmarks):
            print(f"All {len(self.landmarks)} points are selected. Press s to save, or right-click to undo.")
            return
        index = len(self.clicks)
        name, x_m, y_m = self.landmarks[index]
        print(f"Click {index + 1}/{len(self.landmarks)}: {name} at court ({x_m:.2f}, {y_m:.2f}) m.")

    def render(self) -> np.ndarray:
        frame = self.saved_overlay if self.saved_overlay is not None else self.frame
        shown = cv2.resize(
            frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA if self.display_width < self.frame.shape[1] else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((self.display_height + 58, self.display_width, 3), dtype=np.uint8)
        canvas[: self.display_height] = shown

        if self.saved_overlay is None:
            for index, (x, y) in enumerate(self.clicks):
                shown_x = round(x * self.display_width / self.frame.shape[1])
                shown_y = round(y * self.display_height / self.frame.shape[0])
                cv2.circle(canvas, (shown_x, shown_y), 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (shown_x + 8, shown_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if len(self.clicks) < len(self.landmarks):
                name, x_m, y_m = self.landmarks[len(self.clicks)]
                status = f"Click {len(self.clicks) + 1}/{len(self.landmarks)}: {name} ({x_m:.2f}, {y_m:.2f} m)"
            else:
                status = "All points selected. Press s to save; right-click undoes; r resets; q quits."
            hint = "Left-click point | Right-click undo | n skip current | r reset | s save | q quit"
        else:
            status = "Calibration saved. Compare colored overlay lines with court markings."
            hint = "Press q or Esc to close."

        cv2.putText(
            canvas,
            status,
            (10, self.display_height + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            hint,
            (10, self.display_height + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        return canvas


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def court_segments() -> list[tuple[tuple[float, float], tuple[float, float], tuple[int, int, int], int]]:
    perimeter_color = (0, 0, 255)  # red in BGR
    marking_color = (255, 255, 0)  # cyan in BGR
    segments = [
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0), perimeter_color, 3),
        ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M), perimeter_color, 3),
        ((0.0, 0.0), (0.0, COURT_LENGTH_M), perimeter_color, 3),
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M), perimeter_color, 3),
        ((0.46, 0.0), (0.46, COURT_LENGTH_M), marking_color, 2),
        ((5.64, 0.0), (5.64, COURT_LENGTH_M), marking_color, 2),
        ((0.0, 4.72), (COURT_WIDTH_M, 4.72), marking_color, 2),
        ((0.0, 8.68), (COURT_WIDTH_M, 8.68), marking_color, 2),
        ((0.0, 0.76), (COURT_WIDTH_M, 0.76), marking_color, 2),
        ((0.0, 12.64), (COURT_WIDTH_M, 12.64), marking_color, 2),
        ((3.05, 0.76), (3.05, 4.72), marking_color, 2),
        ((3.05, 8.68), (3.05, 12.64), marking_color, 2),
    ]
    return segments


def draw_court_overlay(frame: np.ndarray, court_to_pixel: np.ndarray) -> np.ndarray:
    lines = np.zeros_like(frame)
    for start, end, color, thickness in court_segments():
        projected = project_points(np.array([start, end], dtype=np.float64), court_to_pixel)
        p1 = tuple(np.rint(projected[0]).astype(int))
        p2 = tuple(np.rint(projected[1]).astype(int))
        cv2.line(lines, p1, p2, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(frame, 0.70, lines, 0.90, 0)


def save_calibration(
    frame: np.ndarray,
    clicks: list[tuple[float, float]],
    landmarks: list[tuple[str, float, float]],
    source: Path,
    frame_time_sec: float,
    output_json: Path,
    output_overlay: Path,
) -> tuple[np.ndarray, float]:
    court_points = np.array([[x, y] for _name, x, y in landmarks], dtype=np.float64)
    pixel_points = np.array(clicks, dtype=np.float64)
    court_to_pixel, _mask = cv2.findHomography(court_points, pixel_points, method=0)
    if court_to_pixel is None or not np.isfinite(court_to_pixel).all():
        raise RuntimeError("Could not solve a finite court-to-pixel homography")

    pixel_to_court = np.linalg.inv(court_to_pixel)
    projected = project_points(court_points, court_to_pixel)
    errors = np.linalg.norm(projected - pixel_points, axis=1)
    rms_error = float(np.sqrt(np.mean(errors**2)))
    overlay = draw_court_overlay(frame, court_to_pixel)

    output_overlay.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_overlay), overlay):
        raise RuntimeError(f"Could not write overlay image: {output_overlay}")

    height, width = frame.shape[:2]
    payload = {
        "source_video": str(source),
        "frame_time_sec": round(frame_time_sec, 3),
        "frame_size_px": {"width": width, "height": height},
        "court_size_m": {"width": COURT_WIDTH_M, "length": COURT_LENGTH_M},
        "coordinate_system": "x=0 left sideline to 6.1 right sideline; y=0 near baseline to 13.4 far baseline",
        "rms_reprojection_error_px": round(rms_error, 3),
        "landmarks": [
            {
                "name": name,
                "court_m": [x_m, y_m],
                "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)],
                "reprojection_error_px": round(float(error), 3),
            }
            for (name, x_m, y_m), pixel, error in zip(landmarks, pixel_points, errors)
        ],
        "court_to_pixel": np.round(court_to_pixel, 12).tolist(),
        "pixel_to_court": np.round(pixel_to_court, 12).tolist(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return overlay, rms_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the court by clicking at least four known line intersections."
    )
    parser.add_argument("input_video", type=Path, help="Stage 0 proxy video")
    parser.add_argument("output_json", type=Path, help="path for court_calibration.json")
    parser.add_argument("output_overlay", type=Path, help="path for court_overlay.png")
    parser.add_argument(
        "--time-sec", type=float, default=5.0, help="video time for the calibration frame (default: 5 seconds)"
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=[name for name, _x, _y in LANDMARKS],
        default=[],
        metavar="LANDMARK",
        help="omit an unavailable or obscured point; may be repeated (minimum four points)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    skip_names = set(args.skip)
    landmarks = [item for item in LANDMARKS if item[0] not in skip_names]
    if len(landmarks) < 4:
        parser.error("at least four landmarks must remain after --skip options")

    source = args.input_video.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_overlay = args.output_overlay.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input video does not exist: {source}")
    if args.time_sec < 0:
        parser.error("--time-sec must be zero or greater")
    if output_json == output_overlay or source in (output_json, output_overlay):
        parser.error("input and output paths must be different")
    existing = [path for path in (output_json, output_overlay) if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output already exists: " + ", ".join(str(path) for path in existing) + "; pass --overwrite"
        )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        parser.error(f"OpenCV could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time_sec * 1000.0)
    ok, frame = cap.read()
    frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    cap.release()
    if not ok:
        parser.error(f"Could not read a frame at {args.time_sec:.3f} seconds")

    actual_time_sec = frame_index / fps if fps > 0 else args.time_sec
    frame_height, frame_width = frame.shape[:2]
    display_scale = min(1.0, 1200.0 / frame_width, 760.0 / frame_height)
    display_width = max(1, round(frame_width * display_scale))
    display_height = max(1, round(frame_height * display_scale))
    ui = CalibrationUI(frame, display_width, display_height, landmarks, [])
    window_name = "Stage 2 - Court Calibration"

    print(f"Frame: {source} at {actual_time_sec:.3f}s ({frame_width}x{frame_height}).")
    print("Coordinates are viewed from behind the near baseline: x=0 is camera-left.")
    print(f"Using {len(landmarks)} landmarks; omitted: {', '.join(sorted(skip_names)) or 'none'}.")
    print("Click the named line intersections in order; right-click undoes the last point.")
    ui.print_next()
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, ui.mouse)

    try:
        while True:
            cv2.imshow(window_name, ui.render())
            key = cv2.waitKey(25) & 0xFF
            if key in (ord("q"), 27):
                print("Calibration window closed without further changes.")
                break
            if key == ord("r") and ui.saved_overlay is None:
                ui.clicks.clear()
                print("Cleared all clicked points.")
                ui.print_next()
            elif key == ord("n") and ui.saved_overlay is None:
                if len(ui.clicks) >= len(ui.landmarks):
                    print("There is no current point to skip.")
                elif len(ui.landmarks) <= 4:
                    print("At least four landmarks are required; this point cannot be skipped.")
                else:
                    skipped = ui.landmarks.pop(len(ui.clicks))
                    print(f"Skipped unavailable point: {skipped[0]} at ({skipped[1]:.2f}, {skipped[2]:.2f}) m.")
                    ui.print_next()
            elif key == ord("s") and ui.saved_overlay is None:
                if len(ui.clicks) != len(ui.landmarks):
                    print(f"Select all {len(ui.landmarks)} points before saving.")
                    continue
                overlay, rms_error = save_calibration(
                    frame,
                    ui.clicks,
                    ui.landmarks,
                    source,
                    actual_time_sec,
                    output_json,
                    output_overlay,
                )
                ui.saved_overlay = overlay
                print(f"Calibration saved: {output_json}")
                print(f"Court overlay saved: {output_overlay}")
                print(f"RMS landmark reprojection error: {rms_error:.2f} px")
                if rms_error > 12.0:
                    print("Warning: click residual is high; inspect the overlay carefully and recalibrate if needed.")
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
        print(f"Stage 2 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
