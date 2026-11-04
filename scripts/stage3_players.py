#!/usr/bin/env python3
"""Track the selected player identity and opponent from court detections."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    box: list[float]  # normalized x1, y1, x2, y2, confidence
    court_xy: tuple[float, float]
    score: float
    garment_match: float


COURT_MIDLINE_Y_M = 6.7
MIN_COLOR_MATCH = 0.08
COLOR_CHOICES = ("black", "white", "gray", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown")


def transform_pixel_to_court(
    pixel_to_court: np.ndarray, pixel_x: float, pixel_y: float
) -> tuple[float, float]:
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
    projected = cv2.perspectiveTransform(point, pixel_to_court)[0, 0]
    return float(projected[0]), float(projected[1])


def garment_color_match(frame: np.ndarray, box: list[float], garment: str, color: str) -> float:
    """Return the share of a shirt/pants crop matching a named HSV color."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    box_height = (y2 - y1) * height
    left = max(0, min(width - 1, int(round((x1 + 0.20 * (x2 - x1)) * width))))
    right = max(left + 1, min(width, int(round((x1 + 0.80 * (x2 - x1)) * width))))
    top_ratio, bottom_ratio = (0.18, 0.43) if garment == "shirt" else (0.46, 0.72)
    top = max(0, min(height - 1, int(round(y1 * height + top_ratio * box_height))))
    bottom = max(top + 1, min(height, int(round(y1 * height + bottom_ratio * box_height))))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if color == "black":
        mask = value <= 105
    elif color == "white":
        mask = (value >= 170) & (saturation <= 65)
    elif color == "gray":
        mask = (value >= 45) & (value <= 190) & (saturation <= 55)
    elif color == "brown":
        mask = (hue >= 5) & (hue <= 25) & (saturation >= 45) & (value <= 175)
    else:
        hue_ranges = {
            "red": ((0, 10), (170, 179)),
            "orange": ((11, 22),),
            "yellow": ((23, 35),),
            "green": ((36, 84),),
            "cyan": ((85, 99),),
            "blue": ((100, 129),),
            "purple": ((130, 150),),
            "pink": ((151, 169),),
        }
        mask = np.zeros(hue.shape, dtype=bool)
        for lower, upper in hue_ranges[color]:
            mask |= (hue >= lower) & (hue <= upper)
        mask &= (saturation >= 55) & (value >= 45)
    return float(np.mean(mask))


def candidates_for_frame(
    frame: np.ndarray,
    detections: list[list[float]],
    pixel_to_court: np.ndarray,
    garment: str,
    color: str,
    x_margin_m: float,
    max_box_height: float,
) -> tuple[list[Candidate], list[Candidate], int]:
    height, width = frame.shape[:2]
    near_candidates: list[Candidate] = []
    far_candidates: list[Candidate] = []
    filtered_out = 0

    for raw_box in detections:
        if len(raw_box) < 5:
            filtered_out += 1
            continue
        box = [float(value) for value in raw_box[:5]]
        x1, y1, x2, y2, score = box
        if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
            filtered_out += 1
            continue
        if y2 - y1 > max_box_height:
            filtered_out += 1
            continue

        foot_x = (x1 + x2) * 0.5 * width
        foot_y = y2 * height
        court_x, court_y = transform_pixel_to_court(pixel_to_court, foot_x, foot_y)
        if not (-x_margin_m <= court_x <= 6.1 + x_margin_m):
            filtered_out += 1
            continue
        if not (-1.2 <= court_y <= 15.5):
            filtered_out += 1
            continue

        match = garment_color_match(frame, box, garment, color)
        candidate = Candidate(box, (court_x, court_y), score, match)
        if court_y < COURT_MIDLINE_Y_M:
            near_candidates.append(candidate)
        else:
            far_candidates.append(candidate)

    near_candidates.sort(key=lambda item: (-item.garment_match, -item.score))
    far_candidates.sort(key=lambda item: item.score, reverse=True)
    return near_candidates, far_candidates, filtered_out


def is_near_court(candidate: Candidate) -> bool:
    return candidate.court_xy[1] < COURT_MIDLINE_Y_M


def choose_identity_candidate(
    candidates: list[Candidate], previous: Candidate | None
) -> Candidate | None:
    eligible = [candidate for candidate in candidates if candidate.garment_match >= MIN_COLOR_MATCH]
    if not eligible:
        return None
    if previous is None:
        return max(eligible, key=lambda candidate: (candidate.garment_match, candidate.score))
    return max(
        eligible,
        key=lambda candidate: (
            candidate.garment_match - 0.005 * min(4.0, math.dist(candidate.court_xy, previous.court_xy)),
            candidate.score,
        ),
    )


def choose_opponent_candidate(
    identity: Candidate | None,
    near_candidates: list[Candidate],
    far_candidates: list[Candidate],
) -> Candidate | None:
    if identity is None:
        return None
    candidates = far_candidates if is_near_court(identity) else near_candidates
    return max(candidates, key=lambda candidate: candidate.score) if candidates else None


def nearest_anchor(
    track: list[Candidate | None], index: int, direction: int, radius_frames: int
) -> tuple[int, Candidate] | None:
    for distance in range(1, radius_frames + 1):
        other = index + direction * distance
        if other < 0 or other >= len(track):
            break
        if track[other] is not None:
            return other, track[other]  # type: ignore[return-value]
    return None


def fill_near_gaps(
    track: list[Candidate | None],
    candidates_by_frame: list[list[Candidate]],
    fps: float,
    max_gap_seconds: float = 0.5,
    max_distance_m: float = 0.8,
    passes: int = 3,
) -> int:
    radius = max(1, round(max_gap_seconds * fps))
    filled = 0
    for _pass in range(passes):
        updates: list[tuple[int, Candidate]] = []
        for index, current in enumerate(track):
            if current is not None:
                continue
            left = nearest_anchor(track, index, -1, radius)
            right = nearest_anchor(track, index, 1, radius)
            if left is None and right is None:
                continue

            if left is not None and right is not None:
                left_index, left_candidate = left
                right_index, right_candidate = right
                fraction = (index - left_index) / (right_index - left_index)
                predicted_x = left_candidate.court_xy[0] * (1.0 - fraction) + right_candidate.court_xy[0] * fraction
                predicted_y = left_candidate.court_xy[1] * (1.0 - fraction) + right_candidate.court_xy[1] * fraction
            else:
                anchor = left if left is not None else right
                assert anchor is not None
                predicted_x, predicted_y = anchor[1].court_xy

            anchor_halves = {
                is_near_court(anchor[1]) for anchor in (left, right) if anchor is not None
            }
            if len(anchor_halves) != 1:
                continue
            expected_near = next(iter(anchor_halves))
            possible = [
                candidate for candidate in candidates_by_frame[index]
                if is_near_court(candidate) == expected_near
            ]
            if not possible:
                continue
            best = min(
                possible,
                key=lambda item: math.hypot(
                    item.court_xy[0] - predicted_x, item.court_xy[1] - predicted_y
                ),
            )
            distance = math.hypot(best.court_xy[0] - predicted_x, best.court_xy[1] - predicted_y)
            if distance <= max_distance_m:
                updates.append((index, best))

        for index, candidate in updates:
            track[index] = candidate
        filled += len(updates)
        if not updates:
            break
    return filled


def remove_single_frame_spikes(
    track: list[Candidate | None],
    spike_distance_m: float = 0.9,
    neighbor_agreement_m: float = 0.6,
) -> int:
    remove_indices: list[int] = []
    for index in range(1, len(track) - 1):
        previous, current, following = track[index - 1 : index + 2]
        if previous is None or current is None or following is None:
            continue
        neighbor_distance = math.dist(previous.court_xy, following.court_xy)
        if (
            neighbor_distance <= neighbor_agreement_m
            and math.dist(current.court_xy, previous.court_xy) > spike_distance_m
            and math.dist(current.court_xy, following.court_xy) > spike_distance_m
        ):
            remove_indices.append(index)

    for index in remove_indices:
        track[index] = None
    return len(remove_indices)


def track_record(timestamp: float, identity: Candidate | None, opponent: Candidate | None) -> dict[str, object]:
    return {
        "t": round(timestamp, 3),
        "r": [round(value, 4) for value in identity.court_xy] if identity else None,
        "rb": [round(value, 6) for value in identity.box] if identity else None,
        "o": [round(value, 4) for value in opponent.court_xy] if opponent else None,
        "ob": [round(value, 6) for value in opponent.box] if opponent else None,
    }


def draw_contact_sheet(
    preview_frames: dict[int, tuple[float, np.ndarray]],
    records: list[dict[str, object]],
    sample_indices: list[int],
    output_path: Path,
    identity_label: str,
) -> None:
    if not sample_indices:
        raise RuntimeError("No sample frames available for the tracking contact sheet")

    columns = 4
    tile_width = 480
    image_height = 270
    title_height = 62
    tile_height = image_height + title_height
    rows = math.ceil(len(sample_indices) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 32, dtype=np.uint8)

    for position, sample_index in enumerate(sample_indices):
        if sample_index not in preview_frames:
            continue
        timestamp, frame = preview_frames[sample_index]
        record = records[sample_index]
        row, column = divmod(position, columns)
        x0, y0 = column * tile_width, row * tile_height
        height, width = frame.shape[:2]
        thumbnail = cv2.resize(frame, (tile_width, image_height), interpolation=cv2.INTER_AREA)
        sheet[y0 + title_height : y0 + tile_height, x0 : x0 + tile_width] = thumbnail

        r_xy = record["r"]
        o_xy = record["o"]
        r_text = "R=missing" if r_xy is None else f"R=({r_xy[0]:.1f},{r_xy[1]:.1f})m"
        o_text = "O=missing" if o_xy is None else f"O=({o_xy[0]:.1f},{o_xy[1]:.1f})m"
        cv2.putText(sheet, f"R={identity_label} (red box); O=opponent (blue box)",
                    (x0 + 8, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            sheet,
            f"t={timestamp:.1f}s  {r_text}  {o_text}",
            (x0 + 8, y0 + 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for key, label, color in (("rb", "R", (0, 0, 255)), ("ob", "O", (255, 100, 0))):
            box = record[key]
            if box is None:
                continue
            x1, y1, x2, y2 = box[:4]
            left = x0 + round(x1 * tile_width)
            right = x0 + round(x2 * tile_width)
            top = y0 + title_height + round(y1 * image_height)
            bottom = y0 + title_height + round(y2 * image_height)
            cv2.rectangle(sheet, (left, top), (right, bottom), color, 1, cv2.LINE_AA)
            cv2.circle(sheet, ((left + right) // 2, bottom), 7, color, 2, cv2.LINE_AA)
            cv2.putText(
                sheet,
                label,
                (max(x0 + 2, (left + right) // 2 + 6), max(y0 + title_height + 16, bottom - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def swap_identity_labels(records: list[dict[str, object]]) -> None:
    for record in records:
        record["r"], record["o"] = record["o"], record["r"]
        record["rb"], record["ob"] = record["ob"], record["rb"]


def display_identity_review(image_path: Path, identity_label: str) -> str:
    """Show the tracking sheet and let the user confirm or swap R/O."""
    image = cv2.imread(str(image_path))
    if image is not None:
        try:
            cv2.namedWindow("Stage 3 identity review", cv2.WINDOW_NORMAL)
            cv2.imshow("Stage 3 identity review", image)
            print("Review the saved contact sheet; press any key in the image window to choose.")
            cv2.waitKey(0)
        except cv2.error:
            print(f"Could not open an image window; review this file instead: {image_path}")
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    while True:
        answer = input(
            f"Does red R consistently track the player wearing {identity_label}? "
            "[Y]es / [S]wap R and O / [Q]uit before saving players.json: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return "confirmed"
        if answer in {"s", "swap"}:
            return "swapped_by_review"
        if answer in {"q", "quit"}:
            raise RuntimeError("Stage 3 stopped at identity review; players.json was not saved")
        print("Enter Y, S, or Q.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identify player R by selected clothing color, then track the opponent as O."
    )
    parser.add_argument("input_video", type=Path, help="Stage 0 720p proxy video")
    parser.add_argument("detections_json", type=Path, help="dets.json from Stage 1")
    parser.add_argument("calibration_json", type=Path, help="court_calibration.json from Stage 2")
    parser.add_argument("output_json", type=Path, help="path for players.json")
    parser.add_argument("contact_sheet", type=Path, help="path for players_contact_sheet.jpg")
    parser.add_argument("--identity-garment", choices=("pant", "pants", "shirt"), help="clothing area used to identify R; prompted if omitted")
    parser.add_argument("--identity-color", choices=COLOR_CHOICES, help="color of that garment; prompted if omitted")
    parser.add_argument(
        "--x-margin-m",
        type=float,
        default=0.5,
        help="allow this much lateral margin outside the doubles sidelines (default: 0.5 m)",
    )
    parser.add_argument("--swap-identities", action="store_true", help="swap the R/O labels after inspecting the review sheet")
    parser.add_argument("--no-interactive-review", action="store_true", help="write the review image without prompting to confirm or swap R/O")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    identity_garment = args.identity_garment
    identity_color = args.identity_color
    if identity_garment is None or identity_color is None:
        if not sys.stdin.isatty():
            parser.error("provide --identity-garment and --identity-color when running without an interactive terminal")
        if identity_garment is None:
            identity_garment = input("Identify R by which garment? [pants/shirt]: ").strip().lower()
        if identity_color is None:
            identity_color = input(f"What color is the {identity_garment}? [{'/'.join(COLOR_CHOICES)}]: ").strip().lower()
    if identity_garment == "pant":
        identity_garment = "pants"
    if identity_garment not in {"pants", "shirt"}:
        parser.error("identity garment must be pants or shirt")
    if identity_color not in COLOR_CHOICES:
        parser.error(f"identity color must be one of: {', '.join(COLOR_CHOICES)}")
    identity_label = f"{identity_color} {identity_garment}"

    source = args.input_video.expanduser().resolve()
    detections_path = args.detections_json.expanduser().resolve()
    calibration_path = args.calibration_json.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    contact_sheet = args.contact_sheet.expanduser().resolve()
    for path, label in (
        (source, "proxy video"),
        (detections_path, "detections JSON"),
        (calibration_path, "calibration JSON"),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.x_margin_m < 0:
        parser.error("x margin must be zero or greater")
    outputs = (output_json, contact_sheet)
    if output_json == contact_sheet or source in outputs or detections_path in outputs or calibration_path in outputs:
        parser.error("input and output paths must be different")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output already exists: " + ", ".join(str(path) for path in existing) + "; pass --overwrite"
        )

    detection_data = json.loads(detections_path.read_text(encoding="utf-8"))
    calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))
    detection_frames = detection_data.get("frames", [])
    if not detection_frames:
        parser.error("detections JSON contains no frames")
    pixel_to_court = np.asarray(calibration_data.get("pixel_to_court"), dtype=np.float64)
    if pixel_to_court.shape != (3, 3) or not np.isfinite(pixel_to_court).all():
        parser.error("calibration JSON has no valid 3x3 pixel_to_court matrix")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        parser.error(f"OpenCV could not open proxy video: {source}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calibrated_size = calibration_data.get("frame_size_px", {})
    if calibrated_size and (
        calibrated_size.get("width") != source_width or calibrated_size.get("height") != source_height
    ):
        cap.release()
        parser.error(
            f"calibration size {calibrated_size} does not match video {source_width}x{source_height}"
        )
    if source_fps <= 0:
        cap.release()
        parser.error("Could not read a valid video frame rate")

    sample_every = int(detection_data.get("sample_every", 0))
    if sample_every < 1:
        detection_fps = float(detection_data.get("fps", 0))
        if detection_fps <= 0:
            cap.release()
            parser.error("detection JSON needs a positive fps or sample_every value")
        sample_every = max(1, round(source_fps / detection_fps))
    detection_fps = float(detection_data.get("fps", source_fps / sample_every))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_detection_frames = len(detection_frames)
    preview_count = min(12, total_detection_frames)
    preview_indices = sorted(
        {round(i * (total_detection_frames - 1) / max(1, preview_count - 1)) for i in range(preview_count)}
    )
    preview_index_set = set(preview_indices)
    preview_frames: dict[int, tuple[float, np.ndarray]] = {}
    identity_tracks: list[Candidate | None] = []
    court_candidates_by_frame: list[list[Candidate]] = []
    side_candidates_by_frame: list[tuple[list[Candidate], list[Candidate]]] = []
    direct_identity_matches = 0
    filtered_detection_count = 0
    detection_index = 0
    source_frame_index = 0
    start_time = time.monotonic()
    previous_identity: Candidate | None = None

    print(f"Input: {source}")
    print(f"Detections: {detections_path} ({total_detection_frames} sampled frames)")
    print(f"Calibration: {calibration_path} (RMS {calibration_data.get('rms_reprojection_error_px')} px)")
    print(
        f"Court filter: x in [{-args.x_margin_m:.1f}, {6.1 + args.x_margin_m:.1f}] m; "
        f"R identified by {identity_label} color in its clothing region."
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_frame_index = source_frame_index
            source_frame_index += 1
            if current_frame_index % sample_every != 0:
                continue
            if detection_index >= total_detection_frames:
                raise RuntimeError("Video has more sampled frames than dets.json")

            detection_record = detection_frames[detection_index]
            timestamp = float(detection_record["t"])
            expected_timestamp = current_frame_index / source_fps
            if abs(timestamp - expected_timestamp) > max(0.03, 0.5 / source_fps):
                raise RuntimeError(
                    f"Timestamp mismatch at detection {detection_index}: "
                    f"JSON {timestamp:.3f}s vs video {expected_timestamp:.3f}s"
                )

            near_candidates, far_candidates, filtered_count = candidates_for_frame(
                frame,
                detection_record.get("p", []),
                pixel_to_court,
                identity_garment,
                identity_color,
                args.x_margin_m,
                0.55,
            )
            filtered_detection_count += filtered_count
            candidates = near_candidates + far_candidates
            identity_candidate = choose_identity_candidate(candidates, previous_identity)
            if identity_candidate is not None:
                direct_identity_matches += 1
                previous_identity = identity_candidate
            identity_tracks.append(identity_candidate)
            court_candidates_by_frame.append(candidates)
            side_candidates_by_frame.append((near_candidates, far_candidates))
            if detection_index in preview_index_set:
                preview_frames[detection_index] = (timestamp, frame.copy())

            detection_index += 1
            if detection_index % 1000 == 0:
                elapsed = time.monotonic() - start_time
                print(f"Mapped {detection_index}/{total_detection_frames} frames ({elapsed:.0f}s).", flush=True)
    finally:
        cap.release()

    if detection_index != total_detection_frames:
        raise RuntimeError(
            f"Video ended after {detection_index} sampled frames; dets.json has {total_detection_frames}"
        )

    filled_count = fill_near_gaps(identity_tracks, court_candidates_by_frame, detection_fps)
    identity_spikes = remove_single_frame_spikes(identity_tracks)
    opponent_tracks: list[Candidate | None] = []
    identity_anchor_radius = max(1, round(0.5 * detection_fps))
    for index, (identity, (near, far)) in enumerate(zip(identity_tracks, side_candidates_by_frame)):
        reference = identity
        if reference is None:
            anchors = [
                anchor for anchor in (
                    nearest_anchor(identity_tracks, index, -1, identity_anchor_radius),
                    nearest_anchor(identity_tracks, index, 1, identity_anchor_radius),
                ) if anchor is not None
            ]
            if anchors:
                reference = min(anchors, key=lambda anchor: abs(anchor[0] - index))[1]
        opponent_tracks.append(choose_opponent_candidate(reference, near, far))
    opponent_spikes = remove_single_frame_spikes(opponent_tracks)
    records = [
        track_record(float(frame["t"]), identity_tracks[index], opponent_tracks[index])
        for index, frame in enumerate(detection_frames)
    ]

    review_action = "swapped_by_flag" if args.swap_identities else "pending"
    display_identity_label = "manually selected player" if args.swap_identities else identity_label
    if args.swap_identities:
        swap_identity_labels(records)
    draw_contact_sheet(preview_frames, records, preview_indices, contact_sheet, display_identity_label)
    if sys.stdin.isatty() and not args.no_interactive_review and not args.swap_identities:
        review_action = display_identity_review(contact_sheet, identity_label)
        if review_action == "swapped_by_review":
            swap_identity_labels(records)
            display_identity_label = "manually selected player"
            draw_contact_sheet(preview_frames, records, preview_indices, contact_sheet, display_identity_label)
    elif review_action == "pending":
        review_action = "image_only_review_required"
        print(f"Review the R/O assignments in {contact_sheet} before continuing.")

    payload = {
        "fps": round(detection_fps, 6),
        "source_fps": round(source_fps, 6),
        "sample_every": sample_every,
        "coordinate_system": calibration_data.get("coordinate_system"),
        "player_identity": {
            "R": (
                "manually selected player after swapping the color-based assignment"
                if review_action.startswith("swapped") else
                f"player selected by {identity_label} color; can occupy either court half"
            ),
            "O": "the other player in the opposite court half",
            "garment": identity_garment,
            "color": identity_color,
            "minimum_color_match_fraction": MIN_COLOR_MATCH,
            "review_action": review_action,
            "review_image": contact_sheet.name,
        },
        "frames": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
        file.write("\n")
    final_identity = sum(track is not None for track in identity_tracks)
    final_opponent = sum(track is not None for track in opponent_tracks)
    elapsed = time.monotonic() - start_time
    print(f"Player assignment complete: {total_detection_frames} frames at {detection_fps:.2f} fps.")
    print(
        f"R ({identity_label}): {direct_identity_matches} direct matches, {filled_count} gap fills, "
        f"{identity_spikes} spikes removed, {final_identity} positions retained."
    )
    print(f"O: {opponent_spikes} spikes removed, {final_opponent} positions retained.")
    print(f"Out-of-region/invalid detections filtered: {filtered_detection_count}.")
    print(f"Players JSON: {output_json}")
    print(f"Contact sheet: {contact_sheet}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, cv2.error, np.linalg.LinAlgError, json.JSONDecodeError) as error:
        print(f"Stage 3 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
