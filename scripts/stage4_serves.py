#!/usr/bin/env python3
"""Find stable pre-serve windows, guess server/court side, and make review sheets."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shuttle_tracking import (
    is_main_court_candidate,
    load_shuttle_tracks,
    point_box_distance,
    samples_between,
    validate_shuttle_video,
)


NET_Y_M = 6.7
COURT_CENTER_X_M = 3.05
MIN_SERVER_CUE_ADVANTAGE_M = 0.25
DEFAULT_NEAR_STILL_SPEED_M_S = 0.7
DEFAULT_FAR_STILL_SPEED_M_S = 0.9


@dataclass
class ServeWindow:
    start_index: int
    end_index: int
    server: str | None
    side: str | None
    confidence: float | None
    decisive_votes: int


def read_position(frame: dict, key: str) -> tuple[float, float] | None:
    value = frame.get(key)
    if not isinstance(value, list) or len(value) != 2:
        return None
    x, y = float(value[0]), float(value[1])
    if not np.isfinite([x, y]).all():
        return None
    return x, y


def smooth_positions(
    positions: list[tuple[float, float] | None], window_frames: int
) -> list[tuple[float, float] | None]:
    if window_frames % 2 == 0:
        window_frames += 1
    radius = window_frames // 2
    smoothed: list[tuple[float, float] | None] = []
    for index in range(len(positions)):
        start = max(0, index - radius)
        end = min(len(positions), index + radius + 1)
        values = [position for position in positions[start:end] if position is not None]
        if len(values) < max(1, math.ceil(window_frames / 2)):
            smoothed.append(None)
            continue
        xy = np.asarray(values, dtype=np.float64)
        med = np.median(xy, axis=0)
        smoothed.append((float(med[0]), float(med[1])))
    return smoothed


def position_speeds(
    positions: list[tuple[float, float] | None], timestamps: list[float]
) -> list[float | None]:
    speeds: list[float | None] = [None] * len(positions)
    for index in range(len(positions)):
        previous = index - 1 if index > 0 else None
        following = index + 1 if index + 1 < len(positions) else None
        if previous is not None and following is not None:
            a, b = positions[previous], positions[following]
            dt = timestamps[following] - timestamps[previous]
        elif following is not None:
            a, b = positions[index], positions[following]
            dt = timestamps[following] - timestamps[index]
        elif previous is not None:
            a, b = positions[previous], positions[index]
            dt = timestamps[index] - timestamps[previous]
        else:
            continue
        if a is None or b is None or dt <= 0:
            continue
        speeds[index] = math.dist(a, b) / dt
    return speeds


def in_serve_area(
    positions: dict[str, tuple[float, float] | None],
    speeds: dict[str, float | None],
    near_still_speed_m_s: float = DEFAULT_NEAR_STILL_SPEED_M_S,
    far_still_speed_m_s: float = DEFAULT_FAR_STILL_SPEED_M_S,
) -> bool:
    present = [(player, point) for player, point in positions.items() if point is not None]
    if len(present) != 2 or any(speeds.get(player) is None for player, _ in present):
        return False
    near = [(player, point) for player, point in present if point[1] < NET_Y_M]
    far = [(player, point) for player, point in present if point[1] >= NET_Y_M]
    if len(near) != 1 or len(far) != 1:
        return False
    near_player, near_point = near[0]
    far_player, far_point = far[0]
    rx, ry = near_point
    ox, oy = far_point
    return (
        0.2 <= rx <= 5.9
        and 1.0 <= ry <= 4.9
        and 0.2 <= ox <= 5.9
        and 8.5 <= oy <= 12.4
        and speeds[near_player] < near_still_speed_m_s
        and speeds[far_player] < far_still_speed_m_s
    )


def contiguous_windows(mask: list[bool], timestamps: list[float], minimum_duration: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index - 1
            if timestamps[end] - timestamps[start] >= minimum_duration:
                runs.append((start, end))
            start = None
    return runs


def guess_server_and_side(
    start_index: int,
    end_index: int,
    player_positions: dict[str, list[tuple[float, float] | None]],
    minimum_confidence: float = 0.6,
) -> tuple[str | None, str | None, float | None, int]:
    votes: list[str] = []
    for index in range(start_index, end_index + 1):
        r_position = player_positions["R"][index]
        o_position = player_positions["O"][index]
        if r_position is None or o_position is None:
            continue
        r_distance_to_net = abs(NET_Y_M - r_position[1])
        o_distance_to_net = abs(NET_Y_M - o_position[1])
        difference = abs(r_distance_to_net - o_distance_to_net)
        if difference < MIN_SERVER_CUE_ADVANTAGE_M:
            continue
        votes.append("R" if r_distance_to_net < o_distance_to_net else "O")

    if not votes:
        return None, None, None, 0
    r_votes = votes.count("R")
    o_votes = votes.count("O")
    server = "R" if r_votes >= o_votes else "O"
    confidence = max(r_votes, o_votes) / len(votes)
    if confidence < minimum_confidence:
        return None, None, confidence, len(votes)

    server_positions = player_positions[server]
    xs = [server_positions[i][0] for i in range(start_index, end_index + 1) if server_positions[i] is not None]
    ys = [server_positions[i][1] for i in range(start_index, end_index + 1) if server_positions[i] is not None]
    if not xs:
        return None, None, confidence, len(votes)
    median_x = float(np.median(xs))
    is_far_player = float(np.median(ys)) >= NET_Y_M
    # On the far half the player's right service court appears on camera-left.
    on_right = median_x < COURT_CENTER_X_M if is_far_player else median_x >= COURT_CENTER_X_M
    side = "right" if on_right else "left"
    return server, side, confidence, len(votes)


def find_and_merge_windows(
    mask: list[bool],
    timestamps: list[float],
    player_positions: dict[str, list[tuple[float, float] | None]],
    minimum_duration: float,
    merge_gap_seconds: float,
) -> list[list[ServeWindow]]:
    raw_windows: list[ServeWindow] = []
    for start, end in contiguous_windows(mask, timestamps, minimum_duration):
        server, side, confidence, votes = guess_server_and_side(
            start, end, player_positions
        )
        raw_windows.append(ServeWindow(start, end, server, side, confidence, votes))

    merged: list[list[ServeWindow]] = []
    for window in raw_windows:
        if merged:
            previous = merged[-1][-1]
            gap = timestamps[window.start_index] - timestamps[previous.end_index]
            same_identity = (
                window.server is not None
                and window.side is not None
                and window.server == previous.server
                and window.side == previous.side
            )
            if same_identity and gap <= merge_gap_seconds:
                merged[-1].append(window)
                continue
        merged.append([window])
    return merged


def median_position(
    positions: list[tuple[float, float] | None], start: int, end: int
) -> list[float] | None:
    values = [positions[i] for i in range(start, end + 1) if positions[i] is not None]
    if not values:
        return None
    xy = np.median(np.asarray(values, dtype=np.float64), axis=0)
    return [round(float(xy[0]), 4), round(float(xy[1]), 4)]


def event_record(
    number: int,
    windows: list[ServeWindow],
    timestamps: list[float],
    player_positions: dict[str, list[tuple[float, float] | None]],
) -> dict[str, object]:
    first, last = windows[0], windows[-1]
    all_votes = sum(window.decisive_votes for window in windows)
    weighted_confidence = (
        sum((window.confidence or 0.0) * window.decisive_votes for window in windows) / all_votes
        if all_votes
        else None
    )
    review_start, review_end = last.start_index, last.end_index
    review_indices = sorted(
        {
            review_start,
            (review_start + review_end) // 2,
            review_end,
        }
    )
    return {
        "n": number,
        "candidate_n": number,
        # The estimate is the last still frame before movement resumes.
        "t": round(timestamps[last.end_index], 3),
        "start_t": round(timestamps[first.start_index], 3),
        "end_t": round(timestamps[last.end_index], 3),
        "duration": round(timestamps[last.end_index] - timestamps[first.start_index], 3),
        "server": last.server,
        "receiver": "O" if last.server == "R" else "R" if last.server == "O" else None,
        "side": last.side,
        "heuristic_agreement": round(weighted_confidence, 3) if weighted_confidence is not None else None,
        "server_hint_method": "first-pass net-distance estimate; verify by eye",
        "net_server_guess": last.server,
        "net_heuristic_agreement": round(weighted_confidence, 3) if weighted_confidence is not None else None,
        "net_vote_count": all_votes,
        "centerline_server_guess": None,
        "centerline_agreement": None,
        "centerline_vote_count": 0,
        "position_server_guess": last.server,
        "position_side_guess": last.side,
        "r": median_position(player_positions["R"], review_start, review_end),
        "o": median_position(player_positions["O"], review_start, review_end),
        "windows": [
            {
                "start_t": round(timestamps[window.start_index], 3),
                "end_t": round(timestamps[window.end_index], 3),
            }
            for window in windows
        ],
        "review_times": [round(timestamps[index], 3) for index in review_indices],
        "_review_indices": review_indices,
    }


def player_crop(frame: np.ndarray, box: object, scale: float = 2.2) -> np.ndarray | None:
    """Return a padded square around a normalized player box for visual review."""
    if not isinstance(box, list) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None

    height, width = frame.shape[:2]
    box_width, box_height = (x2 - x1) * width, (y2 - y1) * height
    side = max(32, int(math.ceil(max(box_width, box_height) * scale)))
    center_x, center_y = (x1 + x2) * width / 2, (y1 + y2) * height / 2
    origin_x, origin_y = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    source_x0, source_y0 = max(0, origin_x), max(0, origin_y)
    source_x1, source_y1 = min(width, origin_x + side), min(height, origin_y + side)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return None
    target_x0, target_y0 = source_x0 - origin_x, source_y0 - origin_y
    crop[target_y0:target_y0 + source_y1 - source_y0,
         target_x0:target_x0 + source_x1 - source_x0] = frame[source_y0:source_y1, source_x0:source_x1]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA if side > 256 else cv2.INTER_LINEAR)


def side_from_position(server: str, position: list[float] | None) -> str | None:
    if not isinstance(position, list) or len(position) != 2:
        return None
    x, y = float(position[0]), float(position[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    is_far_player = y >= NET_Y_M
    # A far-side player's right service court appears on camera-left.
    on_right = x < COURT_CENTER_X_M if is_far_player else x >= COURT_CENTER_X_M
    return "right" if on_right else "left"


def refine_server_guesses_with_centerline(
    events: list[dict[str, object]],
    groups: list[list[ServeWindow]],
    player_positions: dict[str, list[tuple[float, float] | None]],
    minimum_confidence: float = 0.6,
) -> None:
    """Refine already-merged candidates using center-line proximity only."""
    for event, windows in zip(events, groups):
        votes: list[str] = []
        for window in windows:
            for index in range(window.start_index, window.end_index + 1):
                r_position = player_positions["R"][index]
                o_position = player_positions["O"][index]
                if r_position is None or o_position is None:
                    continue
                r_offset = abs(r_position[0] - COURT_CENTER_X_M)
                o_offset = abs(o_position[0] - COURT_CENTER_X_M)
                difference = abs(r_offset - o_offset)
                if difference < MIN_SERVER_CUE_ADVANTAGE_M:
                    continue
                votes.append("R" if r_offset < o_offset else "O")

        if votes:
            r_votes, o_votes = votes.count("R"), votes.count("O")
            confidence = max(r_votes, o_votes) / len(votes)
            centerline_guess = "R" if r_votes >= o_votes else "O"
        else:
            confidence = None
            centerline_guess = None

        event["centerline_server_guess"] = (
            centerline_guess if confidence is not None and confidence >= minimum_confidence else None
        )
        event["centerline_agreement"] = round(confidence, 3) if confidence is not None else None
        event["centerline_vote_count"] = len(votes)

        if event["centerline_server_guess"] is None:
            event["server_hint_method"] = (
                "first-pass net-distance estimate retained; center-line second pass was inconclusive"
            )
            continue

        refined_server = str(event["centerline_server_guess"])
        refined_side = side_from_position(refined_server, event.get("r" if refined_server == "R" else "o"))
        net_guess = event.get("net_server_guess")
        if net_guess == refined_server:
            method = "center-line second pass corroborated the first-pass net-distance estimate"
        else:
            method = "center-line second pass refined the first-pass net-distance estimate"
        event["server"] = refined_server
        event["receiver"] = "O" if refined_server == "R" else "R"
        event["side"] = refined_side
        event["heuristic_agreement"] = round(confidence, 3) if confidence is not None else None
        event["position_server_guess"] = refined_server
        event["position_side_guess"] = refined_side
        event["server_hint_method"] = method


def write_racket_review_template(review_dir: Path, events: list[dict[str, object]]) -> Path:
    """Write editable labels for the actual racket orientation at each candidate."""
    template_path = review_dir / "racket_orientation_review.template.json"
    template = {
        "instructions": (
            "For each candidate, label the visible racket itself: up, sideways, down, or unclear. "
            "Use R for the player selected in Stage 3 and O for the opponent, regardless of court half. "
            "A clear up vs sideways/down contrast "
            "identifies the up-racket player as receiver."
        ),
        "events": [
            {"candidate_n": event["candidate_n"], "t": event["t"], "R": None, "O": None}
            for event in events
        ],
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return template_path


def apply_racket_orientation_review(
    events: list[dict[str, object]], review_path: Path | None
) -> int:
    """Apply user-reviewed racket directions; ambiguous pairs keep the position estimate."""
    if review_path is None:
        return 0
    data = json.loads(review_path.read_text(encoding="utf-8"))
    rows = data.get("events")
    if not isinstance(rows, list):
        raise ValueError("racket review JSON must contain an events array")

    by_candidate = {int(event["candidate_n"]): event for event in events}
    seen: set[int] = set()
    applied = 0
    allowed = {"up", "sideways", "down", "unclear"}
    for row in rows:
        if not isinstance(row, dict) or "candidate_n" not in row:
            raise ValueError("each racket review row needs candidate_n, R, and O")
        candidate_n = int(row["candidate_n"])
        if candidate_n in seen:
            raise ValueError(f"candidate {candidate_n} appears more than once in racket review JSON")
        if candidate_n not in by_candidate:
            raise ValueError(f"racket review refers to unknown candidate {candidate_n}")
        seen.add(candidate_n)
        orientation = {label: row.get(label) or "unclear" for label in ("R", "O")}
        for label, value in orientation.items():
            if not isinstance(value, str) or value.lower() not in allowed:
                raise ValueError(
                    f"candidate {candidate_n} {label} orientation must be up, sideways, down, or unclear"
                )
            orientation[label] = value.lower()

        event = by_candidate[candidate_n]
        record: dict[str, object] = {**orientation, "status": "ambiguous"}
        server: str | None = None
        if orientation["R"] == "up" and orientation["O"] in {"sideways", "down"}:
            server = "O"
        elif orientation["O"] == "up" and orientation["R"] in {"sideways", "down"}:
            server = "R"

        if server is not None:
            side = side_from_position(server, event.get("r" if server == "R" else "o"))
            if side is not None:
                event["server"] = server
                event["receiver"] = "O" if server == "R" else "R"
                event["side"] = side
                event["server_hint_method"] = (
                    f"racket orientation review: {server} racket sideways/down, opponent racket up; "
                    "side from server court position"
                )
                record["status"] = "clear_orientation_contrast"
                record["server"] = server
                record["side"] = side
                event["_racket_review_used"] = True
                applied += 1
            else:
                record["status"] = "server_side_unavailable"
                event["_racket_review_used"] = False
        else:
            event["_racket_review_used"] = False
        event["racket_orientation_review"] = record

    return applied


def get_video_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def overlay_player_box(
    image: np.ndarray, box: list[float] | None, label: str, color: tuple[int, int, int]
) -> None:
    if box is None:
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box[:4]
    left, right = round(x1 * width), round(x2 * width)
    top, bottom = round(y1 * height), round(y2 * height)
    cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
    cv2.circle(image, ((left + right) // 2, bottom), 8, color, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (max(4, left), max(18, top - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def make_review_pages(
    cap: cv2.VideoCapture,
    events: list[dict[str, object]],
    player_frames: list[dict],
    source_fps: float,
    review_dir: Path,
) -> int:
    if not events:
        return 0
    review_dir.mkdir(parents=True, exist_ok=True)
    frame_cache: dict[int, np.ndarray | None] = {}
    events_per_page = 4
    tile_width, tile_height = 320, 180
    crop_width, crop_height = tile_width // 2, tile_width // 2
    title_height, event_header_height = 25, 34
    event_height = event_header_height + title_height + tile_height + crop_height

    def frame_at(index: int) -> np.ndarray | None:
        if index not in frame_cache:
            seconds = float(player_frames[index]["t"])
            frame_index = round(seconds * source_fps)
            frame_cache[index] = get_video_frame(cap, frame_index)
        return frame_cache[index]

    page_count = math.ceil(len(events) / events_per_page)
    for page_index in range(page_count):
        page_events = events[page_index * events_per_page : (page_index + 1) * events_per_page]
        page_height = len(page_events) * event_height
        sheet = np.full((page_height, tile_width * 3, 3), 32, dtype=np.uint8)
        for row, event in enumerate(page_events):
            y0 = row * event_height
            server = event["server"] or "?"
            side = event["side"] or "?"
            confidence = event["heuristic_agreement"]
            confidence_text = "?" if confidence is None else f"{confidence:.2f}"
            cue = event.get("racket_orientation_review")
            cue_text = (
                f"  racket R={cue['R']} O={cue['O']} ({cue['status']})"
                if isinstance(cue, dict) else ""
            )
            header = (
                f"Candidate {event['candidate_n']}  guess={server}/{side}  agreement={confidence_text}  "
                f"t~{event['t']:.1f}s{cue_text}"
            )
            cv2.putText(
                sheet,
                header,
                (8, y0 + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            sample_indices = event["_review_indices"]
            for column in range(3):
                sample_index = sample_indices[min(column, len(sample_indices) - 1)]
                timestamp = float(player_frames[sample_index]["t"])
                image_y = y0 + event_header_height + title_height
                x0 = column * tile_width
                frame = frame_at(sample_index)
                if frame is None:
                    cv2.putText(sheet, "Frame unavailable", (x0 + 8, image_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    continue
                thumb = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                player = player_frames[sample_index]
                r_box = player.get("rb")
                o_box = player.get("ob")
                overlay_player_box(thumb, r_box, "R", (0, 0, 255))
                overlay_player_box(thumb, o_box, "O", (255, 100, 0))
                sheet[image_y : image_y + tile_height, x0 : x0 + tile_width] = thumb
                crop_y = image_y + tile_height
                for player_box, label, color, panel_x in (
                    (r_box, "R", (0, 0, 255), x0),
                    (o_box, "O", (255, 100, 0), x0 + crop_width),
                ):
                    closeup = player_crop(frame, player_box, scale=2.5)
                    if closeup is None:
                        closeup = np.full((crop_height, crop_width, 3), 48, dtype=np.uint8)
                    else:
                        closeup = cv2.resize(closeup, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(closeup, (0, 0), (crop_width - 1, 22), (32, 32, 32), -1)
                    cv2.putText(closeup, f"{label} racket close-up", (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                                0.38, (255, 255, 255), 1, cv2.LINE_AA)
                    sheet[crop_y : crop_y + crop_height, panel_x : panel_x + crop_width] = closeup
                cv2.putText(
                    sheet,
                    f"t={timestamp:.1f}s",
                    (x0 + 7, y0 + event_header_height + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        output = review_dir / f"serves_page_{page_index + 1:02d}.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise RuntimeError(f"Could not write serve review sheet: {output}")
    return page_count


def parse_review_reply(reply: str) -> tuple[str, str] | None | bool:
    """Return (server, side), False for not a serve, or None for keep the guess."""
    value = reply.strip().lower().replace(" ", "")
    if not value or value in {"k", "keep", "yes", "y"}:
        return None
    if value in {"x", "n", "no", "notaserve", "not-serve", "skip"}:
        return False
    value = value.replace("rigth", "right").replace("rigt", "right")
    parts = value.replace("-", "/").split("/")
    if len(parts) != 2:
        raise ValueError("enter R/left, R/right, O/left, O/right, x, or press Enter to keep")
    server, side = parts
    server = {"r": "R", "o": "O"}.get(server)
    side = {"l": "left", "left": "left", "r": "right", "right": "right"}.get(side)
    if server is None or side is None:
        raise ValueError("enter R/left, R/right, O/left, O/right, x, or press Enter to keep")
    return server, side


def review_candidates_interactively(
    events: list[dict[str, object]], review_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    review_dir.mkdir(parents=True, exist_ok=True)

    print("Review each candidate against the indicated row in the contact sheets.")
    print("Enter keeps the guess; enter R/left, R/right, O/left, O/right to correct; x rejects it as not a serve.")
    for event in events:
        candidate_n = int(event["candidate_n"])
        page = (candidate_n - 1) // 4 + 1
        row = (candidate_n - 1) % 4 + 1
        current_guess = f"{event['server'] or '?'} / {event['side'] or '?'}"
        prompt = (
            f"Candidate {candidate_n} at {event['t']:.1f}s, guess {current_guess}; "
            f"review serves_page_{page:02d}.jpg row {row} [Enter=keep, x=not a serve]: "
        )
        while True:
            try:
                reply = input(prompt)
                parsed = parse_review_reply(reply)
            except ValueError as error:
                print(error)
                continue

            if parsed is False:
                rejected_event = {key: value for key, value in event.items() if not key.startswith("_")}
                rejected_event["review_status"] = "user_rejected_not_a_serve"
                rejected.append(rejected_event)
                decisions.append({"candidate_n": candidate_n, "decision": "not_a_serve"})
                break

            if parsed is None:
                if event["server"] not in ("R", "O") or event["side"] not in ("left", "right"):
                    print("This candidate has no complete guess; enter R/left, R/right, O/left, O/right, or x.")
                    continue
                event["review_status"] = "user_confirmed_guess"
                event["server_source"] = "user_confirmed"
                decisions.append(
                    {
                        "candidate_n": candidate_n,
                        "decision": "keep",
                        "server": event["server"],
                        "side": event["side"],
                    }
                )
            else:
                server, side = parsed
                event["original_guess"] = {"server": event["server"], "side": event["side"]}
                event["server"] = server
                event["side"] = side
                event["review_status"] = "user_corrected"
                event["server_source"] = "user_input"
                decisions.append(
                    {
                        "candidate_n": candidate_n,
                        "decision": "corrected",
                        "server": server,
                        "side": side,
                    }
                )

            event["n"] = len(accepted) + 1
            accepted.append(event)
            break

    correction_log = {
        "reviewed_interactively": True,
        "decisions": decisions,
    }
    with (review_dir / "serve_corrections.json").open("w", encoding="utf-8") as file:
        json.dump(correction_log, file, indent=2)
        file.write("\n")
    return accepted, rejected, decisions


def manual_score_review_draft(events: list[dict[str, object]]) -> dict[str, object]:
    """Create a Stage 5 review draft covering every accepted Stage 4 candidate."""
    score = {"R": 0, "O": 0}
    rows: list[dict[str, object]] = []
    for index, event in enumerate(events):
        rows.append(
            {
                "n": index + 1,
                "candidate_n": event["candidate_n"],
                "server": event.get("server"),
                "side": event.get("side"),
                "score_before": dict(score),
            }
        )
        if index + 1 < len(events):
            next_server = events[index + 1].get("server")
            if next_server in ("R", "O"):
                # The next serve's server is a provisional estimate of this rally's winner.
                score[next_server] += 1

    return {
        "score_order": "R-O",
        "initial_score": {"R": 0, "O": 0},
        "serves": rows,
        "final_point_winner": None,
        "final_score": None,
        "notes": [
            "Stage 4 draft: verify every server, side, and score; fill in the final point result and add any missed points."
        ],
    }


def add_tracknet_server_cues(
    events: list[dict[str, Any]], player_frames: list[dict[str, Any]], tracks: dict[str, Any]
) -> None:
    """Attach a low-confidence launch-origin hint without changing the server guess."""
    times = [float(row["t"]) for row in player_frames]
    width = int(tracks.get("frame_size_px", {}).get("width", 0))
    height = int(tracks.get("frame_size_px", {}).get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("shuttle tracks need positive frame_size_px dimensions")

    for event in events:
        anchor = float(event.get("end_t", event["t"]))
        candidates = [
            row for row in samples_between(tracks, anchor, anchor + 0.65)
            if row.get("visible") and is_main_court_candidate(row)
        ][:5]
        if not candidates:
            event["tracknet_serve_cue"] = {"status": "no_visible_main_court_track"}
            continue

        distances: dict[str, list[float]] = {"R": [], "O": []}
        for row in candidates:
            timestamp = float(row["t"])
            frame_index = bisect.bisect_left(times, timestamp)
            if frame_index >= len(times):
                frame_index = len(times) - 1
            if frame_index > 0 and timestamp - times[frame_index - 1] <= times[frame_index] - timestamp:
                frame_index -= 1
            player_frame = player_frames[frame_index]
            x = float(row["x_px"]) / width
            y = float(row["y_px"]) / height
            for player, box_key in (("R", "rb"), ("O", "ob")):
                distance = point_box_distance(x, y, player_frame.get(box_key))
                if distance is not None:
                    distances[player].append(distance)

        median_distance = {
            player: float(np.median(values)) if values else None
            for player, values in distances.items()
        }
        available = [(value, player) for player, value in median_distance.items() if value is not None]
        available.sort()
        if len(available) == 2:
            gap = available[1][0] - available[0][0]
            if available[0][0] <= 0.16 and gap >= 0.05:
                hint, status = available[0][1], "near_player_launch_hint"
            else:
                hint, status = None, "ambiguous_origin"
        else:
            gap = None
            hint, status = None, "player_box_unavailable"
        event["tracknet_serve_cue"] = {
            "status": status,
            "server_hint": hint,
            "confidence": round(min(1.0, gap / 0.15), 3) if hint and gap is not None else None,
            "confidence_basis": "normalized gap between the shuttle point's median distances to player boxes; heuristic score, not a calibrated probability",
            "first_track_t": round(float(candidates[0]["t"]), 3),
            "first_track_px": [round(float(candidates[0]["x_px"]), 1), round(float(candidates[0]["y_px"]), 1)],
            "candidate_points": len(candidates),
            "median_distance_to_player_boxes": {
                player: round(value, 4) if value is not None else None
                for player, value in median_distance.items()
            },
            "interpretation": "Image-space launch-origin cue; Stage 4 applies it only when explicitly enabled and confidence meets the configured threshold.",
        }


def refine_server_guesses_with_tracknet(
    events: list[dict[str, Any]], minimum_confidence: float = 0.6
) -> int:
    """Apply confident TrackNet origin cues after the position and center-line passes."""
    applied = 0
    for event in events:
        cue = event.get("tracknet_serve_cue")
        hint = cue.get("server_hint") if isinstance(cue, dict) else None
        confidence = cue.get("confidence") if isinstance(cue, dict) else None
        old_server = event.get("server")
        refinement: dict[str, Any] = {
            "status": "no_confident_launch_hint",
            "previous_server": old_server,
            "minimum_confidence": minimum_confidence,
        }
        if hint not in ("R", "O") or not isinstance(confidence, (int, float)):
            event["tracknet_server_refinement"] = refinement
            continue
        refinement["confidence"] = round(float(confidence), 3)
        refinement["server_hint"] = hint
        refinement["receiver_hint"] = "O" if hint == "R" else "R"
        if float(confidence) < minimum_confidence:
            refinement["status"] = "below_confidence_threshold"
            event["tracknet_server_refinement"] = refinement
            continue
        position = event.get("r" if hint == "R" else "o")
        side = side_from_position(hint, position)
        if side is None:
            refinement["status"] = "server_side_unavailable"
            event["tracknet_server_refinement"] = refinement
            continue

        event["server"] = hint
        event["receiver"] = "O" if hint == "R" else "R"
        event["side"] = side
        event["server_hint_method"] = (
            f"TrackNet launch-origin second pass after center-line analysis; "
            f"confidence {float(confidence):.2f} met threshold {minimum_confidence:.2f}; "
            "side from server court position"
        )
        refinement.update({
            "status": "applied",
            "server": hint,
            "receiver": event["receiver"],
            "side": side,
        })
        event["tracknet_server_refinement"] = refinement
        applied += 1
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find serve-position windows and export serve candidates for manual review."
    )
    parser.add_argument("input_video", type=Path, help="Stage 0 proxy video")
    parser.add_argument("players_json", type=Path, help="players.json from Stage 3")
    parser.add_argument("output_json", type=Path, help="path for serves.json")
    parser.add_argument("review_dir", type=Path, help="directory for serve review contact sheets")
    parser.add_argument("--min-duration", type=float, default=0.8, help="minimum still window in seconds")
    parser.add_argument("--merge-gap", type=float, default=6.0, help="merge same server/side windows within seconds")
    parser.add_argument(
        "--still-threshold", type=float,
        help="shared maximum speed in m/s for near- and far-side players (overrides their defaults)",
    )
    parser.add_argument(
        "--near-still-threshold", type=float,
        help="near-side maximum speed in m/s (default: 0.7)",
    )
    parser.add_argument(
        "--far-still-threshold", type=float,
        help="far-side maximum speed in m/s (default: 0.9)",
    )
    parser.add_argument(
        "--racket-review-json", type=Path,
        help="optional JSON with reviewed R/O racket orientations (up, sideways, down, unclear)",
    )
    parser.add_argument(
        "--shuttle-tracks", type=Path,
        help="optional Stage 3A raw TrackNet tracks; enables a launch-origin cue",
    )
    parser.add_argument(
        "--use-shuttle-server-cue", action="store_true",
        help="after center-line analysis, refine server/receiver from a confident TrackNet launch-origin cue",
    )
    parser.add_argument(
        "--shuttle-server-min-confidence", type=float, default=0.6,
        help="minimum TrackNet server-cue heuristic score required for refinement (default: 0.6)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    review_mode = parser.add_mutually_exclusive_group()
    review_mode.add_argument(
        "--interactive-review", dest="interactive_review", action="store_true", help="prompt to confirm/correct every candidate"
    )
    review_mode.add_argument(
        "--no-interactive-review", dest="interactive_review", action="store_false", help="leave guesses unreviewed"
    )
    parser.set_defaults(interactive_review=None)
    args = parser.parse_args()

    video_path = args.input_video.expanduser().resolve()
    players_path = args.players_json.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    manual_review_json = output_json.parent / "manual_score_review.json"
    reviewed_serves_json = output_json.parent / "serves_reviewed.json"
    review_dir = args.review_dir.expanduser().resolve()
    racket_review_json = args.racket_review_json.expanduser().resolve() if args.racket_review_json else None
    shuttle_path = args.shuttle_tracks.expanduser().resolve() if args.shuttle_tracks else None
    if not video_path.is_file() or not players_path.is_file():
        parser.error("input video and players JSON must both exist")
    if racket_review_json is not None and not racket_review_json.is_file():
        parser.error(f"racket orientation review JSON does not exist: {racket_review_json}")
    if shuttle_path is not None and not shuttle_path.is_file():
        parser.error(f"TrackNet shuttle JSON does not exist: {shuttle_path}")
    if args.use_shuttle_server_cue and shuttle_path is None:
        parser.error("--use-shuttle-server-cue requires --shuttle-tracks")
    if not 0.0 <= args.shuttle_server_min_confidence <= 1.0:
        parser.error("--shuttle-server-min-confidence must be between 0 and 1")
    if args.min_duration <= 0 or args.merge_gap < 0:
        parser.error("minimum duration must be positive and merge gap must be zero or greater")
    if args.still_threshold is not None and args.still_threshold <= 0:
        parser.error("still threshold must be greater than zero")
    if args.near_still_threshold is not None and args.near_still_threshold <= 0:
        parser.error("near still threshold must be greater than zero")
    if args.far_still_threshold is not None and args.far_still_threshold <= 0:
        parser.error("far still threshold must be greater than zero")
    if args.still_threshold is not None and (
        args.near_still_threshold is not None or args.far_still_threshold is not None
    ):
        parser.error("use --still-threshold or the separate near/far thresholds, not both")
    if args.still_threshold is not None:
        near_still_speed_m_s = far_still_speed_m_s = args.still_threshold
    else:
        near_still_speed_m_s = (
            DEFAULT_NEAR_STILL_SPEED_M_S
            if args.near_still_threshold is None else args.near_still_threshold
        )
        far_still_speed_m_s = (
            DEFAULT_FAR_STILL_SPEED_M_S
            if args.far_still_threshold is None else args.far_still_threshold
        )
    if output_json in (video_path, players_path) or manual_review_json in (video_path, players_path, output_json):
        parser.error("input and output paths must be different")
    if reviewed_serves_json in (video_path, players_path, output_json, manual_review_json):
        parser.error("generated output paths must be different from inputs and each other")
    if review_dir in (video_path, players_path, output_json, manual_review_json, reviewed_serves_json):
        parser.error("input and output paths must be different")
    if racket_review_json is not None and racket_review_json in (
        video_path, players_path, output_json, manual_review_json, reviewed_serves_json, shuttle_path
    ):
        parser.error("racket review JSON must be separate from all inputs and generated outputs")
    if shuttle_path is not None and shuttle_path in (
        video_path, players_path, output_json, manual_review_json, reviewed_serves_json, racket_review_json
    ):
        parser.error("TrackNet JSON must be separate from all outputs and other inputs")
    if racket_review_json == review_dir / "racket_orientation_review.template.json":
        parser.error("copy the generated template to a separate JSON file before editing it")
    existing_outputs = [
        path for path in (output_json, manual_review_json, reviewed_serves_json) if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        parser.error(
            "output already exists: " + ", ".join(str(path) for path in existing_outputs) + "; pass --overwrite"
        )
    if review_dir.exists() and not args.overwrite:
        parser.error(f"review directory already exists: {review_dir} (pass --overwrite)")
    if args.overwrite:
        # A new candidate list invalidates the previous UI decisions.
        reviewed_serves_json.unlink(missing_ok=True)

    data = json.loads(players_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        parser.error("players JSON contains no frames")
    fps = float(data.get("fps", 0.0))
    if fps <= 0:
        parser.error("players JSON needs a positive fps value")
    timestamps = [float(frame["t"]) for frame in frames]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        parser.error("players JSON timestamps must increase")
    start_time = time.monotonic()

    raw_positions = {
        player: [read_position(frame, "r" if player == "R" else "o") for frame in frames]
        for player in ("R", "O")
    }
    smoothing_window = max(1, round(0.5 * fps))
    player_positions = {
        player: smooth_positions(raw_positions[player], smoothing_window)
        for player in ("R", "O")
    }
    player_speeds = {
        player: position_speeds(player_positions[player], timestamps)
        for player in ("R", "O")
    }
    mask = [
        in_serve_area(
            {player: player_positions[player][i] for player in ("R", "O")},
            {player: player_speeds[player][i] for player in ("R", "O")},
            near_still_speed_m_s,
            far_still_speed_m_s,
        )
        for i in range(len(frames))
    ]
    groups = find_and_merge_windows(
        mask,
        timestamps,
        player_positions,
        args.min_duration,
        args.merge_gap,
    )
    events = [
        event_record(i + 1, group, timestamps, player_positions)
        for i, group in enumerate(groups)
    ]
    refine_server_guesses_with_centerline(events, groups, player_positions)
    shuttle_data = load_shuttle_tracks(shuttle_path) if shuttle_path is not None else None
    if shuttle_data is not None:
        validate_shuttle_video(shuttle_data, video_path)
        add_tracknet_server_cues(events, frames, shuttle_data)
        print(f"TrackNet server-origin cues attached: {sum(bool(e.get('tracknet_serve_cue', {}).get('server_hint')) for e in events)}/{len(events)}.")
        if args.use_shuttle_server_cue:
            refined = refine_server_guesses_with_tracknet(
                events, minimum_confidence=args.shuttle_server_min_confidence
            )
            print(
                f"TrackNet refined server/receiver after center-line pass: {refined}/{len(events)} "
                f"(minimum confidence {args.shuttle_server_min_confidence:.2f})."
            )
    racket_overrides = apply_racket_orientation_review(events, racket_review_json)
    if racket_review_json is not None:
        review_records = [
            event["racket_orientation_review"]
            for event in events
            if isinstance(event.get("racket_orientation_review"), dict)
        ]
        ambiguous_count = sum(record.get("status") == "ambiguous" for record in review_records)
        unavailable_count = sum(
            record.get("status") == "server_side_unavailable" for record in review_records
        )
        print(
            f"Racket orientation review: {len(review_records)}/{len(events)} candidate rows read; "
            f"{racket_overrides} overrides applied, {ambiguous_count} ambiguous, "
            f"{unavailable_count} missing server positions."
        )
        if racket_overrides == 0 and review_records:
            print(
                "An override needs one player's value to be 'up' and the other to be "
                "'sideways' or 'down'; null and 'unclear' do not establish that contrast."
            )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video for review frames: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        cap.release()
        raise RuntimeError("Could not read a valid frame rate for serve review frames")
    try:
        if review_dir.exists() and args.overwrite:
            for old_page in review_dir.glob("serves_page_*.jpg"):
                old_page.unlink()
        page_count = make_review_pages(cap, events, frames, source_fps, review_dir)
        template_path = write_racket_review_template(review_dir, events)
    finally:
        cap.release()

    interactive_review = args.interactive_review
    if interactive_review is None:
        interactive_review = sys.stdin.isatty()
    if interactive_review:
        events, rejected_events, decisions = review_candidates_interactively(events, review_dir)
    else:
        rejected_events = []
        decisions = []
        for event in events:
            event["review_status"] = "unreviewed_guess"
            event["server_source"] = (
                "racket_orientation_review" if event.get("_racket_review_used") else "heuristic"
            )
        review_dir.mkdir(parents=True, exist_ok=True)
        with (review_dir / "serve_corrections.json").open("w", encoding="utf-8") as file:
            json.dump({"reviewed_interactively": False, "decisions": []}, file, indent=2)
            file.write("\n")

    for event in events:
        event["receiver"] = (
            "O" if event.get("server") == "R" else "R" if event.get("server") == "O" else None
        )

    clean_events = [
        {key: value for key, value in event.items() if not key.startswith("_")}
        for event in events
    ]
    clean_rejected = [
        {key: value for key, value in event.items() if not key.startswith("_")}
        for event in rejected_events
    ]
    payload = {
        "fps": fps,
        "still_speed_threshold_m_s": {
            "near": near_still_speed_m_s,
            "far": far_still_speed_m_s,
        },
        "source": str(players_path),
        "player_labels": {"R": "Stage 3 identity selection", "O": "opponent"},
        "player_identity": data.get("player_identity", {}),
        "side_convention": "right/left from the server's own perspective; the far player's right is camera-left",
        "server_guess_method": (
            "stationary candidates are generated and merged using the existing position-area, speed, duration, and net-distance flow; "
            "a separate center-line pass refines server/receiver labels after candidates are fixed; "
            "when --use-shuttle-server-cue is set, a confident TrackNet launch-origin cue refines labels after the center-line pass; "
            "user-reviewed racket orientation from --racket-review-json overrides the position heuristic "
            "when there is a clear up vs sideways/down contrast; final guesses still require review"
        ),
        "tracknet_source": shuttle_path.name if shuttle_path else None,
        "reviewed_interactively": bool(interactive_review),
        "events": clean_events,
        "rejected_candidates": clean_rejected,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
        file.write("\n")
    with manual_review_json.open("w", encoding="utf-8") as file:
        json.dump(manual_score_review_draft(events), file, indent=2)
        file.write("\n")

    elapsed = time.monotonic() - start_time
    guess_counts = {"R": 0, "O": 0, "unknown": 0}
    for event in events:
        guess_counts[event["server"] if event["server"] in ("R", "O") else "unknown"] += 1
    print(
        f"Accepted serves: {len(events)}; rejected candidates: {len(rejected_events)} "
        f"(R guesses {guess_counts['R']}, O guesses {guess_counts['O']}, unknown {guess_counts['unknown']})."
    )
    print(f"JSON: {output_json}")
    print(f"Stage 5 manual score review draft: {manual_review_json}")
    print(f"Review sheets: {page_count} page(s) in {review_dir}")
    print(f"Racket orientation template: {template_path}")
    print(
        f"Smoothing window: {smoothing_window} frames; still speed thresholds: "
        f"near < {near_still_speed_m_s:.2f} m/s, far < {far_still_speed_m_s:.2f} m/s; "
        f"minimum still window: {args.min_duration:.1f}s."
    )
    print(f"Elapsed: {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Stage 4 failed: {error}", file=sys.stderr)
        raise SystemExit(1)
