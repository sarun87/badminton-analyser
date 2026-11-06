#!/usr/bin/env python3
"""Run a local browser UI to review Stage 4 serves and edit Stage 5 scoring."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import threading
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_HTML = PROJECT_ROOT / "templates" / "serve_review.html"
VALID_PLAYERS = {"R", "O"}
VALID_SIDES = {"left", "right"}
VALID_ORIENTATIONS = {"up", "sideways", "down", "unclear"}


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_id(event: dict[str, Any]) -> int:
    value = event.get("candidate_n", event.get("n"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid Stage 4 candidate number: {value}")
    return value


class ServeReviewApp:
    def __init__(
        self,
        video_path: Path,
        players_path: Path,
        serves_path: Path,
        review_dir: Path,
        manual_review_path: Path,
        reviewed_serves_path: Path,
    ) -> None:
        self.video_path = video_path
        self.players_path = players_path
        self.serves_path = serves_path
        self.review_dir = review_dir
        self.manual_review_path = manual_review_path
        self.reviewed_serves_path = reviewed_serves_path
        self.serve_data = read_object(serves_path)
        self.players_data = read_object(players_path)
        self.video_frames = self.players_data.get("frames", [])
        if not isinstance(self.video_frames, list) or not self.video_frames:
            raise ValueError("players JSON must contain a non-empty frames array")
        self.player_times = [float(row["t"]) for row in self.video_frames]
        self.video_capture = cv2.VideoCapture(str(video_path))
        if not self.video_capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")
        self.video_fps = float(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.video_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.video_fps <= 0:
            self.video_capture.release()
            raise ValueError("proxy video has an invalid frame rate")
        self.duration = self.video_count / self.video_fps
        self.capture_lock = threading.Lock()
        self.assets_dir = review_dir / "ui_assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        for stale_asset in self.assets_dir.glob("candidate_*.jpg"):
            stale_asset.unlink()
        self._make_candidate_assets()
        self._make_candidate_zoom_assets()
        self.state = self._initial_state()

    def close(self) -> None:
        self.video_capture.release()

    def _make_candidate_assets(self) -> None:
        events = self.serve_data.get("events", [])
        rejected = self.serve_data.get("rejected_candidates", [])
        all_events = [event for event in [*events, *rejected] if isinstance(event, dict)]
        numbers = []
        for event in all_events:
            try:
                numbers.append(candidate_id(event))
            except ValueError:
                continue
        highest_candidate = max(numbers, default=0)
        for event in all_events:
            try:
                number = candidate_id(event)
            except ValueError:
                continue
            output = self.assets_dir / f"candidate_{number:04d}.jpg"
            if output.exists():
                continue
            page_number = (number - 1) // 4 + 1
            row_number = (number - 1) % 4
            page_path = self.review_dir / f"serves_page_{page_number:02d}.jpg"
            page = cv2.imread(str(page_path))
            if page is None:
                continue
            rows_on_page = min(4, highest_candidate - (page_number - 1) * 4)
            if rows_on_page <= 0:
                continue
            # Infer row height so both older (full-frame-only) and current sheets work.
            row_height = max(1, page.shape[0] // rows_on_page)
            y0 = row_number * row_height
            crop = page[y0:min(y0 + row_height, page.shape[0]), :]
            if crop.size:
                cv2.imwrite(str(output), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def _make_candidate_zoom_assets(self) -> None:
        """Build unannotated R/O close-ups at the candidate's review timestamps."""
        events = self.serve_data.get("events", [])
        rejected = self.serve_data.get("rejected_candidates", [])
        all_events = [event for event in [*events, *rejected] if isinstance(event, dict)]
        tile_width = 320
        image_side = 320
        label_height = 28
        tile_height = label_height + image_side

        for event in all_events:
            try:
                number = candidate_id(event)
            except ValueError:
                continue

            raw_times = event.get("review_times")
            times = []
            if isinstance(raw_times, list):
                for value in raw_times:
                    try:
                        timestamp = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(timestamp) and 0 <= timestamp <= self.duration and timestamp not in times:
                        times.append(timestamp)
                    if len(times) == 3:
                        break
            if not times:
                timestamp = float(event.get("t", 0.0))
                if math.isfinite(timestamp) and 0 <= timestamp <= self.duration:
                    times = [timestamp]
            if not times:
                continue

            sheet = np.full((2 * tile_height, len(times) * tile_width, 3), 32, dtype=np.uint8)
            for column, timestamp in enumerate(times):
                frame, player = self._frame_and_player_at(timestamp)
                for row, (box_key, label) in enumerate((("rb", "R"), ("ob", "O"))):
                    x0 = column * tile_width
                    y0 = row * tile_height
                    crop = None if frame is None else self._crop_player(frame, player.get(box_key))
                    cv2.putText(
                        sheet,
                        f"{label}  {timestamp:.2f}s",
                        (x0 + 8, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    if crop is None:
                        cv2.putText(
                            sheet,
                            "Player crop unavailable",
                            (x0 + 12, y0 + label_height + image_side // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (190, 190, 190),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        sheet[y0 + label_height:y0 + tile_height, x0:x0 + tile_width] = crop

            output = self.assets_dir / f"candidate_{number:04d}_zoom.jpg"
            if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"Could not write player zoom preview: {output}")

    def _frame_and_player_at(self, timestamp: float) -> tuple[np.ndarray | None, dict[str, Any]]:
        with self.capture_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = self.video_capture.read()
        index = self._player_frame_index(timestamp)
        return (frame if ok else None), self.video_frames[index]

    def _player_frame_index(self, timestamp: float) -> int:
        index = bisect.bisect_left(self.player_times, timestamp)
        if index >= len(self.player_times):
            return len(self.player_times) - 1
        if index > 0 and timestamp - self.player_times[index - 1] < self.player_times[index] - timestamp:
            return index - 1
        return index

    @staticmethod
    def _crop_player(frame: np.ndarray, box: Any, scale: float = 1.8, output_side: int = 320) -> np.ndarray | None:
        if not isinstance(box, list) or len(box) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
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
        interpolation = cv2.INTER_AREA if side > output_side else cv2.INTER_CUBIC
        return cv2.resize(crop, (output_side, output_side), interpolation=interpolation)

    def _initial_state(self) -> dict[str, Any]:
        raw_events = self.serve_data.get("events", [])
        raw_rejected = self.serve_data.get("rejected_candidates", [])
        if not isinstance(raw_events, list) or not isinstance(raw_rejected, list):
            raise ValueError("serves JSON needs events and rejected_candidates arrays")

        raw_by_id: dict[int, dict[str, Any]] = {}
        for event in [*raw_events, *raw_rejected]:
            if isinstance(event, dict):
                raw_by_id[candidate_id(event)] = event

        reviewed_by_id: dict[int, dict[str, Any]] = {}
        if self.reviewed_serves_path.is_file():
            reviewed_data = read_object(self.reviewed_serves_path)
            for event in reviewed_data.get("events", []):
                if isinstance(event, dict):
                    reviewed_by_id[candidate_id(event)] = event
            accepted_ids = set(reviewed_by_id)
        else:
            accepted_ids = {candidate_id(event) for event in raw_events if isinstance(event, dict)}

        prior_review: dict[str, Any] = {}
        if self.manual_review_path.is_file():
            prior_review = read_object(self.manual_review_path)
        prior_rows = prior_review.get("serves", [])
        prior_candidate_rows: dict[int, dict[str, Any]] = {}
        prior_rows = prior_rows if isinstance(prior_rows, list) else []
        for row in prior_rows:
            if isinstance(row, dict) and isinstance(row.get("candidate_n"), int):
                prior_candidate_rows[row["candidate_n"]] = row

        candidates = []
        for number, raw in sorted(raw_by_id.items(), key=lambda pair: float(pair[1].get("t", 0.0))):
            event = dict(raw)
            event.update(reviewed_by_id.get(number, {}))
            previous_row = prior_candidate_rows.get(number, {})
            if previous_row.get("server") in VALID_PLAYERS:
                event["server"] = previous_row["server"]
            if previous_row.get("side") in VALID_SIDES:
                event["side"] = previous_row["side"]
            racket = event.get("racket_orientation_review")
            if not isinstance(racket, dict):
                racket = {}
            image = self.assets_dir / f"candidate_{number:04d}.jpg"
            zoom_image = self.assets_dir / f"candidate_{number:04d}_zoom.jpg"
            candidates.append(
                {
                    "candidate_n": number,
                    "t": float(event.get("t", 0.0)),
                    "server": event.get("server") if event.get("server") in VALID_PLAYERS else "",
                    "receiver": "O" if event.get("server") == "R" else "R" if event.get("server") == "O" else "",
                    "side": event.get("side") if event.get("side") in VALID_SIDES else "",
                    "server_hint_method": event.get("server_hint_method"),
                    "tracknet_serve_cue": event.get("tracknet_serve_cue"),
                    "tracknet_server_refinement": event.get("tracknet_server_refinement"),
                    "position_server_guess": event.get("position_server_guess", event.get("server")),
                    "position_side_guess": event.get("position_side_guess", event.get("side")),
                    "heuristic_agreement": event.get("heuristic_agreement"),
                    "net_server_guess": event.get("net_server_guess"),
                    "net_heuristic_agreement": event.get("net_heuristic_agreement"),
                    "centerline_server_guess": event.get("centerline_server_guess"),
                    "centerline_agreement": event.get("centerline_agreement"),
                    "accepted": number in accepted_ids,
                    "racket_R": racket.get("R") if racket.get("R") in VALID_ORIENTATIONS else "unclear",
                    "racket_O": racket.get("O") if racket.get("O") in VALID_ORIENTATIONS else "unclear",
                    "image_url": f"/assets/candidate_{number:04d}.jpg" if image.is_file() else None,
                    "zoom_image_url": f"/assets/candidate_{number:04d}_zoom.jpg" if zoom_image.is_file() else None,
                }
            )

        candidate_by_id = {row["candidate_n"]: row for row in candidates}
        timeline: list[dict[str, Any]] = []
        seen_candidates: set[int] = set()
        manual_counter = 0
        for row in prior_rows:
            if not isinstance(row, dict):
                continue
            number = row.get("candidate_n")
            if isinstance(number, int):
                candidate = candidate_by_id.get(number)
                if candidate and candidate["accepted"] and number not in seen_candidates:
                    timeline.append({"kind": "candidate", "candidate_n": number})
                    seen_candidates.add(number)
                continue
            if number is not None:
                continue
            manual_counter += 1
            is_dummy = row.get("dummy_point") is True
            timestamp = 0.0 if is_dummy else row.get("timestamp_sec")
            if not is_dummy and timestamp is not None and not isinstance(timestamp, (int, float)):
                continue
            timeline.append(
                {
                    "kind": "dummy" if is_dummy else "manual",
                    "id": f"manual-{manual_counter}",
                    "timestamp_sec": None if timestamp is None else float(timestamp),
                    "server": row.get("server") if row.get("server") in VALID_PLAYERS else "",
                    "side": row.get("side") if row.get("side") in VALID_SIDES else "",
                }
            )

        # Add any accepted Stage 4 candidate missing from a partial/older manual review.
        for candidate in candidates:
            number = candidate["candidate_n"]
            if not candidate["accepted"] or number in seen_candidates:
                continue
            timestamp = candidate["t"]
            insert_at = next(
                (
                    index for index, entry in enumerate(timeline)
                    if entry["kind"] != "dummy"
                    and self._entry_time(entry, candidate_by_id) is not None
                    and self._entry_time(entry, candidate_by_id) > timestamp
                ),
                len(timeline),
            )
            timeline.insert(insert_at, {"kind": "candidate", "candidate_n": number})
            seen_candidates.add(number)

        initial_score = prior_review.get("initial_score", {"R": 0, "O": 0})
        if not isinstance(initial_score, dict):
            initial_score = {"R": 0, "O": 0}
        return {
            "video_name": self.video_path.name,
            "video_duration_sec": round(self.duration, 3),
            "candidates": candidates,
            "timeline": timeline,
            "initial_score": {
                "R": initial_score.get("R", 0),
                "O": initial_score.get("O", 0),
            },
            "final_point_winner": prior_review.get("final_point_winner"),
            "paths": {
                "reviewed_serves": str(self.reviewed_serves_path),
                "manual_review": str(self.manual_review_path),
            },
        }

    @staticmethod
    def _entry_time(entry: dict[str, Any], candidates: dict[int, dict[str, Any]]) -> float | None:
        if entry["kind"] == "candidate":
            return float(candidates[entry["candidate_n"]]["t"])
        timestamp = entry.get("timestamp_sec")
        return None if timestamp is None else float(timestamp)

    def get_state(self) -> dict[str, Any]:
        return self.state

    def preview_frame(self, timestamp: float) -> bytes:
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > self.duration:
            raise ValueError(f"timestamp must be between 0 and {self.duration:.3f} seconds")
        with self.capture_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = self.video_capture.read()
        if not ok:
            raise ValueError(f"could not read frame at {timestamp:.3f}s")
        index = bisect.bisect_left(self.player_times, timestamp)
        if index >= len(self.player_times):
            index = len(self.player_times) - 1
        elif index > 0 and timestamp - self.player_times[index - 1] < self.player_times[index] - timestamp:
            index -= 1
        player = self.video_frames[index]
        for box_key, label, color in (("rb", "R", (0, 0, 255)), ("ob", "O", (255, 100, 0))):
            box = player.get(box_key)
            if not isinstance(box, list) or len(box) < 4:
                continue
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            p1 = (round(x1 * width), round(y1 * height))
            p2 = (round(x2 * width), round(y2 * height))
            cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (max(4, p1[0]), max(20, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Proxy time {timestamp:.2f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("could not encode preview frame")
        return encoded.tobytes()

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_edits = payload.get("candidates")
        raw_entries = payload.get("timeline")
        if not isinstance(raw_edits, list) or not isinstance(raw_entries, list):
            raise ValueError("review data must include candidates and timeline arrays")

        original_events = self.serve_data.get("events", [])
        original_rejected = self.serve_data.get("rejected_candidates", [])
        original_by_id = {
            candidate_id(event): dict(event)
            for event in [*original_events, *original_rejected]
            if isinstance(event, dict)
        }
        edits_by_id: dict[int, dict[str, Any]] = {}
        for edit in raw_edits:
            if not isinstance(edit, dict):
                raise ValueError("each candidate review must be an object")
            number = edit.get("candidate_n")
            if isinstance(number, bool) or not isinstance(number, int) or number not in original_by_id or number in edits_by_id:
                raise ValueError(f"unknown or duplicate candidate_n: {number}")
            if not isinstance(edit.get("accepted"), bool):
                raise ValueError(f"candidate {number} needs an accepted true/false value")
            edits_by_id[number] = edit
        if set(edits_by_id) != set(original_by_id):
            raise ValueError("review all candidates before saving")

        accepted_ids = {
            number for number, edit in edits_by_id.items() if edit["accepted"]
        }
        entries: list[dict[str, Any]] = []
        seen_candidates: set[int] = set()
        manual_counter = 0
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise ValueError("each timeline entry must be an object")
            kind = raw.get("kind")
            server, side = raw.get("server"), raw.get("side")
            if server not in VALID_PLAYERS or side not in VALID_SIDES:
                raise ValueError("every timeline row needs server R/O and side left/right")
            if kind == "candidate":
                number = raw.get("candidate_n")
                if isinstance(number, bool) or not isinstance(number, int) or number not in accepted_ids or number in seen_candidates:
                    raise ValueError(f"candidate {number} is rejected, unknown, or listed more than once")
                seen_candidates.add(number)
                entries.append({"kind": kind, "candidate_n": number, "server": server, "side": side})
            elif kind in {"manual", "dummy"}:
                manual_counter += 1
                if kind == "dummy":
                    timestamp = 0.0
                else:
                    raw_timestamp = raw.get("timestamp_sec")
                    try:
                        timestamp = None if raw_timestamp is None else float(raw_timestamp)
                    except (TypeError, ValueError):
                        raise ValueError("manual serve timestamp must be a number or blank") from None
                    if timestamp is not None and (
                        not math.isfinite(timestamp) or timestamp <= 0 or timestamp > self.duration
                    ):
                        raise ValueError(f"manual serve timestamp must be between 0 and {self.duration:.3f}s")
                entries.append(
                    {
                        "kind": kind,
                        "id": raw.get("id", f"manual-{manual_counter}"),
                        "timestamp_sec": timestamp,
                        "server": server,
                        "side": side,
                    }
                )
            else:
                raise ValueError(f"unknown timeline entry type: {kind}")
        if seen_candidates != accepted_ids:
            missing = sorted(accepted_ids - seen_candidates)
            raise ValueError(f"accepted candidates missing from timeline: {missing}")

        candidate_time = {
            number: float(original_by_id[number].get("t", 0.0)) for number in accepted_ids
        }
        order_times = [
            candidate_time[row["candidate_n"]] if row["kind"] == "candidate" else row["timestamp_sec"]
            for row in entries
            if row["kind"] != "dummy"
            and (row["kind"] == "candidate" or row["timestamp_sec"] is not None)
        ]
        if any(after <= before for before, after in zip(order_times, order_times[1:])):
            raise ValueError("timeline rows with video timestamps must be in strictly increasing time order")

        initial = payload.get("initial_score", {})
        if not isinstance(initial, dict):
            raise ValueError("initial_score must contain R and O")
        initial_score = {}
        for player in ("R", "O"):
            value = initial.get(player)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"initial_score.{player} must be a non-negative integer")
            initial_score[player] = value
        final_winner = payload.get("final_point_winner") or None
        if final_winner not in (None, "R", "O"):
            raise ValueError("final point winner must be R, O, or blank")

        score = dict(initial_score)
        review_rows: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            number = index + 1
            if entry["kind"] == "candidate":
                row = {
                    "n": number,
                    "candidate_n": entry["candidate_n"],
                    "server": entry["server"],
                    "side": entry["side"],
                    "score_before": dict(score),
                }
            elif entry["kind"] == "dummy":
                row = {
                    "n": number,
                    "candidate_n": None,
                    "dummy_point": True,
                    "timestamp_sec": 0,
                    "server": entry["server"],
                    "side": entry["side"],
                    "score_before": dict(score),
                    "note": "Uncounted score-only point; excluded from video-time analysis",
                }
            else:
                row = {
                    "n": number,
                    "candidate_n": None,
                    "dummy_point": False,
                    "timestamp_sec": (
                        None if entry["timestamp_sec"] is None else round(entry["timestamp_sec"], 3)
                    ),
                    "server": entry["server"],
                    "side": entry["side"],
                    "score_before": dict(score),
                    "note": "Manually added video serve",
                }
            review_rows.append(row)
            if index + 1 < len(entries):
                next_server = entries[index + 1]["server"]
                score[next_server] += 1

        final_score = None
        if entries and final_winner in VALID_PLAYERS:
            final_score = dict(review_rows[-1]["score_before"])
            final_score[final_winner] += 1

        manual_review = {
            "score_order": "R-O",
            "initial_score": initial_score,
            "serves": review_rows,
            "final_point_winner": final_winner,
            "final_score": final_score,
            "notes": ["Reviewed locally in the Stage 4 serve review UI."],
        }

        accepted_events = []
        rejected_by_id = {
            candidate_id(event): dict(event)
            for event in original_rejected if isinstance(event, dict)
        }
        for number, original in original_by_id.items():
            edit = edits_by_id[number]
            event = dict(original)
            if edit["accepted"]:
                server, side = edit.get("server"), edit.get("side")
                if server not in VALID_PLAYERS or side not in VALID_SIDES:
                    raise ValueError(f"candidate {number} needs server R/O and side left/right")
                event["server"], event["side"] = server, side
                event["server_source"] = "local_review_ui"
                event["review_status"] = "ui_confirmed"
                orientation = {
                    "R": edit.get("racket_R", "unclear"),
                    "O": edit.get("racket_O", "unclear"),
                    "status": "manually_reviewed",
                    "server": server,
                    "side": side,
                }
                for player in ("R", "O"):
                    if orientation[player] not in VALID_ORIENTATIONS:
                        raise ValueError(f"candidate {number} {player} racket orientation is invalid")
                event["racket_orientation_review"] = orientation
                accepted_events.append(event)
                rejected_by_id.pop(number, None)
            else:
                event["review_status"] = "ui_rejected_not_a_serve"
                rejected_by_id[number] = event

        accepted_events.sort(key=lambda event: float(event.get("t", 0.0)))
        reviewed_serves = dict(self.serve_data)
        reviewed_serves["events"] = accepted_events
        reviewed_serves["rejected_candidates"] = [rejected_by_id[key] for key in sorted(rejected_by_id)]
        reviewed_serves["reviewed_interactively"] = True
        reviewed_serves["reviewed_with_local_ui"] = True
        reviewed_serves["server_guess_method"] = (
            "Stage 4 position and optional racket cues, with final server and court decisions reviewed in the local UI"
        )

        write_json_atomic(self.reviewed_serves_path, reviewed_serves)
        write_json_atomic(self.manual_review_path, manual_review)
        self.state = self._state_from_saved(edits_by_id, entries, initial_score, final_winner)
        return {
            "message": "Review saved. Use serves_reviewed.json and manual_score_review.json for Stage 5.",
            "reviewed_serves": str(self.reviewed_serves_path),
            "manual_review": str(self.manual_review_path),
            "point_count": len(review_rows),
            "accepted_candidate_count": len(accepted_events),
        }

    def _state_from_saved(
        self,
        edits_by_id: dict[int, dict[str, Any]],
        entries: list[dict[str, Any]],
        initial_score: dict[str, int],
        final_winner: str | None,
    ) -> dict[str, Any]:
        candidates = []
        for raw in [*self.serve_data.get("events", []), *self.serve_data.get("rejected_candidates", [])]:
            if not isinstance(raw, dict):
                continue
            number = candidate_id(raw)
            if any(row["candidate_n"] == number for row in candidates):
                continue
            edit = edits_by_id[number]
            racket = raw.get("racket_orientation_review", {})
            candidates.append(
                {
                    "candidate_n": number,
                    "t": float(raw.get("t", 0.0)),
                    "server": edit.get("server", ""),
                    "receiver": "O" if edit.get("server") == "R" else "R" if edit.get("server") == "O" else "",
                    "side": edit.get("side", ""),
                    "server_hint_method": raw.get("server_hint_method"),
                    "tracknet_serve_cue": raw.get("tracknet_serve_cue"),
                    "tracknet_server_refinement": raw.get("tracknet_server_refinement"),
                    "position_server_guess": raw.get("position_server_guess", raw.get("server")),
                    "position_side_guess": raw.get("position_side_guess", raw.get("side")),
                    "heuristic_agreement": raw.get("heuristic_agreement"),
                    "net_server_guess": raw.get("net_server_guess"),
                    "net_heuristic_agreement": raw.get("net_heuristic_agreement"),
                    "centerline_server_guess": raw.get("centerline_server_guess"),
                    "centerline_agreement": raw.get("centerline_agreement"),
                    "accepted": edit["accepted"],
                    "racket_R": edit.get("racket_R", racket.get("R", "unclear")),
                    "racket_O": edit.get("racket_O", racket.get("O", "unclear")),
                    "image_url": f"/assets/candidate_{number:04d}.jpg",
                    "zoom_image_url": f"/assets/candidate_{number:04d}_zoom.jpg",
                }
            )
        timeline = [dict(entry) for entry in entries]
        return {
            **self.state,
            "candidates": candidates,
            "timeline": timeline,
            "initial_score": initial_score,
            "final_point_winner": final_winner,
        }


class ReviewRequestHandler(BaseHTTPRequestHandler):
    app: ServeReviewApp

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/save"):
            super().log_message(format, *args)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        self._send(json.dumps(value).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(UI_HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(self.app.get_state())
            return
        if parsed.path == "/api/frame":
            try:
                timestamp = float(parse_qs(parsed.query).get("time", [""])[0])
                self._send(self.app.preview_frame(timestamp), "image/jpeg")
            except (ValueError, cv2.error) as error:
                self._send_json({"error": str(error)}, 400)
            return
        if parsed.path.startswith("/assets/"):
            name = parsed.path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"candidate_\d{4}(?:_zoom)?\.jpg", name):
                self._send_json({"error": "unknown asset"}, 404)
                return
            path = self.app.assets_dir / name
            if not path.is_file():
                self._send_json({"error": "candidate preview unavailable"}, 404)
                return
            self._send(path.read_bytes(), "image/jpeg")
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            result = self.app.save(payload)
            self._send_json(result)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Stage 4 serve candidates in a local browser UI.")
    parser.add_argument("video", type=Path, help="Stage 0 proxy video")
    parser.add_argument("players_json", type=Path, help="players.json from Stage 3")
    parser.add_argument("serves_json", type=Path, help="serves.json from Stage 4")
    parser.add_argument("review_dir", type=Path, help="Stage 4 review directory with contact sheets")
    parser.add_argument("--manual-review-json", type=Path, help="Stage 5 review JSON; defaults beside serves.json")
    parser.add_argument("--reviewed-serves-json", type=Path, help="reviewed Stage 4 serves output; defaults to serves_reviewed.json")
    parser.add_argument("--port", type=int, default=8765, help="localhost port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    players = args.players_json.expanduser().resolve()
    serves = args.serves_json.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    manual_review = (args.manual_review_json or serves.parent / "manual_score_review.json").expanduser().resolve()
    reviewed_serves = (args.reviewed_serves_json or serves.parent / "serves_reviewed.json").expanduser().resolve()
    if args.port < 1024 or args.port > 65535:
        parser.error("port must be between 1024 and 65535")
    for path in (video, players, serves, UI_HTML):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if not review_dir.is_dir():
        parser.error(f"Stage 4 review directory does not exist: {review_dir}")
    if manual_review in (video, players, serves, reviewed_serves) or reviewed_serves in (video, players, serves):
        parser.error("review output paths must differ from input paths and from each other")

    try:
        app = ServeReviewApp(video, players, serves, review_dir, manual_review, reviewed_serves)
        handler = type("BoundReviewRequestHandler", (ReviewRequestHandler,), {"app": app})
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except (OSError, ValueError, json.JSONDecodeError, cv2.error) as error:
        print(f"Serve review UI failed: {error}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serve review UI: {url}")
    print(f"Save outputs: {reviewed_serves} and {manual_review}")
    print("Keep this terminal open while reviewing; press Ctrl+C here to stop the local UI.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServe review UI stopped.")
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
