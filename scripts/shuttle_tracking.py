"""Shared readers and summaries for optional raw TrackNet shuttle tracks."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_shuttle_tracks(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    fps = data.get("fps")
    if not isinstance(frames, list) or not finite(fps) or fps <= 0:
        raise ValueError("shuttle tracks need a frames array and positive fps")
    previous_t = -math.inf
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or not finite(row.get("t")):
            raise ValueError(f"shuttle frame {index} needs a finite timestamp")
        timestamp = float(row["t"])
        if timestamp <= previous_t:
            raise ValueError("shuttle track timestamps must increase")
        previous_t = timestamp
        if not isinstance(row.get("visible"), bool):
            raise ValueError(f"shuttle frame {index} needs a boolean visible value")
        if row["visible"] and not (finite(row.get("x_px")) and finite(row.get("y_px"))):
            raise ValueError(f"visible shuttle frame {index} needs finite pixel coordinates")
    data["_times"] = [float(row["t"]) for row in frames]
    return data


def validate_shuttle_video(tracks: dict[str, Any], video_path: Path) -> None:
    """Reject shuttle tracks generated from a different video or proxy format."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video to validate TrackNet alignment: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    source_name = tracks.get("source_video")
    if source_name and Path(str(source_name)).name != video_path.name:
        raise ValueError(
            f"TrackNet tracks were generated from {Path(str(source_name)).name}, not {video_path.name}"
        )
    size = tracks.get("frame_size_px", {})
    if (int(size.get("width", -1)), int(size.get("height", -1))) != (width, height):
        raise ValueError("TrackNet frame size differs from the video")
    if int(tracks.get("frame_count", -1)) != frame_count:
        raise ValueError("TrackNet frame count differs from the video")
    track_fps = tracks.get("fps")
    if not finite(track_fps) or fps <= 0 or abs(float(track_fps) - fps) > 0.01:
        raise ValueError("TrackNet frame rate differs from the video")


def samples_between(
    tracks: dict[str, Any], start_sec: float, end_sec: float
) -> list[dict[str, Any]]:
    frames = tracks["frames"]
    times = tracks.get("_times") or [float(row["t"]) for row in frames]
    left = bisect.bisect_left(times, start_sec)
    right = bisect.bisect_right(times, end_sec)
    return frames[left:right]


def is_main_court_candidate(row: dict[str, Any]) -> bool:
    """Return whether a raw point is inside or close to the calibrated court image polygon."""
    return row.get("court_region", "unknown") != "outside"


def summarize_window(
    tracks: dict[str, Any], start_sec: float, end_sec: float
) -> dict[str, Any]:
    samples = samples_between(tracks, start_sec, end_sec)
    visible = [row for row in samples if row.get("visible") is True]
    main_court = [row for row in visible if is_main_court_candidate(row)]
    last = main_court[-1] if main_court else None
    first = main_court[0] if main_court else None
    return {
        "sample_count": len(samples),
        "visible_count": len(visible),
        "main_court_candidate_count": len(main_court),
        "outside_court_count": sum(row.get("court_region") == "outside" for row in visible),
        "main_court_share_pct": round(100 * len(main_court) / len(visible), 1) if visible else None,
        "first_main_court_t": first.get("t") if first else None,
        "last_main_court_t": last.get("t") if last else None,
        "last_main_court_point_px": (
            [round(float(last["x_px"]), 1), round(float(last["y_px"]), 1)] if last else None
        ),
        "last_main_court_region": last.get("court_region") if last else None,
    }


def point_box_distance(x: float, y: float, box: Any) -> float | None:
    """Distance from a normalized image point to a normalized XYXY player box."""
    if not isinstance(box, list) or len(box) < 4 or not all(finite(v) for v in box[:4]):
        return None
    x1, y1, x2, y2 = map(float, box[:4])
    if x2 <= x1 or y2 <= y1:
        return None
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)
