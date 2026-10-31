#!/usr/bin/env python3
"""Create a standard local proxy and save per-video analysis setup."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any


PROXY_WIDTH = 1280
PROXY_HEIGHT = 720
PROXY_FPS = 30
HDR_TRANSFERS = {"smpte2084", "smpte_st_2084", "arib-std-b67"}


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in (Path("/opt/homebrew/bin") / name, Path("/usr/local/bin") / name):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"Could not find {name}. Run scripts/init_macos.sh first.")


def run_json_command(command: list[str], description: str) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"{description} failed:\n{detail}") from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} returned invalid JSON.") from exc


def parse_fps(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def probe_video(ffprobe: str, input_path: Path) -> dict[str, Any]:
    data = run_json_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ],
        "Video metadata probe",
    )
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("The input file does not contain a video stream.")

    width = video.get("width")
    height = video.get("height")
    avg_fps = parse_fps(video.get("avg_frame_rate"))
    nominal_fps = parse_fps(video.get("r_frame_rate"))
    duration_value = data.get("format", {}).get("duration") or video.get("duration")
    try:
        duration_seconds = float(duration_value) if duration_value is not None else None
    except (TypeError, ValueError):
        duration_seconds = None

    return {
        "width": width,
        "height": height,
        "avg_frame_rate": video.get("avg_frame_rate"),
        "r_frame_rate": video.get("r_frame_rate"),
        "fps": avg_fps or nominal_fps,
        "duration_seconds": duration_seconds,
        "color_transfer": video.get("color_transfer"),
        "color_primaries": video.get("color_primaries"),
        "color_space": video.get("color_space"),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def prompt_choice(title: str, choices: list[tuple[str, str]]) -> str:
    print(f"\n{title}")
    for number, (label, _) in enumerate(choices, start=1):
        print(f"  {number}. {label}")
    while True:
        answer = input("Choose a number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][1]
        print(f"Enter a number from 1 to {len(choices)}.")


def prompt_resolution(detected: str | None) -> str | None:
    label = detected or "unknown"
    while True:
        answer = input(
            f"Source resolution, width x height [detected {label}; Enter accepts]: "
        ).strip()
        if not answer:
            return detected
        if re.fullmatch(r"\d{2,6}x\d{2,6}", answer.lower()):
            return answer.lower()
        print("Enter a resolution such as 3840x2160, or press Enter to accept the detected value.")


def prompt_fps(detected: float | None) -> float | None:
    label = f"{detected:g}" if detected is not None else "unknown"
    while True:
        answer = input(f"Source FPS [detected {label}; Enter accepts]: ").strip()
        if not answer:
            return detected
        try:
            value = float(answer)
        except ValueError:
            value = 0
        if math.isfinite(value) and value > 0:
            return value
        print("Enter a positive frame rate, such as 29.97, or press Enter to accept the detected value.")


def prompt_required(title: str, examples: str) -> str:
    while True:
        answer = input(f"{title} ({examples}): ").strip()
        if answer:
            return answer
        print("This field is required for the per-video analysis setup.")


def collect_setup(metadata: dict[str, Any]) -> dict[str, Any]:
    print("\nChoose settings for this video. They are saved in job.json for later stages.")
    camera_view = prompt_choice(
        "Filming position",
        [("Behind a baseline", "behind_baseline"), ("At the side of the court", "side")],
    )
    camera_motion = prompt_choice(
        "Camera movement",
        [
            ("Fixed in one position", "fixed"),
            ("Repositioned between segments or breaks", "repositioned_between_segments"),
            ("Panning, tilting, or moving during play", "moving_during_play"),
        ],
    )
    print("\nMatch format: singles only (doubles analysis is not implemented).")

    detected_resolution = None
    if metadata.get("width") and metadata.get("height"):
        detected_resolution = f"{metadata['width']}x{metadata['height']}"
    source_resolution = prompt_resolution(detected_resolution)
    source_fps = prompt_fps(metadata.get("fps"))
    identity = prompt_required(
        "How should the pipeline identify you?",
        "describe clothing and color, for example black shorts or a red shirt",
    )

    return {
        "camera_view": camera_view,
        "camera_movement": camera_motion,
        "match_format": "singles",
        "source_resolution_confirmed": source_resolution,
        "source_fps_confirmed": source_fps,
        "player_identity_description": identity,
    }


def build_video_filter(is_hdr: bool) -> str:
    filters: list[str] = []
    if is_hdr:
        filters.extend(
            [
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "tonemap=tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:p=bt709:r=tv",
                "format=yuv420p",
            ]
        )
    filters.extend(
        [
            "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
            "fps=30",
        ]
    )
    return ",".join(filters)


def make_proxy(ffmpeg: str, input_path: Path, output_path: Path, metadata: dict[str, Any]) -> bool:
    transfer = (metadata.get("color_transfer") or "").lower()
    is_hdr = transfer in HDR_TRANSFERS
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if partial_path.exists():
        raise RuntimeError(f"Temporary output already exists: {partial_path}. Remove it or choose another output directory.")

    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "-1",
        "-vf",
        build_video_filter(is_hdr),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(partial_path),
    ]
    print("\nCreating the 1280x720, 30 FPS proxy. This can take a while for a full match.")
    try:
        subprocess.run(command, check=True)
        partial_path.replace(output_path)
    except subprocess.CalledProcessError as exc:
        partial_path.unlink(missing_ok=True)
        detail = f"FFmpeg exited with status {exc.returncode}."
        if is_hdr:
            detail += " HDR conversion requires an FFmpeg build with zscale and tonemap filters."
        raise RuntimeError(detail) from exc
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return is_hdr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local 1280x720, 30 FPS SDR proxy and collect setup details for this video."
    )
    parser.add_argument("--input", required=True, help="Path to the original match video.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where proxy.mp4 and job.json will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = output_dir / "proxy.mp4"
    job_path = output_dir / "job.json"

    if not input_path.is_file():
        print(f"Input video does not exist or is not a file: {input_path}", file=sys.stderr)
        return 2
    if output_path.exists() or job_path.exists():
        print(
            f"Output already exists in {output_dir}. Choose a new output directory to preserve existing work.",
            file=sys.stderr,
        )
        return 2

    try:
        ffprobe = find_tool("ffprobe")
        ffmpeg = find_tool("ffmpeg")
        metadata = probe_video(ffprobe, input_path)
        setup = collect_setup(metadata)
        output_dir.mkdir(parents=True, exist_ok=True)
        was_tonemapped = make_proxy(ffmpeg, input_path, output_path, metadata)

        job = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_video": {
                "path": str(input_path),
                "ffprobe": metadata,
            },
            "analysis_setup": setup,
            "stage_0_proxy": {
                "path": str(output_path),
                "resolution": f"{PROXY_WIDTH}x{PROXY_HEIGHT}",
                "fps": PROXY_FPS,
                "video_codec": "H.264",
                "audio_preserved": bool(metadata.get("has_audio")),
                "hdr_tonemapped": was_tonemapped,
                "aspect_ratio_policy": "fit_and_pad",
                "color_space": "BT.709 SDR",
            },
        }
        temporary_job = job_path.with_suffix(".json.partial")
        try:
            temporary_job.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            temporary_job.replace(job_path)
        except OSError:
            temporary_job.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        print(f"Stage 0 failed: {exc}", file=sys.stderr)
        return 1

    print("\nStage 0 complete.")
    print(f"Proxy: {output_path}")
    print(f"Per-video settings: {job_path}")
    print("The proxy is 1280x720 at 30 FPS, with the source aspect ratio preserved using padding.")
    print("Later stages will read job.json to choose camera and match-specific processing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
