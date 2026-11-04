#!/usr/bin/env python3
"""Create a 720p, 30 fps SDR working copy for the badminton analysis pipeline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from video_encoding import h264_encoder_args, select_h264_encoder


HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
SDR_TRANSFERS = {
    "bt709",
    "smpte170m",
    "gamma22",
    "gamma28",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
}


def probe_video(ffprobe: str, path: Path) -> dict[str, str]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,color_transfer,color_primaries,color_space",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe could not read {path}")

    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    return streams[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Make a 1280x720, 30 fps SDR proxy video."
    )
    parser.add_argument("input_video", type=Path, help="path to the original match video")
    parser.add_argument("output_video", type=Path, help="path for the generated proxy MP4")
    parser.add_argument(
        "--match-start-sec", type=float, required=True,
        help="timestamp in the original video of the first serve at score 0-0; the proxy starts here",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it already exists"
    )
    parser.add_argument(
        "--video-encoder",
        choices=("auto", "libx264", "h264_videotoolbox"),
        default="auto",
        help="auto uses VideoToolbox on macOS when available, otherwise libx264 (default: auto)",
    )
    parser.add_argument(
        "--videotoolbox-bitrate",
        default="2500k",
        help="target H.264 bitrate for VideoToolbox encoding (default: 2500k)",
    )
    parser.add_argument(
        "--quiet-ffmpeg",
        action="store_true",
        help="hide FFmpeg warnings and encoding progress while keeping errors visible",
    )
    args = parser.parse_args()

    source = args.input_video.expanduser().resolve()
    output = args.output_video.expanduser().resolve()
    metadata_path = output.with_suffix(".proxy.json")
    if not source.is_file():
        parser.error(f"input video does not exist: {source}")
    if source == output:
        parser.error("input and output paths must be different")
    if not math.isfinite(args.match_start_sec) or args.match_start_sec < 0:
        parser.error("match start time must be a finite number greater than or equal to 0")
    if (output.exists() or metadata_path.exists()) and not args.overwrite:
        parser.error(f"proxy output already exists (pass --overwrite): {output} or {metadata_path}")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("ffmpeg and ffprobe must be installed and available on PATH")

    try:
        video_encoder = select_h264_encoder(ffmpeg, args.video_encoder)
        info = probe_video(ffprobe, source)
        source_duration = float(info.get("duration", 0.0))
        if source_duration > 0 and args.match_start_sec >= source_duration:
            raise ValueError(
                f"first-serve timestamp {args.match_start_sec:.3f}s is beyond video duration {source_duration:.3f}s"
            )
        transfer = info.get("color_transfer", "").lower()
        if transfer in HDR_TRANSFERS:
            print(f"Input transfer is {transfer}; applying HDR-to-SDR tone mapping.")
            video_filter = (
                "zscale=t=linear:npl=100,format=gbrpf32le,"
                "tonemap=tonemap=hable:desat=0,"
                "zscale=p=bt709:t=bt709:m=bt709:r=tv,"
                "scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
            )
        elif transfer in SDR_TRANSFERS:
            print(f"Input transfer is {transfer}; source is SDR, no tone mapping needed.")
            video_filter = (
                "scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
            )
        else:
            raise RuntimeError(
                "Could not safely identify the input color transfer "
                f"({transfer or 'missing'}). Inspect it with ffprobe before converting."
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, "-hide_banner"]
        if args.quiet_ffmpeg:
            command.extend(["-loglevel", "error"])
        command.extend([
            "-n" if not args.overwrite else "-y",
            "-ss",
            f"{args.match_start_sec:.6f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            video_filter,
            *h264_encoder_args(
                video_encoder,
                bitrate=args.videotoolbox_bitrate,
                preset="medium",
                crf=20,
            ),
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-fps_mode",
            "cfr",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ])
        print(f"Input:  {source}")
        print(f"Output: {output}")
        print(f"Video encoder: {video_encoder}")
        if video_encoder == "h264_videotoolbox":
            print(f"VideoToolbox target bitrate: {args.videotoolbox_bitrate}")
        subprocess.run(command, check=True)

        result = probe_video(ffprobe, output)
        proxy_duration = float(result.get("duration", 0.0))
        metadata = {
            "source_video": str(source),
            "proxy_video": output.name,
            "video_encoder": video_encoder,
            "first_serve_at_0_0_source_sec": round(args.match_start_sec, 3),
            "source_duration_sec": round(source_duration, 3) if source_duration > 0 else None,
            "proxy_duration_sec": round(proxy_duration, 3) if proxy_duration > 0 else None,
            "timestamp_convention": "downstream timestamps are relative to the trimmed proxy; add first_serve_at_0_0_source_sec for source-video time",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(
            "Proxy complete: "
            f"{result.get('width')}x{result.get('height')}, "
            f"{result.get('avg_frame_rate', result.get('r_frame_rate'))} fps, "
            f"{result.get('color_transfer')} transfer."
        )
        print(f"First serve source timestamp: {args.match_start_sec:.3f}s; analysis timestamps start at 0.000s.")
        print(f"Proxy metadata: {metadata_path}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        print(f"Stage 0 failed: {detail or error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
