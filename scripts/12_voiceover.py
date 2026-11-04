#!/usr/bin/env python3
"""Create a local spoken match summary and mux it into the rendered video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import artifact_path, load_job, read_json
from visual_stats import build_match_stats, narration_text


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in (Path("/opt/homebrew/bin") / name, Path("/usr/local/bin") / name):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"Could not find {name}; run scripts/init_macos.sh first.")


def run(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"{description} failed:\n{detail}") from exc


def duration_seconds(ffprobe: str, path: Path) -> tuple[float, bool]:
    result = run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        f"Probing {path.name}",
    )
    try:
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read media duration from {path}.") from exc
    has_audio = any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))
    if duration <= 0:
        raise RuntimeError(f"Could not read a positive media duration from {path}.")
    return duration, has_audio


def coaching_note(report_path: Path) -> str | None:
    if not report_path.is_file():
        return None
    report = report_path.read_text(encoding="utf-8")
    if "## Coaching notes" not in report:
        return None
    section = report.split("## Coaching notes", 1)[1].split("\n## ", 1)[0]
    notes = [line.strip()[2:].strip() for line in section.splitlines() if line.strip().startswith("- ")]
    if not notes or notes[0].startswith("The current metrics do not show"):
        return None
    return notes[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to job.json.")
    parser.add_argument("--video", help="Rendered video; defaults to the path in render_manifest.json.")
    parser.add_argument("--manifest", help="Render manifest; defaults to render_manifest.json beside job.json.")
    parser.add_argument("--points", help="Points JSON; defaults to points.json beside job.json.")
    parser.add_argument("--motion", help="Movement JSON; defaults to motion.json beside job.json when present.")
    parser.add_argument("--report", help="Report Markdown; defaults to report.md beside job.json when present.")
    parser.add_argument("--voice", help="Optional built-in macOS voice name, such as Samantha or Alex.")
    parser.add_argument("--rate", type=int, default=175, help="Speech rate in words per minute (default: 175).")
    parser.add_argument("--audio-output", help="Narration audio path; defaults to coach_voiceover.aiff beside job.json.")
    parser.add_argument("--script-output", help="Narration text path; defaults to coach_narration.txt beside job.json.")
    parser.add_argument("--output", help="Video output; defaults to overlay_with_voiceover.mp4 beside job.json.")
    args = parser.parse_args()
    job_path = Path(args.job).expanduser().resolve()
    try:
        load_job(job_path)
        say = shutil.which("say")
        if not say:
            raise RuntimeError("The macOS 'say' command is unavailable; Stage 12 voiceover requires macOS.")
        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else artifact_path(job_path, "render_manifest.json")
        manifest = read_json(manifest_path)
        video_path = Path(args.video).expanduser().resolve() if args.video else Path(str(manifest.get("video_path", ""))).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Rendered video not found: {video_path}. Run Stage 11 first.")
        points_path = Path(args.points).expanduser().resolve() if args.points else artifact_path(job_path, "points.json")
        motion_path = Path(args.motion).expanduser().resolve() if args.motion else artifact_path(job_path, "motion.json")
        report_path = Path(args.report).expanduser().resolve() if args.report else artifact_path(job_path, "report.md")
        audio_path = Path(args.audio_output).expanduser().resolve() if args.audio_output else artifact_path(job_path, "coach_voiceover.aiff")
        script_path = Path(args.script_output).expanduser().resolve() if args.script_output else artifact_path(job_path, "coach_narration.txt")
        output_path = Path(args.output).expanduser().resolve() if args.output else artifact_path(job_path, "overlay_with_voiceover.mp4")
        if output_path.resolve() == video_path.resolve():
            raise ValueError("Choose a separate --output path so the rendered source is preserved.")
        if args.rate < 100 or args.rate > 300:
            raise ValueError("--rate must be between 100 and 300 words per minute.")

        point_data = read_json(points_path)
        points = point_data.get("points", [])
        if not points:
            raise ValueError("points.json has no points. Run Stages 5 and 6 before creating narration.")
        motion = read_json(motion_path) if motion_path.is_file() else {}
        stats = build_match_stats(points, motion)
        text = narration_text(stats, include_graphs=bool(manifest.get("stats_enabled", False)))
        note = coaching_note(report_path)
        if note:
            text += f" Coach's note: {note}"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(text.strip() + "\n", encoding="utf-8")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        command = [say, "-r", str(args.rate), "-f", str(script_path), "-o", str(audio_path)]
        if args.voice:
            command[1:1] = ["-v", args.voice]
        run(command, "Generating local macOS narration")

        source_duration, source_has_audio = duration_seconds(ffprobe, video_path)
        voice_duration, _ = duration_seconds(ffprobe, audio_path)
        match_duration = float(manifest.get("match_seconds", source_duration))
        heatmap_seconds = float(manifest.get("heatmap_seconds", 0))
        stats_enabled = bool(manifest.get("stats_enabled", False))
        voice_start = match_duration + (heatmap_seconds if stats_enabled else 0.0)
        final_duration = max(source_duration, voice_start + voice_duration)
        extra_video = max(0.0, final_duration - source_duration)
        delay_ms = int(round(voice_start * 1000))
        if source_has_audio:
            audio_filter = f"[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[bg];[1:a]adelay={delay_ms}|{delay_ms},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[voice];[bg][voice]amix=inputs=2:duration=longest:dropout_transition=0[a]"
        else:
            audio_filter = f"[1:a]adelay={delay_ms}|{delay_ms},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a]"
        video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={extra_video:.3f},setpts=PTS-STARTPTS[v]"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-filter_complex",
                f"{video_filter};{audio_filter}",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-t",
                f"{final_duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            "Adding narration to the rendered video",
        )
        print(f"Saved narration text to {script_path} and local speech audio to {audio_path}.")
        print(f"Saved narrated video to {output_path}.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Voiceover failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
