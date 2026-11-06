"""Shared FFmpeg H.264 encoder selection for pipeline video stages."""

from __future__ import annotations

import subprocess
import sys


LIBX264 = "libx264"
VIDEOTOOLBOX = "h264_videotoolbox"


def available_encoders(ffmpeg: str) -> set[str]:
    """Return FFmpeg's available video encoder names."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg could not list its encoders")

    encoders: set[str] = set()
    # FFmpeg versions differ on whether encoder listings go to stdout or stderr.
    output = f"{result.stdout}\n{result.stderr}"
    for line in output.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) >= 2 and fields[0].startswith("V"):
            encoders.add(fields[1])
    return encoders


def probe_videotoolbox(ffmpeg: str) -> tuple[bool, str]:
    """Check whether FFmpeg can open a real VideoToolbox encode session."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:r=1:d=1", "-frames:v", "1", "-an",
        "-c:v", VIDEOTOOLBOX, "-b:v", "500k", "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    details = (result.stderr or result.stdout).strip()
    return result.returncode == 0, details


def select_h264_encoder(ffmpeg: str, preference: str) -> str:
    """Resolve auto/explicit selection and verify the selected encoder exists."""
    encoders = available_encoders(ffmpeg)
    if preference == "auto":
        if sys.platform == "darwin" and VIDEOTOOLBOX in encoders:
            usable, details = probe_videotoolbox(ffmpeg)
            if usable:
                selected = VIDEOTOOLBOX
            elif LIBX264 in encoders:
                print(
                    "FFmpeg lists VideoToolbox but could not initialize its encoder; using libx264. "
                    f"Probe details: {details or 'no details returned'}",
                    file=sys.stderr,
                )
                selected = LIBX264
            else:
                raise RuntimeError(
                    "FFmpeg lists VideoToolbox but could not initialize it, and libx264 is unavailable. "
                    f"Probe details: {details or 'no details returned'}"
                )
        else:
            selected = LIBX264
    else:
        selected = preference

    if selected == VIDEOTOOLBOX and preference != "auto":
        usable, details = probe_videotoolbox(ffmpeg)
        if not usable:
            raise RuntimeError(
                "FFmpeg could not initialize the requested VideoToolbox encoder. "
                f"Probe details: {details or 'no details returned'}"
            )

    if selected not in encoders:
        if preference == "auto" and selected == LIBX264:
            raise RuntimeError("FFmpeg does not provide libx264, which is needed as the software fallback")
        raise RuntimeError(
            f"FFmpeg does not provide the requested encoder '{selected}'. "
            "Install an FFmpeg build that includes it, or choose --video-encoder auto."
        )
    return selected


def h264_encoder_args(
    encoder: str,
    *,
    bitrate: str,
    preset: str,
    crf: int,
) -> list[str]:
    """Build encoder-specific H.264 options for proxy and analysis output."""
    if encoder == LIBX264:
        return ["-c:v", LIBX264, "-preset", preset, "-crf", str(crf)]
    if encoder == VIDEOTOOLBOX:
        # FFmpeg's VideoToolbox encoder uses a target bitrate instead of x264's CRF.
        return ["-c:v", VIDEOTOOLBOX, "-b:v", bitrate]
    raise ValueError(f"unsupported H.264 encoder: {encoder}")
