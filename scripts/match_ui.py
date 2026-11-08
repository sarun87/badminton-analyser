#!/usr/bin/env python3
"""Browser-driven, local-first runner and review hub for badminton matches."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from stage2_calibrate import LANDMARKS, save_calibration
from stage4_review_ui import ServeReviewApp
from stage7_net_plane_calibrate import POINT_LABELS, project as project_plane


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
HTML_PATH = ROOT / "templates" / "match_ui.html"
SERVE_HTML_PATH = ROOT / "templates" / "serve_review.html"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
MAX_JSON_BYTES = 8_000_000
MAX_UPLOAD_BYTES = 80 * 1024**3

STAGE_ORDER = ["stage0", "stage1", "stage2", "stage3", "stage3a", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "stage10", "stage10a", "stage11"]
STAGE_INFO: dict[str, dict[str, Any]] = {
    "stage0": {"label": "0 · Proxy", "about": "Trim at the first 0–0 serve, create the working video, then check the result.", "review": True},
    "stage1": {"label": "1 · Person detection", "about": "Find people in sampled proxy frames.", "review": True},
    "stage2": {"label": "2 · Court calibration", "about": "Click known court lines in the browser and review the projected court overlay.", "review": True},
    "stage3": {"label": "3 · Player tracking", "about": "Track R by clothing color and O as the other player.", "review": True},
    "stage3a": {"label": "3A · Shuttle tracking (optional)", "about": "Run TrackNet in its separate environment and review the preview.", "review": True},
    "stage4": {"label": "4 · Serve candidates", "about": "Suggest serve moments, server, and court side.", "review": True},
    "stage5": {"label": "5 · Score", "about": "Validate the reviewed serve list and score timeline.", "review": False},
    "stage6": {"label": "6 · Rally times", "about": "Estimate rally ends and correct them against video frames.", "review": True},
    "stage7": {"label": "7 · Movement", "about": "Calculate movement, heatmaps, and optional shuttle shot estimates.", "review": True},
    "stage8": {"label": "8 · Pose", "about": "Analyze selected-player pose cues and inspect the skeleton preview.", "review": True},
    "stage9": {"label": "9 · Pickup evidence", "about": "Review bends and correct likely shuttle landing side.", "review": True},
    "stage10": {"label": "10 · Report", "about": "Combine score, movement, pose, and pickup evidence.", "review": True},
    "stage10a": {"label": "10A · AI coach (optional)", "about": "Export a coaching context or use a local Ollama model.", "review": True},
    "stage11": {"label": "11 · Analysis video (optional)", "about": "Render full-match or short analysis video.", "review": True},
}
DEPENDENCIES: dict[str, list[str]] = {
    "stage0": ["source_video"],
    "stage1": ["match-720p.mp4", "yolox_model"],
    "stage2": ["match-720p.mp4"],
    "stage3": ["match-720p.mp4", "dets.json", "court_calibration.json"],
    "stage3a": ["match-720p.mp4", "court_calibration.json", "tracknet"],
    "stage4": ["match-720p.mp4", "players.json"],
    "stage5": ["serves_reviewed.json", "manual_score_review.json"],
    "stage6": ["match-720p.mp4", "points.json", "players.json"],
    "stage7": ["players.json", "rallies_review.json"],
    "stage8": ["match-720p.mp4", "players.json", "rallies_review.json", "pose_model"],
    "stage9": ["match-720p.mp4", "players.json", "rallies_review.json"],
    "stage10": ["points.json", "rallies_review.json", "motion.json", "players.json", "pose_stats.json", "pickups.json"],
    "stage10a": ["report_data.json"],
    "stage11": ["match-720p.mp4", "players.json", "rallies_review.json", "pose_stats.json", "report_data.json"],
}
# A file can exist even though its browser review has not been completed, or
# because it predates a rerun. Enforce the review gates when a later stage is
# launched manually as well as when Continue workflow is used.
STAGE_PREREQUISITES: dict[str, list[tuple[str, bool]]] = {
    "stage1": [("stage0", True)],
    "stage2": [("stage0", True)],
    "stage3": [("stage1", True), ("stage2", True)],
    "stage3a": [("stage2", True)],
    "stage4": [("stage3", True)],
    "stage5": [("stage4", True)],
    "stage6": [("stage5", False)],
    "stage7": [("stage6", True)],
    "stage8": [("stage6", True)],
    "stage9": [("stage6", True)],
    "stage10": [("stage7", True), ("stage8", True), ("stage9", True)],
    "stage10a": [("stage10", True)],
    "stage11": [("stage10", True)],
}
INVALIDATES: dict[str, list[str]] = {
    "stage0": STAGE_ORDER[1:],
    "stage1": STAGE_ORDER[3:],
    "stage2": STAGE_ORDER[3:],
    "stage3": STAGE_ORDER[4:],
    "stage3a": ["stage4", "stage6", "stage7", "stage9", "stage10", "stage10a", "stage11"],
    "stage4": STAGE_ORDER[STAGE_ORDER.index("stage5"):],
    "stage5": STAGE_ORDER[STAGE_ORDER.index("stage6"):],
    "stage6": STAGE_ORDER[STAGE_ORDER.index("stage7"):],
    "stage7": STAGE_ORDER[STAGE_ORDER.index("stage8"):],
    "stage8": STAGE_ORDER[STAGE_ORDER.index("stage10"):],
    "stage9": STAGE_ORDER[STAGE_ORDER.index("stage10"):],
    "stage10": ["stage10a", "stage11"],
    "stage10a": ["stage11"],
    "stage11": [],
}
FLOW_ORDER = ["stage0", "stage1", "stage2", "stage3", "stage3a", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "stage10", "stage10a", "stage11"]
INSPECTION_ONLY_REVIEWS = {"stage0", "stage1", "stage2", "stage3a", "stage8", "stage11"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid match run id")
    path = (RUNS / run_id).resolve()
    if path.parent != RUNS.resolve():
        raise ValueError("match path is outside runs/")
    return path


def relative_file(run_dir: Path, name: str) -> Path:
    path = (run_dir / name).resolve()
    if path != run_dir and run_dir not in path.parents:
        raise ValueError("file path must stay inside this match run")
    return path


def fresh_stage() -> dict[str, Any]:
    return {"status": "not_started", "review_status": "not_needed", "progress": None, "message": "", "started_at": None, "finished_at": None}


def make_run_state(run_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    stages = {key: fresh_stage() for key in STAGE_ORDER}
    stages["stage2"]["review_status"] = "needed"
    return {
        "id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "settings": settings,
        "source_video": None,
        "net_plane_calibration_review_status": "not_needed",
        "active_stage": None,
        "active_status": "idle",
        "stages": stages,
        "logs": [],
    }


class RunManager:
    def __init__(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.worker_busy = False
        self.review_busy = False
        self.process: subprocess.Popen[str] | None = None
        self.process_run_id: str | None = None
        self.stop_requested = False
        self.serve_apps: dict[str, ServeReviewApp] = {}
        self.progress_re = re.compile(
            r"(?:Mapped|Processed|Rallies processed|Serve detection frames processed|Pickup points processed|"
            r"Pose frames processed|Pose overlay samples|Full match frames rendered):?\s*(\d+)\s*(?:/|of)\s*(\d+)", re.I
        )
        for path in RUNS.iterdir():
            state_path = path / "ui_state.json"
            if not path.is_dir() or not state_path.is_file():
                continue
            try:
                state = self.read_state(path.name)
                if state.get("active_stage"):
                    stage = state["stages"].get(state["active_stage"], {})
                    stage.update({"status": "interrupted", "message": "Server stopped while this stage was running."})
                    state["active_stage"] = None
                    state["active_status"] = "idle"
                    atomic_json(state_path, state)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    def read_state(self, run_id: str) -> dict[str, Any]:
        path = safe_run_dir(run_id) / "ui_state.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("id") != run_id:
            raise ValueError("invalid match run state")
        settings = value.setdefault("settings", {})
        has_tracknet = bool(settings.get("tracknet_enabled", False))
        settings.setdefault("shuttle_overlay_enabled", has_tracknet)
        settings.setdefault("shot_count_enabled", has_tracknet)
        return value

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_json(safe_run_dir(state["id"]) / "ui_state.json", state)

    def list_runs(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(RUNS.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_dir() or not (path / "ui_state.json").is_file():
                continue
            try:
                state = self.read_state(path.name)
                result.append({
                    "id": state["id"],
                    "created_at": state.get("created_at"),
                    "source_video": state.get("source_video"),
                    "active_status": state.get("active_status"),
                    "active_stage": state.get("active_stage"),
                    "completed_stages": sum(row.get("status") == "completed" for row in state.get("stages", {}).values()),
                })
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return result

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "match").strip()
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "match"
        run_id = f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        settings = {
            "name": name[:80],
            "r_name": str(payload.get("r_name") or "Player R").strip()[:80],
            "o_name": str(payload.get("o_name") or "Player O").strip()[:80],
            "identity_garment": str(payload.get("identity_garment") or "").strip().lower(),
            "identity_color": str(payload.get("identity_color") or "").strip().lower(),
            "first_serve_sec": float(payload.get("first_serve_sec", 0) or 0),
            "analysis_subject": payload.get("analysis_subject", "R"),
            "tracknet_enabled": bool(payload.get("tracknet_enabled", False)),
            "tracknet_repo": str(payload.get("tracknet_repo", "")),
            "tracknet_python": str(payload.get("tracknet_python", "")),
            "tracknet_checkpoint": str(payload.get("tracknet_checkpoint", "")),
            "pose_model": str(payload.get("pose_model", str(ROOT / "models" / "pose_landmarker_heavy.task"))),
            "short_mode": bool(payload.get("short_mode", False)),
            "video_enabled": bool(payload.get("video_enabled", True)),
            "voice_enabled": bool(payload.get("voice_enabled", True)),
            "pose_overlay_enabled": bool(payload.get("pose_overlay_enabled", True)),
            "shuttle_overlay_enabled": bool(payload.get("shuttle_overlay_enabled", payload.get("tracknet_enabled", False))),
            "shot_count_enabled": bool(payload.get("shot_count_enabled", payload.get("tracknet_enabled", False))),
            "ai_coach_enabled": bool(payload.get("ai_coach_enabled", False)),
            "ai_mode": payload.get("ai_mode", "context"),
            "coach_categories": {},
        }
        if settings["identity_garment"] not in ("pants", "pant", "shirt"):
            raise ValueError("choose shirt or pants as the identity garment")
        if settings["identity_color"] not in {"black", "white", "gray", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown"}:
            raise ValueError("choose a supported identity color: black, white, gray, red, orange, yellow, green, cyan, blue, purple, pink, or brown")
        if settings["analysis_subject"] not in ("R", "O"):
            raise ValueError("analysis subject must be R or O")
        if settings["ai_mode"] not in ("context", "ollama"):
            raise ValueError("Stage 10A mode must be context or ollama")
        if not math.isfinite(settings["first_serve_sec"]) or settings["first_serve_sec"] < 0:
            raise ValueError("first serve timestamp must be a non-negative number")
        run_dir = safe_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "logs").mkdir()
        state = make_run_state(run_id, settings)
        atomic_json(run_dir / "settings.json", settings)
        self.save_state(state)
        return state

    def update_settings(self, run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            state = self.read_state(run_id)
            if state.get("active_stage") or self.review_busy:
                raise ValueError("match settings cannot change while a stage is running")
            allowed = set(state["settings"])
            previous = dict(state["settings"])
            for key, value in updates.items():
                if key not in allowed:
                    continue
                state["settings"][key] = value
            settings = state["settings"]
            settings.setdefault("video_enabled", True)
            settings["name"] = str(settings.get("name") or "Match").strip()[:80]
            settings["r_name"] = str(settings.get("r_name") or "Player R").strip()[:80]
            settings["o_name"] = str(settings.get("o_name") or "Player O").strip()[:80]
            settings["identity_garment"] = str(settings.get("identity_garment") or "").strip().lower()
            settings["identity_color"] = str(settings.get("identity_color") or "").strip().lower()
            settings["first_serve_sec"] = float(settings.get("first_serve_sec", 0))
            for key in ("tracknet_repo", "tracknet_python", "tracknet_checkpoint", "pose_model"):
                settings[key] = str(settings.get(key) or "")
            for key in ("tracknet_enabled", "short_mode", "video_enabled", "voice_enabled", "pose_overlay_enabled", "shuttle_overlay_enabled", "shot_count_enabled", "ai_coach_enabled"):
                if not isinstance(settings.get(key), bool):
                    raise ValueError(f"{key} must be true or false")
            if settings["identity_garment"] not in ("pants", "pant", "shirt"):
                raise ValueError("choose shirt or pants as the identity garment")
            if settings["identity_color"] not in {"black", "white", "gray", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown"}:
                raise ValueError("choose a supported clothing color")
            if not math.isfinite(settings["first_serve_sec"]) or settings["first_serve_sec"] < 0:
                raise ValueError("first serve timestamp must be a non-negative number")
            if settings.get("ai_mode") not in ("context", "ollama"):
                raise ValueError("Stage 10A mode must be context or ollama")
            if settings.get("analysis_subject") not in ("R", "O"):
                raise ValueError("analysis subject must be R or O")
            changed = {key for key in allowed if previous.get(key) != settings.get(key)}

            def mark_stale(stage: str, message: str) -> None:
                row = state["stages"][stage]
                if row.get("status") not in ("not_started", "skipped"):
                    row.update({"status": "stale", "review_status": "needed" if STAGE_INFO[stage]["review"] else "not_needed", "progress": None, "message": message})

            if "first_serve_sec" in changed:
                mark_stale("stage0", "Rerun the proxy after changing the first-serve timestamp.")
                self.invalidate(state, "stage0")
            if {"identity_garment", "identity_color"} & changed:
                mark_stale("stage3", "Rerun player tracking after changing R's clothing identity.")
                self.invalidate(state, "stage3")
            if {"tracknet_enabled", "tracknet_repo", "tracknet_python", "tracknet_checkpoint"} & changed:
                mark_stale("stage3a", "Rerun TrackNet after changing its settings.")
                self.invalidate(state, "stage3a")
            if {"analysis_subject", "pose_model"} & changed:
                mark_stale("stage8", "Rerun pose analysis for the selected player/model.")
                self.invalidate(state, "stage8")
            if {"r_name", "o_name"} & changed:
                mark_stale("stage10", "Rerun the report after changing player names.")
                self.invalidate(state, "stage10")
            if {"ai_coach_enabled", "ai_mode"} & changed:
                mark_stale("stage10a", "Rerun Stage 10A after changing AI-coach settings.")
                self.invalidate(state, "stage10a")
                if not settings.get("ai_coach_enabled"):
                    suffix = "-O" if settings.get("analysis_subject") == "O" else ""
                    (safe_run_dir(run_id) / f"AI-coach-narrative{suffix}.md").unlink(missing_ok=True)
            if {"short_mode", "voice_enabled", "pose_overlay_enabled", "shuttle_overlay_enabled", "shot_count_enabled"} & changed:
                mark_stale("stage11", "Rerun the analysis video after changing render settings.")
            if "video_enabled" in changed and settings["video_enabled"]:
                mark_stale("stage11", "Run the analysis video after enabling Stage 11.")
            atomic_json(safe_run_dir(run_id) / "settings.json", state["settings"])
            self.save_state(state)
            return state

    def upload(self, run_id: str, filename: str, body_reader: Any, length: int) -> dict[str, Any]:
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("video upload must be between 1 byte and 80 GB")
        if not filename.lower().endswith((".mp4", ".mov", ".m4v", ".mkv", ".avi")):
            raise ValueError("upload a supported video file (.mp4, .mov, .m4v, .mkv, .avi)")
        name = Path(filename).name
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:160] or "match-video.mp4"
        run_dir = safe_run_dir(run_id)
        target = run_dir / "source-video" / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            current_state = self.read_state(run_id)
            if self.worker_busy or self.review_busy:
                raise ValueError("wait for the active stage to finish before uploading")
            if current_state.get("source_video"):
                raise ValueError("this run already has a source video; create a new run for another match")
        temporary = target.with_suffix(target.suffix + ".uploading")
        received = 0
        try:
            with temporary.open("wb") as output:
                while received < length:
                    chunk = body_reader(min(1024 * 1024, length - received))
                    if not chunk:
                        raise ValueError("video upload ended before all bytes arrived")
                    output.write(chunk)
                    received += len(chunk)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        with self.lock:
            state = self.read_state(run_id)
            state["source_video"] = str(target.relative_to(run_dir))
            for key in STAGE_ORDER:
                state["stages"][key] = fresh_stage()
            state["stages"]["stage2"]["review_status"] = "needed"
            state["active_stage"] = None
            state["active_status"] = "idle"
            self.invalidate(state, "stage0")
            self.save_state(state)
        return {"filename": safe_name, "size_bytes": received, "run": state}

    def artifact(self, run_id: str, relative: str) -> Path:
        run_dir = safe_run_dir(run_id)
        return relative_file(run_dir, relative)

    def check_dependencies(self, run_id: str, stage: str) -> list[str]:
        state = self.read_state(run_id)
        run_dir = safe_run_dir(run_id)
        missing = []
        for item in DEPENDENCIES[stage]:
            if item == "source_video":
                if not state.get("source_video") or not (run_dir / state["source_video"]).is_file():
                    missing.append("Upload the match video")
            elif item == "tracknet":
                settings = state["settings"]
                for label, key in (("TrackNet source", "tracknet_repo"), ("TrackNet Python", "tracknet_python"), ("TrackNet checkpoint", "tracknet_checkpoint")):
                    if not settings.get(key) or not Path(settings[key]).expanduser().is_file() and key != "tracknet_repo":
                        missing.append(f"Set a valid {label} path in match settings")
                if settings.get("tracknet_repo") and not (Path(settings["tracknet_repo"]).expanduser() / "predict.py").is_file():
                    missing.append("TrackNet source folder must contain predict.py")
            elif item == "pose_model":
                path = Path(state["settings"].get("pose_model", "")).expanduser()
                if not path.is_file():
                    missing.append("Set a valid MediaPipe pose model path")
            elif item == "yolox_model":
                if not (ROOT / "models" / "yolox_s.onnx").is_file():
                    missing.append("Add models/yolox_s.onnx (see the installation guide)")
            elif not self.artifact(run_id, item).is_file():
                missing.append(item)
        # Existing files can be stale after an upstream rerun. Keep manual stage
        # launches available, but require users to rerun stale producers first.
        producer = {
            "match-720p.mp4": "stage0", "dets.json": "stage1",
            "court_calibration.json": "stage2", "players.json": "stage3",
            "shuttle_tracks.json": "stage3a", "serves_reviewed.json": "stage4",
            "manual_score_review.json": "stage4", "points.json": "stage5",
            "rallies_review.json": "stage6", "motion.json": "stage7",
            "pose_stats.json": "stage8", "pickups.json": "stage9",
            "report_data.json": "stage10",
        }
        for item in DEPENDENCIES[stage]:
            producing_stage = producer.get(item)
            if producing_stage and state["stages"][producing_stage]["status"] == "stale":
                missing.append(f"Rerun {STAGE_INFO[producing_stage]['label']} because its output is stale")
        for prerequisite, require_review in STAGE_PREREQUISITES.get(stage, []):
            row = state["stages"][prerequisite]
            status = row.get("status")
            if status == "needs_attention":
                missing.append(f"Resolve {STAGE_INFO[prerequisite]['label']} score issues before continuing")
            elif status != "completed":
                missing.append(f"Complete {STAGE_INFO[prerequisite]['label']} first (current status: {status or 'unknown'})")
            elif require_review and row.get("review_status") != "done":
                missing.append(f"Complete the {STAGE_INFO[prerequisite]['label']} browser review first")

        settings = state["settings"]
        if settings.get("tracknet_enabled"):
            tracknet_consumers = {"stage4", "stage6", "stage7", "stage9", "stage11"}
            if stage in tracknet_consumers:
                row = state["stages"]["stage3a"]
                if row.get("status") != "completed":
                    missing.append("Complete Stage 3A TrackNet first because shuttle tracking is enabled")
                elif row.get("review_status") != "done":
                    missing.append("Complete the Stage 3A shuttle preview review first")
                if not (run_dir / "shuttle_tracks.json").is_file():
                    missing.append("shuttle_tracks.json is missing; rerun Stage 3A")
            if stage == "stage7" and not (run_dir / "net_plane_calibration.json").is_file():
                missing.append("Calibrate the Stage 7 net plane before movement analysis")
            elif stage == "stage7" and state.get("net_plane_calibration_review_status") != "done":
                missing.append("Review the Stage 7 net-plane calibration preview")

        if stage == "stage6":
            points_path = run_dir / "points.json"
            if points_path.is_file():
                try:
                    points = json.loads(points_path.read_text(encoding="utf-8"))
                    if not points.get("stage5_passed"):
                        missing.append("Stage 5 score validation has not passed")
                except (OSError, json.JSONDecodeError):
                    missing.append("Could not read Stage 5 score validation; rerun Stage 5")
        if stage == "stage11" and settings.get("ai_coach_enabled"):
            row = state["stages"]["stage10a"]
            if row.get("status") != "completed":
                missing.append("Complete Stage 10A AI coach before rendering the configured video")
            elif row.get("review_status") != "done":
                missing.append("Complete the Stage 10A narrative review first")
        return missing

    def invalidate(self, state: dict[str, Any], stage: str) -> None:
        for downstream in INVALIDATES[stage]:
            if downstream in state["stages"]:
                old = state["stages"][downstream]
                if old.get("status") not in ("not_started", "skipped"):
                    old.update({"status": "stale", "review_status": "needed" if STAGE_INFO[downstream]["review"] else "not_needed", "progress": None, "message": f"Needs rerun because {stage} changed."})

    def stage_command(self, run_id: str, stage: str, options: dict[str, Any]) -> list[str]:
        state = self.read_state(run_id)
        settings = state["settings"]
        run_dir = safe_run_dir(run_id)
        video = run_dir / "match-720p.mp4"
        players = run_dir / "players.json"
        rallies = run_dir / "rallies_review.json"
        points = run_dir / "points.json"
        shuttle = run_dir / "shuttle_tracks.json"
        script = lambda name: str(ROOT / "scripts" / name)
        py = sys.executable
        overwrite = ["--overwrite"]

        if stage == "stage0":
            if not state.get("source_video"):
                raise ValueError("upload a match video first")
            return [py, script("stage0_proxy.py"), str(run_dir / state["source_video"]), str(video), "--match-start-sec", str(settings["first_serve_sec"]), "--machine-progress", *overwrite]
        if stage == "stage1":
            command = [py, script("stage1_detect.py"), str(video), str(ROOT / "models" / "yolox_s.onnx"), str(run_dir / "dets.json"), str(run_dir / "detections_contact_sheet.jpg")]
            if options.get("execution_provider") in ("auto", "cpu", "coreml"):
                command.extend(["--execution-provider", options["execution_provider"]])
            if options.get("confidence"):
                command.extend(["--confidence", str(options["confidence"])])
            return [*command, *overwrite]
        if stage == "stage3":
            return [py, script("stage3_players.py"), str(video), str(run_dir / "dets.json"), str(run_dir / "court_calibration.json"), str(players), str(run_dir / "players_contact_sheet.jpg"), "--identity-garment", str(settings["identity_garment"]), "--identity-color", str(settings["identity_color"]), "--no-interactive-review", *overwrite]
        if stage == "stage3a":
            command = [py, script("stage3a_tracknet.py"), str(video), str(shuttle), "--tracknet-repo", str(Path(settings["tracknet_repo"]).expanduser()), "--tracknet-python", str(Path(settings["tracknet_python"]).expanduser()), "--checkpoint", str(Path(settings["tracknet_checkpoint"]).expanduser()), "--calibration-json", str(run_dir / "court_calibration.json")]
            if options.get("batch_size"):
                command.extend(["--batch-size", str(int(options["batch_size"]))])
            return [*command, *overwrite]
        if stage == "stage4":
            command = [py, script("stage4_serves.py"), str(video), str(players), str(run_dir / "serves.json"), str(run_dir / "serves_review"), "--no-interactive-review"]
            if options.get("still_threshold") not in (None, ""):
                command.extend(["--still-threshold", str(options["still_threshold"])])
            else:
                for key, arg in (("near_still_threshold", "--near-still-threshold"), ("far_still_threshold", "--far-still-threshold")):
                    if options.get(key) not in (None, ""):
                        command.extend([arg, str(options[key])])
            if settings.get("tracknet_enabled") and shuttle.is_file():
                command.extend(["--shuttle-tracks", str(shuttle)])
                if options.get("use_shuttle_server_cue"):
                    command.append("--use-shuttle-server-cue")
            return [*command, *overwrite]
        if stage == "stage5":
            return [py, script("stage5_score.py"), str(run_dir / "serves_reviewed.json"), str(points), "--manual-review", str(run_dir / "manual_score_review.json"), *overwrite]
        if stage == "stage6":
            command = [py, script("stage6_rallies.py"), str(video), str(points), str(run_dir / "rally_review"), str(rallies), "--estimate-rally-ends", str(players)]
            if options.get("pause_duration"):
                command.extend(["--pause-duration-sec", str(options["pause_duration"])])
            if settings.get("tracknet_enabled") and shuttle.is_file():
                command.extend(["--shuttle-tracks", str(shuttle)])
                if options.get("rally_end_source") == "shuttle":
                    command.extend(["--rally-end-source", "shuttle"])
            return [*command, *overwrite]
        if stage == "stage7":
            command = [py, script("stage7_movement.py"), str(players), str(rallies), str(run_dir / "motion.json"), str(run_dir / "movement_heatmaps")]
            if settings.get("tracknet_enabled") and shuttle.is_file():
                command.extend(["--shuttle-tracks", str(shuttle)])
                if (run_dir / "net_plane_calibration.json").is_file():
                    command.extend(["--net-plane-json", str(run_dir / "net_plane_calibration.json")])
            return [*command, *overwrite]
        if stage == "stage8":
            code = str(options.get("analysis_subject") or settings["analysis_subject"])
            return [py, script("stage8_pose.py"), str(video), str(players), str(rallies), str(Path(settings["pose_model"]).expanduser()), str(run_dir / "pose_stats.json"), str(run_dir / "pose_review.jpg"), "--player-code", code, "--racket-side", str(options.get("racket_side", "auto")), *overwrite]
        if stage == "stage9":
            command = [py, script("stage9_pickups.py"), str(video), str(players), str(rallies), str(run_dir / "pickups.json"), str(run_dir / "pickup_review.jpg")]
            if settings.get("tracknet_enabled") and shuttle.is_file():
                command.extend(["--shuttle-tracks", str(shuttle)])
            return [*command, *overwrite]
        if stage == "stage10":
            code = str(options.get("analysis_subject") or settings["analysis_subject"])
            # Stage 10 intentionally overwrites its two reports, and its CLI
            # does not expose the --overwrite switch used by other stages.
            return [py, script("stage10_report.py"), str(points), str(rallies), str(run_dir / "motion.json"), str(players), str(run_dir / "pose_stats.json"), str(run_dir / "pickups.json"), str(run_dir / "report_data.json"), str(run_dir / "coach_report.md"), "--player-code", code, "--r-name", str(settings["r_name"]), "--o-name", str(settings["o_name"])]
        if stage == "stage10a":
            command = [py, script("stage10a_ai_coach.py"), str(run_dir / "report_data.json"), "--player-code", str(options.get("analysis_subject") or settings["analysis_subject"]), "--mode", str(options.get("mode", settings.get("ai_mode", "context")))]
            selection = run_dir / "coach_review_selection.json"
            if selection.is_file():
                command.extend(["--review-selection-json", str(selection)])
            if options.get("model"):
                command.extend(["--model", str(options["model"])])
            return [*command, *overwrite]
        if stage == "stage11":
            code = str(options.get("analysis_subject") or settings["analysis_subject"])
            short_mode = options.get("short_mode", settings.get("short_mode", False))
            if not isinstance(short_mode, bool):
                raise ValueError("Stage 11 short_mode option must be true or false")
            output_name = "match-analysis-short.mp4" if short_mode else "match-analysis.mp4"
            command = [py, script("stage11_video.py"), str(video), str(players), str(rallies), str(run_dir / "pose_stats.json"), str(run_dir / "report_data.json"), str(run_dir / output_name), "--player-code", code, "--r-name", str(settings["r_name"]), "--o-name", str(settings["o_name"]), "--pose-model", str(Path(settings["pose_model"]).expanduser())]
            if short_mode:
                command.append("--short-mode")
            voice_enabled = options.get("voice_enabled", settings.get("voice_enabled", True))
            pose_enabled = options.get("pose_overlay_enabled", settings.get("pose_overlay_enabled", True))
            shuttle_overlay_enabled = options.get("shuttle_overlay_enabled", settings.get("shuttle_overlay_enabled", True))
            shot_count_enabled = options.get("shot_count_enabled", settings.get("shot_count_enabled", True))
            for name, value in (("voice_enabled", voice_enabled), ("pose_overlay_enabled", pose_enabled), ("shuttle_overlay_enabled", shuttle_overlay_enabled), ("shot_count_enabled", shot_count_enabled)):
                if not isinstance(value, bool):
                    raise ValueError(f"Stage 11 {name} option must be true or false")
            if not voice_enabled or options.get("no_voice"):
                command.append("--no-voice")
            if not pose_enabled or options.get("no_pose"):
                command.append("--no-pose")
            use_shuttle_tracks = settings.get("tracknet_enabled") and shuttle.is_file() and shuttle_overlay_enabled
            if use_shuttle_tracks:
                command.extend(["--shuttle-tracks", str(shuttle)])
                if not shot_count_enabled:
                    command.append("--no-shot-count")
            elif shot_count_enabled:
                command.append("--shot-count-only")
            return [*command, *overwrite]
        raise ValueError(f"unknown stage: {stage}")

    def run_stage(self, run_id: str, stage: str, options: dict[str, Any] | None = None) -> None:
        if stage not in STAGE_INFO or stage in ("stage2",):
            raise ValueError("choose an executable stage; calibration is performed in the browser")
        options = options or {}
        missing = self.check_dependencies(run_id, stage)
        if missing:
            raise ValueError("Stage prerequisites are missing: " + ", ".join(missing))
        run_dir = safe_run_dir(run_id)
        subject = str(options.get("analysis_subject") or self.read_state(run_id)["settings"]["analysis_subject"])
        if stage in ("stage10", "stage11") and (run_dir / "pose_stats.json").is_file():
            pose_data = json.loads((run_dir / "pose_stats.json").read_text(encoding="utf-8"))
            if pose_data.get("player_code", "R") != subject:
                raise ValueError(f"pose_stats.json is for {pose_data.get('player_code', 'R')}; run Stage 8 for {subject} first")
        if stage == "stage11" and (run_dir / "report_data.json").is_file():
            report_data = json.loads((run_dir / "report_data.json").read_text(encoding="utf-8"))
            if report_data.get("analysis_subject_code", "R") != subject:
                raise ValueError(f"report_data.json is for {report_data.get('analysis_subject_code', 'R')}; run Stage 10 for {subject} first")
        if stage == "stage10a" and (run_dir / "report_data.json").is_file():
            report_data = json.loads((run_dir / "report_data.json").read_text(encoding="utf-8"))
            if report_data.get("analysis_subject_code", "R") != subject:
                raise ValueError(f"report_data.json is for {report_data.get('analysis_subject_code', 'R')}; run Stage 10 for {subject} first")
        command = self.stage_command(run_id, stage, options)
        with self.lock:
            if self.worker_busy or self.review_busy:
                raise ValueError("a stage or review save is already active; wait for it to finish")
            state = self.read_state(run_id)
            if stage == "stage10":
                for old in [*run_dir.glob("AI-coach-context*.md"), *run_dir.glob("AI-coach-narrative*.md"), *run_dir.glob("coach_review_selection.resolved*.json")]:
                    old.unlink(missing_ok=True)
            elif stage == "stage10a":
                suffix = "-O" if subject == "O" else ""
                for old in (run_dir / f"AI-coach-narrative{suffix}.md", run_dir / f"coach_review_selection.resolved{suffix}.json"):
                    old.unlink(missing_ok=True)
            if stage == "stage4" and run_id in self.serve_apps:
                self.serve_apps.pop(run_id).close()
            self.invalidate(state, stage)
            stage_state = state["stages"][stage]
            stage_state.update({"status": "running", "review_status": "not_needed", "progress": None, "progress_total_seconds": None, "progress_total_frames": None, "message": "Starting…", "started_at": utc_now(), "finished_at": None})
            if stage == "stage11":
                stage_state["output_file"] = Path(command[7]).name
            state["active_stage"] = stage
            state["active_status"] = "running"
            state["last_command"] = [Path(command[1]).name, *command[2:]]
            self.save_state(state)
            self.worker_busy = True
            self.stop_requested = False
            worker = threading.Thread(target=self._worker, args=(run_id, stage, command), daemon=True)
            worker.start()

    def _worker(self, run_id: str, stage: str, command: list[str]) -> None:
        run_dir = safe_run_dir(run_id)
        log_path = run_dir / "logs" / f"{stage}.log"
        process: subprocess.Popen[str] | None = None
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self.lock:
                self.process = process
                self.process_run_id = run_id
                stop_immediately = self.stop_requested
            if stop_immediately:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (AttributeError, ProcessLookupError, PermissionError):
                    process.terminate()
            assert process.stdout is not None
            with log_path.open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    clean_line = line.rstrip("\n")
                    machine_progress_line = stage == "stage0" and re.fullmatch(r"[A-Za-z0-9_]+=.*", clean_line) is not None
                    if not machine_progress_line:
                        self._append_log(run_id, stage, clean_line)
                    self._update_progress_from_line(run_id, stage, line)
            return_code = process.wait()
            state = self.read_state(run_id)
            row = state["stages"][stage]
            row["finished_at"] = utc_now()
            if self.stop_requested:
                row.update({"status": "stopped", "progress": None, "message": "Stage stopped by user."})
                state["active_status"] = "idle"
            elif return_code != 0:
                row.update({"status": "failed", "progress": None, "message": f"Stage exited with code {return_code}. Check the log."})
                state["active_status"] = "failed"
            else:
                if stage == "stage11":
                    output_path = relative_file(run_dir, row.get("output_file", ""))
                    if not output_path.is_file() or output_path.stat().st_size == 0:
                        row.update({"status": "failed", "progress": None, "message": "Stage 11 exited successfully but did not create its selected video output."})
                        state["active_status"] = "failed"
                        state["active_stage"] = None
                        self.save_state(state)
                        return
                message = f"Stage finished. Output: {row['output_file']}" if stage == "stage11" else "Stage finished."
                row.update({"status": "completed", "progress": 100, "message": message})
                row["review_status"] = "needed" if STAGE_INFO[stage]["review"] else "not_needed"
                if stage == "stage5":
                    try:
                        points = json.loads((run_dir / "points.json").read_text(encoding="utf-8"))
                        if not points.get("stage5_passed"):
                            row.update({"status": "needs_attention", "message": "Score review did not pass. See the score report and correct the serve list."})
                    except (OSError, json.JSONDecodeError):
                        row.update({"status": "failed", "message": "Stage 5 did not create points.json."})
                state["active_status"] = "idle"
            state["active_stage"] = None
            self.save_state(state)
        except Exception as error:
            try:
                state = self.read_state(run_id)
                state["stages"][stage].update({"status": "failed", "progress": None, "message": str(error), "finished_at": utc_now()})
                state["active_stage"] = None
                state["active_status"] = "failed"
                self.save_state(state)
                self._append_log(run_id, stage, f"Runner error: {error}")
            except Exception:
                pass
        finally:
            with self.lock:
                self.process = None
                self.process_run_id = None
                self.worker_busy = False
                self.stop_requested = False

    def _append_log(self, run_id: str, stage: str, line: str) -> None:
        with self.lock:
            state = self.read_state(run_id)
            logs = state.setdefault("logs", [])
            logs.append({"stage": stage, "time": datetime.now().strftime("%H:%M:%S"), "line": line[:1000]})
            state["logs"] = logs[-240:]
            state["stages"][stage]["message"] = line[:240]
            self.save_state(state)

    def _update_progress_from_line(self, run_id: str, stage: str, line: str) -> None:
        duration_match = re.search(r"Proxy duration seconds:\s*([0-9.]+)", line, re.I)
        if duration_match:
            with self.lock:
                state = self.read_state(run_id)
                state["stages"][stage]["progress_total_seconds"] = float(duration_match.group(1))
                self.save_state(state)
        if stage == "stage3a":
            video_length_match = re.search(r"Video\s+length\s*[:=]?\s*(\d+)", line, re.I)
            batch_match = re.search(r"\b(\d+)\s*it\b", line, re.I)
            explicit_batches_match = re.search(r"\b(\d+)\s*/\s*(\d+)\s*(?:it|\[)", line, re.I)
            if video_length_match or batch_match or explicit_batches_match:
                with self.lock:
                    state = self.read_state(run_id)
                    row = state["stages"][stage]
                    if video_length_match:
                        row["progress_total_frames"] = int(video_length_match.group(1))
                    current_batches = int(
                        (explicit_batches_match or batch_match).group(1)
                    ) if (explicit_batches_match or batch_match) else None
                    total_batches = int(explicit_batches_match.group(2)) if explicit_batches_match else None
                    if current_batches is not None:
                        if total_batches:
                            row["progress"] = min(99, round(100 * current_batches / total_batches))
                        else:
                            total_frames = int(row.get("progress_total_frames", 0))
                            command = state.get("last_command", [])
                            batch_size = 8
                            if isinstance(command, list) and "--batch-size" in command:
                                try:
                                    batch_size = max(1, int(command[command.index("--batch-size") + 1]))
                                except (IndexError, TypeError, ValueError):
                                    pass
                            if total_frames > 0:
                                row["progress"] = min(99, round(100 * current_batches * batch_size / total_frames))
                    self.save_state(state)
        if stage == "stage11":
            progress = None
            phase_match = re.search(r"Pose overlay samples:?\s*(\d+)\s*/\s*(\d+)", line, re.I)
            if phase_match:
                current, total = (int(value) for value in phase_match.groups())
                progress = round(20 * current / total) if total > 0 else None
            else:
                phase_match = re.search(r"Coach voice segments prepared:?\s*(\d+)\s*/\s*(\d+)", line, re.I)
                if phase_match:
                    current, total = (int(value) for value in phase_match.groups())
                    progress = 20 + round(15 * current / total) if total > 0 else None
                else:
                    phase_match = re.search(r"Stage 11 render frames:?\s*(\d+)\s*/\s*(\d+)", line, re.I)
                    if phase_match:
                        current, total = (int(value) for value in phase_match.groups())
                        progress = 35 + round(60 * current / total) if total > 0 else None
                    elif "Stage 11 preparation complete" in line:
                        progress = 35
                    elif "Stage 11 audio mux started" in line:
                        progress = 96
                    elif "Stage 11 audio mux complete" in line:
                        progress = 99
            if progress is not None:
                with self.lock:
                    state = self.read_state(run_id)
                    state["stages"][stage]["progress"] = min(99, progress)
                    self.save_state(state)
            return
        time_match = re.search(r"out_time_us=(\d+)", line)
        if stage == "stage0" and time_match:
            with self.lock:
                state = self.read_state(run_id)
                total = float(state["stages"][stage].get("progress_total_seconds", 0))
                if total > 0:
                    current_sec = int(time_match.group(1)) / 1_000_000
                    state["stages"][stage]["progress"] = min(99, round(100 * current_sec / total))
                    self.save_state(state)
        match = self.progress_re.search(line)
        if not match:
            return
        current, total = int(match.group(1)), int(match.group(2))
        if total <= 0:
            return
        with self.lock:
            state = self.read_state(run_id)
            state["stages"][stage]["progress"] = min(99, round(100 * current / total))
            self.save_state(state)

    def stop(self, run_id: str) -> None:
        with self.lock:
            state = self.read_state(run_id)
            if not self.worker_busy or state.get("active_stage") is None:
                raise ValueError("no active stage belongs to this match")
            if self.process_run_id not in (None, run_id):
                raise ValueError("no active stage belongs to this match")
            self.stop_requested = True
            process = self.process if self.process_run_id == run_id else None
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (AttributeError, ProcessLookupError, PermissionError):
                    process.terminate()

    def mark_reviewed(self, run_id: str, stage: str) -> dict[str, Any]:
        with self.lock:
            if self.worker_busy or self.review_busy:
                raise ValueError("wait for the active stage or review save to finish")
            if stage == "stage7_calibration":
                run_dir = safe_run_dir(run_id)
                state = self.read_state(run_id)
                if not (run_dir / "net_plane_calibration.json").is_file():
                    raise ValueError("save the Stage 7 net-plane calibration before reviewing it")
                if state["stages"]["stage2"].get("status") != "completed" or state["stages"]["stage2"].get("review_status") != "done":
                    raise ValueError("complete and review Stage 2 court calibration first")
                state["net_plane_calibration_review_status"] = "done"
                state["net_plane_calibration_reviewed_at"] = utc_now()
                self.save_state(state)
                return state
            if stage not in INSPECTION_ONLY_REVIEWS:
                raise ValueError("this stage has editable review controls; save its review to complete the review gate")
            state = self.read_state(run_id)
            row = state["stages"].get(stage)
            if not row or row.get("status") != "completed":
                raise ValueError("run this stage before marking its review complete")
            row["review_status"] = "done"
            row["reviewed_at"] = utc_now()
            state["active_status"] = "idle"
            self.save_state(state)
            return state

    def begin_review_save(self) -> None:
        with self.lock:
            if self.worker_busy or self.review_busy:
                raise ValueError("a stage or review save is already active")
            self.review_busy = True

    def end_review_save(self) -> None:
        with self.lock:
            self.review_busy = False

    def continue_action(self, run_id: str) -> dict[str, Any]:
        state = self.read_state(run_id)
        if state.get("active_stage"):
            return {"action": "running", "stage": state["active_stage"]}
        if not state.get("source_video"):
            return {"action": "upload"}
        settings = state["settings"]
        for stage in FLOW_ORDER:
            if stage == "stage3a" and not settings.get("tracknet_enabled"):
                if state["stages"][stage]["status"] != "skipped":
                    state["stages"][stage].update({"status": "skipped", "review_status": "not_needed", "message": "TrackNet disabled for this match."})
                    self.save_state(state)
                continue
            if stage == "stage10a" and not settings.get("ai_coach_enabled"):
                if state["stages"][stage]["status"] != "skipped":
                    state["stages"][stage].update({"status": "skipped", "review_status": "not_needed", "message": "Optional AI coach skipped."})
                    self.save_state(state)
                continue
            if stage == "stage11" and not settings.get("video_enabled", True):
                if state["stages"][stage]["status"] != "skipped":
                    state["stages"][stage].update({"status": "skipped", "review_status": "not_needed", "message": "Analysis video disabled in match settings."})
                    self.save_state(state)
                continue
            row = state["stages"][stage]
            if row.get("status") == "completed" and row.get("review_status") != "needed":
                continue
            if row.get("status") == "completed" and row.get("review_status") == "needed":
                return {"action": "review", "stage": stage}
            if row.get("status") == "needs_attention":
                return {"action": "review", "stage": "stage4" if stage == "stage5" else stage}
            missing = self.check_dependencies(run_id, stage)
            calibration_issue = "Calibrate the Stage 7 net plane before movement analysis"
            calibration_review_issue = "Review the Stage 7 net-plane calibration preview"
            if stage == "stage7" and (calibration_issue in missing or calibration_review_issue in missing):
                other_missing = [
                    item for item in missing
                    if item not in (calibration_issue, calibration_review_issue)
                ]
                if other_missing:
                    return {"action": "blocked", "stage": stage, "missing": other_missing}
                if calibration_issue in missing:
                    return {"action": "calibrate_net_plane", "stage": stage}
                return {"action": "review_net_plane", "stage": stage}
            if missing:
                return {"action": "blocked", "stage": stage, "missing": missing}
            if stage == "stage2":
                return {"action": "calibrate", "stage": stage}
            return {"action": "run", "stage": stage}
        return {"action": "complete"}

    def delete_run(self, run_id: str) -> None:
        with self.lock:
            if self.worker_busy or self.review_busy:
                raise ValueError("stop the active stage before deleting this match")
            self.read_state(run_id)
            app = self.serve_apps.pop(run_id, None)
            if app:
                app.close()
            shutil.rmtree(safe_run_dir(run_id))

    def serve_app(self, run_id: str) -> ServeReviewApp:
        self.read_state(run_id)
        if run_id in self.serve_apps:
            return self.serve_apps[run_id]
        run_dir = safe_run_dir(run_id)
        video = run_dir / "match-720p.mp4"
        paths = (video, run_dir / "players.json", run_dir / "serves.json")
        if not all(path.is_file() for path in paths):
            raise ValueError("run Stage 4 before opening serve review")
        app = ServeReviewApp(
            video, run_dir / "players.json", run_dir / "serves.json",
            run_dir / "serves_review", run_dir / "manual_score_review.json",
            run_dir / "serves_reviewed.json",
        )
        self.serve_apps[run_id] = app
        return app


def frame_at(video_path: Path, timestamp: float) -> tuple[np.ndarray, float, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        capture.release()
        raise ValueError("video has invalid frame properties")
    duration = count / fps
    if not math.isfinite(timestamp) or timestamp < 0 or timestamp >= duration:
        capture.release()
        raise ValueError(f"frame time must be between 0 and {duration:.3f} seconds")
    index = min(count - 1, max(0, round(timestamp * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"could not read frame at {timestamp:.3f}s")
    return frame, index / fps, fps


def image_bytes(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("could not encode preview frame")
    return encoded.tobytes()


def review_payload(run_id: str, stage: str) -> dict[str, Any]:
    run_dir = safe_run_dir(run_id)
    files_by_stage = {
        "stage0": ["match-720p.mp4"],
        "stage1": ["detections_contact_sheet.jpg", "dets.json"],
        "stage2": ["court_overlay.png", "court_calibration.json"],
        "stage3": ["players_contact_sheet.jpg", "players.json"],
        "stage3a": ["shuttle_tracks.json", "shuttle_tracking_preview.mp4"],
        "stage6": ["rallies_review.json", "rally_review/index.html"],
        "stage7": ["motion.json", "movement_heatmaps/R_heatmap.png", "movement_heatmaps/O_heatmap.png", "net_plane_calibration.preview.jpg"],
        "stage8": ["pose_review.jpg", "pose_stats.json"],
        "stage9": ["pickup_review.jpg", "pickups.json"],
        "stage10": ["coach_report.md", "report_data.json", "coach_review_selection.json"],
        "stage10a": ["AI-coach-context.md", "AI-coach-context-O.md", "AI-coach-narrative.md", "AI-coach-narrative-O.md"],
        "stage11": [],
    }
    if stage == "stage11":
        state = json.loads((run_dir / "ui_state.json").read_text(encoding="utf-8"))
        output_name = state.get("stages", {}).get("stage11", {}).get("output_file")
        if not output_name:
            output_name = "match-analysis-short.mp4" if state.get("settings", {}).get("short_mode") else "match-analysis.mp4"
        files_by_stage["stage11"] = [output_name]
    files = []
    for relative in files_by_stage.get(stage, []):
        path = relative_file(run_dir, relative)
        if path.is_file():
            files.append({"path": relative, "name": path.name, "url": f"/api/runs/{run_id}/file?path={relative}"})
    payload: dict[str, Any] = {"stage": stage, "files": files}
    primary = {
        "stage6": "rallies_review.json", "stage7": "motion.json", "stage9": "pickups.json",
        "stage10": "report_data.json",
    }.get(stage)
    if primary:
        path = run_dir / primary
        if path.is_file():
            payload["data"] = json.loads(path.read_text(encoding="utf-8"))
    if stage == "stage10a":
        try:
            settings = json.loads((run_dir / "ui_state.json").read_text(encoding="utf-8")).get("settings", {})
        except (OSError, json.JSONDecodeError):
            settings = {}
        suffix = "-O" if settings.get("analysis_subject") == "O" else ""
        selected = run_dir / f"AI-coach-narrative{suffix}.md"
        if selected.is_file():
            payload["narrative"] = selected.read_text(encoding="utf-8")
            payload["narrative_path"] = selected.name
    return payload


class MatchRequestHandler(BaseHTTPRequestHandler):
    manager: RunManager
    lan_mode = False

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/"):
            super().log_message(format, *args)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: Any, status: int = 200) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def read_json(self, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > limit:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def run_state(self, run_id: str) -> dict[str, Any]:
        state = self.manager.read_state(run_id)
        calibration_path = safe_run_dir(run_id) / "net_plane_calibration.json"
        state["net_plane_calibration_exists"] = calibration_path.is_file()
        saved_calibration_review = state.get("net_plane_calibration_review_status")
        state["net_plane_calibration_review_status"] = (
            "done" if calibration_path.is_file() and saved_calibration_review == "done"
            else "needed" if calibration_path.is_file()
            else "not_needed"
        )
        state["stages"] = {
            key: {**row, "info": STAGE_INFO[key], "missing": self.manager.check_dependencies(run_id, key)}
            for key, row in state["stages"].items()
        }
        return state

    def serve_review_request(self, run_id: str, subpath: str) -> None:
        app = self.manager.serve_app(run_id)
        prefix = f"/runs/{run_id}/stage4"
        api_prefix = f"/api/runs/{run_id}/stage4"
        if subpath == "":
            html = SERVE_HTML_PATH.read_text(encoding="utf-8")
            html = html.replace("<body>", f'<body data-base-path="{api_prefix}">', 1)
            html = html.replace('fetch("/api/', 'fetch(basePath + "/api/')
            html = html.replace('`/api/', '`${basePath}/api/')
            self.send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            return
        local = subpath
        if local == "/api/state":
            value = app.get_state()
            for candidate in value.get("candidates", []):
                for key in ("image_url", "zoom_image_url"):
                    if candidate.get(key):
                        candidate[key] = prefix + candidate[key]
            self.send_json(value)
        elif local == "/api/frame":
            try:
                timestamp = float(parse_qs(urlparse(self.path).query).get("time", [""])[0])
                self.send_bytes(app.preview_frame(timestamp), "image/jpeg")
            except (ValueError, cv2.error) as error:
                self.send_json({"error": str(error)}, 400)
        elif local.startswith("/assets/"):
            name = local.rsplit("/", 1)[-1]
            if not re.fullmatch(r"candidate_\d{4}(?:_zoom)?\.jpg", name):
                self.send_json({"error": "unknown asset"}, 404)
                return
            path = app.assets_dir / name
            if not path.is_file():
                self.send_json({"error": "candidate preview unavailable"}, 404)
                return
            self.send_bytes(path.read_bytes(), "image/jpeg")
        else:
            self.send_json({"error": "not found"}, 404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            mounted_serve = re.fullmatch(r"/runs/([a-z0-9-]+)/stage4(.*)", path)
            if mounted_serve:
                subpath = mounted_serve.group(2)
                self.serve_review_request(mounted_serve.group(1), subpath)
                return
            if path == "/":
                self.send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                self.send_json({"runs": self.manager.list_runs(), "stages": STAGE_INFO})
                return
            match = re.fullmatch(r"/api/runs/([a-z0-9-]+)(?:/(.*))?", path)
            if match:
                run_id, action = match.group(1), match.group(2) or ""
                self.manager.read_state(run_id)
                if not action:
                    self.send_json(self.run_state(run_id))
                    return
                if action == "continue":
                    self.send_json(self.manager.continue_action(run_id))
                    return
                if action == "review":
                    self.send_json({"stages": STAGE_INFO, "state": self.run_state(run_id)})
                    return
                review_match = re.fullmatch(r"review/(stage[0-9a]+)", action)
                if review_match:
                    self.send_json(review_payload(run_id, review_match.group(1)))
                    return
                if action == "frame":
                    query = parse_qs(parsed.query)
                    timestamp = float(query.get("time", ["0"])[0])
                    run_dir = safe_run_dir(run_id)
                    state = self.manager.read_state(run_id)
                    video = run_dir / ("match-720p.mp4" if (run_dir / "match-720p.mp4").is_file() else state.get("source_video", ""))
                    frame, actual_time, _fps = frame_at(video, timestamp)
                    calibration_path = run_dir / "court_calibration.json"
                    if query.get("net", ["0"])[0] == "1" and calibration_path.is_file():
                        court = json.loads(calibration_path.read_text(encoding="utf-8"))
                        matrix = np.asarray(court.get("court_to_pixel"), dtype=np.float64)
                        width = float(court.get("court_size_m", {}).get("width", 6.1))
                        length = float(court.get("court_size_m", {}).get("length", 13.4))
                        left = project_plane(matrix, (0.0, length / 2.0))
                        right = project_plane(matrix, (width, length / 2.0))
                        cv2.line(frame, tuple(round(v) for v in left), tuple(round(v) for v in right), (0, 210, 255), 2, cv2.LINE_AA)
                    self.send_bytes(image_bytes(frame), "image/jpeg", extra={"X-Frame-Time": str(actual_time)})
                    return
                if action == "file":
                    relative = parse_qs(parsed.query).get("path", [""])[0]
                    target = self.manager.artifact(run_id, unquote(relative))
                    download = parse_qs(parsed.query).get("download", ["0"])[0] == "1"
                    self.send_file(target, download=download)
                    return
                if action.startswith("stage4"):
                    subpath = action[len("stage4"):]
                    if subpath and not subpath.startswith("/"):
                        self.send_json({"error": "not found"}, 404)
                    else:
                        self.serve_review_request(run_id, subpath)
                    return
            self.send_json({"error": "not found"}, 404)
        except (OSError, ValueError, TypeError, IndexError, KeyError, json.JSONDecodeError, cv2.error) as error:
            self.send_json({"error": str(error)}, 400)

    def send_file(self, path: Path, download: bool = False) -> None:
        if not path.is_file():
            self.send_json({"error": "file not found"}, 404)
            return
        size = path.stat().st_size
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        attachment_headers = {}
        if download:
            filename = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name) or "download"
            attachment_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        range_header = self.headers.get("Range")
        if range_header and re.fullmatch(r"bytes=\d*-\d*", range_header):
            start_raw, end_raw = range_header[6:].split("-", 1)
            if not start_raw and not end_raw:
                self.send_json({"error": "invalid byte range"}, 416)
                return
            if start_raw:
                start = int(start_raw)
                end = int(end_raw) if end_raw else size - 1
            else:
                suffix = int(end_raw)
                start, end = max(0, size - suffix), size - 1
            if start >= size or end < start:
                self.send_bytes(b"", mime, 416, {"Content-Range": f"bytes */{size}"})
                return
            end = min(end, size - 1)
            self.stream_file(path, mime, start, end - start + 1, 206, {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}", **attachment_headers})
            return
        self.stream_file(path, mime, 0, size, 200, {"Accept-Ranges": "bytes", **attachment_headers})

    def stream_file(self, path: Path, mime: str, offset: int, length: int, status: int, extra: dict[str, str]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        with path.open("rb") as source:
            source.seek(offset)
            remaining = length
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/runs":
                state = self.manager.create_run(self.read_json())
                self.send_json(state, 201)
                return
            match = re.fullmatch(r"/api/runs/([a-z0-9-]+)/(upload|settings|stage|stop|continue|reviewed|review/(stage[0-9a]+)|calibration/(court|net-plane)|stage4/api/save)", path)
            if not match:
                self.send_json({"error": "not found"}, 404)
                return
            run_id, action = match.group(1), match.group(2)
            self.manager.read_state(run_id)
            if action == "upload":
                filename = unquote(self.headers.get("X-Filename", "match-video.mp4"))
                result = self.manager.upload(run_id, filename, self.rfile.read, int(self.headers.get("Content-Length", "0")))
                self.send_json(result, 201)
                return
            if action == "settings":
                self.send_json(self.manager.update_settings(run_id, self.read_json()))
                return
            if action == "stage":
                payload = self.read_json()
                stage = str(payload.get("stage", ""))
                options = payload.get("options", {})
                if not isinstance(options, dict):
                    raise ValueError("stage options must be an object")
                self.manager.run_stage(run_id, stage, options)
                self.send_json({"ok": True, "state": self.run_state(run_id)}, 202)
                return
            if action == "stop":
                self.manager.stop(run_id)
                self.send_json({"ok": True})
                return
            if action == "continue":
                payload = self.read_json() if int(self.headers.get("Content-Length", "0")) > 0 else {}
                options = payload.get("options", {})
                if not isinstance(options, dict):
                    raise ValueError("stage options must be an object")
                action_info = self.manager.continue_action(run_id)
                if action_info["action"] == "run":
                    self.manager.run_stage(run_id, action_info["stage"], options)
                self.send_json(action_info, 202)
                return
            if action == "reviewed":
                payload = self.read_json()
                self.send_json(self.manager.mark_reviewed(run_id, str(payload.get("stage", ""))))
                return
            review_match = re.fullmatch(r"review/(stage[0-9a]+)", action)
            if review_match:
                self.manager.begin_review_save()
                try:
                    self.save_review(run_id, review_match.group(1), self.read_json())
                finally:
                    self.manager.end_review_save()
                return
            calibration_match = re.fullmatch(r"calibration/(court|net-plane)", action)
            if calibration_match:
                self.manager.begin_review_save()
                try:
                    self.save_calibration(run_id, calibration_match.group(1), self.read_json())
                finally:
                    self.manager.end_review_save()
                return
            if action == "stage4/api/save":
                self.manager.begin_review_save()
                try:
                    app = self.manager.serve_app(run_id)
                    result = app.save(self.read_json(2_000_000))
                    with self.manager.lock:
                        state = self.manager.read_state(run_id)
                        state["stages"]["stage4"].update({"status": "completed", "review_status": "done", "message": "Serve and score review saved."})
                        self.manager.invalidate(state, "stage4")
                        self.manager.save_state(state)
                    self.send_json(result)
                finally:
                    self.manager.end_review_save()
        except (OSError, ValueError, TypeError, IndexError, KeyError, json.JSONDecodeError, cv2.error, np.linalg.LinAlgError) as error:
            self.send_json({"error": str(error)}, 400)

    def do_DELETE(self) -> None:
        match = re.fullmatch(r"/api/runs/([a-z0-9-]+)", urlparse(self.path).path)
        if not match:
            self.send_json({"error": "not found"}, 404)
            return
        try:
            self.manager.read_state(match.group(1))
            self.manager.delete_run(match.group(1))
            self.send_json({"ok": True})
        except (OSError, ValueError) as error:
            self.send_json({"error": str(error)}, 400)

    def save_calibration(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        run_dir = safe_run_dir(run_id)
        video = run_dir / "match-720p.mp4"
        timestamp = float(payload.get("frame_time_sec", 0))
        frame, actual_time, _fps = frame_at(video, timestamp)
        if kind == "court":
            raw_clicks = payload.get("clicks")
            omitted = payload.get("omitted", [])
            if not isinstance(raw_clicks, list) or not isinstance(omitted, list):
                raise ValueError("court calibration needs clicked points and skipped landmark names")
            skipped = {str(name) for name in omitted}
            landmarks = [row for row in LANDMARKS if row[0] not in skipped]
            if len(landmarks) < 4 or len(raw_clicks) != len(landmarks) or any(not isinstance(point, list) or len(point) != 2 for point in raw_clicks):
                raise ValueError("select one pixel point for each remaining landmark (minimum four)")
            clicks = [(float(point[0]), float(point[1])) for point in raw_clicks]
            height, width = frame.shape[:2]
            if any(not math.isfinite(v) or v < 0 or v >= limit for point in clicks for v, limit in zip(point, (width, height))):
                raise ValueError("court clicks must be finite pixels inside the displayed frame")
            _overlay, rms = save_calibration(
                frame, clicks, landmarks, video.resolve(), actual_time,
                run_dir / "court_calibration.json", run_dir / "court_overlay.png",
            )
            # The optional net-plane transform was reviewed against the old
            # court calibration. Force a new browser calibration after Stage 2
            # changes so it cannot silently be reused with mismatched geometry.
            (run_dir / "net_plane_calibration.json").unlink(missing_ok=True)
            (run_dir / "net_plane_calibration.preview.jpg").unlink(missing_ok=True)
            with self.manager.lock:
                state = self.manager.read_state(run_id)
                state["net_plane_calibration_review_status"] = "not_needed"
                state.pop("net_plane_calibration_reviewed_at", None)
                row = state["stages"]["stage2"]
                row.update({"status": "completed", "review_status": "needed", "progress": 100, "message": f"Court calibration saved; inspect the overlay (RMS {rms:.2f}px)."})
                self.manager.invalidate(state, "stage2")
                self.manager.save_state(state)
            self.send_json({"ok": True, "rms_error_px": rms, "overlay": f"/api/runs/{run_id}/file?path=court_overlay.png"})
            return
        calibration_path = run_dir / "court_calibration.json"
        if not calibration_path.is_file():
            raise ValueError("complete Stage 2 court calibration first")
        state = self.manager.read_state(run_id)
        if state["stages"]["stage2"].get("status") != "completed" or state["stages"]["stage2"].get("review_status") == "needed":
            raise ValueError("complete and review Stage 2 court calibration before calibrating the net plane")
        court = json.loads(calibration_path.read_text(encoding="utf-8"))
        court_size = court.get("court_size_m", {})
        width = float(court_size.get("width", 6.1))
        length = float(court_size.get("length", 13.4))
        net_height = float(payload.get("net_top_height_m", 1.55))
        flight_height = float(payload.get("crossing_height_m", 4.5))
        clicks_raw = payload.get("clicks")
        if not isinstance(clicks_raw, list) or len(clicks_raw) != 4 or any(not isinstance(point, list) or len(point) != 2 for point in clicks_raw):
            raise ValueError("net calibration needs four points: floor left/right, tape left/right")
        click_values = [[float(p[0]), float(p[1])] for p in clicks_raw]
        if any(not math.isfinite(value) or value < 0 for point in click_values for value in point):
            raise ValueError("net-plane clicks must be finite non-negative pixels")
        clicks = [[int(round(point[0])), int(round(point[1]))] for point in click_values]
        if net_height <= 0 or flight_height <= net_height or clicks[0][0] >= clicks[1][0] or clicks[2][0] >= clicks[3][0]:
            raise ValueError("invalid net height or net corner ordering")
        h, w = frame.shape[:2]
        for bottom, top, label in ((clicks[0], clicks[2], "left"), (clicks[1], clicks[3], "right")):
            gap = bottom[1] - top[1]
            if top[1] <= 10 or gap < 15 or gap > 0.45 * h or abs(top[0] - bottom[0]) > 0.25 * w:
                raise ValueError(f"{label} tape point must be above and close to its floor point; click the physical net tape")
        world = np.asarray([[0, 0], [width, 0], [0, net_height], [width, net_height]], dtype=np.float64)
        matrix, _mask = cv2.findHomography(world, np.asarray(clicks, dtype=np.float64), method=0)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("could not calculate net-plane homography")
        extended_left = project_plane(matrix, (0.0, flight_height))
        extended_right = project_plane(matrix, (width, flight_height))
        polygon = [clicks[0], clicks[1], list(extended_right), list(extended_left)]
        if any(x >= w or y >= h for x, y in clicks) or min(float(point[1]) for point in polygon) < -0.45 * h:
            raise ValueError("net-plane points or projected corridor fall outside the frame; check the four clicks")
        if any(not math.isfinite(float(v)) for point in polygon for v in point) or abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32)))) < 100:
            raise ValueError("net-plane extension is invalid; check the four physical net corners")
        output = {
            "schema_version": 1, "source_video": video.name,
            "source_court_calibration": calibration_path.name,
            "frame_time_sec": round(actual_time, 3), "frame_size_px": {"width": w, "height": h},
            "court_width_m": width, "net_top_height_m": net_height,
            "crossing_height_m": flight_height, "point_order": list(POINT_LABELS),
            "net_plane_corner_px": clicks, "net_plane_homography_xz_to_pixel": matrix.tolist(),
            "extended_plane_polygon_px": polygon,
            "calibration_note": "The four clicks mark the physical net rectangle; the polygon is an image-space crossing corridor.",
        }
        atomic_json(run_dir / "net_plane_calibration.json", output)
        preview = frame.copy()
        cv2.polylines(preview, [np.asarray([[round(x), round(y)] for x, y in polygon], dtype=np.int32)], True, (60, 240, 90), 2, cv2.LINE_AA)
        court_matrix = np.asarray(court.get("court_to_pixel"), dtype=np.float64)
        floor_left = project_plane(court_matrix, (0.0, length / 2.0))
        floor_right = project_plane(court_matrix, (width, length / 2.0))
        cv2.line(preview, tuple(round(v) for v in floor_left), tuple(round(v) for v in floor_right), (0, 210, 255), 2, cv2.LINE_AA)
        cv2.line(preview, tuple(clicks[0]), tuple(clicks[1]), (0, 210, 255), 2, cv2.LINE_AA)
        cv2.line(preview, tuple(clicks[2]), tuple(clicks[3]), (255, 180, 30), 2, cv2.LINE_AA)
        cv2.imwrite(str(run_dir / "net_plane_calibration.preview.jpg"), preview)
        with self.manager.lock:
            state = self.manager.read_state(run_id)
            state["net_plane_calibration_review_status"] = "needed"
            state.pop("net_plane_calibration_reviewed_at", None)
            movement = state["stages"]["stage7"]
            if movement.get("status") not in ("not_started", "skipped"):
                movement.update({"status": "stale", "review_status": "needed", "message": "Rerun movement analysis after changing net-plane calibration."})
            self.manager.invalidate(state, "stage7")
            self.manager.save_state(state)
        self.send_json({"ok": True, "preview": f"/api/runs/{run_id}/file?path=net_plane_calibration.preview.jpg"})

    def save_review(self, run_id: str, stage: str, payload: dict[str, Any]) -> None:
        run_dir = safe_run_dir(run_id)
        if stage == "stage3":
            path = run_dir / "players.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("swap_players") is True:
                for frame in document.get("frames", []):
                    frame["r"], frame["o"] = frame.get("o"), frame.get("r")
                    frame["rb"], frame["ob"] = frame.get("ob"), frame.get("rb")
                document["player_identity"] = {
                    **document.get("player_identity", {}),
                    "R": "manually selected player after swapping the color-based assignment",
                    "O": "the other tracked player",
                    "review_action": "swapped in browser review",
                }
            else:
                document["player_identity"] = {
                    **document.get("player_identity", {}),
                    "review_action": "confirmed in browser review",
                }
            atomic_json(path, document)
            self.after_review_edit(run_id, stage)
            self.send_json({"ok": True, "swapped": payload.get("swap_players") is True})
            return
        if stage == "stage6":
            path = run_dir / "rallies_review.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            updates = payload.get("points")
            if not isinstance(updates, list):
                raise ValueError("rally review needs a points array")
            by_number = {int(row["point_n"]): row for row in document.get("points", [])}
            for update in updates:
                number = int(update["point_n"])
                row = by_number.get(number)
                if row is None:
                    raise ValueError(f"unknown rally point {number}")
                end_t = float(update["rally_end_sec"])
                serve_t = float(row["serve_t"])
                limit = float(row.get("next_serve_t") or document.get("video_duration_sec", end_t))
                if not math.isfinite(end_t) or end_t < serve_t or end_t >= limit:
                    raise ValueError(f"s{number} end time must be between its serve and the next serve")
                row["rally_end_sec"] = round(end_t, 3)
                row["rally_duration_sec"] = round(end_t - serve_t, 3)
                row["rally_end_method"] = "browser_manual_review"
                row["rally_end_estimated"] = False
                row["rally_end_confidence"] = "reviewed"
            atomic_json(path, document)
            self.after_review_edit(run_id, stage)
            self.send_json({"ok": True})
            return
        if stage == "stage7":
            path = run_dir / "motion.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            updates = payload.get("points")
            if not isinstance(updates, list):
                raise ValueError("movement review needs a points array")
            by_number = {int(row["point_n"]): row for row in document.get("points", [])}
            for update in updates:
                row = by_number.get(int(update["point_n"]))
                if row is None:
                    raise ValueError("unknown movement point")
                value = update.get("manual_shot_count")
                if value in (None, ""):
                    analysis = row.get("shuttle_shot_analysis")
                    if isinstance(analysis, dict):
                        analysis["manual_shot_count"] = None
                        if analysis.get("shot_count_source") == "manual_review":
                            analysis.pop("shot_count_source", None)
                    continue
                count = int(value)
                if count < 0:
                    raise ValueError("shot count must be zero or greater")
                analysis = row.get("shuttle_shot_analysis")
                if not isinstance(analysis, dict):
                    analysis = {}
                    row["shuttle_shot_analysis"] = analysis
                analysis["manual_shot_count"] = count
                analysis["shot_count_source"] = "manual_review"
            atomic_json(path, document)
            self.after_review_edit(run_id, stage)
            self.send_json({"ok": True})
            return
        if stage == "stage9":
            path = run_dir / "pickups.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            updates = payload.get("points")
            if not isinstance(updates, list):
                raise ValueError("pickup review needs a points array")
            by_number = {int(row["point_n"]): row for row in document.get("points", [])}
            for update in updates:
                row = by_number.get(int(update["point_n"]))
                if row is None:
                    raise ValueError("unknown pickup point")
                side, status = update.get("landing_side"), update.get("status")
                if side not in ("R", "O", None) or status not in ("detected", "ambiguous", "not_found"):
                    raise ValueError("choose a valid landing side and evidence status")
                row["landing_side"] = side
                row["landing_side_label"] = {"R": "R analyzed player side", "O": "O opponent side"}.get(side)
                row["status"] = status
                winner = row.get("winner")
                row["relation_to_winner_side"] = (
                    "winner_side" if side == winner else "losing_player_side"
                ) if side in ("R", "O") and winner in ("R", "O") else None
                row["manual_review"] = True
            atomic_json(path, document)
            self.after_review_edit(run_id, stage)
            self.send_json({"ok": True})
            return
        if stage == "stage10":
            if payload.get("automatic") is True:
                (run_dir / "coach_review_selection.json").unlink(missing_ok=True)
                for stale in run_dir.glob("coach_review_selection.resolved*.json"):
                    stale.unlink(missing_ok=True)
                state = self.manager.read_state(run_id)
                suffix = "-O" if state["settings"].get("analysis_subject") == "O" else ""
                (run_dir / f"AI-coach-narrative{suffix}.md").unlink(missing_ok=True)
                self.after_review_edit(run_id, stage)
                self.send_json({"ok": True, "selection_source": "automatic"})
                return
            selection = payload.get("categories")
            if not isinstance(selection, dict):
                raise ValueError("choose supported coaching categories")
            template = json.loads((ROOT / "templates" / "coach_review_selection.json").read_text(encoding="utf-8"))
            allowed = template["categories"]
            if any(key not in allowed or not isinstance(value, bool) for key, value in selection.items()):
                raise ValueError("coaching categories must use supported names and true/false values")
            categories = {key: selection.get(key, False) for key in allowed}
            for stale in run_dir.glob("coach_review_selection.resolved*.json"):
                stale.unlink(missing_ok=True)
            state = self.manager.read_state(run_id)
            suffix = "-O" if state["settings"].get("analysis_subject") == "O" else ""
            (run_dir / f"AI-coach-narrative{suffix}.md").unlink(missing_ok=True)
            atomic_json(run_dir / "coach_review_selection.json", {"version": 1, "categories": categories, "category_help": template["category_help"]})
            self.after_review_edit(run_id, stage)
            self.send_json({"ok": True, "categories": categories})
            return
        if stage == "stage10a":
            narrative = payload.get("narrative")
            if not isinstance(narrative, str) or len(narrative) > 500_000:
                raise ValueError("coach narrative must be text under 500 KB")
            state = self.manager.read_state(run_id)
            subject = state["settings"].get("analysis_subject", "R")
            suffix = "-O" if subject == "O" else ""
            subject_match = re.search(r"<!--\s*analysis_subject_code:\s*([RO])\s*-->", narrative, re.IGNORECASE)
            if subject_match and subject_match.group(1).upper() != subject:
                raise ValueError(f"coach narrative metadata identifies {subject_match.group(1).upper()}, but this run analyzes {subject}")
            if not subject_match:
                narrative = f"<!-- analysis_subject_code: {subject} -->\n\n" + narrative
            path = run_dir / f"AI-coach-narrative{suffix}.md"
            path.write_text(narrative, encoding="utf-8")
            self.after_review_edit(run_id, stage)
            self.send_json({"ok": True, "path": path.name})
            return
        raise ValueError(f"{stage} does not have editable review fields")

    def after_review_edit(self, run_id: str, stage: str) -> None:
        with self.manager.lock:
            state = self.manager.read_state(run_id)
            state["stages"][stage]["review_status"] = "done"
            state["stages"][stage]["reviewed_at"] = utc_now()
            self.manager.invalidate(state, stage)
            self.manager.save_state(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the browser-driven badminton match pipeline.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--lan", action="store_true", help="allow access from the trusted local network")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    manager = RunManager()
    address = "0.0.0.0" if args.lan else "127.0.0.1"
    handler = type("BoundMatchRequestHandler", (MatchRequestHandler,), {"manager": manager, "lan_mode": args.lan})
    try:
        server = ThreadingHTTPServer((address, args.port), handler)
    except OSError as error:
        print(f"Could not start match UI: {error}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Badminton match UI: {url}")
    if args.lan:
        print("Trusted-LAN access is enabled; open this machine's LAN IP on the same network.")
    print("One stage runs at a time. Keep this terminal open; press Ctrl+C to stop the server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMatch UI stopped.")
    finally:
        server.server_close()
        for app in manager.serve_apps.values():
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
