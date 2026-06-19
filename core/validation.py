"""Validation / test mode — record annotated footage + per-detection log for offline analysis.

The production system deliberately stores no images. Validation mode is the explicit
exception used to measure identification accuracy in a lab experiment: for the duration
of a named session it records, per camera, the *annotated* video (boxes + names +
confidence) and a structured JSONL with one record per detected face per frame —
including the raw cosine distance and the nearest template, so a threshold sweep
(FAR/FRR/DET/EER) can be computed offline without re-running the cameras.

All artifacts live under {data_dir}/validation/<session_id>/, OUTSIDE the biometric DB.
A single `ValidationManager` singleton is driven from the web UI / CLI; the pipeline
calls `record()` once per processed frame while a session is active.
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from loguru import logger

from config.settings import get_settings

_settings = get_settings()
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
_NOMINAL_FPS = 15.0  # playback fps; review syncs by frame_index, not wall-clock


def validation_root() -> Path:
    return Path(_settings.data_dir) / "validation"


class _CameraSink:
    def __init__(self, video_path: Path):
        self._video_path = video_path
        self._writer: Optional[cv2.VideoWriter] = None
        self.frame_index = -1  # incremented to 0 on first frame

    def write(self, annotated):
        if self._writer is None:
            h, w = annotated.shape[:2]
            self._writer = cv2.VideoWriter(str(self._video_path), _FOURCC, _NOMINAL_FPS, (w, h))
        self.frame_index += 1
        self._writer.write(annotated)
        return self.frame_index

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class ValidationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._session_id: Optional[str] = None
        self._dir: Optional[Path] = None
        self._jsonl = None
        self._sinks: Dict[str, _CameraSink] = {}
        self._threshold: float = _settings.match_threshold
        self._meta: dict = {}

    # ── lifecycle ───────────────────────────────────────────────────────────
    def is_active(self, camera_id: Optional[str] = None) -> bool:
        return self._active

    def start(self, name: str, enrolled: List[dict], threshold: float) -> dict:
        with self._lock:
            if self._active:
                raise RuntimeError("Sessione di validazione già attiva")
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if name:
                safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40]
                if safe:
                    session_id = f"{session_id}_{safe}"
            self._dir = validation_root() / session_id
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / "video").mkdir(exist_ok=True)  # footage kept in a subfolder (§6 layout)
            self._jsonl = (self._dir / "detections.jsonl").open("w", encoding="utf-8")
            self._sinks = {}
            self._threshold = threshold
            self._session_id = session_id
            self._meta = {
                "session_id": session_id,
                "name": name,
                "start": datetime.now().isoformat(),
                "end": None,
                "threshold": threshold,
                "enrolled": enrolled,
                "cameras": [],
            }
            self._write_manifest()
            self._active = True
            logger.success(f"[Validation] Sessione avviata: {session_id} ({self._dir})")
            return {"session_id": session_id, "dir": str(self._dir)}

    def stop(self) -> dict:
        with self._lock:
            if not self._active:
                return {"active": False}
            for sink in self._sinks.values():
                sink.close()
            if self._jsonl:
                self._jsonl.flush()
                self._jsonl.close()
            self._meta["end"] = datetime.now().isoformat()
            self._meta["cameras"] = sorted(self._sinks.keys())
            self._meta["frames"] = {cam: s.frame_index + 1 for cam, s in self._sinks.items()}
            self._write_manifest()
            sid = self._session_id
            self._active = False
            self._jsonl = None
            self._sinks = {}
            logger.success(f"[Validation] Sessione terminata: {sid}")
            return {"active": False, "session_id": sid}

    def status(self) -> dict:
        with self._lock:
            if not self._active:
                return {"active": False}
            return {
                "active": True,
                "session_id": self._session_id,
                "threshold": self._threshold,
                "frames": {cam: s.frame_index + 1 for cam, s in self._sinks.items()},
            }

    def _write_manifest(self):
        (self._dir / "session.json").write_text(
            json.dumps(self._meta, indent=2), encoding="utf-8"
        )

    # ── per-frame recording ───────────────────────────────────────────────────
    def record(self, camera_id: str, frame, results, details: List[dict]) -> None:
        """Write one annotated video frame + the JSONL detection records for it.

        `results` is the pipeline's [(location, pid, name, conf), ...]; `details` the
        aligned list of per-face dicts (raw distance, best match, bbox). Called for every
        processed frame while active — including frames with zero detections (continuous video).
        """
        if not self._active:
            return
        from ui.display import annotate_frame  # lazy: avoids core↔ui import cycle

        annotated = annotate_frame(frame, results)
        ts = time.time()
        with self._lock:
            if not self._active:
                return
            sink = self._sinks.get(camera_id)
            if sink is None:
                sink = _CameraSink(self._dir / "video" / f"cam_{camera_id}.mp4")
                self._sinks[camera_id] = sink
            frame_index = sink.write(annotated)
            for face_id, d in enumerate(details):
                rec = {
                    "session_id": self._session_id,
                    "timestamp_ms": int(round(ts * 1000)),
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "face_id": face_id,
                    "predicted_identity": d["predicted_identity"],
                    "predicted_person_id": d["predicted_person_id"],
                    "confidence": d["confidence"],
                    "raw_cosine_distance": d["raw_cosine_distance"],
                    "best_match_person_id": d["best_match_person_id"],
                    "candidates": d.get("candidates", []),
                    "bbox": d["bbox"],
                }
                self._jsonl.write(json.dumps(rec) + "\n")
            self._jsonl.flush()


validation_manager = ValidationManager()
