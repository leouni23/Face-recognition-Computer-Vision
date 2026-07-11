import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config.settings import get_settings
from core.detector import FaceLocation, detect_and_encode, detect_faces, embed_faces
from core.geometry import project_point, project_polar
from core.jetson_sysfs import sysfs_telemetry
from core.perfcounters import perf_counters
from core.profile import get_profile
from core.metrics import PerfTracker
from core.notifier import get_notifier
from core.recognizer import FaceRecognizer, Identity
from core.validation import validation_manager
from database.repository import CalibrationRepository, PersonRepository
from database.session import get_session

_settings = get_settings()

RecognitionResult = Tuple[FaceLocation, Optional[int], str, float]


class FaceIdPipeline:
    _TEMPLATE_RELOAD_INTERVAL = 60   # seconds between template refreshes
    _LOG_COOLDOWN = 10               # seconds between DB event writes per person
    _EMA_ALPHA = 0.4                 # smoothing of map positions (face size is noisy)
    _EMA_RESET_AFTER = 5.0           # seconds without sightings → restart smoothing
    _UNKNOWN_GRACE = 0.6             # seconds: brief detection gaps don't reset the unknown streak

    def __init__(self):
        self._recognizer = FaceRecognizer(threshold=_settings.match_threshold)
        self._last_reload = 0.0
        self._reload_lock = threading.Lock()
        self._last_logged: Dict[Optional[int], float] = defaultdict(float)
        self._last_position: Dict[Tuple[str, int], float] = defaultdict(float)
        self._projectors: Dict[str, dict] = {}  # camera_id → {"mode", ...params}
        self._world_ema: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
        self.perf = PerfTracker()  # rolling-window FPS / per-stage latency (read by /api/metrics)
        self._unknown_buf: Dict[str, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=30))
        self._last_unknown_alert: Dict[str, float] = defaultdict(float)
        self._unknown_since: Dict[str, float] = {}      # start of the current unknown streak
        self._unknown_last_seen: Dict[str, float] = {}  # last frame with an unknown (for grace)
        self._start_monotonic = time.monotonic()        # for the startup warm-up window
        # Optimized-TX2 per-camera state (unused by the Standard profile)
        self._frame_count: Dict[str, int] = {}          # for frame-skip
        self._last_results: Dict[str, List[RecognitionResult]] = {}  # reused on skipped frames
        self._trackers: Dict[str, object] = {}          # camera_id → FaceTracker

    def force_reload(self) -> None:
        self._last_reload = 0.0
        # Drop optimized per-camera state so a profile switch doesn't carry stale tracks/caches.
        self._frame_count.clear()
        self._last_results.clear()
        self._trackers.clear()

    def _reload_templates_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_reload < self._TEMPLATE_RELOAD_INTERVAL:
            return
        with self._reload_lock:
            if now - self._last_reload < self._TEMPLATE_RELOAD_INTERVAL:
                return
            with get_session() as session:
                # Only templates from the active profile's recognition model (embeddings from a
                # different pack are in an incompatible space → would never match).
                templates = PersonRepository(session).load_all_templates(get_profile().model_pack)
                projectors = {
                    p["camera_id"]: p["params"]
                    for p in CalibrationRepository(session).load_projectors()
                }
            self._recognizer.load(templates)
            self._projectors = projectors
            self._last_reload = now
            logger.debug(
                f"Template ricaricati: {len(templates)} persona/e, "
                f"{len(projectors)} camera/e calibrate"
            )

    def _should_log(self, person_id: Optional[int]) -> bool:
        now = time.monotonic()
        if now - self._last_logged[person_id] > self._LOG_COOLDOWN:
            self._last_logged[person_id] = now
            return True
        return False

    def _should_log_position(self, camera_id: str, person_id: int) -> bool:
        now = time.monotonic()
        key = (camera_id, person_id)
        if now - self._last_position[key] > _settings.position_log_interval:
            self._last_position[key] = now
            return True
        return False

    def _project_world(
        self, camera_id: str, loc: FaceLocation, frame_w: int
    ) -> Optional[Tuple[float, float]]:
        """Map the face onto the floor plan using the camera's calibration mode."""
        params = self._projectors.get(camera_id)
        if params is None:
            return None
        top, right, bottom, left = loc
        if params["mode"] == "polar":
            x_norm = ((left + right) / 2.0) / float(frame_w)
            return project_polar(params, x_norm, float(bottom - top))
        # homography: project the bbox bottom-centre (closest point to the floor plane)
        return project_point(params["H"], (left + right) / 2.0, float(bottom))

    def _update_unknown_alert(
        self, camera_id: str, unknown_embeddings: List[np.ndarray]
    ) -> None:
        """Alert only after a subject has been continuously 'unknown' for >= a minimum
        duration — so a brief head turn (sub-second) does not trigger a notification.

        The streak resets the moment a frame has no unknown face (e.g. the person turns
        back and is recognised, or leaves)."""
        now = time.monotonic()
        if unknown_embeddings:
            self._unknown_buf[camera_id].extend(unknown_embeddings)
            self._unknown_last_seen[camera_id] = now
            self._unknown_since.setdefault(camera_id, now)
        else:
            # reset only after a SUSTAINED absence — a single missed frame (detection
            # flicker) must not restart the timer, otherwise the alert is delayed for seconds
            if (camera_id in self._unknown_since
                    and now - self._unknown_last_seen.get(camera_id, 0.0) > self._UNKNOWN_GRACE):
                self._unknown_since.pop(camera_id, None)
                self._unknown_buf[camera_id].clear()
            return

        started = self._unknown_since[camera_id]
        if (now - self._start_monotonic >= _settings.unknown_alert_warmup   # skip startup warm-up
                and now - started >= _settings.unknown_alert_min_duration
                and now - self._last_unknown_alert[camera_id] > _settings.unknown_alert_cooldown):
            buf = self._unknown_buf[camera_id]
            if not buf:
                return
            mean = np.mean(np.stack(buf), axis=0)
            self._last_unknown_alert[camera_id] = now
            buf.clear()
            threading.Thread(
                target=get_notifier().alert_unknown, args=(camera_id, mean), daemon=True
            ).start()

    def _smooth_world(
        self, camera_id: str, person_id: int, world: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Exponential moving average — face pixel size jitters, so raw distances do too."""
        key = (camera_id, person_id)
        now = time.monotonic()
        prev = self._world_ema.get(key)
        if prev is not None and now - prev[2] < self._EMA_RESET_AFTER:
            a = self._EMA_ALPHA
            world = (a * world[0] + (1 - a) * prev[0], a * world[1] + (1 - a) * prev[1])
        self._world_ema[key] = (world[0], world[1], now)
        return world

    def _recognize(self, face_data, camera_id, frame, recording, notify_unknown):
        """Shared per-face recognition: match each embedding, log events/positions, build the
        (results, details, unknown_embeddings, match_ms). Matching is cheap (vector ops); used
        by both the Standard and Optimized paths once `face_data = [(loc, embedding)]` is ready."""
        threshold = self._recognizer.threshold
        match_ms = 0.0
        results: List[RecognitionResult] = []
        details: List[dict] = []
        unknown_embeddings: List[np.ndarray] = []
        with get_session() as session:
            repo = PersonRepository(session)
            for loc, embedding in face_data:
                t_m = time.perf_counter()
                best = self._recognizer.match(embedding)  # nearest template (pre-threshold)
                match_ms += (time.perf_counter() - t_m) * 1000.0

                if best is None:
                    pid, name, conf, raw_dist, best_pid = None, "Sconosciuto", 0.0, None, None
                else:
                    best_pid, best_name, best_dist = best
                    conf = float(min(1.0, max(0.0, 1.0 - best_dist)))
                    raw_dist = best_dist
                    pid, name = (best_pid, best_name) if best_dist < threshold else (None, "Sconosciuto")

                if self._should_log(pid):
                    repo.log_event(camera_id, pid, conf)
                    if pid is not None:
                        repo.update_last_seen(pid)
                if pid is not None and self._should_log_position(camera_id, pid):
                    world = self._project_world(camera_id, loc, frame.shape[1])
                    if world is not None:
                        world = self._smooth_world(camera_id, pid, world)
                    repo.log_position(camera_id, pid, loc, conf, frame.shape, world=world)
                elif pid is None and notify_unknown:
                    unknown_embeddings.append(embedding)

                results.append((loc, pid, name, conf))
                if recording:
                    top, right, bottom, left = loc
                    # Full candidate ranking (closest first) → enables CMC / Rank-k offline.
                    ranked = self._recognizer.rank_candidates(embedding)
                    details.append({
                        "predicted_identity": name,
                        "predicted_person_id": pid,
                        "confidence": round(conf, 4),
                        "raw_cosine_distance": round(raw_dist, 6) if raw_dist is not None else None,
                        "best_match_person_id": best_pid if best is not None else None,
                        "candidates": [[cpid, round(cdist, 6)] for cpid, _, cdist in ranked],
                        "bbox": [int(top), int(right), int(bottom), int(left)],
                    })
        return results, details, unknown_embeddings, match_ms

    @staticmethod
    def _downsample(frame: np.ndarray, det_size):
        """Resize `frame` so it fits within `det_size` (never upscale). Returns (work_frame,
        scale) where original_coord = work_coord / scale."""
        if not det_size:
            return frame, 1.0
        h, w = frame.shape[:2]
        dw, dh = det_size
        scale = min(dw / float(w), dh / float(h))
        if scale >= 1.0:  # frame already within budget → no upscaling
            return frame, 1.0
        work = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
        return work, scale

    @staticmethod
    def _remap(loc: FaceLocation, scale: float) -> FaceLocation:
        if scale == 1.0:
            return loc
        return tuple(int(round(c / scale)) for c in loc)  # type: ignore[return-value]

    def process_frame(
        self, frame: np.ndarray, camera_id: str
    ) -> List[RecognitionResult]:
        self._reload_templates_if_needed()

        instrument = _settings.metrics_enabled
        recording = validation_manager.is_active(camera_id)
        notify_unknown = get_notifier().enabled
        t_start = time.perf_counter()
        timing: Optional[Dict[str, float]] = {} if instrument else None

        prof = get_profile()
        if prof.is_optimized:
            return self._process_optimized(
                frame, camera_id, prof, instrument, recording, notify_unknown, t_start, timing)

        # ── Standard profile — unchanged behaviour ────────────────────────────────
        cyc0 = perf_counters.snap() if recording else None
        face_data = detect_and_encode(frame, timing=timing)
        if not face_data:
            if notify_unknown:  # no faces → unknown streak resets
                self._update_unknown_alert(camera_id, [])
            if recording:  # keep the footage continuous even with zero detections
                validation_manager.record(camera_id, frame, [], [])
            if instrument:
                self.perf.record(
                    camera_id, detect_ms=timing.get("detect_ms", 0.0),
                    embed_ms=timing.get("embed_ms", 0.0), match_ms=0.0,
                    total_ms=(time.perf_counter() - t_start) * 1000.0, faces=0,
                )
            return []

        results, details, unknown_embeddings, match_ms = self._recognize(
            face_data, camera_id, frame, recording, notify_unknown)

        if notify_unknown:
            self._update_unknown_alert(camera_id, unknown_embeddings)
        if recording:
            validation_manager.record(
                camera_id, frame, results, details,
                stages=self._stage_telemetry(timing, 0.0, match_ms, t_start, cyc0,
                                             frame.shape, len(face_data)))
        if instrument:
            self.perf.record(
                camera_id, detect_ms=timing.get("detect_ms", 0.0),
                embed_ms=timing.get("embed_ms", 0.0), match_ms=match_ms,
                total_ms=(time.perf_counter() - t_start) * 1000.0, faces=len(face_data),
            )

        return results

    @staticmethod
    def _stage_telemetry(timing, pre_ms, match_ms, t_start, cyc0, det_shape, n_faces):
        """Tier-A per-frame telemetry attached to every detections.jsonl record: per-stage wall
        times, CPU cycles/instructions over detect+embed+match (perf_event), detector input
        resolution and face count — keyed per model/precision via the session manifest."""
        t = timing or {}
        out = {
            "t_ms": {
                "pre": round(pre_ms, 3),
                "det": round(t.get("detect_ms", 0.0), 3),
                "emb": round(t.get("embed_ms", 0.0), 3),
                "match": round(match_ms, 3),
                "total": round((time.perf_counter() - t_start) * 1000.0, 3),
            },
            "n_faces": n_faces,
            "det_input_wh": [int(det_shape[1]), int(det_shape[0])],
        }
        if cyc0 is not None:
            cyc1 = perf_counters.snap()
            if cyc1 is not None:
                out["cycles"] = cyc1[0] - cyc0[0]
                out["instructions"] = cyc1[1] - cyc0[1]
        hw = sysfs_telemetry.sample()   # in-process sysfs snapshot, TTL-cached, synced to frame
        if hw:
            out["hw"] = hw
        return out

    def _process_optimized(self, frame, camera_id, prof, instrument, recording,
                           notify_unknown, t_start, timing):
        """Optimized-TX2 path: frame-skip → downsample → detect → tracker (skip re-embedding
        of carried faces) → batch-embed the rest → shared recognition. Gated entirely by the
        profile; Standard never reaches here."""
        # Frame-skip: reuse the last results on skipped frames (no recording on skips).
        cnt = self._frame_count.get(camera_id, -1) + 1
        self._frame_count[camera_id] = cnt
        if prof.frame_skip > 1 and (cnt % prof.frame_skip) != 0:
            cached = self._last_results.get(camera_id, [])
            if instrument:
                self.perf.record(camera_id, detect_ms=0.0, embed_ms=0.0, match_ms=0.0,
                                 total_ms=(time.perf_counter() - t_start) * 1000.0, faces=len(cached))
            return cached

        cyc0 = perf_counters.snap() if recording else None
        t_pre = time.perf_counter()
        work, scale = self._downsample(frame, prof.det_size)
        pre_ms = (time.perf_counter() - t_pre) * 1000.0
        pairs = detect_faces(work, timing=timing)  # [(loc_ds, face)]
        if not pairs:
            self._last_results[camera_id] = []
            if notify_unknown:
                self._update_unknown_alert(camera_id, [])
            if recording:
                validation_manager.record(camera_id, frame, [], [])
            if instrument:
                self.perf.record(
                    camera_id, detect_ms=(timing or {}).get("detect_ms", 0.0), embed_ms=0.0,
                    match_ms=0.0, total_ms=(time.perf_counter() - t_start) * 1000.0, faces=0)
            return []

        # Tracker carries identity: only embed faces without a cached embedding.
        if prof.tracker:
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                from core.tracker import FaceTracker
                tracker = self._trackers[camera_id] = FaceTracker()
            tracked = tracker.step([loc for loc, _ in pairs])
        else:
            tracked = [(None, None)] * len(pairs)

        embs: List[Optional[np.ndarray]] = [None] * len(pairs)
        to_embed = [i for i, (_, emb) in enumerate(tracked) if emb is None]
        if to_embed:
            new = embed_faces(work, [pairs[i] for i in to_embed],
                              batch=prof.batch_embed, timing=timing)
            for idx, (_, emb) in zip(to_embed, new):
                embs[idx] = emb
                if prof.tracker:
                    tracker.set_embedding(tracked[idx][0], emb)
        elif timing is not None:
            timing["embed_ms"] = 0.0  # all faces carried by the tracker → no embedding this frame
        for i, (_, emb) in enumerate(tracked):
            if embs[i] is None:
                embs[i] = emb  # reuse cached embedding

        # Remap detection boxes back to original-frame coordinates for display/logging.
        face_data = [(self._remap(loc, scale), embs[i]) for i, (loc, _) in enumerate(pairs)]

        results, details, unknown_embeddings, match_ms = self._recognize(
            face_data, camera_id, frame, recording, notify_unknown)
        self._last_results[camera_id] = results
        if notify_unknown:
            self._update_unknown_alert(camera_id, unknown_embeddings)
        if recording:
            validation_manager.record(
                camera_id, frame, results, details,
                stages=self._stage_telemetry(timing, pre_ms, match_ms, t_start, cyc0,
                                             work.shape, len(face_data)))
        if instrument:
            self.perf.record(
                camera_id, detect_ms=(timing or {}).get("detect_ms", 0.0),
                embed_ms=(timing or {}).get("embed_ms", 0.0), match_ms=match_ms,
                total_ms=(time.perf_counter() - t_start) * 1000.0, faces=len(face_data))
        return results
