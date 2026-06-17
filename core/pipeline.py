import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings
from core.detector import FaceLocation, detect_and_encode
from core.geometry import project_point, project_polar
from core.metrics import PerfTracker
from core.recognizer import FaceRecognizer, Identity
from database.repository import CalibrationRepository, PersonRepository
from database.session import get_session

_settings = get_settings()

RecognitionResult = Tuple[FaceLocation, Optional[int], str, float]


class FaceIdPipeline:
    _TEMPLATE_RELOAD_INTERVAL = 60   # seconds between template refreshes
    _LOG_COOLDOWN = 10               # seconds between DB event writes per person
    _EMA_ALPHA = 0.4                 # smoothing of map positions (face size is noisy)
    _EMA_RESET_AFTER = 5.0           # seconds without sightings → restart smoothing

    def __init__(self):
        self._recognizer = FaceRecognizer(threshold=_settings.match_threshold)
        self._last_reload = 0.0
        self._reload_lock = threading.Lock()
        self._last_logged: Dict[Optional[int], float] = defaultdict(float)
        self._last_position: Dict[Tuple[str, int], float] = defaultdict(float)
        self._projectors: Dict[str, dict] = {}  # camera_id → {"mode", ...params}
        self._world_ema: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
        self.perf = PerfTracker()  # rolling-window FPS / per-stage latency (read by /api/metrics)

    def force_reload(self) -> None:
        self._last_reload = 0.0

    def _reload_templates_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_reload < self._TEMPLATE_RELOAD_INTERVAL:
            return
        with self._reload_lock:
            if now - self._last_reload < self._TEMPLATE_RELOAD_INTERVAL:
                return
            with get_session() as session:
                templates = PersonRepository(session).load_all_templates()
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

    def process_frame(
        self, frame: np.ndarray, camera_id: str
    ) -> List[RecognitionResult]:
        self._reload_templates_if_needed()

        instrument = _settings.metrics_enabled
        t_start = time.perf_counter()
        timing: Optional[Dict[str, float]] = {} if instrument else None

        face_data = detect_and_encode(frame, timing=timing)
        if not face_data:
            if instrument:
                self.perf.record(
                    camera_id, detect_ms=timing.get("detect_ms", 0.0),
                    embed_ms=timing.get("embed_ms", 0.0), match_ms=0.0,
                    total_ms=(time.perf_counter() - t_start) * 1000.0, faces=0,
                )
            return []

        match_ms = 0.0
        results: List[RecognitionResult] = []
        with get_session() as session:
            repo = PersonRepository(session)
            for loc, embedding in face_data:
                t_m = time.perf_counter()
                pid, name, conf = self._recognizer.identify(embedding)
                match_ms += (time.perf_counter() - t_m) * 1000.0
                if self._should_log(pid):
                    repo.log_event(camera_id, pid, conf)
                    if pid is not None:
                        repo.update_last_seen(pid)
                if pid is not None and self._should_log_position(camera_id, pid):
                    world = self._project_world(camera_id, loc, frame.shape[1])
                    if world is not None:
                        world = self._smooth_world(camera_id, pid, world)
                    repo.log_position(camera_id, pid, loc, conf, frame.shape, world=world)
                results.append((loc, pid, name, conf))

        if instrument:
            self.perf.record(
                camera_id, detect_ms=timing.get("detect_ms", 0.0),
                embed_ms=timing.get("embed_ms", 0.0), match_ms=match_ms,
                total_ms=(time.perf_counter() - t_start) * 1000.0, faces=len(face_data),
            )

        return results
