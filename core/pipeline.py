import json
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings
from core.detector import FaceLocation, detect_and_encode
from core.geometry import project_point
from core.recognizer import FaceRecognizer, Identity
from database.repository import CalibrationRepository, PersonRepository
from database.session import get_session

_settings = get_settings()

RecognitionResult = Tuple[FaceLocation, Optional[int], str, float]


class FaceIdPipeline:
    _TEMPLATE_RELOAD_INTERVAL = 60   # seconds between template refreshes
    _LOG_COOLDOWN = 10               # seconds between DB event writes per person

    def __init__(self):
        self._recognizer = FaceRecognizer(threshold=_settings.match_threshold)
        self._last_reload = 0.0
        self._reload_lock = threading.Lock()
        self._last_logged: Dict[Optional[int], float] = defaultdict(float)
        self._last_position: Dict[Tuple[str, int], float] = defaultdict(float)
        self._homographies: Dict[str, np.ndarray] = {}

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
                calibs = CalibrationRepository(session).get_all()
                homographies = {
                    c.camera_id: np.array(json.loads(c.homography_json), dtype=np.float64)
                    for c in calibs
                }
            self._recognizer.load(templates)
            self._homographies = homographies
            self._last_reload = now
            logger.debug(
                f"Template ricaricati: {len(templates)} persona/e, "
                f"{len(homographies)} camera/e calibrate"
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

    def process_frame(
        self, frame: np.ndarray, camera_id: str
    ) -> List[RecognitionResult]:
        self._reload_templates_if_needed()

        face_data = detect_and_encode(frame)
        if not face_data:
            return []

        results: List[RecognitionResult] = []
        with get_session() as session:
            repo = PersonRepository(session)
            for loc, embedding in face_data:
                pid, name, conf = self._recognizer.identify(embedding)
                if self._should_log(pid):
                    repo.log_event(camera_id, pid, conf)
                    if pid is not None:
                        repo.update_last_seen(pid)
                if pid is not None and self._should_log_position(camera_id, pid):
                    world = None
                    H = self._homographies.get(camera_id)
                    if H is not None:
                        top, right, bottom, left = loc
                        # project bbox bottom-centre (closest to the floor plane)
                        world = project_point(H, (left + right) / 2.0, float(bottom))
                    repo.log_position(camera_id, pid, loc, conf, frame.shape, world=world)
                results.append((loc, pid, name, conf))

        return results
