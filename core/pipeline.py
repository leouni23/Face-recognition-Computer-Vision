import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings
from core.detector import FaceLocation, detect_and_encode
from core.recognizer import FaceRecognizer, Identity
from database.repository import PersonRepository
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

    def _reload_templates_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_reload < self._TEMPLATE_RELOAD_INTERVAL:
            return
        with self._reload_lock:
            if now - self._last_reload < self._TEMPLATE_RELOAD_INTERVAL:
                return
            with get_session() as session:
                templates = PersonRepository(session).load_all_templates()
            self._recognizer.load(templates)
            self._last_reload = now
            logger.debug(f"Template ricaricati: {len(templates)} persona/e")

    def _should_log(self, person_id: Optional[int]) -> bool:
        now = time.monotonic()
        if now - self._last_logged[person_id] > self._LOG_COOLDOWN:
            self._last_logged[person_id] = now
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
                results.append((loc, pid, name, conf))

        return results
