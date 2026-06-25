"""Thread-safe frame + event broadcaster for Flask workers."""
import queue
import threading
from datetime import datetime
from typing import Dict, List, Optional, Set

import cv2
import numpy as np


class Broadcaster:
    def __init__(self) -> None:
        self._frame_lock = threading.Lock()
        self._frames: Dict[str, bytes] = {}
        self._raw_frames: Dict[str, np.ndarray] = {}
        self._camera_ids: Set[str] = set()
        self._viewers: Dict[str, int] = {}  # MJPEG viewers per camera
        self.pipeline = None  # set by main.py for forced template reload

        self._sub_lock = threading.Lock()
        self._subscribers: List[queue.Queue] = []

    # --- called from worker threads ---

    def push_frame(self, camera_id: str, frame: np.ndarray) -> None:
        # Encode only when someone is actually watching the MJPEG stream
        if not self.has_viewers(camera_id):
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        with self._frame_lock:
            self._frames[camera_id] = buf.tobytes()

    def push_raw_frame(self, camera_id: str, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._raw_frames[camera_id] = frame
            self._camera_ids.add(camera_id)

    def get_raw_frame(self, camera_id: str) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._raw_frames.get(camera_id)

    def push_event(self, camera_id: str, name: str, confidence: float) -> None:
        event = {
            "camera": camera_id,
            "name": name,
            "confidence": round(confidence, 3),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    # --- called from Flask routes ---

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        with self._frame_lock:
            return self._frames.get(camera_id)

    def add_viewer(self, camera_id: str) -> None:
        with self._frame_lock:
            self._viewers[camera_id] = self._viewers.get(camera_id, 0) + 1

    def remove_viewer(self, camera_id: str) -> None:
        with self._frame_lock:
            n = self._viewers.get(camera_id, 0) - 1
            if n > 0:
                self._viewers[camera_id] = n
            else:
                self._viewers.pop(camera_id, None)
                self._frames.pop(camera_id, None)  # free the last encoded frame

    def has_viewers(self, camera_id: str) -> bool:
        with self._frame_lock:
            return self._viewers.get(camera_id, 0) > 0

    @property
    def camera_ids(self) -> List[str]:
        with self._frame_lock:
            return sorted(self._camera_ids)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


broadcaster = Broadcaster()
