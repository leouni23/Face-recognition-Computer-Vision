"""Thread-safe singleton shared by worker threads (push) and the FastAPI app (read)."""
import asyncio
import threading
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np


class Broadcaster:
    def __init__(self) -> None:
        self._frame_lock = threading.Lock()
        self._frames: dict[str, bytes] = {}       # camera_id → JPEG bytes
        self._camera_ids: set[str] = set()

        self._sub_lock = threading.Lock()
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --- called once by FastAPI on startup ---

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # --- called from worker threads ---

    def push_frame(self, camera_id: str, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        with self._frame_lock:
            self._frames[camera_id] = buf.tobytes()
            self._camera_ids.add(camera_id)

    def push_event(self, camera_id: str, name: str, confidence: float) -> None:
        if self._loop is None:
            return
        event = {
            "camera": camera_id,
            "name": name,
            "confidence": round(confidence, 3),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        with self._sub_lock:
            for q in self._subscribers:
                # thread-safe bridge into the asyncio event loop
                self._loop.call_soon_threadsafe(_safe_put, q, event)

    # --- called from asyncio (FastAPI) ---

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        with self._frame_lock:
            return self._frames.get(camera_id)

    @property
    def camera_ids(self) -> list[str]:
        with self._frame_lock:
            return sorted(self._camera_ids)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


def _safe_put(q: asyncio.Queue, item: object) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


broadcaster = Broadcaster()
