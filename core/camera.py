import queue
import time
import threading
from typing import Optional, Union

import cv2
import numpy as np
from loguru import logger

_RTSP_PREFIXES = ("rtsp://", "rtsps://", "rtmp://")
_RECONNECT_DELAY = 2.0   # seconds between reconnect attempts
_MAX_RECONNECTS = 30     # give up after this many consecutive failures


class CameraStream:
    """Threaded camera reader — supports local indices and RTSP/RTMP URLs.

    RTSP streams use the FFMPEG backend with buffer=1 for minimum latency and
    auto-reconnect on dropout.
    """

    def __init__(self, source: str, camera_id: Optional[str] = None):
        self._source = source
        self._is_rtsp = source.lower().startswith(_RTSP_PREFIXES)
        self.camera_id = camera_id or source

        self._cap = self._open()
        self._queue = queue.Queue(maxsize=2)  # type: queue.Queue
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"cam-{self.camera_id}"
        )

    def _open(self) -> cv2.VideoCapture:
        if self._is_rtsp:
            cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            src: Union[int, str] = int(self._source) if self._source.isdigit() else self._source
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la camera: {self._source!r}")
        return cap

    def start(self) -> "CameraStream":
        self._thread.start()
        logger.info(f"Camera {self.camera_id!r} avviata {'(RTSP)' if self._is_rtsp else ''}")
        return self

    def _capture_loop(self) -> None:
        reconnects = 0
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if not self._is_rtsp:
                    logger.warning(f"Camera {self.camera_id!r}: fine stream")
                    break

                reconnects += 1
                if reconnects > _MAX_RECONNECTS:
                    logger.error(f"Camera {self.camera_id!r}: troppi errori, abbandono")
                    break
                logger.warning(
                    f"Camera {self.camera_id!r}: connessione persa "
                    f"(tentativo {reconnects}/{_MAX_RECONNECTS}), riconnessione..."
                )
                self._cap.release()
                time.sleep(_RECONNECT_DELAY)
                try:
                    self._cap = self._open()
                except RuntimeError:
                    pass
                continue

            reconnects = 0
            # Drop the oldest frame if the consumer is too slow
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        self._cap.release()
        logger.info(f"Camera {self.camera_id!r} fermata")
