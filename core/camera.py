import os
import queue
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

_RTSP_PREFIXES = ("rtsp://", "rtsps://", "rtmp://")
_URL_PREFIXES = _RTSP_PREFIXES + ("http://", "https://")
_RECONNECT_DELAY = 2.0    # base seconds between reconnect attempts
_RECONNECT_MAX = 15.0     # backoff cap

# Fail fast on dead RTSP: TCP transport + short open/read timeout (microseconds).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)


def validate_source(source: str) -> Optional[str]:
    """Return an error string if `source` is not a valid camera source, else None.
    Valid: an integer index ("0") or an rtsp://|rtsps://|rtmp://|http(s):// URL."""
    s = (source or "").strip()
    if not s:
        return "Sorgente vuota"
    if s.isdigit():
        return None
    if s.lower().startswith(_URL_PREFIXES):
        return None
    return "Sorgente non valida: usa un indice USB (es. 0) o un URL rtsp://|http://"


def _is_url(source: str) -> bool:
    return source.lower().startswith(_URL_PREFIXES)


def _open_capture(source: str) -> "cv2.VideoCapture":
    if _is_url(source):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        src: Union[int, str] = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
    return cap


def _apply_resolution(cap: "cv2.VideoCapture", resolution: Optional[str]) -> None:
    if not resolution:
        return
    try:
        w, h = (int(x) for x in resolution.lower().split("x", 1))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    except (ValueError, TypeError):
        logger.warning(f"Risoluzione ignorata (formato non valido): {resolution!r}")


def probe_source(source: str, timeout: float = 6.0) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """Open `source` briefly and grab one frame. Runs the (possibly blocking) open+read in a
    thread with a join timeout so a dead RTSP URL fails fast. Returns (ok, error, frame)."""
    err = validate_source(source)
    if err:
        return False, err, None

    result: dict = {}

    def _work() -> None:
        cap = None
        try:
            cap = _open_capture(source)
            if not cap.isOpened():
                result["err"] = "Impossibile aprire la sorgente"
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                result["err"] = "Aperta ma nessun frame ricevuto"
                return
            result["frame"] = frame
        except Exception as exc:  # never propagate
            result["err"] = str(exc)
        finally:
            if cap is not None:
                cap.release()

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, f"Timeout dopo {timeout:.0f}s (sorgente irraggiungibile)", None
    if "frame" in result:
        return True, None, result["frame"]
    return False, result.get("err", "Errore sconosciuto"), None


class CameraStream:
    """Threaded camera reader for USB indices and RTSP/RTMP/HTTP URLs.

    Robust by design: the constructor NEVER opens the device (so an unreachable URL can't crash
    startup). The capture loop opens lazily and reconnects with backoff, exposing a live
    `status` ("connecting"/"connected"/"error") + `last_error` instead of raising.
    """

    def __init__(self, source: str, camera_id: Optional[str] = None,
                 resolution: Optional[str] = None):
        self._source = source
        self._resolution = resolution
        self._is_url = _is_url(source)
        self.camera_id = camera_id or source
        self._cap: Optional["cv2.VideoCapture"] = None
        self._queue = queue.Queue(maxsize=2)  # type: queue.Queue
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status = "connecting"
        self._last_error: Optional[str] = None
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"cam-{self.camera_id}"
        )

    def _set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._status_lock:
            self._status = status
            self._last_error = error

    def status(self) -> dict:
        with self._status_lock:
            return {"status": self._status, "last_error": self._last_error}

    def start(self) -> "CameraStream":
        self._thread.start()
        logger.info(f"Camera {self.camera_id!r} avviata {'(URL)' if self._is_url else ''}")
        return self

    def _capture_loop(self) -> None:
        delay = _RECONNECT_DELAY
        while not self._stop_event.is_set():
            if self._cap is None:
                self._set_status("connecting")
                try:
                    cap = _open_capture(self._source)
                    _apply_resolution(cap, self._resolution)
                except Exception as exc:
                    cap = None
                    self._set_status("error", str(exc))
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    self._set_status("error", "Impossibile aprire la sorgente")
                    if self._stop_event.wait(delay):
                        break
                    delay = min(delay * 1.5, _RECONNECT_MAX)
                    continue
                self._cap = cap
                delay = _RECONNECT_DELAY

            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._cap.release()
                self._cap = None
                self._set_status("error", "Connessione persa")
                logger.warning(f"Camera {self.camera_id!r}: connessione persa, riconnessione...")
                if self._stop_event.wait(delay):
                    break
                delay = min(delay * 1.5, _RECONNECT_MAX)
                continue

            self._set_status("connected")
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
        if self._cap is not None:
            self._cap.release()
        logger.info(f"Camera {self.camera_id!r} fermata")
