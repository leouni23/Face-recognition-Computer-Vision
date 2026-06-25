import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from core.pipeline import RecognitionResult

_KNOWN_COLOR = (34, 197, 94)     # green
_UNKNOWN_COLOR = (59, 130, 246)  # blue
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_frame(frame: np.ndarray, results: List[RecognitionResult]) -> np.ndarray:
    """Return a copy of `frame` with bounding boxes and labels drawn."""
    out = frame.copy()
    for (top, right, bottom, left), pid, name, conf in results:
        color = _KNOWN_COLOR if pid is not None else _UNKNOWN_COLOR
        cv2.rectangle(out, (left, top), (right, bottom), color, 2)
        label = f"{name} {conf:.0%}" if pid is not None else name
        cv2.rectangle(out, (left, bottom - 26), (right, bottom), color, cv2.FILLED)
        cv2.putText(out, label, (left + 4, bottom - 6), _FONT, 0.55, (255, 255, 255), 1)
    return out


class Display:
    """Manages named OpenCV windows. Must be called from the main thread on macOS."""

    def __init__(self) -> None:
        self._fps = _FPSCounter()

    def show(
        self,
        frame: np.ndarray,
        results: List[RecognitionResult],
        window: str = "Face ID",
    ) -> bool:
        """Draw annotation + FPS overlay and render. Returns False when user presses Q."""
        out = annotate_frame(frame, results)
        fps = self._fps.tick()
        cv2.putText(out, f"FPS {fps:.1f}", (8, 26), _FONT, 0.7, (220, 220, 220), 2)
        cv2.imshow(window, out)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    @staticmethod
    def close_all() -> None:
        cv2.destroyAllWindows()


class _FPSCounter:
    def __init__(self, window: int = 30):
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
