"""Cosine-distance matcher for 512-d ArcFace embeddings (InsightFace).

InsightFace returns L2-normalised embeddings, so cosine_distance = 1 - dot(a, b).
Recommended MATCH_THRESHOLD: 0.5  (stricter → lower value).
"""
from typing import List, Optional, Tuple

import numpy as np

Identity = Tuple[Optional[int], str, float]  # person_id, name, confidence

# (matrix NxD, templates) — swapped atomically so identify() never sees stati misti
_LoadedData = Tuple[np.ndarray, List[Tuple[int, str, np.ndarray]]]


class FaceRecognizer:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._data: Optional[_LoadedData] = None

    def load(self, templates: List[Tuple[int, str, np.ndarray]]) -> None:
        if templates:
            matrix = np.stack([t[2] for t in templates]).astype(np.float32)  # (N, 512)
            self._data = (matrix, list(templates))
        else:
            self._data = None

    def identify(self, embedding: np.ndarray) -> Identity:
        data = self._data
        if data is None:
            return None, "Sconosciuto", 0.0

        matrix, templates = data
        distances = 1.0 - (matrix @ embedding)  # dot = cosine sim (embeddings L2-normalised)

        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = float(min(1.0, max(0.0, 1.0 - best_dist)))

        if best_dist < self.threshold:
            pid, name, _ = templates[best_idx]
            return pid, name, confidence
        return None, "Sconosciuto", confidence
