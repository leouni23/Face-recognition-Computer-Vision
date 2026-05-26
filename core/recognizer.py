"""Cosine-distance matcher for 512-d ArcFace embeddings (InsightFace).

InsightFace returns L2-normalised embeddings, so cosine_distance = 1 - dot(a, b).
Recommended MATCH_THRESHOLD: 0.5  (stricter → lower value).
"""
from typing import List, Optional, Tuple

import numpy as np

Identity = Tuple[Optional[int], str, float]  # person_id, name, confidence


class FaceRecognizer:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._templates: List[Tuple[int, str, np.ndarray]] = []

    def load(self, templates: List[Tuple[int, str, np.ndarray]]) -> None:
        self._templates = templates

    def identify(self, embedding: np.ndarray) -> Identity:
        if not self._templates:
            return None, "Sconosciuto", 0.0

        # Vectorised cosine distance (embeddings are already L2-normalised)
        matrix = np.stack([t[2] for t in self._templates])  # (N, 512)
        distances = 1.0 - (matrix @ embedding)               # dot = cosine sim

        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = float(max(0.0, 1.0 - best_dist))

        if best_dist < self.threshold:
            pid, name, _ = self._templates[best_idx]
            return pid, name, confidence
        return None, "Sconosciuto", confidence
