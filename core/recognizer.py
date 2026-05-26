from typing import List, Optional, Tuple

import numpy as np

Identity = Tuple[Optional[int], str, float]  # person_id, name, confidence


class FaceRecognizer:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self._templates: List[Tuple[int, str, np.ndarray]] = []

    def load(self, templates: List[Tuple[int, str, np.ndarray]]) -> None:
        self._templates = templates

    def identify(self, encoding: np.ndarray) -> Identity:
        if not self._templates:
            return None, "Sconosciuto", 0.0

        # Vectorised cosine distance against all stored templates
        matrix = np.stack([t[2] for t in self._templates])  # (N, 128)
        norms = np.linalg.norm(matrix, axis=1)
        probe_norm = np.linalg.norm(encoding)
        distances = 1.0 - (matrix @ encoding) / (norms * probe_norm + 1e-8)

        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = float(max(0.0, 1.0 - best_dist))

        if best_dist < self.threshold:
            pid, name, _ = self._templates[best_idx]
            return pid, name, confidence
        return None, "Sconosciuto", confidence
