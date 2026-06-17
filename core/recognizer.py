"""Cosine-distance matcher for 512-d ArcFace embeddings (InsightFace).

InsightFace returns L2-normalised embeddings, so cosine_distance = 1 - dot(a, b).
Recommended MATCH_THRESHOLD: 0.5  (stricter → lower value).
"""
import time
from typing import Dict, List, Optional, Tuple

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

    def match(self, embedding: np.ndarray) -> Optional[Tuple[int, str, float]]:
        """Nearest template regardless of threshold: (best_person_id, best_name, best_distance).

        Returns None if no templates are loaded. Used by validation logging to record the
        raw cosine distance + best match even when the subject is ultimately rejected.
        """
        data = self._data
        if data is None:
            return None
        matrix, templates = data
        distances = 1.0 - (matrix @ embedding)  # dot = cosine sim (embeddings L2-normalised)
        best_idx = int(np.argmin(distances))
        pid, name, _ = templates[best_idx]
        return pid, name, float(distances[best_idx])

    def identify(self, embedding: np.ndarray) -> Identity:
        best = self.match(embedding)
        if best is None:
            return None, "Sconosciuto", 0.0
        pid, name, best_dist = best
        confidence = float(min(1.0, max(0.0, 1.0 - best_dist)))
        if best_dist < self.threshold:
            return pid, name, confidence
        return None, "Sconosciuto", confidence


def _random_unit_embeddings(n: int, dim: int = 512) -> np.ndarray:
    """`n` L2-normalised float32 vectors, like real ArcFace embeddings."""
    v = np.random.randn(n, dim).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def benchmark_matching(
    sizes: Tuple[int, ...] = (10, 100, 500, 1000, 5000),
    dim: int = 512,
    trials: int = 200,
) -> List[Dict[str, float]]:
    """Measure identify() latency as a function of database size (synthetic templates).

    Returns one row per size: matching latency mean/p95 in milliseconds over `trials`
    queries. Pure matching cost — no detection/embedding — so the matcher optimisation
    (precomputed contiguous matrix vs per-call np.stack) is measurable in isolation.
    """
    rows: List[Dict[str, float]] = []
    for n in sizes:
        embs = _random_unit_embeddings(n, dim)
        templates = [(i, f"p{i}", embs[i]) for i in range(n)]
        rec = FaceRecognizer(threshold=0.5)
        rec.load(templates)
        queries = _random_unit_embeddings(trials, dim)

        rec.identify(queries[0])  # warm-up (BLAS init)
        times_ms = np.empty(trials, dtype=np.float64)
        for i in range(trials):
            t0 = time.perf_counter()
            rec.identify(queries[i])
            times_ms[i] = (time.perf_counter() - t0) * 1000.0

        rows.append({
            "db_size": n,
            "mean_ms": round(float(times_ms.mean()), 4),
            "p95_ms": round(float(np.percentile(times_ms, 95)), 4),
            "max_ms": round(float(times_ms.max()), 4),
            "trials": trials,
        })
    return rows
