"""Face detection and embedding via InsightFace (ArcFace).

GPU:  set USE_GPU=true  → uses CUDAExecutionProvider (NVIDIA)
CPU:  set USE_GPU=false → uses CPUExecutionProvider (fallback automatic)

Models are downloaded on first run (~200 MB, cached in ~/.insightface/).
"""
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings

FaceLocation = Tuple[int, int, int, int]   # top, right, bottom, left
FaceData = Tuple[FaceLocation, np.ndarray]  # location + 512-d ArcFace embedding

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from insightface.app import FaceAnalysis

        settings = get_settings()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.use_gpu
            else ["CPUExecutionProvider"]
        )
        logger.info(f"InsightFace: caricamento modelli (providers={providers})")
        _analyzer = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        _analyzer.prepare(ctx_id=0 if settings.use_gpu else -1, det_size=(640, 640))
        logger.info("InsightFace: modelli pronti")
    return _analyzer


def detect_and_encode(frame: np.ndarray) -> List[FaceData]:
    """Detect all faces in `frame` and return their locations + ArcFace embeddings."""
    analyzer = _get_analyzer()
    faces = analyzer.get(frame)  # expects BGR (same as OpenCV)
    results: List[FaceData] = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        location: FaceLocation = (y1, x2, y2, x1)  # → top, right, bottom, left
        results.append((location, face.normed_embedding.astype(np.float32)))
    return results
