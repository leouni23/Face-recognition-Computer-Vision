"""Face detection and embedding via InsightFace (ArcFace).

GPU:  set USE_GPU=true  → uses CUDAExecutionProvider (NVIDIA)
CPU:  set USE_GPU=false → uses CPUExecutionProvider (fallback automatic)

Models are downloaded on first run (~200 MB, cached in ~/.insightface/).
"""
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings

# On Windows, register CUDA DLL directories installed via pip (nvidia-* packages)
# so onnxruntime-gpu can find cublasLt64_12.dll, cudart64_12.dll, cudnn64_9.dll etc.
# onnxruntime-gpu's C++ loader resolves CUDA DLLs via PATH (not the Python DLL list),
# so PATH mutation is required here — os.add_dll_directory() alone is insufficient.
if sys.platform == "win32":
    _site = Path(sys.executable).parent / "Lib" / "site-packages" / "nvidia"
    _cuda_dirs = [
        str(_site / sub / "bin")
        for sub in ("cublas", "cuda_runtime", "cuda_nvrtc", "cudnn", "cufft",
                    "curand", "cusolver", "cusparse", "nvjitlink")
        if (_site / sub / "bin").is_dir()
    ]
    if _cuda_dirs:
        os.environ["PATH"] = ";".join(_cuda_dirs) + ";" + os.environ.get("PATH", "")

FaceLocation = Tuple[int, int, int, int]   # top, right, bottom, left
FaceData = Tuple[FaceLocation, np.ndarray]  # location + 512-d ArcFace embedding

_analyzer = None
_analyzer_lock = threading.Lock()


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                from insightface.app import FaceAnalysis

                settings = get_settings()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if settings.use_gpu
                    else ["CPUExecutionProvider"]
                )
                logger.info(f"InsightFace: caricamento modelli (providers={providers})")
                instance = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                instance.prepare(ctx_id=0 if settings.use_gpu else -1, det_size=(640, 640))
                logger.info("InsightFace: modelli pronti")
                _analyzer = instance
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
