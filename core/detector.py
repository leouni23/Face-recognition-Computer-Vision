"""Face detection and embedding via InsightFace (ArcFace).

GPU:  set USE_GPU=true  → uses CUDAExecutionProvider (NVIDIA)
CPU:  set USE_GPU=false → uses CPUExecutionProvider (fallback automatic)

Models are downloaded on first run (~200 MB, cached in ~/.insightface/).
"""
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _preload_cuda_libs() -> None:
    """On Linux, load the CUDA 12 / cuDNN 9 runtime shipped by the pip `nvidia-*`
    wheels. LD_LIBRARY_PATH is read once at process start, so mutating it from
    Python (as the win32 PATH trick above does) is too late — instead use
    onnxruntime's own loader, which dlopen()s the libs from site-packages at
    runtime. No-op on the Docker GPU image (libs come from the cudnn base) and
    on the CPU wheel (preload_dlls absent or finds nothing)."""
    if sys.platform == "win32":
        return
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as exc:
        logger.debug(f"preload_dlls non disponibile/non necessario: {exc}")


def _has_cuda_provider() -> bool:
    """True if this onnxruntime build can offer CUDA. The plain `onnxruntime`
    (CPU) wheel lists only CPU/Azure providers; only `onnxruntime-gpu` exposes
    `CUDAExecutionProvider`. Requesting CUDA on the CPU wheel does NOT error — it
    silently runs on CPU — so we check up front to warn instead of pretending."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _active_providers(analyzer) -> set:
    """Providers actually bound by the loaded ONNX sessions (post-`prepare`).
    This is the ground truth — `prepare()` accepts the requested provider list
    even when CUDA is unavailable and quietly drops down to CPU."""
    active: set = set()
    for model in getattr(analyzer, "models", {}).values():
        session = getattr(model, "session", None)
        if session is not None:
            try:
                active.update(session.get_providers())
            except Exception:
                pass
    return active


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
                want_gpu = settings.use_gpu
                if want_gpu:
                    _preload_cuda_libs()
                if want_gpu and not _has_cuda_provider():
                    # CPU-only `onnxruntime` wheel installed → CUDA can never bind.
                    # Don't advertise a provider we can't honour: fall back explicitly
                    # so the log reflects reality instead of a silent CPU run.
                    logger.warning(
                        "USE_GPU=true ma questo onnxruntime NON espone CUDAExecutionProvider "
                        "(è installato il wheel CPU 'onnxruntime'). Inferenza su CPU. "
                        "Per la GPU: pip install onnxruntime-gpu (+ runtime CUDA 12 / cuDNN 9), "
                        "oppure usa l'immagine Docker GPU."
                    )
                    want_gpu = False
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if want_gpu
                    else ["CPUExecutionProvider"]
                )
                logger.info(f"InsightFace: caricamento modelli (providers={providers})")
                instance = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                instance.prepare(
                    ctx_id=0 if want_gpu else -1,
                    det_size=(640, 640),
                    det_thresh=settings.det_threshold,
                )
                active = _active_providers(instance)
                if want_gpu and "CUDAExecutionProvider" not in active:
                    # CUDA was offered but didn't bind — almost always missing CUDA 12 /
                    # cuDNN 9 runtime libs (e.g. a Docker base without cuDNN). onnxruntime
                    # degrades to CPU without raising, so surface it loudly here.
                    logger.warning(
                        "USE_GPU=true ma CUDA non si è agganciato (provider attivi: "
                        f"{sorted(active)}); inferenza su CPU. Verifica le librerie CUDA 12 / "
                        "cuDNN 9 e i driver NVIDIA (--gpus all nel container)."
                    )
                logger.info(
                    f"InsightFace: modelli pronti (provider attivi={sorted(active)}, "
                    f"det_thresh={settings.det_threshold}, min_face_px={settings.min_face_px})"
                )
                _analyzer = instance
    return _analyzer


def _faces_to_data(faces) -> List[FaceData]:
    """Convert InsightFace results to (location, embedding), dropping faces shorter than
    `min_face_px` — this filters out mirror reflections and distant low-quality faces,
    the main source of spurious 'unknown' detections."""
    min_px = get_settings().min_face_px
    results: List[FaceData] = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (y2 - y1) < min_px:
            continue
        location: FaceLocation = (y1, x2, y2, x1)  # → top, right, bottom, left
        results.append((location, face.normed_embedding.astype(np.float32)))
    return results


def detect_and_encode(
    frame: np.ndarray, timing: Optional[Dict[str, float]] = None
) -> List[FaceData]:
    """Detect all faces in `frame` and return their locations + ArcFace embeddings.

    If `timing` (a mutable dict) is given, it is filled with `detect_ms` and `embed_ms`.
    To split the two stages we replicate FaceAnalysis.get() — detection first, then the
    per-face recognition loop — and fall back to the combined call if InsightFace's
    internals differ from what we expect (embed_ms then reported as 0).
    """
    analyzer = _get_analyzer()

    if timing is None:
        return _faces_to_data(analyzer.get(frame))

    try:
        from insightface.app.common import Face

        t0 = time.perf_counter()
        bboxes, kpss = analyzer.det_model.detect(frame, max_num=0, metric="default")
        t1 = time.perf_counter()
        faces = []
        for i in range(bboxes.shape[0]):
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bboxes[i, 0:4], kps=kps, det_score=bboxes[i, 4])
            for taskname, model in analyzer.models.items():
                if taskname == "detection":
                    continue
                model.get(frame, face)
            faces.append(face)
        t2 = time.perf_counter()
        timing["detect_ms"] = (t1 - t0) * 1000.0
        timing["embed_ms"] = (t2 - t1) * 1000.0
        return _faces_to_data(faces)
    except Exception as exc:  # internals changed → combined timing, never crash
        logger.debug(f"Timing split non disponibile, uso get() combinato: {exc}")
        t0 = time.perf_counter()
        faces = analyzer.get(frame)
        timing["detect_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["embed_ms"] = 0.0
        return _faces_to_data(faces)
