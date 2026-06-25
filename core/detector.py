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
from core.profile import get_profile

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


def _has_trt_provider() -> bool:
    """True only if this onnxruntime build exposes the TensorRT EP (the JP4.6 Jetson wheel does;
    the x86 pip wheel does not). Avoids requesting a phantom provider — passing an unavailable EP
    in the explicit list makes onnxruntime drop to CPU-only instead of binding CUDA."""
    try:
        import onnxruntime as ort
        return "TensorrtExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


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


def _build_providers(want_gpu: bool, prof) -> list:
    """ONNX Runtime providers for the active profile. Standard → CUDA (or CPU). Optimized →
    prepend the TensorRT EP (FP16/INT8 on the TX2's Maxwell GPU) with the engine cache on the
    external disk, falling back to CUDA then CPU. If the TRT EP is unavailable (e.g. the x86
    pip wheel), onnxruntime simply ignores it and CUDA binds — surfaced by `_active_providers`."""
    if not want_gpu:
        return ["CPUExecutionProvider"]
    # Optimized + FP16/INT8 → TensorRT EP first, but only if this build actually has it
    # (the JP4.6 Jetson wheel does; x86 doesn't → fall back to CUDA cleanly, no phantom provider).
    if prof.is_optimized and prof.precision in ("fp16", "int8") and _has_trt_provider():
        try:
            Path(prof.engine_cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": prof.engine_cache_dir,
            "trt_fp16_enable": True,  # FP16 for both fp16 and int8 (INT8 still benefits)
        }
        if prof.precision == "int8":
            trt_opts["trt_int8_enable"] = True
        return [("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


# Profile-switch rebuild state. A switch must NOT block the app: the heavy rebuild (new model
# pack + TensorRT engine build, minutes) runs in a background thread while the old analyzer keeps
# serving frames; the new one is swapped in atomically when ready.
_rebuild_state_lock = threading.Lock()
_rebuilding = False
_rebuild_again = False


def _build_analyzer():
    """Construct + prepare a FaceAnalysis for the *currently active* profile. Heavy (model load +
    TensorRT engine build). Callers must NOT hold `_analyzer_lock` while this runs."""
    from insightface.app import FaceAnalysis

    settings = get_settings()
    want_gpu = settings.use_gpu
    if want_gpu:
        _preload_cuda_libs()
    if want_gpu and not _has_cuda_provider():
        # CPU-only `onnxruntime` wheel installed → CUDA can never bind.
        logger.warning(
            "USE_GPU=true ma questo onnxruntime NON espone CUDAExecutionProvider "
            "(è installato il wheel CPU 'onnxruntime'). Inferenza su CPU. "
            "Per la GPU: pip install onnxruntime-gpu (+ runtime CUDA 12 / cuDNN 9), "
            "oppure usa l'immagine Docker GPU."
        )
        want_gpu = False
    prof = get_profile()
    providers = _build_providers(want_gpu, prof)
    logger.info(
        f"InsightFace: profilo={prof.name} pack={prof.model_pack} "
        f"precisione={prof.precision} (providers={[p[0] if isinstance(p, tuple) else p for p in providers]})"
    )
    instance = FaceAnalysis(
        name=prof.model_pack,
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
        logger.warning(
            "USE_GPU=true ma CUDA non si è agganciato (provider attivi: "
            f"{sorted(active)}); inferenza su CPU. Verifica le librerie CUDA 12 / "
            "cuDNN 9 e i driver NVIDIA (--gpus all nel container)."
        )
    logger.info(
        f"InsightFace: modelli pronti (provider attivi={sorted(active)}, "
        f"det_thresh={settings.det_threshold}, min_face_px={settings.min_face_px})"
    )
    return instance


def _rebuild_worker() -> None:
    global _analyzer, _rebuilding, _rebuild_again
    while True:
        try:
            prof = get_profile()
            logger.info(f"InsightFace: ricostruzione analyzer in background (profilo '{prof.name}')...")
            new = _build_analyzer()
            with _analyzer_lock:
                _analyzer = new
            logger.info("InsightFace: nuovo profilo attivo (analyzer sostituito senza riavvio)")
        except Exception:
            logger.exception("Ricostruzione profilo fallita; mantengo l'analyzer precedente")
        with _rebuild_state_lock:
            if _rebuild_again:
                _rebuild_again = False
                continue  # a newer switch arrived mid-build → rebuild for the latest profile
            _rebuilding = False
            return


def reset_for_profile_change() -> None:
    """Apply a runtime profile switch WITHOUT freezing the app. The analyzer rebuild (new model
    pack / providers / TensorRT engines — can take minutes on the TX2) runs in a background thread
    and is swapped in atomically; the previous analyzer keeps serving frames meanwhile, so the
    camera feed and web UI stay responsive. Rapid switches are coalesced to the latest profile."""
    global _rebuilding, _rebuild_again
    if _analyzer is None:
        # Never built yet → the lazy first build in _get_analyzer (warm-up) handles it.
        return
    with _rebuild_state_lock:
        if _rebuilding:
            _rebuild_again = True
            logger.info("InsightFace: cambio profilo accodato (ricostruzione già in corso)")
            return
        _rebuilding = True
    threading.Thread(target=_rebuild_worker, daemon=True, name="profile-rebuild").start()


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = _build_analyzer()
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


# ── Optimized profile: detection / embedding split (for the tracker + batch embedding) ──────
# Standard keeps detect_and_encode (above) byte-for-byte. The optimized pipeline detects boxes
# every frame but only embeds faces that the tracker can't carry over — and can batch them.

def detect_faces(frame: np.ndarray, timing: Optional[Dict[str, float]] = None):
    """Detect faces and return [(location, face)] WITHOUT computing embeddings. `face` is the
    InsightFace Face (bbox + kps), passed later to `embed_faces`. Filtered by `min_face_px`."""
    from insightface.app.common import Face

    analyzer = _get_analyzer()
    min_px = get_settings().min_face_px
    t0 = time.perf_counter()
    bboxes, kpss = analyzer.det_model.detect(frame, max_num=0, metric="default")
    if timing is not None:
        timing["detect_ms"] = (time.perf_counter() - t0) * 1000.0
    out = []
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i, 0:4].astype(int)
        if (y2 - y1) < min_px:
            continue
        kps = kpss[i] if kpss is not None else None
        face = Face(bbox=bboxes[i, 0:4], kps=kps, det_score=bboxes[i, 4])
        location: FaceLocation = (int(y1), int(x2), int(y2), int(x1))  # top, right, bottom, left
        out.append((location, face))
    return out


def embed_faces(frame: np.ndarray, pairs, batch: bool = False,
                timing: Optional[Dict[str, float]] = None) -> List[FaceData]:
    """Compute embeddings for the given [(location, face)] pairs. With `batch=True`, align all
    crops and run the recogniser in a single forward pass (one expensive call per frame instead
    of one per face); otherwise the standard per-face path. Returns [(location, embedding)]."""
    analyzer = _get_analyzer()
    t0 = time.perf_counter()
    result: List[FaceData] = []
    rec = analyzer.models.get("recognition") if hasattr(analyzer, "models") else None
    if batch and rec is not None and pairs:
        try:
            from insightface.utils import face_align
            crops = [face_align.norm_crop(frame, landmark=face.kps) for _, face in pairs]
            feats = rec.get_feat(crops)  # (N, 512)
            for (loc, _), feat in zip(pairs, feats):
                emb = feat / (np.linalg.norm(feat) + 1e-9)
                result.append((loc, emb.astype(np.float32)))
            if timing is not None:
                timing["embed_ms"] = (time.perf_counter() - t0) * 1000.0
            return result
        except Exception as exc:  # any internals mismatch → fall back to per-face
            logger.debug(f"Batch embedding non disponibile, uso per-volto: {exc}")
            result = []
    for loc, face in pairs:
        for taskname, model in analyzer.models.items():
            if taskname == "detection":
                continue
            model.get(frame, face)
        result.append((loc, face.normed_embedding.astype(np.float32)))
    if timing is not None:
        timing["embed_ms"] = (time.perf_counter() - t0) * 1000.0
    return result


def warmup() -> None:
    """Eagerly load InsightFace models (+ one dummy inference) at startup so the first
    real camera frame is not blocked by the slow pure-python protobuf model load."""
    t0 = time.perf_counter()
    logger.info("InsightFace: pre-caricamento modelli all'avvio (puo' richiedere alcuni minuti)...")
    try:
        detect_and_encode(np.zeros((640, 640, 3), dtype=np.uint8))
    except Exception:
        logger.exception("Pre-caricamento modelli fallito; caricamento posticipato al primo frame")
        return
    logger.info(f"InsightFace: pre-caricamento completato in {time.perf_counter() - t0:.1f}s")
