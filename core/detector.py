"""Face detection and embedding via InsightFace (ArcFace).

GPU:  set USE_GPU=true  → uses CUDAExecutionProvider (NVIDIA)
CPU:  set USE_GPU=false → uses CPUExecutionProvider (fallback automatic)

Models are downloaded on first run (~200 MB, cached in ~/.insightface/).
"""
import gc
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings, snap_det_dim
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

# Build epoch: bumped on every profile change. A build captures the epoch at its start; before
# publishing _analyzer it checks the epoch is still current, else discards its result. Prevents a
# slow warm-up of the PREVIOUS profile, finishing after a newer rebuild, from clobbering the new
# analyzer (rapid profile switches).
_build_epoch = 0
_epoch_lock = threading.Lock()


def _bump_epoch() -> int:
    global _build_epoch
    with _epoch_lock:
        _build_epoch += 1
        return _build_epoch


def _current_epoch() -> int:
    with _epoch_lock:
        return _build_epoch

# Warm-up progress (model load is ~9 min on the TX2 pure-python protobuf base). Instrumented in
# _build_analyzer; read by GET /api/warmup so the UI can show a % bar that auto-hides when ready.
_warmup_state = {"active": False, "phase": "", "pct": 0.0, "eta_s": None, "ready": False}
_warmup_lock = threading.Lock()

# What the current analyzer ACTUALLY loaded (pack/precision/providers) — set at the end of every
# _build_analyzer. Read by GET /api/settings/profile/status so the UI can verify the requested
# profile really took effect (green light) instead of trusting settings (buffalo_l incident).
_loaded_info = {"model_pack": None, "precision": None, "providers": [], "profile": None,
                "det_input": None, "loaded_at": None}
_loaded_lock = threading.Lock()


def get_loaded_info() -> dict:
    with _loaded_lock:
        return dict(_loaded_info)


def get_detection_settings() -> dict:
    """Parametri di rilevamento richiesti + quelli realmente attivi sull'analyzer caricato.
    Divergono solo tra la POST di /api/settings/detection e il primo frame successivo."""
    s = get_settings()
    out = {
        "min_face_px": int(s.min_face_px),
        "det_threshold": float(s.det_threshold),
        "det_input": list(s.det_input),
        "live": {},
    }
    det_model = getattr(_analyzer, "det_model", None) if _analyzer is not None else None
    if det_model is not None:
        size = getattr(det_model, "input_size", None)
        out["live"] = {
            "det_threshold": float(getattr(det_model, "det_thresh", 0.0)),
            "det_input": list(size) if size else None,
        }
    return out


def apply_detection_settings(min_face_px=None, det_threshold=None, det_input=None) -> dict:
    """Applica i parametri del rilevatore A CALDO, senza ricostruire l'analyzer (il warm-up
    costa ~9 min sul TX2).

    `min_face_px` è letto a ogni frame da `_faces_to_data` / `detect_faces` via get_settings(),
    quindi basta aggiornare l'oggetto Settings (che è un singleton lru_cache). `det_thresh` e
    `input_size` vivono sul modello SCRFD e si possono riassegnare: il grafo ONNX ha input
    dinamico (è `prepare()` a fissare input_size, non il modello), quindi la nuova forma è
    valida senza ricaricare nulla.

    ATTENZIONE (profilo optimized): sotto TensorRT una forma di input mai vista fa costruire un
    nuovo motore ai primi frame — lento una volta sola, poi in cache in {data_dir}/engines.

    Ritorna lo stato risultante (come get_detection_settings()).
    """
    s = get_settings()
    if min_face_px is not None:
        s.min_face_px = max(0, int(min_face_px))
    if det_threshold is not None:
        s.det_threshold = float(det_threshold)
    if det_input is not None:
        s.det_input_width = snap_det_dim(det_input[0])
        s.det_input_height = snap_det_dim(det_input[1])

    det_model = getattr(_analyzer, "det_model", None) if _analyzer is not None else None
    if det_model is not None:
        if det_threshold is not None:
            det_model.det_thresh = float(s.det_threshold)
        if det_input is not None:
            det_model.input_size = s.det_input
        with _loaded_lock:
            _loaded_info["det_input"] = list(s.det_input)
    logger.info(
        f"[Detection] min_face_px={s.min_face_px} det_thresh={s.det_threshold} "
        f"det_input={s.det_input[0]}x{s.det_input[1]} "
        f"(applicato a caldo: {det_model is not None})"
    )
    return get_detection_settings()


def get_warmup_state() -> dict:
    with _warmup_lock:
        st = dict(_warmup_state)
    st["ready"] = is_analyzer_ready()
    return st


def _warmup_set(**kw) -> None:
    with _warmup_lock:
        _warmup_state.update(kw)


def _warmup_file() -> Path:
    return Path(get_settings().data_dir) / "warmup_last.json"


def _warmup_last_durations():
    """(total_s, prepare_s) from the previous run, or (None, None)."""
    try:
        import json
        d = json.loads(_warmup_file().read_text())
        return d.get("total_s"), d.get("prepare_s")
    except Exception:
        return None, None


def _warmup_save_durations(total_s: float, prepare_s: float) -> None:
    try:
        import json
        _warmup_file().write_text(json.dumps({"total_s": round(total_s, 1),
                                              "prepare_s": round(prepare_s, 1)}))
    except Exception:
        logger.debug("Impossibile salvare la durata warm-up")


def _load_ramp(start: float, est_load, stop_ev) -> None:
    """Phase 1 (ONNX protobuf parse is atomic pure-Python, no native progress): advance pct 0→85
    over the previous run's load duration. The onnx.load hook (size-weighted) can push pct ahead;
    we use max() so the bar never moves backward. eta_s counts down off the same estimate."""
    est = est_load if est_load and est_load > 0 else 480.0   # ~8min default on TX2 (no prior run)
    while not stop_ev.is_set():
        elapsed = time.perf_counter() - start
        frac = min(1.0, elapsed / float(est))
        with _warmup_lock:
            cur = _warmup_state.get("pct", 0.0)
            _warmup_state["pct"] = round(max(cur, 85.0 * frac), 1)
            _warmup_state["eta_s"] = max(0, int(est - elapsed))
        if stop_ev.wait(0.5):
            break


def _prepare_ramp(start: float, last_prep, stop_ev) -> None:
    """Phase 2 (CUDA/TensorRT prepare is blocking, no native progress): advance pct 85→99 over the
    last run's prepare duration."""
    est = last_prep if last_prep and last_prep > 0 else 60.0
    while not stop_ev.is_set():
        frac = min(0.99, (time.perf_counter() - start) / float(est))
        _warmup_set(pct=round(min(99.0, 85.0 + 14.0 * frac), 1))
        if stop_ev.wait(0.5):
            break


def _warmup_done() -> None:
    _warmup_set(active=False, ready=True, pct=100.0, phase="", eta_s=None)


def _pack_onnx_total_bytes(pack: str) -> int:
    """Sum of *.onnx sizes for the active pack, trying the usual InsightFace roots."""
    roots = []
    home = os.environ.get("INSIGHTFACE_HOME")
    if home:
        roots.append(Path(home) / "models" / pack)
    roots.append(Path(os.path.expanduser("~/.insightface")) / "models" / pack)
    for d in roots:
        try:
            files = list(d.glob("*.onnx"))
            if files:
                return sum(f.stat().st_size for f in files)
        except Exception:
            pass
    return 0


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

    # ── warm-up progress: size-weighted ONNX parse (phase 1) + prepare ramp (phase 2) ──
    t_begin = time.perf_counter()
    prepare_s = 0.0
    total_bytes = _pack_onnx_total_bytes(prof.model_pack)
    last_total, last_prep = _warmup_last_durations()
    est_load = (last_total - (last_prep or 0.0)) if last_total else None  # load = total − prepare (prep may be ~0 on CPU)
    _warmup_set(active=True, ready=False, phase="Caricamento modelli", pct=0.0,
                eta_s=(int(last_total) if last_total else None))
    loaded = {"b": 0}
    import onnx
    _orig_load = onnx.load
    _orig_load_model = getattr(onnx, "load_model", None)

    def _wrap(orig):
        def inner(f, *a, **k):
            r = orig(f, *a, **k)
            try:
                sz = os.path.getsize(f) if isinstance(f, str) else 0
            except Exception:
                sz = 0
            loaded["b"] += sz
            if total_bytes > 0:
                want = round(85.0 * min(1.0, loaded["b"] / float(total_bytes)), 1)
                with _warmup_lock:          # max(): never drag the bar below the time ramp
                    _warmup_state["pct"] = max(_warmup_state.get("pct", 0.0), want)
            return r
        return inner

    onnx.load = _wrap(_orig_load)
    if _orig_load_model is not None:
        onnx.load_model = _wrap(_orig_load_model)
    stop_ramp = threading.Event()
    stop_load = threading.Event()
    # Phase-1 time ramp: drives pct 0→85 while the atomic ONNX parse blocks with no native progress.
    threading.Thread(target=_load_ramp, args=(time.perf_counter(), est_load, stop_load),
                     daemon=True).start()
    try:
        instance = FaceAnalysis(
            name=prof.model_pack,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        stop_load.set()                       # models loaded → end phase-1 ramp before phase 2
        _warmup_set(phase="Preparazione (CUDA/TensorRT)", pct=85.0)
        prep_start = time.perf_counter()
        threading.Thread(target=_prepare_ramp, args=(prep_start, last_prep, stop_ramp),
                         daemon=True).start()
        instance.prepare(
            ctx_id=0 if want_gpu else -1,
            det_size=settings.det_input,   # era hardcoded (640, 640) → 1080p visto a 640x360
            det_thresh=settings.det_threshold,
        )
        prepare_s = time.perf_counter() - prep_start
    finally:
        stop_load.set()
        stop_ramp.set()
        onnx.load = _orig_load
        if _orig_load_model is not None:
            onnx.load_model = _orig_load_model
    _warmup_set(pct=99.0)
    _warmup_save_durations(time.perf_counter() - t_begin, prepare_s)

    active = _active_providers(instance)
    if want_gpu and "CUDAExecutionProvider" not in active:
        logger.warning(
            "USE_GPU=true ma CUDA non si è agganciato (provider attivi: "
            f"{sorted(active)}); inferenza su CPU. Verifica le librerie CUDA 12 / "
            "cuDNN 9 e i driver NVIDIA (--gpus all nel container)."
        )
    det_input = settings.det_input
    logger.info(
        f"InsightFace: modelli pronti (provider attivi={sorted(active)}, "
        f"det_input={det_input[0]}x{det_input[1]}, det_thresh={settings.det_threshold}, "
        f"min_face_px={settings.min_face_px})"
    )
    # Su un frame landscape è la LARGHEZZA a fissare il fattore di scala del letterbox
    # (new_width = det_input[0]), quindi il confronto utile è width-vs-width: l'altezza del
    # canvas è in parte padding e darebbe falsi allarmi (es. OPT_DET 1280x720 vs input 1280x736).
    if prof.det_size and prof.det_size[0] < det_input[0]:
        # Il pre-downsample del profilo optimized butta via risoluzione PRIMA del rilevatore:
        # oltre questa soglia non fa più risparmiare nulla al rilevatore (che comunque rimpicciolisce
        # al suo canvas) e diventa solo un tetto alla portata, oltre a ridurre il crop di embedding.
        logger.warning(
            f"OPT_DET {prof.det_size[0]}x{prof.det_size[1]} è più piccolo dell'input rilevatore "
            f"{det_input[0]}x{det_input[1]}: il pre-downsample limita la portata. "
            "Alzare OPT_DET_WIDTH/OPT_DET_HEIGHT (o metterli a 0 = piena risoluzione)."
        )
    # Ground truth of what is ACTUALLY loaded (semaforo UI): the buffalo_l lab incident happened
    # because settings said one thing and nothing showed what the analyzer really ran.
    with _loaded_lock:
        _loaded_info.update({
            "model_pack": prof.model_pack, "precision": prof.precision,
            "providers": sorted(active), "profile": prof.name,
            "det_input": list(det_input),
            "loaded_at": datetime.now().isoformat(),
        })
    return instance


def _dispose_analyzer(old) -> None:
    """Release a replaced FaceAnalysis: drop its onnxruntime InferenceSessions (which own the
    native CUDA/TensorRT arenas) and force a GC pass. Without this the old profile's GPU/unified
    memory leaks across switches and starves the TX2's 8 GB → fps collapses."""
    if old is None:
        return
    try:
        models = getattr(old, "models", None)
        if isinstance(models, dict):
            for m in list(models.values()):
                if hasattr(m, "session"):
                    m.session = None
            models.clear()
        for attr in ("det_model", "models"):
            if hasattr(old, attr):
                try:
                    setattr(old, attr, None)
                except Exception:
                    pass
    except Exception:
        logger.debug("Dispose analyzer: cleanup parziale")
    del old
    gc.collect()


def _rebuild_worker() -> None:
    global _analyzer, _rebuilding, _rebuild_again
    while True:
        try:
            prof = get_profile()
            epoch = _current_epoch()
            logger.info(f"InsightFace: ricostruzione analyzer in background (profilo '{prof.name}')...")
            new = _build_analyzer()
            if epoch != _current_epoch():
                # A newer switch arrived while we built → our result is stale. Discard it (free the
                # native memory) and loop to rebuild for the latest profile; never publish.
                logger.info("InsightFace: build obsoleta (epoch superata) — scartata")
                _dispose_analyzer(new)
            else:
                with _analyzer_lock:
                    old = _analyzer
                    _analyzer = new
                _dispose_analyzer(old)  # free the old onnxruntime/TensorRT native memory (TX2 8 GB)
                _warmup_done()
                logger.info("InsightFace: nuovo profilo attivo (analyzer sostituito senza riavvio)")
                if get_settings().profile_switch_restart:
                    # Fallback: onnxruntime r32.7 may not release native arenas in-process. Exit so
                    # Docker (restart: unless-stopped) brings the process back fresh on the new
                    # profile (already persisted) — zero leak, ~seconds feed gap.
                    logger.warning("profile_switch_restart attivo: riavvio pulito del processo per liberare la GPU")
                    os._exit(0)
        except Exception:
            logger.exception("Ricostruzione profilo fallita; mantengo l'analyzer precedente")
            _warmup_set(active=False)
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
    _bump_epoch()  # invalidate any in-flight build (incl. the initial warm-up) for the old profile
    if _analyzer is None:
        # Never built yet → the lazy first build in _get_analyzer (warm-up) handles it. The epoch
        # bump makes that build re-check and discard if it's now for a superseded profile.
        return
    with _rebuild_state_lock:
        if _rebuilding:
            _rebuild_again = True
            logger.info("InsightFace: cambio profilo accodato (ricostruzione già in corso)")
            return
        _rebuilding = True
    threading.Thread(target=_rebuild_worker, daemon=True, name="profile-rebuild").start()


def is_analyzer_ready() -> bool:
    """True once InsightFace models are loaded (background warm-up / first build done). Lets the
    camera worker skip detection without blocking on the (slow, ~9 min) model load → live feed
    during warm-up. During a profile switch the OLD analyzer stays set, so this stays True."""
    return _analyzer is not None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                epoch = _current_epoch()
                built = _build_analyzer()
                if _analyzer is None and epoch == _current_epoch():
                    _analyzer = built
                    _warmup_done()
                elif _analyzer is None:
                    # A profile switch landed during this slow first build → it's for the old
                    # profile. Publish it anyway so the app has SOME analyzer (rebuild will follow),
                    # but let reset_for_profile_change's queued rebuild converge to the new profile.
                    _analyzer = built
                    _warmup_done()
                    reset_for_profile_change()
                else:
                    _dispose_analyzer(built)  # a rebuild already published → drop our stale build
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

def detect_faces(frame: np.ndarray, scale: float = 1.0,
                 timing: Optional[Dict[str, float]] = None):
    """Detect faces and return [(location, face)] WITHOUT computing embeddings. `face` is the
    InsightFace Face (bbox + kps), passed later to `embed_faces`. Filtered by `min_face_px`.

    `frame` may be a DOWNSAMPLED work frame (optimized profile) whose coordinates are `scale` ×
    the original. The `min_face_px` cut must be measured in ORIGINAL-frame pixels — exactly like
    the Standard path (`_faces_to_data`) — otherwise the downsample makes the filter ~1/scale ×
    stricter and silently drops normally-sized faces (a subject at moderate distance yields no
    detections at all → the optimized profile appears dead: no ID, no 'unknown')."""
    from insightface.app.common import Face

    analyzer = _get_analyzer()
    min_px = get_settings().min_face_px
    t0 = time.perf_counter()
    bboxes, kpss = analyzer.det_model.detect(frame, max_num=0, metric="default")
    if timing is not None:
        timing["detect_ms"] = (time.perf_counter() - t0) * 1000.0
    out = []
    scale = scale if scale > 0 else 1.0
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i, 0:4].astype(int)
        if (y2 - y1) / scale < min_px:  # original-frame height (frame may be downsampled)
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
