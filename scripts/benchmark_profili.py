#!/usr/bin/env python3
"""Benchmark prestazionale dei due profili InsightFace su Jetson TX2.

Eseguire DENTRO il container:
  python3 /app/scripts/benchmark_profili.py --profile standard  --video /data/validation/.../cam_0_000.mp4 --output /tmp/perf_standard.json
  python3 /app/scripts/benchmark_profili.py --profile optimized-tx2 --video /data/validation/.../cam_0_000.mp4 --output /tmp/perf_optimized-tx2.json

Se --video non è fornito o non esiste, cerca automaticamente in /data/validation e in
mancanza usa frame sintetici (720p, volto simulato — solo per test provider, non per paper).
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Porta /app nel path così gli import core/ funzionano dentro il container.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── preflight (Task ARM-proof): fallisce SUBITO con elenco chiaro, mai a meta' run ──

def preflight(verbose=True):
    """Verifica import + modelli. Ritorna lista problemi (vuota = ok)."""
    problems = []
    for mod in ("numpy", "cv2", "onnxruntime", "insightface"):
        try:
            __import__(mod)
        except Exception as exc:
            problems.append("modulo mancante/rotto: {} ({})".format(mod, exc))
    try:
        from core.perfcounters import perf_counters
        if verbose:
            print("perf_event: {}".format("disponibile" if perf_counters.available
                                          else "NON disponibile (--cap-add PERFMON per abilitare)"))
    except Exception as exc:
        problems.append("core.perfcounters non importabile: {}".format(exc))
    home = os.environ.get("INSIGHTFACE_HOME", "/data/models")
    if not Path(home).is_dir():
        problems.append("INSIGHTFACE_HOME non esiste: {}".format(home))
    else:
        for pack in ("buffalo_l", "buffalo_s"):
            d = Path(home) / "models" / pack
            if not (d.is_dir() and list(d.glob("*.onnx"))):
                problems.append("pack {} assente in {}/models (verra' scaricato al primo uso)".format(pack, home))
    if verbose:
        for p in problems:
            print("[PREFLIGHT] " + p)
        if not problems:
            print("[PREFLIGHT] OK — tutte le dipendenze presenti")
    return problems


# ── provider selection ──────────────────────────────────────────────────────

def _available_providers():
    import onnxruntime as ort
    return ort.get_available_providers()


def _build_providers(profile: str, engine_cache: str) -> list:
    avail = _available_providers()
    if profile == "optimized-tx2":
        if "TensorrtExecutionProvider" in avail:
            Path(engine_cache).mkdir(parents=True, exist_ok=True)
            trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": engine_cache,
                "trt_fp16_enable": True,
            }
            return [("TensorrtExecutionProvider", trt_opts),
                    "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in avail:
            print("[WARN] TensorrtExecutionProvider non disponibile — uso CUDA senza TRT")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("[WARN] Nessun provider GPU disponibile — inferenza su CPU")
            return ["CPUExecutionProvider"]
    else:
        # standard: CUDA con fallback CPU
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[WARN] CUDAExecutionProvider non disponibile — inferenza su CPU")
        return ["CPUExecutionProvider"]


def _active_providers(analyzer) -> list:
    """Provider effettivamente agganciati dalle sessioni ONNX."""
    active = set()
    for model in getattr(analyzer, "models", {}).values():
        sess = getattr(model, "session", None)
        if sess is not None:
            try:
                active.update(sess.get_providers())
            except Exception:
                pass
    return sorted(active)


# ── frame source ────────────────────────────────────────────────────────────

def _find_video(hint):
    if hint and Path(hint).is_file():
        return hint
    # cerca in /data/validation
    candidates = sorted(glob.glob("/data/validation/*/video/cam_0_000.mp4"))
    if candidates:
        return candidates[0]
    return None


def _frame_iter_video(path: str, n_total: int):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while len(frames) < n_total:
        ok, frame = cap.read()
        if not ok:
            if not frames:
                raise RuntimeError("Video vuoto")
            # riavvolge
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(frame)
    cap.release()
    return frames


def _frame_iter_synthetic(n_total: int):
    """Frame sintetici 720p BGR — nessun volto reale, ma carica comunque il modello."""
    print("[WARN] Nessun video trovato — uso frame sintetici (no volti reali, solo stress test provider)")
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(n_total)]


# ── timing split (replicate core/detector.py logic standalone) ─────────────

def _process_frame(analyzer, frame) -> dict:
    """Ritorna detect_ms, embed_ms, total_ms, n_faces."""
    from insightface.app.common import Face

    t0 = time.perf_counter()
    try:
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
        return {
            "detect_ms": (t1 - t0) * 1000.0,
            "embed_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
            "n_faces": len(faces),
        }
    except Exception as exc:
        # fallback: combined timing
        t0 = time.perf_counter()
        faces = analyzer.get(frame)
        total = (time.perf_counter() - t0) * 1000.0
        return {"detect_ms": total, "embed_ms": 0.0, "total_ms": total, "n_faces": len(faces)}


# ── ORT built-in per-operator profiling (Tier B: MAI durante sessioni di accuratezza) ────────

def _ort_op_profile(pack, providers, insightface_home, out_dir, n=50):
    """Per-operator timings via SessionOptions.enable_profiling on RAW det+rec sessions of the
    pack (synthetic inputs of the right shape — profiles the GRAPH, per-op bottleneck view).
    Returns {det: [top ops], rec: [...]}; raw ORT profile JSONs are kept next to the output."""
    import onnxruntime as ort
    pack_dir = Path(insightface_home) / "models" / pack
    targets = {}
    for f in sorted(pack_dir.glob("*.onnx")):
        name = f.name.lower()
        if name.startswith("det_"):
            targets["det"] = (str(f), (1, 3, 640, 640))
        elif name.startswith("w600k"):
            targets["rec"] = (str(f), (1, 3, 112, 112))
    result = {}
    for key, (path, shape) in targets.items():
        try:
            so = ort.SessionOptions()
            so.enable_profiling = True
            so.profile_file_prefix = str(Path(out_dir) / "ortprof_{}_{}".format(pack, key))
            sess = ort.InferenceSession(path, sess_options=so, providers=[
                p if not isinstance(p, tuple) else p[0] for p in providers])
            inp = sess.get_inputs()[0]
            x = np.random.rand(*shape).astype(np.float32)
            for _ in range(n):
                sess.run(None, {inp.name: x})
            prof_file = sess.end_profiling()
            events = json.loads(Path(prof_file).read_text())
            agg = {}
            for e in events:
                if e.get("cat") == "Node":
                    op = e.get("args", {}).get("op_name", e.get("name", "?"))
                    agg[op] = agg.get(op, 0) + e.get("dur", 0)
            top = sorted(agg.items(), key=lambda kv: -kv[1])[:10]
            result[key] = {"profile_file": prof_file,
                           "top_ops_us": [{"op": k, "total_us": v} for k, v in top]}
        except Exception as exc:
            result[key] = {"error": str(exc)}
    return result


# ── main ────────────────────────────────────────────────────────────────────

def run(args):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    profile = args.profile
    n_frames = args.n_frames
    n_warmup = args.warmup
    engine_cache = args.engine_cache or "/data/engines"

    # ── 1. Diagnosi provider ──
    avail = _available_providers()
    print(f"\n{'='*60}")
    print(f"ORT versione   : {ort.__version__}")
    print(f"Provider disponibili: {avail}")

    # ── 2. Pack e provider ──
    pack = "buffalo_l" if profile == "standard" else "buffalo_s"
    providers = _build_providers(profile, engine_cache)
    print(f"Profilo        : {profile}")
    print(f"Model pack     : {pack}")
    print(f"Provider richiesti: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
    print(f"{'='*60}\n")

    # ── 3. Carica analyzer ──
    insightface_home = os.environ.get("INSIGHTFACE_HOME", "/data/models")
    print(f"INSIGHTFACE_HOME={insightface_home}")
    t_load = time.perf_counter()
    analyzer = FaceAnalysis(
        name=pack,
        root=insightface_home,
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    print(f"Modelli caricati in {time.perf_counter() - t_load:.1f}s")

    active = _active_providers(analyzer)
    print(f"Provider ATTIVI (da sessioni ONNX): {active}")
    gpu_used = any("CUDA" in p or "Tensorrt" in p or "TensorRT" in p for p in active)
    print(f"GPU in uso: {'SI' if gpu_used else 'NO — inferenza su CPU!'}\n")

    # ── 4. Frame source ──
    video_path = _find_video(args.video)
    if video_path:
        print(f"Video: {video_path}")
        frames = _frame_iter_video(video_path, n_frames + n_warmup)
    else:
        frames = _frame_iter_synthetic(n_frames + n_warmup)

    # ── 5. Warm-up ──
    print(f"Warm-up: {n_warmup} frame...")
    for frame in frames[:n_warmup]:
        _process_frame(analyzer, frame)

    # ── 6. Misura (con cicli/istruzioni CPU per inferenza, se perf_event disponibile) ──
    from core.perfcounters import perf_counters
    print(f"Misura: {n_frames} frame... (perf_event: {'on' if perf_counters.available else 'off'})")
    rows = []
    t_wall_start = time.perf_counter()
    for frame in frames[n_warmup:n_warmup + n_frames]:
        c0 = perf_counters.snap()
        r = _process_frame(analyzer, frame)
        c1 = perf_counters.snap()
        if c0 is not None and c1 is not None:
            r["cycles"] = c1[0] - c0[0]
            r["instructions"] = c1[1] - c0[1]
        rows.append(r)
    t_wall = time.perf_counter() - t_wall_start

    # ── 6b. Overhead della telemetria Tier-A (snap perf) misurato, per il report ──
    overhead_pct = None
    if perf_counters.available and len(frames) > n_warmup:
        probe = frames[n_warmup:n_warmup + min(50, n_frames)]
        t0 = time.perf_counter()
        for f in probe:
            _process_frame(analyzer, f)
        base = time.perf_counter() - t0
        t0 = time.perf_counter()
        for f in probe:
            a = perf_counters.snap(); _process_frame(analyzer, f); perf_counters.snap()
        instr = time.perf_counter() - t0
        if base > 0:
            overhead_pct = round(max(0.0, (instr - base) / base) * 100.0, 2)

    # ── 7. Statistiche ──
    detect_arr = np.array([r["detect_ms"] for r in rows])
    embed_arr  = np.array([r["embed_ms"]  for r in rows])
    total_arr  = np.array([r["total_ms"]  for r in rows])
    fps = n_frames / t_wall if t_wall > 0 else 0.0

    def stats(arr):
        return {
            "mean_ms":   round(float(arr.mean()), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms":    round(float(np.percentile(arr, 95)), 2),
            "min_ms":    round(float(arr.min()), 2),
            "max_ms":    round(float(arr.max()), 2),
        }

    cyc = [r["cycles"] for r in rows if "cycles" in r]
    ins = [r["instructions"] for r in rows if "instructions" in r]
    result = {
        "profile": profile,
        "model_pack": pack,
        "ort_version": ort.__version__,
        "providers_requested": [p[0] if isinstance(p, tuple) else p for p in providers],
        "providers_active": active,
        "gpu_used": gpu_used,
        "video_source": video_path or "synthetic",
        "n_frames": n_frames,
        "n_warmup": n_warmup,
        "fps_sustained": round(fps, 2),
        "wall_time_s": round(t_wall, 3),
        "detect": stats(detect_arr),
        "embed":  stats(embed_arr),
        "total":  stats(total_arr),
        "cycles_per_inference": (int(np.median(cyc)) if cyc else None),
        "instructions_per_inference": (int(np.median(ins)) if ins else None),
        "perf_counters": "available" if cyc else "unavailable",
        "telemetry_overhead_pct": overhead_pct,
        "per_frame": rows,
    }
    if args.ort_profile:
        print("ORT per-op profiling (sessioni raw det+rec, input sintetici)...")
        result["ort_profile"] = _ort_op_profile(
            pack, providers, insightface_home, Path(args.output).parent)

    # ── 8. Output ──
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nRisultati scritti in: {out}")

    print(f"\n{'='*60}")
    print(f"PROFILO: {profile}  |  PACK: {pack}  |  FPS: {fps:.1f}")
    print(f"Total  : mediana={result['total']['median_ms']:.1f}ms  p95={result['total']['p95_ms']:.1f}ms")
    print(f"Detect : mediana={result['detect']['median_ms']:.1f}ms  p95={result['detect']['p95_ms']:.1f}ms")
    print(f"Embed  : mediana={result['embed']['median_ms']:.1f}ms  p95={result['embed']['p95_ms']:.1f}ms")
    print(f"GPU in uso: {'SI' if gpu_used else 'NO'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark profili InsightFace — Jetson TX2")
    p.add_argument("--profile", choices=["standard", "optimized-tx2"])
    p.add_argument("--video", default=None, help="Path video mp4 (dentro il container)")
    p.add_argument("--n-frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--engine-cache", default="/data/engines",
                   help="Directory cache engine TensorRT")
    p.add_argument("--output", help="Path JSON output")
    p.add_argument("--ort-profile", action="store_true",
                   help="Profiling per-operatore ONNX Runtime (sessioni raw det+rec)")
    p.add_argument("--preflight", action="store_true",
                   help="Verifica solo dipendenze/modelli ed esce")
    args = p.parse_args()
    if args.preflight:
        sys.exit(2 if preflight() else 0)
    if not args.profile or not args.output:
        p.error("--profile e --output sono obbligatori (oppure usa --preflight)")
    probs = preflight(verbose=False)
    hard = [x for x in probs if "modulo mancante" in x or "INSIGHTFACE_HOME" in x]
    if hard:
        print("[PREFLIGHT] Dipendenze mancanti — impossibile procedere:")
        for x in hard:
            print("  - " + x)
        sys.exit(2)
    run(args)
