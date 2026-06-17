#!/usr/bin/env python
"""Controlled performance benchmark — writes a machine-readable report under data/benchmarks/.

Sections:
  1. matching latency vs database size (synthetic templates; no camera/GPU needed),
  2. per-stage pipeline latency + FPS on real frames from a camera (skipped if none),
  3. resource usage (GPU/CPU/RAM/power/temp) sampled during the run.

The report is labelled with platform + backend so the SAME scenarios can be compared
across hardware targets (x86+CUDA, Jetson TX2, Jetson Orin).

Usage:
  python scripts/benchmark.py                       # full run (camera = CAMERA_SOURCES[0])
  python scripts/benchmark.py --no-live             # matching benchmark only
  python scripts/benchmark.py --camera 0 --duration 15
"""
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import get_settings
from core.metrics import get_resource_metrics
from core.recognizer import benchmark_matching

_settings = get_settings()


def _benchmarks_dir() -> Path:
    d = Path(_settings.data_dir) / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _live_pipeline(camera: str, duration: float) -> dict:
    """Run the real pipeline on a camera for `duration` s and return perf + resource stats."""
    from core.camera import CameraStream
    from core.pipeline import FaceIdPipeline

    logger.info(f"Benchmark live: camera={camera!r} per {duration}s (modelli in caricamento)...")
    try:
        cam = CameraStream(camera).start()
    except Exception as exc:
        logger.warning(f"Camera non disponibile ({exc}) — salto la parte live")
        return {"available": False, "reason": str(exc)}

    pipeline = FaceIdPipeline()
    res_samples = []
    deadline = time.perf_counter() + duration
    frames = 0
    try:
        while time.perf_counter() < deadline:
            frame = cam.read(timeout=0.5)
            if frame is None:
                continue
            pipeline.process_frame(frame, camera)
            frames += 1
            if frames % 15 == 0:
                res_samples.append(get_resource_metrics())
    finally:
        cam.stop()

    snap = pipeline.perf.snapshot()
    return {
        "available": True,
        "frames_processed": frames,
        "duration_s": duration,
        "perf": snap,
        "resource_samples": res_samples,
    }


def run_benchmark(
    db_sizes=(10, 100, 500, 1000, 5000),
    do_live: bool = True,
    camera: str = None,
    duration: float = 15.0,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _benchmarks_dir() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "use_gpu": _settings.use_gpu,
            "resource_backend": get_resource_metrics().get("backend"),
            "gpu_name": get_resource_metrics().get("gpu_name"),
        },
        "matching": None,
        "live": None,
    }

    logger.info("Benchmark matching vs dimensione DB...")
    report["matching"] = benchmark_matching(sizes=tuple(db_sizes))

    if do_live:
        cam = camera if camera is not None else _settings.cameras[0]
        report["live"] = _live_pipeline(cam, duration)

    # JSON: full report
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # CSV: matching table (easy to plot/compare across targets)
    with (out_dir / "matching.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["db_size", "mean_ms", "p95_ms", "max_ms", "trials"])
        w.writeheader()
        w.writerows(report["matching"])

    logger.success(f"Report scritto in {out_dir}")
    _print_summary(report)
    return out_dir


def _print_summary(report: dict) -> None:
    print("\n=== MATCHING (latenza identify vs dimensione DB) ===")
    print(f'{"DB size":>8} | {"mean ms":>9} | {"p95 ms":>9}')
    for r in report["matching"]:
        print(f'{r["db_size"]:>8} | {r["mean_ms"]:>9} | {r["p95_ms"]:>9}')
    live = report.get("live")
    if live and live.get("available"):
        cams = live["perf"]["cameras"]
        print("\n=== PIPELINE LIVE (media su finestra) ===")
        for cam_id, s in cams.items():
            print(f'  camera {cam_id}: {s["fps"]} FPS | detect {s["detect_ms"]}ms '
                  f'| embed {s["embed_ms"]}ms | match {s["match_ms"]}ms '
                  f'| E2E {s["total_ms"]}ms | {s["faces"]} volti/frame')
    elif live:
        print(f"\n(parte live saltata: {live.get('reason')})")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Face ID — benchmark prestazioni")
    p.add_argument("--no-live", action="store_true", help="Solo benchmark matching (senza camera)")
    p.add_argument("--camera", default=None, help="Sorgente camera (default: CAMERA_SOURCES[0])")
    p.add_argument("--duration", type=float, default=15.0, help="Durata parte live in secondi")
    p.add_argument("--db-sizes", default="10,100,500,1000,5000", help="Dimensioni DB separate da virgola")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=_settings.log_level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    run_benchmark(
        db_sizes=[int(x) for x in args.db_sizes.split(",")],
        do_live=not args.no_live,
        camera=args.camera,
        duration=args.duration,
    )
