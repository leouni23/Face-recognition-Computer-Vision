"""Performance instrumentation: pipeline timing + cross-platform resource metrics.

Two independent pieces:

- `PerfTracker`: thread-safe rolling window of per-frame, per-camera stage timings
  (detection / embedding / matching / end-to-end) used to compute smoothed FPS and
  average latency. One shared instance lives on the pipeline.

- Resource metrics provider: returns GPU/CPU/RAM/power/temperature, auto-selecting a
  backend at runtime — NVML (`pynvml`) on x86+NVIDIA, `tegrastats` on Jetson (L4T),
  `psutil`-only fallback elsewhere. Every field degrades to None if unavailable.
"""
import platform
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from loguru import logger

# ── Pipeline timing ───────────────────────────────────────────────────────────


class PerfTracker:
    """Rolling-window per-camera frame timings → smoothed FPS and average latency."""

    def __init__(self, window: int = 60):
        self._window = window
        self._lock = threading.Lock()
        self._frames: Dict[str, Deque[dict]] = {}

    def record(
        self,
        camera_id: str,
        *,
        detect_ms: float,
        embed_ms: float,
        match_ms: float,
        total_ms: float,
        faces: int,
        ts: Optional[float] = None,
    ) -> None:
        sample = {
            "ts": ts if ts is not None else time.perf_counter(),
            "detect_ms": detect_ms,
            "embed_ms": embed_ms,
            "match_ms": match_ms,
            "total_ms": total_ms,
            "faces": faces,
        }
        with self._lock:
            dq = self._frames.get(camera_id)
            if dq is None:
                dq = deque(maxlen=self._window)
                self._frames[camera_id] = dq
            dq.append(sample)

    @staticmethod
    def _camera_stats(dq: Deque[dict]) -> dict:
        samples = list(dq)
        n = len(samples)
        if n == 0:
            return {"fps": 0.0, "samples": 0}
        span = samples[-1]["ts"] - samples[0]["ts"]
        fps = (n - 1) / span if n >= 2 and span > 0 else 0.0

        def avg(key: str) -> float:
            return round(sum(s[key] for s in samples) / n, 2)

        return {
            "fps": round(fps, 2),
            "samples": n,
            "detect_ms": avg("detect_ms"),
            "embed_ms": avg("embed_ms"),
            "match_ms": avg("match_ms"),
            "total_ms": avg("total_ms"),
            "faces": round(sum(s["faces"] for s in samples) / n, 2),
        }

    def snapshot(self) -> dict:
        with self._lock:
            per_camera = {cam: self._camera_stats(dq) for cam, dq in self._frames.items()}
        aggregate_fps = round(sum(c["fps"] for c in per_camera.values()), 2)
        return {"cameras": per_camera, "aggregate_fps": aggregate_fps}


# ── Resource metrics (GPU / CPU / RAM / power / temperature) ────────────────────


class _BaseProvider:
    backend = "none"

    def read(self) -> dict:  # pragma: no cover - overridden
        return _empty_metrics()


def _empty_metrics() -> dict:
    return {
        "backend": "none",
        "gpu_name": None,
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_power_w": None,
        "gpu_temp_c": None,
        "cpu_pct": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
    }


def _read_psutil(metrics: dict) -> None:
    """Fill CPU% and RAM in-place; no-op if psutil missing."""
    try:
        import psutil
    except ImportError:
        return
    try:
        metrics["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        metrics["ram_used_mb"] = round((vm.total - vm.available) / 1e6)
        metrics["ram_total_mb"] = round(vm.total / 1e6)
    except Exception:
        pass


class _NvmlProvider(_BaseProvider):
    """x86 (or any) NVIDIA GPU via NVML (pynvml / nvidia-ml-py)."""

    backend = "nvidia-nvml"

    def __init__(self):
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(self._handle)
        self._name = name.decode() if isinstance(name, bytes) else name

    def read(self) -> dict:
        m = _empty_metrics()
        m["backend"] = self.backend
        m["gpu_name"] = self._name
        nvml, h = self._nvml, self._handle
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(h)
            m["gpu_util_pct"] = int(util.gpu)
        except Exception:
            pass
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            m["gpu_mem_used_mb"] = round(mem.used / 1e6)
            m["gpu_mem_total_mb"] = round(mem.total / 1e6)
        except Exception:
            pass
        try:
            m["gpu_power_w"] = round(nvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
        except Exception:
            pass
        try:
            m["gpu_temp_c"] = int(nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        _read_psutil(m)
        return m


class _TegrastatsProvider(_BaseProvider):
    """Jetson (L4T) GPU/CPU/RAM/power/temp by parsing one line of `tegrastats`."""

    backend = "jetson-tegrastats"
    _RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    _GR3D = re.compile(r"GR3D_FREQ (\d+)%")
    _GPU_TEMP = re.compile(r"(?:GPU|gpu)@([\d.]+)C")
    _CPU_TEMP = re.compile(r"(?:CPU|cpu)@([\d.]+)C")
    # power rails differ across modules (VDD_IN, POM_5V_IN, VDD_GPU_SOC ...)
    _GPU_POWER = re.compile(r"(?:VDD_GPU_SOC|POM_5V_GPU|VDD_GPU) (\d+)mW")
    _IN_POWER = re.compile(r"(?:VDD_IN|POM_5V_IN) (\d+)mW")

    def __init__(self):
        self._bin = shutil.which("tegrastats") or "/usr/bin/tegrastats"

    def _sample_line(self) -> str:
        proc = subprocess.Popen(
            [self._bin, "--interval", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
        )
        try:
            line = proc.stdout.readline() if proc.stdout else ""
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        return line

    def read(self) -> dict:
        m = _empty_metrics()
        m["backend"] = self.backend
        m["gpu_name"] = "Jetson integrated GPU"
        try:
            line = self._sample_line()
        except Exception:
            _read_psutil(m)
            return m
        mm = self._RAM.search(line)
        if mm:
            m["ram_used_mb"], m["ram_total_mb"] = int(mm.group(1)), int(mm.group(2))
        mm = self._GR3D.search(line)
        if mm:
            m["gpu_util_pct"] = int(mm.group(1))
        mm = self._GPU_TEMP.search(line)
        if mm:
            m["gpu_temp_c"] = round(float(mm.group(1)))
        mm = self._GPU_POWER.search(line) or self._IN_POWER.search(line)
        if mm:
            m["gpu_power_w"] = round(int(mm.group(1)) / 1000.0, 1)
        # tegrastats has no single CPU%, leave psutil to fill cpu_pct
        _read_psutil(m)
        return m


class _PsutilOnlyProvider(_BaseProvider):
    backend = "cpu-psutil"

    def read(self) -> dict:
        m = _empty_metrics()
        m["backend"] = self.backend
        _read_psutil(m)
        return m


def _build_provider() -> _BaseProvider:
    # Jetson first: an L4T board also exposes NVML poorly, tegrastats is the right tool
    if Path("/etc/nv_tegra_release").exists() or shutil.which("tegrastats"):
        try:
            p = _TegrastatsProvider()
            logger.info("Metrics: backend Jetson tegrastats")
            return p
        except Exception as exc:
            logger.warning(f"Metrics: tegrastats non disponibile ({exc})")
    try:
        p = _NvmlProvider()
        logger.info(f"Metrics: backend NVML ({p._name})")
        return p
    except Exception:
        logger.info("Metrics: backend solo CPU (psutil) — GPU non rilevata")
        return _PsutilOnlyProvider()


_provider: Optional[_BaseProvider] = None
_provider_lock = threading.Lock()


def get_resource_metrics() -> dict:
    """Current GPU/CPU/RAM/power/temperature. Auto-selects the backend on first call."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = _build_provider()
    try:
        return _provider.read()
    except Exception as exc:  # never let metrics crash the request
        logger.debug(f"Metrics read fallita: {exc}")
        return _empty_metrics()


# ── Platform label (for the validation manifest / PROTOCOL.md) ──────────────────


def _arch() -> str:
    m = platform.machine().lower()
    if m in ("amd64", "x86_64", "x64"):
        return "x86"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m or "unknown"


def _slug_gpu(name: str) -> str:
    """'NVIDIA GeForce RTX 2080' → 'rtx2080' (compact, filesystem/label-safe)."""
    s = name.lower()
    for noise in ("nvidia", "geforce", "(r)", "(tm)"):
        s = s.replace(noise, " ")
    return re.sub(r"[^a-z0-9]+", "", s) or "gpu"


_jetson_model_warned = False


def _jetson_model() -> Optional[str]:
    """Read the Jetson board name from the device tree → jetson-tx2 / jetson-orin / …
    Inside the container /proc/device-tree must be bind-mounted (docker-compose.jetson.yml);
    if it's unreadable we log the reason ONCE and fall back to the arch label."""
    global _jetson_model_warned
    try:
        txt = Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except Exception as exc:
        if not _jetson_model_warned and Path("/etc/nv_tegra_release").exists():
            _jetson_model_warned = True
            logger.info("Platform: /proc/device-tree/model illeggibile ({}); montalo read-only nel "
                        "compose per l'etichetta jetson-* — fallback su arch".format(exc))
        return None
    for key in ("tx2", "orin", "xavier", "nano", "tx1"):
        if key in txt:
            return f"jetson-{key}"
    return "jetson" if "jetson" in txt or "tegra" in txt else None


def detect_platform_label() -> str:
    """A short hardware label (e.g. 'jetson-tx2', 'jetson-orin', 'x86-rtx2080', 'x86-cpu').

    Auto-detected so each validation session records where it ran. Reuses the resource
    provider's backend/GPU name; on Jetson the board is read from the device tree.
    """
    # Device-tree FIRST: inside the container tegrastats may be unreadable (backend falls back to
    # psutil) but /proc/device-tree/model is always there → without this a TX2 run was mislabelled
    # "arm64-cpu" in every session manifest.
    jm = _jetson_model()
    if jm:
        base = jm
    else:
        metrics = get_resource_metrics()
        backend = metrics.get("backend")
        if backend == _NvmlProvider.backend and metrics.get("gpu_name"):
            base = f"{_arch()}-{_slug_gpu(metrics['gpu_name'])}"
        else:
            base = _arch()
    return base + _compute_suffix()


def _compute_suffix() -> str:
    """Compute suffix from the providers ACTUALLY bound by the live InsightFace session (same
    source as the profile semaforo) — never a static fallback: -trt (TensorRT), -gpu (CUDA),
    -cpu (analyzer loaded on CPU), '' (analyzer not built yet → unknown, no claim)."""
    try:
        from core.detector import get_loaded_info
        provs = (get_loaded_info() or {}).get("providers") or []
    except Exception:
        provs = []
    if not provs:
        return ""
    if any("Tensorrt" in p or "TensorRT" in p for p in provs):
        return "-trt"
    if any("CUDA" in p for p in provs):
        return "-gpu"
    return "-cpu"
