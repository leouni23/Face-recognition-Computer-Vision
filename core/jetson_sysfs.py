"""In-process hardware telemetry from Linux sysfs — no external deps, py3.6.

tegrastats is only a frontend over sysfs; on the TX2 (L4T r32.7) GPU load, per-rail power
(INA3221), temperatures and CPU/GPU/EMC frequencies are all plain files. This module reads them
directly, in-process, synchronized to each frame (~µs per read), so every validation embedding
carries the hardware state without any external process or orchestration. The sysfs paths are
bind-mounted read-only into the container (see docker-compose.jetson.yml).

Discovery runs ONCE (glob) on first use; the hot path only opens already-resolved files. A 20 ms
TTL cache keeps overhead negligible at 30 fps (~1 real read per frame). Everything degrades
gracefully: on x86 or with missing nodes, sample() returns only the channels found (or {}), never
raises, and the missing nodes are logged once.
"""
import glob
import os
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from loguru import logger

def _ttl_s() -> float:
    """Cache TTL in seconds. Default 100 ms (power/temp change slowly; at 30 fps this caps real
    I2C reads at ~10/s so the ~ms-scale INA rail reads stay negligible on the inference path).
    Override with SYSFS_TTL_MS."""
    try:
        return max(0.0, float(os.environ.get("SYSFS_TTL_MS", "100"))) / 1000.0
    except (TypeError, ValueError):
        return 0.1


_TTL_S = _ttl_s()


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _read_int(path: str) -> Optional[int]:
    v = _read(path)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


class SysfsTelemetry:
    """Auto-discovering sysfs sampler. Singleton `sysfs_telemetry` at module bottom."""

    def __init__(self):
        self._discovered = False
        self._gpu_load: Optional[str] = None
        self._rails: List[Tuple[str, str]] = []          # (rail_name, power_input path)
        self._rail_is_hwmon = False                       # hwmon power is µW; iio is mW
        self._temps: List[Tuple[str, str]] = []          # (zone_type, temp path)
        self._cpu_freq: List[Tuple[str, str, Optional[int]]] = []  # (label, cur path, max khz)
        self._gpu_freq: Optional[str] = None
        self._emc_freq: Optional[str] = None
        self._emc_actmon: Optional[str] = None
        self._emc_freq_max: Optional[int] = None
        self._cache: Dict = {}
        self._cache_ts = 0.0
        self._cost_hist: Deque[float] = deque(maxlen=64)  # steady-state read costs (µs)
        self._cost_first_skipped = False                  # drop the cold first read
        self._missing_logged = False
        self.available = False

    # ── discovery (once) ──────────────────────────────────────────────────────

    def _discover(self):
        self._discovered = True
        missing = []

        # GPU load: /sys/devices/gpu.0/load (0-1000 permille) + fallback devfreq device/load
        for cand in ("/sys/devices/gpu.0/load",):
            if os.path.exists(cand):
                self._gpu_load = cand
                break
        if self._gpu_load is None:
            fb = glob.glob("/sys/class/devfreq/*gp10b*/device/load")
            self._gpu_load = fb[0] if fb else None
        if self._gpu_load is None:
            missing.append("gpu_load")

        self._discover_rails() or missing.append("power_rails")

        # Temperatures: thermal_zone*/type + temp
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            ztype = _read(os.path.join(zone, "type"))
            tpath = os.path.join(zone, "temp")
            if ztype and os.path.exists(tpath):
                self._temps.append((ztype, tpath))
        if not self._temps:
            missing.append("temperature")

        # CPU freq per core + max (throttle reference)
        for cpud in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")):
            cur = os.path.join(cpud, "scaling_cur_freq")
            if os.path.exists(cur):
                label = os.path.basename(os.path.dirname(cpud))  # "cpu0"
                mx = _read_int(os.path.join(cpud, "cpuinfo_max_freq"))
                self._cpu_freq.append((label, cur, mx))
        if not self._cpu_freq:
            missing.append("cpu_freq")

        # GPU / EMC frequency via devfreq
        for pat in ("/sys/class/devfreq/*gp10b*/cur_freq", "/sys/devices/*.gp10b/devfreq/*/cur_freq"):
            g = glob.glob(pat)
            if g:
                self._gpu_freq = g[0]
                break
        emc = glob.glob("/sys/class/devfreq/*emc*/cur_freq")
        if emc:
            self._emc_freq = emc[0]
            mx = glob.glob(os.path.join(os.path.dirname(emc[0]), "max_freq"))
            self._emc_freq_max = _read_int(mx[0]) if mx else None
        else:
            # No devfreq EMC node on this L4T build → try debugfs (only if mounted). If neither is
            # present the EMC channel is simply omitted (graceful — documented in DOCUMENTAZIONE).
            for cand in ("/sys/kernel/debug/clk/emc/clk_rate", "/sys/kernel/debug/bpmp/debug/clk/emc/rate"):
                if os.path.exists(cand):
                    self._emc_freq = cand
                    break

        # EMC utilization via actmon (optional)
        for cand in ("/sys/kernel/actmon_avg_activity/mc_all",):
            if os.path.exists(cand):
                self._emc_actmon = cand
                break

        self.available = any((self._gpu_load, self._rails, self._temps, self._cpu_freq,
                              self._gpu_freq, self._emc_freq))
        if missing and not self._missing_logged:
            self._missing_logged = True
            if self.available:
                logger.info("SysfsTelemetry: canali assenti (degradato): {}".format(", ".join(missing)))
            else:
                logger.info("SysfsTelemetry: nessun nodo sysfs (probabile non-Jetson) — telemetria hw off")

    def _discover_rails(self) -> bool:
        """INA3221 power rails. Primary: ina3221x iio:device (rail_name_N + in_powerN_input).
        Fallback: ina3221 (no x) hwmon (inN_label + powerN_input). Returns True if any found."""
        for base in glob.glob("/sys/bus/i2c/drivers/ina3221x/*/iio:device*"):
            for name_path in sorted(glob.glob(os.path.join(base, "rail_name_*"))):
                n = name_path.rsplit("_", 1)[-1]
                rail = _read(name_path)
                power = os.path.join(base, "in_power{}_input".format(n))
                if rail and os.path.exists(power):
                    self._rails.append((rail, power))
        if self._rails:
            return True
        self._rail_is_hwmon = True
        for hp in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
            for label_path in sorted(glob.glob(os.path.join(hp, "in*_label"))):
                base = os.path.basename(label_path)  # "in1_label"
                n = base[2:].split("_", 1)[0]
                rail = _read(label_path)
                power = os.path.join(hp, "power{}_input".format(n))
                if rail and os.path.exists(power):
                    self._rails.append((rail, power))
        if not self._rails:
            self._rail_is_hwmon = False  # nothing found either way
        return bool(self._rails)

    # ── sampling (hot path, TTL-cached) ───────────────────────────────────────

    def sample(self) -> Dict:
        """Flat hardware snapshot. TTL-cached (default 100 ms). Never raises. {} when nothing
        available. Real (cache-miss) read costs feed a steady-state history for probe_cost_us."""
        if not self._discovered:
            self._discover()
        now = time.monotonic()
        if self._cache and (now - self._cache_ts) < _TTL_S:
            return self._cache
        t0 = time.perf_counter()
        out = self._read_all()
        cost_us = (time.perf_counter() - t0) * 1e6
        if out:
            if not self._cost_first_skipped:
                # First real read is cold (discovery page-cache miss, ~100s of ms): don't let it
                # poison the persisted cost — record the reference but exclude it from the median.
                self._cost_first_skipped = True
                logger.info("SysfsTelemetry: primo campione (cold) in {:.1f} µs, {} canali".format(
                    cost_us, self._channel_count(out)))
            else:
                self._cost_hist.append(cost_us)
        self._cache = out
        self._cache_ts = now
        return out

    def _read_all(self) -> Dict:
        out: Dict = {}
        if self._gpu_load is not None:
            v = _read_int(self._gpu_load)
            if v is not None:
                out["gpu_load_pct"] = round(v / 10.0, 1)  # permille → %

        power = {}
        for rail, path in self._rails:
            v = _read_int(path)
            if v is not None:
                power[rail] = round(v / 1000.0) if self._rail_is_hwmon else v  # hwmon µW→mW; iio mW
        if power:
            out["power_mw"] = power

        temps = {}
        for ztype, path in self._temps:
            v = _read_int(path)
            if v is not None:
                temps[ztype] = round(v / 1000.0, 1)  # m°C → °C
        if temps:
            out["temp_c"] = temps

        freq = {}
        throttling = False
        for label, path, mx in self._cpu_freq:
            v = _read_int(path)
            if v is not None:
                freq[label] = round(v / 1000.0)  # kHz → MHz
                if mx and v < 0.9 * mx:
                    throttling = True
        if self._gpu_freq is not None:
            v = _read_int(self._gpu_freq)
            if v is not None:
                freq["gpu"] = round(v / 1e6)  # Hz → MHz
        if self._emc_freq is not None:
            v = _read_int(self._emc_freq)
            if v is not None:
                freq["emc"] = round(v / 1e6)
                if self._emc_freq_max:
                    out["emc_util_pct"] = round(100.0 * v / self._emc_freq_max, 1)
        if freq:
            out["freq_mhz"] = freq
        if self._cpu_freq:
            out["throttling"] = throttling
        return out

    @staticmethod
    def _channel_count(out: Dict) -> int:
        n = 0
        for v in out.values():
            n += len(v) if isinstance(v, dict) else 1
        return n

    def probe_cost_us(self) -> Optional[float]:
        """STEADY-STATE median cost of a real (cache-miss) sample() in µs — the cold first read is
        excluded, so this reflects the true recurring overhead, not the one-off discovery hit."""
        if not self._cost_hist:
            return None
        vs = sorted(self._cost_hist)
        return round(vs[len(vs) // 2], 1)


sysfs_telemetry = SysfsTelemetry()
