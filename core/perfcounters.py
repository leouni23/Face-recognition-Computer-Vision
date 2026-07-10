"""Linux perf_event counters (CPU cycles + instructions) via ctypes — no external deps, py3.6.

Used for per-inference telemetry on the TX2: cycles/instructions per identification quantify the
CPU cost of each model independently of clock scaling. Counters are opened once per thread
(pid=0 → calling thread, cpu=-1 → any CPU, exclude_kernel) and read as deltas around the
detect+embed block. If the kernel forbids it (perf_event_paranoid>2 in-container, missing
CAP_PERFMON) everything degrades silently: `available=False` and the manifest records
"unavailable" — run the container with `--cap-add PERFMON` (or `--privileged`, or host
`sysctl kernel.perf_event_paranoid=1`) to enable.
"""
import ctypes
import os
import platform
import struct
import threading
from pathlib import Path

from loguru import logger

# perf_event_open syscall number per arch (stable ABI)
_SYSCALL_NR = {"x86_64": 298, "aarch64": 241, "armv8l": 241, "armv7l": 364}

# perf_event_attr constants
_PERF_TYPE_HARDWARE = 0
_PERF_COUNT_HW_CPU_CYCLES = 0
_PERF_COUNT_HW_INSTRUCTIONS = 1
_ATTR_SIZE = 112  # PERF_ATTR_SIZE_VER3 — accepted by every modern kernel (>=3.x)


def _open_counter(config):
    """perf_event_open(attr, pid=0, cpu=-1, group_fd=-1, flags=0) → fd or None."""
    nr = _SYSCALL_NR.get(platform.machine())
    if nr is None:
        return None
    # struct perf_event_attr {u32 type; u32 size; u64 config; u64 sample_period;
    #   u64 sample_type; u64 read_format; u64 flags_bitfield; ...} — zero the rest.
    # flags bitfield: bit0 disabled, bit5 exclude_kernel, bit6 exclude_hv → 0x60 (enabled).
    attr = struct.pack("<IIQQQQQ", _PERF_TYPE_HARDWARE, _ATTR_SIZE, config, 0, 0, 0, 0x60)
    attr = attr + b"\x00" * (_ATTR_SIZE - len(attr))
    buf = ctypes.create_string_buffer(attr, _ATTR_SIZE)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(nr, buf, 0, -1, -1, 0)
    return fd if fd >= 0 else None


def _diagnose_perf_failure() -> str:
    """Distinguish WHY perf_event_open failed so the operator applies the right fix:
    paranoid sysctl (host) vs seccomp (Docker default profile) vs missing capability."""
    try:
        paranoid = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
    except Exception:
        paranoid = None
    if paranoid is not None and paranoid > 2:
        return ("kernel.perf_event_paranoid={} sull'HOST: abbassa con "
                "'sudo sysctl kernel.perf_event_paranoid=1'".format(paranoid))
    return ("syscall probabilmente bloccata dal profilo seccomp di Docker "
            "(paranoid={} ok): avvia il container con security_opt "
            "seccomp:./docker/seccomp-perf.json (gia' in docker-compose.jetson.yml)".format(paranoid))


class PerfCounters:
    """Per-thread cycles+instructions reader. Thread-safe via thread-local fds."""

    def __init__(self):
        self._local = threading.local()
        self._warned = False
        self.available = self._probe()

    def _probe(self):
        fd = _open_counter(_PERF_COUNT_HW_CPU_CYCLES)
        if fd is None:
            return False
        os.close(fd)
        return True

    def _fds(self):
        fds = getattr(self._local, "fds", None)
        if fds is None:
            c = _open_counter(_PERF_COUNT_HW_CPU_CYCLES)
            i = _open_counter(_PERF_COUNT_HW_INSTRUCTIONS)
            if c is None or i is None:
                if c is not None:
                    os.close(c)
                if i is not None:
                    os.close(i)
                fds = ()
                if not self._warned:
                    self._warned = True
                    logger.info("perf_event non disponibile: " + _diagnose_perf_failure())
            else:
                fds = (c, i)
            self._local.fds = fds
        return fds

    def snap(self):
        """(cycles, instructions) cumulative for this thread, or None if unavailable."""
        fds = self._fds()
        if not fds:
            return None
        try:
            # perf fds are not seekable: every read() returns the CURRENT 8-byte value.
            c = struct.unpack("<Q", os.read(fds[0], 8))[0]
            i = struct.unpack("<Q", os.read(fds[1], 8))[0]
            return (c, i)
        except OSError:
            return None


perf_counters = PerfCounters()
