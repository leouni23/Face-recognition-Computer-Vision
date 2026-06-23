"""Destination-volume detection and validation for the validation experiment.

Lets the operator put the bulky, append-only validation artifacts (video, JSONL, CSV, JSON)
on an external HDD/SSD. Cross-platform via psutil (already a dependency) — no lsblk/df parsing.

Hard rules enforced/surfaced here:
- The app never mounts drives or installs drivers (needs root; on Jetson exFAT may need a
  kernel rebuild). It works with already-mounted paths; an unmounted/missing path → guidance.
- The live SQLite DB is NEVER placed on a removable FS (FAT/exFAT/NTFS have POSIX permission /
  file-locking issues). Only the append-only validation artifacts go to the external disk;
  this module only ever validates a *validation* target and says so.
- FAT32's 4 GB single-file limit is handled by segmenting the video (see core/validation.py);
  validation just notes it.
"""
import os
from pathlib import Path
from typing import List, Optional

# Video segmentation thresholds (used by the recorder). Rotating at ~3.5 GB keeps every
# segment under FAT32's 4 GB limit on every filesystem; the time cap keeps segments reviewable.
SEG_MAX_BYTES = int(3.5 * 1024 ** 3)
SEG_MAX_FRAMES = 15 * 60 * 10          # ~10 min at 15 fps
_DEFAULT_BITRATE_BPS = 4_000_000        # rough annotated-mp4 bitrate for the space estimate

_PSEUDO_FS = {
    "proc", "sysfs", "devtmpfs", "tmpfs", "devpts", "cgroup", "cgroup2", "overlay",
    "squashfs", "ramfs", "autofs", "mqueue", "hugetlbfs", "debugfs", "tracefs", "fusectl",
    "configfs", "securityfs", "pstore", "binfmt_misc", "nsfs", "snap",
}


def estimate_bytes(minutes: float, bitrate_bps: int = _DEFAULT_BITRATE_BPS) -> int:
    """Rough storage need for `minutes` of annotated video (JSONL/CSV are negligible)."""
    return int(max(0.0, minutes) * 60.0 * bitrate_bps / 8.0)


def classify_fs(fstype: str) -> str:
    """Map a raw filesystem name to a class: fat32 | exfat | ntfs | ext | other."""
    f = (fstype or "").lower()
    if "exfat" in f:
        return "exfat"
    if f in ("vfat", "fat", "fat32", "msdos") or "fat32" in f:
        return "fat32"
    if "ntfs" in f or f == "fuseblk":  # fuseblk is usually ntfs-3g (could be exfat-fuse)
        return "ntfs"
    if f.startswith("ext") or f in ("xfs", "btrfs", "zfs", "f2fs", "apfs", "hfs", "hfsplus"):
        return "ext"
    return "other"


def list_drives() -> List[dict]:
    """Currently mounted real partitions: mountpoint, fstype, fs class, total/free bytes."""
    import psutil

    out: List[dict] = []
    for part in psutil.disk_partitions(all=False):
        if part.fstype.lower() in _PSEUDO_FS:
            continue
        mp = part.mountpoint
        if mp.startswith(("/proc", "/sys", "/run", "/dev", "/snap")):
            continue
        try:
            u = psutil.disk_usage(mp)
            total, free = u.total, u.free
        except Exception:
            total = free = None  # e.g. an empty optical drive on Windows
        out.append({
            "device": part.device,
            "mountpoint": mp,
            "fstype": part.fstype,
            "fs_class": classify_fs(part.fstype),
            "total_bytes": total,
            "free_bytes": free,
            "removable": "removable" in (part.opts or "") or False,
        })
    return out


def _fstype_for_path(path: Path) -> Optional[str]:
    """Filesystem of the partition that contains `path` (longest matching mountpoint)."""
    import psutil

    try:
        target = str(path.resolve())
    except Exception:
        target = str(path)
    best, best_len = None, -1
    for part in psutil.disk_partitions(all=False):
        mp = part.mountpoint
        if (target == mp or target.startswith(mp.rstrip("\\/") + os.sep) or
                (os.name == "nt" and target.lower().startswith(mp.lower()))):
            if len(mp) > best_len:
                best, best_len = part.fstype, len(mp)
    return best


def _is_writable(path: Path) -> bool:
    probe_dir = path if path.is_dir() else path.parent
    try:
        probe = probe_dir / ".faceid_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def validate_target(path: str, est_bytes: int = 0) -> dict:
    """Check a chosen destination before recording: exists / writable / free space / filesystem.

    Returns a status the UI can show as OK / warning / error. Only the validation volume is
    validated here — the DB stays on internal storage regardless.
    """
    warnings: List[str] = []
    errors: List[str] = []
    p = Path(path) if path else None

    if not path:
        return {"ok": False, "exists": False, "errors": ["Nessun percorso indicato."],
                "warnings": [], "path": ""}

    exists = p.exists()
    is_dir = p.is_dir()
    if not exists:
        # Not failing silently: the app does not mount disks — guide the operator.
        errors.append("Percorso inesistente. Monta il disco (l'app non monta da sola) o crea la cartella.")
    elif not is_dir:
        errors.append("Il percorso non è una cartella.")

    writable = bool(exists and is_dir and _is_writable(p))
    if exists and is_dir and not writable:
        errors.append("Cartella non scrivibile (permessi del filesystem montato?).")

    free = total = None
    if exists:
        try:
            import psutil
            u = psutil.disk_usage(str(p))
            free, total = u.free, u.total
        except Exception:
            pass
    if est_bytes and free is not None and free < est_bytes:
        warnings.append(f"Spazio stimato insufficiente: servono ~{est_bytes // 1_000_000} MB, liberi ~{free // 1_000_000} MB.")

    fstype = _fstype_for_path(p) if exists else None
    fs_class = classify_fs(fstype) if fstype else "unknown"
    if fs_class == "fat32":
        warnings.append("FAT32: limite 4 GB per file — il video è registrato a segmenti automaticamente (gestito).")
    elif fs_class == "exfat":
        warnings.append("exFAT: ok per gli artefatti. Su Jetson richiede exfat-fuse/exfatprogs (e talvolta il driver kernel).")
    elif fs_class == "ntfs":
        warnings.append("NTFS: ok via ntfs-3g (scritture sequenziali).")
    elif fs_class == "other" and fstype:
        warnings.append(f"Filesystem '{fstype}': scritture sequenziali ok; verifica i permessi.")

    return {
        "ok": not errors,
        "path": str(p),
        "exists": exists,
        "writable": writable,
        "free_bytes": free,
        "total_bytes": total,
        "fstype": fstype,
        "fs_class": fs_class,
        "est_bytes": est_bytes,
        "warnings": warnings,
        "errors": errors,
        "note": "Solo gli artefatti di validazione vanno qui; il database biometrico resta su storage interno.",
    }
