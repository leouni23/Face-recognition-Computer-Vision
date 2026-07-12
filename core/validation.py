"""Validation / test mode — record annotated footage + per-detection log for offline analysis.

The production system deliberately stores no images. Validation mode is the explicit
exception used to measure identification accuracy in a lab experiment: for the duration
of a named session it records, per camera, the *annotated* video (boxes + names +
confidence) and a structured JSONL with one record per detected face per frame —
including the raw cosine distance and the nearest template, so a threshold sweep
(FAR/FRR/DET/EER) can be computed offline without re-running the cameras.

All artifacts live under {data_dir}/validation/<session_id>/, OUTSIDE the biometric DB.
A single `ValidationManager` singleton is driven from the web UI / CLI; the pipeline
calls `record()` once per processed frame while a session is active.
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from loguru import logger

from config.settings import get_settings
from core.metrics import detect_platform_label
from core.profile import profile_slug, profile_summary
from core.storage import SEG_MAX_BYTES, SEG_MAX_FRAMES
from core.validation_presets import load_presets, load_subjects, subject_truth

_settings = get_settings()
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
_NOMINAL_FPS = 15.0  # playback fps; review syncs by frame_index, not wall-clock

# Active destination root for validation artifacts (may be an external disk chosen by the
# operator). The biometric DB always stays on internal storage, separate from this.
_dest_root: Optional[Path] = None


def _default_root() -> Path:
    base = _settings.validation_dir.strip() if _settings.validation_dir else ""
    return Path(base) if base else Path(_settings.data_dir) / "validation"


def validation_root() -> Path:
    return _dest_root if _dest_root is not None else _default_root()


def set_destination(path: Optional[str]) -> Path:
    """Point the validation volume at `path` (…/validation under it), or reset to default
    when falsy. Only the bulky append-only artifacts go here — never the DB."""
    global _dest_root
    _dest_root = (Path(path) / "validation") if path else None
    return validation_root()


# ── Daily validation folder + auto-generated session names ────────────────────────
# Sessions nest automatically under validation/<YYYYMMDD>/ (created on first start of the day —
# no manual "open/close validation" step). The session name is NEVER typed by the operator: it is
# derived from the registry selection (subjects) + preset + profile, with a globally-unique provaN
# counter, so name↔ground-truth can no longer diverge (the free-text-name lab incident).


def _slugify(name: str, maxlen: int = 40) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in (name or "").strip())
    safe = "-".join(p for p in safe.split("-") if p)  # collapse repeats
    return safe[:maxlen] or "x"


def day_folder(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d")


def make_trial_key(subject_labels: List[str], preset_id: Optional[str], profile: str) -> str:
    """Stable identity of a trial combination: same subjects + preset + profile ⇒ same key."""
    return "|".join(sorted(subject_labels)) + "|" + (preset_id or "none") + "|" + profile


def count_trials(trial_key: str) -> int:
    """How many existing sessions (whole validation/ tree, any layout) share `trial_key`.
    Global counter per decision — provaN never repeats across days."""
    root = validation_root()
    n = 0
    if not root.is_dir():
        return 0
    for d in root.iterdir():
        candidates = [d] if (d / "session.json").is_file() else \
                     ([s for s in d.iterdir() if (s / "session.json").is_file()] if d.is_dir() else [])
        for s in candidates:
            try:
                meta = json.loads((s / "session.json").read_text(encoding="utf-8"))
                if meta.get("trial_key") == trial_key:
                    n += 1
            except Exception:
                pass
    return n


def _proc_io() -> dict:
    """read_bytes/write_bytes from /proc/self/io (0s if unavailable, e.g. non-Linux)."""
    out = {"read_bytes": 0, "write_bytes": 0}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in out:
                out[k] = int(v.strip())
    except Exception:
        pass
    return out


def _flatten_numeric(d: dict, prefix: str = "") -> Dict[str, float]:
    """Flatten a hw snapshot to dotted numeric keys (power_mw.VDD_IN, temp_c.CPU-therm, …),
    skipping bools (throttling handled separately) so only real measurements aggregate."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = prefix + k
        if isinstance(v, bool):
            continue
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, key + "."))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _device_for_path(path: str) -> Optional[str]:
    """Block device backing a path (longest mountpoint prefix in /proc/mounts) — records which
    physical disk (SD/USB/eMMC) the session was written to. Best-effort, Linux only."""
    try:
        best_mp, best_dev = "", None
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 2 or not parts[0].startswith("/dev/"):
                continue
            mp = parts[1].replace("\\040", " ")
            if (path == mp or path.startswith(mp.rstrip("/") + "/")) and len(mp) > len(best_mp):
                best_mp, best_dev = mp, parts[0]
        return best_dev
    except Exception:
        return None


def _vmrss_kb() -> Optional[int]:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass
    return None


def _configured_cameras() -> List[str]:
    """Enabled cameras from the runtime registry (fallback: CAMERA_SOURCES). Lazy import keeps
    core.validation free of a hard dependency on the camera manager."""
    try:
        from core.camera_manager import camera_manager
        ids = camera_manager.active_ids()
        if ids:
            return ids
    except Exception:
        pass
    return list(_settings.cameras)


def _perf_line(perf: Optional[dict]) -> str:
    """One-line human summary of the optimization parameters for PROTOCOL.md §2."""
    perf = perf or {}
    if perf.get("profile") != "optimized-tx2":
        return f"pack {perf.get('model_pack', 'buffalo_l')}, FP32, comportamento standard"
    det = perf.get("det_size")
    return (f"pack {perf.get('model_pack')}, {perf.get('precision')}, "
            f"det {('×'.join(map(str, det)) if det else 'full')}, frame-skip {perf.get('frame_skip')}, "
            f"tracker {'on' if perf.get('tracker') else 'off'}, batch {'on' if perf.get('batch_embed') else 'off'}")


def build_protocol_md(meta: dict) -> str:
    """Human-readable, self-documenting protocol for a session (written at start, §2).

    States the test type (open-set 1:N), the exact data schema, the metrics and how they
    are computed, the gallery / platform / cameras / threshold / timestamps, and the
    labeling instructions + verdict taxonomy — so each run is reproducible and explains itself.
    """
    enrolled = meta.get("enrolled", []) or []
    gallery = "\n".join(f"| `{p.get('id')}` | {p.get('name')} |" for p in enrolled) or "| — | (nessun iscritto) |"
    cameras = meta.get("cameras") or []
    cam_line = ", ".join(f"`{c}`" for c in cameras) if cameras else "_(rilevate all'avvio della registrazione)_"
    repo_protocol = Path(__file__).resolve().parent.parent / "protocollo_test_laboratorio.md"
    ref = (
        f"\n> Questo run segue il protocollo di laboratorio del repository "
        f"(`{repo_protocol.name}`); quanto sotto ne è la versione operativa per questa sessione.\n"
        if repo_protocol.exists() else ""
    )
    record_video = meta.get("record_video", True)
    gt_labels = {
        "crossing-declared": "soggetto dichiarato per ogni attraversamento (automatica)",
        "seating-map": "mappa dei posti dichiarata per la sessione comune (automatica)",
        "manual-live": "assegnazione live click-to-assign (manuale)",
    }
    gt_source = gt_labels.get(meta.get("ground_truth_source"), "dichiarata / manuale")
    # §3 + §6 footage rows only when video is actually recorded.
    video_schema = (
        "\n`video/cam_<id>.mp4` — video **annotato** (box + nome + confidenza), sincronizzato per\n"
        "`frame_index`. È materiale sensibile (reintroduce immagini): resta solo nel volume, separato dal\n"
        "DB biometrico, ed è cancellabile con `scripts/clear_validation.py`.\n"
        if record_video else
        "\n**Nessun video** è registrato in questa sessione: nessuna immagine viene scritta su disco\n"
        "(ripristino della postura «no images on disk»). La verità a terra è stabilita **dal vivo**\n"
        "(presentazione controllata: soggetto dichiarato o mappa dei posti) — vedi\n"
        "`validazione_senza_video_addendum.md`. Compromesso: **nessun audit trail a posteriori**\n"
        "senza il video.\n"
    )
    video_layout = "\n  metrics.json  det_curve.csv  cmc.csv            video/cam_<id>.mp4" if record_video \
        else "\n  metrics.json  det_curve.csv  cmc.csv            runs.jsonl   _(nessun video)_"
    return f"""# Protocollo di test — sessione `{meta.get('session_id')}`

_Generato automaticamente all'avvio della sessione di validazione._
{ref}
## 1. Tipo di test
**Identificazione open-set 1:N** (scenario watchlist / controllo accessi): una *galleria* di
soggetti iscritti e dei *probe* che includono anche persone **non iscritte** (non-mate) che il
sistema deve **rifiutare**. Valutazione di scenario secondo **ISO/IEC 19795**; metriche primarie
**FPIR/FNIR** come in **NIST FRVT/FRTE 1:N**, con curva DET ed EER.
Non è una verifica 1:1: FAR/FRR sono forniti solo come alias colloquiali di FPIR/FNIR.

## 2. Configurazione della sessione
- **Tipo sessione:** {'sessione comune (soggetti statici)' if meta.get('session_type') == 'common' else 'attraversamento singolo-soggetto (ground-truth automatica dal soggetto dichiarato)'}
- **Registrazione video:** {'sì (video annotato per camera)' if record_video else '**no** — nessuna immagine su disco'}
- **Verità a terra:** {gt_source}
- **Profilo prestazioni:** `{(meta.get('performance') or {}).get('profile', 'standard')}` — {_perf_line(meta.get('performance'))}
- **Piattaforma:** `{meta.get('platform', 'n/d')}`
- **Destinazione artefatti:** `{(meta.get('destination') or {}).get('root', 'n/d')}`
- **Telecamere:** {cam_line}
- **Soglia operativa** (distanza coseno, accetta se `<` soglia): **{meta.get('threshold')}**
- **Avvio:** {meta.get('start')}
- **Galleria iscritti (mate):**

| person_id | nome |
| --- | --- |
{gallery}

## 3. Schema dati registrato
`detections.jsonl` — un record per **ogni volto in ogni frame**:
`session_id, timestamp_ms, camera_id, frame_index, face_id, predicted_person_id,
predicted_identity, confidence, raw_cosine_distance, best_match_person_id, candidates, bbox`.
- `raw_cosine_distance` + `best_match_person_id` + `candidates` (ranking completo) permettono di
  ricalcolare **tutte** le metriche a qualsiasi soglia **offline**, senza riavviare le camere.

`labels.jsonl` — verità a terra per detection/evento: `true_person_id` (iscritto) **oppure**
`non_mate: true`. Da questa, soglia-indipendente, si derivano verdetti e curve.
{video_schema}

## 4. Etichettatura (verità a terra)
L'operatore assegna a ogni **evento** (comparsa continua di un soggetto) la sua **identità vera**:
un iscritto della galleria, oppure *Non-mate* (sconosciuto). Il verdetto a soglia operativa è
derivato automaticamente. Tassonomia:

| Verdetto | Significato | Contributo |
| --- | --- | --- |
| **Mate corretto** | iscritto identificato come sé stesso | TP (identificazione corretta) |
| **Mate mancato** | iscritto riportato come Sconosciuto | → **FNIR** |
| **Scambio** | iscritto A riportato come iscritto B | → **FNIR** (conteggiato a parte) |
| **Falso positivo** | non-mate riportato come un iscritto | → **FPIR** (critico per la sicurezza) |
| **Non-mate corretto** | sconosciuto riportato come Sconosciuto | TN |

## 5. Metriche calcolate (azione "Calcola metriche")
Unità di analisi primaria = **evento/track** (i frame consecutivi dello stesso soggetto si
aggregano per voto di maggioranza); il per-frame è riportato come secondario (i frame sono
correlati → N effettivo minore).
- **FPIR** = eventi non-mate con un candidato sopra soglia / totale eventi non-mate.
- **FNIR** = eventi mate non identificati correttamente sopra soglia / totale eventi mate.
- **Curva DET**: spazza la soglia su tutto il range di `raw_cosine_distance`, ricalcola (FPIR,FNIR)
  ad ogni soglia → `det_curve.csv`.
- **EER**: soglia dove FPIR≈FNIR (valore + soglia operativa).
- **FNIR a FPIR fisso** (0.01 e 0.003): punti operativi orientati alla sicurezza.
- **Rank-1 e CMC** sui probe mate (complemento closed-set) → `cmc.csv`.
- **Intervalli di confidenza** (Wilson) su FPIR/FNIR; **rule-of-3** (limite ~3/N a zero errori) e
  **flag rule-of-30** (Doddington) quando si raccolgono <30 errori al punto operativo.
- **FAR/FRR** = alias etichettati di FPIR/FNIR.
Tutto ricomputabile da `detections.jsonl` + `labels.jsonl`: ri-etichettare e ricalcolare basta.

## 6. Layout dei file
```
{meta.get('session_id')}/
  PROTOCOL.md   session.json   detections.jsonl   labels.jsonl{video_layout}
```
"""


class _CameraSink:
    """Writes the annotated footage for one camera as rotating segments.

    The video is split into `cam_<id>_NNN.mp4` segments (rotated at SEG_MAX_FRAMES or
    SEG_MAX_BYTES) so FAT32's 4 GB single-file limit is never hit and review stays manageable.
    `frame_index` is GLOBAL and continuous across segments; a segment index records, per
    segment, the global first/last frame and timestamps so the review UI can play seamlessly.
    """

    def __init__(self, video_dir: Path, camera_id: str):
        self._dir = video_dir
        self._cam = camera_id
        self._writer: Optional[cv2.VideoWriter] = None
        self.frame_index = -1            # global, incremented to 0 on the first frame
        self.segments: List[dict] = []   # finalized segments
        self._seg_no = -1
        self._seg_first = 0
        self._seg_first_ts = 0
        self._frames_in_seg = 0
        self._cur_file = ""
        self._last_ts = 0

    def _open_segment(self, w: int, h: int, ts: float) -> None:
        self._seg_no += 1
        self._seg_first = self.frame_index + 1
        self._seg_first_ts = int(round(ts * 1000))
        self._frames_in_seg = 0
        self._cur_file = f"cam_{self._cam}_{self._seg_no:03d}.mp4"
        self._writer = cv2.VideoWriter(str(self._dir / self._cur_file), _FOURCC, _NOMINAL_FPS, (w, h))

    def _close_segment(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        if self._frames_in_seg > 0:
            self.segments.append({
                "segment": self._seg_no, "file": self._cur_file,
                "first_frame": self._seg_first, "last_frame": self.frame_index,
                "first_ts": self._seg_first_ts, "last_ts": self._last_ts,
            })

    def write(self, annotated, ts: float, frame_index: int) -> int:
        """Write one annotated frame. `frame_index` is the manager-owned global counter
        (single source of truth, shared with the JSONL) — the sink no longer increments
        its own, so video segments and detections can never drift apart."""
        h, w = annotated.shape[:2]
        if self._writer is None:
            self._open_segment(w, h, ts)
        # rotate on frame cap, or on size (checked periodically — VideoWriter hides bytes written)
        rotate = self._frames_in_seg >= SEG_MAX_FRAMES
        if not rotate and self._frames_in_seg and self._frames_in_seg % 300 == 0:
            try:
                if (self._dir / self._cur_file).stat().st_size >= SEG_MAX_BYTES:
                    rotate = True
            except OSError:
                pass
        if rotate:
            self._close_segment()
            self._open_segment(w, h, ts)
        self.frame_index = frame_index
        self._frames_in_seg += 1
        self._last_ts = int(round(ts * 1000))
        self._writer.write(annotated)
        return self.frame_index

    def segment_count(self) -> int:
        return len(self.segments) + (1 if self._writer is not None else 0)

    def index(self) -> List[dict]:
        """Segment index including the still-open current segment (for live status/review)."""
        idx = list(self.segments)
        if self._writer is not None and self._frames_in_seg > 0:
            idx.append({
                "segment": self._seg_no, "file": self._cur_file,
                "first_frame": self._seg_first, "last_frame": self.frame_index,
                "first_ts": self._seg_first_ts, "last_ts": self._last_ts,
            })
        return idx

    def close(self) -> None:
        self._close_segment()


class ValidationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._session_id: Optional[str] = None
        self._dir: Optional[Path] = None
        self._jsonl = None
        self._sinks: Dict[str, _CameraSink] = {}
        self._threshold: float = _settings.match_threshold
        self._meta: dict = {}
        self._session_type: str = "crossing"
        self._record_video: bool = _settings.validation_record_video
        # Per-camera global frame counter — the single source of truth for frame_index,
        # whether or not video is recorded (when recording, it is also passed to the sink).
        self._frames: Dict[str, int] = {}
        # Latest processed frame per camera (boxes only) for live click-to-assign; never persisted.
        self._last_frame: Dict[str, dict] = {}
        # Seating map for a common session: ordered list of resolved seats
        # [{"seat", "label", "true_person_id", "non_mate"}], left→right then row by row.
        self._seating: Optional[List[dict]] = None
        # Current run context (which subject is crossing + under which condition preset).
        # Set via set_run_context(); used to tag detections and derive automatic ground truth.
        self._run: dict = {"subject_label": None, "preset_id": None, "true_person_id": None, "non_mate": None}

    # ── lifecycle ───────────────────────────────────────────────────────────
    def is_active(self, camera_id: Optional[str] = None) -> bool:
        return self._active

    def start(self, name: str, enrolled: List[dict], threshold: float, *,
              session_type: str = "crossing", subjects: Optional[dict] = None,
              presets: Optional[List[dict]] = None, destination: Optional[str] = None,
              record_video: Optional[bool] = None,
              trial_subjects: Optional[List[dict]] = None,
              preset_id: Optional[str] = None) -> dict:
        with self._lock:
            if self._active:
                raise RuntimeError("Sessione di validazione già attiva")
            if destination is not None:
                set_destination(destination)  # external disk (validated by the caller)
            self._record_video = (
                _settings.validation_record_video if record_video is None else bool(record_video)
            )
            now = datetime.now()
            trial_key = None
            trial_n = None
            if trial_subjects:
                # AUTO-generated name (operator never types it): derived from the registry
                # selection + preset + profile → name↔GT cannot diverge. provaN = global counter.
                prof_name = profile_summary()["profile"]
                labels = [s.get("label", "?") for s in trial_subjects]
                trial_key = make_trial_key(labels, preset_id, prof_name)
                trial_n = count_trials(trial_key) + 1
                names_part = _slugify("-".join(
                    (s.get("name") or s.get("label") or "?") for s in trial_subjects), maxlen=60)
                preset_part = _slugify(preset_id or "nopreset", maxlen=20)
                session_id = "{}_{}_{}_{}".format(
                    now.strftime("%Y%m%d_%H%M%S"), names_part, preset_part, profile_slug())
                base = "{}_prova{}".format(session_id, trial_n)
            else:
                # Legacy/CLI path (main.py --validation NAME): old naming preserved.
                session_id = now.strftime("%Y%m%d_%H%M%S")
                if name:
                    safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40]
                    if safe:
                        session_id = f"{session_id}_{safe}"
                # Tag the profile in the folder name so Standard / Optimized sessions are
                # unambiguously separate on disk (e.g. …_standard / …_optTX2).
                base = f"{session_id}_{profile_slug()}"

            root = validation_root()
            # Robustness (Task E5): never silently fall back to internal storage. If an external
            # destination is configured, its mount point must exist & be writable now — otherwise
            # warn and STOP. Only the default internal root is auto-created.
            external = _dest_root is not None or bool(_settings.validation_dir.strip())
            if external:
                if not (root.parent.exists() and os.access(root.parent, os.W_OK)):
                    raise RuntimeError(
                        f"Destinazione esterna non disponibile/scrivibile ({root.parent}). "
                        "Monta il disco esterno (o correggi VALIDATION_DIR/destinazione) — "
                        "nessun fallback su storage interno."
                    )
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(f"Impossibile usare la root di validazione {root}: {exc}")

            # Nest the session inside the automatic DAILY folder validation/<YYYYMMDD>/ —
            # created/reused on the first start of the day, no manual open/close step.
            parent = root / day_folder(now)
            parent.mkdir(parents=True, exist_ok=True)

            # Never overwrite a session: create a NEW folder, appending _2/_3… on any collision.
            self._dir = parent / base
            n = 1
            while True:
                try:
                    self._dir.mkdir(parents=False, exist_ok=False)
                    break
                except FileExistsError:
                    n += 1
                    self._dir = parent / f"{base}_{n}"
                except OSError as exc:
                    raise RuntimeError(f"Impossibile creare la cartella di sessione in {root}: {exc}")
            session_id = self._dir.name  # actual id (possibly suffixed)
            if self._record_video:
                (self._dir / "video").mkdir(exist_ok=True)  # footage kept in a subfolder (§6 layout)
            self._jsonl = (self._dir / "detections.jsonl").open("w", encoding="utf-8")
            self._write_ms: List[float] = []   # JSONL append latencies (Tier-A disk telemetry)
            self._io0 = _proc_io()             # /proc/self/io baseline → delta at stop
            self._hw_samples: List[dict] = []  # per-embedding sysfs hw snapshots (aggregated at stop)
            self._energy: List[float] = []     # per-record VDD_IN·emb_ms (µJ) — only on real re-embed
            self._energy_frame: List[float] = []  # per-frame VDD_IN·t_total (µJ) — every processed frame
            self._sinks = {}
            self._frames = {}
            self._last_frame = {}
            self._seating = None
            self._threshold = threshold
            self._session_id = session_id
            self._session_type = session_type if session_type in ("crossing", "common") else "crossing"
            self._run = {"subject_label": None, "preset_id": None, "true_person_id": None, "non_mate": None}
            # How ground truth is established this session (no-video modes derive it live).
            gt_source = "crossing-declared" if self._session_type == "crossing" else "manual-live"
            subjects = subjects if subjects is not None else load_subjects()
            preset_list = presets if presets is not None else load_presets()
            self._meta = {
                "session_id": session_id,
                "name": name,
                "session_type": self._session_type,
                # explicit operator-facing label (Singolo/Gruppo) + trial identity for provaN
                "session_type_label": "Gruppo" if self._session_type == "common" else "Singolo",
                "subjects_full": trial_subjects or [],
                "trial_key": trial_key,
                "trial_n": trial_n,
                "record_video": self._record_video,
                "ground_truth_source": gt_source,
                # Active performance profile + its optimization parameters (Phase 1 vs Phase 2).
                "profile": profile_summary()["profile"],
                "performance": profile_summary(),
                "platform": detect_platform_label(),
                "start": datetime.now().isoformat(),
                "end": None,
                "threshold": threshold,
                "enrolled": enrolled,
                "subjects": subjects,
                # full preset parameters snapshotted → record is self-contained if a preset is later edited
                "presets": {p["preset_id"]: p for p in preset_list},
                "destination": {"root": str(self._dir.parent)},
                "cameras": _configured_cameras(),  # registry enabled; finalized to seen-cameras at stop
            }
            self._write_manifest()
            self._write_protocol()
            self._active = True
            mode = "video" if self._record_video else "no-video"
            logger.success(
                f"[Validation] Sessione avviata: {session_id} ({self._dir}) "
                f"[{self._session_type} · {mode}]")
            return {
                "session_id": session_id, "dir": str(self._dir),
                "session_type": self._session_type, "record_video": self._record_video,
            }

    def set_run_context(self, subject_label: Optional[str], preset_id: Optional[str]) -> dict:
        """Set which subject is now crossing and under which condition preset. Subsequent
        detections are tagged with this and, for a declared subject, get automatic ground
        truth (S→mate person_id, U→non-mate). Logged to runs.jsonl with the frame index."""
        with self._lock:
            if not self._active:
                raise RuntimeError("Nessuna sessione di validazione attiva")
            gt = subject_truth(subject_label) if subject_label else None
            if subject_label and gt is None:
                raise ValueError(f"Soggetto non nel registro: {subject_label}")
            if preset_id and preset_id not in self._meta.get("presets", {}):
                raise ValueError(f"Preset non in sessione: {preset_id}")
            self._run = {
                "subject_label": subject_label or None,
                "preset_id": preset_id or None,
                "true_person_id": (gt or {}).get("true_person_id"),
                "non_mate": (gt or {}).get("non_mate"),
            }
            frame_index = max(self._frames.values(), default=-1) + 1
            rec = {"timestamp_ms": int(round(time.time() * 1000)), "frame_index": frame_index, **self._run}
            with (self._dir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info(f"[Validation] Run context: soggetto={subject_label} preset={preset_id} @frame {frame_index}")
            return dict(self._run)

    def set_seating_map(self, seats: List[dict]) -> dict:
        """Declare the seating map for a common session: an ordered list of occupied positions
        (left→right, rows top→bottom) → subject label (`S…`/`U…`). Each detected face is then
        assigned the ground truth of the matching seat (deterministic, no video, no clicking).
        Labels are resolved against the registry up front (unknown → error)."""
        with self._lock:
            if not self._active:
                raise RuntimeError("Nessuna sessione di validazione attiva")
            if self._session_type != "common":
                raise RuntimeError("La seating-map si applica solo alle sessioni comuni")
            resolved: List[dict] = []
            for pos, s in enumerate(seats or []):
                label = (s.get("label") or "").strip()
                if not label:
                    continue
                gt = subject_truth(label)
                if gt is None:
                    raise ValueError(f"Soggetto non nel registro: {label}")
                resolved.append({
                    "seat": s.get("seat", pos),
                    "label": label,
                    "true_person_id": gt.get("true_person_id"),
                    "non_mate": gt.get("non_mate"),
                })
            self._seating = resolved or None
            self._meta["ground_truth_source"] = "seating-map" if resolved else "manual-live"
            self._meta["seating"] = resolved
            self._write_manifest()
            rec = {"timestamp_ms": int(round(time.time() * 1000)),
                   "frame_index": max(self._frames.values(), default=-1) + 1,
                   "seating": resolved}
            with (self._dir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info(f"[Validation] Seating-map: {len(resolved)} posti dichiarati")
            return {"seating": resolved}

    def live_detections(self, camera_id: str) -> dict:
        """Latest processed frame's boxes for one camera (for live click-to-assign). Returns
        metadata only — frame_index + per-face bbox/predicted identity; no image is kept."""
        with self._lock:
            if not self._active:
                return {"active": False}
            snap = self._last_frame.get(camera_id) or {}
            return {
                "active": True,
                "session_id": self._session_id,
                "camera_id": camera_id,
                "frame_index": snap.get("frame_index"),
                "faces": snap.get("faces", []),
            }

    def stop(self) -> dict:
        with self._lock:
            if not self._active:
                return {"active": False}
            if self._jsonl:
                self._jsonl.flush()
                self._jsonl.close()
            if self._record_video:
                for sink in self._sinks.values():
                    sink.close()
                # Persist the per-camera segment index so the review UI can play across segments.
                segments = {cam: s.index() for cam, s in self._sinks.items()}
                (self._dir / "video" / "segments.json").write_text(
                    json.dumps(segments, indent=2), encoding="utf-8")
            # Cameras/frames come from the global counter (works with or without video).
            self._meta["end"] = datetime.now().isoformat()
            # Refresh the platform label: at stop the analyzer is loaded, so the compute suffix
            # (-trt/-gpu/-cpu) reflects the REAL session even if start happened during warm-up.
            try:
                self._meta["platform"] = detect_platform_label()
            except Exception:
                pass
            self._meta["cameras"] = sorted(self._frames.keys())
            self._meta["frames"] = {cam: idx + 1 for cam, idx in self._frames.items()}
            # Tier-A session telemetry: disk I/O delta, RSS, JSONL append latency (mean/p95).
            io1 = _proc_io()
            io0 = getattr(self, "_io0", None) or {"read_bytes": 0, "write_bytes": 0}
            self._meta["io"] = {"read_bytes": io1["read_bytes"] - io0["read_bytes"],
                                "write_bytes": io1["write_bytes"] - io0["write_bytes"]}
            self._meta["mem_vmrss_kb"] = _vmrss_kb()
            w = sorted(getattr(self, "_write_ms", []) or [])
            if w:
                self._meta["jsonl_write_ms"] = {
                    "n": len(w), "mean": round(sum(w) / len(w), 3),
                    "p95": round(w[min(len(w) - 1, int(0.95 * len(w)))], 3)}
            try:
                from core.perfcounters import perf_counters
                self._meta["perf_counters"] = "available" if perf_counters.available else "unavailable"
            except Exception:
                self._meta["perf_counters"] = "unavailable"
            # Hardware telemetry aggregates (sysfs, in-process): per-channel mean/median/p95/peak,
            # energy per identification (∫VDD_IN over embed time), sampling overhead, throttling.
            self._finalize_hw_telemetry()
            if not self._meta.get("destination_device"):
                dev = _device_for_path(str(self._dir))
                if dev:
                    self._meta["destination_device"] = dev
            self._write_manifest()
            sid = self._session_id
            self._active = False
            self._jsonl = None
            self._sinks = {}
            self._frames = {}
            self._last_frame = {}
            logger.success(f"[Validation] Sessione terminata: {sid}")
            return {"active": False, "session_id": sid}

    def status(self) -> dict:
        with self._lock:
            if not self._active:
                return {"active": False}
            return {
                "active": True,
                "session_id": self._session_id,
                "session_type": self._session_type,
                "record_video": self._record_video,
                "ground_truth_source": self._meta.get("ground_truth_source"),
                "threshold": self._threshold,
                "run": dict(self._run),
                "seating": self._seating,
                "destination": self._meta.get("destination"),
                "frames": {cam: idx + 1 for cam, idx in self._frames.items()},
                "segments": {cam: s.segment_count() for cam, s in self._sinks.items()},
            }

    def _write_manifest(self):
        (self._dir / "session.json").write_text(
            json.dumps(self._meta, indent=2), encoding="utf-8"
        )

    def _write_protocol(self):
        (self._dir / "PROTOCOL.md").write_text(build_protocol_md(self._meta), encoding="utf-8")

    # ── seating-map ordering ──────────────────────────────────────────────────
    @staticmethod
    def _order_faces_by_seat(details: List[dict]) -> List[int]:
        """Return detection indices in reading order: rows top→bottom, left→right within a
        row. Rows are grouped by vertical center with a tolerance of ~0.6× the median face
        height, so small vertical jitter doesn't split a row. bbox = [top, right, bottom, left].
        """
        if not details:
            return []
        items = []
        for i, d in enumerate(details):
            top, right, bottom, left = d["bbox"]
            items.append((i, (left + right) / 2.0, (top + bottom) / 2.0, max(1.0, bottom - top)))
        med_h = sorted(x[3] for x in items)[len(items) // 2]
        tol = 0.6 * med_h
        rows: List[List[tuple]] = []
        for it in sorted(items, key=lambda x: x[2]):           # by vertical center
            if rows and (it[2] - rows[-1][0][2]) <= tol:        # within tol of the row anchor
                rows[-1].append(it)
            else:
                rows.append([it])
        order: List[int] = []
        for row in rows:
            order.extend(it[0] for it in sorted(row, key=lambda x: x[1]))  # left→right
        return order

    def _seating_truth(self, details: List[dict]) -> Dict[int, dict]:
        """Map each detection (by face_id) to its declared seat's ground truth. Extra faces
        beyond the declared seats stay unlabeled rather than being mis-assigned."""
        out: Dict[int, dict] = {}
        for seat_pos, face_idx in enumerate(self._order_faces_by_seat(details)):
            if seat_pos >= len(self._seating):
                break
            seat = self._seating[seat_pos]
            out[face_idx] = {
                "subject_label": seat.get("label"),
                "true_person_id": seat.get("true_person_id"),
                "non_mate": seat.get("non_mate"),
            }
        return out

    # ── per-frame recording ───────────────────────────────────────────────────
    def _finalize_hw_telemetry(self) -> None:
        """Aggregate the accumulated per-frame sysfs snapshots into the session manifest:
        hw_aggregate (mean/median/p95/peak per numeric channel), energy_per_id (µJ), the measured
        sampling cost, and the throttling fraction. No-op if no hw samples were captured."""
        samples = getattr(self, "_hw_samples", None) or []
        if samples:
            # Flatten each snapshot to dotted numeric keys, then aggregate per key.
            series: Dict[str, List[float]] = {}
            throttled = 0
            for s in samples:
                if s.get("throttling"):
                    throttled += 1
                for k, v in _flatten_numeric(s).items():
                    series.setdefault(k, []).append(v)
            agg = {}
            for k, vals in series.items():
                vs = sorted(vals)
                n = len(vs)
                agg[k] = {
                    "mean": round(sum(vs) / n, 2),
                    "median": round(vs[n // 2], 2),
                    "p95": round(vs[min(n - 1, int(0.95 * n))], 2),
                    "peak": round(vs[-1], 2),
                }
            self._meta["hw_aggregate"] = agg
            self._meta["throttling_frac"] = round(throttled / len(samples), 3)
        # Energy per identification (∫VDD_IN over embed time). With the tracker, a persistent face
        # skips re-embedding (emb_ms=0) → no energy sample: report that EXPLICITLY, never a bare
        # None that reads as a failed measurement.
        energy = getattr(self, "_energy", None) or []
        frame_energy = getattr(self, "_energy_frame", None) or []
        if energy:
            self._meta["energy_per_id"] = {
                "n": len(energy),
                "total_uj": round(sum(energy), 1),
                "mean_uj": round(sum(energy) / len(energy), 1),
            }
        elif frame_energy:
            self._meta["energy_per_id"] = {"n": 0, "note": "no re-embed nel periodo (tracker hit)"}
        else:
            self._meta["energy_per_id"] = {"note": "misura non disponibile (VDD_IN assente)"}
        # Per-frame mean energy (∫VDD_IN over every processed frame) — always available whenever
        # power was sampled, so the metric is non-empty even under full tracking.
        if frame_energy:
            self._meta["energy_per_frame"] = {
                "n": len(frame_energy),
                "total_uj": round(sum(frame_energy), 1),
                "mean_uj": round(sum(frame_energy) / len(frame_energy), 1),
            }
        try:
            from core.jetson_sysfs import sysfs_telemetry
            cost = sysfs_telemetry.probe_cost_us()
            if cost is not None:
                self._meta["telemetry_sampling_cost_us"] = cost
        except Exception:
            pass

    def record(self, camera_id: str, frame, results, details: List[dict],
               stages: Optional[dict] = None) -> None:
        """Write the per-detection JSONL for one frame (and, when video is on, an annotated
        video frame). `results` is the pipeline's [(location, pid, name, conf), ...]; `details`
        the aligned per-face dicts; `stages` the Tier-A per-frame telemetry (per-stage ms,
        cycles/instructions, n_faces, det input resolution) merged into each record. Called for
        every processed frame while active — including zero-detection frames. With video off,
        no image bytes are ever written to disk.
        """
        if not self._active:
            return
        annotated = None
        if self._record_video:
            from ui.display import annotate_frame  # lazy: avoids core↔ui import cycle
            annotated = annotate_frame(frame, results)
        ts = time.time()
        ts_ms = int(round(ts * 1000))
        with self._lock:
            if not self._active:
                return
            # Manager-owned global frame counter (single source of truth, video or not).
            frame_index = self._frames.get(camera_id, -1) + 1
            self._frames[camera_id] = frame_index
            if self._record_video:
                sink = self._sinks.get(camera_id)
                if sink is None:
                    sink = _CameraSink(self._dir / "video", camera_id)
                    self._sinks[camera_id] = sink
                sink.write(annotated, ts, frame_index)
            # Seating-map automatic ground truth (common session): position → declared seat.
            seat_gt = (self._seating_truth(details)
                       if self._session_type == "common" and self._seating else None)
            faces_live = []
            for face_id, d in enumerate(details):
                rec = {
                    "session_id": self._session_id,
                    "timestamp_ms": ts_ms,
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "face_id": face_id,
                    "predicted_identity": d["predicted_identity"],
                    "predicted_person_id": d["predicted_person_id"],
                    "confidence": d["confidence"],
                    "raw_cosine_distance": d["raw_cosine_distance"],
                    "best_match_person_id": d["best_match_person_id"],
                    "candidates": d.get("candidates", []),
                    "bbox": d["bbox"],
                }
                # Condition tag (per-condition breakdown) — independent of the GT source.
                if self._run.get("preset_id") is not None:
                    rec["preset_id"] = self._run["preset_id"]
                # Ground truth: seating map for a common session; otherwise the crossing run
                # context. Manual labels.jsonl still override either at metric time.
                gt = seat_gt.get(face_id) if seat_gt is not None else None
                if gt is not None:
                    for k in ("subject_label", "true_person_id", "non_mate"):
                        if gt.get(k) is not None:
                            rec[k] = gt[k]
                elif seat_gt is None:
                    for field in ("subject_label", "true_person_id", "non_mate"):
                        if self._run.get(field) is not None:
                            rec[field] = self._run[field]
                if stages:
                    rec.update(stages)   # Tier-A telemetry: t_ms{}, cycles, instructions, …
                t_w = time.perf_counter()
                self._jsonl.write(json.dumps(rec) + "\n")
                self._write_ms.append((time.perf_counter() - t_w) * 1000.0)
                faces_live.append({"face_id": face_id, "bbox": d["bbox"],
                                   "predicted_identity": d["predicted_identity"]})
            self._jsonl.flush()
            # Per-FRAME hw accumulation (one snapshot/frame, not per face) for the session
            # aggregates and the energy-per-identification estimate. mW·ms = µJ exactly.
            if stages and isinstance(stages.get("hw"), dict):
                hw = stages["hw"]
                self._hw_samples.append(hw)
                vdd = (hw.get("power_mw") or {}).get("VDD_IN")
                t_ms = stages.get("t_ms") or {}
                emb_ms = t_ms.get("emb")
                if vdd is not None:
                    # per-identification energy only when a real re-embed happened this frame
                    if emb_ms:
                        self._energy.append(float(vdd) * float(emb_ms))
                    # per-frame energy always (works under full tracking too)
                    total_ms = t_ms.get("total")
                    if total_ms:
                        self._energy_frame.append(float(vdd) * float(total_ms))
            # Latest boxes for live click-to-assign — metadata only, never an image on disk.
            self._last_frame[camera_id] = {
                "frame_index": frame_index, "ts_ms": ts_ms, "faces": faces_live,
            }


validation_manager = ValidationManager()
