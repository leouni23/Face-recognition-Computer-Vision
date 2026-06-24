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
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from loguru import logger

from config.settings import get_settings
from core.metrics import detect_platform_label
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
              record_video: Optional[bool] = None) -> dict:
        with self._lock:
            if self._active:
                raise RuntimeError("Sessione di validazione già attiva")
            if destination is not None:
                set_destination(destination)  # external disk (validated by the caller)
            self._record_video = (
                _settings.validation_record_video if record_video is None else bool(record_video)
            )
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if name:
                safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40]
                if safe:
                    session_id = f"{session_id}_{safe}"
            self._dir = validation_root() / session_id
            self._dir.mkdir(parents=True, exist_ok=True)
            if self._record_video:
                (self._dir / "video").mkdir(exist_ok=True)  # footage kept in a subfolder (§6 layout)
            self._jsonl = (self._dir / "detections.jsonl").open("w", encoding="utf-8")
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
                "record_video": self._record_video,
                "ground_truth_source": gt_source,
                "platform": detect_platform_label(),
                "start": datetime.now().isoformat(),
                "end": None,
                "threshold": threshold,
                "enrolled": enrolled,
                "subjects": subjects,
                # full preset parameters snapshotted → record is self-contained if a preset is later edited
                "presets": {p["preset_id"]: p for p in preset_list},
                "destination": {"root": str(self._dir.parent)},
                "cameras": list(_settings.cameras),  # configured; finalized to seen-cameras at stop
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
            self._meta["cameras"] = sorted(self._frames.keys())
            self._meta["frames"] = {cam: idx + 1 for cam, idx in self._frames.items()}
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
    def record(self, camera_id: str, frame, results, details: List[dict]) -> None:
        """Write the per-detection JSONL for one frame (and, when video is on, an annotated
        video frame). `results` is the pipeline's [(location, pid, name, conf), ...]; `details`
        the aligned per-face dicts. Called for every processed frame while active — including
        zero-detection frames. With video off, no image bytes are ever written to disk.
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
                self._jsonl.write(json.dumps(rec) + "\n")
                faces_live.append({"face_id": face_id, "bbox": d["bbox"],
                                   "predicted_identity": d["predicted_identity"]})
            self._jsonl.flush()
            # Latest boxes for live click-to-assign — metadata only, never an image on disk.
            self._last_frame[camera_id] = {
                "frame_index": frame_index, "ts_ms": ts_ms, "faces": faces_live,
            }


validation_manager = ValidationManager()
