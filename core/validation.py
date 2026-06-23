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
- **Tipo sessione:** {'sessione comune (soggetti statici, labeling manuale)' if meta.get('session_type') == 'common' else 'attraversamento singolo-soggetto (ground-truth automatica dal soggetto dichiarato)'}
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

`video/cam_<id>.mp4` — video **annotato** (box + nome + confidenza), sincronizzato per
`frame_index`. È materiale sensibile (reintroduce immagini): resta solo nel volume, separato dal
DB biometrico, ed è cancellabile con `scripts/clear_validation.py`.

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
  PROTOCOL.md   session.json   detections.jsonl   labels.jsonl
  metrics.json  det_curve.csv  cmc.csv            video/cam_<id>.mp4
```
"""


class _CameraSink:
    def __init__(self, video_path: Path):
        self._video_path = video_path
        self._writer: Optional[cv2.VideoWriter] = None
        self.frame_index = -1  # incremented to 0 on first frame

    def write(self, annotated):
        if self._writer is None:
            h, w = annotated.shape[:2]
            self._writer = cv2.VideoWriter(str(self._video_path), _FOURCC, _NOMINAL_FPS, (w, h))
        self.frame_index += 1
        self._writer.write(annotated)
        return self.frame_index

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


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
        # Current run context (which subject is crossing + under which condition preset).
        # Set via set_run_context(); used to tag detections and derive automatic ground truth.
        self._run: dict = {"subject_label": None, "preset_id": None, "true_person_id": None, "non_mate": None}

    # ── lifecycle ───────────────────────────────────────────────────────────
    def is_active(self, camera_id: Optional[str] = None) -> bool:
        return self._active

    def start(self, name: str, enrolled: List[dict], threshold: float, *,
              session_type: str = "crossing", subjects: Optional[dict] = None,
              presets: Optional[List[dict]] = None, destination: Optional[str] = None) -> dict:
        with self._lock:
            if self._active:
                raise RuntimeError("Sessione di validazione già attiva")
            if destination is not None:
                set_destination(destination)  # external disk (validated by the caller)
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if name:
                safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40]
                if safe:
                    session_id = f"{session_id}_{safe}"
            self._dir = validation_root() / session_id
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / "video").mkdir(exist_ok=True)  # footage kept in a subfolder (§6 layout)
            self._jsonl = (self._dir / "detections.jsonl").open("w", encoding="utf-8")
            self._sinks = {}
            self._threshold = threshold
            self._session_id = session_id
            self._session_type = session_type if session_type in ("crossing", "common") else "crossing"
            self._run = {"subject_label": None, "preset_id": None, "true_person_id": None, "non_mate": None}
            subjects = subjects if subjects is not None else load_subjects()
            preset_list = presets if presets is not None else load_presets()
            self._meta = {
                "session_id": session_id,
                "name": name,
                "session_type": self._session_type,
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
            logger.success(f"[Validation] Sessione avviata: {session_id} ({self._dir}) [{self._session_type}]")
            return {"session_id": session_id, "dir": str(self._dir), "session_type": self._session_type}

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
            frame_index = max((s.frame_index for s in self._sinks.values()), default=-1) + 1
            rec = {"timestamp_ms": int(round(time.time() * 1000)), "frame_index": frame_index, **self._run}
            with (self._dir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info(f"[Validation] Run context: soggetto={subject_label} preset={preset_id} @frame {frame_index}")
            return dict(self._run)

    def stop(self) -> dict:
        with self._lock:
            if not self._active:
                return {"active": False}
            for sink in self._sinks.values():
                sink.close()
            if self._jsonl:
                self._jsonl.flush()
                self._jsonl.close()
            self._meta["end"] = datetime.now().isoformat()
            self._meta["cameras"] = sorted(self._sinks.keys())
            self._meta["frames"] = {cam: s.frame_index + 1 for cam, s in self._sinks.items()}
            self._write_manifest()
            sid = self._session_id
            self._active = False
            self._jsonl = None
            self._sinks = {}
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
                "threshold": self._threshold,
                "run": dict(self._run),
                "destination": self._meta.get("destination"),
                "frames": {cam: s.frame_index + 1 for cam, s in self._sinks.items()},
            }

    def _write_manifest(self):
        (self._dir / "session.json").write_text(
            json.dumps(self._meta, indent=2), encoding="utf-8"
        )

    def _write_protocol(self):
        (self._dir / "PROTOCOL.md").write_text(build_protocol_md(self._meta), encoding="utf-8")

    # ── per-frame recording ───────────────────────────────────────────────────
    def record(self, camera_id: str, frame, results, details: List[dict]) -> None:
        """Write one annotated video frame + the JSONL detection records for it.

        `results` is the pipeline's [(location, pid, name, conf), ...]; `details` the
        aligned list of per-face dicts (raw distance, best match, bbox). Called for every
        processed frame while active — including frames with zero detections (continuous video).
        """
        if not self._active:
            return
        from ui.display import annotate_frame  # lazy: avoids core↔ui import cycle

        annotated = annotate_frame(frame, results)
        ts = time.time()
        with self._lock:
            if not self._active:
                return
            sink = self._sinks.get(camera_id)
            if sink is None:
                sink = _CameraSink(self._dir / "video" / f"cam_{camera_id}.mp4")
                self._sinks[camera_id] = sink
            frame_index = sink.write(annotated)
            for face_id, d in enumerate(details):
                rec = {
                    "session_id": self._session_id,
                    "timestamp_ms": int(round(ts * 1000)),
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
                # Run-context tags + automatic ground truth (crossing sessions). For a common
                # session the run context is unset → no GT here (manual labelling is used).
                run = self._run
                for field in ("subject_label", "preset_id", "true_person_id", "non_mate"):
                    if run.get(field) is not None:
                        rec[field] = run[field]
                self._jsonl.write(json.dumps(rec) + "\n")
            self._jsonl.flush()


validation_manager = ValidationManager()
