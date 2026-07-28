import hmac
import json
import os
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from loguru import logger

from config.settings import get_settings, snap_det_dim
from core.camera import probe_source
from core.camera_manager import camera_manager
from core.detector import (
    apply_detection_settings, detect_and_encode, get_detection_settings, get_loaded_info,
    get_warmup_state, reset_for_profile_change,
)
from core.profile import OPTIMIZED, STANDARD, get_profile, profile_summary
from core.geometry import compute_homography, solve_polar_calibration
from core.validation import validation_manager, validation_root
from core.validation_metrics import compute_and_export
from core.validation_presets import (
    delete_preset, duplicate_preset, load_presets, load_subjects, save_subjects, upsert_preset,
)
from core.storage import estimate_bytes, list_drives, validate_target
from database.models import Person
from database.repository import CalibrationRepository, PersonRepository
from database.session import get_session
from web.broadcaster import broadcaster

_settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"
MAPS_DIR = STATIC_DIR / "maps"
_MAP_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Nome persona: lettere/numeri/spazi e .'-  — max 100 caratteri (anche difesa XSS in profondità)
_NAME_RE = re.compile(r"^[\w\s'.\-]{1,100}$", re.UNICODE)

app = Flask(__name__, static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # upload max 10 MB


@app.before_request
def _guard_request():
    # /healthz è l'UNICO path esente da autenticazione: serve al HEALTHCHECK Docker (che non ha
    # credenziali) e non espone alcun dato sensibile. Ogni altro path resta protetto.
    if request.path == "/healthz":
        return None
    # CSRF: rifiuta richieste cross-site che modificano lo stato (Sec-Fetch-Site è
    # impostato dai browser moderni e non è falsificabile da una pagina malevola)
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and \
            request.headers.get("Sec-Fetch-Site") == "cross-site":
        return jsonify({"error": "Richiesta cross-site non consentita"}), 403

    # Basic Auth opzionale (WEB_PASSWORD nel .env) — obbligatoria se esposta oltre localhost
    password = _settings.web_password
    if password:
        auth = request.authorization
        if not (auth and auth.password and hmac.compare_digest(auth.password, password)):
            return Response(
                "Autenticazione richiesta", 401,
                {"WWW-Authenticate": 'Basic realm="Face ID"'},
            )


@app.after_request
def _security_headers(resp: Response) -> Response:
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    # Always revalidate HTML so UI updates are never masked by a stale cached page
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp

# ── Enrollment session (one at a time) ────────────────────────────────────────
_enroll_lock = threading.Lock()
_enroll_session: dict = {}  # {"name": str, "embeddings": List[np.ndarray], "required": int}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ── MJPEG stream ──────────────────────────────────────────────────────────────

@app.route("/stream/<path:camera_id>")
def video_stream(camera_id: str):
    def generate():
        broadcaster.add_viewer(camera_id)
        last = None
        try:
            while True:
                jpeg = broadcaster.get_frame(camera_id)
                # send only NEW frames — avoids re-pushing identical JPEGs to the client
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(0.04)   # ~25 FPS cap
        finally:
            broadcaster.remove_viewer(camera_id)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Server-Sent Events ────────────────────────────────────────────────────────

@app.route("/api/events")
def events_stream():
    def generate():
        q = broadcaster.subscribe()
        try:
            while True:
                try:
                    event = q.get(timeout=0.5)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            broadcaster.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Enrollment API ────────────────────────────────────────────────────────────

@app.route("/api/enroll/start", methods=["POST"])
def enroll_start():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Campo 'name' obbligatorio"}), 400
    if not _NAME_RE.match(name):
        return jsonify({"error": "Nome non valido: max 100 caratteri, solo lettere, numeri, spazi e .'-"}), 400
    try:
        required = max(1, min(20, int(data.get("samples", 5))))
    except (TypeError, ValueError):
        required = 5
    continuous = bool(data.get("continuous", False))
    with _enroll_lock:
        _enroll_session.clear()
        # continuous mode: the client auto-captures frames until Stop; cap bounds the memory (the
        # mean embedding has long since stabilised well before this).
        _enroll_session.update({"name": name, "embeddings": [], "required": required,
                                "continuous": continuous, "cap": 300})
    return jsonify({"message": f"Sessione avviata per '{name}'", "required": required,
                    "continuous": continuous})


@app.route("/api/enroll/capture", methods=["POST"])
def enroll_capture():
    with _enroll_lock:
        if not _enroll_session:
            return jsonify({"error": "Nessuna sessione attiva. Chiama /api/enroll/start prima."}), 400
        name = _enroll_session["name"]
        required = _enroll_session["required"]
        continuous = _enroll_session.get("continuous", False)
        cap = _enroll_session.get("cap", 300)
        collected = len(_enroll_session["embeddings"])

    # Continuous mode is a client-driven ~5 fps loop: a frame with no / multiple faces must NOT
    # error the loop — return 200 with captured=false so it just keeps going. Manual mode keeps
    # the strict 422 responses. A safety cap bounds the collected samples.
    def soft(reason, code=200):
        if continuous:
            return jsonify({"collected": collected, "required": required, "continuous": True,
                            "captured": False, "reason": reason}), code
        return jsonify({"error": reason}), 422

    if continuous and collected >= cap:
        return jsonify({"collected": collected, "required": required, "continuous": True,
                        "captured": False, "capped": True}), 200

    camera_id = (broadcaster.camera_ids or ["0"])[0]
    frame = broadcaster.get_raw_frame(camera_id)
    if frame is None:
        if continuous:
            return jsonify({"collected": collected, "required": required, "continuous": True,
                            "captured": False, "reason": "Nessun frame dalla camera"}), 200
        return jsonify({"error": "Nessun frame disponibile dalla camera"}), 503

    faces = detect_and_encode(frame)
    if not faces:
        return soft("Nessun volto rilevato. Avvicinati alla camera.")
    if len(faces) > 1:
        return soft("Più volti nell'inquadratura. Assicurati di essere solo.")

    _, embedding = faces[0]
    with _enroll_lock:
        # session may have been cancelled while the frame was being processed
        if not _enroll_session:
            return jsonify({"error": "Sessione annullata"}), 409
        _enroll_session["embeddings"].append(embedding)
        collected = len(_enroll_session["embeddings"])
        done = (not continuous) and collected >= required

    return jsonify({"collected": collected, "required": required, "continuous": continuous,
                    "captured": True, "done": done})


@app.route("/api/enroll/save", methods=["POST"])
def enroll_save():
    with _enroll_lock:
        if not _enroll_session:
            return jsonify({"error": "Nessuna sessione attiva"}), 400
        name = _enroll_session["name"]
        embeddings = list(_enroll_session["embeddings"])

    if not embeddings:
        return jsonify({"error": "Nessun campione acquisito"}), 400

    mean_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    mean_embedding /= np.linalg.norm(mean_embedding)

    with get_session() as session:
        repo = PersonRepository(session)
        person = repo.get_by_name(name)
        if person:
            action = "aggiornata"
        else:
            person = repo.create(name)
            action = "iscritta"
        repo.give_consent(person)
        repo.add_template(person, mean_embedding)

    with _enroll_lock:
        _enroll_session.clear()

    if broadcaster.pipeline:
        broadcaster.pipeline.force_reload()

    logger.success(f"[Enrollment web] '{name}' {action} con {len(embeddings)} campioni.")
    return jsonify({"message": f"Persona '{name}' {action} con successo.", "samples": len(embeddings)})


@app.route("/api/enroll/cancel", methods=["POST"])
def enroll_cancel():
    with _enroll_lock:
        _enroll_session.clear()
    return jsonify({"message": "Sessione annullata"})


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route("/api/cameras")
def list_cameras():
    """Registry cameras with live status (connected/connecting/error/stopped/disabled)."""
    return jsonify(camera_manager.list_cameras())


@app.route("/api/cameras", methods=["POST"])
def add_camera():
    data = request.get_json(silent=True) or {}
    try:
        res = camera_manager.add(
            name=data.get("name", ""), source=data.get("source", ""),
            enabled=bool(data.get("enabled", True)), resolution=data.get("resolution") or None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(res), 201


@app.route("/api/cameras/<path:camera_id>", methods=["PATCH", "PUT"])
def edit_camera(camera_id: str):
    data = request.get_json(silent=True) or {}
    fields = {k: data[k] for k in ("name", "source", "enabled", "resolution") if k in data}
    try:
        ok = camera_manager.update(camera_id, **fields)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not ok:
        return jsonify({"error": "Camera non trovata"}), 404
    return jsonify({"ok": True})


@app.route("/api/cameras/<path:camera_id>", methods=["DELETE"])
def delete_camera(camera_id: str):
    if not camera_manager.remove(camera_id):
        return jsonify({"error": "Camera non trovata"}), 404
    return jsonify({"ok": True})


@app.route("/api/cameras/<path:camera_id>/test", methods=["POST"])
def test_camera(camera_id: str):
    """Probe a source (the posted one, or the registered camera's) and return ok + a snapshot."""
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or "").strip()
    if not source:
        cam = next((c for c in camera_manager.list_cameras() if c["cam_id"] == camera_id), None)
        if cam is None:
            return jsonify({"error": "Camera non trovata"}), 404
        # A running camera holds its device open (USB can't be opened twice) — test via its
        # live frame instead of re-probing.
        if cam["status"] == "connected":
            live = broadcaster.get_raw_frame(camera_id)
            if live is not None:
                ok, err, frame = True, None, live
            else:
                ok, err, frame = True, "Connessa (nessun frame in cache)", None
            source = None  # skip probe below
        else:
            source = cam["source"]
    if source:
        ok, err, frame = probe_source(source)
    snapshot = None
    if ok and frame is not None:
        enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if enc:
            import base64
            snapshot = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return jsonify({"ok": ok, "error": err, "snapshot": snapshot})


def _active_pack() -> str:
    """The recognition model pack ACTUALLY loaded (semaforo source); falls back to the requested
    profile before the analyzer is built."""
    from core.detector import get_loaded_info
    from core.profile import get_profile
    return (get_loaded_info() or {}).get("model_pack") or get_profile().model_pack


def _person_suffix(packs, active_pack):
    """UI-only model tag next to a name. Dual-pack → the ACTIVE model only (user rule); single
    pack → that pack; no template → '' . Never stored, purely visual."""
    from database.repository import pack_letter
    if not packs:
        return ""
    if active_pack in packs or len(packs) > 1:
        return "_" + pack_letter(active_pack)
    return "_" + pack_letter(packs[0])


@app.route("/api/persons")
def list_persons():
    from database.repository import pack_letter
    active_pack = _active_pack()
    with get_session() as session:
        repo = PersonRepository(session)
        packs_by_person = repo.person_packs()
        persons = (
            session.query(Person)
            .filter(Person.active == True, Person.consent_given == True)
            .order_by(Person.name)
            .all()
        )
        return jsonify([
            {
                "id": p.id,
                "name": p.name,
                "enrolled_at": p.enrolled_at.isoformat(),
                "last_seen": p.last_seen.isoformat() if p.last_seen else None,
                "packs": packs_by_person.get(p.id, []),
                "suffix": _person_suffix(packs_by_person.get(p.id, []), active_pack),
                "active_pack": active_pack,
                "active_letter": pack_letter(active_pack),
                # True only if this person has a template for the ACTIVE pack, i.e. can actually be
                # recognized right now. The UI shows these as the gallery; the others are listed
                # apart as "registrati su un altro modello" (no phantom residue after a switch).
                "for_active_pack": active_pack in packs_by_person.get(p.id, []),
            }
            for p in persons
        ])


@app.route("/api/persons/<name>", methods=["DELETE"])
def delete_person(name: str):
    with get_session() as session:
        count = PersonRepository(session).delete_person(name)
    if not count:
        return jsonify({"error": f"Persona '{name}' non trovata"}), 404
    # Reload templates NOW: otherwise the recognizer keeps the deleted person's
    # template for up to 60 s and would log events/positions for a person_id that
    # no longer exists → FOREIGN KEY violation that kills the camera worker.
    if broadcaster.pipeline is not None:
        broadcaster.pipeline.force_reload()
    logger.info(f"[API] Dati biometrici di '{name}' eliminati (GDPR Art. 17)")
    return jsonify({"message": f"Persona '{name}' eliminata ({count} record)."})


@app.route("/api/persons/templates/wipe", methods=["POST"])
def wipe_templates():
    """One-off cleanup: drop ALL face templates (keeps persons/consent). For a DB polluted with
    mixed/untagged embeddings from different recognition models. Re-enroll afterwards."""
    with get_session() as session:
        n = PersonRepository(session).delete_all_templates()
    if broadcaster.pipeline is not None:
        broadcaster.pipeline.force_reload()
    logger.info(f"[API] Wipe template volto: {n} eliminati")
    return jsonify({"deleted": n})


@app.route("/api/persons/<int:person_id>/trajectory")
def person_trajectory(person_id: int):
    """Storico posizioni di un soggetto. Query param opzionale ?minutes=N (default 60)."""
    try:
        minutes = max(1, min(1440, int(request.args.get("minutes", 60))))
    except (TypeError, ValueError):
        minutes = 60
    since = datetime.utcnow() - timedelta(minutes=minutes)

    with get_session() as session:
        points = PersonRepository(session).get_trajectory(person_id, since=since)
        return jsonify([
            {
                "camera": p.camera_id,
                "time": p.timestamp.isoformat(),
                "cx": p.bbox_cx,
                "cy": p.bbox_cy,
                "w": p.bbox_w,
                "h": p.bbox_h,
                "frame_w": p.frame_w,
                "frame_h": p.frame_h,
                "world_x": p.world_x,
                "world_y": p.world_y,
                "confidence": p.confidence,
            }
            for p in points
        ])


# ── Performance metrics ───────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    """FPS (per camera + aggregata), latenza per-stage e risorse GPU/CPU/RAM/power/temp."""
    from core.metrics import get_resource_metrics

    perf = {"cameras": {}, "aggregate_fps": 0.0}
    if broadcaster.pipeline is not None:
        perf = broadcaster.pipeline.perf.snapshot()
    out = {"perf": perf, "resources": get_resource_metrics()}
    from core.jetson_sysfs import sysfs_telemetry
    hw = sysfs_telemetry.sample()  # in-process sysfs channels (Jetson); {} elsewhere
    if hw:
        out["sysfs"] = hw
    return jsonify(out)


# ── Validation / test mode ────────────────────────────────────────────────────

def _session_dir(session_id: str):
    """Resolve a session folder safely (reject path traversal). Nest-aware: a session lives either
    directly under the root (legacy flat) or one level down inside a validation group."""
    if "/" in session_id or "\\" in session_id or session_id in ("", ".", ".."):
        return None
    root = validation_root().resolve()
    if not root.is_dir():
        return None
    candidates = [root]
    for d in root.iterdir():
        if d.is_dir():
            candidates.append(d)
    for parent in candidates:
        target = (parent / session_id).resolve()
        if target.is_dir() and (target / "session.json").is_file() and root in target.parents:
            return target
    return None


def _read_jsonl_file(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@app.route("/validation")
def validation_page():
    return send_from_directory(STATIC_DIR, "validation.html")


def _resolve_trial_subjects(subjects):
    """Attach full person names to the wizard's registry selection [{label, person_id}] so the
    manifest and the auto-generated session name carry real names, never bare S1/S2."""
    if not subjects:
        return None
    ids = [s.get("person_id") for s in subjects if s.get("person_id") is not None]
    names = {}
    if ids:
        with get_session() as session:
            for p in session.query(Person).filter(Person.id.in_(ids)).all():
                names[p.id] = p.name
    out = []
    for s in subjects:
        pid = s.get("person_id")
        out.append({
            "label": (s.get("label") or "").strip() or "?",
            "person_id": pid,
            "name": names.get(pid) or s.get("name") or ("Sconosciuto" if pid is None else "?"),
        })
    return out


@app.route("/api/validation/start", methods=["POST"])
def validation_start():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if name and not _NAME_RE.match(name):
        return jsonify({"error": "Nome sessione non valido"}), 400
    # Destination is MANDATORY server-side (it was UI-only → curl could start onto nothing):
    # resolve the effective destination (request or the configured VALIDATION_DIR default) and
    # RE-validate it NOW — a disk validated in the wizard may have been unmounted since.
    destination = (data.get("destination") or "").strip() or None
    if destination is None and _settings.validation_dir.strip():
        destination = str(Path(_settings.validation_dir.strip()).parent)
    if destination is None:
        return jsonify({"error": "Selezionare e validare una destinazione prima di avviare"}), 400
    v = validate_target(destination)
    if not v.get("ok"):
        return jsonify({"error": "Destinazione non valida allo start: "
                        + "; ".join(v.get("errors") or ["sconosciuto"])}), 400
    trial_subjects = _resolve_trial_subjects(data.get("subjects"))
    with get_session() as session:
        enrolled = [
            {"id": p.id, "name": p.name}
            for p in session.query(Person)
            .filter(Person.active == True, Person.consent_given == True)
            .order_by(Person.name)
            .all()
        ]
    record_video = data.get("record_video")
    try:
        info = validation_manager.start(
            name, enrolled, _settings.match_threshold,
            session_type=data.get("session_type", "crossing"),
            destination=(data.get("destination") or None),
            record_video=(None if record_video is None else bool(record_video)),
            trial_subjects=trial_subjects,
            preset_id=(data.get("preset_id") or None),
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({**info, "enrolled": enrolled, "threshold": _settings.match_threshold})


@app.route("/api/validation/preview_name", methods=["POST"])
def validation_preview_name():
    """Preview of the auto-generated session name (summary step) without creating anything.
    provaN is counted INSIDE the active campaign (a campaign that isn't open yet would be the
    auto daily one → the preview then correctly shows prova1)."""
    from core.profile import profile_summary as _ps
    from core.validation import active_campaign, count_trials, make_trial_key, _slugify
    data = request.get_json(silent=True) or {}
    trial_subjects = _resolve_trial_subjects(data.get("subjects")) or []
    preset_id = data.get("preset_id") or None
    prof = _ps()["profile"]
    labels = [s["label"] for s in trial_subjects]
    key = make_trial_key(labels, preset_id, prof)
    camp = active_campaign()
    n = count_trials(key, scope=(Path(camp["dir"]) if camp else None)) + 1
    names_part = _slugify("-".join(s["name"] for s in trial_subjects), maxlen=60) if trial_subjects else "x"
    from core.profile import profile_slug
    name = "{}_{}_{}_{}_prova{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"), names_part,
        _slugify(preset_id or "nopreset", maxlen=20), profile_slug(), n)
    return jsonify({"name": name, "trial_n": n, "trial_key": key,
                    "campaign": (camp or {}).get("name"),
                    "campaign_folder": (camp or {}).get("folder")})


@app.route("/api/validation/run", methods=["POST"])
def validation_set_run():
    """Set the current run context (which subject is crossing + condition preset).
    Subsequent detections are tagged and get automatic ground truth (crossing sessions)."""
    data = request.get_json(silent=True) or {}
    try:
        ctx = validation_manager.set_run_context(data.get("subject_label"), data.get("preset_id"))
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(ctx)


@app.route("/api/validation/seating", methods=["POST"])
def validation_seating():
    """Declare the seating map for a common session (no-video automatic ground truth):
    an ordered list of {seat, label} positions, left→right then row by row."""
    data = request.get_json(silent=True) or {}
    seats = data.get("seats") or data.get("seating") or []
    try:
        result = validation_manager.set_seating_map(seats)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/api/validation/live/<camera_id>")
def validation_live(camera_id: str):
    """Latest processed frame's boxes for live click-to-assign (metadata only, no image)."""
    return jsonify(validation_manager.live_detections(camera_id))


@app.route("/api/validation/stop", methods=["POST"])
def validation_stop():
    return jsonify(validation_manager.stop())


@app.route("/api/validation/status")
def validation_status():
    return jsonify(validation_manager.status())


def _read_session_manifests():
    """All session.json manifests, flat (legacy) + nested one level under validation groups."""
    root = validation_root()
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        if (d / "session.json").is_file():
            out.append((d, None))
        elif d.is_dir():
            for sub in sorted(d.iterdir(), reverse=True):
                if (sub / "session.json").is_file():
                    out.append((sub, d.name))
    res = []
    for sdir, group in out:
        try:
            m = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
            m["group"] = group
            res.append(m)
        except Exception:
            pass
    return res


@app.route("/api/validation/sessions")
def validation_sessions():
    return jsonify(_read_session_manifests())


@app.route("/api/validation/groups")
def validation_groups():
    """Archivio: sessions grouped by CAMPAIGN, plus every legacy container (daily folders, manual
    val_* groups) and flat sessions under "(senza campagna)". Read-only."""
    from core.compare import scan_sessions
    from core.validation import active_campaign
    groups = {}
    order = []
    camp = active_campaign()
    active_folder = camp["folder"] if camp else None
    for s in scan_sessions():
        key = s.get("group") or "__flat__"
        if key not in groups:
            gname = s.get("group_name") or "(senza campagna)"
            groups[key] = {
                "folder": s.get("group"), "name": gname,
                "is_test": s.get("is_test", False), "sessions": [], "profiles": set(),
            }
            order.append(key)
        g = groups[key]
        g["sessions"].append(s)
        if s.get("profile"):
            g["profiles"].add(s["profile"])
    # An empty campaign has no sessions, so scan_sessions can't surface it — merge them in.
    for c in _campaigns():
        if c["folder"] not in groups:
            groups[c["folder"]] = {"folder": c["folder"], "name": c["name"],
                                   "is_test": c["is_test"], "sessions": [], "profiles": set()}
            order.append(c["folder"])
    out = []
    for key in sorted(order, key=lambda k: (groups[k]["folder"] or ""), reverse=True):
        g = groups[key]
        out.append({
            "folder": g["folder"], "name": g["name"], "is_test": g["is_test"],
            "active": g["folder"] is not None and g["folder"] == active_folder,
            "n_sessions": len(g["sessions"]), "profiles": sorted(g["profiles"]),
            "sessions": g["sessions"],
        })
    return jsonify(out)


# ── Validation campaigns (experiment containers; provaN restarts inside each) ──────

def _campaigns():
    from core.validation import list_campaigns
    return list_campaigns()


@app.route("/api/validation/campaign/active")
def validation_campaign_active():
    from core.validation import active_campaign
    return jsonify({"active": active_campaign()})


@app.route("/api/validation/campaigns")
def validation_campaigns():
    return jsonify(_campaigns())


@app.route("/api/validation/campaign/start", methods=["POST"])
def validation_campaign_start():
    """Open a campaign. 409 if one is already open — the operator closes it explicitly (no
    implicit close), so sessions can never land in the wrong container."""
    from core.validation import start_campaign
    data = request.get_json(silent=True) or {}
    try:
        camp = start_campaign(name=data.get("name", ""), is_test=bool(data.get("is_test")),
                              notes=data.get("notes", ""))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    except OSError as exc:
        return jsonify({"error": "Impossibile creare la validazione: {}".format(exc)}), 500
    return jsonify(camp)


@app.route("/api/validation/campaign/close", methods=["POST"])
def validation_campaign_close():
    from core.validation import close_campaign
    if validation_manager.is_active():
        return jsonify({"error": "Ferma prima la sessione in corso"}), 409
    return jsonify(close_campaign())


@app.route("/api/validation/campaign/<folder>", methods=["DELETE"])
def validation_campaign_delete(folder: str):
    """Delete a TEST campaign and its sessions. Non-test → 403 (real data is never deletable
    from the UI)."""
    import shutil
    from core.validation import active_campaign, close_campaign, validation_root
    if "/" in folder or "\\" in folder or folder in ("", ".", ".."):
        return jsonify({"error": "id non valido"}), 400
    root = validation_root().resolve()
    d = (root / folder).resolve()
    if not (d.is_dir() and root in d.parents and (d / "campaign.json").is_file()):
        return jsonify({"error": "Validazione non trovata"}), 404
    try:
        meta = json.loads((d / "campaign.json").read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    if not meta.get("is_test"):
        return jsonify({"error": "Solo le validazioni di prova (test) sono eliminabili"}), 403
    if validation_manager.is_active():
        return jsonify({"error": "Ferma prima la sessione in corso"}), 409
    camp = active_campaign()
    if camp and camp["folder"] == folder:
        close_campaign()
    shutil.rmtree(str(d), ignore_errors=True)
    logger.info(f"[Validation] Validazione di prova eliminata: {folder}")
    return jsonify({"deleted": folder})


@app.route("/api/validation/compare")
def validation_compare():
    """Offline Phase-1 (Standard) vs Phase-2 (Optimized-TX2) comparison, recomputed from the
    saved JSONL of all sessions (no camera). `?by_preset=1` also splits by condition preset;
    `?format=csv` returns the per-profile CSV for the paper."""
    from core.compare import compare, to_csv
    by_preset = request.args.get("by_preset") in ("1", "true", "yes")
    result = compare(by_preset=by_preset)
    if request.args.get("format") == "csv":
        return Response(to_csv(result), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=phase_comparison.csv"})
    return jsonify(result)


@app.route("/api/validation/<session_id>/session")
def validation_session(session_id: str):
    """The session manifest (gallery, platform, threshold, cameras) for the review UI."""
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    manifest = d / "session.json"
    if not manifest.is_file():
        return jsonify({"error": "Manifest non trovato"}), 404
    return jsonify(json.loads(manifest.read_text(encoding="utf-8")))


@app.route("/api/validation/<session_id>/detections")
def validation_detections(session_id: str):
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    dets = _read_jsonl_file(d / "detections.jsonl")
    labels = {}  # detection key → ground truth ({"true_person_id":..} or {"non_mate":True})
    for r in _read_jsonl_file(d / "labels.jsonl"):  # append-only; later entries override
        try:
            k = (str(r["camera_id"]), int(r["frame_index"]), int(r["face_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        if r.get("non_mate") is True:
            labels[k] = {"non_mate": True}
        elif r.get("true_person_id") is not None:
            labels[k] = {"true_person_id": int(r["true_person_id"])}
    for det in dets:
        k = (str(det["camera_id"]), int(det["frame_index"]), int(det["face_id"]))
        if k in labels:                              # manual label wins (correction / common)
            det["truth"] = labels[k]
        elif det.get("non_mate") is True:            # else automatic GT from the run context
            det["truth"] = {"non_mate": True}
        elif det.get("true_person_id") is not None:
            det["truth"] = {"true_person_id": int(det["true_person_id"])}
        else:
            det["truth"] = None
    return jsonify(dets)


@app.route("/api/validation/<session_id>/labels", methods=["POST"])
def validation_save_labels(session_id: str):
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    data = request.get_json(silent=True) or {}
    items = data.get("labels") if "labels" in data else [data]
    rows = []
    try:
        for it in items:
            row = {
                "camera_id": str(it["camera_id"]),
                "frame_index": int(it["frame_index"]),
                "face_id": int(it["face_id"]),
            }
            # Ground truth, threshold-independent: an enrolled subject OR a non-mate.
            if it.get("non_mate") is True:
                row["non_mate"] = True
            elif it.get("true_person_id") is not None:
                row["true_person_id"] = int(it["true_person_id"])
            else:
                return jsonify({"error": "Label senza identità vera (true_person_id o non_mate)"}), 400
            rows.append(row)
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Payload label non valido"}), 400
    with (d / "labels.jsonl").open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return jsonify({"saved": len(rows)})


@app.route("/api/validation/<session_id>/metrics", methods=["POST"])
def validation_compute_metrics(session_id: str):
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    return jsonify(compute_and_export(d))


# ── Condition presets + subject registry (validation experiment setup) ──────────

@app.route("/api/validation/presets", methods=["GET", "POST"])
def validation_presets():
    if request.method == "GET":
        return jsonify(load_presets())
    preset = upsert_preset(request.get_json(silent=True) or {})
    return jsonify(preset)


@app.route("/api/validation/presets/<preset_id>", methods=["DELETE"])
def validation_delete_preset(preset_id: str):
    return jsonify({"deleted": delete_preset(preset_id)})


@app.route("/api/validation/presets/<preset_id>/duplicate", methods=["POST"])
def validation_duplicate_preset(preset_id: str):
    copy = duplicate_preset(preset_id)
    if copy is None:
        return jsonify({"error": "Preset non trovato"}), 404
    return jsonify(copy)


@app.route("/api/validation/subjects", methods=["GET", "POST"])
def validation_subjects():
    if request.method == "GET":
        return jsonify(load_subjects())
    return jsonify(save_subjects(request.get_json(silent=True) or {}))


# ── Destination volume (external-disk picker) ──────────────────────────────────

@app.route("/api/storage/drives")
def storage_drives():
    """Currently mounted partitions (mountpoint, filesystem, free space) for the picker."""
    return jsonify(list_drives())


@app.route("/api/storage/validate", methods=["POST"])
def storage_validate():
    """Validate a chosen destination (writable / free space / filesystem) before Start."""
    data = request.get_json(silent=True) or {}
    minutes = data.get("minutes")
    est = estimate_bytes(float(minutes)) if minutes else 0
    return jsonify(validate_target(data.get("path", ""), est))


def _persist_env(key: str, value: str) -> bool:
    """Best-effort: update/insert KEY=value in ${DATA_DIR}/runtime.env (on the external disk) so
    the change survives a restart. Deliberately NOT /app/.env: pydantic re-reads .env at boot,
    which hard-requires python-dotenv and crash-looped the container after a UI switch. runtime.env
    is loaded into os.environ by config.settings._load_runtime_env, best-effort — the boot never
    depends on it."""
    env_path = Path(os.environ.get("DATA_DIR", "data")) / "runtime.env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        out, found = [], False
        for line in lines:
            if line.strip().startswith(f"{key}="):
                out.append(f"{key}={value}"); found = True
            else:
                out.append(line)
        if not found:
            out.append(f"{key}={value}")
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        logger.warning(f"Impossibile scrivere .env: {exc}")
        return False


@app.route("/api/settings/match_threshold", methods=["POST"])
def set_match_threshold():
    """Apply a recognition threshold live (e.g. the EER point from validation) and persist it."""
    data = request.get_json(silent=True) or {}
    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        return jsonify({"error": "Valore non valido"}), 400
    if not (0.0 < value < 2.0):
        return jsonify({"error": "Soglia fuori range (0–2)"}), 400
    if broadcaster.pipeline is not None:
        broadcaster.pipeline._recognizer.threshold = value  # live update
    persisted = _persist_env("MATCH_THRESHOLD", f"{value}")
    logger.info(f"[Settings] MATCH_THRESHOLD = {value} (persistita nel .env: {persisted})")
    return jsonify({"value": value, "persisted": persisted})


@app.route("/api/settings/detection", methods=["GET", "POST"])
def settings_detection():
    """Parametri di portata del rilevatore, tarabili DAL VIVO (nessun riavvio: il warm-up dei
    modelli costa ~9 min sul TX2).

    - `min_face_px`: altezza minima del volto, in px del frame a piena risoluzione.
    - `det_threshold`: confidenza minima SCRFD.
    - `det_input`: [w, h] del canvas del rilevatore. Il frame ci viene inscritto in letterbox,
      quindi questo valore fissa la portata: con 640x640 un 1080p arriva alla rete a 640x360
      (det_scale 0.333) e un volto sotto ~50 px reali non viene proprio visto. I lati vengono
      arrotondati a multipli di 32 (vincolo delle griglie di anchor SCRFD).
    """
    if request.method == "GET":
        return jsonify(get_detection_settings())
    data = request.get_json(silent=True) or {}
    min_face_px = det_threshold = det_input = None
    try:
        if data.get("min_face_px") is not None:
            min_face_px = int(data["min_face_px"])
            if not (0 <= min_face_px <= 400):
                return jsonify({"error": "min_face_px fuori range (0–400)"}), 400
        if data.get("det_threshold") is not None:
            det_threshold = float(data["det_threshold"])
            if not (0.05 <= det_threshold <= 0.95):
                return jsonify({"error": "det_threshold fuori range (0.05–0.95)"}), 400
        if data.get("det_input") is not None:
            w, h = data["det_input"]
            det_input = (snap_det_dim(w), snap_det_dim(h))
    except (TypeError, ValueError):
        return jsonify({"error": "Valori non validi"}), 400
    if min_face_px is None and det_threshold is None and det_input is None:
        return jsonify({"error": "Nessun parametro da applicare"}), 400

    before = get_detection_settings()
    state = apply_detection_settings(min_face_px=min_face_px, det_threshold=det_threshold,
                                     det_input=det_input)
    persisted = True
    if min_face_px is not None:
        persisted &= _persist_env("MIN_FACE_PX", str(state["min_face_px"]))
    if det_threshold is not None:
        persisted &= _persist_env("DET_THRESHOLD", str(state["det_threshold"]))
    if det_input is not None:
        persisted &= _persist_env("DET_INPUT_WIDTH", str(state["det_input"][0]))
        persisted &= _persist_env("DET_INPUT_HEIGHT", str(state["det_input"][1]))
    warning = None
    if det_input is not None and det_input != tuple(before["det_input"]) \
            and get_profile().is_optimized:
        warning = ("Profilo optimized: TensorRT deve costruire il motore per la nuova "
                   "risoluzione: i primi frame saranno lenti, poi resta in cache.")
    logger.info(f"[Settings] detection {state} (persistito: {persisted})")
    return jsonify(dict(state, persisted=persisted, warning=warning))


@app.route("/api/settings/profile", methods=["GET", "POST"])
def settings_profile():
    """Get or set the performance profile (Standard vs Optimized-TX2). On POST: persist
    PERFORMANCE_PROFILE, apply it live, and rebuild the analyzer (new model pack / providers)
    + drop optimized per-camera state — no full restart needed."""
    if request.method == "GET":
        summary = profile_summary()
        with get_session() as session:
            summary["templates_for_profile"] = PersonRepository(session).count_templates(summary["model_pack"])
        return jsonify({"options": [STANDARD, OPTIMIZED], **summary})
    data = request.get_json(silent=True) or {}
    value = (data.get("profile") or "").strip().lower()
    if value not in (STANDARD, OPTIMIZED):
        return jsonify({"error": f"Profilo non valido (usa {STANDARD} o {OPTIMIZED})"}), 400
    get_settings().performance_profile = value  # live update (read by get_profile())
    reset_for_profile_change()                  # rebuild InsightFace with the new pack/providers
    if broadcaster.pipeline is not None:
        broadcaster.pipeline.force_reload()     # clear optimized per-camera caches/trackers
    persisted = _persist_env("PERFORMANCE_PROFILE", value)
    logger.info(f"[Settings] PERFORMANCE_PROFILE = {value} (persistito nel .env: {persisted})")
    return jsonify({"persisted": persisted, **profile_summary()})


@app.route("/healthz")
def healthz():
    """Unauthenticated liveness probe for the Docker HEALTHCHECK (the previous target was
    auth-gated -> permanent 401 -> container flagged unhealthy). No sensitive data."""
    w = get_warmup_state()
    return jsonify({"status": "ok", "warmup_ready": bool(w.get("ready")),
                    "warmup_active": bool(w.get("active"))})


@app.route("/api/warmup")
def warmup_status():
    """Model warm-up progress for the UI bar: {active, phase, pct, eta_s, ready}."""
    return jsonify(get_warmup_state())


@app.route("/api/settings/profile/status")
def profile_status():
    """Semaforo: requested profile vs what the analyzer ACTUALLY loaded (pack/precision/providers
    read from the live build, not from settings). match=True only when loaded == requested and
    warm-up is done — a silent misconfig (e.g. OPT_MODEL_PACK pointing both profiles at buffalo_l)
    shows up as an explicit mismatch instead of passing unnoticed."""
    requested = profile_summary()
    loaded = get_loaded_info()
    warmup = get_warmup_state()
    match = bool(
        warmup.get("ready")
        and loaded.get("model_pack") == requested.get("model_pack")
        and loaded.get("precision") == requested.get("precision")
        and loaded.get("profile") == requested.get("profile")
    )
    state = "switching" if (warmup.get("active") or not warmup.get("ready")) else \
            ("ok" if match else "mismatch")
    return jsonify({"state": state, "match": match, "requested": requested,
                    "loaded": loaded, "warmup": warmup})


_CONTAINER = os.environ.get("FACEID_CONTAINER", "faceid")


def _delayed(fn, delay=0.4):
    def _run():
        time.sleep(delay)
        fn()
    threading.Thread(target=_run, daemon=True).start()


@app.route("/api/system/power", methods=["POST"])
def system_power():
    """Restart or shut down the container from the UI. Gated by `_guard_request` (WEB_PASSWORD when
    set + cross-site reject); requires `confirm: true`. restart = clean exit (Docker
    `--restart unless-stopped` brings it back). shutdown = `docker stop` via the mounted Docker
    socket (needs the socket mount + docker CLI in the image; 501 if unavailable → restart still works)."""
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip().lower()
    if not data.get("confirm"):
        return jsonify({"error": "Conferma richiesta"}), 400
    if action == "restart":
        logger.warning("[System] Riavvio richiesto dalla UI — uscita pulita (Docker riavvia)")
        _delayed(lambda: os._exit(0))
        return jsonify({"action": "restart", "message": "Riavvio in corso…"})
    if action == "shutdown":
        try:
            proc = subprocess.run(["docker", "stop", _CONTAINER], timeout=30,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            return jsonify({"error": "docker CLI non disponibile nell'immagine (vedi run.sh / "
                            "Dockerfile.jetson per il mount del socket)"}), 501
        except Exception as exc:
            return jsonify({"error": f"docker stop fallito: {exc}"}), 500
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip() or "docker stop fallito"
            return jsonify({"error": err}), 502  # socket non montato / container non trovato
        logger.warning("[System] Spegnimento richiesto dalla UI — docker stop %s", _CONTAINER)
        return jsonify({"action": "shutdown", "message": "Container fermato."})
    return jsonify({"error": "action deve essere 'restart' o 'shutdown'"}), 400


def _segments_index(d: Path, camera_id: str) -> list:
    f = d / "video" / "segments.json"
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get(str(camera_id), [])
        except Exception:
            return []
    return []


def _locate_frame(d: Path, camera_id: str, index: int):
    """Map a GLOBAL frame index → (segment file Path, local index). Falls back to a legacy
    single-file recording. Returns (None, 0) when nothing matches."""
    for s in _segments_index(d, camera_id):
        if s["first_frame"] <= index <= s["last_frame"]:
            return d / "video" / s["file"], index - s["first_frame"]
    legacy = d / "video" / f"cam_{camera_id}.mp4"  # pre-segmentation sessions
    if legacy.is_file():
        return legacy, index
    segs = _segments_index(d, camera_id)
    if segs:  # out-of-range index → clamp to the first segment
        return d / "video" / segs[0]["file"], 0
    return None, 0


@app.route("/api/validation/<session_id>/video/<camera_id>")
def validation_video(session_id: str, camera_id: str):
    """Download a recorded segment (?seg=N; default the first). Segments avoid FAT32's 4 GB limit."""
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    segs = _segments_index(d, camera_id)
    if segs:
        seg = request.args.get("seg", type=int)
        chosen = next((s for s in segs if s["segment"] == seg), segs[0]) if seg is not None else segs[0]
        fname = chosen["file"]
    else:
        fname = f"cam_{camera_id}.mp4"  # legacy single-file
    if not (d / "video" / fname).is_file():
        return jsonify({"error": "Video non trovato"}), 404
    return send_from_directory(d / "video", fname, mimetype="video/mp4", conditional=True)


# Frame-by-frame review: the recorded mp4 uses a codec browsers can't decode in
# <video>, so the review UI requests decoded JPEG frames by index (exact sync with
# the detection log). One cached VideoCapture per file makes sequential playback fast.
_frame_caps: dict = {}
_frame_caps_lock = threading.Lock()


def _read_frame(path: str, index: int):
    with _frame_caps_lock:
        c = _frame_caps.get(path)
        if c is None:
            c = {"cap": cv2.VideoCapture(path), "next": -1}
            _frame_caps[path] = c
        cap = c["cap"]
        if c["next"] != index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        c["next"] = index + 1 if ok else -1
        return frame if ok else None


@app.route("/api/validation/<session_id>/frame/<camera_id>/<int:index>")
def validation_frame(session_id: str, camera_id: str, index: int):
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    path, local = _locate_frame(d, camera_id, max(0, index))  # global index → segment + local
    if path is None or not path.is_file():
        return jsonify({"error": "Video non trovato"}), 404
    frame = _read_frame(str(path), max(0, local))
    if frame is None:
        return jsonify({"error": "Frame non disponibile"}), 404
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return jsonify({"error": "Encoding fallito"}), 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ── Floor-plan map + calibration (Phase 2) ────────────────────────────────────

def _map_file() -> "Optional[Path]":
    if MAPS_DIR.is_dir():
        for f in sorted(MAPS_DIR.glob("floorplan.*")):
            return f
    return None


@app.route("/api/map", methods=["GET"])
def get_map():
    f = _map_file()
    if f is None:
        return jsonify({"exists": False, "url": None})
    # ?v=mtime busts the browser cache when the floor plan is replaced
    return jsonify({"exists": True, "url": f"/static/maps/{f.name}?v={int(f.stat().st_mtime)}"})


@app.route("/api/map", methods=["POST"])
def upload_map():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Nessun file caricato"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in _MAP_EXTS:
        return jsonify({"error": "Formato non supportato (png/jpg/webp)"}), 415
    # Verify the payload really is a decodable image (the file lands in /static)
    payload = file.read()
    if cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) is None:
        return jsonify({"error": "Il file non è un'immagine valida"}), 415
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    for old in MAPS_DIR.glob("floorplan.*"):
        old.unlink()
    dest = MAPS_DIR / f"floorplan{ext}"
    dest.write_bytes(payload)
    logger.info(f"[Mappa] Planimetria caricata: {dest.name}")
    return jsonify({"exists": True, "url": f"/static/maps/{dest.name}"})


@app.route("/api/cameras/<path:camera_id>/snapshot")
def camera_snapshot(camera_id: str):
    frame = broadcaster.get_raw_frame(camera_id)
    if frame is None:
        return jsonify({"error": "Nessun frame disponibile dalla camera"}), 503
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return jsonify({"error": "Encoding fallito"}), 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/calibration")
def list_calibration():
    with get_session() as session:
        calibs = CalibrationRepository(session).get_all()
        out = []
        for c in calibs:
            raw = json.loads(c.points_json)
            if isinstance(raw, list):  # legacy homography rows
                raw = {"mode": "homography", "points": raw}
            out.append({
                "camera": c.camera_id,
                "mode": raw.get("mode", "homography"),
                "raw": raw,
                "updated_at": c.updated_at.isoformat(),
            })
        return jsonify(out)


@app.route("/api/calibration/<camera_id>/sample", methods=["POST"])
def calibration_sample(camera_id: str):
    """Detect the (single) face in the current frame and return its horizontal
    position + pixel height — one polar-calibration reference sample."""
    frame = broadcaster.get_raw_frame(camera_id)
    if frame is None:
        return jsonify({"error": "Nessun frame disponibile dalla camera"}), 503

    faces = detect_and_encode(frame)
    if not faces:
        return jsonify({"error": "Nessun volto rilevato. Mettiti nel punto scelto e riprova."}), 422
    if len(faces) > 1:
        return jsonify({"error": "Più volti nell'inquadratura. Assicurati di essere solo."}), 422

    (top, right, bottom, left), _ = faces[0]
    frame_w = frame.shape[1]
    return jsonify({
        "x_norm": float((left + right) / 2.0 / frame_w),
        "h_px": float(bottom - top),
    })


@app.route("/api/calibration/<camera_id>", methods=["POST"])
def save_calibration(camera_id: str):
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "homography")

    try:
        if mode == "polar":
            cam = data.get("camera") or {}
            samples = data.get("samples") or []
            params = solve_polar_calibration((cam["mx"], cam["my"]), samples)
            raw = {"camera": cam, "samples": samples}
            detail = f"{len(samples)} campioni"
        else:
            points = data.get("points") or []
            if len(points) < 4:
                return jsonify({"error": "Servono almeno 4 punti di calibrazione"}), 400
            params = {"H": compute_homography(points).tolist()}
            raw = {"points": points}
            detail = f"{len(points)} punti"
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": f"Calibrazione non calcolabile: {exc}"}), 400

    with get_session() as session:
        CalibrationRepository(session).upsert(camera_id, mode, raw, params)

    if broadcaster.pipeline:
        broadcaster.pipeline.force_reload()

    logger.success(f"[Calibrazione] Camera {camera_id} calibrata in modalità {mode} ({detail}).")
    return jsonify({"message": f"Calibrazione {mode} camera {camera_id} salvata ({detail})."})


@app.route("/api/positions/map")
def positions_map():
    try:
        minutes = max(1, min(1440, int(request.args.get("minutes", 5))))
    except (TypeError, ValueError):
        minutes = 5
    since = datetime.utcnow() - timedelta(minutes=minutes)
    with get_session() as session:
        return jsonify(PersonRepository(session).latest_world_positions(since))


@app.route("/calibrate")
def calibrate_page():
    return send_from_directory(STATIC_DIR, "calibrate.html")
