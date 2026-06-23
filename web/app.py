import hmac
import json
import queue
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from loguru import logger

from config.settings import get_settings
from core.detector import detect_and_encode
from core.geometry import compute_homography, solve_polar_calibration
from core.validation import validation_manager, validation_root
from core.validation_metrics import compute_and_export
from core.validation_presets import (
    delete_preset, duplicate_preset, load_presets, load_subjects, save_subjects, upsert_preset,
)
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
_enroll_session: dict = {}  # {"name": str, "embeddings": list[np.ndarray], "required": int}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ── MJPEG stream ──────────────────────────────────────────────────────────────

@app.route("/stream/<camera_id>")
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
    with _enroll_lock:
        _enroll_session.clear()
        _enroll_session.update({"name": name, "embeddings": [], "required": required})
    return jsonify({"message": f"Sessione avviata per '{name}'", "required": required})


@app.route("/api/enroll/capture", methods=["POST"])
def enroll_capture():
    with _enroll_lock:
        if not _enroll_session:
            return jsonify({"error": "Nessuna sessione attiva. Chiama /api/enroll/start prima."}), 400
        name = _enroll_session["name"]
        required = _enroll_session["required"]
        collected = len(_enroll_session["embeddings"])

    # grab the latest raw frame from the first available camera
    camera_id = (broadcaster.camera_ids or ["0"])[0]
    frame = broadcaster.get_raw_frame(camera_id)
    if frame is None:
        return jsonify({"error": "Nessun frame disponibile dalla camera"}), 503

    faces = detect_and_encode(frame)
    if not faces:
        return jsonify({"error": "Nessun volto rilevato. Avvicinati alla camera."}), 422
    if len(faces) > 1:
        return jsonify({"error": "Più volti nell'inquadratura. Assicurati di essere solo."}), 422

    _, embedding = faces[0]
    with _enroll_lock:
        # session may have been cancelled while the frame was being processed
        if not _enroll_session:
            return jsonify({"error": "Sessione annullata"}), 409
        _enroll_session["embeddings"].append(embedding)
        collected = len(_enroll_session["embeddings"])
        done = collected >= required

    return jsonify({"collected": collected, "required": required, "done": done})


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
    return jsonify(broadcaster.camera_ids)


@app.route("/api/persons")
def list_persons():
    with get_session() as session:
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
    return jsonify({"perf": perf, "resources": get_resource_metrics()})


# ── Validation / test mode ────────────────────────────────────────────────────

def _session_dir(session_id: str):
    """Resolve a session folder safely (reject path traversal)."""
    root = validation_root().resolve()
    target = (root / session_id).resolve()
    if target.is_dir() and root in target.parents:
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


@app.route("/api/validation/start", methods=["POST"])
def validation_start():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if name and not _NAME_RE.match(name):
        return jsonify({"error": "Nome sessione non valido"}), 400
    with get_session() as session:
        enrolled = [
            {"id": p.id, "name": p.name}
            for p in session.query(Person)
            .filter(Person.active == True, Person.consent_given == True)
            .order_by(Person.name)
            .all()
        ]
    try:
        info = validation_manager.start(
            name, enrolled, _settings.match_threshold,
            session_type=data.get("session_type", "crossing"),
            destination=(data.get("destination") or None),
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({**info, "enrolled": enrolled, "threshold": _settings.match_threshold})


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


@app.route("/api/validation/stop", methods=["POST"])
def validation_stop():
    return jsonify(validation_manager.stop())


@app.route("/api/validation/status")
def validation_status():
    return jsonify(validation_manager.status())


@app.route("/api/validation/sessions")
def validation_sessions():
    root = validation_root()
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            manifest = d / "session.json"
            if manifest.is_file():
                try:
                    out.append(json.loads(manifest.read_text(encoding="utf-8")))
                except Exception:
                    pass
    return jsonify(out)


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
        det["truth"] = labels.get(
            (str(det["camera_id"]), int(det["frame_index"]), int(det["face_id"]))
        )
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


def _persist_env(key: str, value: str) -> bool:
    """Best-effort: update/insert KEY=value in the project .env so the change survives restart."""
    env_path = Path(__file__).parent.parent / ".env"
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


@app.route("/api/validation/<session_id>/video/<camera_id>")
def validation_video(session_id: str, camera_id: str):
    d = _session_dir(session_id)
    if d is None:
        return jsonify({"error": "Sessione non trovata"}), 404
    fname = f"cam_{camera_id}.mp4"
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
    path = d / "video" / f"cam_{camera_id}.mp4"
    if not path.is_file():
        return jsonify({"error": "Video non trovato"}), 404
    frame = _read_frame(str(path), max(0, index))
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


@app.route("/api/cameras/<camera_id>/snapshot")
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
