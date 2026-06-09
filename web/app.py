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
from core.geometry import compute_homography
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
        return jsonify([
            {
                "camera": c.camera_id,
                "points": json.loads(c.points_json),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in calibs
        ])


@app.route("/api/calibration/<camera_id>", methods=["POST"])
def save_calibration(camera_id: str):
    data = request.get_json(silent=True) or {}
    points = data.get("points") or []
    if len(points) < 4:
        return jsonify({"error": "Servono almeno 4 punti di calibrazione"}), 400
    try:
        homography = compute_homography(points)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": f"Omografia non calcolabile: {exc}"}), 400

    with get_session() as session:
        CalibrationRepository(session).upsert(camera_id, points, homography)

    if broadcaster.pipeline:
        broadcaster.pipeline.force_reload()

    logger.success(f"[Calibrazione] Camera {camera_id} calibrata con {len(points)} punti.")
    return jsonify({"message": f"Calibrazione camera {camera_id} salvata ({len(points)} punti)."})


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
