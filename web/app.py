import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from loguru import logger

from core.detector import detect_and_encode
from database.models import Person
from database.repository import PersonRepository
from database.session import get_session
from web.broadcaster import broadcaster

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))

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
        while True:
            jpeg = broadcaster.get_frame(camera_id)
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.04)   # ~25 FPS cap

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
    required = max(1, min(20, int(data.get("samples", 5))))
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
