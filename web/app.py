import json
import queue
import time
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory, stream_with_context
from loguru import logger

from database.models import Person
from database.repository import PersonRepository
from database.session import get_session
from web.broadcaster import broadcaster

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))


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
