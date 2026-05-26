import asyncio
import json
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from database.models import Person
from database.repository import PersonRepository
from database.session import get_session
from web.broadcaster import broadcaster

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Face ID", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def _startup() -> None:
    broadcaster.set_loop(asyncio.get_running_loop())
    logger.info("Web UI disponibile — broadcaster loop registrato")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ── Video streams (MJPEG) ─────────────────────────────────────────────────────

@app.get("/stream/{camera_id}")
async def video_stream(camera_id: str) -> StreamingResponse:
    async def generate():
        while True:
            jpeg = broadcaster.get_frame(camera_id)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.04)   # ~25 FPS cap

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Server-Sent Events ────────────────────────────────────────────────────────

@app.get("/api/events")
async def events_stream(request: Request) -> StreamingResponse:
    q = broadcaster.subscribe()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = q.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.QueueEmpty:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.5)
        finally:
            broadcaster.unsubscribe(q)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/cameras")
def list_cameras() -> List[str]:
    return broadcaster.camera_ids


@app.get("/api/persons")
def list_persons():
    with get_session() as session:
        persons = (
            session.query(Person)
            .filter(Person.active == True, Person.consent_given == True)
            .order_by(Person.name)
            .all()
        )
        return [
            {
                "id": p.id,
                "name": p.name,
                "enrolled_at": p.enrolled_at.isoformat(),
                "last_seen": p.last_seen.isoformat() if p.last_seen else None,
            }
            for p in persons
        ]


@app.delete("/api/persons/{name}")
def delete_person(name: str):
    with get_session() as session:
        count = PersonRepository(session).delete_person(name)
    if not count:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' non trovata")
    logger.info(f"[API] Dati biometrici di '{name}' eliminati (GDPR Art. 17)")
    return {"message": f"Persona '{name}' eliminata ({count} record)."}
