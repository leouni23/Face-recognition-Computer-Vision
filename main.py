#!/usr/bin/env python
"""Face ID — real-time face recognition, GDPR-compliant.

Usage:
  python main.py                  # local OpenCV windows
  python main.py --web            # web UI on http://localhost:8000
  python main.py --web --local    # both web UI and local windows
"""
import argparse
import queue
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings
from core.camera import CameraStream
from core.pipeline import FaceIdPipeline, RecognitionResult
from database.models import Base
from database.session import engine, get_session
from privacy.retention import run_retention
from ui.display import Display, annotate_frame
from web.broadcaster import broadcaster

_settings = get_settings()

_QueueItem = Tuple[str, np.ndarray, List[RecognitionResult]]


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=_settings.log_level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )


def _worker(
    cam: CameraStream,
    pipeline: FaceIdPipeline,
    result_queue: Optional["queue.Queue[_QueueItem]"],
    stop_event: threading.Event,
    web_mode: bool,
) -> None:
    while not stop_event.is_set():
        frame = cam.read(timeout=0.1)
        if frame is None:
            continue

        results = pipeline.process_frame(frame, cam.camera_id)

        if web_mode:
            broadcaster.push_raw_frame(cam.camera_id, frame)
            if broadcaster.has_viewers(cam.camera_id):
                annotated = annotate_frame(frame, results)
                broadcaster.push_frame(cam.camera_id, annotated)
            for _, pid, name, conf in results:
                if pid is not None:
                    broadcaster.push_event(cam.camera_id, name, conf)

        if result_queue is not None:
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put((cam.camera_id, frame, results))


def _start_flask(host: str, port: int) -> None:
    from web.app import app as flask_app
    logger.info(f"Web UI: http://{host}:{port}")
    flask_app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Face ID")
    parser.add_argument("--web", action="store_true", help="Avvia la Web UI (Flask)")
    parser.add_argument("--local", action="store_true", help="Mostra finestre OpenCV locali")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host web (default: 127.0.0.1 — usa 0.0.0.0 per esporre sulla LAN, con WEB_PASSWORD impostata)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Porta web (default: 8000)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Esegui il benchmark prestazioni e termina (report in data/benchmarks/)")
    parser.add_argument("--validation", nargs="?", const="", default=None, metavar="NOME",
                        help="Avvia una sessione di validazione all'avvio (registra video + detections)")
    args = parser.parse_args()

    show_local = args.local or not args.web

    _setup_logging()

    if args.benchmark:
        from scripts.benchmark import run_benchmark
        run_benchmark()
        return

    if args.web and args.host not in ("127.0.0.1", "localhost", "::1") and not _settings.web_password:
        logger.warning(
            f"Web UI esposta su {args.host} SENZA autenticazione: chiunque sulla rete può vedere "
            "le camere e gestire i dati biometrici. Imposta WEB_PASSWORD nel file .env"
        )

    Base.metadata.create_all(engine)

    with get_session() as session:
        deleted = run_retention(session, _settings.data_retention_days)
    if deleted:
        logger.info(f"Retention GDPR: {deleted} persona/e eliminate per scadenza dati")

    cameras = [CameraStream(src).start() for src in _settings.cameras]
    pipeline = FaceIdPipeline()
    broadcaster.pipeline = pipeline

    if args.validation is not None:
        from core.validation import validation_manager
        from database.models import Person
        with get_session() as session:
            enrolled = [
                {"id": p.id, "name": p.name}
                for p in session.query(Person)
                .filter(Person.active == True, Person.consent_given == True).all()
            ]
        info = validation_manager.start(args.validation, enrolled, _settings.match_threshold)
        logger.info(f"Validazione: sessione '{info['session_id']}' — registrazione attiva")

    result_queues: Dict[str, "queue.Queue[_QueueItem]"] = (
        {cam.camera_id: queue.Queue(maxsize=2) for cam in cameras}
        if show_local else {}
    )

    stop_event = threading.Event()
    workers = [
        threading.Thread(
            target=_worker,
            args=(cam, pipeline, result_queues.get(cam.camera_id), stop_event, args.web),
            daemon=True,
            name=f"worker-{cam.camera_id}",
        )
        for cam in cameras
    ]
    for w in workers:
        w.start()

    if args.web:
        threading.Thread(
            target=_start_flask, args=(args.host, args.port), daemon=True
        ).start()

    mode = []
    if show_local:
        mode.append("finestre locali")
    if args.web:
        mode.append(f"web http://{args.host}:{args.port}")
    logger.info(f"Face ID avviato — {len(cameras)} camera/e — {', '.join(mode)}. Premi Q o Ctrl-C per uscire.")

    display = Display() if show_local else None

    try:
        while not stop_event.is_set():
            if show_local and display:
                for cam_id, q in result_queues.items():
                    try:
                        _, frame, results = q.get_nowait()
                        if not display.show(frame, results, window=f"Face ID — Camera {cam_id}"):
                            stop_event.set()
                    except queue.Empty:
                        pass
            else:
                stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Interruzione ricevuta")
        stop_event.set()
    finally:
        from core.validation import validation_manager
        if validation_manager.is_active():
            validation_manager.stop()
        for cam in cameras:
            cam.stop()
        Display.close_all()
        logger.info("Face ID terminato.")


if __name__ == "__main__":
    main()
