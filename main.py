#!/usr/bin/env python
"""Face ID — real-time face recognition, GDPR-compliant.

Usage:
  python main.py                  # local OpenCV windows
  python main.py --web            # web UI on http://localhost:8000
  python main.py --web --local    # both web UI and local windows
"""
import argparse
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings
from core.camera_manager import camera_manager
from core.pipeline import FaceIdPipeline, RecognitionResult
from database.models import Base
from database.session import engine, get_session, migrate_schema
from privacy.retention import run_retention
from ui.display import Display
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

    # Bootstrap difensivo: models/engines/validation DEVONO esistere (il symlink
    # /root/.insightface -> /data/models altrimenti pende e il warm-up crasha su disco vuoto).
    # /data non scrivibile = errore fatale esplicito — MAI fallback silenzioso su eMMC.
    data_root = Path(_settings.data_dir)
    try:
        for sub in ("models", "engines", "validation"):
            (data_root / sub).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(f"DATA_DIR {data_root} non scrivibile ({exc}): monta il disco esterno "
                     "e riavvia. Nessun fallback su storage interno.")
        sys.exit(1)
    if not os.access(str(data_root), os.W_OK):
        logger.error(f"DATA_DIR {data_root} non scrivibile: monta il disco esterno e riavvia.")
        sys.exit(1)

    Base.metadata.create_all(engine)
    migrate_schema()  # idempotent: adds FaceTemplate.model_pack to an existing DB

    with get_session() as session:
        deleted = run_retention(session, _settings.data_retention_days)
    if deleted:
        logger.info(f"Retention GDPR: {deleted} persona/e eliminate per scadenza dati")

    # Log the EFFECTIVE profile knobs at boot so a misconfig (e.g. OPT_MODEL_PACK=buffalo_l on
    # both profiles) is visible in the container logs, not only in the dashboard semaforo.
    logger.info(
        "Config profilo: PERFORMANCE_PROFILE={} | OPT_MODEL_PACK={} OPT_PRECISION={} "
        "OPT_DET={}x{} OPT_FRAME_SKIP={} OPT_TRACKER={} OPT_BATCH_EMBED={}".format(
            _settings.performance_profile, _settings.opt_model_pack, _settings.opt_precision,
            _settings.opt_det_width, _settings.opt_det_height, _settings.opt_frame_skip,
            _settings.opt_tracker, _settings.opt_batch_embed))

    pipeline = FaceIdPipeline()
    broadcaster.pipeline = pipeline

    from core.detector import warmup as _warmup_models
    threading.Thread(target=_warmup_models, daemon=True, name="model-warmup").start()

    from core.notifier import build_notifier, set_notifier
    notifier = build_notifier(_settings)
    set_notifier(notifier)
    notifier.start()
    if notifier.enabled:
        logger.info("Notifiche Telegram attive (alert soggetti sconosciuti)")

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

    # Camera lifecycle (registry-driven, hot-reloadable) is owned by the manager.
    camera_manager.start_all(pipeline, web_mode=args.web, show_local=show_local)
    result_queues = camera_manager.result_queues

    stop_event = threading.Event()
    if args.web:
        threading.Thread(
            target=_start_flask, args=(args.host, args.port), daemon=True
        ).start()

    mode = []
    if show_local:
        mode.append("finestre locali")
    if args.web:
        mode.append(f"web http://{args.host}:{args.port}")
    logger.info(f"Face ID avviato — {len(result_queues) or 'N'} camera/e attive — {', '.join(mode)}. Premi Q o Ctrl-C per uscire.")

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
        camera_manager.stop_all()
        Display.close_all()
        logger.info("Face ID terminato.")


if __name__ == "__main__":
    main()
