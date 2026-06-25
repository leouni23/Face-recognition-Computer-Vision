"""Runtime camera lifecycle owner.

Single source of truth for which cameras run. Seeds the `camera_configs` registry from
settings on first run, owns one CameraStream + worker thread per enabled camera, and hot-reloads
(start/stop threads) on registry changes without restarting the app. Peer of `broadcaster` /
`validation_manager`. Unreachable cameras surface as status, never crash the app.
"""
import queue
import threading
from typing import Dict, List, Optional

from loguru import logger

from config.settings import get_settings
from core.camera import CameraStream, probe_source, validate_source
from database.repository import CameraRepository
from database.session import get_session
from web.broadcaster import broadcaster

_settings = get_settings()


class CameraManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cams: Dict[str, CameraStream] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._worker_stops: Dict[str, threading.Event] = {}
        self.result_queues: Dict[str, "queue.Queue"] = {}  # for --local display (main.py)
        self._pipeline = None
        self._web_mode = False
        self._show_local = False
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start_all(self, pipeline, web_mode: bool, show_local: bool) -> None:
        self._pipeline = pipeline
        self._web_mode = web_mode
        self._show_local = show_local
        self._started = True
        self._seed_if_empty()
        self.reload()

    def _seed_if_empty(self) -> None:
        with get_session() as session:
            repo = CameraRepository(session)
            if repo.count() > 0:
                return
            for src in _settings.cameras:
                name = "Webcam USB 0" if src == "0" else src
                repo.add(cam_id=src, name=name, source=src, enabled=True)
            logger.info(f"[Camere] Registry inizializzato da CAMERA_SOURCES: {_settings.cameras}")

    def _enabled_rows(self) -> List[dict]:
        with get_session() as session:
            return [
                {"cam_id": r.cam_id, "source": r.source, "resolution": r.resolution}
                for r in CameraRepository(session).all() if r.enabled
            ]

    def reload(self) -> None:
        """Diff the enabled registry against running threads; start/stop to converge."""
        if not self._started:
            return
        with self._lock:
            want = {r["cam_id"]: r for r in self._enabled_rows()}
            running = set(self._cams)
            # Stop removed/disabled, or restart changed source/resolution.
            for cam_id in list(running):
                r = want.get(cam_id)
                cam = self._cams[cam_id]
                if r is None or r["source"] != cam._source or r["resolution"] != cam._resolution:
                    self._stop_one(cam_id)
            # Start new ones.
            for cam_id, r in want.items():
                if cam_id not in self._cams:
                    self._start_one(cam_id, r["source"], r["resolution"])

    def _start_one(self, cam_id: str, source: str, resolution: Optional[str]) -> None:
        cam = CameraStream(source, camera_id=cam_id, resolution=resolution).start()
        stop_ev = threading.Event()
        self._cams[cam_id] = cam
        self._worker_stops[cam_id] = stop_ev
        if self._show_local:
            self.result_queues[cam_id] = queue.Queue(maxsize=2)
        t = threading.Thread(target=self._worker, args=(cam, stop_ev),
                             daemon=True, name=f"worker-{cam_id}")
        self._workers[cam_id] = t
        t.start()
        logger.info(f"[Camere] avviata {cam_id!r} ({source})")

    def _stop_one(self, cam_id: str) -> None:
        ev = self._worker_stops.pop(cam_id, None)
        if ev is not None:
            ev.set()
        cam = self._cams.pop(cam_id, None)
        if cam is not None:
            cam.stop()
        self._workers.pop(cam_id, None)
        self.result_queues.pop(cam_id, None)
        logger.info(f"[Camere] fermata {cam_id!r}")

    def stop_all(self) -> None:
        with self._lock:
            for cam_id in list(self._cams):
                self._stop_one(cam_id)

    # ── worker (was main._worker) ──────────────────────────────────────────────
    def _worker(self, cam: CameraStream, stop_ev: threading.Event) -> None:
        from ui.display import annotate_frame  # lazy: avoid core<->ui import cycle
        pipeline = self._pipeline
        rq = self.result_queues.get(cam.camera_id)
        while not stop_ev.is_set():
            frame = cam.read(timeout=0.1)
            if frame is None:
                continue
            try:
                results = pipeline.process_frame(frame, cam.camera_id)
            except Exception:
                logger.exception(f"Errore nel frame (camera {cam.camera_id}); continuo")
                continue
            if self._web_mode:
                broadcaster.push_raw_frame(cam.camera_id, frame)
                if broadcaster.has_viewers(cam.camera_id):
                    broadcaster.push_frame(cam.camera_id, annotate_frame(frame, results))
                for _, pid, name, conf in results:
                    if pid is not None:
                        broadcaster.push_event(cam.camera_id, name, conf)
            if rq is not None:
                if rq.full():
                    try:
                        rq.get_nowait()
                    except queue.Empty:
                        pass
                rq.put((cam.camera_id, frame, results))

    # ── CRUD (persist then hot-reload) ─────────────────────────────────────────
    def list_cameras(self) -> List[dict]:
        with get_session() as session:
            rows = CameraRepository(session).all()
            out = []
            for r in rows:
                live = self._cams[r.cam_id].status() if r.cam_id in self._cams else {}
                out.append({
                    "cam_id": r.cam_id, "name": r.name, "source": r.source,
                    "enabled": r.enabled, "resolution": r.resolution,
                    "status": live.get("status", "disabled" if not r.enabled else "stopped"),
                    "last_error": live.get("last_error"),
                })
            return out

    def add(self, name: str, source: str, enabled: bool = True,
            resolution: Optional[str] = None) -> dict:
        err = validate_source(source)
        if err:
            raise ValueError(err)
        source = source.strip()
        with get_session() as session:
            repo = CameraRepository(session)
            cam_id = self._unique_cam_id(repo, source)
            repo.add(cam_id=cam_id, name=(name or source).strip(), source=source,
                     enabled=enabled, resolution=resolution)
        self.reload()
        return {"cam_id": cam_id}

    def update(self, cam_id: str, **fields) -> bool:
        if fields.get("source"):
            err = validate_source(fields["source"])
            if err:
                raise ValueError(err)
        with get_session() as session:
            ok = CameraRepository(session).update(cam_id, **fields) is not None
        if ok:
            self.reload()
        return ok

    def active_ids(self) -> List[str]:
        """Enabled camera ids from the registry (the cameras that drive both modes)."""
        with get_session() as session:
            return [r.cam_id for r in CameraRepository(session).all() if r.enabled]

    def remove(self, cam_id: str) -> bool:
        with get_session() as session:
            ok = CameraRepository(session).delete(cam_id)
        if ok:
            self.reload()
        return ok

    @staticmethod
    def _unique_cam_id(repo: CameraRepository, source: str) -> str:
        # USB index keeps its index as the (stable) cam_id; URLs get cam<N>.
        base = source if source.isdigit() else None
        if base is not None and repo.get(base) is None:
            return base
        n = 1
        while True:
            cid = f"cam{n}"
            if repo.get(cid) is None:
                return cid
            n += 1


camera_manager = CameraManager()
