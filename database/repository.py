import json
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import (
    CameraCalibration, CameraConfig, FaceTemplate, Person, PositionLog, RecognitionEvent,
)
from privacy.crypto import decrypt_embedding, encrypt_embedding

_settings = get_settings()


class PersonRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str) -> Person:
        person = Person(name=name)
        self.session.add(person)
        self.session.flush()
        return person

    def get_by_name(self, name: str) -> Optional[Person]:
        return (
            self.session.query(Person)
            .filter(Person.name == name, Person.active == True)
            .first()
        )

    def give_consent(self, person: Person) -> None:
        person.consent_given = True
        person.consent_date = datetime.utcnow()

    def add_template(self, person: Person, encoding: np.ndarray) -> FaceTemplate:
        # Tag the embedding with the recognition model that produced it (active profile pack),
        # so matching never mixes incompatible embedding spaces across profiles.
        from core.profile import get_profile
        encrypted = encrypt_embedding(encoding, _settings.biometric_secret_key)
        template = FaceTemplate(person_id=person.id, encoding_encrypted=encrypted,
                                model_pack=get_profile().model_pack)
        self.session.add(template)
        return template

    def delete_all_templates(self) -> int:
        """Drop every face template (keeps persons + consent). One-off cleanup for the polluted,
        untagged DB before per-model tagging takes effect."""
        n = self.session.query(FaceTemplate).delete()
        return n

    def count_templates(self, model_pack: str) -> int:
        return (
            self.session.query(FaceTemplate)
            .join(Person)
            .filter(Person.active == True, Person.consent_given == True,
                    FaceTemplate.model_pack == model_pack)
            .count()
        )

    def load_all_templates(self, model_pack: Optional[str] = None) -> List[Tuple[int, str, np.ndarray]]:
        """Return (person_id, name, encoding) for active, consenting people. When `model_pack`
        is given, only templates from that recognition model (embeddings from other packs are in
        an incompatible space and must not be matched against)."""
        q = (
            self.session.query(Person, FaceTemplate)
            .join(FaceTemplate)
            .filter(Person.active == True, Person.consent_given == True)
        )
        if model_pack is not None:
            q = q.filter(FaceTemplate.model_pack == model_pack)
        rows = q.all()
        result = []
        for person, template in rows:
            try:
                enc = decrypt_embedding(
                    template.encoding_encrypted, _settings.biometric_secret_key
                )
                result.append((person.id, person.name, enc))
            except Exception:
                # Skip templates that cannot be decrypted (e.g. after key rotation)
                pass
        return result

    def update_last_seen(self, person_id: int) -> None:
        self.session.query(Person).filter(Person.id == person_id).update(
            {"last_seen": datetime.utcnow()}
        )

    def log_event(
        self, camera_id: str, person_id: Optional[int], confidence: float
    ) -> None:
        event = RecognitionEvent(
            camera_id=camera_id, person_id=person_id, confidence=confidence
        )
        self.session.add(event)

    def log_position(
        self,
        camera_id: str,
        person_id: int,
        location: Tuple[int, int, int, int],
        confidence: float,
        frame_shape: Optional[Tuple[int, ...]] = None,
        world: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Persist a position point (bounding-box centre + size) for an identified subject.

        Coordinates arrive as numpy ints (face.bbox.astype(int)); cast to native
        Python types so the sqlite3 driver stores them as INTEGER, not BLOB.
        `world` is the floor-plan projection (Phase 2), stored when the camera is calibrated.
        """
        top, right, bottom, left = (int(v) for v in location)
        frame_h, frame_w = (
            (int(frame_shape[0]), int(frame_shape[1])) if frame_shape is not None else (None, None)
        )
        world_x, world_y = (float(world[0]), float(world[1])) if world else (None, None)
        self.session.add(
            PositionLog(
                person_id=person_id,
                camera_id=camera_id,
                bbox_cx=(left + right) // 2,
                bbox_cy=(top + bottom) // 2,
                bbox_w=right - left,
                bbox_h=bottom - top,
                frame_w=frame_w,
                frame_h=frame_h,
                confidence=float(confidence),
                world_x=world_x,
                world_y=world_y,
            )
        )

    def get_trajectory(
        self,
        person_id: int,
        since: Optional[datetime] = None,
        limit: int = 5000,
    ) -> List[PositionLog]:
        """Return a subject's saved positions, oldest first, optionally since a timestamp."""
        query = self.session.query(PositionLog).filter(PositionLog.person_id == person_id)
        if since is not None:
            query = query.filter(PositionLog.timestamp >= since)
        return query.order_by(PositionLog.timestamp).limit(limit).all()

    def latest_world_positions(self, since: datetime) -> List[dict]:
        """Most recent map position per person (only calibrated points, since `since`)."""
        rows = (
            self.session.query(PositionLog, Person.name)
            .join(Person, Person.id == PositionLog.person_id)
            .filter(PositionLog.world_x.isnot(None), PositionLog.timestamp >= since)
            .order_by(PositionLog.timestamp.desc())
            .all()
        )
        latest: dict = {}
        for pos, name in rows:
            if pos.person_id not in latest:
                latest[pos.person_id] = {
                    "person_id": pos.person_id,
                    "name": name,
                    "world_x": pos.world_x,
                    "world_y": pos.world_y,
                    "camera": pos.camera_id,
                    "time": pos.timestamp.isoformat(),
                }
        return list(latest.values())

    def delete_person(self, name: str) -> int:
        """GDPR Art. 17 — elimina la persona e, via cascade ORM, template e posizioni."""
        persons = self.session.query(Person).filter(Person.name == name).all()
        for person in persons:
            self.session.delete(person)
        return len(persons)


class CalibrationRepository:
    """Stores per-camera calibration. Two modes share the same columns via a JSON
    envelope: `points_json` holds the raw user input (re-editable), `homography_json`
    holds the computed projector params — {"mode":"homography","H":[...]} or
    {"mode":"polar","cam":[x,y],"heading":...,"scale":...,"k":...}. Legacy rows
    (plain 3x3 list) are read as homography."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(self) -> List[CameraCalibration]:
        return self.session.query(CameraCalibration).all()

    def get(self, camera_id: str) -> Optional[CameraCalibration]:
        return (
            self.session.query(CameraCalibration)
            .filter(CameraCalibration.camera_id == camera_id)
            .first()
        )

    def load_projectors(self) -> List[dict]:
        """Return parsed [{camera_id, mode, params}] for every calibrated camera."""
        result = []
        for row in self.get_all():
            data = json.loads(row.homography_json)
            if isinstance(data, list):  # legacy plain 3x3 matrix
                data = {"mode": "homography", "H": data}
            result.append({"camera_id": row.camera_id, "mode": data.get("mode", "homography"), "params": data})
        return result

    def upsert(self, camera_id: str, mode: str, raw: dict, params: dict) -> None:
        points_json = json.dumps({"mode": mode, **raw})
        params_json = json.dumps({"mode": mode, **params})
        existing = self.get(camera_id)
        if existing:
            existing.points_json = points_json
            existing.homography_json = params_json
            existing.updated_at = datetime.utcnow()
        else:
            self.session.add(
                CameraCalibration(
                    camera_id=camera_id,
                    points_json=points_json,
                    homography_json=params_json,
                )
            )


class CameraRepository:
    """Runtime camera registry CRUD (see models.CameraConfig)."""

    def __init__(self, session: Session):
        self.session = session

    def all(self) -> List[CameraConfig]:
        return self.session.query(CameraConfig).order_by(CameraConfig.id).all()

    def get(self, cam_id: str) -> Optional[CameraConfig]:
        return self.session.query(CameraConfig).filter(CameraConfig.cam_id == cam_id).first()

    def count(self) -> int:
        return self.session.query(CameraConfig).count()

    def add(self, cam_id: str, name: str, source: str, enabled: bool = True,
            resolution: Optional[str] = None, notes: Optional[str] = None) -> CameraConfig:
        row = CameraConfig(cam_id=cam_id, name=name, source=source, enabled=enabled,
                           resolution=resolution or None, notes=notes or None)
        self.session.add(row)
        self.session.flush()
        return row

    def update(self, cam_id: str, **fields) -> Optional[CameraConfig]:
        row = self.get(cam_id)
        if row is None:
            return None
        for k in ("name", "source", "enabled", "resolution", "notes"):
            if k in fields and fields[k] is not None:
                setattr(row, k, fields[k])
        return row

    def delete(self, cam_id: str) -> bool:
        row = self.get(cam_id)
        if row is None:
            return False
        self.session.delete(row)
        return True
