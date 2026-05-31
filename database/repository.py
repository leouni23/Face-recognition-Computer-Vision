from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import FaceTemplate, Person, PositionLog, RecognitionEvent
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
        encrypted = encrypt_embedding(encoding, _settings.biometric_secret_key)
        template = FaceTemplate(person_id=person.id, encoding_encrypted=encrypted)
        self.session.add(template)
        return template

    def load_all_templates(self) -> List[Tuple[int, str, np.ndarray]]:
        """Return (person_id, name, encoding) for every active, consenting person."""
        rows = (
            self.session.query(Person, FaceTemplate)
            .join(FaceTemplate)
            .filter(Person.active == True, Person.consent_given == True)
            .all()
        )
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
    ) -> None:
        """Persist a position point (bounding-box centre + size) for an identified subject.

        Coordinates arrive as numpy ints (face.bbox.astype(int)); cast to native
        Python types so the sqlite3 driver stores them as INTEGER, not BLOB.
        """
        top, right, bottom, left = (int(v) for v in location)
        frame_h, frame_w = (
            (int(frame_shape[0]), int(frame_shape[1])) if frame_shape is not None else (None, None)
        )
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

    def delete_person(self, name: str) -> int:
        count = (
            self.session.query(Person).filter(Person.name == name).delete()
        )
        return count
