from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import FaceTemplate, Person, RecognitionEvent
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

    def delete_person(self, name: str) -> int:
        count = (
            self.session.query(Person).filter(Person.name == name).delete()
        )
        return count
