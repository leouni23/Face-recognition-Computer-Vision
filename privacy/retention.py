from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import or_
from sqlalchemy.orm import Session

from database.models import Person


def run_retention(session: Session, days: int) -> int:
    """Hard-delete persons whose last activity is older than `days` days (GDPR Art. 5 data minimization)."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    expired = (
        session.query(Person)
        .filter(
            Person.active == True,
            Person.enrolled_at < cutoff,
            or_(Person.last_seen == None, Person.last_seen < cutoff),
        )
        .all()
    )
    for person in expired:
        logger.info(f"Retention: eliminazione '{person.name}' (id={person.id}, ultimo accesso={person.last_seen})")
        session.delete(person)
    return len(expired)
