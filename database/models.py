from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    consent_given = Column(Boolean, nullable=False, default=False)
    consent_date = Column(DateTime, nullable=True)
    enrolled_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=True)
    active = Column(Boolean, nullable=False, default=True)

    templates = relationship("FaceTemplate", back_populates="person", cascade="all, delete-orphan")
    events = relationship("RecognitionEvent", back_populates="person")


class FaceTemplate(Base):
    __tablename__ = "face_templates"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    # AES-128-CBC + HMAC (Fernet) encrypted 128-d float32 face embedding — never stored in plaintext
    encoding_encrypted = Column(LargeBinary, nullable=False)
    enrolled_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    person = relationship("Person", back_populates="templates")


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(64), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    person = relationship("Person", back_populates="events")
