from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
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
    positions = relationship("PositionLog", back_populates="person", cascade="all, delete-orphan")


class FaceTemplate(Base):
    __tablename__ = "face_templates"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    # AES-128-CBC + HMAC (Fernet) encrypted 512-d float32 face embedding — never stored in plaintext
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


class PositionLog(Base):
    """Time-stamped position of an identified subject within a camera frame.

    Cross-camera tracking is implicit: the same `person_id` (ArcFace identity)
    links a subject's points across every camera. Pixel coordinates are stored
    now (Phase 1); `world_x`/`world_y` are reserved for the floor-plan homography
    projection (Phase 2) and stay NULL until per-camera calibration exists.
    """
    __tablename__ = "position_logs"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Pixel-space position inside the camera frame (bounding-box centre + size)
    bbox_cx = Column(Integer, nullable=False)
    bbox_cy = Column(Integer, nullable=False)
    bbox_w = Column(Integer, nullable=False)
    bbox_h = Column(Integer, nullable=False)
    # Frame dimensions → enables normalisation and future homography
    frame_w = Column(Integer, nullable=True)
    frame_h = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)

    # Real-world map coordinates (Phase 2 — filled once per-camera homography exists)
    world_x = Column(Float, nullable=True)
    world_y = Column(Float, nullable=True)

    person = relationship("Person", back_populates="positions")


class CameraCalibration(Base):
    """Per-camera homography mapping frame pixels to a shared floor-plan plane.

    `points_json` stores the pixel<->map correspondences (so calibration can be
    re-edited), `homography_json` the computed 3x3 matrix used at runtime.
    """
    __tablename__ = "camera_calibrations"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(64), nullable=False, unique=True, index=True)
    points_json = Column(Text, nullable=False)
    homography_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
