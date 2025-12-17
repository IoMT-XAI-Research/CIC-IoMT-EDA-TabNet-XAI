from sqlalchemy import Column, Integer, String, Float, ForeignKey, Enum, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base
import enum
from datetime import datetime

class UserRole(str, enum.Enum):
    TECH_STAFF = "TECH_STAFF"
    MANAGER = "MANAGER"
    ADMIN = "ADMIN"

class DeviceStatus(str, enum.Enum):
    SAFE = "SAFE"
    ATTACK = "ATTACK"
    ISOLATED = "ISOLATED"

class EventType(str, enum.Enum):
    CRITICAL_ALERT = "CRITICAL_ALERT"
    ISOLATION = "ISOLATION"
    INFO = "INFO"

class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    unique_code = Column(String, unique=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id")) # New: Owner

    owner = relationship("User", back_populates="owned_hospitals", foreign_keys=[owner_id])
    users = relationship("User", back_populates="hospital", foreign_keys="User.hospital_id")
    devices = relationship("Device", back_populates="hospital", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="hospital")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(Enum(UserRole), default=UserRole.TECH_STAFF)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)

    hospital = relationship("Hospital", back_populates="users", foreign_keys=[hospital_id])
    owned_hospitals = relationship("Hospital", back_populates="owner", foreign_keys="Hospital.owner_id", cascade="all, delete-orphan") # New linkage

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    ip_address = Column(String)
    status = Column(Enum(DeviceStatus), default=DeviceStatus.SAFE)
    last_risk_score = Column(Float, default=0.0)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"))

    hospital = relationship("Hospital", back_populates="devices")
    analyses = relationship("Analysis", back_populates="device")
    events = relationship("Event", back_populates="device")

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    type = Column(Enum(EventType))
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"))

    hospital = relationship("Hospital", back_populates="events")
    device = relationship("Device", back_populates="events")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    prediction_score = Column(Float)
    xai_data = Column(JSON) # Stores SHAP values and other XAI data

    device = relationship("Device", back_populates="analyses")

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

