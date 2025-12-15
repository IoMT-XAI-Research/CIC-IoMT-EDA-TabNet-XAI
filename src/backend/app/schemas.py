from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .models import UserRole, DeviceStatus, EventType
from datetime import datetime

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None
    hospital_id: Optional[int] = None
    role: Optional[UserRole] = None

# User Schemas
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    hospital_code: Optional[str] = None

class UserResponse(UserBase):
    id: int
    role: UserRole
    hospital_id: Optional[int] = None

    class Config:
        orm_mode = True

# Hospital Schemas
class HospitalBase(BaseModel):
    name: str
    unique_code: str

class HospitalCreate(HospitalBase):
    pass

class HospitalResponse(HospitalBase):
    id: int

    class Config:
        orm_mode = True

# Device Schemas
class DeviceBase(BaseModel):
    name: str
    ip_address: str

class DeviceCreate(DeviceBase):
    hospital_unique_code: str

class DeviceResponse(DeviceBase):
    id: int
    status: DeviceStatus
    last_risk_score: float
    hospital_id: int

    class Config:
        orm_mode = True

# Analysis Schemas
class XAIForcePlot(BaseModel):
    base_value: float
    final_value: float
    features: List[Dict[str, Any]]

class FeatureImportance(BaseModel):
    name: str
    percentage: float
    value_desc: str

class AnalysisResponse(BaseModel):
    device_status: str
    risk_score: float
    summary_text: str
    xai_force_plot: XAIForcePlot
    feature_importance_list: List[FeatureImportance]

class TrafficPredictionRequest(BaseModel):
    hospital_id: int
    device_id: int
    features: Dict[str, Any]

# Event Schemas
class EventBase(BaseModel):
    type: EventType
    message: str

class EventCreate(EventBase):
    device_id: int

class EventResponse(EventBase):
    id: int
    device_id: int
    timestamp: datetime
    hospital_id: int

    class Config:
        orm_mode = True
