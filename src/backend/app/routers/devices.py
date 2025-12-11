# src/backend/app/routers/devices.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from app import models, schemas, dependencies   # 🔹 modeller + şemalar + current_user
from app.database import get_db                 # 🔹 ortak get_db

router = APIRouter(
    prefix="/devices",
    tags=["devices"]
)

# --- 1) Cihaz oluşturma endpoint'i (YENİ) ---
@router.post("/", response_model=schemas.DeviceResponse)
def create_device(
    payload: schemas.DeviceCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # Sadece TECH_STAFF yeni cihaz ekleyebilsin
    if current_user.role != models.UserRole.TECH_STAFF:
        raise HTTPException(status_code=403, detail="Not authorized to create devices")

    # Yeni cihazı, kullanıcının kendi hastanesine bağla
    device = models.Device(
        name=payload.name,
        ip_address=payload.ip_address,
        status=models.DeviceStatus.SAFE,   # Enum kullanıyoruz
        last_risk_score=0.0,
        hospital_id=current_user.hospital_id,
    )

    db.add(device)
    db.commit()
    db.refresh(device)

    return device


# --- 2) Cihazları listeleme ---
@router.get("/", response_model=List[schemas.DeviceResponse])
def read_devices(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # Kullanıcının hastanesine ait cihazlar
    devices = (
        db.query(models.Device)
        .filter(models.Device.hospital_id == current_user.hospital_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return devices


# --- 3) Cihaz izole etme ---
@router.post("/{device_id}/isolate", response_model=schemas.DeviceResponse)
def isolate_device(
    device_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # Yetki kontrolü
    if current_user.role != models.UserRole.TECH_STAFF:
        raise HTTPException(status_code=403, detail="Not authorized to isolate devices")

    # Aynı hastaneye ait cihaz mı?
    device = (
        db.query(models.Device)
        .filter(
            models.Device.id == device_id,
            models.Device.hospital_id == current_user.hospital_id,
        )
        .first()
    )
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    device.status = models.DeviceStatus.ISOLATED

    # Event kaydı (isteğe bağlı ama güzel durur)
    event = models.Event(
        device_id=device.id,
        type=models.EventType.ISOLATION,
        message=f"Device {device.name} isolated by {current_user.email}",
        hospital_id=current_user.hospital_id,
    )
    db.add(event)
    db.commit()
    db.refresh(device)

    return device


# --- 4) Cihaz durumunu manuel güncelleme ---
class StatusUpdate(BaseModel):
    status: str


@router.put("/{device_id}/status")
def update_device_status(
    device_id: int,
    status_update: StatusUpdate,
    db: Session = Depends(get_db),
):
    device = db.query(models.Device).filter(models.Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Cihaz bulunamadı")

    device.status = status_update.status
    db.commit()
    db.refresh(device)
    return {
        "message": f"Cihaz {device.name} durumu '{device.status}' olarak güncellendi!",
        "device": device,
    }
# --- BİTİŞ ---
