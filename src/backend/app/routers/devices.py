from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from app import models, schemas, dependencies   # 🔹 EKLENDİ
from app.database import get_db                 # 🔹 BUNU KULLANACAĞIZ

router = APIRouter(
    prefix="/devices",
    tags=["devices"]
)


@router.get("/", response_model=List[schemas.DeviceResponse])
def read_devices(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),  # 🔹 dependencies.get_db DEĞİL
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # Enforce hospital isolation
    devices = (
        db.query(models.Device)
        .filter(models.Device.hospital_id == current_user.hospital_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return devices


@router.post("/{device_id}/isolate", response_model=schemas.DeviceResponse)
def isolate_device(
    device_id: int,
    db: Session = Depends(get_db),  # 🔹 Burada da aynı
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # Check permissions
    if current_user.role != models.UserRole.TECH_STAFF:
        raise HTTPException(status_code=403, detail="Not authorized to isolate devices")

    # Enforce hospital isolation
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

    # Create Isolation Event
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


# --- Status update endpoint ---
class StatusUpdate(BaseModel):
    status: str


@router.put("/{device_id}/status")
def update_device_status(
    device_id: int,
    status_update: StatusUpdate,
    db: Session = Depends(get_db),  # 🔹 Aynı get_db
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
