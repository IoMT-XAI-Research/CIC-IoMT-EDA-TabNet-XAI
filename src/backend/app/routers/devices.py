# src/backend/app/routers/devices.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from .. import models, schemas, dependencies   # 🔹 modeller + şemalar + current_user
from src.backend.app.database import get_db                 # 🔹 ortak get_db

from .logs import create_activity_log

router = APIRouter(
    prefix="/devices",
    tags=["devices"]
)

# --- 1) Cihaz oluşturma endpoint'i (YENİ) ---
@router.post("/", response_model=schemas.DeviceResponse)
def create_device(
    payload: schemas.DeviceCreate,
    db: Session = Depends(get_db)
):
    # Lookup Hospital by unique_code
    hospital = db.query(models.Hospital).filter(models.Hospital.unique_code == payload.hospital_unique_code).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found with provided code")

    # Create new device linked to this hospital
    device = models.Device(
        name=payload.name,
        ip_address=payload.ip_address,
        room_number=payload.room_number, # Added
        status=models.DeviceStatus.SAFE,
        last_risk_score=0.0,
        hospital_id=hospital.id,
    )

    db.add(device)
    db.commit()
    db.refresh(device)
    
    # LOGGING
    create_activity_log(db, "Yeni Cihaz", f"{device.name} eklendi.", "INFO", hospital.id)

    return device


# --- 2) Cihazları listeleme ---
@router.get("/", response_model=List[schemas.DeviceResponse])
def read_devices(
    hospital_unique_code: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # Lookup Hospital first
    hospital = db.query(models.Hospital).filter(models.Hospital.unique_code == hospital_unique_code).first()
    if not hospital:
        return [] 

    # SECURITY CHECK: Ensure the requesting user OWNS this hospital
    if hospital.owner_id != current_user.id:
        # Optionally checking if user is "employed" there (hospital_id) could be another case,
        # but the requirement says "Group structure" / "Owns", so we enforce ownership.
        # If TECH_STAFF needs access, we might check `current_user.hospital_id == hospital.id` too.
        # For now, let's allow OWNER or EMPLOYEE.
        if current_user.hospital_id != hospital.id:
             raise HTTPException(status_code=403, detail="Not authorized to view devices for this hospital")

    try:
        devices = (
            db.query(models.Device)
            .filter(models.Device.hospital_id == hospital.id)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return devices
    except Exception as e:
        print(f"CRITICAL DB ERROR in /devices/: {e}")
        # Return empty list or raise 500? Use 500 to signal retry or check logs
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


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
    
    create_activity_log(db, "Cihaz İzole Edildi", f"{device.name} izole edildi.", "DANGER", device.hospital_id)

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

@router.delete("/{device_id}", status_code=204)
def delete_device(
    device_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # 1. Fetch the device
    device = db.query(models.Device).filter(models.Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # 2. Authorization (Only Admin or Owner/Tech of that hospital)
    # For simplicity in this fix, ensure user belongs to the same hospital or is Admin
    if current_user.role != models.UserRole.ADMIN:
         if device.hospital_id != current_user.hospital_id:
             raise HTTPException(status_code=403, detail="Not authorized to delete this device")

    # 3. Delete
    hospital_id = device.hospital_id
    device_name = device.name
    db.delete(device)
    db.commit()
    
    create_activity_log(db, "Cihaz Silindi", f"{device_name} silindi.", "DANGER", hospital_id)
    return None
