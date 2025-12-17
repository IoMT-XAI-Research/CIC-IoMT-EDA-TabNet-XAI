from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas, database, dependencies
from . import logs
from .logs import create_activity_log

router = APIRouter(
    prefix="/hospitals",
    tags=["hospitals"]
)

@router.put("/{hospital_id}", response_model=schemas.HospitalResponse)
def update_hospital(
    hospital_id: int,
    hospital_update: schemas.HospitalCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    if current_user.role != models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only Admins can update hospitals")

    db_hospital = db.query(models.Hospital).filter(models.Hospital.id == hospital_id).first()
    if not db_hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    
    # Check if admin owns this hospital
    if db_hospital.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this hospital")

    db_hospital.name = hospital_update.name
    db_hospital.unique_code = hospital_update.unique_code
    db.commit()
    db.refresh(db_hospital)
    logs.log_activity(db, f"Hospital Updated: {db_hospital.name} by {current_user.email}") 
    return db_hospital

@router.post("/", response_model=schemas.HospitalResponse, status_code=status.HTTP_201_CREATED)
def create_hospital(
    hospital: schemas.HospitalCreate, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # Check if hospital unique_code already exists
    db_hospital = db.query(models.Hospital).filter(models.Hospital.unique_code == hospital.unique_code).first()
    if db_hospital:
        raise HTTPException(status_code=400, detail="Hospital with this unique code already exists")
    
    # Check if name exists
    db_hospital_name = db.query(models.Hospital).filter(models.Hospital.name == hospital.name).first()
    if db_hospital_name:
        raise HTTPException(status_code=400, detail="Hospital with this name already exists")
    
    new_hospital = models.Hospital(
        name=hospital.name,
        unique_code=hospital.unique_code,
        owner_id=current_user.id # Assign owner
    )
    db.add(new_hospital)
    db.commit()
    db.refresh(new_hospital)
    
    # LOGGING (Safe)
    try:
        create_activity_log(db, "Hastane Eklendi", f"{new_hospital.name} sisteme kaydedildi.", "SUCCESS")
    except Exception as e:
        print(f"Logging Error: {e}")

    return new_hospital

@router.get("/", response_model=List[schemas.HospitalResponse])
def read_hospitals(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    if current_user.role == models.UserRole.ADMIN:
        # Admin sees hospitals they own
        hospitals = db.query(models.Hospital).filter(models.Hospital.owner_id == current_user.id).offset(skip).limit(limit).all()
    else:
        # Staff sees only their assigned hospital
        if current_user.hospital_id:
            hospitals = db.query(models.Hospital).filter(models.Hospital.id == current_user.hospital_id).all()
        else:
            hospitals = []
    
    return hospitals

@router.delete("/{hospital_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_hospital(
    hospital_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    if current_user.role != models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only Admins can delete hospitals")

    db_hospital = db.query(models.Hospital).filter(models.Hospital.id == hospital_id).first()
    if not db_hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    
    # Check ownership
    if db_hospital.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this hospital")

    db.delete(db_hospital)
    db.commit()
    try:
        create_activity_log(db, "Hastane Silindi", f"{db_hospital.name} silindi.", "WARNING")
    except Exception as e:
        print(f"Logging Error: {e}")
    return None
