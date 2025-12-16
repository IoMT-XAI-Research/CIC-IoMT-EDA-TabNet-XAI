from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas, database, dependencies

router = APIRouter(
    prefix="/hospitals",
    tags=["hospitals"]
)

from . import logs # Import logs router helper

@router.post("/", response_model=schemas.HospitalResponse, status_code=status.HTTP_201_CREATED)
def create_hospital(
    hospital: schemas.HospitalCreate, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # RBAC: Only Admin can create hospitals
    if current_user.role != models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only Admins can create hospitals")

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
    
    # Log Activity
    logs.log_activity(db, f"Hospital Created: {new_hospital.name} ({new_hospital.unique_code}) by {current_user.email}")
    
    return new_hospital

@router.delete("/{hospital_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_hospital(
    hospital_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # RBAC: Only Admin can delete
    if current_user.role != models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only Admins can delete hospitals")
        
    db_hospital = db.query(models.Hospital).filter(models.Hospital.id == hospital_id).first()
    if not db_hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
        
    db.delete(db_hospital)
    db.commit()
    
    logs.log_activity(db, f"Hospital Deleted: {db_hospital.name} by {current_user.email}")
    return None

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
