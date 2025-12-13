from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas, database, dependencies

router = APIRouter(
    prefix="/hospitals",
    tags=["hospitals"]
)

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
    return new_hospital

@router.get("/", response_model=List[schemas.HospitalResponse])
def read_hospitals(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # Only return hospitals owned by the user
    hospitals = db.query(models.Hospital).filter(models.Hospital.owner_id == current_user.id).offset(skip).limit(limit).all()
    return hospitals
