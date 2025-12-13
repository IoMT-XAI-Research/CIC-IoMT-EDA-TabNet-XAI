from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import models, schemas, database, auth
from typing import List

router = APIRouter(
    prefix="/admin",
    tags=["admin"]
)

@router.post("/seed", status_code=status.HTTP_201_CREATED)
def seed_data(db: Session = Depends(database.get_db)):
    """
    Seeds the database with test data:
    1. Admin User (admin@doruk.com)
    2. Hospitals (Doruk Nilufer, Doruk Yildirim)
    3. Devices for each
    """
    
    # 1. Create User
    email = "admin@doruk.com"
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        hashed_password = auth.get_password_hash("password123")
        user = models.User(
            email=email,
            password_hash=hashed_password,
            role=models.UserRole.ADMIN
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # 2. Create Hospitals (Owned by this user)
    hospitals_data = [
        {"name": "Doruk Nilufer", "code": "DORUK-NIL"},
        {"name": "Doruk Yildirim", "code": "DORUK-YIL"}
    ]
    
    created_hospitals = []
    
    for h_data in hospitals_data:
        hospital = db.query(models.Hospital).filter(models.Hospital.unique_code == h_data["code"]).first()
        if not hospital:
            hospital = models.Hospital(
                name=h_data["name"],
                unique_code=h_data["code"],
                owner_id=user.id
            )
            db.add(hospital)
            db.commit()
            db.refresh(hospital)
        created_hospitals.append(hospital)
        
    # 3. Create Devices
    files_per_hospital = {
        "DORUK-NIL": [
            ("MRI Scanner - Room 101", "192.168.10.5"),
            ("Infusion Pump - ICU", "192.168.10.12")
        ],
        "DORUK-YIL": [
            ("X-Ray Machine - ER", "192.168.20.8"),
            ("Patient Monitor - Room 205", "192.168.20.15")
        ]
    }
    
    for hospital in created_hospitals:
        devices_to_add = files_per_hospital.get(hospital.unique_code, [])
        for d_name, d_ip in devices_to_add:
            exists = db.query(models.Device).filter(models.Device.hospital_id == hospital.id, models.Device.name == d_name).first()
            if not exists:
                device = models.Device(
                    name=d_name,
                    ip_address=d_ip,
                    status=models.DeviceStatus.SAFE,
                    hospital_id=hospital.id
                )
                db.add(device)
    
    db.commit()
    
    return {"message": "Database seeded successfully", "user": email}

@router.delete("/database", status_code=status.HTTP_200_OK)
def reset_database(db: Session = Depends(database.get_db)):
    """
    Resets the database:
    1. Drops all tables
    2. Recreates all tables
    3. Seeds initial data
    """
    # Drop all tables
    database.Base.metadata.drop_all(bind=database.engine)
    
    # Re-create all tables
    database.Base.metadata.create_all(bind=database.engine)
    
    # Seed data
    return seed_data(db)
