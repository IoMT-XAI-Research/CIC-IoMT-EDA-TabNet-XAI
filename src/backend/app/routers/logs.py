from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database, dependencies

router = APIRouter(
    prefix="/activity-logs",
    tags=["logs"]
)

# HELPER FUNCTION
def create_activity_log(db: Session, title: str, description: str, log_type: str, hospital_id: int = None):
    log = models.ActivityLog(
        title=title,
        description=description,
        log_type=log_type,
        hospital_id=hospital_id
    )
    db.add(log)
    db.commit()

@router.get("/", response_model=List[schemas.ActivityLogResponse])
def get_activity_logs(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user),
    limit: int = 50
):
    # --- DEBUG BLOĞU BAŞLANGIÇ ---
    print(f"🛑 DEBUG: İstek Yapan: {current_user.email}, Role: {current_user.role}, HospitalID: {current_user.hospital_id}")
    # --- DEBUG BLOĞU BİTİŞ ---

    # If Admin, show all. If User, show only their hospital's logs + general logs (hospital_id is None)
    query = db.query(models.ActivityLog)
    
    # REFRESH USER to get latest hospital_id
    db.refresh(current_user)

    # 1. Hospital Assignment TRUMPS Admin Role.
    if current_user.hospital_id is not None:
        print(f"✅ DEBUG: Kullanıcının hastanesi var ({current_user.hospital_id}). Loglar filtreleniyor.")
        query = query.filter(models.ActivityLog.hospital_id == current_user.hospital_id)
    # 2. If no hospital ID (New Admin), show NOTHING.
    else:
        print("⛔ DEBUG: Kullanıcının hastanesi YOK. BOŞ LİSTE DÖNÜLÜYOR.")
        return []
    
    return query.order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()
