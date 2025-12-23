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
    # If Admin, show all. If User, show only their hospital's logs + general logs (hospital_id is None)
    query = db.query(models.ActivityLog)
    
    # REFINED LOGIC:
    # 1. If user belongs to a hospital (regardless of role), show ONLY that hospital's logs.
    # 2. If user has NO hospital (Global User):
    #    - If ADMIN: Show ALL logs.
    #    - If not ADMIN: Show NOTHING (Strict Isolation).
    
    if current_user.hospital_id:
        query = query.filter(models.ActivityLog.hospital_id == current_user.hospital_id)
    elif current_user.role == models.UserRole.ADMIN:
        # Super Admin sees everything
        pass
    else:
        # Global non-admin sees nothing
        query = query.filter(models.ActivityLog.hospital_id == -1)
    
    return query.order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()
