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
    
    # DEBUG LOGGING (Nuclear Isolation Check)
    print(f"[LOGS] User {current_user.id} ({current_user.email}) Role: {current_user.role}, Hospital ID: {current_user.hospital_id}")

    # NUCLEAR ISOLATION LOGIC:
    # 1. Hospital Assignment TRUMPS Admin Role.
    #    If a user has a hospital_id (e.g., 2), they MUST be restricted to that hospital.
    #    It does NOT matter if they are an ADMIN. They are an Admin of *that* hospital.
    if current_user.hospital_id is not None:
        query = query.filter(models.ActivityLog.hospital_id == current_user.hospital_id)
        
    # 2. Only if hospital_id is None (Global User) do we check Role.
    elif current_user.role == models.UserRole.ADMIN:
        # Super Admin sees everything
        pass
        
    # 3. Everyone else (Global non-admin?) sees NOTHING.
    else:
        # Strict Isolation
        query = query.filter(models.ActivityLog.hospital_id == -1)
    
    return query.order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()
