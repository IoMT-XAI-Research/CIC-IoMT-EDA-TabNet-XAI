from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
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
    
    # FIX: Re-fetch the user within the current session to ensure persistence and get latest data
    current_user = db.query(models.User).filter(models.User.id == current_user.id).first()

    # STRICT ISOLATION LOGIC
    if current_user.hospital_id is not None:
        # User is assigned to a specific hospital -> SEE ONLY THAT HOSPITAL'S LOGS
        query = query.filter(models.ActivityLog.hospital_id == current_user.hospital_id)
    else:
        # User is NOT assigned to a hospital (Global Admin or Unassigned)
        # MUST NOT see logs from other hospitals. Only see System Logs (where hospital_id is NULL).
        query = query.filter(models.ActivityLog.hospital_id.is_(None))
    
    return query.order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()

@router.get("/stats/traffic")
def get_log_traffic_stats(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # Count logs created in the last 5 seconds
    cutoff_time = datetime.utcnow() - timedelta(seconds=5)
    
    query = db.query(models.ActivityLog).filter(models.ActivityLog.timestamp >= cutoff_time)
    
    # Apply Isolation Logic to ensure stats reflect what the user CAN see
    if current_user.hospital_id is not None:
         query = query.filter(models.ActivityLog.hospital_id == current_user.hospital_id)
    else:
         # Unassigned users see System Logs only
         query = query.filter(models.ActivityLog.hospital_id.is_(None))

    count = query.count()
    
    # Threshold for "Attack" simulation
    status = "high" if count > 10 else "normal"
    
    return {"count": count, "status": status}
