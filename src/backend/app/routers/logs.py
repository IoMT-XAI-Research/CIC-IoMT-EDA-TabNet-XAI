from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List
from datetime import datetime

from .. import models, schemas, database, dependencies

router = APIRouter(
    prefix="/logs",
    tags=["logs"]
)

def log_activity(db: Session, description: str):
    """
    Helper function to create an activity log entry.
    """
    new_log = models.ActivityLog(
        description=description,
        timestamp=datetime.utcnow()
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return new_log

@router.get("/", response_model=List[schemas.ActivityLogResponse])
def read_logs(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # Logs are generally viewable by all authenticated users, 
    # but we could restrict if needed.
    logs = db.query(models.ActivityLog).order_by(desc(models.ActivityLog.timestamp)).offset(skip).limit(limit).all()
    return logs
