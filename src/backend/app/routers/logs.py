from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database, dependencies

router = APIRouter(
    prefix="/logs",
    tags=["logs"]
)

# Helper function to log activity (to be imported by other routers)
def log_activity(db: Session, description: str):
    new_log = models.ActivityLog(description=description)
    db.add(new_log)
    db.commit()

@router.get("/", response_model=List[schemas.ActivityLogResponse])
def read_logs(
    limit: int = 5, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    return db.query(models.ActivityLog).order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()
