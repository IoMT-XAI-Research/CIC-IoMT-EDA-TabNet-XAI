from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas, dependencies
from ..services import ml_service

router = APIRouter(
    prefix="/devices",
    tags=["analysis"]
)

@router.get("/{device_id}/analysis/latest", response_model=schemas.AnalysisResponse)
def get_analysis(device_id: int, db: Session = Depends(dependencies.get_db), current_user: models.User = Depends(dependencies.get_current_user)):
    # Enforce hospital isolation
    device = db.query(models.Device).filter(models.Device.id == device_id, models.Device.hospital_id == current_user.hospital_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return ml_service.get_mock_analysis(device_id)
