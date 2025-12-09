from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import schemas, models, database
from ..services.ai_engine import ai_engine
from typing import Dict, Any

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)

@router.post("/traffic", response_model=schemas.AnalysisResponse)
def predict_traffic(request: schemas.TrafficPredictionRequest, db: Session = Depends(database.get_db)):
    # 1. Get Prediction from AI Engine
    analysis = ai_engine.predict(request.features)
    
    # 2. Update Device Status in DB
    device = db.query(models.Device).filter(models.Device.id == request.device_id).first()
    if device:
        # Update status
        if analysis.device_status == "ATTACK":
            device.status = models.DeviceStatus.ATTACK
        else:
            device.status = models.DeviceStatus.SAFE
        
        # Update risk score
        device.last_risk_score = analysis.risk_score
        
        db.commit()
    else:
        # If device not found, we might still want to return the analysis, but log warning
        print(f"[WARN] Device ID {request.device_id} not found in database.")
    
    return analysis
