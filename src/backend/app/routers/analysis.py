# src/backend/app/routers/analysis.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas, dependencies
from ..services import ml_service

router = APIRouter(
    prefix="/devices",
    tags=["analysis"],
)

@router.get("/{device_id}/analysis/latest", response_model=schemas.AnalysisResponse)
def get_analysis(
    device_id: int,
    db: Session = Depends(dependencies.get_db),
    current_user: models.User = Depends(dependencies.get_current_user),
):
    # 1. Cihaz gerçekten bu kullanıcının hastanesine mi ait, kontrol et
    device = (
        db.query(models.Device)
        .filter(
            models.Device.id == device_id,
            models.Device.hospital_id == current_user.hospital_id,
        )
        .first()
    )

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # 2. Artık MOCK DEĞİL → AIEngine + TabNet + SHAP pipeline'ına gidiyor
    return ml_service.get_model_analysis_for_device(db, device)

