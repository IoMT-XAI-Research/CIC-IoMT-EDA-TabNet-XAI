from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import List, Dict
from sqlalchemy.orm import Session
from .. import dependencies, models, auth
from jose import jwt, JWTError

router = APIRouter(
    prefix="/ws",
    tags=["websocket"]
)

class ConnectionManager:
    def __init__(self):
        # Map hospital_id to list of websockets
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, hospital_id: int):
        await websocket.accept()
        if hospital_id not in self.active_connections:
            self.active_connections[hospital_id] = []
        self.active_connections[hospital_id].append(websocket)

    def disconnect(self, websocket: WebSocket, hospital_id: int):
        if hospital_id in self.active_connections:
            self.active_connections[hospital_id].remove(websocket)

    async def broadcast_to_hospital(self, message: dict, hospital_id: int):
        if hospital_id in self.active_connections:
            for connection in self.active_connections[hospital_id]:
                await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/alerts")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        hospital_id: int = payload.get("hospital_id")
        if hospital_id is None:
            await websocket.close(code=1008)
            return
    except JWTError:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, hospital_id)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket, hospital_id)

# Internal endpoint to trigger alerts (Simulation)
@router.post("/internal/report-attack")
async def report_attack(
    payload: dict,
    db: Session = Depends(dependencies.get_db)  # Need DB for logging
):
    # Payload structure:
    # {
    #   "prediction": { "is_attack": bool, "probability": float, ... },
    #   "flow_details": { "timestamp": float, ... },
    #   "explanation": [ ... ],
    #   "hospital_id": int,  <-- REQUIRED for isolation
    #   "device_id": int     <-- REQUIRED for identification
    # }
    
    hospital_id = payload.get("hospital_id")
    
    # STRICT ISOLATION: Broadcast ONLY to the specific hospital channel
    if hospital_id and isinstance(hospital_id, int):
        await manager.broadcast_to_hospital(payload, hospital_id)
    else:
        # If no hospital_id or invalid, strictly do NOT broadcast globally.
        pass
        
    # Check for Attack and Log
    prediction = payload.get("prediction", {})
    if prediction.get("is_attack"):
        # Create DANGER log
        from .logs import create_activity_log
        
        device_id = payload.get("device_id")
        device_name = f"Device {device_id}" if device_id else "Unknown Device"
        prob = prediction.get("probability", 0.0)
        
        create_activity_log(
            db, 
            "KRİTİK SALDIRI TESPİTİ", 
            f"{device_name} üzerinde saldırı tespit edildi! (Güven: {prob:.2f})", 
            "DANGER", 
            hospital_id
        )
        
    return {"status": "alert processed"}
