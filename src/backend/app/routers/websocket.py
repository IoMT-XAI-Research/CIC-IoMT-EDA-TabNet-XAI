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
        # List of Global Admin websockets (receive EVERYTHING)
        self.admin_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, hospital_id: int):
        # WebSocket is already accepted in endpoint
        if hospital_id not in self.active_connections:
            self.active_connections[hospital_id] = []
        self.active_connections[hospital_id].append(websocket)
        
    async def connect_admin(self, websocket: WebSocket):
        # WebSocket is already accepted in endpoint
        self.admin_connections.append(websocket)

    def disconnect(self, websocket: WebSocket, hospital_id: int):
        if hospital_id in self.active_connections:
            if websocket in self.active_connections[hospital_id]:
                self.active_connections[hospital_id].remove(websocket)
    
    def disconnect_admin(self, websocket: WebSocket):
        if websocket in self.admin_connections:
            self.admin_connections.remove(websocket)

    async def broadcast_to_hospital(self, message: dict, hospital_id: int):
        # 1. Send to Specific Hospital Staff
        if hospital_id in self.active_connections:
            for connection in self.active_connections[hospital_id]:
                await connection.send_json(message)
        
        # 2. Send to ALL Global Admins
        for connection in self.admin_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/alerts")
async def websocket_endpoint(websocket: WebSocket, token: str):
    # ACCEPT IMMEDIATELY to complete handshake
    await websocket.accept()
    
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        role: str = payload.get("role")
        hospital_id = payload.get("hospital_id")
        
        print(f"WS Connection Attempt: Payload={payload}")
        
        # Logic:
        # If ADMIN -> Global Listener (regardless of hospital_id)
        # If NOT ADMIN -> strict hospital_id isolation
        
        if role == "ADMIN" or role == "UserRole.ADMIN":
            await manager.connect_admin(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect_admin(websocket)
        else:
            if hospital_id is None:
                # Regular user with no hospital? Reject.
                await websocket.close(code=1008)
                return
                
            await manager.connect(websocket, int(hospital_id))
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect(websocket, int(hospital_id))
            
    except JWTError:
        print("WS Token Error")
        await websocket.close(code=1008)
        return
    except Exception as e:
        print(f"WS Error: {e}")
        await websocket.close(code=1011)

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
    device_id = payload.get("device_id")
    
    # DYNAMIC LOOKUP: If IDs are missing, try to find device by IP
    if not hospital_id or not device_id:
        target_ip = payload.get("device_ip")
        if not target_ip:
            # Fallback to flow_details
            flow = payload.get("flow_details", {})
            target_ip = flow.get("dst_ip")
            
        if target_ip:
            device = db.query(models.Device).filter(models.Device.ip_address == target_ip).first()
            if device:
                print(f"[WS] Validated Device: {device.name} (ID: {device.id}, Hospital: {device.hospital_id})")
                hospital_id = device.hospital_id
                device_id = device.id
                # Update payload so frontend receives correct context
                payload["hospital_id"] = hospital_id
                payload["device_id"] = device_id
            else:
                print(f"[WS] WARNING: Device with IP {target_ip} not found in DB.")
    
    # Broadcast (To Hospital Staff AND Admins)
    if hospital_id and isinstance(hospital_id, int):
        await manager.broadcast_to_hospital(payload, hospital_id)
    else:
        # If still no hospital_id, perhaps broadcast ONLY to admins?
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
