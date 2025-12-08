from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import List, Dict
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
async def report_attack(event: dict):
    # event should contain hospital_id, device_id, score, etc.
    hospital_id = event.get("hospital_id")
    if hospital_id:
        await manager.broadcast_to_hospital(event, hospital_id)
    return {"status": "alert sent"}
