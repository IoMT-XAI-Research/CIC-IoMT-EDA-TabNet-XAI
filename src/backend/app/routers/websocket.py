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
        self.active_connections: Dict[int, List[WebSocket]] = {}
        self.admin_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, hospital_id: int):
        if hospital_id not in self.active_connections:
            self.active_connections[hospital_id] = []
        self.active_connections[hospital_id].append(websocket)
        
    async def connect_admin(self, websocket: WebSocket):
        self.admin_connections.append(websocket)

    def disconnect(self, websocket: WebSocket, hospital_id: int):
        if hospital_id in self.active_connections:
            if websocket in self.active_connections[hospital_id]:
                self.active_connections[hospital_id].remove(websocket)
    
    def disconnect_admin(self, websocket: WebSocket):
        if websocket in self.admin_connections:
            self.admin_connections.remove(websocket)

    async def broadcast_to_hospital(self, message: dict, hospital_id: int):
        if hospital_id in self.active_connections:
            for connection in self.active_connections[hospital_id]:
                await connection.send_json(message)
        
        for connection in self.admin_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/alerts")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        role: str = payload.get("role")
        hospital_id = payload.get("hospital_id")
        user_email = payload.get("sub")
        
        print(f"WS Connection Attempt: Payload={payload}")
        
        if (role == "ADMIN" or role == "UserRole.ADMIN") and hospital_id is None:
            await manager.connect_admin(websocket)
            print(f"WS: Global Admin connected. (User: {user_email})")
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect_admin(websocket)
        elif hospital_id is not None:
            await manager.connect(websocket, int(hospital_id))
            print(f"WS: User connected to Hospital {hospital_id}. (Role: {role})")
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect(websocket, int(hospital_id))
        else:
            await websocket.close(code=1008)
            return

    except JWTError:
        print("WS Token Error")
        await websocket.close(code=1008)
        return
    except Exception as e:
        print(f"WS Error: {e}")
        await websocket.close(code=1011)

@router.post("/internal/report-attack")
async def report_attack(
    request_data: schemas.AttackNotification,
    db: Session = Depends(dependencies.get_db)
):
    # Compatibility: Convert Pydantic model to dict so existing code using payload.get() works
    payload = request_data.model_dump()
    
    hospital_id = payload.get("hospital_id")
    device_id = payload.get("device_id")
    
    # Cihazı IP adresinden bulma işlemi
    if not hospital_id or not device_id:
        target_ip = payload.get("device_ip")
        if not target_ip:
            flow = payload.get("flow_details", {})
            target_ip = flow.get("dst_ip")
            
        if target_ip:
            device = db.query(models.Device).filter(models.Device.ip_address == target_ip).first()
            if device:
                hospital_id = device.hospital_id
                device_id = device.id
                payload["hospital_id"] = hospital_id
                payload["device_id"] = device_id
                payload["device_name"] = device.name
                
                print(f"[WS] SECURE LOOKUP: Mapped Attack to Device {device.name}")
                
                # --- KRITIK DÜZELTME BURADA ---
                # Veritabanı sadece "ATTACK" veya "SAFE" kabul ediyor.
                # "DDoS" gönderirsek sistem çöker (500 Hatası).
                prediction = payload.get("prediction", {})
                
                if prediction.get("is_attack"):
                    # Saldırı türü ne olursa olsun veritabanına "ATTACK" yazıyoruz.
                    # Mobil uygulama zaten saldırı detayını WebSocket'ten alıp gösterecek.
                    device.status = "ATTACK"
                else:
                    device.status = "SAFE"
                    
                db.commit()      # Değişikliği kaydet
                db.refresh(device) # Cihaz bilgisini yenile
                # ------------------------------
            else:
                print(f"[WS] WARNING: Device with IP {target_ip} not found in DB.")
                pass
    
    # Bildirimi Yayınla
    if hospital_id and isinstance(hospital_id, int):
        await manager.broadcast_to_hospital(payload, hospital_id)
    else:
        pass
        
    # Log Kaydı Oluştur
    prediction = payload.get("prediction", {})
    if prediction.get("is_attack") and hospital_id:
        from .logs import create_activity_log
        
        device_name = payload.get("device_name")
        if not device_name:
             device_name = f"Device {device_id}" if device_id else "Unknown Device"
             
        prob = prediction.get("probability", 0.0)
        attack_type = payload.get("attack_type", "Saldırı")
        
        create_activity_log(
            db, 
            "KRİTİK SALDIRI TESPİTİ", 
            f"{device_name} üzerinde {attack_type} tespit edildi! (Güven: {prob:.2f})", 
            "DANGER", 
            hospital_id
        )
    elif prediction.get("is_attack"):
        print("[WS] Skipped Logging: Attack on Unregistered Device.")
        
    return {"status": "alert processed"}
