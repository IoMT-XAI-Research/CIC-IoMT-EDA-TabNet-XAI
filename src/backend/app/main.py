from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routers import auth, devices, analysis, websocket, prediction, hospitals, admin, logs
from .services.ai_engine import ai_engine
from .database import SessionLocal
from . import models

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="IoMT IDS Backend")

# Initialize AI Engine on Startup
# --- BAŞLANGIÇ: OTOMATİK VERİ YÜKLEME ---
@app.on_event("startup")
async def startup_event():
    # Load AI Model
    ai_engine.load_model()
    
    # Start Background Task for Status Cleanup
    import asyncio
    from datetime import datetime, timedelta
    
    async def cleanup_stale_attacks():
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            try:
                db = SessionLocal()
                # Find devices that are ATTACK but haven't been updated in 10s
                cutoff = datetime.utcnow() - timedelta(seconds=10)
                stale_devices = db.query(models.Device).filter(
                    models.Device.status == models.DeviceStatus.ATTACK,
                    models.Device.last_updated < cutoff
                ).all()
                
                if stale_devices:
                    print(f"[{datetime.utcnow()}] Auto-Recovering {len(stale_devices)} devices...")
                    for device in stale_devices:
                        device.status = models.DeviceStatus.SAFE
                        device.last_risk_score = 0.0
                    db.commit()
                db.close()
            except Exception as e:
                print(f"Cleanup Error: {e}")

    asyncio.create_task(cleanup_stale_attacks())
    # Note: Database seeding is now handled via /admin/seed endpoint

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(devices.router)
app.include_router(analysis.router)
app.include_router(websocket.router)
app.include_router(prediction.router)
app.include_router(hospitals.router)
app.include_router(admin.router)
app.include_router(logs.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to IoMT IDS Backend"}
