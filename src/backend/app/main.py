from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routers import auth, devices, analysis, websocket, prediction, hospitals, admin
from .services.ai_engine import ai_engine
from app.database import SessionLocal
from app import models

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="IoMT IDS Backend")

# Initialize AI Engine on Startup
@app.on_event("startup")
async def startup_event():
    ai_engine.load_model()

# --- BAŞLANGIÇ: OTOMATİK VERİ YÜKLEME ---
@app.on_event("startup")
def startup_db_client():
    db = SessionLocal()
    try:
        # 1. Hastane Kontrolü
        hospital = db.query(models.Hospital).filter(models.Hospital.code == "HST-001").first()
        if not hospital:
            print("🏥 Hastane oluşturuluyor (Render)...")
            hospital = models.Hospital(name="Merkez Şehir Hastanesi", code="HST-001")
            db.add(hospital)
            db.commit()
            db.refresh(hospital)
        
        # 2. Cihaz Kontrolü (Opsiyonel ama iyi olur)
        if db.query(models.Device).count() == 0:
            print("📟 Cihazlar ekleniyor...")
            devices = [
                models.Device(name="Oksijen Sensörü", ip_address="192.168.1.10", status="SAFE", hospital_id=hospital.id),
                models.Device(name="Akıllı Tansiyon", ip_address="192.168.1.11", status="SAFE", hospital_id=hospital.id),
            ]
            db.add_all(devices)
            db.commit()
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
    finally:
        db.close()
# --- BİTİŞ ---

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

@app.get("/")
def read_root():
    return {"message": "Welcome to IoMT IDS Backend"}
