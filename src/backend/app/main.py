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
# --- BAŞLANGIÇ: OTOMATİK VERİ YÜKLEME ---
@app.on_event("startup")
async def startup_event():
    # Load AI Model
    ai_engine.load_model()
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

@app.get("/")
def read_root():
    return {"message": "Welcome to IoMT IDS Backend"}
