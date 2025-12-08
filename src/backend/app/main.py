from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routers import auth, devices, analysis, websocket

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="IoMT IDS Backend")

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

@app.get("/")
def read_root():
    return {"message": "Welcome to IoMT IDS Backend"}
