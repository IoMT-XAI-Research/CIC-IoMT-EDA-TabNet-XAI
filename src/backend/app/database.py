from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import os

# Check for DATABASE_URL environment variable (Render)
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Fix Render's quirk where it provides postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_DATABASE_URL = DATABASE_URL
    # PostgreSQL does not need check_same_thread
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
else:
    # Fallback to local SQLite (use absolute path for robustness)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "sql_app.db")
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

    # SQLite specific argument
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# 🔹 Tüm router'ların kullanacağı DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
