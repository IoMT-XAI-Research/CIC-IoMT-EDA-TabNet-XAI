from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Şimdilik SQLite (lokal dosya) kullanıyorsun
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# SQLite için özel argüman
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
