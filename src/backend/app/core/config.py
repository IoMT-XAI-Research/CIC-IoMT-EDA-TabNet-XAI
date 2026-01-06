import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamic absolute path to .env file
# config.py is in src/backend/app/core/ -> Go up 3 levels to src/backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding='utf-8', 
        extra='ignore',
        case_sensitive=True
    )

    # App Config
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # SMTP Config
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "emirsozer1622e@gmail.com"
    SMTP_PASSWORD: str = "mhdb zops phjh jkhn" # App Password
    EMAILS_FROM_EMAIL: str = "emirsozer1622e@gmail.com"
    
    # Server Url
    SERVER_HOST: str = "http://4.231.100.94:8000"

settings = Settings()
