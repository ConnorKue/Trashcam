# backend/app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    ENV: str = os.getenv("ENV", "dev")
    APP_NAME: str = os.getenv("APP_NAME", "MyBackend")
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()
