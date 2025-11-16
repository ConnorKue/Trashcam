# backend/app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    ENV: str = os.getenv("ENV", "dev")
    APP_NAME: str = os.getenv("APP_NAME", "MyBackend")
    PORT: int = int(os.getenv("PORT"))
    UDP_SERVER_HOST: str = os.getenv("UDP_SERVER_HOST", "129.161.144.78")
    UDP_SERVER_PORT: int = int(os.getenv("UDP_SERVER_PORT", "5002"))
settings = Settings()
