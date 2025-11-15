# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import health

def create_app():
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0"
    )
    
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)

    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    print("FastAPI backend started")

@app.on_event("shutdown")
async def shutdown_event():
    print("FastAPI backend shutting down")
