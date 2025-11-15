# backend/app/main.py
from fastapi import FastAPI
from .config import settings
from .routers import health

def create_app():
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0"
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
