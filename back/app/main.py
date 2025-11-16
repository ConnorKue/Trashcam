# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import health
from .routers import clearData
from .routers import totalTrash
from .routers import log   
from .routers import fill

def create_app():
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(clearData.router)
    app.include_router(totalTrash.router)
    app.include_router(log.router)
    app.include_router(fill.router)

    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    print("FastAPI backend started")

@app.on_event("shutdown")
async def shutdown_event():
    print("FastAPI backend shutting down")
