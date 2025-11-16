# backend/app/routers/health.py
from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health Check"]
)

@router.get("/")
async def health_check():
    return {"status": "ok"}




