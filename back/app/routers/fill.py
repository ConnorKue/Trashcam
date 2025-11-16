# backend/app/routers/health.py
from fastapi import APIRouter

router = APIRouter(
    prefix="/fill",
    tags=["Fill Level Check"]
)

@router.get("/")
async def health_check():
    #this is temp logic will be fixed later this will be in percent
    return {"fill": 15}




