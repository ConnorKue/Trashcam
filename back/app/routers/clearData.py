# backend/app/routers/clearData.py
import os
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(
    prefix="/clearData",
    tags=["Clear Data"],
)

def get_current_csv_path() -> Path:
    # This resolves to: RCOS/back/current.csv
    return Path(__file__).resolve().parents[2] / "current.csv"

@router.delete("/")
async def clear_data():
    file_path = get_current_csv_path()

    try:
        # Make sure the folder exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Delete if it exists
        if file_path.exists():
            file_path.unlink()

        # Recreate an empty file
        file_path.touch()

        return {"message": f"File {file_path.name} cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear file: {str(e)}"}
