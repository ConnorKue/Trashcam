from fastapi import APIRouter
from pathlib import Path
import csv
from datetime import datetime

router = APIRouter(prefix="/logs", tags=["Logs"])

def get_logs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "current.csv"


@router.get("/")
def get_logs():
    file_path = get_logs_path()

    logs = []

    with file_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"]).strftime("%H:%M:%S")
            except:
                ts = row["timestamp"]

            logs.append({
                "timestamp": ts,
                "item": row["item"],
                "class": row["class"]
            })

    return {"logs": logs}
