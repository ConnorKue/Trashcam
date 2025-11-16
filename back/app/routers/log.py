from fastapi import APIRouter
from pathlib import Path
import csv
from datetime import datetime

router = APIRouter(prefix="/logs", tags=["Logs"])


def get_logs_path() -> Path:
    # back/current.csv relative to this file
    return Path(__file__).resolve().parents[2] / "current.csv"


@router.get("/")
def get_logs():
    file_path = get_logs_path()

    # If current.csv doesn't exist yet, just return empty list
    if not file_path.exists():
        return {"logs": []}

    logs = []

    with file_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not row:
                continue

            # --- timestamp handling ---
            raw_ts = (
                row.get("timestamp")
                or row.get("time")
                or row.get("ts")
                or ""
            )

            ts = raw_ts
            if raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts).strftime("%H:%M:%S")
                except Exception:
                    # leave as-is if format is weird
                    ts = raw_ts

            # --- item name ---
            item = (
                row.get("item")
                or row.get("label")
                or row.get("object")
                or ""
            )

            # --- class/category label ---
            cls = (
                row.get("classification")
                or row.get("category")
                or row.get("coarse_type")
                or row.get("type")
                or ""
            )

            logs.append(
                {
                    "timestamp": ts,
                    "item": item,
                    "classification": cls,
                }
            )

    return {"logs": logs}
