import cv2
import time
from datetime import datetime

import torch
from ultralytics import YOLO

from dotenv import load_dotenv
import os

# =========================
# Stream setup
# =========================

load_dotenv()

VIDEO_URL = "http://"+str(os.getenv("PI_USER"))+":"+str(os.getenv("PI_PASSWORD"))+"@"+str(os.getenv("PI_IP"))+":"+str(os.getenv("PI_PORT"))+"/?action=stream.mjpeg"


def open_stream():
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# =========================
# YOLO-World: item classes
# =========================
# You can tweak/extend this list any time.
ITEM_CLASSES = [
    # Fruits / veg
    "apple",
    "banana",
    "orange",
    "lettuce",
    "salad",
    "tomato",
    "broccoli",
    "carrot",

    # Common foods
    "pizza slice",
    "burger",
    "sandwich",
    "french fries",
    "chicken wing",
    "chicken nugget",
    "fried chicken",
    "donut",
    "cookie",

    # Packaging / containers
    "water bottle",
    "plastic bottle",
    "aluminum can",
    "soda can",
    "glass bottle",
    "paper cup",
    "coffee cup",
    "cardboard box",
    "paper bag",
    "plastic bag",
]

# Which things count as recycling?
RECYCLING_ITEMS = {
    "water bottle",
    "plastic bottle",
    "aluminum can",
    "soda can",
    "glass bottle",
    "paper cup",
    "coffee cup",
    "cardboard box",
    "paper bag",
}


# =========================
# Category + CO2 profiles
# (rough demo values; tweak as you like)
# =========================

CATEGORY_MAP = {
    # fruit
    "apple": "fruit",
    "banana": "fruit",
    "orange": "fruit",

    # veg / salad
    "lettuce": "veg",
    "salad": "veg",
    "tomato": "veg",
    "broccoli": "veg",
    "carrot": "veg",

    # foods / meals
    "pizza slice": "meal",
    "burger": "meal",
    "sandwich": "meal",
    "french fries": "snack",
    "chicken wing": "meat",
    "chicken nugget": "meat",
    "fried chicken": "meat",
    "donut": "snack",
    "cookie": "snack",

    # packaging / containers
    "water bottle": "packaging",
    "plastic bottle": "packaging",
    "aluminum can": "packaging",
    "soda can": "packaging",
    "glass bottle": "packaging",
    "paper cup": "packaging",
    "coffee cup": "packaging",
    "cardboard box": "packaging",
    "paper bag": "packaging",
    "plastic bag": "packaging",
}

# Per-item CO2 profiles (kg CO2e per item) + fraction saved if recycled
ITEM_CO2_PROFILE = {
    # Fruits (~0.06 kg per apple, ~0.1 kg per banana: rough typical LCAs)
    "apple": {"co2_item_kg": 0.06, "saving_fraction": 0.0},
    "banana": {"co2_item_kg": 0.10, "saving_fraction": 0.0},
    "orange": {"co2_item_kg": 0.08, "saving_fraction": 0.0},

    # Veg / salad (very rough)
    "lettuce": {"co2_item_kg": 0.03, "saving_fraction": 0.0},
    "salad": {"co2_item_kg": 0.15, "saving_fraction": 0.0},
    "tomato": {"co2_item_kg": 0.03, "saving_fraction": 0.0},
    "broccoli": {"co2_item_kg": 0.05, "saving_fraction": 0.0},
    "carrot": {"co2_item_kg": 0.02, "saving_fraction": 0.0},

    # Meals / meat / snacks (order-of-magnitude demo numbers)
    "pizza slice": {"co2_item_kg": 1.0, "saving_fraction": 0.0},
    "burger": {"co2_item_kg": 3.0, "saving_fraction": 0.0},
    "sandwich": {"co2_item_kg": 1.0, "saving_fraction": 0.0},
    "french fries": {"co2_item_kg": 0.5, "saving_fraction": 0.0},
    "chicken wing": {"co2_item_kg": 2.0, "saving_fraction": 0.0},
    "chicken nugget": {"co2_item_kg": 1.5, "saving_fraction": 0.0},
    "fried chicken": {"co2_item_kg": 2.5, "saving_fraction": 0.0},
    "donut": {"co2_item_kg": 0.3, "saving_fraction": 0.0},
    "cookie": {"co2_item_kg": 0.2, "saving_fraction": 0.0},

    # Packaging
    # ~0.08 kg CO2 per 500 ml plastic water bottle (typical bottled water LCA range)
    "water bottle": {"co2_item_kg": 0.08, "saving_fraction": 0.7},
    "plastic bottle": {"co2_item_kg": 0.08, "saving_fraction": 0.7},
    # ~0.10 kg CO2 per aluminum can, very large benefit from recycling
    "aluminum can": {"co2_item_kg": 0.10, "saving_fraction": 0.9},
    "soda can": {"co2_item_kg": 0.10, "saving_fraction": 0.9},
    # Glass bottle: slightly higher CO2, modest recycling benefit
    "glass bottle": {"co2_item_kg": 0.15, "saving_fraction": 0.3},

    "paper cup": {"co2_item_kg": 0.03, "saving_fraction": 0.5},
    "coffee cup": {"co2_item_kg": 0.05, "saving_fraction": 0.3},
    "cardboard box": {"co2_item_kg": 0.04, "saving_fraction": 0.5},
    "paper bag": {"co2_item_kg": 0.02, "saving_fraction": 0.5},
    "plastic bag": {"co2_item_kg": 0.03, "saving_fraction": 0.0},  # usually not recycled
}

DEFAULT_CATEGORY_CO2 = {
    "fruit": {"co2_item_kg": 0.08, "saving_fraction": 0.0},
    "veg": {"co2_item_kg": 0.04, "saving_fraction": 0.0},
    "meal": {"co2_item_kg": 1.5, "saving_fraction": 0.0},
    "snack": {"co2_item_kg": 0.3, "saving_fraction": 0.0},
    "meat": {"co2_item_kg": 2.0, "saving_fraction": 0.0},
    "packaging": {"co2_item_kg": 0.05, "saving_fraction": 0.5},
    "other": {"co2_item_kg": 0.05, "saving_fraction": 0.0},
}


def estimate_co2(item_label: str, bin_type: str):
    """
    Returns: (category, co2_item_kg, co2_saved_kg)
    """
    category = CATEGORY_MAP.get(item_label, "other")
    profile = ITEM_CO2_PROFILE.get(item_label)

    if profile is None:
        profile = DEFAULT_CATEGORY_CO2.get(category, DEFAULT_CATEGORY_CO2["other"])

    co2_item = profile["co2_item_kg"]
    frac = profile.get("saving_fraction", 0.0)
    co2_saved = co2_item * frac if bin_type == "recycling" else 0.0

    return category, co2_item, co2_saved


# =========================
# Load YOLO-World model (GPU if available)
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# With 16 GB VRAM you could also try "yolov8m-worldv2.pt" for more accuracy.
model = YOLO("yolov8s-worldv2.pt").to(device)
model.set_classes(ITEM_CLASSES)  # open-vocab prompts


# =========================
# Simple text "database"
# =========================

LOG_PATH = "trash_log.txt"


def log_item(
    track_id,
    frame_idx,
    item_label,
    bin_type,
    w,
    h,
    conf,
    category,
    co2_item_kg,
    co2_saved_kg,
    x_center,
    y_center,
):
    """
    Append a line to trash_log.txt:
    timestamp, track_id, frame, item, bin, category, size, position, conf, CO2
    """
    size_px = w * h
    ts = datetime.now().isoformat(timespec="seconds")

    line = (
        f"{ts}, "
        f"track={track_id}, "
        f"frame={frame_idx}, "
        f"item={item_label}, "
        f"bin={bin_type}, "
        f"category={category}, "
        f"area_px={size_px}, w={w}, h={h}, "
        f"x_center={x_center:.3f}, y_center={y_center:.3f}, "
        f"conf={conf:.3f}, "
        f"co2_kg={co2_item_kg:.4f}, co2_saved_kg={co2_saved_kg:.4f}\n"
    )

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


# =========================
# Main loop with tracking
# =========================

cap = open_stream()
frame_idx = 0

# Track IDs we've already logged (so each physical object is logged once)
logged_track_ids = set()

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: failed to read frame, reopening stream...")
            cap.release()
            time.sleep(0.5)
            cap = open_stream()
            if not cap.isOpened():
                print("Error: could not reopen stream, exiting.")
                break
            continue

        frame_idx += 1

        # Downscale for speed (you can bump this up if FPS is great)
        frame_small = cv2.resize(frame, (640, 480))
        h_img, w_img, _ = frame_small.shape

        # --- YOLO-World + ByteTrack tracking ---
        # persist=True keeps track IDs across frames
        # tracker="bytetrack.yaml" uses the built-in ByteTrack config
        # Docs: Ultralytics track mode + persist IDs 
        results = model.track(
            frame_small,
            conf=0.35,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
        )[0]

        boxes = results.boxes
        ids = boxes.id  # Tensor of track IDs (or None if no tracking)

        if ids is None:
            # No tracks this frame
            cv2.imshow("TrashCam – YOLO-World", frame_small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Convert IDs to CPU list so we can zip with boxes
        track_ids = ids.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            # Some trackers can produce -1 for "unassigned" tracks; skip those
            if track_id is None or track_id < 0:
                continue

            cls_idx = int(box.cls[0].item())
            if cls_idx < 0 or cls_idx >= len(ITEM_CLASSES):
                continue

            item_label = ITEM_CLASSES[cls_idx]
            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            w_box = x2 - x1
            h_box = y2 - y1

            # High-level bin type: recycling vs trash
            bin_type = "recycling" if item_label in RECYCLING_ITEMS else "trash"

            # Center position (normalized 0–1)
            x_center = (x1 + x2) / 2.0 / w_img
            y_center = (y1 + y2) / 2.0 / h_img

            # CO2 estimates
            category, co2_item_kg, co2_saved_kg = estimate_co2(item_label, bin_type)

            # ---- draw for debugging every frame ----
            color = (0, 255, 0) if bin_type == "recycling" else (0, 0, 255)
            debug_text = (
                f"ID {track_id} | {item_label} | {bin_type} | "
                f"{co2_item_kg*1000:.0f} gCO2"
            )

            cv2.rectangle(frame_small, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_small,
                debug_text,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )

            # ---- LOG ONLY ON FIRST TIME WE SEE THIS track_id ----
            if track_id not in logged_track_ids:
                log_item(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    item_label=item_label,
                    bin_type=bin_type,
                    w=w_box,
                    h=h_box,
                    conf=conf,
                    category=category,
                    co2_item_kg=co2_item_kg,
                    co2_saved_kg=co2_saved_kg,
                    x_center=x_center,
                    y_center=y_center,
                )
                logged_track_ids.add(track_id)

        cv2.imshow("TrashCam – YOLO-World", frame_small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
