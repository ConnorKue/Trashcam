import cv2
import time
import os
from datetime import datetime
import math

import torch
from ultralytics import YOLO   # note: YOLO-World weights, same API

from dotenv import load_dotenv
import csv

# --------------------------
# YOLO-World setup
# --------------------------
# Use YOLO-World weights
model = YOLO("yolov8s-world.pt")

# Define the open-vocab classes you care about
WORLD_CLASSES = [
#recycling
"bottle", "cup", "paper plate", "pizza box", "soda can",
#trash
"paper napkin", "paper towel", "wrapper", "chip bag", "straw", "tissue", 
"cloth", "pen", "pencil", "battery", "earbuds", "duct tape", 
"paper bag", "laptop", "cell phone", "coat", "capri sun",
#compost
"pizza", "chocolate", "person", "meat", "Orange", "Apple", 
"Sandwiches"
]

# Tell YOLO-World to detect only these things
model.set_classes(WORLD_CLASSES)

load_dotenv()

VIDEO_URL = "http://129.161.144.78:8080/color"
# VIDEO_URL = "http://"+str(os.getenv("PI_USER"))+":"+str(os.getenv("PI_PASSWORD"))+"@"+str(os.getenv("PI_IP"))+":"+str(os.getenv("PI_PORT"))+"/?action=stream.mjpeg"

# --------------------------
# Category mapping
# item (prompt text) -> classification
# tweak these strings however you want
# --------------------------
CATEGORY_MAP = {
#recycling
"bottle": "recycling",
"cup": "recycling",
"paper plate": "recycling",
"pizza box": "recycling",
"soda can": "recycling",
#trash
"paper napkin":"trash",
"paper towel": "trash",
"wrapper": "trash",
"chip bag": "trash",
"straw": "trash",
"tissue": "trash",
"cloth": "trash",
"pen": "trash",
"pencil": "trash",
"battery": "trash",
"earbuds": "trash",
"duct tape": "trash",
"paper bag": "trash",
"laptop": "trash",
"cell phone": "trash",
"coat": "trash",
"capri sun": "trash",
#compost
"pizza": "compost",
"chocolate": "compost",
"person": "compost",
"meat": "compost",
"Orange": "compost",
"Apple": "compost",
"Sandwiches": "compost",

}

currentItems = []


def classify_into_3(item: str) -> str:
    return CATEGORY_MAP.get(item, "unknown")


def addToCan(timestamp, location, item, classification):
    currentItems.append({
        "timestamp": timestamp,
        "location": f"{location}",  # "(x, y)"
        "item": item,
        "classification": classification
    })


def parseLocation(loc_str: str):
    # "(123, 456)" -> [123, 456]
    inner = loc_str.strip("()")
    parts = inner.split(",")
    x = int(parts[0].strip())
    y = int(parts[1].strip())
    return [x, y]


def updateCan():
    with open('current.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["timestamp", "location", "item", "classification"]
        )
        writer.writeheader()

        # if you ever want to filter/dedup, do it here before writing
        for entry in currentItems:
            writer.writerow(entry)


def process_frame(frame):
    # YOLO-World inference; same API as YOLOv8
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls)
        conf = float(box.conf)

        # With YOLO-World + set_classes, names are exactly your WORLD_CLASSES
        item = model.names[cls]  # or results.names[cls]

        category = classify_into_3(item)

        # Log to can
        addToCan(time.time(), (int(x1), int(y1)), item, category)

        # Draw box + labels
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{item} {conf:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{category}",
            (int(x1), int(y1) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame


def main():
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open stream")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        frame = process_frame(frame)
        cv2.imshow("YOLO-World Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # dump whatever we saw this session
    updateCan()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
