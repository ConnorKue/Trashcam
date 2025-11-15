import cv2
import time
import os
from datetime import datetime

import torch
from ultralytics import YOLO

from dotenv import load_dotenv
import csv
import math


model = YOLO("yolo11s.pt")
load_dotenv()


VIDEO_URL = "http://129.161.144.78:8080/color"
# VIDEO_URL = "http://"+str(os.getenv("PI_USER"))+":"+str(os.getenv("PI_PASSWORD"))+"@"+str(os.getenv("PI_IP"))+":"+str(os.getenv("PI_PORT"))+"/?action=stream.mjpeg"


CATEGORY_MAP = {}
currentItems = []


def load_category_map():
    global CATEGORY_MAP
    CATEGORY_MAP = {}
    try:
        with open('whatis.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cls = row.get("class_name")
                cat = row.get("category", "")
                if cls:
                    CATEGORY_MAP[cls] = cat
    except FileNotFoundError:
        # If file doesn't exist, create it with just header on save
        CATEGORY_MAP = {}


# def save_category_map():
#     with open('whatis.csv', 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=["class_name", "category"])
#         writer.writeheader()
#         for cls, cat in CATEGORY_MAP.items():
#             writer.writerow({"class_name": cls, "category": cat})




def parseLocation(loc_str):
    # Remove parentheses: "(123, 456)" → "123, 456"
    inner = loc_str.strip("()")
    
    # Split on the comma
    parts = inner.split(",")
    
    # Convert pieces to integers and strip spaces
    x = int(parts[0].strip())
    y = int(parts[1].strip())
    
    return [x, y]

def addToCan(timestamp, location, item, classification):
    currentItems.append({
        "timestamp": timestamp,
        "location": f"{location}",  # store as string "(x, y)"
        "item": item,
        "classification": classification
    })

def updateCan():
    with open('current.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["timestamp", "location", "item", "classification"]
        )
        writer.writeheader()

        # nothing to write
        if not currentItems:
            return

        # sort and remove extraneous
        # sort by item first, then by timestamp so "prev" makes sense
        currentItems.sort(key=lambda entry: (entry["item"], entry["timestamp"]))

        newItems = []

        prev = None
        for entry in currentItems:
            # first item always kept
            if prev is None:
                newItems.append(entry)
                prev = entry
                continue

            # different item? always keep
            if entry["item"] != prev["item"]:
                newItems.append(entry)
                prev = entry
                continue

            # same item: check location + time
            loc_cur = parseLocation(entry["location"])
            x_i, y_i = loc_cur[0], loc_cur[1]

            loc_prev = parseLocation(prev["location"])
            x_prev, y_prev = loc_prev[0], loc_prev[1]

            distanceApart = math.sqrt((x_i - x_prev) ** 2 + (y_i - y_prev) ** 2)
            timeApart = abs(entry["timestamp"] - prev["timestamp"])

            # keep only if far enough in BOTH space and time
            if distanceApart > 15 and timeApart > 10:
                newItems.append(entry)
                prev = entry
            # else: skip as extraneous; prev stays as last kept

        # finally write filtered items
        for entry in newItems:
            writer.writerow(entry)




def classify_into_3(class_name: str) -> str:
    return CATEGORY_MAP.get(class_name, "unknown")

def process_frame(frame):
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # bounding box coords
        cls = int(box.cls)           # class id
        conf = float(box.conf)       # confidence

        class_name = results.names[cls]


        if class_name not in CATEGORY_MAP:
            # no category assigned yet; will show as "unknown"
            CATEGORY_MAP[class_name] = ""

        category = classify_into_3(class_name)

        addToCan(time.time(), (int(x1), int(y1)), class_name, category)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(frame, f"{results.names[cls]} {conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"{category} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText( frame, f"{category}", (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2 )
    return frame




def main():
    
    load_category_map()

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


        cv2.imshow("MJPEG Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    updateCan()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()