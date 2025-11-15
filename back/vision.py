import cv2
import time
import os
from datetime import datetime

import torch
from ultralytics import YOLO

from dotenv import load_dotenv
import os

model = YOLO("yolov8n.pt")
load_dotenv()

VIDEO_URL = "http://"+str(os.getenv("PI_USER"))+":"+str(os.getenv("PI_PASSWORD"))+"@"+str(os.getenv("PI_IP"))+":"+str(os.getenv("PI_PORT"))+"/?action=stream.mjpeg"




def process_frame(frame):
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # bounding box coords
        cls = int(box.cls)           # class id
        conf = float(box.conf)       # confidence

        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        cv2.putText(frame,
                    f"{results.names[cls]} {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    return frame




def main():
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open stream")
        return

    while True:
        ret, frame = cap.read()

        frame = process_frame(frame)

        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("MJPEG Stream", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()