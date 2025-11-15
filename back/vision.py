import cv2
import time
from dotenv import load_dotenv
import os


# "http://"+os.getenv("PI_IP")+":"+os.getenv("PI_PORT")+"/stream.mjpeg/?action=stream"






load_dotenv()

VIDEO_URL = "http://"+os.getenv("PI_USER")+":"+os.getenv("PI_PASSWORD")+"@"+os.getenv("PI_IP")+":"+os.getenv("PI_PORT")+"/?action=stream.mjpeg"

def open_stream():
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap

cap = open_stream()

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Warning: failed to read frame, reopening stream...")
            cap.release()
            time.sleep(0.5)  # small backoff
            cap = open_stream()
            if not cap.isOpened():
                print("Error: could not reopen stream, exiting.")
                break
            continue



        try:
            cv2.imshow("HTTP Stream", frame)
        except cv2.error as e:
            print("OpenCV display error:", e)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()