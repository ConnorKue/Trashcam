from hcsr04 import HCSR04
import time
from machine import Pin

TRIGGER_PIN = 27
ECHO_PIN = 28

sensor = HCSR04(trigger_pin=TRIGGER_PIN, echo_pin=ECHO_PIN)

while True:
    try:
        distance = sensor.distance_cm()
        print(distance)  # <-- This goes over serial to the Pi
    except OSError as e:
        print("Error:", e)
    time.sleep(0.5)
