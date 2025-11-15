import cv2
import time
import os
from datetime import datetime
from collections import defaultdict, deque
import json
import numpy as np
import csv

import torch
from ultralytics import YOLO
from dotenv import load_dotenv

# =========================
# Load env BEFORE using os.getenv
# =========================
load_dotenv()

# =========================
# Config
# =========================
CONF_THRES = 0.20
IOU_THRES = 0.30  # For NMS

LOG_FILE = "detections.log"
UNKNOWN_LOG_FILE = "unknown_labels.log"
STATS_FILE = "detection_stats.json"
CSV_FILE = "current.csv"

DEBUG_LOG_UNKNOWN = True
ENABLE_TRACKING = False  # Disabled for YOLO-World compatibility
FRAME_SKIP = 0  # Process every N frames (0 = process all)
USE_FP16 = False  # Disabled for YOLO-World compatibility
STATS_SAVE_INTERVAL = 30  # Save stats every N seconds

# Spatial dedupe: group top-left corners into buckets of this size (pixels)
POS_MARGIN_PX = 45   # 45px region in x and y

# Map from detected prompt/label -> coarse type (keys kept lowercase)
PROMPT_TO_COARSE = {
    # ------------ PLASTIC (recycling) ------------
    "plastic bottle": "plastic",
    "water bottle": "plastic",
    "soda bottle": "plastic",
    "plastic cup": "plastic",
    "disposable plastic cup": "plastic",
    "plastic food container": "plastic",
    "takeout container": "plastic",
    "plastic clamshell container": "plastic",
    "tupperware": "plastic",

    "plastic bag": "plastic",
    "grocery bag": "plastic",
    "shopping bag": "plastic",
    "trash bag": "plastic",

    "chip bag": "plastic",
    "snack bag": "plastic",
    "snack wrapper": "plastic",
    "candy wrapper": "plastic",
    "plastic wrapper": "plastic",
    "straw": "plastic",
    "plastic straw": "plastic",
    "cup lid": "plastic",
    "plastic lid": "plastic",

    # ------------ METAL (recycling) ------------
    "aluminum can": "metal",
    "soda can": "metal",
    "beer can": "metal",
    "metal can": "metal",
    "tin can": "metal",
    "energy drink can": "metal",

    "aluminum foil": "metal",
    "foil": "metal",

    # ------------ GLASS (recycling) ------------
    "glass bottle": "glass",
    "glass jar": "glass",
    "glass cup": "glass",

    # ------------ PAPER / CARDBOARD (recycling) ------------
    "paper cup": "paper",
    "coffee cup": "paper",
    "paper plate": "paper",
    "paper bowl": "paper",

    "cardboard box": "paper",
    "shipping box": "paper",
    "pizza box": "paper",
    "cereal box": "paper",

    "paper bag": "paper",
    "paper wrapper": "paper",
    "newspaper": "paper",
    "magazine": "paper",
    "paper flyer": "paper",
    "paper napkin": "paper",
    "paper towel": "paper",

    # ------------ FOOD / ORGANICS (trash) ------------
    "food": "food",
    "food scraps": "food",
    "leftover food": "food",
    "leftover meal": "food",

    "banana peel": "fruit",
    "apple core": "fruit",
    "orange peel": "fruit",
    "fruit": "fruit",

    "vegetable": "vegetable",
    "salad": "vegetable",
    "lettuce": "vegetable",
    "broccoli": "vegetable",
    "carrot": "vegetable",

    "pizza slice": "food",
    "sandwich": "food",
    "burger": "food",
    "french fries": "food",

    "chicken wing": "meat",
    "chicken bone": "meat",
    "meat": "meat",

    # ------------ DRINKS / LIQUID CONTEXT ------------
    "drink cup": "liquid",
    "soft drink cup": "liquid",
    "cold drink cup": "liquid",
    "iced coffee cup": "liquid",
    "smoothie cup": "liquid",

    "juice box": "liquid",
    "juice pouch": "liquid",
    "capri sun": "liquid",
    "coffee cup with sleeve": "liquid",
}

BASE_WORD_TO_COARSE = {
    "bottle": "plastic",
    "can": "metal",
    "cup": "paper",
    "mug": "paper",
    "box": "paper",
    "bag": "plastic",
    "plate": "paper",
}

COARSE_TO_BIN = {
    "plastic": "recycling",
    "metal": "recycling",
    "glass": "recycling",
    "paper": "recycling",
    "food": "trash",
    "fruit": "trash",
    "vegetable": "trash",
    "meat": "trash",
    "liquid": "trash",
}

COARSE_CO2 = {
    "plastic":   {"co2_item_kg": 0.08, "saving_fraction": 0.7},
    "metal":     {"co2_item_kg": 0.10, "saving_fraction": 0.9},
    "glass":     {"co2_item_kg": 0.15, "saving_fraction": 0.3},
    "paper":     {"co2_item_kg": 0.04, "saving_fraction": 0.5},
    "food":      {"co2_item_kg": 1.0,  "saving_fraction": 0.0},
    "fruit":     {"co2_item_kg": 0.08, "saving_fraction": 0.0},
    "vegetable": {"co2_item_kg": 0.05, "saving_fraction": 0.0},
    "meat":      {"co2_item_kg": 3.0,  "saving_fraction": 0.0},
    "liquid":    {"co2_item_kg": 0.02, "saving_fraction": 0.0},
    "other":     {"co2_item_kg": 0.05, "saving_fraction": 0.0},
}

# =========================
# GPU/CUDA Verification
# =========================
def check_gpu_availability():
    """Check and print GPU/CUDA availability"""
    print("\n" + "="*50)
    print("GPU/CUDA DIAGNOSTICS")
    print("="*50)
    
    # PyTorch CUDA info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory - Total: {total_memory:.2f}GB, Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        device = "cuda:0"
    else:
        print("⚠️  WARNING: CUDA not available, running on CPU")
        print("This will be significantly slower!")
        device = "cpu"
    
    print("="*50 + "\n")
    return device

# =========================
# Statistics Tracking
# =========================
class DetectionStats:
    """Track comprehensive detection statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.items_by_type = defaultdict(int)
        self.items_by_bin = defaultdict(int)
        self.total_co2_saved = 0.0
        self.total_co2_footprint = 0.0
        self.confidence_scores = []
        self.processing_times = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)
        self.unique_tracked_items = set()
        self.last_save_time = time.time()
        
    def update(self, detections_data):
        """Update stats with new frame data"""
        self.frame_count += 1
        
        for det in detections_data:
            self.detection_count += 1
            self.items_by_type[det['coarse']] += 1
            self.items_by_bin[det['bin_type']] += 1
            self.total_co2_saved += det['co2_saved']
            self.total_co2_footprint += det['co2_item']
            self.confidence_scores.append(det['conf'])
            
            if det.get('track_id') is not None:
                track_key = f"{det['label']}_{det['track_id']}"
                self.unique_tracked_items.add(track_key)
    
    def add_processing_time(self, proc_time):
        """Add frame processing time"""
        self.processing_times.append(proc_time)
    
    def add_fps(self, fps):
        """Add FPS measurement"""
        self.fps_history.append(fps)
    
    def get_summary(self):
        """Get current statistics summary"""
        runtime = time.time() - self.start_time
        avg_conf = np.mean(self.confidence_scores) if self.confidence_scores else 0
        avg_proc_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        recycling_rate = (
            self.items_by_bin['recycling'] / max(1, sum(self.items_by_bin.values())) * 100
        )
        
        return {
            'runtime_seconds': runtime,
            'frames_processed': self.frame_count,
            'total_detections': self.detection_count,
            'unique_items': len(self.unique_tracked_items),
            'avg_confidence': float(avg_conf),
            'avg_processing_time_ms': float(avg_proc_time * 1000),
            'avg_fps': float(avg_fps),
            'items_by_type': dict(self.items_by_type),
            'items_by_bin': dict(self.items_by_bin),
            'recycling_rate_percent': float(recycling_rate),
            'total_co2_saved_kg': float(self.total_co2_saved),
            'total_co2_footprint_kg': float(self.total_co2_footprint),
            'detections_per_hour': float(self.detection_count / max(runtime/3600, 0.001)),
        }
    
    def save_to_file(self):
        """Save statistics to JSON file"""
        try:
            with open(STATS_FILE, 'w') as f:
                json.dump(self.get_summary(), f, indent=2)
        except Exception as e:
            print(f"Failed to save stats: {e}")
    
    def print_summary(self):
        """Print current statistics to console"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("DETECTION STATISTICS")
        print("="*50)
        print(f"Runtime: {summary['runtime_seconds']:.1f}s")
        print(f"Frames Processed: {summary['frames_processed']}")
        print(f"Avg FPS: {summary['avg_fps']:.1f}")
        print(f"Avg Processing Time: {summary['avg_processing_time_ms']:.1f}ms")
        print(f"Total Detections: {summary['total_detections']}")
        print(f"Unique Items: {summary['unique_items']}")
        print(f"Avg Confidence: {summary['avg_confidence']:.3f}")
        print(f"Recycling Rate: {summary['recycling_rate_percent']:.1f}%")
        print(f"CO₂ Saved: {summary['total_co2_saved_kg']:.3f}kg")
        print(f"CO₂ Footprint: {summary['total_co2_footprint_kg']:.3f}kg")
        print("\nItems by Type:")
        for item_type, count in sorted(summary['items_by_type'].items()):
            print(f"  {item_type}: {count}")
        print("="*50 + "\n")

# Track unknown labels we've already noted (for debug log only)
SEEN_UNKNOWN = set()

# Global set of event keys we've already logged (for CSV + detections.log)
SEEN_EVENTS = set()  # e.g. "plastic:2:5", "unknown:4:7"

def estimate_co2(coarse_category: str, bin_type: str):
    profile = COARSE_CO2.get(coarse_category, COARSE_CO2["other"])
    co2_item = profile["co2_item_kg"]
    frac = profile.get("saving_fraction", 0.0)
    co2_saved = co2_item * frac if bin_type == "recycling" else 0.0
    return coarse_category, co2_item, co2_saved

def classify_item(label: str):
    label_l = label.lower().strip()
    coarse = PROMPT_TO_COARSE.get(label_l)
    
    if coarse is None:
        for key, value in PROMPT_TO_COARSE.items():
            if key in label_l or label_l in key:
                coarse = value
                break
    
    if coarse is None:
        words = label_l.replace("-", " ").split()
        for w in words:
            if w in BASE_WORD_TO_COARSE:
                coarse = BASE_WORD_TO_COARSE[w]
                break
    
    if coarse is None:
        return None, None
    
    bin_type = COARSE_TO_BIN.get(coarse)
    if bin_type is None:
        return None, None
    
    return coarse, bin_type

def log_unknown_label(label: str):
    if not DEBUG_LOG_UNKNOWN:
        return
    key = label.lower()
    if key in SEEN_UNKNOWN:
        return
    SEEN_UNKNOWN.add(key)
    line = f"{datetime.now().isoformat()} - UNKNOWN_LABEL: '{label}'\n"
    try:
        with open(UNKNOWN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write unknown label log: {e}")

def log_new_item(label: str, coarse: str, bin_type: str, co2_item_kg: float, co2_saved_kg: float):
    """Log every NEW detection event to detections.log"""
    line = (
        f"{datetime.now().isoformat()} - {label} -> {coarse} -> {bin_type} | "
        f"co2_item_kg={co2_item_kg:.4f}, co2_saved_kg={co2_saved_kg:.4f}\n"
    )
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write to log file: {e}")

# ========== CSV OUTPUT HELPERS ==========

def init_csv():
    """Create / overwrite the CSV file with header."""
    try:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "TopCornerOfBoundary", "item", "class"])
    except Exception as e:
        print(f"Failed to init CSV file: {e}")

def log_csv_detection(x1: int, y1: int, label: str, cls_str: str):
    """Append one detection row to the CSV."""
    timestamp = datetime.now().isoformat()
    top_corner_str = f"{x1},{y1}"  # (x1,y1) as "x,y"
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, top_corner_str, label, cls_str])
    except Exception as e:
        print(f"Failed to write CSV detection: {e}")

# ========== SPATIAL DEDUPE (NO TIME WINDOW) ==========

def is_new_spatial_event(x1: int, y1: int, coarse: str, label: str) -> bool:
    """
    Returns True if this looks like a NEW item in this region
    for the entire runtime of the script.
    
    We define:
      - key_class = coarse if available, else 'unknown'
      - region_x = x1 // POS_MARGIN_PX
      - region_y = y1 // POS_MARGIN_PX
      - event_key = f"{key_class}:{region_x}:{region_y}"
    """
    global SEEN_EVENTS

    key_class = coarse if coarse is not None else "unknown"
    region_x = x1 // POS_MARGIN_PX
    region_y = y1 // POS_MARGIN_PX

    event_key = f"{key_class}:{region_x}:{region_y}"

    if event_key in SEEN_EVENTS:
        return False

    SEEN_EVENTS.add(event_key)
    return True

# ========================================

def process_frame(frame, model, stats):
    """Process a single frame with timing"""
    start_time = time.time()
    detections_data = []
    
    # Run YOLO inference
    results = model(frame, verbose=False, conf=CONF_THRES, iou=IOU_THRES)
    results = results[0]
    
    for box in results.boxes:
        conf = float(box.conf)
        if conf < CONF_THRES:
            continue
        
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(box.cls)
        label = results.names[cls]

        # Tracking disabled for YOLO-World
        track_id = None
        
        coarse, bin_type = classify_item(label)

        # ---- Unknown / unmapped items ----
        if bin_type is None:
            # spatial dedupe for unknowns too (by label + region)
            if is_new_spatial_event(x1, y1, None, label):
                log_unknown_label(label)
                log_csv_detection(x1, y1, label, "unknown")
            # stats currently ignore unknowns
            continue

        # Known + mapped item: spatial dedupe for the entire run
        if is_new_spatial_event(x1, y1, coarse, label):
            # Only NEW events update logs / stats / CSV
            coarse_cat, co2_item_kg, co2_saved_kg = estimate_co2(coarse, bin_type)

            log_new_item(label, coarse_cat, bin_type, co2_item_kg, co2_saved_kg)

            detections_data.append({
                'label': label,
                'coarse': coarse_cat,
                'bin_type': bin_type,
                'co2_item': co2_item_kg,
                'co2_saved': co2_saved_kg,
                'conf': conf,
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2)
            })

            log_csv_detection(x1, y1, label, coarse_cat)
        else:
            # Even if not new, we still compute CO2 for overlay only
            coarse_cat, co2_item_kg, co2_saved_kg = estimate_co2(coarse, bin_type)

        # Draw bounding box (always, so user sees live tracking)
        color = (0, 255, 0) if bin_type == "recycling" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        co2_item_g = co2_item_kg * 1000.0
        co2_saved_g = co2_saved_kg * 1000.0
        
        if co2_saved_g > 0:
            co2_text = f"{co2_saved_g:.0f}g CO₂ saved"
        else:
            co2_text = f"{co2_item_g:.0f}g CO₂"
        
        text = f"{label} | {bin_type} | {co2_text} ({conf:.2f})"
        
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Update stats with only NEW events
    stats.update(detections_data)
    processing_time = time.time() - start_time
    stats.add_processing_time(processing_time)
    
    return frame

def draw_info_panel(frame, stats):
    """Draw info panel with real-time statistics"""
    summary = stats.get_summary()
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw background rectangle
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Draw text
    y_offset = 35
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    
    lines = [
        f"FPS: {summary['avg_fps']:.1f}",
        f"Processing: {summary['avg_processing_time_ms']:.1f}ms",
        f"Detections (unique-ish): {summary['total_detections']}",
        f"Unique Items: {summary['unique_items']}",
        f"Recycling Rate: {summary['recycling_rate_percent']:.1f}%",
        f"CO2 Saved: {summary['total_co2_saved_kg']:.3f}kg",
    ]
    
    for line in lines:
        cv2.putText(frame, line, (20, y_offset), font, font_scale, color, 1)
        y_offset += line_height
    
    return frame

def main():
    # Check GPU availability
    device = check_gpu_availability()
    
    # Initialize model
    print("Loading YOLO model...")
    model = YOLO("yolov8s-worldv2.pt")
    
    # Set classes for YOLO-World (do this BEFORE any inference)
    DETECTION_PROMPTS = list(PROMPT_TO_COARSE.keys())
    try:
        model.set_classes(DETECTION_PROMPTS)
        print(f"Set YOLO-World classes to {len(DETECTION_PROMPTS)} categories")
    except Exception as e:
        print(f"Warning: model.set_classes failed: {e}")
    
    print("Model loaded successfully on GPU" if torch.cuda.is_available() else "Model loaded on CPU")
    
    # Build video URL
    VIDEO_URL = f"{os.getenv('PI_URL')}"
    
    print(f"\nConnecting to stream...")
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Failed to open stream")
        print(f"VIDEO_URL: {VIDEO_URL}")
        return
    
    print("✅ Stream opened successfully")
    print(f"Press 'q' to quit, 's' to print statistics\n")
    
    # Init CSV output
    init_csv()
    
    # Initialize stats
    stats = DetectionStats()
    frame_counter = 0
    fps_timer = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame")
            break
        
        # Frame skipping for performance
        if FRAME_SKIP > 0 and frame_counter % (FRAME_SKIP + 1) != 0:
            frame_counter += 1
            continue
        
        frame_counter += 1
        
        # Process frame
        frame = process_frame(frame, model, stats)
        
        # Draw info panel
        frame = draw_info_panel(frame, stats)
        
        # Calculate FPS
        current_time = time.time()
        if current_time - fps_timer > 0:
            fps = 1.0 / (current_time - fps_timer)
            stats.add_fps(fps)
        fps_timer = current_time
        
        # Save stats periodically
        if current_time - stats.last_save_time > STATS_SAVE_INTERVAL:
            stats.save_to_file()
            stats.last_save_time = current_time
        
        # Display
        cv2.imshow("Waste Detection System", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            stats.print_summary()
    
    # Final summary
    stats.print_summary()
    stats.save_to_file()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Program terminated successfully")

if __name__ == "__main__":
    main()
