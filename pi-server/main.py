import pyrealsense2 as rs
import numpy as np
import cv2
from mjpeg_streamer import MjpegServer, Stream
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import os
import json
import time
import threading
import socket
from socket import AF_INET, SOCK_DGRAM

CALIBRATION_FILE = "depth_calibration.json"
CALIBRATION_SAMPLES = 30  # Number of frames to average for calibration

# Post-processing parameters - tuned for maximum stability
TEMPORAL_FILTER_ALPHA = 0.08  # Lower = more smoothing, more stable (0.05-0.15 for high stability)
SPATIAL_FILTER_ENABLED = True
SPATIAL_FILTER_MAGNITUDE = 5  # Increased strength of spatial filtering (2-5 range)
SPATIAL_FILTER_SMOOTH_ALPHA = 0.7  # Higher = more aggressive smoothing
SPATIAL_FILTER_SMOOTH_DELTA = 50  # Larger threshold = more smoothing across depth changes
HOLE_FILLING_MODE = 2  # 2 = nearest from around (more conservative than farthest)
DEPTH_BUFFER_SIZE = 20  # Increased from 10 to 20 for more stable median
DEPTH_CHANGE_THRESHOLD = 5  # Ignore changes smaller than 5mm to prevent jitter

# Global state
current_state = {
    "empty_depth": None,
    "current_depth": None,
    "fill_percentage": None,
    "calibration_date": None,
    "calibrating": False,
    "bounding_box": {
        "shape": "rectangle",  # rectangle, circle, polygon
        "center_x": 0.5,  # normalized 0-1
        "center_y": 0.5,  # normalized 0-1
        "width": 0.6,  # normalized 0-1
        "height": 0.6,  # normalized 0-1
        "rotation": 0,  # degrees
        "num_sides": 6  # for polygon shape
    },
    "depth_socket": None,  # UDP socket for depth streaming
    "depth_clients": set(),  # Set of client addresses subscribed to depth updates
    "depth_buffer": [],  # Temporal buffer for median filtering
    "last_stable_depth": None  # Last known good depth reading
}

# Flask app for API
app = Flask(__name__)

@app.route('/calibrate', methods=['POST'])
def api_calibrate():
    """API endpoint to trigger calibration."""
    global current_state
    
    if current_state["calibrating"]:
        return jsonify({"success": False, "error": "Calibration already in progress"}), 400
    
    try:
        current_state["calibrating"] = True
        empty_depth = calibrate_empty_depth(current_state["pipe"])
        
        if empty_depth:
            current_state["empty_depth"] = empty_depth
            return jsonify({
                "success": True,
                "empty_depth": empty_depth,
                "message": "Calibration successful"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to collect valid depth samples"
            }), 500
    finally:
        current_state["calibrating"] = False

@app.route('/status', methods=['GET'])
def api_status():
    """API endpoint to get current status."""
    return jsonify({
        "success": True,
        "empty_depth": current_state["empty_depth"],
        "current_depth": current_state["current_depth"],
        "fill_percentage": current_state["fill_percentage"],
        "calibration_date": current_state["calibration_date"],
        "calibrating": current_state["calibrating"],
        "bounding_box": current_state["bounding_box"]
    })

@app.route('/bounding_box', methods=['POST'])
def api_update_bounding_box():
    """API endpoint to update bounding box parameters."""
    global current_state
    
    try:
        data = request.get_json()
        
        # Update bounding box parameters
        if "shape" in data:
            current_state["bounding_box"]["shape"] = data["shape"]
        if "center_x" in data:
            current_state["bounding_box"]["center_x"] = float(data["center_x"])
        if "center_y" in data:
            current_state["bounding_box"]["center_y"] = float(data["center_y"])
        if "width" in data:
            current_state["bounding_box"]["width"] = float(data["width"])
        if "height" in data:
            current_state["bounding_box"]["height"] = float(data["height"])
        if "rotation" in data:
            current_state["bounding_box"]["rotation"] = float(data["rotation"])
        if "num_sides" in data:
            current_state["bounding_box"]["num_sides"] = int(data["num_sides"])
        
        return jsonify({
            "success": True,
            "bounding_box": current_state["bounding_box"]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def run_flask():
    """Run Flask API server in a separate thread."""
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)


def apply_depth_filters(depth_frame):
    """
    Apply post-processing filters to reduce noise and artifacts.
    Returns the filtered depth frame.
    """
    # Spatial filter - edge-preserving smoothing
    if SPATIAL_FILTER_ENABLED:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, SPATIAL_FILTER_MAGNITUDE)
        spatial.set_option(rs.option.filter_smooth_alpha, SPATIAL_FILTER_SMOOTH_ALPHA)
        spatial.set_option(rs.option.filter_smooth_delta, SPATIAL_FILTER_SMOOTH_DELTA)
        depth_frame = spatial.process(depth_frame)
    
    # Temporal filter - reduce flickering over time
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, TEMPORAL_FILTER_ALPHA)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    depth_frame = temporal.process(depth_frame)
    
    # Hole filling - fill in missing depth data
    if HOLE_FILLING_MODE > 0:
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, HOLE_FILLING_MODE)
        depth_frame = hole_filling.process(depth_frame)
    
    return depth_frame


def apply_median_temporal_filter(current_depth):
    """
    Apply median filter over time to reduce sudden jumps.
    Includes hysteresis to ignore small fluctuations.
    Returns stabilized depth value.
    """
    global current_state
    
    if current_depth is None:
        return current_state.get("last_stable_depth")
    
    # Add to buffer
    current_state["depth_buffer"].append(current_depth)
    
    # Keep buffer at fixed size
    if len(current_state["depth_buffer"]) > DEPTH_BUFFER_SIZE:
        current_state["depth_buffer"].pop(0)
    
    # Calculate median of buffer for stability
    if len(current_state["depth_buffer"]) > 0:
        median_depth = np.median(current_state["depth_buffer"])
        
        # Apply hysteresis: only update if change exceeds threshold
        last_stable = current_state.get("last_stable_depth")
        if last_stable is None:
            stable_depth = median_depth
        else:
            depth_change = abs(median_depth - last_stable)
            if depth_change > DEPTH_CHANGE_THRESHOLD:
                # Smooth the transition using weighted average
                stable_depth = last_stable * 0.7 + median_depth * 0.3
            else:
                # Keep the previous value (dead zone)
                stable_depth = last_stable
        
        current_state["last_stable_depth"] = stable_depth
        return stable_depth
    
    return current_depth


def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "localhost"


def get_rotated_rect_points(center_x, center_y, width, height, angle_deg):
    """Get the four corner points of a rotated rectangle."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Half dimensions
    hw = width / 2
    hh = height / 2
    
    # Corner offsets (before rotation)
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate and translate
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y
    
    return rotated_corners.astype(np.int32)


def get_polygon_points(center_x, center_y, radius, num_sides, rotation_deg):
    """Get points for a regular polygon."""
    points = []
    angle_step = 360 / num_sides
    start_angle = rotation_deg
    
    for i in range(num_sides):
        angle = np.radians(start_angle + i * angle_step)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append([x, y])
    
    return np.array(points, dtype=np.int32)


def create_region_mask(img_height, img_width, bbox_params):
    """Create a binary mask for the region of interest based on bounding box parameters."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Convert normalized coordinates to pixel coordinates
    center_x = int(bbox_params["center_x"] * img_width)
    center_y = int(bbox_params["center_y"] * img_height)
    width = int(bbox_params["width"] * img_width)
    height = int(bbox_params["height"] * img_height)
    rotation = bbox_params["rotation"]
    shape = bbox_params["shape"]
    
    if shape == "rectangle":
        if rotation == 0:
            # Simple rectangle (no rotation)
            x1 = center_x - width // 2
            y1 = center_y - height // 2
            x2 = center_x + width // 2
            y2 = center_y + height // 2
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        else:
            # Rotated rectangle
            points = get_rotated_rect_points(center_x, center_y, width, height, rotation)
            cv2.fillPoly(mask, [points], 255)
    
    elif shape == "circle":
        radius = min(width, height) // 2
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    elif shape == "polygon":
        num_sides = bbox_params["num_sides"]
        radius = min(width, height) // 2
        points = get_polygon_points(center_x, center_y, radius, num_sides, rotation)
        cv2.fillPoly(mask, [points], 255)
    
    return mask


def create_cropped_depth_view(colorized_depth, bbox_params):
    """
    Create a cropped view centered on the bounding box midpoint.
    Shows as much of the depth map as possible while maintaining square aspect ratio.
    """
    img_height, img_width = colorized_depth.shape[:2]
    
    # Get center point of the bounding box
    center_x = int(bbox_params["center_x"] * img_width)
    center_y = int(bbox_params["center_y"] * img_height)
    
    # Determine the maximum square crop size that fits in the image
    # Use the smaller dimension to ensure we don't crop outside the image
    max_crop_size = min(
        center_x * 2,  # Can extend this far left and right
        center_y * 2,  # Can extend this far up and down
        (img_width - center_x) * 2,  # Space to the right
        (img_height - center_y) * 2,  # Space to the bottom
        img_width,  # Total width
        img_height  # Total height
    )
    
    # Calculate crop coordinates centered on the bounding box
    half_size = max_crop_size // 2
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(img_width, center_x + half_size)
    y2 = min(img_height, center_y + half_size)
    
    # Crop the image
    cropped = colorized_depth[y1:y2, x1:x2]
    
    # Resize to square for consistent display
    if cropped.size > 0:
        cropped_resized = cv2.resize(cropped, (300, 300), interpolation=cv2.INTER_AREA)
        return cropped_resized
    
    return np.zeros((300, 300, 3), dtype=np.uint8)


def add_inset_to_image(main_image, inset_image, position='bottom-right', margin=10):
    """
    Add a small inset image to the corner of the main image.
    """
    main_h, main_w = main_image.shape[:2]
    inset_h, inset_w = inset_image.shape[:2]
    
    # Calculate position
    if position == 'bottom-right':
        y1 = main_h - inset_h - margin
        x1 = main_w - inset_w - margin
    elif position == 'bottom-left':
        y1 = main_h - inset_h - margin
        x1 = margin
    elif position == 'top-right':
        y1 = margin
        x1 = main_w - inset_w - margin
    else:  # top-left
        y1 = margin
        x1 = margin
    
    y2 = y1 + inset_h
    x2 = x1 + inset_w
    
    # Create a copy to avoid modifying the original
    result = main_image.copy()
    
    # Add border around inset
    cv2.rectangle(result, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 2)
    
    # Overlay the inset
    result[y1:y2, x1:x2] = inset_image
    
    return result


def draw_bounding_shape(image, bbox_params, color=(0, 255, 0), thickness=2):
    """Draw the bounding shape on the image."""
    img_height, img_width = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    center_x = int(bbox_params["center_x"] * img_width)
    center_y = int(bbox_params["center_y"] * img_height)
    width = int(bbox_params["width"] * img_width)
    height = int(bbox_params["height"] * img_height)
    rotation = bbox_params["rotation"]
    shape = bbox_params["shape"]
    
    if shape == "rectangle":
        if rotation == 0:
            # Simple rectangle (no rotation)
            x1 = center_x - width // 2
            y1 = center_y - height // 2
            x2 = center_x + width // 2
            y2 = center_y + height // 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        else:
            # Rotated rectangle
            points = get_rotated_rect_points(center_x, center_y, width, height, rotation)
            cv2.polylines(image, [points], True, color, thickness)
    
    elif shape == "circle":
        radius = min(width, height) // 2
        cv2.circle(image, (center_x, center_y), radius, color, thickness)
    
    elif shape == "polygon":
        num_sides = bbox_params["num_sides"]
        radius = min(width, height) // 2
        points = get_polygon_points(center_x, center_y, radius, num_sides, rotation)
        cv2.polylines(image, [points], True, color, thickness)
    
    return image



def calibrate_empty_depth(pipe):
  """
  Calibrate the empty trash can depth by sampling the region defined by bounding box.
  Takes multiple samples and returns the median depth as baseline.
  """
  print("Starting depth calibration...")
  print("Please ensure the trash can is EMPTY")
  print("Collecting samples in 3 seconds...")
  time.sleep(3)
  
  depth_samples = []
  
  for i in range(CALIBRATION_SAMPLES):
    try:
      frames = pipe.wait_for_frames()
      depth_frame = frames.get_depth_frame()
      
      if depth_frame:
        # Apply filters for more stable calibration
        depth_frame_filtered = apply_depth_filters(depth_frame)
        
        # Get depth data as numpy array
        depth_image = np.asanyarray(depth_frame_filtered.get_data())
        
        # Create mask for the bounding box region
        height, width = depth_image.shape
        mask = create_region_mask(height, width, current_state["bounding_box"])
        
        # Apply mask to get only depths within the bounding box
        masked_depths = depth_image[mask > 0]
        
        # Filter out zero values (invalid depth readings)
        valid_depths = masked_depths[masked_depths > 0]
        
        if len(valid_depths) > 0:
          median_depth = np.median(valid_depths)
          depth_samples.append(median_depth)
          print(f"Sample {i+1}/{CALIBRATION_SAMPLES}: {median_depth:.1f}mm ({len(valid_depths)} valid pixels)")
        else:
          print(f"Sample {i+1}/{CALIBRATION_SAMPLES}: No valid depth data in bounding box")
      else:
        print(f"Sample {i+1}/{CALIBRATION_SAMPLES}: No depth frame received")
      
      time.sleep(0.1)  # Small delay between samples
    except Exception as e:
      print(f"Sample {i+1}/{CALIBRATION_SAMPLES}: Error - {e}")
      continue
  
  if len(depth_samples) > 0:
    # Use median of all samples for robustness
    empty_depth = np.median(depth_samples)
    print(f"\nCalibration complete!")
    print(f"Empty depth baseline: {empty_depth:.1f}mm ({empty_depth/10:.1f}cm)")
    
    # Save calibration to file
    calibration_data = {
      "empty_depth_mm": float(empty_depth),
      "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
      "samples_taken": len(depth_samples),
      "bounding_box": current_state["bounding_box"]
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
      json.dump(calibration_data, f, indent=2)
    
    print(f"Calibration saved to {CALIBRATION_FILE}")
    
    # Update global state
    current_state["calibration_date"] = calibration_data["calibration_date"]
    
    return empty_depth
  else:
    print("ERROR: No valid depth samples collected!")
    return None

def load_calibration():
  """Load previously saved calibration data."""
  if os.path.exists(CALIBRATION_FILE):
    with open(CALIBRATION_FILE, 'r') as f:
      data = json.load(f)
      print(f"Loaded calibration from {data['calibration_date']}")
      print(f"Empty depth baseline: {data['empty_depth_mm']:.1f}mm")
      current_state["calibration_date"] = data['calibration_date']
      
      # Load bounding box settings if available
      if "bounding_box" in data:
        current_state["bounding_box"] = data["bounding_box"]
        print(f"Loaded bounding box: {data['bounding_box']['shape']}")
      
      return data['empty_depth_mm']
  return None


def depth_socket_init(local_ip):
  """Initialize UDP socket for depth streaming."""
  global current_state
  try:
    serverSocket = socket.socket(AF_INET, SOCK_DGRAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind((local_ip, 5002))
    serverSocket.setblocking(False)  # Non-blocking to avoid hanging main loop
    current_state["depth_socket"] = serverSocket
    print(f"Depth socket initialized on {local_ip}:5002")
    
    # Start listener thread for client subscriptions
    listener_thread = threading.Thread(target=depth_socket_listener, daemon=True)
    listener_thread.start()
    
    return serverSocket
  except Exception as e:
    print(f"Error initializing depth socket: {e}")
    return None

def depth_socket_listener():
  """Listen for client subscription messages."""
  global current_state
  
  while True:
    try:
      if current_state["depth_socket"] is None:
        time.sleep(1)
        continue
      
      # Receive subscription messages from clients
      data, addr = current_state["depth_socket"].recvfrom(1024)
      message = data.decode('utf-8').strip()
      
      if message == "SUBSCRIBE":
        current_state["depth_clients"].add(addr)
        print(f"Client subscribed: {addr}")
        # Send acknowledgment
        current_state["depth_socket"].sendto(b"SUBSCRIBED", addr)
      elif message == "UNSUBSCRIBE":
        current_state["depth_clients"].discard(addr)
        print(f"Client unsubscribed: {addr}")
        
    except socket.error:
      # No data available (non-blocking socket)
      time.sleep(0.1)
    except Exception as e:
      print(f"Error in depth socket listener: {e}")
      time.sleep(1)

def send_depth():
  """Send current depth data to all subscribed clients via UDP socket."""
  global current_state
  
  if current_state["depth_socket"] is None or len(current_state["depth_clients"]) == 0:
    return
  
  try:
    data = {
      "current_depth": current_state.get("current_depth"),
      "fill_percentage": current_state.get("fill_percentage"),
      "empty_depth": current_state.get("empty_depth"),
      "timestamp": time.time()
    }
    
    # Convert to JSON and encode
    message = json.dumps(data).encode('utf-8')
    
    # Send to all subscribed clients
    disconnected_clients = set()
    for client_addr in current_state["depth_clients"]:
      try:
        current_state["depth_socket"].sendto(message, client_addr)
      except Exception as e:
        print(f"Error sending to {client_addr}: {e}")
        disconnected_clients.add(client_addr)
    
    # Remove disconnected clients
    current_state["depth_clients"] -= disconnected_clients
    
  except Exception as e:
    print(f"Error sending depth data: {e}")


if __name__ == "__main__":
  load_dotenv()
  pipe = rs.pipeline()
  config = rs.config()
  
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  
  pipe.start(config)
  print("RealSense pipeline started")
  
  # Store pipe in global state for API access
  current_state["pipe"] = pipe
  
  # Start Flask API server in background thread
  flask_thread = threading.Thread(target=run_flask, daemon=True)
  flask_thread.start()
  
  local_ip = get_local_ip()
  depth_socket_init(local_ip)
  print(f"Flask API server started on http://{local_ip}:5001")
  print(f"API accessible at http://{local_ip}:5001/status")
  
  # Check if we should recalibrate or load existing calibration
  recalibrate = os.getenv("RECALIBRATE", "false").lower() == "true"
  
  if recalibrate or not os.path.exists(CALIBRATION_FILE):
    empty_depth = calibrate_empty_depth(pipe)
  else:
    empty_depth = load_calibration()
    print(f"Calibration can be triggered via web interface at http://{local_ip}:5000")
  
  if empty_depth is None:
    print("WARNING: No calibration available. Use web interface to calibrate.")
  
  current_state["empty_depth"] = empty_depth
  
  server = MjpegServer("0.0.0.0", 8080)
  colorStream = Stream("color", size=(640, 480), quality=85, fps=30)
  depthStream = Stream("depth", size=(640, 480), quality=85, fps=30)
  depthRawStream = Stream("depth_raw", size=(640, 480), quality=85, fps=30)
  depthCroppedStream = Stream("depth_cropped", size=(300, 300), quality=85, fps=30)
  server.add_stream(colorStream)
  server.add_stream(depthStream)
  server.add_stream(depthRawStream)
  server.add_stream(depthCroppedStream)
  server.start()
  print(f"MJPEG server started on http://{local_ip}:8080")
  print(f"Color stream: http://{local_ip}:8080/color")
  print(f"Depth stream (filtered): http://{local_ip}:8080/depth")
  print(f"Depth raw (unfiltered): http://{local_ip}:8080/depth_raw")
  print(f"Depth cropped (bounding box only): http://{local_ip}:8080/depth_cropped")
  print(f"\nWeb interface available at: http://{local_ip}:5000")
  print(f"   Access from any device on your network")
  
  # Colorizer for depth visualization
  colorizer = rs.colorizer()
  
  try:
    while True:
      # Wait for frames
      frames = pipe.wait_for_frames()
      color_frame = frames.get_color_frame()
      depth_frame = frames.get_depth_frame()
      
      if color_frame:
        # Convert to numpy array (already in BGR format)
        color_image = np.asanyarray(color_frame.get_data())
        
        # Push frame to color stream
        colorStream.set_frame(color_image)
      
      # Process depth and create visualization
      if depth_frame:
        # Keep original for comparison
        depth_frame_raw = depth_frame
        
        # Apply post-processing filters
        depth_frame_filtered = apply_depth_filters(depth_frame)
        
        # Create colorized versions for both
        colorized_depth_raw = np.asanyarray(colorizer.colorize(depth_frame_raw).get_data())
        colorized_depth_filtered = np.asanyarray(colorizer.colorize(depth_frame_filtered).get_data())
        
        # Use filtered depth for main stream
        # Use filtered depth for main stream
        colorized_depth = colorized_depth_filtered.copy()
        
        # Get depth data from filtered frame
        depth_image = np.asanyarray(depth_frame_filtered.get_data())
        
        # Get image dimensions
        height, width = depth_image.shape
        
        # Draw bounding shape on visualizations
        draw_bounding_shape(colorized_depth, current_state["bounding_box"])
        draw_bounding_shape(colorized_depth_raw, current_state["bounding_box"])
        
        # Calculate fill percentage if calibrated
        fill_percentage = None
        current_depth = None
        
        # Use the current state's empty_depth which gets updated by calibration
        if current_state["empty_depth"] is not None:
          # Create mask for the bounding box region
          mask = create_region_mask(height, width, current_state["bounding_box"])
          
          # Apply mask to get only depths within the bounding box
          masked_depths = depth_image[mask > 0]
          valid_depths = masked_depths[masked_depths > 0]
          
          if len(valid_depths) > 0:
            raw_depth = np.median(valid_depths)
            
            # Apply temporal median filter to stabilize readings
            current_depth = apply_median_temporal_filter(raw_depth)
            
            fill_depth = current_state["empty_depth"] - current_depth
            fill_percentage = (fill_depth / current_state["empty_depth"]) * 100
            fill_percentage = max(0, min(100, fill_percentage))
            
            # Update global state
            current_state["current_depth"] = current_depth
            current_state["fill_percentage"] = fill_percentage
            
            # Send depth data to subscribed clients
            send_depth()
            
            # Add title and percentage text to main stream (filtered)
            title_text = "Trash Can Fill Level"
            percentage_text = f"{fill_percentage:.1f}% Full"
            depth_text = f"Depth: {current_depth:.0f}mm / {current_state['empty_depth']:.0f}mm"
            
            # Background rectangles for text
            cv2.rectangle(colorized_depth, (10, 10), (400, 100), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(colorized_depth, title_text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(colorized_depth, percentage_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(colorized_depth, depth_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add raw data to raw stream (for comparison page)
            raw_fill_depth = current_state["empty_depth"] - raw_depth
            raw_fill_percentage = (raw_fill_depth / current_state["empty_depth"]) * 100
            raw_fill_percentage = max(0, min(100, raw_fill_percentage))
            
            cv2.rectangle(colorized_depth_raw, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.putText(colorized_depth_raw, "Raw (Unfiltered)", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(colorized_depth_raw, f"{raw_fill_percentage:.1f}% Full", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(colorized_depth_raw, f"Depth: {raw_depth:.0f}mm / {current_state['empty_depth']:.0f}mm", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
          # No calibration message
          cv2.rectangle(colorized_depth, (10, 10), (350, 60), (0, 0, 0), -1)
          cv2.putText(colorized_depth, "NOT CALIBRATED", (20, 35), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          cv2.putText(colorized_depth, "Use web interface to calibrate", (20, 55), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
          
          # Same for raw
          cv2.rectangle(colorized_depth_raw, (10, 10), (350, 60), (0, 0, 0), -1)
          cv2.putText(colorized_depth_raw, "NOT CALIBRATED", (20, 35), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Create cropped view showing only the bounding box region
        cropped_depth = create_cropped_depth_view(colorized_depth_filtered, current_state["bounding_box"])
        
        # Push depth visualizations to streams (no inset)
        depthStream.set_frame(colorized_depth)
        depthRawStream.set_frame(colorized_depth_raw)
        depthCroppedStream.set_frame(cropped_depth)
              
  except KeyboardInterrupt:
    print("\nStopping stream...")
  finally:
    server.stop()
    pipe.stop()
    print("Stopped")
