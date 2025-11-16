# Trashcam - Smart Trash Can Monitor

## Setup

1. Install dependencies:
```bash
cd pi-server
pip install -r requirements.txt
```

## Running the Application

### Option 1: Two-Process Setup (Recommended)

Run both servers in separate terminals:

**Terminal 1 - Main Server (Camera & API):**
```bash
python main.py
```

**Terminal 2 - Web Interface:**
```bash
python web_interface.py
```

The servers will display their network IP addresses. You can access the web interface from:
- **Same machine**: http://localhost:5000
- **Other devices on network**: http://YOUR_IP:5000 (displayed when server starts)

### Option 2: Main Server Only

```bash
python main.py
```

Access streams directly:
- Color stream: http://YOUR_IP:8080/color
- Depth stream: http://YOUR_IP:8080/depth
- API status: http://YOUR_IP:5001/status

(Replace YOUR_IP with the IP address displayed when the server starts)

## Features

### Web Interface (http://localhost:5000)
- **Live Streams**: View both color and depth analysis side-by-side
- **Calibration Button**: Click to calibrate the empty trash can depth
- **Real-time Status**: See fill percentage, current depth, and calibration date
- **Visual Progress Bar**: Color-coded fill level indicator
- **Bounding Box Controls**: Adjust the monitored region:
  - **Shape**: Rectangle, Circle, or Polygon
  - **Position**: Center X/Y sliders
  - **Size**: Width and Height sliders
  - **Rotation**: 0-360 degrees
  - **Polygon Sides**: 3-12 sides (when polygon selected)

### Depth Analysis Stream
- Bounding shape (rectangle/circle/polygon) shows the monitored region
- Adjustable via web interface controls
- Title: "Trash Can Fill Level"
- Fill percentage displayed in real-time
- Current depth vs empty baseline shown
- Changes color and shape based on your settings

### Calibration
- Click "Calibrate Empty" button in web interface
- Ensure trash can is completely empty
- System takes 30 samples over 3 seconds
- Calibration saved to `depth_calibration.json`
- Persists across restarts

## API Endpoints

### POST /calibrate
Trigger calibration process.

**Response:**
```json
{
  "success": true,
  "empty_depth": 450.5,
  "message": "Calibration successful"
}
```

### GET /status
Get current system status.

**Response:**
```json
{
  "success": true,
  "empty_depth": 450.5,
  "current_depth": 350.2,
  "fill_percentage": 22.3,
  "calibration_date": "2025-11-15 14:30:00",
  "calibrating": false
}
```

## How It Works

1. **Camera Positioning**: RealSense camera looks straight down into trash can
2. **Region of Interest**: Central 60% of frame monitored (avoids edges)
3. **Empty Calibration**: Measures depth to bottom when empty
4. **Fill Detection**: As trash fills, depth decreases
5. **Percentage Calculation**: `fill% = (empty_depth - current_depth) / empty_depth × 100`

## Ports

- **5000**: Web interface (Flask) - accessible from any device on your network
- **5001**: API server (Flask) - accessible from any device on your network
- **8080**: MJPEG streams (color & depth) - accessible from any device on your network

All servers bind to `0.0.0.0` and display their network IP address on startup.

## Environment Variables

Create a `.env` file:

```bash
# Force recalibration on startup
RECALIBRATE=false

# Main server URL (for web interface)
MAIN_SERVER_URL=http://localhost:5001
```
