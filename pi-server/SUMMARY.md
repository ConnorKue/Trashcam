# 🎉 Trashcam System Complete!

I've successfully implemented a complete trash can monitoring system with a web interface and calibration button!

## 📁 Files Created/Modified

### New Files:
1. **`web_interface.py`** - Flask web application with UI
2. **`templates/index.html`** - Beautiful web interface with live streams
3. **`requirements.txt`** - Python dependencies
4. **`README.md`** - Complete documentation
5. **`start.sh`** - Convenient launcher script

### Modified Files:
1. **`main.py`** - Enhanced with:
   - Second MJPEG stream for depth visualization
   - Flask API endpoints for calibration
   - CV2 bounding box and text overlay
   - Real-time fill percentage display

## ✨ Features Implemented

### 1. **Web Interface** (http://localhost:5000)
- Clean, modern UI with gradient background
- Live view of BOTH streams side-by-side:
  - Color stream (left)
  - Depth analysis stream (right)
- Real-time status display:
  - Fill percentage (large, prominent)
  - Empty baseline depth
  - Current depth
  - Calibration date
- Color-coded progress bar (green → yellow → red)
- **Big Calibration Button** - Click to calibrate!

### 2. **Depth Visualization Stream**
- Green bounding box showing monitored region (central 60%)
- Text overlay with:
  - "Trash Can Fill Level" title
  - Percentage display (e.g., "45.2% Full")
  - Depth comparison (e.g., "Depth: 350mm / 450mm")
- Black background boxes for readable text
- Colorized depth map for easy visualization

### 3. **Calibration System**
- Click button in web interface
- Automatic 30-sample calibration over 3 seconds
- Saves to `depth_calibration.json`
- Real-time feedback during calibration
- Persists across restarts

### 4. **API Endpoints**
- `POST /calibrate` - Trigger calibration remotely
- `GET /status` - Get current stats (fill %, depths, etc.)

## 🚀 How to Run

### Easy Method:
```bash
cd pi-server
./start.sh
```
Then open browser to: **http://localhost:5000**

### Manual Method:
Terminal 1:
```bash
cd pi-server
source ../trashcam/bin/activate
python main.py
```

Terminal 2:
```bash
cd pi-server
source ../trashcam/bin/activate
python web_interface.py
```

## 🎯 How It Works

1. **Camera Setup**: RealSense camera looks down into trash can
2. **Bounding Box**: Green box shows central 60% region being monitored
3. **Empty Calibration**: Click button → system measures empty depth
4. **Real-time Monitoring**: 
   - As trash fills up, depth decreases
   - Percentage calculated: `(empty - current) / empty × 100`
5. **Visual Feedback**: 
   - Progress bar changes color
   - Depth stream shows percentage overlay
   - Web UI updates every 2 seconds

## 🎨 UI Features

- Responsive design (works on phones/tablets)
- Auto-refreshing streams
- Success/Error message notifications
- Disabled button during calibration
- Beautiful gradient purple background
- Clean card-based layout

## 📊 Streams Available

1. **Color Stream**: http://localhost:8080/color
2. **Depth Stream**: http://localhost:8080/depth (with overlay)
3. **Web Interface**: http://localhost:5000 (both streams + controls)

## 🔧 Technical Details

- Uses OpenCV (cv2) for drawing bounding box and text
- RealSense colorizer for depth visualization
- Flask runs in background thread (doesn't block camera)
- MJPEG streams run on separate server
- Global state synchronization for real-time updates

Enjoy your smart trash can monitoring system! 🗑️✨
