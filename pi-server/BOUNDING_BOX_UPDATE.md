# 🎯 Bounding Box Controls Added!

## New Features

I've added comprehensive bounding box controls to the web interface that allow you to customize the monitored region in real-time!

### 🎨 What's New

#### Web Interface Controls Panel
A new "Bounding Box Controls" panel has been added below the progress bar with:

1. **Shape Selection**
   - Rectangle (with optional rotation)
   - Circle
   - Polygon (3-12 sides)

2. **Position Controls**
   - Center X: Horizontal position (0-100%)
   - Center Y: Vertical position (0-100%)

3. **Size Controls**
   - Width: Horizontal size (10-100%)
   - Height: Vertical size (10-100%)

4. **Rotation Control**
   - 0-360 degrees
   - Works with rectangles and polygons

5. **Polygon Sides**
   - 3-12 sides (only visible when polygon is selected)

### 🔧 Technical Implementation

**New Functions in `main.py`:**
- `get_rotated_rect_points()` - Calculates rotated rectangle corners
- `get_polygon_points()` - Generates regular polygon points
- `create_region_mask()` - Creates binary mask for any shape
- `draw_bounding_shape()` - Draws the shape on the video feed

**New API Endpoint:**
- `POST /bounding_box` - Updates bounding box parameters in real-time

**Updated Functions:**
- `calibrate_empty_depth()` - Now uses the custom bounding box region
- `load_calibration()` - Restores saved bounding box settings
- Main loop - Uses mask-based region detection instead of fixed rectangle

### 📊 How It Works

1. **Real-time Updates**: Adjust any slider/dropdown → shape updates immediately on the depth stream
2. **Debounced API Calls**: Changes are sent to the server after 100ms of inactivity (smooth performance)
3. **Mask-Based Detection**: Creates a binary mask of the selected region for accurate depth sampling
4. **Persistent Settings**: Bounding box configuration is saved with calibration data

### 🎮 Usage

1. Open web interface: http://localhost:5000
2. Scroll to "Bounding Box Controls" panel
3. Adjust shape, position, size, and rotation as needed
4. Watch the depth stream update in real-time
5. Click "Calibrate Empty" to save settings with calibration

### 🌟 Example Configurations

**Rectangle (Default):**
- Shape: Rectangle
- Center: 50%, 50%
- Size: 60%, 60%
- Rotation: 0°

**Circular Region:**
- Shape: Circle
- Center: 50%, 50%
- Size: 60%, 60%

**Hexagon:**
- Shape: Polygon
- Sides: 6
- Center: 50%, 50%
- Size: 60%, 60%
- Rotation: 0° (or any angle to rotate)

**Rotated Rectangle:**
- Shape: Rectangle
- Center: 50%, 50%
- Size: 80%, 40%
- Rotation: 45°

### 💡 Tips

- **Toggle Controls**: Click "Hide" button to collapse the panel when not needed
- **Live Preview**: Watch the depth stream to see your changes in real-time
- **Shape Visibility**: The green outline shows exactly what region is being monitored
- **Calibration**: Always calibrate after changing the bounding box for accurate measurements

All settings are automatically saved when you calibrate and restored on next startup!
