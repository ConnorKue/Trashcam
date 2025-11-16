#!/bin/bash
# Trashcam Launcher Script

echo "🗑️  Starting Trashcam System..."
echo ""

# Activate virtual environment
source ../trashcam/bin/activate

# Start main server in background
echo "Starting main server (camera & API)..."
python main.py &
MAIN_PID=$!

# Wait a bit for main server to initialize
sleep 3

# Start web interface
echo ""
echo "Starting web interface..."
echo ""
echo "========================================="
echo "The web interface URL will be displayed above"
echo "You can access it from any device on your network!"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

python web_interface.py

# When web interface stops, kill main server too
kill $MAIN_PID 2>/dev/null
echo "Stopped all servers"
