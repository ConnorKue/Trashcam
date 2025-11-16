from flask import Flask, render_template, jsonify, request
import requests
import os
import socket

app = Flask(__name__)

# URL of the main server
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "http://localhost:5001")

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

@app.route('/')
def index():
    # Pass the server IP to the template
    server_ip = get_local_ip()
    return render_template('index.html', server_ip=server_ip)

@app.route('/touchscreen')
def touchscreen():
    # Dedicated touchscreen interface for 1024x600 display
    server_ip = get_local_ip()
    return render_template('touchscreen.html', server_ip=server_ip)

@app.route('/compare')
def compare():
    # Comparison page for raw vs filtered depth streams
    server_ip = get_local_ip()
    return render_template('compare.html', server_ip=server_ip)

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Trigger calibration on the main server."""
    try:
        response = requests.post(f"{MAIN_SERVER_URL}/calibrate", timeout=60)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get current calibration status and fill percentage."""
    try:
        response = requests.get(f"{MAIN_SERVER_URL}/status")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/bounding_box', methods=['POST'])
def update_bounding_box():
    """Update bounding box parameters."""
    try:
        data = request.get_json()
        response = requests.post(f"{MAIN_SERVER_URL}/bounding_box", json=data)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    local_ip = get_local_ip()
    print("Web Interface Server")
    print("   Desktop UI:     http://localhost:5000")
    print(f"   Desktop UI:     http://{local_ip}:5000")
    print(f"   Touchscreen UI: http://{local_ip}:5000/touchscreen")
    print(f"   Comparison:     http://{local_ip}:5000/compare")
    print("Access from any device on your network")
    app.run(host='0.0.0.0', port=5000, debug=True)
