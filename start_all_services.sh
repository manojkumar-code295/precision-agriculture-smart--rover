#!/usr/bin/env bash
# ==============================================================================
# Precision Agriculture Smart Rover - Start All Services
# Restarts the camera service, AI inference, and desktop live camera viewer.
# ==============================================================================

echo "Starting all AgriRover services..."

# 1. Remove stop flag
rm -f /home/agrirover/Plant_Disease_Detection_AI/SERVICE_STOPPED

# 2. Wake up main service
pkill -TERM -f "Raspberry_Pi/main.py" 2>/dev/null

# If sudo available, start systemd
if sudo -n true 2>/dev/null; then
    sudo systemctl restart agrirover-camera.service 2>/dev/null
fi

# 3. Launch live camera viewer window on desktop
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
nohup /home/agrirover/Plant_Disease_Detection_AI/open_live_camera_display.sh > /tmp/camera_viewer.log 2>&1 &

echo "======================================================="
echo "   [OK] AgriRover services are now STARTING.          "
echo "   - Camera service initializing...                    "
echo "   - Desktop camera viewer will open automatically.   "
echo "======================================================="
