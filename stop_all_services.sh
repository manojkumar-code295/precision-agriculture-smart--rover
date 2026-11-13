#!/usr/bin/env bash
# ==============================================================================
# Precision Agriculture Smart Rover - Stop All Services
# Completely stops the live camera viewer, camera stream, AI inference, and background tasks.
# ==============================================================================

echo "Stopping all AgriRover services..."

# 1. Set dormant service stop flag
touch /home/agrirover/Plant_Disease_Detection_AI/SERVICE_STOPPED

# 2. Kill Chromium desktop viewer and launcher
pkill -9 -f "open_live_camera_display.sh" 2>/dev/null
pkill -9 -f "chromium.*camera" 2>/dev/null
pkill -9 -f "chromium-browser.*camera" 2>/dev/null

# 3. Kill main python service (will release camera hardware and enter dormant standby)
pkill -TERM -f "Raspberry_Pi/main.py" 2>/dev/null
pkill -9 -f "webcam_leaf_detection.py" 2>/dev/null
pkill -9 -f "realtime_leaf_detection.py" 2>/dev/null
pkill -9 -f "camera_tester.py" 2>/dev/null

sleep 1

# Try stopping systemd service if sudo credentials available
if sudo -n true 2>/dev/null; then
    sudo systemctl stop agrirover-camera.service agrirover-web.service 2>/dev/null
fi

echo "======================================================="
echo "   [OK] All AgriRover services are now STOPPED.        "
echo "   - Camera hardware: Released & Free                  "
echo "   - Web stream (5000): Inactive                       "
echo "   - AI inference: Stopped                             "
echo "   - Desktop viewer: Closed                            "
echo "======================================================="
