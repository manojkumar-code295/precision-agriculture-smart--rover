#!/usr/bin/env bash
# ==============================================================================
# AgriRover AI - Resilient Desktop Camera & Leaf Detection Viewer
# Automatically launches and maintains live camera HUD window upon desktop login.
# ==============================================================================

export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

API_STATUS_URL="http://127.0.0.1:5000/api/status"
VIEWER_URL="http://127.0.0.1:5000/camera"

echo "[AgriRover Autostart] Waiting for Core Camera Service..."

# Resilient polling loop: wait up to 120 seconds during cold boot
MAX_WAIT_SEC=120
for i in $(seq 1 $MAX_WAIT_SEC); do
    if curl -s "$API_STATUS_URL" 2>/dev/null | grep -qE '"camera_connected":\s*true|"status":\s*"STREAMING"'; then
        echo "[AgriRover Autostart] Camera Service verified ready in ${i}s."
        break
    fi
    sleep 1
done

# Small stabilization delay
sleep 1

# Detect browser binary
BROWSER_BIN=""
if command -v chromium >/dev/null 2>&1; then
    BROWSER_BIN="chromium"
elif command -v chromium-browser >/dev/null 2>&1; then
    BROWSER_BIN="chromium-browser"
fi

if [ -z "$BROWSER_BIN" ]; then
    echo "[AgriRover Autostart] Chromium not found, falling back to python realtime detector."
    exec /home/agrirover/Plant_Disease_Detection_AI/venv/bin/python /home/agrirover/realtime_leaf_detection.py
fi

echo "[AgriRover Autostart] Launching Live Camera Viewer via $BROWSER_BIN..."

# Continuous supervisor loop: ensures the camera window stays alive
while true; do
    "$BROWSER_BIN" \
        --app="$VIEWER_URL" \
        --window-size=960,720 \
        --window-position=50,50 \
        --no-first-run \
        --no-default-browser-check \
        --disable-session-crashed-bubble \
        --disable-infobars \
        --check-for-update-interval=31536000 \
        --autoplay-policy=no-user-gesture-required

    EXIT_CODE=$?
    echo "[AgriRover Autostart] Camera window exited (code: $EXIT_CODE). Restarting in 2 seconds..."
    sleep 2
done
