#!/usr/bin/env bash
# ==============================================================================
# AgriRover AI - Automatic Desktop Camera Viewer
# Displays live camera feed with real-time green leaf detection bounding boxes
# automatically upon boot into Raspberry Pi OS desktop.
# ==============================================================================

# Wait up to 25 seconds for camera service to be ready
for i in $(seq 1 25); do
    if curl -s http://127.0.0.1:5000/api/status 2>/dev/null | grep -q '"camera_connected": true'; then
        break
    fi
    sleep 1
done

# Launch Chromium in lightweight app window
if command -v chromium >/dev/null 2>&1; then
    exec chromium --app="http://127.0.0.1:5000/api/camera/stream" \
                  --window-size=680,560 \
                  --window-position=40,40 \
                  --no-first-run \
                  --no-default-browser-check
elif command -v chromium-browser >/dev/null 2>&1; then
    exec chromium-browser --app="http://127.0.0.1:5000/api/camera/stream" \
                          --window-size=680,560 \
                          --window-position=40,40 \
                          --no-first-run \
                          --no-default-browser-check
fi
