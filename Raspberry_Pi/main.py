"""
Precision Agriculture Smart Rover - Raspberry Pi Core Service
Main Flask REST API service integrating Camera, Edge AI Inference,
Motor Control, Ultrasonic Obstacle Detection, Sprayer, Database, and Cloud Sync.
"""

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import config
from database import RoverDatabase
from inference import LeafDiseaseClassifier
from camera import RoverCamera
from rover_control import RoverController
from sensor_manager import SensorManager
from sprayer_control import SprayerController
from sync_service import CloudSyncService

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.RASPBERRY_PI_DIR / "rover_service.log")
    ]
)
logger = logging.getLogger("RoverMain")

# Flask App
app = Flask(__name__)
CORS(app)

# Check Stop Flag before initializing hardware
STOP_FLAG = config.PROJECT_ROOT / "SERVICE_STOPPED"
if STOP_FLAG.exists():
    logger.info("Service STOP flag detected (%s). Holding in dormant standby...", STOP_FLAG)
    while STOP_FLAG.exists():
        time.sleep(1.0)
    logger.info("Service STOP flag removed. Proceeding with subsystem initialization...")

# Initialize Subsystems
logger.info("Initializing Smart Rover Subsystems...")
db = RoverDatabase()
classifier = LeafDiseaseClassifier()

def handle_auto_capture(image_path: Path):
    """Callback for automated capture when a chili leaf stabilizes in camera view."""
    try:
        logger.info(f"Running automated leaf classification on: {image_path.name}")
        det_result = classifier.predict(image_path)
        # Only record genuine confirmed plant leaf detections into the database
        if not det_result.get("leaf_detected", False):
            logger.info(f"Non-leaf image suppressed from database: {det_result.get('display_label')} ({image_path.name})")
            return

        det_id = db.insert_detection({
            "detection_id": det_result["detection_id"],
            "timestamp": det_result["timestamp"],
            "image_filename": det_result["image_filename"],
            "image_path": str(image_path),
            "predicted_disease": det_result["predicted_disease"],
            "confidence": det_result["confidence"],
            "is_low_confidence": det_result["is_low_confidence"],
            "recommendation": det_result["recommendation"],
            "sprayer_status": "OFF"
        })
        logger.info(f"Auto-capture verified leaf: {det_result['predicted_disease']} ({det_result['confidence']}%) [DB ID: {det_id}]")
    except Exception as e:
        logger.error(f"Auto-capture classification failed: {e}")

camera = RoverCamera(on_auto_capture=handle_auto_capture)
rover = RoverController()
sensor_manager = SensorManager(ultrasonic=rover.obstacle_detector.sensor)
sprayer = SprayerController()
sync_service = CloudSyncService(
    db=db,
    rover=rover,
    camera=camera,
    sprayer=sprayer,
    sensor_manager=sensor_manager
)

# Start background sync
sync_service.start()

# Background Sensor Logger
def sensor_telemetry_loop():
    while True:
        try:
            sensors_data = sensor_manager.get_all_sensors()
            ultra = sensors_data["sensors"]["ultrasonic_distance"]["reading"]
            obs_detected = rover.obstacle_detector.check_obstacle()["obstacle_detected"]
            db.insert_sensor_log({
                "ultrasonic_distance": ultra,
                "obstacle_detected": obs_detected,
                "soil_moisture": None,
                "temperature": None,
                "humidity": None
            })
        except Exception as e:
            logger.error(f"Sensor logging error: {e}")
        time.sleep(10.0)

sensor_thread = threading.Thread(target=sensor_telemetry_loop, daemon=True)
sensor_thread.start()


# -------------------------------------------------------------
# Static Media Route
# -------------------------------------------------------------
@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(config.CAPTURES_DIR, filename)


# -------------------------------------------------------------
# Frontend Static Asset Path & Single-Page Application Routes
# -------------------------------------------------------------
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Website", "backend", "dist_client"))


@app.route("/", methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def index(path=""):
    if path.startswith("api/") or path.startswith("captures/") or path in ("video_feed", "camera", "live"):
        return jsonify({"error": "Endpoint not found"}), 404
    if FRONTEND_DIST and os.path.exists(FRONTEND_DIST):
        target_file = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(target_file):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "service": "Plant Disease Detection AI - Smart Rover Core",
        "device_id": config.DEVICE_ID,
        "device_name": config.DEVICE_NAME,
        "status": "Online",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/health", methods=["GET"])
def health():
    import shutil
    total, used, free = shutil.disk_usage("/")
    return jsonify({
        "status": "OK",
        "device_id": config.DEVICE_ID,
        "disk_free_gb": round(free / (1024**3), 2),
        "disk_total_gb": round(total / (1024**3), 2),
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/status", methods=["GET"])
def system_status():
    cam_status = camera.get_status()
    rover_status = rover.get_status()
    sensor_status = sensor_manager.get_all_sensors()
    sprayer_status = sprayer.get_status()
    sync_status = sync_service.get_status()
    latest_det = db.get_latest_detection()
    stats = db.get_detection_statistics()

    return jsonify({
        "device_id": config.DEVICE_ID,
        "device_name": config.DEVICE_NAME,
        "pi_online": True,
        "pi_status": "ONLINE",
        "camera_status": cam_status.get("status", "STREAMING" if cam_status.get("camera_connected") else "INTERRUPTED"),
        "ai_status": "RUNNING",
        "fps": cam_status.get("live_fps", 0.0),
        "camera": cam_status,
        "ai": {
            "status": "RUNNING",
            "model_version": "2.1.0-Edge",
            "architecture": "MobileNetV2 CNN",
            "input_shape": "160x160x3",
            "classes": config.CLASS_NAMES
        },
        "rover": rover_status,
        "sensors": sensor_status,
        "sprayer": sprayer_status,
        "battery": {
            "percentage": 94,
            "voltage": 12.4,
            "status": "GOOD"
        },
        "sync": sync_status,
        "latest_detection": latest_det,
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    })


# -------------------------------------------------------------
# Camera & Disease Detection Endpoints
# -------------------------------------------------------------
@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    return jsonify(camera.get_status())


@app.route("/api/camera/live_frame", methods=["GET"])
def camera_live_frame():
    """Returns single latest frame as JPEG."""
    frame_bytes = camera.get_jpeg_frame()
    if frame_bytes is not None:
        return Response(
            frame_bytes,
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return Response(b"", status=204)


@app.route("/api/camera/stream", methods=["GET"])
@app.route("/video_feed", methods=["GET"])
def camera_stream():
    """Streams real-time MJPEG live video feed with active green leaf detection bounding-box overlays."""
    def generate():
        while True:
            try:
                frame_bytes = camera.get_jpeg_frame()
                if frame_bytes is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                    time.sleep(0.08)
                else:
                    time.sleep(0.05)
            except (GeneratorExit, BrokenPipeError):
                break
            except Exception:
                time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
        }
    )


@app.route("/camera", methods=["GET"])
@app.route("/live", methods=["GET"])
def live_camera_page():
    """Serves high-reliability live camera monitor with auto-reconnection and diagnostics."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriRover AI - Live Camera & Keras Leaf Detection</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f1715; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header { background: #13221c; padding: 10px 18px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f3d30; }
        .title { font-size: 16px; font-weight: 700; color: #34d399; display: flex; align-items: center; gap: 8px; }
        .badges { display: flex; gap: 8px; align-items: center; }
        .badge { font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 9999px; }
        .badge-live { background: #065f46; color: #a7f3d0; border: 1px solid #059669; }
        .badge-model { background: #1e3a5f; color: #bfdbfe; border: 1px solid #3b82f6; }
        .main-container { flex: 1; display: flex; justify-content: center; align-items: center; padding: 12px; position: relative; background: #09100e; }
        .video-box { position: relative; width: 100%; max-width: 900px; max-height: 90vh; aspect-ratio: 4/3; background: #000; border-radius: 8px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.6); border: 2px solid #1f3d30; display: flex; justify-content: center; align-items: center; }
        #stream-img { width: 100%; height: 100%; object-fit: contain; display: block; }
        #placeholder { position: absolute; display: flex; flex-direction: column; align-items: center; gap: 12px; color: #94a3b8; }
        .spinner { width: 44px; height: 44px; border: 4px solid #1e3a2e; border-top-color: #34d399; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        footer { background: #13221c; padding: 8px 18px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1f3d30; font-size: 12px; color: #94a3b8; }
        .btn { background: #059669; color: white; border: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn:hover { background: #10b981; }
        .btn-reload { background: #334155; }
        .btn-reload:hover { background: #475569; }
    </style>
</head>
<body>
    <header>
        <div class="title">
            <span>🌿</span> AgriRover AI – Real-Time Chili Leaf Detection
        </div>
        <div class="badges">
            <span class="badge badge-model">Keras MobileNetV2</span>
            <span class="badge badge-live" id="cam-status-badge">Connecting...</span>
        </div>
    </header>

    <div class="main-container">
        <div class="video-box">
            <div id="placeholder">
                <div class="spinner"></div>
                <p id="placeholder-text">Connecting to Live Camera Stream...</p>
            </div>
            <img id="stream-img" src="" alt="Live Stream" style="display: none;">
        </div>
    </div>

    <footer>
        <div id="telemetry">Telemetry: Initializing sensor and AI monitor...</div>
        <div style="display: flex; gap: 8px;">
            <button class="btn btn-reload" onclick="forceReconnect()">↻ Reload Stream</button>
            <button class="btn" onclick="captureManual()">📸 Capture Leaf</button>
        </div>
    </footer>

    <script>
        const img = document.getElementById("stream-img");
        const placeholder = document.getElementById("placeholder");
        const placeholderText = document.getElementById("placeholder-text");
        const badge = document.getElementById("cam-status-badge");
        const telemetry = document.getElementById("telemetry");

        let reconnectTimer = null;
        let isConnected = false;

        function connectStream() {
            img.src = "/api/camera/stream?t=" + Date.now();
        }

        img.onload = () => {
            isConnected = true;
            placeholder.style.display = "none";
            img.style.display = "block";
            badge.textContent = "ONLINE (STREAMING)";
            badge.style.background = "#065f46";
            badge.style.borderColor = "#059669";
        };

        img.onerror = () => {
            isConnected = false;
            placeholder.style.display = "flex";
            img.style.display = "none";
            placeholderText.textContent = "Camera Stream Interrupted. Reconnecting...";
            badge.textContent = "RECONNECTING...";
            badge.style.background = "#854d0e";
            badge.style.borderColor = "#ca8a04";

            clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(connectStream, 1500);
        };

        function forceReconnect() {
            placeholder.style.display = "flex";
            placeholderText.textContent = "Refreshing Camera Stream...";
            connectStream();
        }

        function captureManual() {
            fetch("/api/camera/capture", { method: "POST" })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert("Snapshot captured: " + (data.detection?.predicted_disease || "Saved"));
                    } else {
                        alert("Capture error: " + (data.error || "Failed"));
                    }
                })
                .catch(err => alert("Request error: " + err));
        }

        // Live Telemetry Poller
        setInterval(() => {
            fetch("/api/camera/status")
                .then(r => r.json())
                .then(data => {
                    if (data.camera_connected && data.status === "STREAMING") {
                        telemetry.textContent = `FPS: ${data.live_fps || 0.0} | Leaves in View: ${data.detected_leaves_count || 0} | Sensor: ${data.camera_type || 'RPi CSI'} | Leaf Guard: Active`;
                        if (!isConnected && img.style.display === "none") {
                            connectStream();
                        }
                    } else {
                        telemetry.textContent = `Camera Status: ${data.status || 'Offline'} (${data.error_message || 'Waiting'})`;
                    }
                })
                .catch(() => {
                    telemetry.textContent = "Rover Service Offline. Waiting for service restart...";
                });
        }, 1500);

        // Initial connection
        connectStream();
    </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.route("/api/camera/auto-detect", methods=["GET", "POST"])
def auto_detect_control():
    """Controls or queries automated chili leaf detection and capture."""
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        if "enabled" in data:
            camera.set_auto_detect(bool(data["enabled"]))
        if "interval" in data:
            camera.set_auto_interval(float(data["interval"]))
        return jsonify({
            "success": True,
            "auto_detect_enabled": camera.auto_detect_enabled,
            "auto_interval_sec": camera.auto_interval_sec
        })
    return jsonify({
        "auto_detect_enabled": camera.auto_detect_enabled,
        "auto_interval_sec": camera.auto_interval_sec
    })


@app.route("/api/camera/capture", methods=["POST"])
def capture_leaf():
    """Captures a leaf image from the camera and immediately classifies it."""
    auto_classify = request.args.get("classify", "true").lower() in ("1", "true", "yes")

    success, path, msg = camera.capture_image()
    if not success or not path:
        return jsonify({"success": False, "error": msg}), 500

    image_url = f"/captures/{path.name}"
    response_data = {
        "success": True,
        "image_filename": path.name,
        "image_url": image_url,
        "message": msg
    }

    if auto_classify:
        try:
            det_result = classifier.predict(path)
            # Store in local database
            det_id = db.insert_detection({
                "detection_id": det_result["detection_id"],
                "timestamp": det_result["timestamp"],
                "image_filename": det_result["image_filename"],
                "image_path": str(path),
                "predicted_disease": det_result["predicted_disease"],
                "confidence": det_result["confidence"],
                "is_low_confidence": det_result["is_low_confidence"],
                "recommendation": det_result["recommendation"],
                "sprayer_status": "OFF"
            })
            det_result["db_id"] = det_id
            det_result["image_url"] = image_url
            response_data["detection"] = det_result
        except Exception as e:
            logger.error(f"Classification failed on capture: {e}")
            response_data["classification_error"] = str(e)

    return jsonify(response_data)


@app.route("/api/ai/classify", methods=["POST"])
def classify_uploaded_leaf():
    """Classifies an uploaded leaf image."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    save_path = config.CAPTURES_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file.save(str(save_path))

    try:
        det_result = classifier.predict(save_path)
        det_id = db.insert_detection({
            "detection_id": det_result["detection_id"],
            "timestamp": det_result["timestamp"],
            "image_filename": save_path.name,
            "image_path": str(save_path),
            "predicted_disease": det_result["predicted_disease"],
            "confidence": det_result["confidence"],
            "is_low_confidence": det_result["is_low_confidence"],
            "recommendation": det_result["recommendation"],
            "sprayer_status": "OFF"
        })
        det_result["db_id"] = det_id
        det_result["image_url"] = f"/captures/{save_path.name}"
        return jsonify({"success": True, "detection": det_result})
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/detections/latest", methods=["GET"])
def latest_detection():
    det = db.get_latest_detection()
    if det:
        det["image_url"] = f"/captures/{det['image_filename']}"
        return jsonify({"success": True, "detection": det})
    return jsonify({"success": True, "detection": None, "message": "No detections recorded yet."})


@app.route("/api/detections", methods=["GET"])
def detection_history():
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    disease = request.args.get("disease", None)
    min_conf = request.args.get("min_confidence", None)
    min_conf_val = float(min_conf) if min_conf else None

    records = db.get_detections(limit=limit, offset=offset, disease=disease, min_confidence=min_conf_val)
    for r in records:
        r["image_url"] = f"/captures/{r['image_filename']}"

    stats = db.get_detection_statistics()
    return jsonify({
        "success": True,
        "count": len(records),
        "statistics": stats,
        "detections": records
    })


# -------------------------------------------------------------
# Rover Motor & Obstacle Endpoints
# -------------------------------------------------------------
@app.route("/api/rover/move", methods=["POST"])
def rover_move():
    data = request.get_json(force=True, silent=True) or {}
    command = str(data.get("command") or data.get("direction") or "").upper()
    speed = data.get("speed", config.MOTOR_DEFAULT_SPEED)

    result = rover.execute_command(command, speed=int(speed))
    db.insert_rover_event(
        command=command,
        speed=speed,
        mode=rover.mode,
        obstacle_stopped=(not result.get("success", False) and result.get("reason") == "OBSTACLE_DETECTED"),
        details=result.get("message", "")
    )
    return jsonify(result)


@app.route("/api/rover/stop", methods=["POST"])
def rover_stop():
    rover.stop()
    db.insert_rover_event("STOP", 0, rover.mode, False, "User stop requested")
    return jsonify({"success": True, "state": "STOPPED", "message": "Rover stopped."})


@app.route("/api/rover/emergency-stop", methods=["POST"])
def rover_emergency_stop():
    result = rover.emergency_stop()
    db.insert_rover_event("EMERGENCY_STOP", 0, rover.mode, False, "Emergency stop triggered")
    return jsonify(result)


@app.route("/api/rover/mode", methods=["POST"])
def set_rover_mode():
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "MANUAL")
    res = rover.set_mode(mode)
    return jsonify(res)


# -------------------------------------------------------------
# Sensors & Sprayer Endpoints
# -------------------------------------------------------------
@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    return jsonify(sensor_manager.get_all_sensors())


@app.route("/api/sprayer/activate", methods=["POST"])
def activate_sprayer():
    data = request.get_json(force=True, silent=True) or {}
    duration = float(data.get("duration", config.DEFAULT_SPRAY_DURATION_SEC))
    confirmed = bool(data.get("confirmed", False))
    disease_context = data.get("disease_context", None)

    result = sprayer.activate_spray(duration_sec=duration, user_confirmed=confirmed, disease_context=disease_context)
    if result["success"]:
        db.insert_sprayer_event("ON", duration, str(disease_context), confirmed)
    return jsonify(result)


@app.route("/api/sprayer/stop", methods=["POST"])
def stop_sprayer():
    sprayer.stop_spray()
    db.insert_sprayer_event("OFF", 0.0, "Manual Stop", True)
    return jsonify({"success": True, "sprayer_active": False, "message": "Sprayer pump stopped."})


@app.route("/api/sync/trigger", methods=["POST"])
def trigger_sync():
    sync_res = sync_service.perform_sync()
    return jsonify(sync_res)


# -------------------------------------------------------------
# Signal Handling & Safe Shutdown
# -------------------------------------------------------------
def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received. Stopping hardware devices safely...")
    rover.shutdown()
    sprayer.cleanup()
    camera.release()
    sync_service.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if __name__ == "__main__":
    logger.info(f"Starting Smart Rover Core API on {config.LOCAL_API_HOST}:{config.LOCAL_API_PORT}...")
    app.run(host=config.LOCAL_API_HOST, port=config.LOCAL_API_PORT, debug=False, threaded=True)
