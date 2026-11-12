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
from flask import Flask, request, jsonify, send_from_directory
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

# Initialize Subsystems
logger.info("Initializing Smart Rover Subsystems...")
db = RoverDatabase()
classifier = LeafDiseaseClassifier()
camera = RoverCamera()
rover = RoverController()
sensor_manager = SensorManager(ultrasonic=rover.obstacle_detector.sensor)
sprayer = SprayerController()
sync_service = CloudSyncService(db=db)

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
# System & Status Endpoints
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
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
        "camera": cam_status,
        "rover": rover_status,
        "sensors": sensor_status,
        "sprayer": sprayer_status,
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
    command = data.get("command", "").upper()
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
