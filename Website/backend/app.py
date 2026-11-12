"""
Precision Agriculture Smart Rover - Cloud API Service (Render Web Service)
Central cloud REST API for receiving telemetry from Raspberry Pi, persisting detection
records, generating analytics, and serving the Angular web dashboard.
"""

import os
import sys
import sqlite3
import csv
import io
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [CloudAPI]: %(message)s"
)
logger = logging.getLogger("CloudAPI")

app = Flask(__name__)
# Enable CORS for all origins so Angular frontend on Render Static Site can connect seamlessly
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
DB_PATH = os.getenv("CLOUD_DB_PATH", "cloud_rover.db")
API_KEY = os.getenv("ROVER_API_KEY", "smartrover_secure_key_2026")
PI_HEARTBEAT_TIMEOUT_SEC = int(os.getenv("PI_HEARTBEAT_TIMEOUT_SEC", "60"))

# In-memory heartbeat cache for live status
LIVE_STATUS = {
    "last_seen": None,
    "pi_online": False,
    "camera": {
        "camera_connected": False,
        "camera_type": "Awaiting Telemetry",
        "last_capture": None,
        "error_message": None
    },
    "rover": {
        "mode": "MANUAL",
        "is_moving": False,
        "motor_status": {"state": "STOPPED", "speed": 0, "hardware_enabled": False},
        "obstacle_status": {"obstacle_detected": False, "distance_cm": None, "hazard_level": "CLEAR"}
    },
    "sensors": {
        "ultrasonic_distance": {"name": "HC-SR04", "reading": None, "status": "Awaiting Telemetry", "unit": "cm"},
        "soil_moisture": {"name": "Soil Moisture", "reading": None, "status": "Not Installed", "unit": "%"},
        "ambient_environment": {"name": "Temp/Humidity", "reading": {"temperature": None, "humidity": None}, "status": "Not Installed"}
    },
    "sprayer": {
        "sprayer_active": False,
        "hardware_enabled": False,
        "last_activated": None
    }
}

# Pending remote commands queue
PENDING_COMMANDS = []

# -------------------------------------------------------------
# Database Setup
# -------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cloud_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id TEXT UNIQUE NOT NULL,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                image_url TEXT,
                predicted_disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_low_confidence INTEGER NOT NULL DEFAULT 0,
                recommendation TEXT NOT NULL,
                sprayer_status TEXT DEFAULT 'OFF',
                synced_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cloud_sensors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ultrasonic_distance REAL,
                obstacle_detected INTEGER NOT NULL DEFAULT 0,
                soil_moisture REAL,
                temperature REAL,
                humidity REAL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cloud_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

# -------------------------------------------------------------
# Agricultural Disease Knowledge Base
# -------------------------------------------------------------
DISEASE_ENCYCLOPEDIA = [
    {
        "id": "bacterial_spot",
        "name": "Bacterial Spot",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "category": "Bacterial Disease",
        "severity": "High",
        "symptoms": [
            "Small water-soaked spots on leaves that turn dark brown to black.",
            "Lesions often have yellow halos around necrotic margins.",
            "Severe infections trigger massive defoliation, exposing fruit to sunscald."
        ],
        "causes": [
            "Warm, humid rainy weather (24°C - 30°C).",
            "Overhead sprinkler irrigation spreading bacteria via splash.",
            "Contaminated seeds or crop residues."
        ],
        "prevention": [
            "Use certified disease-free seeds and resistant cultivars.",
            "Practice minimum 2-year crop rotation with non-solanaceous crops.",
            "Sanitize pruning tools and tractor/rover equipment."
        ],
        "chemical_treatment": {
            "spray_name": "Copper Oxychloride 50% WP + Streptocycline",
            "dosage": "2.5 g Copper Oxychloride per liter + 1 g Streptocycline per 10 liters of water",
            "application_interval": "Spray at first symptom onset; repeat after 7-10 days if rainy conditions persist.",
            "pre_harvest_interval": "7 days"
        },
        "biological_treatment": "Apply Bacillus subtilis or Pseudomonas fluorescens formulations @ 5 g/L to roots and foliage."
    },
    {
        "id": "cercospora_leaf_spot",
        "name": "Cercospora Leaf Spot (Frogeye)",
        "scientific_name": "Cercospora capsici",
        "category": "Fungal Disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Circular to oval lesions with light gray or white centers.",
            "Dark brown, slightly raised margins giving characteristic 'frogeye' appearance.",
            "Leaves turn chlorotic (yellow) and drop prematurely."
        ],
        "causes": [
            "High relative humidity (>80%) and warm temperatures (25°C - 32°C).",
            "Windblown fungal conidia and water splash."
        ],
        "prevention": [
            "Widen plant spacing to facilitate rapid canopy drying and air circulation.",
            "Avoid overhead irrigation during evening hours.",
            "Collect and burn infected plant debris immediately."
        ],
        "chemical_treatment": {
            "spray_name": "Mancozeb 75% WP or Chlorothalonil 75% WP",
            "dosage": "2.5 g Mancozeb OR 2.0 g Chlorothalonil per liter of water. Alternatively Azoxystrobin 23% SC @ 1.0 ml/L.",
            "application_interval": "10-14 days during periods of high humidity.",
            "pre_harvest_interval": "10 days"
        },
        "biological_treatment": "Trichoderma harzianum or Trichoderma viride foliar spray @ 4 g/L."
    },
    {
        "id": "curl_virus",
        "name": "Chili Leaf Curl Virus (ChiLCV)",
        "scientific_name": "Begomovirus (Family Geminiviridae)",
        "category": "Viral Disease",
        "severity": "Severe / Critical",
        "symptoms": [
            "Upward and downward curling and crinkling of leaves.",
            "Thickened veins, puckering, and blister-like leaf surface.",
            "Stunted plant stature and severe reduction in flower bud development."
        ],
        "causes": [
            "Transmission exclusively by Whitefly (Bemisia tabaci) insect vector.",
            "Not transmitted by mechanical contact or soil."
        ],
        "prevention": [
            "Install 15-20 yellow sticky traps per acre to monitor and trap whiteflies.",
            "Grow barrier crops like maize, sorghum, or pearl millet around chili fields.",
            "IMMEDIATELY rogue (uproot and burn) symptomatic plants to prevent vector feeding."
        ],
        "chemical_treatment": {
            "spray_name": "Vector Control: Diafenthiuron 50% WP OR Imidacloprid 17.8% SL",
            "dosage": "Diafenthiuron @ 1.2 g/L OR Imidacloprid @ 0.3 ml/L of water. Note: Chemical sprays manage whitefly vectors only; they do not kill the virus inside plant tissue.",
            "application_interval": "Alternate chemical classes to prevent whitefly pesticide resistance.",
            "pre_harvest_interval": "14 days"
        },
        "biological_treatment": "Foliar spray with Neem Seed Kernel Extract (NSKE 5%) or cold-pressed Neem Oil @ 3-5 ml/L."
    },
    {
        "id": "healthy_leaf",
        "name": "Healthy Leaf",
        "scientific_name": "Capsicum annuum",
        "category": "Optimal Plant Health",
        "severity": "None",
        "symptoms": [
            "Uniform vibrant green pigmentation.",
            "Intact cuticle without necrotic lesions, spots, or chlorosis.",
            "Smooth margin and healthy turgor pressure."
        ],
        "causes": ["Proper nutrition, adequate moisture, and effective pest vigilance."],
        "prevention": ["Maintain balanced N-P-K fertigation and preventive soil aeration."],
        "chemical_treatment": {
            "spray_name": "No chemical treatment needed",
            "dosage": "None. Do not apply unnecessary pesticides to healthy foliage.",
            "application_interval": "None",
            "pre_harvest_interval": "0 days"
        },
        "biological_treatment": "Periodic root drench with beneficial mycorrhizae or organic humic acid."
    }
]

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist/frontend/browser"))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    if os.path.exists(FRONTEND_DIST):
        target_file = os.path.join(FRONTEND_DIST, path)
        if path and os.path.exists(target_file) and os.path.isfile(target_file):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "service": "Plant Disease Detection AI - Cloud API Service",
        "version": "2.0.0",
        "status": "Operational",
        "deployed_on": "Render",
        "pi_online": LIVE_STATUS["pi_online"],
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "service": "Cloud API",
        "pi_connected": LIVE_STATUS["pi_online"],
        "last_pi_contact": LIVE_STATUS["last_seen"],
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/status", methods=["GET"])
def cloud_status():
    # Evaluate Pi connection status
    now = datetime.now()
    pi_online = False
    if LIVE_STATUS["last_seen"]:
        try:
            last_dt = datetime.fromisoformat(LIVE_STATUS["last_seen"])
            pi_online = (now - last_dt).total_seconds() <= PI_HEARTBEAT_TIMEOUT_SEC
        except Exception:
            pass

    LIVE_STATUS["pi_online"] = pi_online

    # Fetch DB summary
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM cloud_detections")
        total_detections = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) AS healthy FROM cloud_detections WHERE predicted_disease = 'Healthy_Leaf'")
        healthy_count = cursor.fetchone()["healthy"]

    return jsonify({
        "pi_online": pi_online,
        "last_seen": LIVE_STATUS["last_seen"],
        "camera": LIVE_STATUS["camera"],
        "rover": LIVE_STATUS["rover"],
        "sensors": LIVE_STATUS["sensors"],
        "sprayer": LIVE_STATUS["sprayer"],
        "total_detections": total_detections,
        "healthy_count": healthy_count,
        "diseased_count": total_detections - healthy_count,
        "timestamp": now.isoformat()
    })

# -------------------------------------------------------------
# Synchronization Ingestion from Raspberry Pi
# -------------------------------------------------------------
@app.route("/api/sync/heartbeat", methods=["POST"])
def sync_heartbeat():
    data = request.get_json(force=True, silent=True) or {}
    now_str = datetime.now().isoformat()
    LIVE_STATUS["last_seen"] = now_str
    LIVE_STATUS["pi_online"] = True

    if "camera" in data: LIVE_STATUS["camera"] = data["camera"]
    if "rover" in data: LIVE_STATUS["rover"] = data["rover"]
    if "sensors" in data: LIVE_STATUS["sensors"] = data["sensors"]
    if "sprayer" in data: LIVE_STATUS["sprayer"] = data["sprayer"]

    # Check if there are pending remote commands for Pi to execute
    commands_to_send = []
    global PENDING_COMMANDS
    if PENDING_COMMANDS:
        commands_to_send = list(PENDING_COMMANDS)
        PENDING_COMMANDS = []

    return jsonify({
        "success": True,
        "acknowledged_at": now_str,
        "pending_commands": commands_to_send
    })

@app.route("/api/sync/detections", methods=["POST"])
def sync_detections():
    data = request.get_json(force=True, silent=True) or {}
    device_id = data.get("device_id", "unknown-device")
    records = data.get("records", [])

    if not records:
        return jsonify({"success": True, "inserted": 0})

    now_str = datetime.now().isoformat()
    inserted_count = 0

    with get_db() as conn:
        cursor = conn.cursor()
        for r in records:
            try:
                cursor.execute("""
                    INSERT INTO cloud_detections (
                        detection_id, device_id, timestamp, image_filename,
                        image_url, predicted_disease, confidence, is_low_confidence,
                        recommendation, sprayer_status, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(detection_id) DO UPDATE SET
                        confidence = excluded.confidence,
                        sprayer_status = excluded.sprayer_status,
                        synced_at = excluded.synced_at
                """, (
                    r["detection_id"],
                    device_id,
                    r.get("timestamp", now_str),
                    r.get("image_filename", "unknown.jpg"),
                    r.get("image_url", ""),
                    r.get("predicted_disease", "Unknown"),
                    float(r.get("confidence", 0.0)),
                    1 if r.get("is_low_confidence", False) else 0,
                    r.get("recommendation", ""),
                    r.get("sprayer_status", "OFF"),
                    now_str
                ))
                inserted_count += 1
            except Exception as e:
                logger.error(f"Error inserting detection {r.get('detection_id')}: {e}")
        conn.commit()

    LIVE_STATUS["last_seen"] = now_str
    LIVE_STATUS["pi_online"] = True

    return jsonify({
        "success": True,
        "inserted": inserted_count,
        "synced_at": now_str
    }), 201

@app.route("/api/sync/sensors", methods=["POST"])
def sync_sensors():
    data = request.get_json(force=True, silent=True) or {}
    device_id = data.get("device_id", "unknown-device")
    records = data.get("records", [])

    now_str = datetime.now().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        for r in records:
            cursor.execute("""
                INSERT INTO cloud_sensors (
                    device_id, timestamp, ultrasonic_distance,
                    obstacle_detected, soil_moisture, temperature,
                    humidity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                device_id,
                r.get("timestamp", now_str),
                r.get("ultrasonic_distance"),
                1 if r.get("obstacle_detected", False) else 0,
                r.get("soil_moisture"),
                r.get("temperature"),
                r.get("humidity"),
                now_str
            ))
        conn.commit()

    LIVE_STATUS["last_seen"] = now_str
    LIVE_STATUS["pi_online"] = True
    return jsonify({"success": True, "count": len(records)})

# -------------------------------------------------------------
# Frontend Consumption Endpoints
# -------------------------------------------------------------
@app.route("/api/detections/latest", methods=["GET"])
def get_latest_detection():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cloud_detections ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return jsonify({
            "success": True,
            "detection": dict(row) if row else None
        })

@app.route("/api/detections", methods=["GET"])
def get_detections():
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    disease = request.args.get("disease", None)
    min_confidence = request.args.get("min_confidence", None)

    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM cloud_detections WHERE 1=1"
        params = []
        if disease:
            query += " AND predicted_disease = ?"
            params.append(disease)
        if min_confidence:
            query += " AND confidence >= ?"
            params.append(float(min_confidence))
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        records = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT COUNT(*) as count FROM cloud_detections")
        total_count = cursor.fetchone()["count"]

    return jsonify({
        "success": True,
        "total": total_count,
        "count": len(records),
        "detections": records
    })

@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM cloud_detections")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) AS healthy FROM cloud_detections WHERE predicted_disease = 'Healthy_Leaf'")
        healthy = cursor.fetchone()["healthy"]

        cursor.execute("""
            SELECT predicted_disease, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM cloud_detections
            GROUP BY predicted_disease
        """)
        distribution = {
            row["predicted_disease"]: {
                "count": row["count"],
                "avg_confidence": round(row["avg_conf"], 2)
            } for row in cursor.fetchall()
        }

        # Daily Trend (Last 7 days)
        cursor.execute("""
            SELECT substr(timestamp, 1, 10) as day, COUNT(*) as count
            FROM cloud_detections
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
        """)
        daily_trend = [{"date": row["day"], "count": row["count"]} for row in cursor.fetchall()]

        return jsonify({
            "total_detections": total,
            "healthy_count": healthy,
            "diseased_count": total - healthy,
            "disease_distribution": distribution,
            "daily_trend": daily_trend
        })

@app.route("/api/disease-info", methods=["GET"])
def get_disease_info():
    return jsonify({
        "success": True,
        "diseases": DISEASE_ENCYCLOPEDIA
    })

@app.route("/api/reports/export.csv", methods=["GET"])
def export_csv():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT detection_id, device_id, timestamp, predicted_disease,
                   confidence, is_low_confidence, recommendation, sprayer_status
            FROM cloud_detections
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Detection ID", "Device ID", "Timestamp", "Predicted Disease",
        "Confidence (%)", "Low Confidence Flag", "Treatment Recommendation", "Sprayer Status"
    ])
    for r in rows:
        writer.writerow([
            r["detection_id"], r["device_id"], r["timestamp"], r["predicted_disease"],
            r["confidence"], r["is_low_confidence"], r["recommendation"], r["sprayer_status"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=leaf_disease_detections.csv"}
    )

# -------------------------------------------------------------
# Remote Command Relay (From Website to Pi via Heartbeat)
# -------------------------------------------------------------
@app.route("/api/rover/command", methods=["POST"])
def queue_rover_command():
    data = request.get_json(force=True, silent=True) or {}
    cmd = data.get("command", "").upper()
    speed = data.get("speed", 75)

    if not cmd:
        return jsonify({"success": False, "error": "Missing command"}), 400

    command_obj = {
        "id": f"CMD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "type": "ROVER_MOVE",
        "command": cmd,
        "speed": speed,
        "created_at": datetime.now().isoformat()
    }
    PENDING_COMMANDS.append(command_obj)
    logger.info(f"Queued remote rover command: {command_obj}")

    return jsonify({
        "success": True,
        "command_queued": command_obj,
        "message": f"Command {cmd} queued for rover."
    })

@app.route("/api/sprayer/command", methods=["POST"])
def queue_sprayer_command():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action", "OFF").upper()
    duration = float(data.get("duration", 3.0))
    confirmed = bool(data.get("confirmed", False))

    if action == "ON" and not confirmed:
        return jsonify({
            "success": False,
            "error": "Confirmation required before spraying chemicals."
        }), 400

    command_obj = {
        "id": f"SPRAY-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "type": "SPRAYER_ACTUATE",
        "action": action,
        "duration": duration,
        "confirmed": confirmed,
        "created_at": datetime.now().isoformat()
    }
    PENDING_COMMANDS.append(command_obj)
    return jsonify({
        "success": True,
        "command_queued": command_obj,
        "message": f"Sprayer action {action} queued for rover."
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting Cloud API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
