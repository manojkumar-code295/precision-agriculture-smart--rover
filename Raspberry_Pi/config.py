"""
Precision Agriculture Smart Rover - Configuration Module
Centralized, safe, and configurable hardware and AI settings.
"""

import os
import json
from pathlib import Path

# Base Paths
RASPBERRY_PI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RASPBERRY_PI_DIR.parent
AI_MODEL_DIR = PROJECT_ROOT / "AI_Model"
CAPTURES_DIR = RASPBERRY_PI_DIR / "captured_leaves"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = RASPBERRY_PI_DIR / "rover_local.db"

# AI Model Configuration
TFLITE_MODEL_PATH = AI_MODEL_DIR / "model.tflite"
KERAS_MODEL_PATH = AI_MODEL_DIR / "best_chili_leaf_model.keras"
CLASS_NAMES_PATH = AI_MODEL_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "70.0"))

# Four Verified Classes
CLASS_NAMES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf"
]

# Device Identity
DEVICE_ID = os.getenv("DEVICE_ID", "agrirover-pi4-01")
DEVICE_NAME = os.getenv("DEVICE_NAME", "Precision Smart Rover Pi4")

# Hardware Pin Verification Flag
# GPIO pins will not physically drive outputs until confirmed by the user
GPIO_HARDWARE_ENABLED = os.getenv("GPIO_HARDWARE_ENABLED", "false").lower() in ("1", "true", "yes")

# Motor Driver Pin Mapping (BCM Numbers)
# Standard L298N / L293D dual H-bridge motor driver interface
# Left Motor: IN1, IN2, ENA (PWM)
# Right Motor: IN3, IN4, ENB (PWM)
MOTOR_PINS = {
    "LEFT_IN1": int(os.getenv("MOTOR_LEFT_IN1", "17")),
    "LEFT_IN2": int(os.getenv("MOTOR_LEFT_IN2", "27")),
    "LEFT_ENA": int(os.getenv("MOTOR_LEFT_ENA", "22")),
    "RIGHT_IN3": int(os.getenv("MOTOR_RIGHT_IN3", "23")),
    "RIGHT_IN4": int(os.getenv("MOTOR_RIGHT_IN4", "24")),
    "RIGHT_ENB": int(os.getenv("MOTOR_RIGHT_ENB", "25")),
}

MOTOR_DEFAULT_SPEED = int(os.getenv("MOTOR_DEFAULT_SPEED", "75"))  # 0 to 100%
MOTOR_COMMAND_TIMEOUT = float(os.getenv("MOTOR_COMMAND_TIMEOUT", "1.5"))  # Seconds before auto-stop

# Ultrasonic Sensor Configuration (HC-SR04)
# Note: ECHO output must use a voltage divider (e.g. 1k + 2k resistors) to convert 5V to 3.3V
ULTRASONIC_PINS = {
    "TRIG": int(os.getenv("ULTRASONIC_TRIG", "5")),
    "ECHO": int(os.getenv("ULTRASONIC_ECHO", "6")),
}
ULTRASONIC_ENABLED = os.getenv("ULTRASONIC_ENABLED", "false").lower() in ("1", "true", "yes")
OBSTACLE_SAFETY_DISTANCE_CM = float(os.getenv("OBSTACLE_SAFETY_DISTANCE_CM", "25.0"))
CRITICAL_STOP_DISTANCE_CM = float(os.getenv("CRITICAL_STOP_DISTANCE_CM", "15.0"))

# Sprayer Relay Configuration
SPRAYER_PIN = int(os.getenv("SPRAYER_PIN", "26"))
SPRAYER_ENABLED = os.getenv("SPRAYER_ENABLED", "false").lower() in ("1", "true", "yes")
DEFAULT_SPRAY_DURATION_SEC = float(os.getenv("DEFAULT_SPRAY_DURATION_SEC", "3.0"))
MAX_SPRAY_DURATION_SEC = 10.0  # Absolute safety hard limit

# Other Optional Sensors (Soil moisture, Temperature, Humidity)
# Default to NOT INSTALLED unless physically present and verified
OPTIONAL_SENSORS = {
    "soil_moisture": {
        "installed": os.getenv("SENSOR_SOIL_INSTALLED", "false").lower() in ("1", "true", "yes"),
        "pin": int(os.getenv("SENSOR_SOIL_PIN", "16")),
        "type": "analog_or_digital",
    },
    "temperature_humidity": {
        "installed": os.getenv("SENSOR_DHT_INSTALLED", "false").lower() in ("1", "true", "yes"),
        "pin": int(os.getenv("SENSOR_DHT_PIN", "4")),
        "model": "DHT11",
    }
}

# Cloud Synchronization Configuration
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "https://precision-agriculture-smart--rover-api.onrender.com")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "smartrover_secure_key_2026")
SYNC_INTERVAL_SEC = int(os.getenv("SYNC_INTERVAL_SEC", "30"))
SYNC_BATCH_SIZE = int(os.getenv("SYNC_BATCH_SIZE", "20"))

# Local Storage Limits
MAX_LOCAL_IMAGES = int(os.getenv("MAX_LOCAL_IMAGES", "500"))
CLEANUP_CHECK_INTERVAL_SEC = 3600  # 1 hour

# Flask API Port
LOCAL_API_PORT = int(os.getenv("LOCAL_API_PORT", "5000"))
LOCAL_API_HOST = os.getenv("LOCAL_API_HOST", "0.0.0.0")
