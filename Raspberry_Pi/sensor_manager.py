"""
Precision Agriculture Smart Rover - Sensor Manager Module
Modular telemetry collector. Reads active sensors and explicitly marks
non-installed sensors as 'Not Installed' (No fabricated values).
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import config
from ultrasonic_sensor import UltrasonicSensor

logger = logging.getLogger("SensorManager")


class SensorManager:
    def __init__(self, ultrasonic: Optional[UltrasonicSensor] = None):
        self.ultrasonic = ultrasonic or UltrasonicSensor()
        self.optional_config = config.OPTIONAL_SENSORS

    def get_all_sensors(self) -> Dict[str, Any]:
        """Collects telemetry from installed sensors and marks missing sensors accurately."""
        now = datetime.now().isoformat()
        sensors_report = {}

        # 1. Ultrasonic Distance Sensor
        distance = self.ultrasonic.get_distance_cm()
        sensors_report["ultrasonic_distance"] = {
            "name": "HC-SR04 Ultrasonic Distance Sensor",
            "installed": True,
            "connected": self.ultrasonic.enabled,
            "reading": distance,
            "unit": "cm",
            "status": "Operational" if self.ultrasonic.enabled else "Simulation / Not Verified",
            "last_updated": now,
            "error": None if self.ultrasonic.enabled else "GPIO hardware not enabled in configuration"
        }

        # 2. Soil Moisture Sensor (Check physical installation flag)
        soil_conf = self.optional_config.get("soil_moisture", {})
        if soil_conf.get("installed", False):
            # If enabled in configuration, attempt reading
            sensors_report["soil_moisture"] = {
                "name": "Capacitive Soil Moisture Sensor",
                "installed": True,
                "connected": False,
                "reading": None,
                "unit": "%",
                "status": "Hardware wiring verification required",
                "last_updated": now,
                "error": "Awaiting analog-to-digital converter (ADC) or digital pin confirmation"
            }
        else:
            sensors_report["soil_moisture"] = {
                "name": "Soil Moisture Sensor",
                "installed": False,
                "connected": False,
                "reading": None,
                "unit": "%",
                "status": "Not Installed",
                "last_updated": now,
                "error": "Sensor hardware not physically installed on rover"
            }

        # 3. Ambient Temperature & Humidity Sensor (DHT11/DHT22)
        dht_conf = self.optional_config.get("temperature_humidity", {})
        if dht_conf.get("installed", False):
            sensors_report["ambient_environment"] = {
                "name": f"{dht_conf.get('model', 'DHT11')} Temperature & Humidity Sensor",
                "installed": True,
                "connected": False,
                "reading": {"temperature": None, "humidity": None},
                "unit": {"temperature": "°C", "humidity": "%"},
                "status": "Hardware wiring verification required",
                "last_updated": now,
                "error": "Awaiting GPIO pin confirmation"
            }
        else:
            sensors_report["ambient_environment"] = {
                "name": "Temperature & Humidity Sensor",
                "installed": False,
                "connected": False,
                "reading": {"temperature": None, "humidity": None},
                "unit": {"temperature": "°C", "humidity": "%"},
                "status": "Not Installed",
                "last_updated": now,
                "error": "Sensor hardware not physically installed on rover"
            }

        return {
            "timestamp": now,
            "sensors": sensors_report
        }
