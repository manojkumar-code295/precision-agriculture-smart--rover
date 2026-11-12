"""
Precision Agriculture Smart Rover - Obstacle Detection Module
Evaluates proximity hazards, triggers automatic motor stops, and locks out forward movement.
"""

import logging
from typing import Dict, Any, Optional
import config
from ultrasonic_sensor import UltrasonicSensor

logger = logging.getLogger("ObstacleDetection")


class ObstacleDetector:
    def __init__(self, sensor: Optional[UltrasonicSensor] = None):
        self.sensor = sensor or UltrasonicSensor()
        self.safety_distance = config.OBSTACLE_SAFETY_DISTANCE_CM
        self.critical_distance = config.CRITICAL_STOP_DISTANCE_CM

    def check_obstacle(self) -> Dict[str, Any]:
        """Queries the ultrasonic sensor and determines safety clearances."""
        distance = self.sensor.get_distance_cm()

        if distance is None:
            # Sensor timeout or invalid reading
            return {
                "sensor_connected": self.sensor.enabled,
                "distance_cm": None,
                "obstacle_detected": False,
                "hazard_level": "UNKNOWN",
                "safe_to_move_forward": True,
                "message": "Ultrasonic sensor offline or timeout"
            }

        is_critical = distance <= self.critical_distance
        is_obstacle = distance <= self.safety_distance

        if is_critical:
            hazard_level = "CRITICAL"
            safe_forward = False
            msg = f"CRITICAL OBSTACLE at {distance} cm! Forward movement locked."
        elif is_obstacle:
            hazard_level = "WARNING"
            safe_forward = False
            msg = f"Obstacle detected within safety buffer ({distance} cm). Forward blocked."
        else:
            hazard_level = "CLEAR"
            safe_forward = True
            msg = f"Clear path ({distance} cm)."

        return {
            "sensor_connected": self.sensor.enabled,
            "distance_cm": distance,
            "obstacle_detected": is_obstacle,
            "is_critical": is_critical,
            "hazard_level": hazard_level,
            "safe_to_move_forward": safe_forward,
            "message": msg
        }
