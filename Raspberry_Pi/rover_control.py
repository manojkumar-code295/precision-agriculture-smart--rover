"""
Precision Agriculture Smart Rover - Rover Control Module
Integrates motor drive commands, obstacle avoidance interlocks, watchdog safety timers,
and 3 operating modes (Manual, Assisted Detection, and Auto Monitoring).
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
import config
from motor_driver import MotorDriver
from obstacle_detection import ObstacleDetector

logger = logging.getLogger("RoverControl")


class RoverController:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(RoverController, cls).__new__(cls)
        return cls._instance

    def __init__(self, motor_driver: Optional[MotorDriver] = None,
                 obstacle_detector: Optional[ObstacleDetector] = None):
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            self.driver = motor_driver or MotorDriver()
            self.obstacle_detector = obstacle_detector or ObstacleDetector()
            self.mode = "MANUAL"  # "MANUAL", "ASSISTED", "AUTO"
            self.is_moving = False
            self.last_command_time = time.time()
            self.watchdog_thread = None
            self.running = True

            # Start watchdog safety timer thread
            self._start_watchdog()
            self._initialized = True

    def _start_watchdog(self):
        """Monitors command timestamps and continuously checks obstacle proximity during forward motion."""
        def watchdog_loop():
            while self.running:
                # 1. Real-time active obstacle detection and automatic rover stop
                if self.is_moving and self.driver.current_state == "FORWARD":
                    try:
                        obs = self.obstacle_detector.check_obstacle()
                        if obs.get("obstacle_detected") or not obs.get("safe_to_move_forward"):
                            dist = obs.get("distance_cm")
                            logger.warning(
                                f"AUTOMATIC OBSTACLE BRAKE ACTIVATED: Object detected at {dist} cm "
                                f"(Safety threshold: {self.obstacle_detector.safety_distance} cm). "
                                "Halting rover immediately!"
                            )
                            self.driver.stop()
                            self.is_moving = False
                    except Exception as e:
                        logger.error(f"Error in obstacle collision watchdog: {e}")

                # 2. Watchdog timeout for manual teleoperation safety
                if self.is_moving and self.mode == "MANUAL":
                    if time.time() - self.last_command_time > config.MOTOR_COMMAND_TIMEOUT:
                        logger.info("Watchdog timeout reached: Auto-stopping rover motors.")
                        self.stop()
                time.sleep(0.08)

        self.watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
        self.watchdog_thread.start()

    def execute_command(self, command: str, speed: Optional[int] = None) -> Dict[str, Any]:
        """Executes directional commands with safety interlocks."""
        cmd = command.upper().strip()
        motor_speed = speed if speed is not None else config.MOTOR_DEFAULT_SPEED

        # Check obstacle status first
        obs_status = self.obstacle_detector.check_obstacle()

        if cmd == "FORWARD":
            if not obs_status["safe_to_move_forward"]:
                self.driver.stop()
                self.is_moving = False
                logger.warning(f"Movement command FORWARD blocked by obstacle interlock: {obs_status['message']}")
                return {
                    "success": False,
                    "state": "BLOCKED",
                    "reason": "OBSTACLE_DETECTED",
                    "obstacle_info": obs_status,
                    "message": "Cannot move forward: Obstacle detected within safety distance."
                }
            self.driver.forward(motor_speed)
            self.is_moving = True
            self.last_command_time = time.time()

        elif cmd == "BACKWARD":
            self.driver.backward(motor_speed)
            self.is_moving = True
            self.last_command_time = time.time()

        elif cmd == "LEFT":
            self.driver.turn_left(motor_speed)
            self.is_moving = True
            self.last_command_time = time.time()

        elif cmd == "RIGHT":
            self.driver.turn_right(motor_speed)
            self.is_moving = True
            self.last_command_time = time.time()

        elif cmd in ("STOP", "BRAKE"):
            self.driver.stop()
            self.is_moving = False

        elif cmd in ("EMERGENCY_STOP", "ESTOP"):
            self.driver.emergency_stop()
            self.is_moving = False
            return {
                "success": True,
                "state": "EMERGENCY_STOPPED",
                "message": "Rover emergency stop executed successfully."
            }

        else:
            return {
                "success": False,
                "state": self.driver.current_state,
                "reason": "INVALID_COMMAND",
                "message": f"Unrecognized movement command '{command}'"
            }

        return {
            "success": True,
            "state": self.driver.current_state,
            "speed": self.driver.current_speed,
            "mode": self.mode,
            "obstacle_status": obs_status,
            "message": f"Command {cmd} executed."
        }

    def set_mode(self, new_mode: str) -> Dict[str, Any]:
        """Switches between MANUAL, ASSISTED, and AUTO modes safely."""
        valid_modes = ["MANUAL", "ASSISTED", "AUTO"]
        mode_upper = new_mode.upper().strip()
        if mode_upper not in valid_modes:
            return {"success": False, "message": f"Invalid mode. Choose from {valid_modes}"}

        # Stop motors before mode change
        self.stop()
        self.mode = mode_upper
        logger.info(f"Rover mode changed to {self.mode}")
        return {"success": True, "mode": self.mode, "message": f"Rover mode set to {self.mode}"}

    def stop(self):
        self.driver.stop()
        self.is_moving = False

    def emergency_stop(self):
        self.driver.emergency_stop()
        self.is_moving = False
        return {
            "success": True,
            "state": "EMERGENCY_STOPPED",
            "message": "Rover emergency stop executed successfully."
        }

    def get_status(self) -> Dict[str, Any]:
        obs = self.obstacle_detector.check_obstacle()
        return {
            "mode": self.mode,
            "is_moving": self.is_moving,
            "motor_status": self.driver.get_status(),
            "obstacle_status": obs
        }

    def shutdown(self):
        self.running = False
        self.driver.cleanup()
        self.obstacle_detector.sensor.cleanup()
        logger.info("Rover controller shutdown complete.")
