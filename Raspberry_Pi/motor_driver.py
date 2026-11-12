"""
Precision Agriculture Smart Rover - DC Motor Driver Module
Controls dual DC motors via H-Bridge (L298N / L293D) using safe GPIO / PWM.
"""

import logging
import time
from typing import Dict, Any, Optional
import config

logger = logging.getLogger("MotorDriver")

# Try importing hardware GPIO libraries safely
GPIO_LIB_AVAILABLE = False
try:
    import gpiozero
    from gpiozero import PWMOutputDevice, DigitalOutputDevice
    GPIO_LIB_AVAILABLE = True
except ImportError:
    pass


class MotorDriver:
    def __init__(self, pins: Optional[Dict[str, int]] = None):
        self.pins = pins or config.MOTOR_PINS
        self.hardware_enabled = config.GPIO_HARDWARE_ENABLED and GPIO_LIB_AVAILABLE
        self.current_state = "STOPPED"
        self.current_speed = 0

        # Device handles
        self.left_in1 = None
        self.left_in2 = None
        self.left_ena = None
        self.right_in3 = None
        self.right_in4 = None
        self.right_enb = None

        self._init_gpio()

    def _init_gpio(self):
        """Initializes GPIO pins safely with all outputs LOW."""
        if not self.hardware_enabled:
            logger.info("MotorDriver operating in SAFE SIMULATION mode (Hardware pins disabled in config).")
            return

        try:
            self.left_in1 = DigitalOutputDevice(self.pins["LEFT_IN1"], initial_value=False)
            self.left_in2 = DigitalOutputDevice(self.pins["LEFT_IN2"], initial_value=False)
            self.left_ena = PWMOutputDevice(self.pins["LEFT_ENA"], initial_value=0.0)

            self.right_in3 = DigitalOutputDevice(self.pins["RIGHT_IN3"], initial_value=False)
            self.right_in4 = DigitalOutputDevice(self.pins["RIGHT_IN4"], initial_value=False)
            self.right_enb = PWMOutputDevice(self.pins["RIGHT_ENB"], initial_value=0.0)
            logger.info(f"GPIO Motor pins initialized successfully: {self.pins}")
        except Exception as e:
            logger.error(f"Failed to initialize GPIO pins: {e}. Falling back to simulation mode.")
            self.hardware_enabled = False

    def forward(self, speed: int = config.MOTOR_DEFAULT_SPEED):
        """Drives both left and right motors forward."""
        norm_speed = max(0.0, min(1.0, speed / 100.0))
        self.current_state = "FORWARD"
        self.current_speed = speed

        if self.hardware_enabled:
            # Left forward: IN1=HIGH, IN2=LOW
            self.left_in1.on()
            self.left_in2.off()
            self.left_ena.value = norm_speed

            # Right forward: IN3=HIGH, IN4=LOW
            self.right_in3.on()
            self.right_in4.off()
            self.right_enb.value = norm_speed

        logger.info(f"Rover moving FORWARD at {speed}% speed")

    def backward(self, speed: int = config.MOTOR_DEFAULT_SPEED):
        """Drives both left and right motors backward."""
        norm_speed = max(0.0, min(1.0, speed / 100.0))
        self.current_state = "BACKWARD"
        self.current_speed = speed

        if self.hardware_enabled:
            # Left backward: IN1=LOW, IN2=HIGH
            self.left_in1.off()
            self.left_in2.on()
            self.left_ena.value = norm_speed

            # Right backward: IN3=LOW, IN4=HIGH
            self.right_in3.off()
            self.right_in4.on()
            self.right_enb.value = norm_speed

        logger.info(f"Rover moving BACKWARD at {speed}% speed")

    def turn_left(self, speed: int = config.MOTOR_DEFAULT_SPEED):
        """Turns left (Left motor backward/stopped, Right motor forward)."""
        norm_speed = max(0.0, min(1.0, speed / 100.0))
        self.current_state = "LEFT"
        self.current_speed = speed

        if self.hardware_enabled:
            # Left reverse or stop
            self.left_in1.off()
            self.left_in2.on()
            self.left_ena.value = norm_speed * 0.5

            # Right forward
            self.right_in3.on()
            self.right_in4.off()
            self.right_enb.value = norm_speed

        logger.info(f"Rover turning LEFT at {speed}% speed")

    def turn_right(self, speed: int = config.MOTOR_DEFAULT_SPEED):
        """Turns right (Left motor forward, Right motor backward/stopped)."""
        norm_speed = max(0.0, min(1.0, speed / 100.0))
        self.current_state = "RIGHT"
        self.current_speed = speed

        if self.hardware_enabled:
            # Left forward
            self.left_in1.on()
            self.left_in2.off()
            self.left_ena.value = norm_speed

            # Right reverse
            self.right_in3.off()
            self.right_in4.on()
            self.right_enb.value = norm_speed * 0.5

        logger.info(f"Rover turning RIGHT at {speed}% speed")

    def stop(self):
        """Stops all motor rotation and sets PWM outputs to zero."""
        self.current_state = "STOPPED"
        self.current_speed = 0

        if self.hardware_enabled:
            if self.left_in1: self.left_in1.off()
            if self.left_in2: self.left_in2.off()
            if self.left_ena: self.left_ena.value = 0.0

            if self.right_in3: self.right_in3.off()
            if self.right_in4: self.right_in4.off()
            if self.right_enb: self.right_enb.value = 0.0

        logger.info("Rover motors STOPPED")

    def emergency_stop(self):
        """Immediate hard stop; cuts all GPIO signals."""
        logger.warning("EMERGENCY STOP TRIGGERED!")
        self.stop()

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.current_state,
            "speed": self.current_speed,
            "hardware_enabled": self.hardware_enabled,
            "pins": self.pins
        }

    def cleanup(self):
        """Closes all hardware devices cleanly."""
        self.stop()
        if self.hardware_enabled:
            for dev in [self.left_in1, self.left_in2, self.left_ena,
                        self.right_in3, self.right_in4, self.right_enb]:
                if dev:
                    try:
                        dev.close()
                    except Exception:
                        pass
        logger.info("Motor driver cleaned up.")
