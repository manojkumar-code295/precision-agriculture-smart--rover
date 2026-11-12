"""
Precision Agriculture Smart Rover - Ultrasonic Distance Sensor Module
Handles distance measurement via HC-SR04 with pulse timeout protection and noise filtering.

IMPORTANT ELECTRICAL SAFETY:
- Raspberry Pi GPIO operates at 3.3V logic levels.
- The HC-SR04 ECHO pin outputs 5V signals.
- A voltage divider (e.g., 1kΩ and 2kΩ resistors) MUST be connected between
  the ECHO pin and the Raspberry Pi GPIO input to step down 5V to ~3.3V.
"""

import logging
import time
import statistics
from typing import Optional, Dict, Any
import config

logger = logging.getLogger("UltrasonicSensor")

# Check GPIO availability
GPIO_LIB_AVAILABLE = False
try:
    import gpiozero
    from gpiozero import DigitalOutputDevice, DigitalInputDevice
    GPIO_LIB_AVAILABLE = True
except ImportError:
    pass


class UltrasonicSensor:
    def __init__(self, pins: Optional[Dict[str, int]] = None):
        self.pins = pins or config.ULTRASONIC_PINS
        self.enabled = config.ULTRASONIC_ENABLED and GPIO_LIB_AVAILABLE
        self.trig = None
        self.echo = None
        self.last_valid_distance = 100.0  # Safe default 100cm

        self._init_sensor()

    def _init_sensor(self):
        if not self.enabled:
            logger.info("UltrasonicSensor running in SIMULATION/UNVERIFIED mode.")
            return

        try:
            self.trig = DigitalOutputDevice(self.pins["TRIG"], initial_value=False)
            self.echo = DigitalInputDevice(self.pins["ECHO"], pull_up=False)
            time.sleep(0.05)
            logger.info(f"Ultrasonic sensor initialized on TRIG={self.pins['TRIG']}, ECHO={self.pins['ECHO']}")
        except Exception as e:
            logger.error(f"Failed to initialize ultrasonic GPIO: {e}. Running in simulation.")
            self.enabled = False

    def _measure_single_pulse(self) -> Optional[float]:
        """Sends a 10µs trigger pulse and measures echo return time with timeout."""
        if not self.enabled or self.trig is None or self.echo is None:
            return None

        # Ensure trigger is low
        self.trig.off()
        time.sleep(0.000002)

        # Send 10 microsecond pulse
        self.trig.on()
        time.sleep(0.000010)
        self.trig.off()

        # Wait for echo to go high (timeout 0.03s -> ~5m max range)
        timeout_start = time.time()
        while not self.echo.is_active:
            if time.time() - timeout_start > 0.03:
                return None
        pulse_start = time.time()

        # Wait for echo to go low
        while self.echo.is_active:
            if time.time() - pulse_start > 0.03:
                return None
        pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        # Speed of sound = 34300 cm/s
        distance_cm = (pulse_duration * 34300.0) / 2.0

        # Validate range (2 cm to 400 cm)
        if 2.0 <= distance_cm <= 400.0:
            return round(distance_cm, 1)
        return None

    def get_distance_cm(self) -> Optional[float]:
        """Takes 3 rapid readings and returns the median filtered distance."""
        if not self.enabled:
            # Return safe default or simulated clear reading
            return 120.0

        readings = []
        for _ in range(3):
            d = self._measure_single_pulse()
            if d is not None:
                readings.append(d)
            time.sleep(0.01)

        if readings:
            filtered_distance = round(statistics.median(readings), 1)
            self.last_valid_distance = filtered_distance
            return filtered_distance

        return self.last_valid_distance

    def cleanup(self):
        if self.enabled:
            if self.trig:
                try: self.trig.close()
                except Exception: pass
            if self.echo:
                try: self.echo.close()
                except Exception: pass
        logger.info("Ultrasonic sensor cleaned up.")
