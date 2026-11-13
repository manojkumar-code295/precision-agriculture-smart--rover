"""
Precision Agriculture Smart Rover - Ultrasonic Distance Sensor Module
Handles physical distance measurement via HC-SR04 on Raspberry Pi GPIO
with microsecond pulse timing, median noise filtering, and automatic obstacle safety.
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
    import RPi.GPIO as GPIO
    GPIO_LIB_AVAILABLE = True
except ImportError:
    try:
        import gpiozero
        GPIO_LIB_AVAILABLE = True
    except ImportError:
        pass


class UltrasonicSensor:
    def __init__(self, pins: Optional[Dict[str, int]] = None):
        self.pins = pins or config.ULTRASONIC_PINS
        self.trig_pin = int(self.pins.get("TRIG", 5))
        self.echo_pin = int(self.pins.get("ECHO", 6))
        self.enabled = config.ULTRASONIC_ENABLED and GPIO_LIB_AVAILABLE
        self.last_valid_distance = 150.0  # Safe default 150cm (clear path)

        self._init_sensor()

    def _init_sensor(self):
        if not self.enabled:
            logger.info("UltrasonicSensor running in SIMULATION/UNVERIFIED mode.")
            return

        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.output(self.trig_pin, False)
            time.sleep(0.05)
            logger.info(f"HC-SR04 Ultrasonic sensor initialized (TRIG=BCM {self.trig_pin}, ECHO=BCM {self.echo_pin})")
        except Exception as e:
            logger.error(f"Failed to initialize ultrasonic GPIO: {e}. Running in simulation.")
            self.enabled = False

    def _measure_single_pulse(self) -> Optional[float]:
        """Sends a 12µs trigger pulse and measures echo pulse duration with microsecond accuracy."""
        if not self.enabled:
            return None

        try:
            import RPi.GPIO as GPIO
            # Ensure trigger is low before pulse
            GPIO.output(self.trig_pin, False)
            time.sleep(0.000002)

            # Send 12 microsecond trigger pulse
            GPIO.output(self.trig_pin, True)
            time.sleep(0.000012)
            GPIO.output(self.trig_pin, False)

            # Wait for echo to go high (timeout 25ms -> ~4.2m max range)
            t_start = time.perf_counter()
            while GPIO.input(self.echo_pin) == 0:
                if time.perf_counter() - t_start > 0.025:
                    return None

            pulse_start = time.perf_counter()

            # Wait for echo to go low (timeout 30ms -> ~5m max range)
            timeout_limit = pulse_start + 0.030
            while GPIO.input(self.echo_pin) == 1:
                if time.perf_counter() > timeout_limit:
                    # Echo took > 30ms: clear open path
                    return 400.0

            pulse_end = time.perf_counter()
            duration = pulse_end - pulse_start
            distance_cm = (duration * 34300.0) / 2.0

            if 2.0 <= distance_cm <= 400.0:
                return round(distance_cm, 1)
            elif distance_cm > 400.0:
                return 400.0
        except Exception as e:
            logger.debug(f"Pulse read error: {e}")
        return None

    def get_distance_cm(self) -> Optional[float]:
        """Takes rapid readings and returns median filtered distance in centimeters."""
        if not self.enabled:
            return 150.0

        readings = []
        for _ in range(3):
            d = self._measure_single_pulse()
            if d is not None:
                readings.append(d)
            time.sleep(0.01)

        if readings:
            filtered = round(statistics.median(readings), 1)
            self.last_valid_distance = filtered
            return filtered

        return self.last_valid_distance

    def cleanup(self):
        if self.enabled:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup([self.trig_pin, self.echo_pin])
            except Exception:
                pass
        logger.info("Ultrasonic sensor cleaned up.")
