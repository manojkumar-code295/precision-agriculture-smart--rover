"""
Precision Agriculture Smart Rover - Sprayer & Fertilizer Control Module
Safely actuates DC pump via relay with explicit user confirmation and hard timeout safety.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import config

logger = logging.getLogger("SprayerControl")

GPIO_LIB_AVAILABLE = False
try:
    import gpiozero
    from gpiozero import DigitalOutputDevice
    GPIO_LIB_AVAILABLE = True
except ImportError:
    pass


class SprayerController:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(SprayerController, cls).__new__(cls)
        return cls._instance

    def __init__(self, relay_pin: Optional[int] = None):
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            self.relay_pin = relay_pin or config.SPRAYER_PIN
            self.enabled = config.SPRAYER_ENABLED and GPIO_LIB_AVAILABLE
            self.relay = None
            self.is_spraying = False
            self.last_spray_time = None
            self.spray_timer = None

            self._init_relay()
            self._initialized = True

    def _init_relay(self):
        """Initializes relay pin with pump strictly OFF."""
        if not self.enabled:
            logger.info("SprayerController in SIMULATION mode (Hardware relay disabled in config).")
            return

        try:
            # Active low or active high depending on relay module; initial_value=False keeps relay open (OFF)
            self.relay = DigitalOutputDevice(self.relay_pin, initial_value=False)
            logger.info(f"Sprayer relay initialized on GPIO {self.relay_pin} (PUMP OFF).")
        except Exception as e:
            logger.error(f"Failed to initialize sprayer GPIO: {e}. Running in simulation.")
            self.enabled = False

    def activate_spray(self, duration_sec: Optional[float] = None,
                       user_confirmed: bool = False,
                       disease_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Activates sprayer for a strictly bounded duration.
        SAFETY REQUIREMENT: Explicit user confirmation is strictly mandatory.
        """
        with self._lock:
            if not user_confirmed:
                return {
                    "success": False,
                    "sprayer_active": False,
                    "reason": "CONFIRMATION_REQUIRED",
                    "message": "Spraying chemical treatments requires explicit user authorization and safety confirmation."
                }

            if self.is_spraying:
                return {
                    "success": False,
                    "sprayer_active": True,
                    "reason": "ALREADY_ACTIVE",
                    "message": "Sprayer pump is currently operating."
                }

            # Enforce strict duration bounds
            duration = duration_sec if duration_sec is not None else config.DEFAULT_SPRAY_DURATION_SEC
            duration = max(0.5, min(duration, config.MAX_SPRAY_DURATION_SEC))

            logger.info(f"Activating sprayer pump for {duration:.1f} seconds. Disease: {disease_context.get('disease') if disease_context else 'Manual'}")

            # Turn ON relay
            if self.enabled and self.relay:
                self.relay.on()

            self.is_spraying = True
            self.last_spray_time = datetime.now().isoformat()

            # Automatic cutoff timer thread
            def auto_cutoff():
                time.sleep(duration)
                self.stop_spray()

            self.spray_timer = threading.Thread(target=auto_cutoff, daemon=True)
            self.spray_timer.start()

            return {
                "success": True,
                "sprayer_active": True,
                "duration_seconds": duration,
                "timestamp": self.last_spray_time,
                "message": f"Sprayer activated for {duration} seconds."
            }

    def stop_spray(self):
        """Immediately turns OFF the sprayer relay."""
        with self._lock:
            if self.enabled and self.relay:
                self.relay.off()

            self.is_spraying = False
            logger.info("Sprayer pump OFF.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "sprayer_active": self.is_spraying,
            "relay_pin": self.relay_pin,
            "hardware_enabled": self.enabled,
            "last_activated": self.last_spray_time,
            "default_duration_sec": config.DEFAULT_SPRAY_DURATION_SEC,
            "max_duration_sec": config.MAX_SPRAY_DURATION_SEC
        }

    def cleanup(self):
        self.stop_spray()
        if self.enabled and self.relay:
            try:
                self.relay.close()
            except Exception:
                pass
        logger.info("Sprayer relay cleaned up.")
