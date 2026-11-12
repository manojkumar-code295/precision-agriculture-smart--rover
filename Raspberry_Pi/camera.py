"""
Precision Agriculture Smart Rover - Camera Controller Module
Handles automatic initialization, libcamera / Picamera2 detection, USB webcam fallback,
and safe still-image capture for AI leaf classification.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import config

logger = logging.getLogger("RoverCamera")


class RoverCamera:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(RoverCamera, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            self.camera_type = "None"
            self.is_connected = False
            self.error_message = None
            self.picam2 = None
            self.cv_cap = None
            self.last_capture_path = None
            self.captures_dir = config.CAPTURES_DIR

            self.initialize()
            self._initialized = True

    def initialize(self) -> Tuple[bool, str]:
        """Automatically attempts to detect and initialize the camera stack."""
        with self._lock:
            logger.info("Initializing Raspberry Pi Camera subsystem...")

            # 1. Attempt libcamera / Picamera2 initialization
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                camera_info = cam.camera_controls
                # Attempt to configure still capture mode
                cam_config = cam.create_still_configuration(main={"size": (1920, 1080)})
                cam.configure(cam_config)
                cam.start()
                time.sleep(1.0)  # Sensor warmup and auto-exposure settling

                self.picam2 = cam
                self.camera_type = "Picamera2 (libcamera)"
                self.is_connected = True
                self.error_message = None
                logger.info("Picamera2 initialized successfully.")
                return True, "Picamera2 initialized successfully."
            except Exception as e:
                logger.warning(f"Picamera2 initialization failed: {e}")
                self.picam2 = None

            # 2. Attempt standard V4L2 / USB webcam fallback only if /dev/video0 exists
            import os
            if os.path.exists("/dev/video0"):
                try:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            self.cv_cap = cap
                            self.camera_type = "USB / V4L2 Camera"
                            self.is_connected = True
                            self.error_message = None
                            logger.info("USB / V4L2 camera detected and initialized.")
                            return True, "USB / V4L2 camera initialized."
                        else:
                            cap.release()
                except Exception as e:
                    logger.warning(f"USB/V4L2 camera check failed: {e}")
                    self.cv_cap = None

            # 3. No physical camera currently detected
            self.camera_type = "Hardware Disconnected"
            self.is_connected = False
            self.error_message = (
                "No hardware camera detected. If using the Raspberry Pi Camera Module, check that the "
                "CSI ribbon cable is seated securely in the CAM port with contacts facing the HDMI ports. "
                "Run 'rpicam-hello --list-cameras' to verify connection."
            )
            logger.warning(self.error_message)
            return False, self.error_message

    def capture_image(self, custom_filename: Optional[str] = None) -> Tuple[bool, Optional[Path], str]:
        """Captures a still image from the active camera source."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = custom_filename or f"leaf_capture_{timestamp_str}.jpg"
        save_path = self.captures_dir / filename

        with self._lock:
            # 1. Picamera2 capture
            if self.picam2 is not None:
                try:
                    self.picam2.capture_file(str(save_path))
                    self.last_capture_path = save_path
                    return True, save_path, "Image captured successfully via Picamera2."
                except Exception as e:
                    logger.error(f"Picamera2 capture error: {e}")
                    return False, None, f"Capture error: {str(e)}"

            # 2. OpenCV capture
            if self.cv_cap is not None and self.cv_cap.isOpened():
                try:
                    ret, frame = self.cv_cap.read()
                    if ret and frame is not None:
                        cv2.imwrite(str(save_path), frame)
                        self.last_capture_path = save_path
                        return True, save_path, "Image captured successfully via USB camera."
                except Exception as e:
                    logger.error(f"OpenCV capture error: {e}")
                    return False, None, f"Capture error: {str(e)}"

            # 3. If hardware is disconnected, check if a dataset sample can be used for simulation test
            sample_candidates = list(config.PROJECT_ROOT.glob("Dataset/*/*.jpg"))
            if sample_candidates:
                import shutil
                shutil.copyfile(str(sample_candidates[0]), str(save_path))
                self.last_capture_path = save_path
                return True, save_path, f"Hardware camera unavailable; simulated sample captured from dataset ({sample_candidates[0].parent.name})."

            return False, None, self.error_message or "No camera available to capture image."

    def get_status(self) -> Dict[str, Any]:
        """Returns structured camera diagnostic status."""
        return {
            "camera_connected": self.is_connected,
            "camera_type": self.camera_type,
            "last_capture": str(self.last_capture_path) if self.last_capture_path else None,
            "error_message": self.error_message,
            "status_timestamp": datetime.now().isoformat()
        }

    def release(self):
        """Safely stops and releases camera hardware resources."""
        with self._lock:
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except Exception as e:
                    logger.warning(f"Error closing Picamera2: {e}")
                self.picam2 = None

            if self.cv_cap is not None:
                try:
                    self.cv_cap.release()
                except Exception as e:
                    logger.warning(f"Error releasing cv_cap: {e}")
                self.cv_cap = None

            self.is_connected = False
            self.camera_type = "None"
            logger.info("Camera resources released.")
