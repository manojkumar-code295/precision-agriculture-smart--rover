"""
Precision Agriculture Smart Rover - Camera Controller Module
Handles automatic initialization, libcamera / Picamera2 detection, continuous real-time video streaming,
live chili leaf object detection with green bounding-box overlays, and automated stable leaf capture.
"""

import io
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List
import cv2
import numpy as np

import config
from leaf_detector import ChiliLeafDetector, LeafTracker, annotate_frame, DetectedLeaf

logger = logging.getLogger("RoverCamera")


class RoverCamera:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(RoverCamera, cls).__new__(cls)
        return cls._instance

    def __init__(self, on_auto_capture: Optional[Callable[[Path], None]] = None):
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

            # Detection & Streaming Subsystems
            self.detector = ChiliLeafDetector()
            self.tracker = LeafTracker(stability_frames=3, max_displacement_px=25.0, cooldown_sec=5.0)
            self.on_auto_capture = on_auto_capture

            self.auto_detect_enabled = True
            self.auto_interval_sec = 3.0
            self.last_auto_capture_time = 0.0

            self.latest_raw_frame: Optional[np.ndarray] = None
            self.latest_annotated_frame: Optional[np.ndarray] = None
            self.latest_jpeg: Optional[bytes] = None
            self.latest_detected_leaves: List[DetectedLeaf] = []
            self.live_fps: float = 0.0

            self.is_streaming = False
            self.stream_thread = None

            self.initialize()
            self._initialized = True

    def initialize(self) -> Tuple[bool, str]:
        """Automatically attempts to detect and initialize the camera hardware."""
        with self._lock:
            logger.info("Initializing Raspberry Pi Camera subsystem...")

            # 1. Attempt libcamera / Picamera2 initialization (RPi Camera Module ov5647)
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                cam_config = cam.create_preview_configuration(main={"size": (640, 480)})
                cam.configure(cam_config)
                cam.start()
                time.sleep(1.0)  # Allow sensor AE/AWB to settle

                self.picam2 = cam
                self.camera_type = "Picamera2 (libcamera)"
                self.is_connected = True
                self.error_message = None
                self.start_stream_loop()
                logger.info("Picamera2 initialized successfully in preview streaming mode.")
                return True, "Picamera2 initialized successfully."
            except Exception as e:
                logger.warning(f"Picamera2 initialization failed: {e}")
                self.picam2 = None

            # 2. Attempt standard V4L2 / USB webcam fallback
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
                            self.start_stream_loop()
                            logger.info("USB / V4L2 camera detected and initialized.")
                            return True, "USB / V4L2 camera initialized."
                        else:
                            cap.release()
                except Exception as e:
                    logger.warning(f"USB/V4L2 camera check failed: {e}")
                    self.cv_cap = None

            # 3. Disconnected state
            self.camera_type = "Hardware Disconnected"
            self.is_connected = False
            self.error_message = (
                "No hardware camera detected. If using the Raspberry Pi Camera Module, check that the "
                "CSI ribbon cable is seated securely in the CAM port with contacts facing the HDMI ports. "
                "Run 'rpicam-hello --list-cameras' to verify connection."
            )
            logger.warning(self.error_message)
            return False, self.error_message

    def start_stream_loop(self):
        """Starts background worker thread for frame acquisition and leaf detection."""
        if self.is_streaming:
            return
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()

    def _stream_worker(self):
        """Continuous camera loop: captures frames, executes detection & tracking, encodes MJPEG."""
        frame_count = 0
        fps_t0 = time.time()

        while self.is_streaming and self.is_connected:
            raw_bgr = None

            # 1. Capture from active camera
            if self.picam2:
                try:
                    bgra = self.picam2.capture_array()
                    raw_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    logger.error(f"Picamera2 read error: {e}")
                    time.sleep(0.05)
                    continue
            elif self.cv_cap and self.cv_cap.isOpened():
                try:
                    ret, frame = self.cv_cap.read()
                    if ret and frame is not None:
                        raw_bgr = frame
                except Exception as e:
                    logger.error(f"OpenCV read error: {e}")
                    time.sleep(0.05)
                    continue

            if raw_bgr is None:
                time.sleep(0.05)
                continue

            self.latest_raw_frame = raw_bgr.copy()

            # 2. Run Chili Leaf Detection & Tracking
            now = time.time()
            candidates, qual_msg = self.detector.detect_leaf_candidates(raw_bgr)
            detected_leaves = self.tracker.update(candidates)
            self.latest_detected_leaves = detected_leaves

            # 3. Check for Automatic Capture on Confirmed Leaf
            if self.auto_detect_enabled and (now - self.last_auto_capture_time >= self.auto_interval_sec):
                for leaf in detected_leaves:
                    if leaf.status == "CONFIRMED_STABLE" and self.tracker.can_capture(leaf):
                        self.last_auto_capture_time = now
                        logger.info(f"Auto-capture triggered for stable chili leaf at {leaf.box}")
                        self._trigger_auto_capture(raw_bgr, leaf)
                        break

            # 4. Compute FPS
            frame_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                self.live_fps = frame_count / elapsed
                frame_count = 0
                fps_t0 = time.time()

            # 5. Render Detection Overlays (Green / Yellow / Red boxes)
            annotated = annotate_frame(
                raw_bgr,
                detected_leaves,
                quality_status=qual_msg,
                fps=self.live_fps,
                auto_active=self.auto_detect_enabled
            )
            self.latest_annotated_frame = annotated

            # 6. Compress to JPEG for Web Stream
            ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                self.latest_jpeg = jpeg.tobytes()

            # Maintain optimal streaming pace (~20-25 FPS)
            time.sleep(0.035)

    def _trigger_auto_capture(self, frame: np.ndarray, leaf: DetectedLeaf):
        """Saves captured leaf image and invokes classification callback."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"auto_leaf_{timestamp_str}.jpg"
        save_path = self.captures_dir / filename
        try:
            cv2.imwrite(str(save_path), frame)
            self.last_capture_path = save_path
            if self.on_auto_capture:
                threading.Thread(target=self.on_auto_capture, args=(save_path,), daemon=True).start()
        except Exception as e:
            logger.error(f"Error saving auto-capture: {e}")

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Returns the latest annotated JPEG frame bytes for HTTP streaming."""
        return self.latest_jpeg

    def capture_image(self, custom_filename: Optional[str] = None) -> Tuple[bool, Optional[Path], str]:
        """Captures and saves a snapshot of the current camera frame."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = custom_filename or f"leaf_capture_{timestamp_str}.jpg"
        save_path = self.captures_dir / filename

        with self._lock:
            if self.latest_raw_frame is not None:
                try:
                    cv2.imwrite(str(save_path), self.latest_raw_frame)
                    self.last_capture_path = save_path
                    return True, save_path, "Image captured successfully from live camera stream."
                except Exception as e:
                    logger.error(f"Frame save error: {e}")
                    return False, None, f"Capture error: {str(e)}"

            # Fallback if stream worker not ready
            if self.picam2:
                try:
                    self.picam2.capture_file(str(save_path))
                    self.last_capture_path = save_path
                    return True, save_path, "Image captured directly via Picamera2."
                except Exception as e:
                    return False, None, f"Picam2 direct capture error: {e}"

            # If hardware disconnected, check sample dataset fallback for local testing
            sample_candidates = list(config.PROJECT_ROOT.glob("Dataset/*/*.jpg"))
            if sample_candidates:
                import shutil
                shutil.copyfile(str(sample_candidates[0]), str(save_path))
                self.last_capture_path = save_path
                return True, save_path, f"Hardware camera unavailable; simulated sample captured from dataset ({sample_candidates[0].parent.name})."

            return False, None, self.error_message or "No camera available to capture image."

    def set_auto_detect(self, enabled: bool):
        self.auto_detect_enabled = enabled
        logger.info(f"Auto leaf detection set to: {enabled}")

    def set_auto_interval(self, interval_sec: float):
        self.auto_interval_sec = max(1.0, float(interval_sec))
        logger.info(f"Auto capture interval set to: {self.auto_interval_sec}s")

    def get_status(self) -> Dict[str, Any]:
        """Returns structured camera diagnostic status."""
        return {
            "camera_connected": self.is_connected,
            "camera_type": self.camera_type,
            "live_fps": round(self.live_fps, 1),
            "auto_detect_enabled": self.auto_detect_enabled,
            "auto_interval_sec": self.auto_interval_sec,
            "detected_leaves_count": len(self.latest_detected_leaves),
            "last_capture": str(self.last_capture_path) if self.last_capture_path else None,
            "error_message": self.error_message,
            "status_timestamp": datetime.now().isoformat()
        }

    def release(self):
        """Safely stops streaming thread and releases camera hardware."""
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)

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
