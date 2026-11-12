"""
Precision Agriculture Smart Rover - Optimized Camera Controller Module
Features:
- Dedicated continuous hardware acquisition thread (Picamera2 / libcamera or V4L2)
- Target FPS: 8-10 FPS at 640x480 with JPEG quality 70%
- Decoupled AI inference thread (prevents frame drops and acquisition lag)
- Single-slot latest frame buffer (drops stale frames, eliminates queue buildup)
- Automatic reconnection on sensor/bus interruptions
- Real FPS metering and diagnostic health telemetry
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
            self.tracker = LeafTracker(stability_frames=3, max_displacement_px=30.0, cooldown_sec=5.0)
            self.on_auto_capture = on_auto_capture

            self.auto_detect_enabled = True
            self.auto_interval_sec = 3.0
            self.last_auto_capture_time = 0.0

            # Single-Slot Frame Buffers (Atomic latest-frame semantics)
            self._frame_lock = threading.Lock()
            self.latest_raw_frame: Optional[np.ndarray] = None
            self.latest_annotated_frame: Optional[np.ndarray] = None
            self.latest_jpeg: Optional[bytes] = None
            self.latest_detected_leaves: List[DetectedLeaf] = []
            self.latest_qual_msg: str = "GOOD_QUALITY"
            self.live_fps: float = 0.0
            self.raw_capture_fps: float = 0.0

            # Thread Controls
            self.is_streaming = False
            self.capture_thread: Optional[threading.Thread] = None
            self.ai_thread: Optional[threading.Thread] = None
            self.encode_thread: Optional[threading.Thread] = None

            self.initialize()
            self._initialized = True

    def initialize(self) -> Tuple[bool, str]:
        """Automatically attempts to detect and initialize the camera hardware."""
        with self._lock:
            logger.info("Initializing Raspberry Pi Camera subsystem (640x480 @ 8-10 FPS)...")

            # 1. Attempt libcamera / Picamera2 initialization (RPi Camera Module ov5647)
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                cam_config = cam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                cam.configure(cam_config)
                cam.start()
                time.sleep(0.8)  # Allow sensor AE/AWB to stabilize

                self.picam2 = cam
                self.camera_type = "Picamera2 (libcamera ov5647)"
                self.is_connected = True
                self.error_message = None
                self.start_worker_threads()
                logger.info("Picamera2 ov5647 initialized successfully at 640x480.")
                return True, "Picamera2 initialized successfully."
            except Exception as e:
                logger.warning(f"Picamera2 initialization failed: {e}")
                self.picam2 = None

            # 2. Attempt standard V4L2 / USB webcam fallback
            import os
            if os.path.exists("/dev/video0"):
                try:
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 10)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            self.cv_cap = cap
                            self.camera_type = "USB / V4L2 Camera"
                            self.is_connected = True
                            self.error_message = None
                            self.start_worker_threads()
                            logger.info("USB / V4L2 camera detected and initialized at 640x480.")
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

    def start_worker_threads(self):
        """Starts dedicated decoupled worker threads."""
        if self.is_streaming:
            return
        self.is_streaming = True

        # Thread 1: Dedicated Hardware Capture Loop (locked at 8-10 FPS)
        self.capture_thread = threading.Thread(target=self._capture_worker, name="CamCaptureThread", daemon=True)
        self.capture_thread.start()

        # Thread 2: Asynchronous AI Inference & Tracking Loop (200ms cycle)
        self.ai_thread = threading.Thread(target=self._ai_worker, name="CamAIThread", daemon=True)
        self.ai_thread.start()

        # Thread 3: Annotation & JPEG Compression Loop (target 8-10 FPS)
        self.encode_thread = threading.Thread(target=self._encode_worker, name="CamEncodeThread", daemon=True)
        self.encode_thread.start()
        logger.info("Decoupled camera capture, AI inference, and streaming threads started.")

    def _capture_worker(self):
        """
        Dedicated capture thread:
        Runs at steady 8-10 FPS (100ms interval).
        Pushes raw frame into single-slot latest-frame buffer.
        Never blocked by AI inference or JPEG encoding.
        """
        frame_count = 0
        fps_t0 = time.time()
        consecutive_errors = 0

        while self.is_streaming and self.is_connected:
            loop_start = time.time()
            raw_bgr = None

            if self.picam2:
                try:
                    # Picamera2 capture_array returns RGB888 as configured
                    arr = self.picam2.capture_array()
                    # Convert RGB to BGR for standard OpenCV processing
                    raw_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    consecutive_errors = 0
                except Exception as e:
                    logger.error(f"Picamera2 capture error: {e}")
                    consecutive_errors += 1
            elif self.cv_cap and self.cv_cap.isOpened():
                try:
                    ret, frame = self.cv_cap.read()
                    if ret and frame is not None:
                        raw_bgr = frame
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                except Exception as e:
                    logger.error(f"OpenCV capture error: {e}")
                    consecutive_errors += 1

            if raw_bgr is not None:
                with self._frame_lock:
                    self.latest_raw_frame = raw_bgr

                frame_count += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    self.raw_capture_fps = frame_count / elapsed
                    frame_count = 0
                    fps_t0 = time.time()

            # Automatic error recovery if 8 consecutive frame failures
            if consecutive_errors >= 8:
                logger.warning("Camera hardware capture stalled, attempting auto-reconnect...")
                time.sleep(2.0)
                self._auto_reconnect()
                consecutive_errors = 0

            # Target 8-10 FPS pacing (~100-110 ms total cycle time)
            elapsed_loop = time.time() - loop_start
            sleep_needed = max(0.01, 0.105 - elapsed_loop)
            time.sleep(sleep_needed)

    def _ai_worker(self):
        """
        Asynchronous AI worker thread:
        Pulls latest raw frame, executes botanical leaf detection and tracking.
        Does not delay frame acquisition.
        """
        while self.is_streaming:
            frame_to_process = None
            with self._frame_lock:
                if self.latest_raw_frame is not None:
                    frame_to_process = self.latest_raw_frame.copy()

            if frame_to_process is not None:
                try:
                    candidates, qual_msg = self.detector.detect_leaf_candidates(frame_to_process)
                    detected = self.tracker.update(candidates)

                    self.latest_detected_leaves = detected
                    self.latest_qual_msg = qual_msg

                    # Check for Auto-Capture Trigger on Confirmed Leaf
                    now = time.time()
                    if self.auto_detect_enabled and (now - self.last_auto_capture_time >= self.auto_interval_sec):
                        for leaf in detected:
                            if leaf.status == "CONFIRMED_STABLE" and self.tracker.can_capture(leaf):
                                self.last_auto_capture_time = now
                                logger.info(f"Auto-capture triggered for verified leaf at {leaf.box}")
                                self._trigger_auto_capture(frame_to_process, leaf)
                                break
                except Exception as e:
                    logger.error(f"AI detection worker exception: {e}")

            # AI runs at ~4-5 Hz to conserve CPU on Raspberry Pi
            time.sleep(0.20)

    def _encode_worker(self):
        """
        Annotation and JPEG compression worker thread:
        Composes HUD and bounding boxes on latest frame at steady 8-10 FPS.
        Compacts output to JPEG quality 70 for smooth web transmission.
        """
        frame_count = 0
        fps_t0 = time.time()

        while self.is_streaming:
            loop_start = time.time()
            raw_bgr = None
            leaves = []
            qual = "GOOD_QUALITY"

            with self._frame_lock:
                if self.latest_raw_frame is not None:
                    raw_bgr = self.latest_raw_frame.copy()
                    leaves = list(self.latest_detected_leaves)
                    qual = self.latest_qual_msg

            if raw_bgr is not None:
                frame_count += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    self.live_fps = frame_count / elapsed
                    frame_count = 0
                    fps_t0 = time.time()

                # Render Botanical Annotations & HUD
                annotated = annotate_frame(
                    raw_bgr,
                    leaves,
                    quality_status=qual,
                    fps=self.live_fps if self.live_fps > 0 else self.raw_capture_fps,
                    auto_active=self.auto_detect_enabled
                )
                self.latest_annotated_frame = annotated

                # Encode JPEG at quality 70 (optimal low bandwidth, high fidelity)
                ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    self.latest_jpeg = jpeg.tobytes()

            # Target 8-10 FPS pacing
            elapsed_loop = time.time() - loop_start
            sleep_needed = max(0.01, 0.11 - elapsed_loop)
            time.sleep(sleep_needed)

    def _auto_reconnect(self):
        """Attempts to recover disconnected camera without crashing the service."""
        try:
            if self.picam2:
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
        except Exception:
            pass

        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam_config = cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            cam.configure(cam_config)
            cam.start()
            self.picam2 = cam
            self.is_connected = True
            self.error_message = None
            logger.info("Camera auto-reconnect succeeded.")
        except Exception as e:
            logger.error(f"Auto-reconnect failed: {e}")

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

        with self._frame_lock:
            if self.latest_raw_frame is not None:
                try:
                    cv2.imwrite(str(save_path), self.latest_raw_frame)
                    self.last_capture_path = save_path
                    return True, save_path, "Image captured successfully from live camera stream."
                except Exception as e:
                    logger.error(f"Frame save error: {e}")
                    return False, None, f"Capture error: {str(e)}"

        # Fallback direct Picam2 capture
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
            "resolution": "640x480",
            "target_fps": "8-10 FPS",
            "live_fps": round(self.live_fps if self.live_fps > 0 else self.raw_capture_fps, 1),
            "jpeg_quality": 70,
            "auto_detect_enabled": self.auto_detect_enabled,
            "auto_interval_sec": self.auto_interval_sec,
            "detected_leaves_count": len(self.latest_detected_leaves),
            "last_capture": str(self.last_capture_path) if self.last_capture_path else None,
            "error_message": self.error_message,
            "status_timestamp": datetime.now().isoformat()
        }

    def release(self):
        """Safely stops streaming threads and releases camera hardware."""
        self.is_streaming = False
        time.sleep(0.15)

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
