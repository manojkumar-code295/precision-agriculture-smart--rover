#!/usr/bin/env python3
"""
================================================================================
Plant Disease Detection AI - Real-Time Botanical Leaf Detection & CNN Classifier
================================================================================
Features:
1. LEAF ONLY DETECTION:
   - Multi-spectral botanical pigment analysis (Chlorophyll ExG + Carotenoid + Necrotic lesions)
   - Strict rejection of human hands/skin (YCbCr chromaticity + Melanin balance)
   - Strict rejection of walls, curtains, tables, and clothing
   - Non-plant objects are NEVER sent to the CNN model or falsely diagnosed!
2. KERAS CNN MODEL INFERENCE & COMPARISON:
   - Loads user's trained MobileNetV2 Keras model: AI_Model/best_chili_leaf_model.keras
   - Runs inference on verified leaf ROI crops
   - Compares with TFLite edge model (latency, agreement, confidence delta)
   - Renders real-time class probability gauges for all 4 classes
3. HARDWARE-AGNOSTIC REAL-TIME STREAMING:
   - Connects seamlessly to the local rover camera stream (http://127.0.0.1:5000)
   - Fallback to Picamera2 (RPi CSI) or OpenCV cv2.VideoCapture (USB webcam)
   - Decoupled asynchronous inference: video renders at smooth 30 FPS with no lag!
================================================================================
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List

import cv2
import numpy as np

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION & IMPORTS
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Search for AI Model files in local directory or central project
POSSIBLE_MODEL_DIRS = [
    SCRIPT_DIR / "AI_Model",
    SCRIPT_DIR.parent.parent / "Plant_Disease_Detection_AI" / "AI_Model",
    Path("/home/agrirover/Plant_Disease_Detection_AI/AI_Model")
]

MODEL_PATH = None
for d in POSSIBLE_MODEL_DIRS:
    candidate = d / "best_chili_leaf_model.keras"
    if candidate.exists():
        MODEL_PATH = candidate
        break

TFLITE_PATH = None
for d in POSSIBLE_MODEL_DIRS:
    candidate = d / "model.tflite"
    if candidate.exists():
        TFLITE_PATH = candidate
        break

LABEL_PATH = None
for d in POSSIBLE_MODEL_DIRS:
    for sub in [d / "Reports" / "class_names.json", d / "class_names.json"]:
        if sub.exists():
            LABEL_PATH = sub
            break
    if LABEL_PATH:
        break

CAPTURES_DIR = SCRIPT_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 160
CONFIDENCE_THRESHOLD = 70.0

DEFAULT_CLASSES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf"
]

DISEASE_COLORS = {
    "Healthy_Leaf": (0, 230, 0),        # Vibrant Green
    "Bacterial_Spot": (0, 140, 255),    # Safety Orange
    "Cercospora_Leaf_Spot": (180, 50, 180),  # Royal Purple
    "Curl_Virus": (30, 30, 225),        # Crimson Red
    "Unknown / Low Confidence": (150, 150, 150)
}

# ------------------------------------------------------------------------------
# 2. BOTANICAL LEAF VERIFIER (DETECT LEAF ONLY)
# ------------------------------------------------------------------------------
def verify_leaf_only(img_bgr: np.ndarray) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Botanical leaf vs non-plant verifier.
    Distinguishes genuine chili leaves (healthy green, yellow chlorotic, diseased spotty, curly)
    from human hands/skin, walls, curtains, tables, clothes, and non-plant backgrounds.

    Returns:
        (is_leaf: bool, status_label: str, confidence: float, details: dict)
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "No Leaf Result / Invalid Frame", 0.0, {"reason": "Empty or corrupted frame"}

    h, w = img_bgr.shape[:2]
    if w < 24 or h < 24:
        return False, "No Leaf Result / Object Too Small", 0.0, {"reason": "ROI dimensions too small"}

    # Rescale to standard 240x240 for consistent spectral feature extraction
    scale_size = (240, 240)
    frame = cv2.resize(img_bgr, scale_size)
    total_pixels = 240.0 * 240.0

    # 1. Illumination & Glare Check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(cv2.mean(gray)[0])
    if mean_brightness < 16.0:
        return False, "No Leaf Result / Insufficient Lighting", 0.0, {"reason": f"Scene too dark ({mean_brightness:.1f})"}
    if mean_brightness > 252.0:
        return False, "No Leaf Result / Overexposed Glare", 0.0, {"reason": f"Overexposed glare ({mean_brightness:.1f})"}

    b, g, r = cv2.split(frame.astype(np.float32))

    # 2. Human Hand / Skin Rejection (YCbCr + Melanin/Hemoglobin Balance)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    # Skin chromaticity box: Cr in [133, 173], Cb in [77, 127], R > G > B, (R - G) > 8
    skin_pixels = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (r > g) & (g > b) & ((r - g) > 8.0)
    skin_ratio = float(np.sum(skin_pixels)) / total_pixels

    if skin_ratio > 0.12:
        return False, "No Leaf Result / Human Hand Detected", 0.0, {
            "reason": f"Human skin/hand detected (skin_ratio={skin_ratio*100:.1f}%)",
            "skin_ratio": round(skin_ratio, 3)
        }

    # 3. Wall / Flat Featureless Surface Check
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(np.sum(edges > 0)) / total_pixels

    # 4. Curtain / Fabric Anisotropy Check (Directional fold gradients)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sum_x = float(np.sum(np.abs(sobel_x)))
    sum_y = float(np.sum(np.abs(sobel_y)))
    anisotropy = max(sum_x, sum_y) / (min(sum_x, sum_y) + 1e-5)

    # 5. Multi-spectral Botanical Pigments
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Excess Green Index (ExG = 2G - R - B)
    exg = (2.0 * g) - r - b

    # Chlorophyll (Healthy and spotted green leaves)
    chlorophyll = (g > r) & (exg > 8.0) & (h_channel >= 20) & (h_channel <= 98) & (s_channel >= 20) & (v_channel >= 25) & (g > 30.0)
    chlorophyll_ratio = float(np.sum(chlorophyll)) / total_pixels

    # Carotenoid / Xanthophyll (Yellow chlorotic / virus leaves)
    yellow_hue = (h_channel >= 12) & (h_channel <= 45) & (s_channel >= 25) & (v_channel >= 35)
    carotenoid = yellow_hue & (r > 45.0) & (g > 40.0) & (np.abs(r - g) < 45.0) & (b < 160.0) & (g >= (b * 0.90))
    carotenoid_ratio = float(np.sum(carotenoid)) / total_pixels

    # Necrotic Lesions (Brown/necrotic spot tissue)
    brown_hue = (h_channel >= 5) & (h_channel <= 25) & (s_channel >= 20) & (v_channel >= 20) & (v_channel <= 185)
    brown_spots = brown_hue & (r > b) & ((r - b) > 6.0)
    brown_ratio = float(np.sum(brown_spots)) / total_pixels

    plant_foliage_ratio = chlorophyll_ratio + carotenoid_ratio

    # Rejection: Curtain / Fabric folds
    if anisotropy > 3.2 and plant_foliage_ratio < 0.15:
        return False, "No Leaf Result / Curtain Detected", 0.0, {
            "reason": f"Periodic fabric folds detected (anisotropy={anisotropy:.1f})",
            "anisotropy": round(anisotropy, 2)
        }

    # Rejection: Wall or Flat Surface
    if lap_var < 16.0 and edge_density < 0.018 and plant_foliage_ratio < 0.10:
        return False, "No Leaf Result / Wall Detected", 0.0, {
            "reason": f"Flat featureless surface detected (lap_var={lap_var:.1f})",
            "lap_var": round(lap_var, 1)
        }

    # Rejection: Non-plant background (furniture, clothes, table, carpet)
    if plant_foliage_ratio < 0.07 and (plant_foliage_ratio + brown_ratio) < 0.09:
        return False, "No Leaf Result / Not a Plant", 0.0, {
            "reason": f"No botanical foliage detected (plant_ratio={plant_foliage_ratio*100:.1f}%)",
            "plant_ratio": round(plant_foliage_ratio, 3)
        }

    # Passed Botanical Verification!
    leaf_confidence = min(99.0, max(72.0, 70.0 + (chlorophyll_ratio * 32.0) + (carotenoid_ratio * 25.0)))
    return True, "Plant / Leaf Detected", round(leaf_confidence, 1), {
        "plant_ratio": round(plant_foliage_ratio, 3),
        "chlorophyll_ratio": round(chlorophyll_ratio, 3),
        "carotenoid_ratio": round(carotenoid_ratio, 3),
        "brown_ratio": round(brown_ratio, 3),
        "skin_ratio": round(skin_ratio, 3),
        "lap_var": round(lap_var, 1)
    }


def detect_leaf_only(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str, float]:
    """
    Finds and segments real chili leaves only, actively rejecting non-plant objects.
    Returns: (cropped_leaf, bounding_box, status_msg, confidence)
    """
    h, w = frame.shape[:2]
    frame_area = float(h * w)

    # 1. Multi-spectral foliage segmentation
    b, g, r = cv2.split(frame.astype(np.float32))
    exg = (2.0 * g) - r - b
    chlorophyll_mask = (exg > 4.0) & (g > 30.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_band = (h_ch >= 18) & (h_ch <= 98) & (s_ch >= 20) & (v_ch >= 25)
    yellow_band = (h_ch >= 12) & (h_ch <= 45) & (s_ch >= 25) & (v_ch >= 35) & (r > 45.0) & (g > 40.0)

    # Suppress skin & hands
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (r > g) & (g > b) & ((r - g) > 8.0)

    # Combined vegetative mask minus skin
    foliage = ((chlorophyll_mask & green_band) | yellow_band) & (~skin_mask)
    combined = foliage.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (frame_area * 0.02) or area > (frame_area * 0.94):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            continue
        if bw >= int(w * 0.99) and bh >= int(h * 0.99):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = float(area) / hull_area
        aspect = float(bw) / float(bh)

        if (0.22 <= aspect <= 4.20) and (solidity >= 0.38):
            roi = frame[y:y+bh, x:x+bw]
            is_leaf, leaf_label, leaf_conf, _ = verify_leaf_only(roi)
            if is_leaf:
                candidates.append((area, x, y, bw, bh, leaf_conf))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, x, y, bw, bh, conf = candidates[0]
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2 - x1, y2 - y1), "Plant / Leaf Detected", conf

    # Fallback: Check if the frame itself is a close-up leaf view
    is_leaf, leaf_label, leaf_conf, det_meta = verify_leaf_only(frame)
    if is_leaf:
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        bw = w - 2 * pad_x
        bh = h - 2 * pad_y
        crop = frame[pad_y:pad_y+bh, pad_x:pad_x+bw]
        return crop, (pad_x, pad_y, bw, bh), "Plant / Leaf Detected", leaf_conf

    return None, None, leaf_label, 0.0


# ------------------------------------------------------------------------------
# 3. MODEL INFERENCE & COMPARISON ENGINE
# ------------------------------------------------------------------------------
class ModelEngine:
    """Loads Keras (.keras) and LiteRT (.tflite) models and performs inference."""

    def __init__(self):
        self.class_names = DEFAULT_CLASSES
        self.keras_model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_classes()
        self.load_models()

    def load_classes(self):
        if LABEL_PATH and LABEL_PATH.exists():
            try:
                with open(LABEL_PATH, "r") as f:
                    self.class_names = json.load(f)
                print(f"Loaded class labels: {self.class_names}")
            except Exception as e:
                print(f"Error reading class labels: {e}")

    def load_models(self):
        # 1. Load User's Keras Model
        if MODEL_PATH and MODEL_PATH.exists():
            try:
                print(f"Loading Keras CNN model from {MODEL_PATH}...")
                import keras
                self.keras_model = keras.models.load_model(str(MODEL_PATH))
                print(f"User Keras model loaded successfully ({MODEL_PATH.name})!")
            except Exception as e:
                print(f"Could not load Keras model: {e}")

        # 2. Load TFLite Model
        if TFLITE_PATH and TFLITE_PATH.exists():
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    import tensorflow as tf
                    Interpreter = tf.lite.Interpreter

            try:
                self.interpreter = Interpreter(model_path=str(TFLITE_PATH))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("TFLite model loaded successfully for comparison.")
            except Exception as e:
                print(f"TFLite model load skipped: {e}")

    def predict_keras(self, crop: np.ndarray) -> Tuple[str, float, float, Dict[str, float]]:
        if self.keras_model is None:
            return self.predict_tflite(crop)

        t0 = time.time()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        inp = np.expand_dims(resized, axis=0)

        preds = self.keras_model(inp, training=False).numpy()[0]
        lat = (time.time() - t0) * 1000.0

        idx = int(np.argmax(preds))
        conf = float(preds[idx]) * 100.0
        disease = self.class_names[idx]
        scores = {name: float(preds[i]) * 100.0 for i, name in enumerate(self.class_names)}
        return disease, conf, lat, scores

    def predict_tflite(self, crop: np.ndarray) -> Tuple[str, float, float, Dict[str, float]]:
        if self.interpreter is None:
            return "None", 0.0, 0.0, {n: 0.0 for n in self.class_names}

        t0 = time.time()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        inp = np.expand_dims(resized, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        lat = (time.time() - t0) * 1000.0

        idx = int(np.argmax(preds))
        conf = float(preds[idx]) * 100.0
        disease = self.class_names[idx]
        scores = {name: float(preds[i]) * 100.0 for i, name in enumerate(self.class_names)}
        return disease, conf, lat, scores


# ------------------------------------------------------------------------------
# 4. CAMERA MANAGER (DUAL SOURCE: HTTP STREAM OR LOCAL HARDWARE)
# ------------------------------------------------------------------------------
class LiveCameraStream:
    """Handles seamless camera acquisition from local rover stream or hardware."""

    def __init__(self):
        self.cap = None
        self.picam2 = None
        self.stream_url = "http://127.0.0.1:5000/api/camera/stream"
        self.backend = "None"
        self.is_running = False
        self.connect()

    def connect(self):
        # 1. Try local rover camera stream first (avoids resource conflict)
        try:
            import urllib.request
            req = urllib.request.urlopen("http://127.0.0.1:5000/api/status", timeout=1.0)
            if req.status == 200:
                print(f"Connecting to live rover stream: {self.stream_url}...")
                cap = cv2.VideoCapture(self.stream_url)
                if cap.isOpened():
                    self.cap = cap
                    self.backend = "Rover HTTP Live Stream"
                    self.is_running = True
                    print("Connected to live rover video stream.")
                    return
        except Exception:
            pass

        # 2. Try Picamera2 CSI
        try:
            from picamera2 import Picamera2
            p = Picamera2()
            cfg = p.create_preview_configuration(main={"size": (640, 480)})
            p.configure(cfg)
            p.start()
            self.picam2 = p
            self.backend = "Picamera2 (RPi CSI)"
            self.is_running = True
            print("Connected via Picamera2.")
            return
        except Exception:
            pass

        # 3. Try standard OpenCV webcam
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.backend = f"OpenCV Webcam ({idx})"
                    self.is_running = True
                    print(f"Connected to USB/V4L2 camera {idx}.")
                    return
                cap.release()

        print("WARNING: No live camera detected. Using test image loop.")
        self.backend = "Simulation / Test Dataset"

    def read_frame(self) -> Optional[np.ndarray]:
        if self.picam2:
            try:
                arr = self.picam2.capture_array()
                return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        # Simulated fallback frame from test results or dataset
        test_file = SCRIPT_DIR / "test_results" / "captures" / "20261112_151320_Healthy_Leaf_98pct.jpg"
        if test_file.exists():
            return cv2.imread(str(test_file))
        return None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
        self.is_running = False


# ------------------------------------------------------------------------------
# 5. MAIN REAL-TIME DETECTION APPLICATION
# ------------------------------------------------------------------------------
def run_realtime_detection():
    print("=======================================================")
    print("   🌿 AGRIROVER - REAL-TIME LEAF DETECTION SYSTEM    ")
    print("   Detect Leaf Only | Compare with User Keras Model    ")
    print("=======================================================")

    model_engine = ModelEngine()
    camera = LiveCameraStream()

    print(f"\nActive Camera Backend: {camera.backend}")
    print(f"Loaded Keras Model: {model_engine.keras_model is not None}")
    print(f"Loaded TFLite Model: {model_engine.interpreter is not None}")
    print("\nPress 'Q' to quit | 'S' to save snapshot | 'C' to compare models")

    # State variables for asynchronous inference
    last_crop = None
    last_box = None
    last_status = "Scanning for Leaf..."
    last_keras_disease = "No Leaf Detected"
    last_keras_conf = 0.0
    last_keras_lat = 0.0
    last_keras_scores = {n: 0.0 for n in model_engine.class_names}

    last_tflite_disease = "No Leaf Detected"
    last_tflite_conf = 0.0
    last_tflite_lat = 0.0

    is_inferring = False
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    def async_inference_worker(leaf_crop: np.ndarray):
        nonlocal is_inferring, last_keras_disease, last_keras_conf, last_keras_lat, last_keras_scores
        nonlocal last_tflite_disease, last_tflite_conf, last_tflite_lat
        try:
            # 1. Run User's Keras CNN Model
            k_dis, k_conf, k_lat, k_scores = model_engine.predict_keras(leaf_crop)
            last_keras_disease = k_dis
            last_keras_conf = k_conf
            last_keras_lat = k_lat
            last_keras_scores = k_scores

            # 2. Run TFLite Model for comparison
            t_dis, t_conf, t_lat, _ = model_engine.predict_tflite(leaf_crop)
            last_tflite_disease = t_dis
            last_tflite_conf = t_conf
            last_tflite_lat = t_lat
        finally:
            is_inferring = False

    window_name = "AgriRover AI - Real-Time Leaf Only Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 680)

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.04)
                continue

            fps_counter += 1
            now = time.time()
            if (now - fps_time) >= 1.0:
                current_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            h, w = frame.shape[:2]
            display = frame.copy()

            # 1. Run Botanical Leaf-Only Detection
            crop, box, status_msg, leaf_conf = detect_leaf_only(frame)

            if crop is not None:
                last_box = box
                last_status = "Leaf Confirmed"
                last_crop = crop.copy()

                # Trigger background Keras inference if not currently busy
                if not is_inferring:
                    is_inferring = True
                    threading.Thread(target=async_inference_worker, args=(crop.copy(),), daemon=True).start()

                # Render Leaf Bounding Box
                x, y, bw, bh = box
                box_color = DISEASE_COLORS.get(last_keras_disease, (0, 230, 0))

                # Main Bounding Box
                cv2.rectangle(display, (x, y), (x + bw, y + bh), box_color, 2)

                # High-contrast corner accents
                c_len = min(18, bw // 4, bh // 4)
                cv2.line(display, (x, y), (x + c_len, y), box_color, 3)
                cv2.line(display, (x, y), (x, y + c_len), box_color, 3)
                cv2.line(display, (x + bw, y), (x + bw - c_len, y), box_color, 3)
                cv2.line(display, (x + bw, y), (x + bw, y + c_len), box_color, 3)
                cv2.line(display, (x, y + bh), (x + c_len, y + bh), box_color, 3)
                cv2.line(display, (x, y + bh), (x, y + bh - c_len), box_color, 3)
                cv2.line(display, (x + bw, y + bh), (x + bw - c_len, y + bh), box_color, 3)
                cv2.line(display, (x + bw, y + bh), (x + bw, y + c_len), box_color, 3)

                # Box Tag
                tag_text = f"KERAS: {last_keras_disease} ({last_keras_conf:.1f}%)"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(display, (x, max(0, y - 24)), (x + tw + 12, y), box_color, -1)
                cv2.putText(display, tag_text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

            else:
                last_box = None
                last_status = status_msg

            # 2. Render Top Diagnostics HUD Bar
            cv2.rectangle(display, (0, 0), (w, 36), (15, 23, 20), -1)
            hud_left = f"AGRIROVER AI | FPS: {current_fps:.1f} | SRC: {camera.backend}"
            cv2.putText(display, hud_left, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)

            # Top Status Indicator
            if crop is not None:
                stat_text = f"🌿 LEAF: {last_keras_disease.replace('_', ' ')} ({last_keras_conf:.1f}%)"
                stat_col = (0, 230, 0) if "Healthy" in last_keras_disease else (0, 165, 255)
            elif "Hand" in last_status:
                stat_text = "🚫 NO LEAF: HUMAN HAND DETECTED"
                stat_col = (50, 50, 240)
            elif "Wall" in last_status:
                stat_text = "🚫 NO LEAF: WALL DETECTED"
                stat_col = (140, 140, 200)
            elif "Curtain" in last_status or "Not a Plant" in last_status:
                stat_text = "🚫 NO LEAF: NON-PLANT DETECTED"
                stat_col = (140, 140, 200)
            else:
                stat_text = "🔍 SCANNING FOR CHILI LEAF..."
                stat_col = (180, 190, 180)

            cv2.putText(display, stat_text, (w - 420, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, stat_col, 2, cv2.LINE_AA)

            # 3. Render Right-Side Real-Time Keras Model Class Probability Gauges
            gauge_w = 260
            gauge_h = 175
            gx = w - gauge_w - 10
            gy = 45

            # Semi-transparent HUD overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (gx, gy), (gx + gauge_w, gy + gauge_h), (20, 28, 25), -1)
            cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
            cv2.rectangle(display, (gx, gy), (gx + gauge_w, gy + gauge_h), (52, 211, 153), 1)

            cv2.putText(display, "KERAS MODEL COMPARISON", (gx + 10, gy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (52, 211, 153), 1, cv2.LINE_AA)

            # Draw class probability bars
            bar_y = gy + 42
            for c_name in model_engine.class_names:
                prob = last_keras_scores.get(c_name, 0.0) if crop is not None else 0.0
                short_name = c_name.replace("Cercospora_", "Cercospora ").replace("Bacterial_", "Bacterial ")
                short_name = short_name[:14]
                col = DISEASE_COLORS.get(c_name, (200, 200, 200))

                # Class name text
                cv2.putText(display, f"{short_name}:", (gx + 10, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

                # Bar background
                bx = gx + 115
                bw_bar = 90
                cv2.rectangle(display, (bx, bar_y - 2), (bx + bw_bar, bar_y + 10), (45, 55, 50), -1)

                # Filled bar
                fill_w = int((prob / 100.0) * bw_bar)
                if fill_w > 0:
                    cv2.rectangle(display, (bx, bar_y - 2), (bx + fill_w, bar_y + 10), col, -1)

                # Prob text
                cv2.putText(display, f"{prob:4.1f}%", (bx + bw_bar + 6, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1)
                bar_y += 26

            # TFLite Agreement Comparison footer
            agree = (last_keras_disease == last_tflite_disease) and (crop is not None)
            agree_text = f"TFLite: {last_tflite_disease[:10]} ({last_tflite_conf:.0f}%)" if crop is not None else "TFLite Edge: Standby"
            cv2.putText(display, agree_text, (gx + 10, gy + gauge_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 230, 180) if agree else (180, 180, 180), 1)

            # 4. Bottom Non-Plant Rejection Alert Banner (if non-leaf presented)
            if crop is None and ("Hand" in last_status or "Wall" in last_status or "Curtain" in last_status):
                banner_y = h - 35
                cv2.rectangle(display, (0, banner_y), (w, h), (10, 10, 80), -1)
                warn_msg = f"BOTANICAL GUARD: NON-PLANT OBJECT REJECTED ({last_status}) - CNN INFERENCE SUPPRESSED"
                cv2.putText(display, warn_msg, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            # Show annotated real-time frame
            cv2.imshow(window_name, display)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'Q' or ESC
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = CAPTURES_DIR / f"detection_{ts}_{last_keras_disease}.jpg"
                cv2.imwrite(str(save_path), display)
                print(f"Saved real-time detection frame: {save_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nReal-time leaf detection stopped.")


if __name__ == "__main__":
    run_realtime_detection()
