#!/usr/bin/env python3
"""
Plant Disease Detection AI - Desktop Camera Tester
Standalone GUI application for live camera testing, automated leaf capture,
edge AI inference, and dataset verification.
Compatible with Windows, Linux, and Raspberry Pi.
"""

import os
import sys
import time
import json
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Resolve base directories
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "Raspberry_Pi"))
try:
    from leaf_detector import comprehensive_leaf_verifier
except ImportError:
    comprehensive_leaf_verifier = None

AI_DIR = SCRIPT_DIR / "AI_Model"
TFLITE_MODEL_PATH = AI_DIR / "model.tflite"
KERAS_MODEL_PATH = AI_DIR / "best_chili_leaf_model.keras"
CLASS_NAMES_PATH = AI_DIR / "class_names.json"
RESULTS_DIR = SCRIPT_DIR / "test_results"
CAPTURES_DIR = RESULTS_DIR / "captures"
CSV_LOG_PATH = RESULTS_DIR / "camera_test_log.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Try loading AI inference backend
TFLITE_AVAILABLE = False
KERAS_AVAILABLE = False
Interpreter = None
tf = None

try:
    from ai_edge_litert.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        TFLITE_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            TFLITE_AVAILABLE = True
        except ImportError:
            pass

# OpenCV fallback for webcam
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Picamera2 for Raspberry Pi CSI
try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except ImportError:
    PICAM2_AVAILABLE = False

# Default classes
DEFAULT_CLASSES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf"
]

DISEASE_COLORS = {
    "Healthy_Leaf": "#27ae60",         # Vibrant Green
    "Bacterial_Spot": "#e67e22",       # Orange
    "Cercospora_Leaf_Spot": "#8e44ad", # Purple
    "Curl_Virus": "#c0392b",           # Deep Red
    "Unknown / Low Confidence": "#7f8c8d"  # Slate Gray
}

DISEASE_DESCRIPTIONS = {
    "Healthy_Leaf": "Optimal health. No treatment required. Maintain standard irrigation & nutrition.",
    "Bacterial_Spot": "Bacterial Spot: Apply Copper Oxychloride 50% WP (2.5 g/L) + Streptocycline.",
    "Cercospora_Leaf_Spot": "Cercospora (Frogeye): Spray Mancozeb 75% WP (2.5 g/L) or Azoxystrobin.",
    "Curl_Virus": "Curl Virus (ChiLCV): Whitefly vector management. Apply Neem Oil or Diafenthiuron. Rogue infected plants.",
    "Unknown / Low Confidence": "Confidence below threshold or leaf not clearly centered. Reposition leaf."
}


class ModelEngine:
    """Manages model loading and inference with LiteRT / TFLite or Keras."""

    def __init__(self):
        self.class_names = DEFAULT_CLASSES
        self.interpreter = None
        self.keras_model = None
        self.model_type = "None"
        self.input_shape = (160, 160)
        self.load_classes()
        self.load_model()

    def load_classes(self):
        if CLASS_NAMES_PATH.exists():
            try:
                with open(CLASS_NAMES_PATH, "r") as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(f"Error loading class names: {e}")

    def load_model(self) -> bool:
        # Prioritize lightweight TFLite (45ms inference, instant load)
        if TFLITE_AVAILABLE and TFLITE_MODEL_PATH.exists():
            try:
                self.interpreter = Interpreter(model_path=str(TFLITE_MODEL_PATH))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.model_type = "TFLite MobileNetV2"
                return True
            except Exception as e:
                print(f"TFLite load failed: {e}")

        # Fallback to Keras if TFLite failed
        if KERAS_MODEL_PATH.exists():
            try:
                import keras
                self.keras_model = keras.models.load_model(str(KERAS_MODEL_PATH))
                self.model_type = "Keras MobileNetV2"
                return True
            except Exception as e:
                print(f"Keras load failed: {e}")

        self.model_type = "Model Not Loaded"
        return False

    def predict(self, image: Image.Image) -> Tuple[str, float, float, Dict[str, float]]:
        """
        Two-stage AI diagnosis pipeline:
        Stage 1: Real Botanical Leaf Verification
        Stage 2: Model Classification (only if confirmed leaf)
        """
        t0 = time.time()

        # Stage 1: Real Botanical Leaf Verification
        if comprehensive_leaf_verifier is not None:
            img_np = np.asarray(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else img_np[:, :, ::-1]
            is_leaf, leaf_status, leaf_conf, det_meta = comprehensive_leaf_verifier(img_bgr)

            if not is_leaf:
                latency_ms = (time.time() - t0) * 1000.0
                return leaf_status, 0.0, latency_ms, {name: 0.0 for name in self.class_names}

        # Stage 2: Disease Classification
        resized = image.convert("RGB").resize(self.input_shape)
        arr = np.asarray(resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        scores = {}
        if self.model_type.startswith("TFLite") and self.interpreter:
            self.interpreter.set_tensor(self.input_details[0]["index"], arr)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        elif self.keras_model:
            preds = self.keras_model.predict(arr, verbose=0)[0]
        else:
            return "Model Error", 0.0, 0.0, {}

        latency_ms = (time.time() - t0) * 1000.0
        best_idx = int(np.argmax(preds))
        conf = float(preds[best_idx]) * 100.0
        label = self.class_names[best_idx]

        for i, name in enumerate(self.class_names):
            scores[name] = float(preds[i]) * 100.0

        return label, conf, latency_ms, scores


class CameraManager:
    """Hardware abstraction for live video capture across Windows, Linux, and Raspberry Pi."""

    def __init__(self):
        self.backend = None
        self.picam2 = None
        self.cv_cap = None
        self.is_running = False

    def start(self) -> bool:
        if self.is_running:
            return True

        # 1. Try Raspberry Pi Picamera2 first
        if PICAM2_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_preview_configuration(main={"size": (640, 480)})
                self.picam2.configure(cfg)
                self.picam2.start()
                self.backend = "Picamera2 (RPi CSI)"
                self.is_running = True
                return True
            except Exception as e:
                print(f"Picamera2 start error: {e}")
                if self.picam2:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                self.picam2 = None

        # 2. Try OpenCV webcam (Windows / Linux laptop webcam)
        if CV2_AVAILABLE:
            for cam_idx in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(cam_idx)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cv_cap = cap
                            self.backend = f"OpenCV Webcam (Index {cam_idx})"
                            self.is_running = True
                            return True
                        cap.release()
                except Exception as e:
                    print(f"OpenCV cam {cam_idx} error: {e}")

        return False

    def read_frame(self) -> Optional[Image.Image]:
        """Captures a single frame as PIL Image."""
        if not self.is_running:
            return None

        if self.backend == "Picamera2 (RPi CSI)" and self.picam2:
            try:
                bgra = self.picam2.capture_array()
                rgb = bgra[:, :, :3][:, :, ::-1]
                return Image.fromarray(rgb)
            except Exception as e:
                print(f"Picam2 capture error: {e}")
                return None

        if self.cv_cap and self.cv_cap.isOpened():
            try:
                ret, frame = self.cv_cap.read()
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb)
            except Exception as e:
                print(f"OpenCV capture error: {e}")
                return None

        return None

    def stop(self):
        self.is_running = False
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        if self.cv_cap:
            try:
                self.cv_cap.release()
            except Exception:
                pass
            self.cv_cap = None
        self.backend = None


class CameraTesterApp(tk.Tk):
    """Main Desktop GUI Application."""

    def __init__(self):
        super().__init__()
        self.title("Plant Disease Detection AI – Camera Tester")
        self.geometry("960x680")
        self.minsize(820, 580)
        self.configure(bg="#1e272e")

        self.model_engine = ModelEngine()
        self.camera_mgr = CameraManager()

        self.auto_capture_active = False
        self.auto_interval_sec = 3.0
        self.confidence_threshold = 70.0
        self.last_auto_capture_time = 0.0

        self.current_frame = None
        self.last_captured_image = None
        self.test_records = []

        self.setup_ui()
        self.init_csv()

        # Handle clean window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start live preview loop (30 fps)
        self.after(50, self.update_preview)

    def init_csv(self):
        if not CSV_LOG_PATH.exists():
            with open(CSV_LOG_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Source", "Predicted_Disease", "Confidence",
                    "Latency_ms", "Saved_Image_Path", "Status"
                ])

    def setup_ui(self):
        # 1. Header Banner
        header = tk.Frame(self, bg="#2f3542", height=55)
        header.pack(fill="x", side="top")

        title_lbl = tk.Label(
            header,
            text="🌿 Plant Disease Detection AI – Live Camera Tester",
            font=("Helvetica", 14, "bold"),
            bg="#2f3542",
            fg="#f1f2f6"
        )
        title_lbl.pack(side="left", padx=15, pady=10)

        self.model_status_lbl = tk.Label(
            header,
            text=f"Model: {self.model_engine.model_type}",
            font=("Helvetica", 9, "bold"),
            bg="#2ed573",
            fg="#2f3542",
            padx=10,
            pady=3
        )
        self.model_status_lbl.pack(side="right", padx=15, pady=12)

        # 2. Main Content Split (Left: Camera / Image | Right: Inference & Controls)
        content = tk.Frame(self, bg="#1e272e")
        content.pack(fill="both", expand=True, padx=12, pady=10)

        # Left Column: Video Preview
        left_col = tk.Frame(content, bg="#2f3640", bd=2, relief="groove")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 6))

        preview_header = tk.Frame(left_col, bg="#353b48")
        preview_header.pack(fill="x")
        tk.Label(
            preview_header,
            text="Live Camera Preview / Active Image",
            font=("Helvetica", 10, "bold"),
            bg="#353b48",
            fg="#dcdde1"
        ).pack(side="left", padx=10, pady=6)

        self.cam_badge = tk.Label(
            preview_header,
            text="Camera: Stopped",
            font=("Helvetica", 8, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=6,
            pady=2
        )
        self.cam_badge.pack(side="right", padx=10, pady=4)

        # Video Canvas
        self.preview_canvas = tk.Label(left_col, bg="#1e272e", text="Click [Start Camera] or [Load Dataset Image]", fg="#718093")
        self.preview_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Right Column: Controls & Prediction Card
        right_col = tk.Frame(content, bg="#2f3640", width=420, bd=2, relief="groove")
        right_col.pack(side="right", fill="both", padx=(6, 0))
        right_col.pack_propagate(False)

        # Prediction Display Box
        pred_box = tk.LabelFrame(
            right_col,
            text="AI Diagnosis Result",
            font=("Helvetica", 10, "bold"),
            bg="#2f3640",
            fg="#00d2d3",
            padx=10,
            pady=8
        )
        pred_box.pack(fill="x", padx=10, pady=8)

        self.disease_label = tk.Label(
            pred_box,
            text="Awaiting Capture...",
            font=("Helvetica", 15, "bold"),
            bg="#353b48",
            fg="#f5f6fa",
            pady=10
        )
        self.disease_label.pack(fill="x", pady=4)

        # Confidence Bar Frame
        conf_frame = tk.Frame(pred_box, bg="#2f3640")
        conf_frame.pack(fill="x", pady=4)

        self.conf_text_lbl = tk.Label(
            conf_frame,
            text="Confidence: 0.0%",
            font=("Helvetica", 10),
            bg="#2f3640",
            fg="#f5f6fa"
        )
        self.conf_text_lbl.pack(side="left")

        self.latency_lbl = tk.Label(
            conf_frame,
            text="Latency: -- ms",
            font=("Helvetica", 10),
            bg="#2f3640",
            fg="#bdc581"
        )
        self.latency_lbl.pack(side="right")

        self.conf_bar = ttk.Progressbar(pred_box, orient="horizontal", mode="determinate")
        self.conf_bar.pack(fill="x", pady=4)

        self.guide_lbl = tk.Label(
            pred_box,
            text="Position a chili leaf in the camera viewfinder and click Capture & Classify.",
            font=("Helvetica", 9),
            bg="#2f3640",
            fg="#dcdde1",
            wraplength=380,
            justify="left"
        )
        self.guide_lbl.pack(fill="x", pady=4)

        # Controls Frame
        ctrl_frame = tk.LabelFrame(
            right_col,
            text="Camera & Capture Controls",
            font=("Helvetica", 10, "bold"),
            bg="#2f3640",
            fg="#48dbfb",
            padx=10,
            pady=8
        )
        ctrl_frame.pack(fill="x", padx=10, pady=4)

        cam_btn_row = tk.Frame(ctrl_frame, bg="#2f3640")
        cam_btn_row.pack(fill="x", pady=3)

        self.start_cam_btn = tk.Button(
            cam_btn_row,
            text="▶ Start Camera",
            bg="#20bf6b",
            fg="white",
            font=("Helvetica", 9, "bold"),
            command=self.start_camera
        )
        self.start_cam_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.stop_cam_btn = tk.Button(
            cam_btn_row,
            text="⏹ Stop Camera",
            bg="#eb3b5a",
            fg="white",
            font=("Helvetica", 9, "bold"),
            state="disabled",
            command=self.stop_camera
        )
        self.stop_cam_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # Manual Buttons
        self.capture_btn = tk.Button(
            ctrl_frame,
            text="📸 Capture and Classify Now",
            bg="#3867d6",
            fg="white",
            font=("Helvetica", 10, "bold"),
            command=self.manual_capture
        )
        self.capture_btn.pack(fill="x", pady=4)

        self.load_img_btn = tk.Button(
            ctrl_frame,
            text="📁 Select & Test Dataset Image",
            bg="#4b6584",
            fg="white",
            font=("Helvetica", 9),
            command=self.load_dataset_image
        )
        self.load_img_btn.pack(fill="x", pady=3)

        # Auto Capture Settings
        auto_row = tk.Frame(ctrl_frame, bg="#2f3640")
        auto_row.pack(fill="x", pady=4)

        self.auto_capture_var = tk.BooleanVar(value=False)
        self.auto_chk = tk.Checkbutton(
            auto_row,
            text="Auto-Capture Every:",
            variable=self.auto_capture_var,
            font=("Helvetica", 9),
            bg="#2f3640",
            fg="#f5f6fa",
            selectcolor="#353b48",
            activebackground="#2f3640",
            activeforeground="#f5f6fa",
            command=self.toggle_auto_capture
        )
        self.auto_chk.pack(side="left")

        self.interval_spin = ttk.Spinbox(
            auto_row,
            from_=1.0,
            to=10.0,
            increment=0.5,
            width=5
        )
        self.interval_spin.set("3.0")
        self.interval_spin.pack(side="left", padx=5)
        tk.Label(auto_row, text="sec", font=("Helvetica", 9), bg="#2f3640", fg="#dcdde1").pack(side="left")

        # Threshold slider
        thresh_row = tk.Frame(ctrl_frame, bg="#2f3640")
        thresh_row.pack(fill="x", pady=4)

        tk.Label(thresh_row, text="Min Confidence Threshold:", font=("Helvetica", 9), bg="#2f3640", fg="#dcdde1").pack(side="left")
        self.thresh_val_lbl = tk.Label(thresh_row, text="70%", font=("Helvetica", 9, "bold"), bg="#2f3640", fg="#f5f6fa")
        self.thresh_val_lbl.pack(side="right")

        self.thresh_scale = ttk.Scale(
            ctrl_frame,
            from_=40.0,
            to=95.0,
            value=70.0,
            command=self.on_thresh_change
        )
        self.thresh_scale.pack(fill="x", pady=2)

        # 3. Bottom Log / History Table
        bottom_frame = tk.LabelFrame(
            right_col,
            text="Session Test History",
            font=("Helvetica", 9, "bold"),
            bg="#2f3640",
            fg="#a5b1c2",
            padx=5,
            pady=5
        )
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        columns = ("Time", "Source", "Prediction", "Confidence")
        self.history_tree = ttk.Treeview(bottom_frame, columns=columns, show="headings", height=5)
        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("Source", text="Source")
        self.history_tree.heading("Prediction", text="Prediction")
        self.history_tree.heading("Confidence", text="Confidence")

        self.history_tree.column("Time", width=70)
        self.history_tree.column("Source", width=50)
        self.history_tree.column("Prediction", width=120)
        self.history_tree.column("Confidence", width=70)
        self.history_tree.pack(fill="both", expand=True)

    def on_thresh_change(self, val):
        self.confidence_threshold = float(val)
        self.thresh_val_lbl.config(text=f"{int(self.confidence_threshold)}%")

    def toggle_auto_capture(self):
        self.auto_capture_active = self.auto_capture_var.get()
        try:
            self.auto_interval_sec = max(1.0, float(self.interval_spin.get()))
        except ValueError:
            self.auto_interval_sec = 3.0

    def start_camera(self):
        if self.camera_mgr.start():
            self.start_cam_btn.config(state="disabled")
            self.stop_cam_btn.config(state="normal")
            self.cam_badge.config(text=f"Camera: {self.camera_mgr.backend}", bg="#20bf6b")
        else:
            messagebox.showwarning(
                "Camera Not Found",
                "Could not connect to a webcam or Raspberry Pi camera.\n"
                "You can still use [Select & Test Dataset Image] to verify the model!"
            )

    def stop_camera(self):
        self.camera_mgr.stop()
        self.start_cam_btn.config(state="normal")
        self.stop_cam_btn.config(state="disabled")
        self.cam_badge.config(text="Camera: Stopped", bg="#e74c3c")
        self.auto_capture_var.set(False)
        self.auto_capture_active = False

    def update_preview(self):
        now = time.time()
        if self.camera_mgr.is_running:
            frame = self.camera_mgr.read_frame()
            if frame:
                self.current_frame = frame
                # Resize for canvas fit
                cw = self.preview_canvas.winfo_width() or 460
                ch = self.preview_canvas.winfo_height() or 340
                disp_img = frame.copy()
                disp_img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(disp_img)
                self.preview_canvas.config(image=self.tk_img, text="")

                # Check auto capture
                if self.auto_capture_active and (now - self.last_auto_capture_time >= self.auto_interval_sec):
                    self.last_auto_capture_time = now
                    self.classify_image(frame, source="Auto-Cam")

        self.after(33, self.update_preview)

    def manual_capture(self):
        if self.current_frame is not None and self.camera_mgr.is_running:
            self.classify_image(self.current_frame, source="Live-Cam")
        elif self.last_captured_image is not None:
            self.classify_image(self.last_captured_image, source="Active-Img")
        else:
            # If camera is not started, prompt user
            messagebox.showinfo("Camera Required", "Please click [Start Camera] or [Select & Test Dataset Image] first.")

    def load_dataset_image(self):
        initial_dir = SCRIPT_DIR / "Dataset"
        if not initial_dir.exists():
            initial_dir = SCRIPT_DIR

        path = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Select Chili Leaf Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if path:
            try:
                img = Image.open(path)
                self.current_frame = img
                self.last_captured_image = img

                # Display loaded image
                cw = self.preview_canvas.winfo_width() or 460
                ch = self.preview_canvas.winfo_height() or 340
                disp_img = img.copy()
                disp_img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(disp_img)
                self.preview_canvas.config(image=self.tk_img, text="")

                self.classify_image(img, source=f"File: {Path(path).name[:10]}")
            except Exception as e:
                messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

    def classify_image(self, img: Image.Image, source: str = "Live"):
        # Run inference in a fast thread so UI does not stutter
        def run_inf():
            raw_label, conf, latency, all_scores = self.model_engine.predict(img)

            # Check threshold
            if conf < self.confidence_threshold:
                display_label = "Unknown / Low Confidence"
                guide = f"Confidence {conf:.1f}% is below {int(self.confidence_threshold)}% threshold. Leaf image is ambiguous or not centered."
            else:
                display_label = raw_label
                guide = DISEASE_DESCRIPTIONS.get(raw_label, "")

            # Save snapshot
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            time_display = datetime.now().strftime("%H:%M:%S")
            img_save_name = f"{timestamp_str}_{raw_label}_{int(conf)}pct.jpg"
            img_save_path = CAPTURES_DIR / img_save_name
            try:
                img.save(img_save_path)
            except Exception:
                pass

            # Write to CSV log
            try:
                with open(CSV_LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(), source, display_label,
                        f"{conf:.2f}%", f"{latency:.1f}ms", str(img_save_path),
                        "VERIFIED" if conf >= self.confidence_threshold else "LOW_CONFIDENCE"
                    ])
            except Exception as e:
                print(f"CSV log error: {e}")

            # Update GUI on main thread
            self.after(0, lambda: self.update_prediction_ui(
                display_label, conf, latency, guide, time_display, source
            ))

        threading.Thread(target=run_inf, daemon=True).start()

    def update_prediction_ui(self, label, conf, latency, guide, time_display, source):
        color = DISEASE_COLORS.get(label, "#353b48")
        self.disease_label.config(text=label.replace("_", " "), bg=color)
        self.conf_text_lbl.config(text=f"Confidence: {conf:.1f}%")
        self.latency_lbl.config(text=f"Latency: {latency:.1f} ms")
        self.conf_bar["value"] = conf
        self.guide_lbl.config(text=guide)

        # Add to history tree
        self.history_tree.insert(
            "",
            0,
            values=(time_display, source, label.replace("_", " "), f"{conf:.1f}%")
        )

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = CameraTesterApp()
    app.mainloop()
