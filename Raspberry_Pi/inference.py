"""
Precision Agriculture Smart Rover - AI Inference Engine
Executes local edge inference using TensorFlow Lite on Raspberry Pi 4B.
Identifies: Bacterial Spot, Cercospora Leaf Spot, Curl Virus, and Healthy Leaf.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

import config

logger = logging.getLogger("LeafInference")

# Treatment and Agricultural Management Database (India Agriculture Standard)
DISEASE_KNOWLEDGE_BASE = {
    "Bacterial_Spot": {
        "disease_name": "Bacterial Leaf Spot",
        "causal_organism": "Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small, water-soaked dark lesions on leaves, turning brown with yellow halos. Leaves eventually drop prematurely.",
        "treatment_type": "Bactericide / Protective Copper Spray",
        "recommended_chemicals": "Copper Oxychloride 50% WP @ 2.5 g/L OR Copper Hydroxide 77% WP @ 2.0 g/L, combined with Streptocycline @ 1 g/10 L of water.",
        "management_guidance": "Avoid overhead sprinkler irrigation to stop water-splash spread. Prune and safely destroy lower infected leaves. Disinfect rover wheels and tools.",
        "safety_warning": "Observe a 7-day pre-harvest interval (PHI). Wear gloves and eye protection when preparing bactericide solution."
    },
    "Cercospora_Leaf_Spot": {
        "disease_name": "Cercospora Leaf Spot (Frogeye Spot)",
        "causal_organism": "Cercospora capsici",
        "symptoms": "Circular to oblong lesions with gray or off-white centers and pronounced dark brown margins, resembling a frogeye.",
        "treatment_type": "Fungicide Spray",
        "recommended_chemicals": "Foliar spray of Mancozeb 75% WP @ 2.5 g/L OR Chlorothalonil 75% WP @ 2.0 g/L. For advanced symptoms, Azoxystrobin 23% SC @ 1.0 ml/L.",
        "management_guidance": "Enhance row spacing to improve plant canopy air circulation. Avoid excessive nitrogen fertilization. Destroy crop debris after harvest.",
        "safety_warning": "Do not apply during high midday heat or strong winds. Ensure uniform coverage on both upper and lower leaf surfaces."
    },
    "Curl_Virus": {
        "disease_name": "Chili Leaf Curl Virus (ChiLCV)",
        "causal_organism": "Begomovirus transmitted primarily by Whitefly (Bemisia tabaci)",
        "symptoms": "Severe upward or downward curling of leaves, puckering, vein thickening, stunted plant growth, and reduced flower/fruit set.",
        "treatment_type": "Vector Management (Viral infections cannot be cured with fungicides or fertilizers)",
        "recommended_chemicals": "Vector Control: Neem Oil 3000 ppm @ 3-5 ml/L or 5% NSKE. In severe whitefly infestation: Diafenthiuron 50% WP @ 1.2 g/L OR Imidacloprid 17.8% SL @ 0.3 ml/L.",
        "management_guidance": "CRITICAL: Uproot and burn or bury severely infected plants immediately (roguing) to prevent spread. Install yellow sticky traps (15-20 traps/acre). Keep borders clean of weed hosts.",
        "safety_warning": "Do NOT apply fungicides or fertilizers expecting to cure viral infections; control the whitefly vector strictly according to label dosage."
    },
    "Healthy_Leaf": {
        "disease_name": "Healthy Chili Leaf",
        "causal_organism": "None",
        "symptoms": "Uniform green pigmentation, smooth leaf texture, normal venation, and vigorous leaf expansion.",
        "treatment_type": "None Required",
        "recommended_chemicals": "None. Do not apply chemical pesticides to healthy crops.",
        "management_guidance": "Healthy leaf detected. Continue regular monitoring, balanced NPK fertigation, and maintain standard soil moisture management.",
        "safety_warning": "No action needed. Periodic surveillance recommended."
    }
}


class LeafDiseaseClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LeafDiseaseClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: Path = config.TFLITE_MODEL_PATH):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = config.CLASS_NAMES
        self.load_model()
        self._initialized = True

    def load_model(self):
        """Loads and allocates tensors for the optimized TFLite model."""
        if not self.model_path.exists():
            logger.error(f"TFLite model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file missing: {self.model_path}")

        logger.info(f"Loading TFLite model from {self.model_path}...")
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class names from json if available
        if config.CLASS_NAMES_PATH.exists():
            with open(config.CLASS_NAMES_PATH, "r") as f:
                self.class_names = json.load(f)

        logger.info(f"Model loaded successfully. Input shape: {self.input_details[0]['shape']}, Classes: {self.class_names}")

    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """Validates that the file exists, is non-empty, and is a readable image."""
        if not image_path.exists():
            return False, f"File does not exist: {image_path}"
        if image_path.stat().st_size == 0:
            return False, f"File is empty (0 bytes): {image_path}"
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True, "Valid image"
        except Exception as e:
            return False, f"Corrupted or invalid image: {str(e)}"

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocesses image to (1, 160, 160, 3) float32 RGB array."""
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(config.IMAGE_SIZE)
            img_array = np.asarray(resized_img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array

    def predict(self, image_path: Path, confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Runs TFLite edge inference on a leaf image and returns comprehensive diagnostics."""
        threshold = confidence_threshold if confidence_threshold is not None else config.DEFAULT_CONFIDENCE_THRESHOLD

        valid, err_msg = self.validate_image(image_path)
        if not valid:
            raise ValueError(f"Image validation failed: {err_msg}")

        input_data = self.preprocess_image(image_path)

        # Set input tensor and invoke interpreter
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Extract output probabilities
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        predicted_idx = int(np.argmax(output_data))
        raw_confidence = float(output_data[predicted_idx]) * 100.0
        predicted_class = self.class_names[predicted_idx]

        is_low_confidence = raw_confidence < threshold

        # Retrieve agricultural recommendations
        disease_info = DISEASE_KNOWLEDGE_BASE.get(predicted_class, {})
        recommendation_text = disease_info.get("management_guidance", "Regular monitoring recommended.")
        if is_low_confidence:
            recommendation_text = f"Low confidence ({raw_confidence:.1f}% < {threshold}%). Re-position camera and re-capture under clearer lighting before taking treatment action."

        detection_record = {
            "detection_id": f"DET-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now().isoformat(),
            "image_filename": image_path.name,
            "image_path": str(image_path.resolve()),
            "predicted_disease": predicted_class if not is_low_confidence else f"{predicted_class} (Low Confidence)",
            "disease_code": predicted_class,
            "confidence": round(raw_confidence, 2),
            "confidence_threshold": threshold,
            "is_low_confidence": is_low_confidence,
            "all_class_probabilities": {
                name: round(float(output_data[i]) * 100.0, 2)
                for i, name in enumerate(self.class_names)
            },
            "disease_details": disease_info,
            "recommendation": recommendation_text,
            "sprayer_recommended": (predicted_class in ("Bacterial_Spot", "Cercospora_Leaf_Spot") and not is_low_confidence),
            "requires_user_confirmation": True
        }

        return detection_record
