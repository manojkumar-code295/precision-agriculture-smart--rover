"""
Precision Agriculture Smart Rover - Chili Leaf Object Detection & Botanical Verifier Module
Performs multi-spectral plant foliage segmentation (Chlorophyll + Carotenoid + Necrotic lesions),
geometric contour morphology validation, botanical discrimination against non-plant objects
(human hands/skin, walls, curtains, tables, clothes), frame-to-frame tracking, and HUD annotation.
"""

import time
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np

logger = logging.getLogger("LeafDetector")


@dataclass
class DetectedLeaf:
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]       # (cx, cy)
    area: float
    solidity: float
    aspect_ratio: float
    status: str                     # "CONFIRMED_STABLE", "SETTLING", "LOW_CONFIDENCE", "NOT_A_LEAF"
    label: str                      # "Plant / Leaf Detected", "Verifying Leaf...", etc.
    disease: Optional[str] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    is_stable: bool = False
    tracking_frames: int = 1


def comprehensive_leaf_verifier(img_bgr: np.ndarray) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Botanical leaf vs non-plant verifier.
    Distinguishes real chili leaves (healthy green, yellow chlorotic, diseased spotty, curly)
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
        return False, "No Leaf Result / Insufficient Lighting", 0.0, {"reason": f"Scene too dark ({mean_brightness:.1f} < 16)"}
    if mean_brightness > 251.0:
        return False, "No Leaf Result / Overexposed Glare", 0.0, {"reason": f"Overexposed glare ({mean_brightness:.1f} > 251)"}

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
    denom = (2.0 * g) + r + b + 1e-4
    gli = exg / denom

    # Chlorophyll (Healthy and spotted green leaves)
    # Green hue in [22, 95], S >= 20, V >= 25, G > R, ExG > 8.0
    chlorophyll = (g > r) & (exg > 8.0) & (h_channel >= 22) & (h_channel <= 95) & (s_channel >= 20) & (v_channel >= 25) & (g > 32.0)
    chlorophyll_ratio = float(np.sum(chlorophyll)) / total_pixels

    # Carotenoid / Xanthophyll (Yellow chlorotic / virus leaves)
    yellow_hue = (h_channel >= 14) & (h_channel <= 42) & (s_channel >= 25) & (v_channel >= 35)
    carotenoid = yellow_hue & (r > 45.0) & (g > 45.0) & (np.abs(r - g) < 40.0) & (b < 155.0) & (g >= (b * 0.95))
    carotenoid_ratio = float(np.sum(carotenoid)) / total_pixels

    # Necrotic Lesions (Brown/necrotic spot tissue)
    brown_hue = (h_channel >= 5) & (h_channel <= 25) & (s_channel >= 20) & (v_channel >= 20) & (v_channel <= 185)
    brown_spots = brown_hue & (r > b) & ((r - b) > 6.0)
    brown_ratio = float(np.sum(brown_spots)) / total_pixels

    plant_foliage_ratio = chlorophyll_ratio + carotenoid_ratio

    # Rejection: Curtain / Fabric folds (natural leaves have isotropic branching veins with anisotropy <= 2.2)
    if anisotropy > 3.2:
        return False, "No Leaf Result / Curtain Detected", 0.0, {
            "reason": f"Periodic fabric folds detected (anisotropy={anisotropy:.1f})",
            "anisotropy": round(anisotropy, 2)
        }

    # Rejection: Wall or Flat Surface
    if lap_var < 18.0 and edge_density < 0.020:
        return False, "No Leaf Result / Wall Detected", 0.0, {
            "reason": f"Flat featureless surface detected (lap_var={lap_var:.1f})",
            "lap_var": round(lap_var, 1)
        }

    # Rejection: Non-plant background (furniture, clothes, table, carpet)
    if plant_foliage_ratio < 0.08 and (plant_foliage_ratio + brown_ratio) < 0.10:
        return False, "No Leaf Result / Not a Plant", 0.0, {
            "reason": f"No botanical foliage detected (plant_ratio={plant_foliage_ratio*100:.1f}%)",
            "plant_ratio": round(plant_foliage_ratio, 3)
        }

    # Passed Botanical Verification!
    # Confidence scaled based on genuine pigment density
    leaf_confidence = min(99.0, max(72.0, 70.0 + (chlorophyll_ratio * 32.0) + (carotenoid_ratio * 25.0)))
    return True, "Plant / Leaf Detected", round(leaf_confidence, 1), {
        "plant_ratio": round(plant_foliage_ratio, 3),
        "chlorophyll_ratio": round(chlorophyll_ratio, 3),
        "carotenoid_ratio": round(carotenoid_ratio, 3),
        "brown_ratio": round(brown_ratio, 3),
        "skin_ratio": round(skin_ratio, 3),
        "lap_var": round(lap_var, 1)
    }


class ChiliLeafDetector:
    """Multi-stage botanical vegetative segmentation and geometric morphology leaf detector."""

    def __init__(
        self,
        min_area_ratio: float = 0.02,    # Minimum 2% of frame area
        max_area_ratio: float = 0.94,    # Support close-up leaves up to 94% of frame area
        min_solidity: float = 0.38,      # Leaf blade contour compactness
        min_aspect: float = 0.22,        # Width to height lower bound
        max_aspect: float = 4.20,        # Width to height upper bound
        blur_threshold: float = 35.0,    # Laplacian variance cutoff
        min_brightness: float = 16.0,    # Minimum mean gray level
        max_brightness: float = 252.0    # Maximum mean gray level
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_solidity = min_solidity
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def check_frame_quality(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Checks overall illumination before running detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(cv2.mean(gray)[0])

        if brightness < self.min_brightness:
            return False, "INSUFFICIENT_LIGHTING"
        if brightness > self.max_brightness:
            return False, "OVEREXPOSED"

        return True, "GOOD_QUALITY"

    def segment_vegetation(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts multi-spectral plant foliage mask:
        - Chlorophyll band (ExG > 4 & HSV green)
        - Carotenoid yellow chlorosis band (HSV yellow & high R/G)
        - Rejects skin (R > G > B & YCbCr) and plain neutral backgrounds.
        """
        b, g, r = cv2.split(frame.astype(np.float32))

        # 1. Excess Green Index
        exg = (2.0 * g) - r - b
        chlorophyll_mask = (exg > 4.0) & (g > 30.0)

        # 2. HSV Green & Yellow Bands
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        green_band = (h >= 18) & (h <= 98) & (s >= 20) & (v >= 25)
        yellow_band = (h >= 12) & (h <= 45) & (s >= 25) & (v >= 35) & (r > 45.0) & (g > 40.0)

        # 3. Suppress Skin and Hands
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (r > g) & (g > b) & ((r - g) > 8.0)

        # Combined foliage mask minus skin
        foliage = ((chlorophyll_mask & green_band) | yellow_band) & (~skin_mask)
        combined = foliage.astype(np.uint8) * 255

        # 4. Morphological closure and opening to clean noise and join leaf regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        return cleaned

    def detect_leaf_candidates(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
        """Finds all candidate leaf bounding boxes validated by botanical discrimination."""
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        valid_qual, qual_msg = self.check_frame_quality(frame)
        if not valid_qual:
            return [], qual_msg

        mask = self.segment_vegetation(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area bounds: must be between min_area_ratio and max_area_ratio
            if area < (frame_area * self.min_area_ratio) or area > (frame_area * self.max_area_ratio):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0:
                continue

            # Strict guard against full-frame border filling only if both width and height touch borders
            if bw >= int(w * 0.99) and bh >= int(h * 0.99):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area) / hull_area
            aspect = float(bw) / float(bh)

            # Validate typical lanceolate / ovate chili leaf morphology
            if (self.min_aspect <= aspect <= self.max_aspect) and (solidity >= self.min_solidity):
                # Crop ROI and run botanical verifier on candidate
                roi = frame[y:y+bh, x:x+bw]
                is_leaf, leaf_label, leaf_conf, det_meta = comprehensive_leaf_verifier(roi)

                if is_leaf:
                    cx = x + bw // 2
                    cy = y + bh // 2
                    candidates.append({
                        "box": (x, y, bw, bh),
                        "centroid": (cx, cy),
                        "area": area,
                        "solidity": solidity,
                        "aspect_ratio": aspect,
                        "confidence": leaf_conf,
                        "contour": cnt
                    })

        # Fallback check: if no individual contour candidate was accepted,
        # verify if the center or full frame is a genuine close-up leaf view
        if not candidates:
            is_full_leaf, full_label, full_conf, full_meta = comprehensive_leaf_verifier(frame)
            if is_full_leaf:
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.05)
                bw = w - 2 * pad_x
                bh = h - 2 * pad_y
                candidates.append({
                    "box": (pad_x, pad_y, bw, bh),
                    "centroid": (w // 2, h // 2),
                    "area": float(bw * bh),
                    "solidity": 0.85,
                    "aspect_ratio": float(bw) / float(bh),
                    "confidence": full_conf,
                    "contour": None
                })
                return candidates, "OK"
            else:
                return [], full_label

        # Sort candidates by area descending (largest leaf first)
        candidates.sort(key=lambda c: c["area"], reverse=True)
        return candidates, "OK"


class LeafTracker:
    """Tracks detected leaf bounding boxes across frames to verify stability before auto-capture."""

    def __init__(self, stability_frames: int = 3, max_displacement_px: float = 35.0, cooldown_sec: float = 5.0):
        self.stability_frames = stability_frames
        self.max_displacement_px = max_displacement_px
        self.cooldown_sec = cooldown_sec

        self.tracked_leaves: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1
        self.last_capture_time: Dict[int, float] = {}

    def update(self, candidates: List[Dict[str, Any]]) -> List[DetectedLeaf]:
        """Matches candidates against existing tracks and determines stability."""
        now = time.time()
        updated_tracks = {}
        detected_results = []

        unmatched_candidates = list(candidates)

        # Match existing tracks with nearest candidate
        for track_id, track in self.tracked_leaves.items():
            best_idx = -1
            best_dist = float("inf")

            for idx, cand in enumerate(unmatched_candidates):
                cx, cy = cand["centroid"]
                tcx, tcy = track["centroid"]
                dist = math.hypot(cx - tcx, cy - tcy)
                if dist < best_dist and dist <= self.max_displacement_px:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                cand = unmatched_candidates.pop(best_idx)
                frames = track["frames"] + 1
                is_stable = frames >= self.stability_frames
                status = "CONFIRMED_STABLE" if is_stable else "SETTLING"

                updated_tracks[track_id] = {
                    "box": cand["box"],
                    "centroid": cand["centroid"],
                    "area": cand["area"],
                    "solidity": cand["solidity"],
                    "aspect_ratio": cand["aspect_ratio"],
                    "confidence": cand.get("confidence", 85.0),
                    "frames": frames,
                    "is_stable": is_stable,
                    "last_seen": now
                }

                leaf = DetectedLeaf(
                    box=cand["box"],
                    centroid=cand["centroid"],
                    area=cand["area"],
                    solidity=cand["solidity"],
                    aspect_ratio=cand["aspect_ratio"],
                    status=status,
                    label="Plant / Leaf Detected" if is_stable else "Verifying Leaf...",
                    confidence=cand.get("confidence", 85.0),
                    is_stable=is_stable,
                    tracking_frames=frames
                )
                detected_results.append(leaf)

        # Register remaining candidates as new tracks
        for cand in unmatched_candidates:
            track_id = self.next_track_id
            self.next_track_id += 1

            updated_tracks[track_id] = {
                "box": cand["box"],
                "centroid": cand["centroid"],
                "area": cand["area"],
                "solidity": cand["solidity"],
                "aspect_ratio": cand["aspect_ratio"],
                "confidence": cand.get("confidence", 80.0),
                "frames": 1,
                "is_stable": False,
                "last_seen": now
            }

            leaf = DetectedLeaf(
                box=cand["box"],
                centroid=cand["centroid"],
                area=cand["area"],
                solidity=cand["solidity"],
                aspect_ratio=cand["aspect_ratio"],
                status="SETTLING",
                label="Verifying Leaf...",
                confidence=cand.get("confidence", 80.0),
                is_stable=False,
                tracking_frames=1
            )
            detected_results.append(leaf)

        self.tracked_leaves = updated_tracks
        return detected_results

    def can_capture(self, leaf: DetectedLeaf) -> bool:
        """Determines if a stable leaf can be auto-captured without violating cooldown."""
        now = time.time()
        for track_id, track in self.tracked_leaves.items():
            if track["centroid"] == leaf.centroid and leaf.is_stable:
                last_time = self.last_capture_time.get(track_id, 0.0)
                if (now - last_time) >= self.cooldown_sec:
                    self.last_capture_time[track_id] = now
                    return True
        return False


def annotate_frame(
    frame: np.ndarray,
    leaves: List[DetectedLeaf],
    quality_status: str = "OK",
    fps: float = 0.0,
    auto_active: bool = True
) -> np.ndarray:
    """
    Renders high-contrast botanical leaf detection bounding boxes and HUD diagnostics.
    - Green: Confirmed Plant / Leaf Detected
    - Yellow: Leaf detected, settling/verifying
    - Red/Alert: Quality warnings or Non-Plant status
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Draw Bounding Boxes
    for leaf in leaves:
        x, y, bw, bh = leaf.box

        # Distinct disease colors for visual clarity
        disease_name = leaf.disease or ""
        if "Healthy" in disease_name:
            box_color = (0, 230, 0)      # Bright Green
        elif "Bacterial" in disease_name:
            box_color = (0, 140, 255)    # Orange
        elif "Cercospora" in disease_name:
            box_color = (180, 50, 180)   # Purple
        elif "Curl" in disease_name:
            box_color = (30, 30, 225)    # Crimson Red
        elif leaf.status == "CONFIRMED_STABLE":
            box_color = (0, 230, 0)      # Neon Green
        elif leaf.status == "SETTLING":
            box_color = (0, 215, 255)    # Amber Yellow
        else:
            box_color = (0, 0, 230)      # Red Alert

        if leaf.disease:
            label_text = f"LEAF: {leaf.disease}"
        elif leaf.status == "CONFIRMED_STABLE":
            label_text = f"CONFIRMED LEAF ({leaf.confidence:.1f}%)"
        elif leaf.status == "SETTLING":
            label_text = f"VERIFYING LEAF... ({leaf.tracking_frames}/3)"
        else:
            label_text = leaf.label

        # Draw main rectangle
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2)

        # Draw corner accents
        corner_len = min(15, bw // 4, bh // 4)
        thickness = 3
        # Top-left
        cv2.line(canvas, (x, y), (x + corner_len, y), box_color, thickness)
        cv2.line(canvas, (x, y), (x, y + corner_len), box_color, thickness)
        # Top-right
        cv2.line(canvas, (x + bw, y), (x + bw - corner_len, y), box_color, thickness)
        cv2.line(canvas, (x + bw, y), (x + bw, y + corner_len), box_color, thickness)
        # Bottom-left
        cv2.line(canvas, (x, y + bh), (x + corner_len, y + bh), box_color, thickness)
        cv2.line(canvas, (x, y + bh), (x, y + bh - corner_len), box_color, thickness)
        # Bottom-right
        cv2.line(canvas, (x + bw, y + bh), (x + bw - corner_len, y + bh), box_color, thickness)
        cv2.line(canvas, (x + bw, y + bh), (x + bw, y + corner_len), box_color, thickness)

        # Label Banner
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y - 22)
        bg_y2 = y
        cv2.rectangle(canvas, (x, bg_y1), (x + tw + 10, bg_y2), box_color, -1)
        text_color = (0, 0, 0)
        cv2.putText(
            canvas,
            label_text,
            (x + 5, bg_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA
        )

    # Top HUD Bar
    hud_bg_height = 32
    cv2.rectangle(canvas, (0, 0), (w, hud_bg_height), (10, 25, 18), -1)

    has_confirmed_leaf = any(l.status == "CONFIRMED_STABLE" for l in leaves)
    first_disease = next((l.disease for l in leaves if l.disease), None)

    if first_disease:
        status_str = f"LEAF: {first_disease}"
        status_color = (0, 230, 0) if "Healthy" in first_disease else (0, 165, 255)
    elif has_confirmed_leaf:
        status_str = "PLANT / LEAF DETECTED"
        status_color = (0, 230, 0)
    elif leaves:
        status_str = "VERIFYING LEAF..."
        status_color = (0, 215, 255)
    elif "Hand" in quality_status:
        status_str = "NO LEAF (HAND SUPPRESSED)"
        status_color = (50, 50, 240)
    elif "Wall" in quality_status:
        status_str = "NO LEAF (WALL SUPPRESSED)"
        status_color = (150, 150, 200)
    elif "Curtain" in quality_status or "Clothes" in quality_status or "Not a Plant" in quality_status:
        status_str = "NO LEAF (NON-PLANT SUPPRESSED)"
        status_color = (150, 150, 200)
    else:
        status_str = "NO LEAF IN VIEW"
        status_color = (160, 175, 165)

    hud_text = f"AGRIROVER AI | FPS: {fps:.1f} | 640x480 | Auto: {'ON' if auto_active else 'OFF'}"
    cv2.putText(canvas, hud_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)

    # Right side HUD status
    cv2.putText(canvas, f"STATUS: {status_str}", (w - 240, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    # Quality warnings if any
    if quality_status not in ("OK", "GOOD_QUALITY"):
        warn_text = f"WARNING: {quality_status.replace('_', ' ')}"
        cv2.putText(canvas, warn_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1, cv2.LINE_AA)

    return canvas
