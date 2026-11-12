"""
Precision Agriculture Smart Rover - Chili Leaf Object Detection & Tracking Module
Performs real-time plant leaf segmentation, vegetative chromaticity filtering (ExG + HSV),
geometric contour morphology validation, blur/lighting checks, frame-to-frame tracking,
and green bounding-box annotation.
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
    label: str                      # "CHILI LEAF", "Unknown", etc.
    disease: Optional[str] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    is_stable: bool = False
    tracking_frames: int = 1


class ChiliLeafDetector:
    """Multi-stage vegetative segmentation and geometric morphology leaf detector."""

    def __init__(
        self,
        min_area_ratio: float = 0.03,    # Minimum 3% of frame area
        max_area_ratio: float = 0.85,    # Maximum 85% of frame area
        min_solidity: float = 0.50,      # Leaf blade contour compactness
        min_aspect: float = 0.35,        # Width to height lower bound
        max_aspect: float = 2.80,        # Width to height upper bound
        blur_threshold: float = 55.0,    # Laplacian variance cutoff
        min_brightness: float = 30.0,    # Minimum mean gray level
        max_brightness: float = 245.0    # Maximum mean gray level
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
        Combines Excess Green index (ExG = 2G - R - B) and HSV chlorophyll band.
        Rejects soil (ExG < 0), skin/hands (R > G), and neutral artificial backgrounds.
        """
        b, g, r = cv2.split(frame.astype(np.float32))

        # 1. Excess Green Index
        exg = (2.0 * g) - r - b
        exg_mask = (exg > 15.0) & (g > 35.0)

        # 2. HSV Chlorophyll Band
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_green, upper_green) > 0

        # 3. ExG and HSV combination
        combined = (exg_mask | hsv_mask).astype(np.uint8) * 255

        # 4. Morphological closure and opening to clean noise and join leaf regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        return cleaned

    def detect_leaf_candidates(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
        """Finds all candidate chili leaf bounding boxes with geometric validation."""
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
            if area < (frame_area * self.min_area_ratio) or area > (frame_area * self.max_area_ratio):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area) / hull_area
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0:
                continue

            aspect = float(bw) / float(bh)

            # Validate typical lanceolate / ovate chili leaf morphology
            if (self.min_aspect <= aspect <= self.max_aspect) and (solidity >= self.min_solidity):
                cx = x + bw // 2
                cy = y + bh // 2
                candidates.append({
                    "box": (x, y, bw, bh),
                    "centroid": (cx, cy),
                    "area": area,
                    "solidity": solidity,
                    "aspect_ratio": aspect,
                    "contour": cnt
                })

        # Sort candidates by area descending (largest leaf first)
        candidates.sort(key=lambda c: c["area"], reverse=True)
        return candidates, "OK"


class LeafTracker:
    """Tracks detected leaf bounding boxes across frames to verify stability before auto-capture."""

    def __init__(self, stability_frames: int = 3, max_displacement_px: float = 28.0, cooldown_sec: float = 5.0):
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
                    label="CHILI LEAF" if is_stable else "DETECTING LEAF...",
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
                label="DETECTING LEAF...",
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
    Renders high-contrast detection bounding boxes and HUD diagnostics on video frame.
    - Green: Confirmed chili leaf
    - Yellow: Leaf detected, settling/verifying
    - Red: Error or quality alert
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Draw Bounding Boxes
    for leaf in leaves:
        x, y, bw, bh = leaf.box

        if leaf.status == "CONFIRMED_STABLE":
            box_color = (0, 230, 0)     # Neon Green
            label_text = f"CHILI LEAF: {leaf.disease or 'HEALTHY/DISEASE CHECK'}"
            if leaf.confidence > 0:
                label_text += f" ({leaf.confidence:.1f}%)"
        elif leaf.status == "SETTLING":
            box_color = (0, 215, 255)   # Amber Yellow
            label_text = f"DETECTING LEAF... ({leaf.tracking_frames}/3)"
        else:
            box_color = (0, 0, 230)     # Red Alert
            label_text = leaf.label

        # Draw main rectangle
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2)

        # Draw stylish corner accents
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
        cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner_len), box_color, thickness)

        # Label Banner
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y - 22)
        bg_y2 = y
        cv2.rectangle(canvas, (x, bg_y1), (x + tw + 10, bg_y2), box_color, -1)
        text_color = (0, 0, 0) if leaf.status != "ALERT" else (255, 255, 255)
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
    cv2.rectangle(canvas, (0, 0), (w, hud_bg_height), (20, 25, 30), -1)

    hud_text = f"AGRIROVER AI | FPS: {fps:.1f} | Leaves: {len(leaves)} | Auto: {'ON' if auto_active else 'OFF'}"
    cv2.putText(canvas, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 230), 1, cv2.LINE_AA)

    # Quality warnings if any
    if quality_status != "OK" and quality_status != "GOOD_QUALITY":
        warn_text = f"WARNING: {quality_status.replace('_', ' ')}"
        cv2.putText(canvas, warn_text, (w - 240, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)

    return canvas
