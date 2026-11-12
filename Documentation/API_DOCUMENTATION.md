# REST API Documentation
**Plant Disease Detection AI – Precision Agriculture Smart Rover**

Both the Raspberry Pi local edge server (`http://<pi-ip>:5000`) and the Render Cloud API (`https://<service-name>.onrender.com`) expose standard JSON REST endpoints.

---

## 1. System Health & Status

### `GET /api/health`
Returns system health, CPU/RAM diagnostics, and storage metrics.
```json
{
  "status": "OK",
  "device_id": "agrirover-pi4-01",
  "disk_free_gb": 18.9,
  "disk_total_gb": 28.8,
  "timestamp": "2026-11-12T13:00:00"
}
```

### `GET /api/status`
Returns live state of all Rover subsystems:
```json
{
  "pi_online": true,
  "camera": { "camera_connected": true, "camera_type": "Picamera2 (libcamera)" },
  "rover": { "mode": "MANUAL", "is_moving": false, "motor_status": { "state": "STOPPED", "speed": 0 } },
  "sensors": { "ultrasonic_distance": { "reading": 120.5, "unit": "cm", "status": "Operational" } },
  "sprayer": { "sprayer_active": false, "last_activated": null }
}
```

---

## 2. Camera & AI Disease Classification

### `POST /api/camera/capture?classify=true`
Captures a high-resolution still image from the camera and triggers the edge AI classifier.
**Response:**
```json
{
  "success": true,
  "image_filename": "leaf_capture_20261112_130000.jpg",
  "image_url": "/captures/leaf_capture_20261112_130000.jpg",
  "detection": {
    "detection_id": "DET-20261112130000-a1b2c3",
    "predicted_disease": "Bacterial_Spot",
    "confidence": 99.74,
    "confidence_threshold": 70.0,
    "is_low_confidence": false,
    "recommendation": "Spray Copper Oxychloride 50% WP @ 2.5 g/L mixed with Streptocycline @ 1 g/10 L.",
    "sprayer_recommended": true
  }
}
```

### `POST /api/ai/classify`
Accepts a multipart file upload (`image`) and returns AI diagnosis.

---

## 3. Rover Directional Movement

### `POST /api/rover/move`
Sends directional commands to the H-Bridge dual motor driver.
**Request Body:**
```json
{
  "command": "FORWARD",
  "speed": 75
}
```
*Valid commands: `FORWARD`, `BACKWARD`, `LEFT`, `RIGHT`, `STOP`.*

**Obstacle Rejection Response (Safety Interlock):**
```json
{
  "success": false,
  "state": "BLOCKED",
  "reason": "OBSTACLE_DETECTED",
  "message": "Cannot move forward: Obstacle detected within safety distance."
}
```

### `POST /api/rover/emergency-stop`
Immediate hard cutoff to all GPIO motor outputs.

---

## 4. Sprayer Actuation

### `POST /api/sprayer/activate`
Activates the chemical pump relay with operator confirmation.
**Request Body:**
```json
{
  "duration": 3.0,
  "confirmed": true,
  "disease_context": { "disease": "Bacterial_Spot" }
}
```

### `POST /api/sprayer/stop`
Immediately stops the sprayer pump relay.

---

## 5. Telemetry & Analytics

### `GET /api/sensors`
Returns real-time telemetry from all connected sensors.

### `GET /api/detections`
Retrieves paginated detection records. Query parameters:
- `limit`: int (default: 50)
- `offset`: int (default: 0)
- `disease`: string (optional filter)
- `min_confidence`: float (optional filter)

### `GET /api/reports/export.csv`
Downloads complete field detection history as a CSV file.
