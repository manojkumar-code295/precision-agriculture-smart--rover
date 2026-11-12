"""
Precision Agriculture Smart Rover - Local SQLite Database Module
Manages local detections, sensor telemetry, rover movement logs, and synchronization queue.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import config

logger = logging.getLogger("RoverDatabase")


class RoverDatabase:
    def __init__(self, db_path: Path = config.DATABASE_PATH):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables with proper indices and schemas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Detections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    predicted_disease TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_low_confidence INTEGER NOT NULL DEFAULT 0,
                    recommendation TEXT NOT NULL,
                    sprayer_status TEXT DEFAULT 'OFF',
                    sync_status INTEGER NOT NULL DEFAULT 0,
                    synced_at TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Sensor Telemetry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ultrasonic_distance REAL,
                    obstacle_detected INTEGER NOT NULL DEFAULT 0,
                    soil_moisture REAL,
                    temperature REAL,
                    humidity REAL,
                    sync_status INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # Rover Movement Event table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rover_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    command TEXT NOT NULL,
                    speed INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    obstacle_stopped INTEGER NOT NULL DEFAULT 0,
                    details TEXT
                )
            """)

            # Sprayer Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sprayer_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    user_confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # Settings KV Store
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indices for quick lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_sync ON detections(sync_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_logs_sync ON sensor_logs(sync_status)")
            conn.commit()

    # -------------------------------------------------------------
    # Detections
    # -------------------------------------------------------------
    def insert_detection(self, data: Dict[str, Any]) -> int:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (
                    detection_id, timestamp, image_filename, image_path,
                    predicted_disease, confidence, is_low_confidence,
                    recommendation, sprayer_status, sync_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                data["detection_id"],
                data.get("timestamp", now),
                data["image_filename"],
                data["image_path"],
                data["predicted_disease"],
                float(data["confidence"]),
                1 if data.get("is_low_confidence", False) else 0,
                data.get("recommendation", ""),
                data.get("sprayer_status", "OFF"),
                now
            ))
            conn.commit()
            return cursor.lastrowid

    def get_latest_detection(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_detections(self, limit: int = 50, offset: int = 0,
                       disease: Optional[str] = None,
                       min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM detections WHERE 1=1"
            params = []
            if disease:
                query += " AND predicted_disease = ?"
                params.append(disease)
            if min_confidence is not None:
                query += " AND confidence >= ?"
                params.append(min_confidence)
            query += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_detection_statistics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM detections")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS healthy FROM detections WHERE predicted_disease = 'Healthy_Leaf'")
            healthy = cursor.fetchone()["healthy"]

            cursor.execute("SELECT predicted_disease, COUNT(*) as count, AVG(confidence) as avg_conf FROM detections GROUP BY predicted_disease")
            distribution = {row["predicted_disease"]: {"count": row["count"], "avg_confidence": round(row["avg_conf"], 2)} for row in cursor.fetchall()}

            return {
                "total_detections": total,
                "healthy_count": healthy,
                "diseased_count": total - healthy,
                "disease_distribution": distribution
            }

    # -------------------------------------------------------------
    # Sensor Logs
    # -------------------------------------------------------------
    def insert_sensor_log(self, data: Dict[str, Any]) -> int:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sensor_logs (
                    timestamp, ultrasonic_distance, obstacle_detected,
                    soil_moisture, temperature, humidity, sync_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                data.get("timestamp", now),
                data.get("ultrasonic_distance"),
                1 if data.get("obstacle_detected", False) else 0,
                data.get("soil_moisture"),
                data.get("temperature"),
                data.get("humidity"),
                now
            ))
            conn.commit()
            return cursor.lastrowid

    def get_latest_sensor_log(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sensor_logs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------
    # Rover and Sprayer Events
    # -------------------------------------------------------------
    def insert_rover_event(self, command: str, speed: int, mode: str,
                           obstacle_stopped: bool = False, details: str = "") -> int:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rover_events (timestamp, command, speed, mode, obstacle_stopped, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now, command, speed, mode, 1 if obstacle_stopped else 0, details))
            conn.commit()
            return cursor.lastrowid

    def insert_sprayer_event(self, action: str, duration_sec: float,
                             trigger_reason: str, user_confirmed: bool) -> int:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sprayer_events (timestamp, action, duration_sec, trigger_reason, user_confirmed, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now, action, duration_sec, trigger_reason, 1 if user_confirmed else 0, now))
            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------------
    # Synchronization
    # -------------------------------------------------------------
    def get_pending_sync_records(self, table: str = "detections", limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} WHERE sync_status = 0 ORDER BY id ASC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def mark_records_synced(self, table: str, ids: List[int]):
        if not ids:
            return
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in ids)
            if table == "detections":
                cursor.execute(f"UPDATE detections SET sync_status = 1, synced_at = ? WHERE id IN ({placeholders})", [now] + ids)
            else:
                cursor.execute(f"UPDATE {table} SET sync_status = 1 WHERE id IN ({placeholders})", ids)
            conn.commit()

    # -------------------------------------------------------------
    # Settings Management
    # -------------------------------------------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key, value, now))
            conn.commit()

    # -------------------------------------------------------------
    # Storage Maintenance
    # -------------------------------------------------------------
    def cleanup_old_images(self, max_images: int = config.MAX_LOCAL_IMAGES):
        """Ensures local microSD storage is not overwhelmed by old captured leaf images."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM detections")
            total = cursor.fetchone()["total"]

            if total > max_images:
                excess = total - max_images
                cursor.execute("SELECT id, image_path FROM detections ORDER BY id ASC LIMIT ?", (excess,))
                old_records = cursor.fetchall()

                for row in old_records:
                    file_path = Path(row["image_path"])
                    if file_path.exists() and file_path.is_file():
                        try:
                            file_path.unlink()
                        except Exception as e:
                            logger.warning(f"Could not remove old file {file_path}: {e}")
