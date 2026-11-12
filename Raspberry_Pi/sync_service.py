"""
Precision Agriculture Smart Rover - Cloud Synchronization Service
Uploads pending local detection and sensor records to the Render cloud API.
Continues safe local caching when internet is unavailable.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import config
from database import RoverDatabase

logger = logging.getLogger("SyncService")


class CloudSyncService:
    def __init__(self, db: Optional[RoverDatabase] = None):
        self.db = db or RoverDatabase()
        self.cloud_url = config.CLOUD_API_URL.rstrip("/")
        self.api_key = config.CLOUD_API_KEY
        self.sync_interval = config.SYNC_INTERVAL_SEC
        self.is_online = False
        self.last_sync_time = None
        self.last_error = None
        self.running = True
        self.sync_thread = None

    def start(self):
        """Starts background periodic sync thread."""
        def sync_worker():
            while self.running:
                try:
                    self.perform_sync()
                except Exception as e:
                    logger.error(f"Unexpected sync worker error: {e}")
                time.sleep(self.sync_interval)

        self.sync_thread = threading.Thread(target=sync_worker, daemon=True)
        self.sync_thread.start()
        logger.info(f"Sync service started. Target cloud: {self.cloud_url}")

    def check_cloud_health(self) -> bool:
        """Pings cloud API health check endpoint."""
        try:
            resp = requests.get(f"{self.cloud_url}/api/health", timeout=5.0)
            if resp.status_code == 200:
                self.is_online = True
                self.last_error = None
                return True
        except requests.RequestException as e:
            self.last_error = f"Cloud connection failed: {e}"
        self.is_online = False
        return False

    def perform_sync(self) -> Dict[str, Any]:
        """Uploads pending batches of detections and sensor logs."""
        if not self.check_cloud_health():
            return {
                "success": False,
                "online": False,
                "error": self.last_error,
                "message": "Cloud API unreachable; records remain safely cached locally."
            }

        headers = {
            "Content-Type": "application/json",
            "X-Device-ID": config.DEVICE_ID,
            "X-API-Key": self.api_key
        }

        synced_detections_count = 0
        synced_sensors_count = 0

        # 1. Sync pending detections
        pending_detections = self.db.get_pending_sync_records("detections", limit=config.SYNC_BATCH_SIZE)
        if pending_detections:
            try:
                resp = requests.post(
                    f"{self.cloud_url}/api/sync/detections",
                    json={"device_id": config.DEVICE_ID, "records": pending_detections},
                    headers=headers,
                    timeout=10.0
                )
                if resp.status_code in (200, 201):
                    ids = [rec["id"] for rec in pending_detections]
                    self.db.mark_records_synced("detections", ids)
                    synced_detections_count = len(ids)
                    logger.info(f"Successfully synced {synced_detections_count} detections to cloud.")
            except requests.RequestException as e:
                logger.warning(f"Detection sync request failed: {e}")

        # 2. Sync pending sensor logs
        pending_sensors = self.db.get_pending_sync_records("sensor_logs", limit=config.SYNC_BATCH_SIZE)
        if pending_sensors:
            try:
                resp = requests.post(
                    f"{self.cloud_url}/api/sync/sensors",
                    json={"device_id": config.DEVICE_ID, "records": pending_sensors},
                    headers=headers,
                    timeout=10.0
                )
                if resp.status_code in (200, 201):
                    ids = [rec["id"] for rec in pending_sensors]
                    self.db.mark_records_synced("sensor_logs", ids)
                    synced_sensors_count = len(ids)
                    logger.info(f"Successfully synced {synced_sensors_count} sensor logs to cloud.")
            except requests.RequestException as e:
                logger.warning(f"Sensor sync request failed: {e}")

        self.last_sync_time = datetime.now().isoformat()
        return {
            "success": True,
            "online": True,
            "synced_detections": synced_detections_count,
            "synced_sensors": synced_sensors_count,
            "last_sync": self.last_sync_time
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "online": self.is_online,
            "cloud_url": self.cloud_url,
            "last_sync": self.last_sync_time,
            "last_error": self.last_error,
            "sync_interval_sec": self.sync_interval
        }

    def stop(self):
        self.running = False
