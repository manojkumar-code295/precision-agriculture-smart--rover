"""
Precision Agriculture Smart Rover - Cloud Backend Entry Point
Root WSGI wrapper forwarding to Website.backend.app for seamless Render deployment.
"""

import os
import sys

# Ensure repository root and Website/backend are on Python sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "Website", "backend")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Import the configured Flask application instance
from Website.backend.app import app

application = app
__all__ = ["app", "application"]

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
