"""
Package fallback for Render placeholder 'your_application'.
Forwards to the actual Flask app instance in Website.backend.app.
"""
from Website.backend.app import app

application = app
__all__ = ["app", "application"]
