"""
Package fallback for Render placeholder 'your_application'.
Forwards to the actual Flask app instance in app.
"""
from app import app

application = app
__all__ = ["app", "application"]
