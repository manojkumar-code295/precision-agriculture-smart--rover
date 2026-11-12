"""
WSGI fallback for Render placeholder 'your_application.wsgi:application'.
Forwards to the actual Flask app instance in app.
"""
from app import app

application = app
__all__ = ["app", "application"]
