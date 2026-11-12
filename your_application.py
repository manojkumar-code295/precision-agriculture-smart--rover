"""
Fallback alias module for Render default placeholder 'your_application:app'.
Resolves 'ModuleNotFoundError: No module named your_application'
by forwarding directly to the actual Flask application instance in Website.backend.app.
"""

from Website.backend.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
