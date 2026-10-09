#!/usr/bin/env python3
"""Run the movie product UI (port 5000). Requires platform on PLATFORM_URL (default localhost:5001)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PLATFORM_URL", "http://localhost:5001")
from product.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
