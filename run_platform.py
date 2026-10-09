#!/usr/bin/env python3
"""Run the recommendation platform API (port 5001)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PORT", "5001")
from rec_platform.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
