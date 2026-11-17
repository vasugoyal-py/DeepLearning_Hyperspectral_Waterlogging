"""
Production entry point for the DeepWaterLog dashboard.

The dashboard only reads saved run records and images, so it needs Flask,
NumPy and Pillow — not PyTorch. That keeps the deployed image small.

Local:
    python wsgi.py

Container / PaaS (port supplied by the platform):
    gunicorn wsgi:app --bind 0.0.0.0:$PORT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ui.app import app  # noqa: E402,F401  (exported for WSGI servers)

# Some hosts (Hugging Face Spaces, Render) inject PORT; default to 7860.
PORT = int(os.environ.get("PORT", os.environ.get("APP_PORT", "7860")))
HOST = os.environ.get("HOST", "0.0.0.0")


def main() -> None:
    try:
        from waitress import serve
    except ImportError:
        # Development fallback if waitress is not installed.
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
        return

    print(f"DeepWaterLog dashboard on http://{HOST}:{PORT}", flush=True)
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
