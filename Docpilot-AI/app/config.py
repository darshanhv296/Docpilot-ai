"""Runtime paths and configuration for source and PyInstaller execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

if os.environ.get("DOCPILOT_DATA_DIR"):
    DATA_ROOT = Path(os.environ["DOCPILOT_DATA_DIR"]).expanduser().resolve()
elif getattr(sys, "frozen", False):
    DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DocPilot AI"
else:
    DATA_ROOT = PROJECT_ROOT / ".data"

FRONTEND_ROOT = PROJECT_ROOT / "frontend"
DEFAULT_HOST = os.environ.get("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
