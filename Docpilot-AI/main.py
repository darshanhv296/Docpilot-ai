"""DocPilot AI standalone entry point.

Run with ``python main.py`` from any working directory.
"""

from app.api.handlers import *
from app.api.handlers import main as _run


if __name__ == "__main__":
    _run()
