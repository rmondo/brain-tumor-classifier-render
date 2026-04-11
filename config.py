"""Minimal runtime config used by notebook Flask orchestration cells."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
BASE_DIR = _HERE if (_HERE / "app.py").exists() else _HERE.parent
LOG_DIR = BASE_DIR / "logs"

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5001


def make_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
