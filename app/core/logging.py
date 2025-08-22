"""Logging setup for production environment."""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.getenv("LOG_DIR", "/app/logs")
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "app.log")

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(process)d %(threadName)s %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(formatter)
    root.addHandler(fh)

__all__ = ["setup_logging"]
