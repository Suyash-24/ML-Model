"""
utils/logger.py  —  Eyecon Logging Utility
"""

import logging
import os
import sys
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_file = os.path.join(LOG_DIR, f"eyecon_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(name)-16s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)


class EyeconLogger:
    def __init__(self, name: str):
        self._log = logging.getLogger(f"Eyecon.{name}")

    def info(self,    msg): self._log.info(msg)
    def debug(self,   msg): self._log.debug(msg)
    def warning(self, msg): self._log.warning(msg)
    def error(self,   msg): self._log.error(msg)
    def critical(self,msg): self._log.critical(msg)
