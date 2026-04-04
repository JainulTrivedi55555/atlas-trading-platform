"""
ATLAS Logging Utility
Provides a consistent logger configuration for all ATLAS modules.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR      = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Returns a configured logger.
    Logs to both console and logs/atlas.log (rotating, max 10MB, 5 backups).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured
    logger.setLevel(level)
    fmt = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # Rotating file handler (max 10MB per file, keep 5 files)
    fh = RotatingFileHandler(
        LOG_DIR / 'atlas.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger