"""Simple logger factory."""
import logging


def get_logger(name: str = __name__, level: int = logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger
