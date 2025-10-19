import logging, sys
from pathlib import Path

def setup_logger(name: str, logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
            logger.addHandler(fh)
    return logger
