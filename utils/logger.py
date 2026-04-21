import logging
import sys
from pathlib import Path


def build_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def build_tensorboard(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return None
