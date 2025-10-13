import os
import logging
import contextlib
import time
import numpy as np
from typing import Optional
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str, level: str = "INFO", to_file: bool = True) -> None:
    """
    共通ログ初期化。標準出力とローテーションファイルロギング（任意）。
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run.log")
        fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3)
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logging.getLogger().addHandler(fh)

def set_log_level(level: str) -> None:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logging.getLogger().setLevel(level_map.get(level.upper(), logging.INFO))

@contextlib.contextmanager
def time_block(name: str, level: int = logging.INFO):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logging.log(level, f"[TIME] {name}: {dt:.4f}s")

def log_ndarray_stats(name: str, arr: np.ndarray, mask: Optional[np.ndarray] = None, level: int = logging.DEBUG) -> None:
    try:
        if mask is not None:
            vals = arr[mask]
        else:
            vals = arr
        vals = vals.astype(np.float32)
        finite = np.isfinite(vals)
        if finite.any():
            v = vals[finite]
            logging.log(level, f"[STAT] {name}: shape={arr.shape}, finite={v.size}, min={np.min(v):.6f}, max={np.max(v):.6f}, mean={np.mean(v):.6f}")
        else:
            logging.log(level, f"[STAT] {name}: shape={arr.shape}, finite=0")
    except Exception as e:
        logging.debug(f"[STAT] {name}: failed to compute stats: {e}")


