import contextlib
import logging
import os
import time
import warnings
from logging.handlers import RotatingFileHandler
from typing import Optional

import numpy as np


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
    # Ensure root logger level and stdout handler are correctly set
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root_logger.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root_logger.addHandler(sh)

    # Noise suppression for verbose CUDA/Numba logs
    if os.getenv("PM_SUPPRESS_NOISY_LOGS", "1") == "1":
        # Suppress specific warning categories/messages (stderr warnings)
        try:
            try:
                from numba.core.errors import NumbaPerformanceWarning  # type: ignore
            except Exception:
                from numba.errors import NumbaPerformanceWarning  # type: ignore
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        except Exception:
            pass
        # Common noisy warnings
        warnings.filterwarnings(
            "ignore",
            message=r"Grid size .*will likely result in GPU under-utilization",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"All-NaN slice encountered",
            module=r"numpy\.lib\.nanfunctions",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"invalid value encountered in cast",
        )

        class MessageExcludeFilter(logging.Filter):
            def __init__(self, substrings):
                super().__init__()
                self.substrings = tuple(substrings)

            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    msg = record.getMessage()
                except Exception:
                    return True
                return not any(s in msg for s in self.substrings)

        exclude = MessageExcludeFilter(
            [
                "cuMemFree_v2",
                "add pending dealloc",
            ]
        )

        # Lower verbosity of known noisy third-party loggers and apply filter there only
        for name in (
            "numba",
            "numba.cuda",
            "numba.cuda.cudadrv",
            "numba.cuda.cudadrv.driver",
            "numba.cuda.cudadrv.memory",
            "llvmlite",
        ):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            lg.addFilter(exclude)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run.log")
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
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


def log_ndarray_stats(
    name: str,
    arr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    level: int = logging.DEBUG,
) -> None:
    try:
        if mask is not None:
            vals = arr[mask]
        else:
            vals = arr
        vals = vals.astype(np.float32)
        finite = np.isfinite(vals)
        if finite.any():
            v = vals[finite]
            logging.log(
                level,
                f"[STAT] {name}: shape={arr.shape}, finite={v.size}, min={np.min(v):.6f}, max={np.max(v):.6f}, mean={np.mean(v):.6f}",
            )
        else:
            logging.log(level, f"[STAT] {name}: shape={arr.shape}, finite=0")
    except Exception as e:
        logging.debug(f"[STAT] {name}: failed to compute stats: {e}")
