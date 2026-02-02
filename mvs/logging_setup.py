"""
ログ初期化モジュール。
"""

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
    共通ログ初期化。標準出力と任意でローテーションファイル（run.log）に出力する。
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

    if os.getenv("PM_SUPPRESS_NOISY_LOGS", "1") == "1":
        try:
            try:
                from numba.core.errors import NumbaPerformanceWarning  # type: ignore
            except Exception:
                from numba.errors import NumbaPerformanceWarning  # type: ignore
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        except Exception:
            pass
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
            """ログメッセージに指定文字列が含まれる場合はそのレコードを破棄するフィルタ。"""

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
    """
    root logger のログレベルを変更する。
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logging.getLogger().setLevel(level_map.get(level.upper(), logging.INFO))


@contextlib.contextmanager
def time_block(name: str, level: int = logging.INFO):
    """
    ブロックの実行時間を計測し、指定レベルで "[TIME] name: 経過秒数" をログ出力するコンテキストマネージャ。
    """
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
    """
    配列の shape と有限値の個数・min/max/mean を指定レベルでログ出力する。
    mask を渡した場合はそのマスクで絞った値で統計を計算する。
    """
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
