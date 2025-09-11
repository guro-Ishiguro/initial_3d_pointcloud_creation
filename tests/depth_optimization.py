import os
import logging
import importlib


def _detect_cuda_available():
    try:
        from numba import cuda  # noqa: F401
    except Exception:
        return False
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def _import_local_module(mod_name: str):
    """Import module from either package context (tests.mod) or local (mod).

    This allows running both `python -m tests.main` and `python tests/main.py`.
    """
    try:
        return importlib.import_module(f"tests.{mod_name}")
    except Exception:
        return importlib.import_module(mod_name)


_FORCE_CPU = os.environ.get("PM_FORCE_CPU", "0") in ("1", "true", "True")

if not _FORCE_CPU and _detect_cuda_available():
    try:
        DepthOptimization = _import_local_module("depth_optimization_gpu").DepthOptimization  # type: ignore[attr-defined]
        USING_GPU = True
        logging.info("DepthOptimization: Using GPU implementation.")
    except Exception as e:
        logging.warning(f"DepthOptimization GPU import failed ({e}); falling back to CPU.")
        DepthOptimization = _import_local_module("depth_optimization_cpu").DepthOptimization  # type: ignore[attr-defined]
        USING_GPU = False
else:
    DepthOptimization = _import_local_module("depth_optimization_cpu").DepthOptimization  # type: ignore[attr-defined]
    USING_GPU = False


def is_gpu_enabled() -> bool:
    return USING_GPU


__all__ = ["DepthOptimization", "USING_GPU", "is_gpu_enabled"]


