import importlib
import logging
import os


def _detect_cupy_available():
    try:
        import cupy as cp  # noqa: F401

        return True
    except Exception:
        return False


def _detect_cuda_available():
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _import_local_module(mod_name: str):
    """Import module from either package context (mvs.mod) or local (mod).

    This allows running both `python -m mvs.main` and `python mvs/main.py`.
    """
    try:
        return importlib.import_module(f"mvs.{mod_name}")
    except Exception:
        return importlib.import_module(mod_name)


backend_pref = os.getenv("MVS_BACKEND", "auto").lower()

DepthOptimization = None  # type: ignore
USING_GPU = False

if backend_pref in ("cupy", "auto") and _detect_cupy_available():
    try:
        DepthOptimization = _import_local_module("depth_optimization_cupy").DepthOptimization  # type: ignore[attr-defined]
        USING_GPU = True
        logging.info("DepthOptimization: Using CuPy backend.")
    except Exception as e:
        logging.warning(f"CuPy backend import failed ({e}).")

if DepthOptimization is None and _detect_cuda_available():
    try:
        DepthOptimization = _import_local_module("depth_optimization_gpu").DepthOptimization  # type: ignore[attr-defined]
        USING_GPU = True
        logging.info("DepthOptimization: Using Numba CUDA backend.")
    except Exception as e:
        logging.warning(f"Numba GPU backend import failed ({e}); falling back to CPU.")

if DepthOptimization is None:
    DepthOptimization = _import_local_module("depth_optimization_cpu").DepthOptimization  # type: ignore[attr-defined]
    USING_GPU = False


def is_gpu_enabled() -> bool:
    return USING_GPU


__all__ = ["DepthOptimization", "USING_GPU", "is_gpu_enabled"]
