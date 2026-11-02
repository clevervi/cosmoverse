"""
Optional GPU acceleration and device audit for COSMOVERSE.

The core framework stays NumPy/SciPy. When CuPy and a CUDA device are
available, the batched Hankel transform used by the SBI training bank runs on
the GPU; otherwise the exact NumPy path is used.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from scipy import integrate, special

_KERNEL_CACHE: dict[tuple[int, int, float, float], np.ndarray] = {}


def _import_cupy():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="CUDA path could not be detected"
            )
            import cupy as cp
        return cp
    except Exception:
        return None


def cupy_available() -> bool:
    """Returns True only if CuPy imports and sees at least one CUDA device."""
    cp = _import_cupy()
    if cp is None:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def device_summary() -> dict[str, Any]:
    """Reports the active array backend and CUDA device details."""
    cp = _import_cupy()
    if cp is None:
        return {
            "backend": "numpy",
            "cupy_available": False,
            "gpu_count": 0,
            "gpu_name": None,
        }
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count == 0:
            return {
                "backend": "numpy",
                "cupy_available": True,
                "gpu_count": 0,
                "gpu_name": None,
            }
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props["name"]
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return {
            "backend": "cupy",
            "cupy_available": True,
            "gpu_count": count,
            "gpu_name": name,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "free_memory_gb": float(free_mem) / 1e9,
            "total_memory_gb": float(total_mem) / 1e9,
        }
    except Exception as exc:
        return {
            "backend": "numpy",
            "cupy_available": True,
            "gpu_count": 0,
            "gpu_name": None,
            "error": str(exc),
        }


def _hankel_kernel(ell: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    """
    Builds the fixed matrix K_ij = ell_i * w_i * J0(ell_i theta_j) / (2 pi),
    where w_i are composite Simpson weights on the ell grid.
    """
    ell_arr = np.asarray(ell, dtype=np.float64)
    theta_arr = np.asarray(theta_rad, dtype=np.float64)
    key = (
        len(ell_arr),
        len(theta_arr),
        float(ell_arr[0]),
        float(ell_arr[-1]),
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    j0_matrix = special.j0(np.outer(ell_arr, theta_arr))
    weights = integrate.simpson(np.eye(len(ell_arr)), x=ell_arr, axis=0)
    kernel = (
        ell_arr[:, np.newaxis]
        * weights[:, np.newaxis]
        * j0_matrix
        / (2.0 * np.pi)
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def batch_hankel_transform(
    c_ell: np.ndarray,
    ell: np.ndarray,
    theta_rad: np.ndarray,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Computes w(theta) for a batch of C_ell arrays:
      w_j = sum_i ell_i w_i C_ell,i J0(ell_i theta_j) / (2 pi).
    """
    c_ell_arr = np.asarray(c_ell, dtype=np.float64)
    kernel = _hankel_kernel(ell, theta_rad)
    if c_ell_arr.ndim != 2 or c_ell_arr.shape[1] != kernel.shape[0]:
        raise ValueError("c_ell must have shape (n_batch, n_ell)")

    if use_gpu and cupy_available():
        cp = _import_cupy()
        c_gpu = cp.asarray(c_ell_arr, dtype=cp.float64)
        k_gpu = cp.asarray(kernel, dtype=cp.float64)
        result = c_gpu @ k_gpu
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(result)

    return c_ell_arr @ kernel


def benchmark_devices(
    n_sims: int = 1600,
    n_ell: int = 1024,
    n_theta: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmarks the batched Hankel transform on CPU (BLAS) and GPU (CuPy)."""
    rng = np.random.default_rng(seed)
    ell = np.linspace(1.0, 8000.0, n_ell)
    theta_rad = np.deg2rad(np.linspace(1.0, 20.0, n_theta))
    c_ell = np.abs(rng.normal(size=(n_sims, n_ell)))
    kernel = _hankel_kernel(ell, theta_rad)

    start = time.perf_counter()
    cpu_result = c_ell @ kernel
    cpu_seconds = time.perf_counter() - start

    summary = device_summary()
    gpu_seconds = None
    max_abs_diff = None
    speedup = None
    if summary["backend"] == "cupy":
        cp = _import_cupy()
        c_gpu = cp.asarray(c_ell, dtype=cp.float64)
        k_gpu = cp.asarray(kernel, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gpu_result = c_gpu @ k_gpu
        cp.cuda.Stream.null.synchronize()
        gpu_seconds = time.perf_counter() - start
        gpu_result_np = cp.asnumpy(gpu_result)
        max_abs_diff = float(np.max(np.abs(gpu_result_np - cpu_result)))
        speedup = cpu_seconds / max(gpu_seconds, 1e-12)

    return {
        "device": summary,
        "n_sims": int(n_sims),
        "n_ell": int(n_ell),
        "n_theta": int(n_theta),
        "cpu_seconds": float(cpu_seconds),
        "gpu_seconds": gpu_seconds,
        "gpu_speedup": speedup,
        "max_abs_difference": max_abs_diff,
    }
