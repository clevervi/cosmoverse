"""
Gaussian and Non-Gaussian summary statistics for Resultado 1 (Emulator Precision):
  1. Power Spectrum MAPE (%) for k < k_max (default 1.0 h/Mpc)
  2. One-Point Overdensity PDF P(delta)
  3. 3D Peak Counts N_peaks(nu) as a function of significance nu = delta / sigma
  4. 3D Minkowski Functionals (V0, V1, V2, V3) on periodic 3D excursion sets
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from cosmoverse.simulation.nbody import estimate_power_spectrum_3d


def compute_pk_mape(
    pk_pred: np.ndarray,
    pk_true: np.ndarray,
    k: Optional[np.ndarray] = None,
    k_max: float = 1.0,
) -> float:
    """
    Computes Mean Absolute Percentage Error (MAPE, in %) between predicted and true P(k)
    restricted to scales k <= k_max (default 1.0 h/Mpc per Master Document Section 4 & 13).
    """
    pk_pred_arr = np.asarray(pk_pred, dtype=np.float64)
    pk_true_arr = np.asarray(pk_true, dtype=np.float64)

    if k is not None:
        mask = (np.asarray(k) <= k_max) & (pk_true_arr > 1e-12)
        if not np.any(mask):
            mask = pk_true_arr > 1e-12
    else:
        mask = pk_true_arr > 1e-12

    rel_err = np.abs(pk_pred_arr[mask] - pk_true_arr[mask]) / np.maximum(np.abs(pk_true_arr[mask]), 1e-12)
    return float(100.0 * np.mean(rel_err))


def compute_one_point_pdf(
    field: np.ndarray,
    n_bins: int = 25,
    nu_range: Tuple[float, float] = (-3.0, 5.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the normalized 1-point Probability Distribution Function (PDF)
    of the standardized density contrast nu = (delta - <delta>) / sigma_delta.

    Returns:
        bin_centers: (n_bins,) centers in units of sigma
        pdf_values: (n_bins,) probability density
    """
    std = float(np.std(field)) + 1e-12
    nu_field = (field - np.mean(field)) / std
    hist, edges = np.histogram(nu_field.ravel(), bins=n_bins, range=nu_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def compute_peak_counts_3d(
    field: np.ndarray,
    n_bins: int = 12,
    nu_range: Tuple[float, float] = (-1.0, 4.5),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies 3D local maxima (peaks) where a voxel is strictly greater than all 26 neighbors
    under periodic boundary conditions, and bins them by peak height significance nu = delta / sigma.

    Returns:
        nu_centers: (n_bins,) threshold significance centers
        peak_density: (n_bins,) normalized fraction of peaks per bin
    """
    std = float(np.std(field)) + 1e-12
    nu_field = (field - np.mean(field)) / std

    is_peak = np.ones(nu_field.shape, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                shifted = np.roll(nu_field, shift=(dx, dy, dz), axis=(0, 1, 2))
                is_peak &= (nu_field > shifted)

    peak_heights = nu_field[is_peak]
    edges = np.linspace(nu_range[0], nu_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(peak_heights, bins=edges)
    # Normalize by total grid volume in voxels
    peak_fraction = counts.astype(np.float64) / float(nu_field.size)
    return centers, peak_fraction


def compute_minkowski_functionals_3d(
    field: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes the four 3D Minkowski Functionals (V0, V1, V2, V3) as functions of
    excursion threshold nu = (delta - <delta>) / sigma_delta on a 3D periodic lattice:
      - V0(nu): Volume fraction of excursion set {nu(x) >= nu_th}
      - V1(nu): Surface area density via boundary face transitions
      - V2(nu): Mean curvature integral density via Crofton plaque/edge configurations
      - V3(nu): Euler characteristic (genus / topology) density chi = V - E + F - C
    """
    if thresholds is None:
        thresholds = np.linspace(-2.5, 3.5, 13)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)

    std = float(np.std(field)) + 1e-12
    nu_field = (field - np.mean(field)) / std
    n_total = float(nu_field.size)

    v0 = np.zeros_like(thresholds)
    v1 = np.zeros_like(thresholds)
    v2 = np.zeros_like(thresholds)
    v3 = np.zeros_like(thresholds)

    for idx, th in enumerate(thresholds):
        exc = (nu_field >= th).astype(np.int8)

        # V0: Volume fraction
        n_cubes = float(np.sum(exc))
        v0[idx] = n_cubes / n_total

        # Shift along principal axes (periodic)
        sx = np.roll(exc, -1, axis=0)
        sy = np.roll(exc, -1, axis=1)
        sz = np.roll(exc, -1, axis=2)

        sxy = np.roll(sx, -1, axis=1)
        sxz = np.roll(sx, -1, axis=2)
        syz = np.roll(sy, -1, axis=2)
        sxyz = np.roll(sxy, -1, axis=2)

        # V1: Boundary surface area density (proportion of active-inactive adjacent pairs)
        b_x = np.sum(np.abs(exc - sx))
        b_y = np.sum(np.abs(exc - sy))
        b_z = np.sum(np.abs(exc - sz))
        v1[idx] = float(b_x + b_y + b_z) / (6.0 * n_total)

        # Discrete cubical complex counts for V2 and V3 (Adler 1981 / Schmalzing & Buchert 1997)
        # Active faces (2-cells): shared by at least one active 3-voxel in the dual lattice
        f_xy = np.sum(exc | sx | sy | sxy)
        f_xz = np.sum(exc | sx | sz | sxz)
        f_yz = np.sum(exc | sy | sz | syz)
        n_faces = float(f_xy + f_xz + f_yz)

        # Active edges (1-cells)
        e_x = np.sum(exc | sy | sz | syz)
        e_y = np.sum(exc | sx | sz | sxz)
        e_z = np.sum(exc | sx | sy | sxy)
        n_edges = float(e_x + e_y + e_z)

        # Active vertices (0-cells)
        n_vertices = float(np.sum(exc | sx | sy | sz | sxy | sxz | syz | sxyz))

        # V2: Integrated mean curvature density (Crofton formula)
        v2[idx] = (2.0 * n_edges - 2.0 * n_faces + 3.0 * n_cubes - n_vertices) / (3.0 * n_total)

        # V3: Euler characteristic density chi = N_v - N_e + N_f - N_c
        v3[idx] = (n_vertices - n_edges + n_faces - n_cubes) / n_total

    return {
        "thresholds": thresholds,
        "V0": v0,
        "V1": v1,
        "V2": v2,
        "V3": v3,
    }


def evaluate_emulator_result1(
    field_pred: np.ndarray,
    field_true: np.ndarray,
    box_size: float,
    k_max: float = 1.0,
    mape_threshold_pct: float = 1.0,
) -> Dict[str, Any]:
    """
    Comprehensive evaluator for Resultado 1 (Master Document Section 4.1 & 13.1):
    Evaluates Gaussian P(k) MAPE at k <= k_max and Non-Gaussian statistics
    (1-point PDF, 3D Peak Counts, 3D Minkowski Functionals).
    """
    k_bins, pk_pred = estimate_power_spectrum_3d(field_pred, box_size, n_bins=20)
    _, pk_true = estimate_power_spectrum_3d(field_true, box_size, n_bins=20)

    pk_mape = compute_pk_mape(pk_pred, pk_true, k=k_bins, k_max=k_max)

    _, pdf_pred = compute_one_point_pdf(field_pred)
    _, pdf_true = compute_one_point_pdf(field_true)
    pdf_l1_error = float(np.mean(np.abs(pdf_pred - pdf_true)))

    _, peaks_pred = compute_peak_counts_3d(field_pred)
    _, peaks_true = compute_peak_counts_3d(field_true)
    peak_counts_mae = float(np.mean(np.abs(peaks_pred - peaks_true)))

    mf_pred = compute_minkowski_functionals_3d(field_pred)
    mf_true = compute_minkowski_functionals_3d(field_true)
    minkowski_errors = {
        key: float(np.mean(np.abs(mf_pred[key] - mf_true[key])))
        for key in ("V0", "V1", "V2", "V3")
    }

    return {
        "pk_mape_pct": pk_mape,
        "passes_pk_threshold": bool(pk_mape < mape_threshold_pct),
        "one_point_pdf_l1_error": pdf_l1_error,
        "peak_counts_mae": peak_counts_mae,
        "minkowski_mae": minkowski_errors,
        "k_bins": k_bins,
        "pk_pred": pk_pred,
        "pk_true": pk_true,
    }
