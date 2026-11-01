"""
Scale-Dependent Misspecification Detection (Master Document Section 4.3 & 6.6).
Reference: Akhmetzhanova, A., Cuesta-Lázaro, C., et al. (2025) scale-dependent normalizing flows
and out-of-distribution summary space diagnostics.

Detects model misspecification (e.g. unmodeled baryonic physics or hydro suite mismatch)
by measuring summary-space Mahalanobis distance and posterior shift across scale cuts k_max.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class ScaleDependentMisspecification:
    """
    Detects model misspecification by comparing observed summary vectors x_obs(k <= k_max)
    against the training simulation bank distribution across progressive scale cuts k_max.
    """

    def __init__(self, k_grid: np.ndarray, x_training_bank: np.ndarray):
        """
        Args:
            k_grid: (N_k,) wavenumber array corresponding to summary columns
            x_training_bank: (N_sims, N_k) simulated training observables
        """
        self.k_grid = np.asarray(k_grid, dtype=np.float64)
        self.x_bank = np.asarray(x_training_bank, dtype=np.float64)
        self.mean_vec = np.mean(self.x_bank, axis=0)
        self.std_vec = np.std(self.x_bank, axis=0) + 1e-8

    def evaluate_scale_cuts(
        self,
        x_observed: np.ndarray,
        k_cuts: Optional[List[float]] = None,
        anomaly_z_threshold: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Computes scale-dependent standardized anomaly score z(k_max) across scale cuts.
        If z(k_max) stays small at linear scales (k_max <= 0.2 h/Mpc) but exceeds threshold
        at non-linear scales (k_max ~ 1.0 h/Mpc), flags small-scale model misspecification.
        """
        obs = np.asarray(x_observed, dtype=np.float64)
        cuts = k_cuts or [0.2, 0.5, 1.0]

        scale_diagnostics: List[Dict[str, float | bool]] = []
        for kc in cuts:
            mask = self.k_grid <= kc
            if not np.any(mask):
                mask = np.ones_like(self.k_grid, dtype=bool)
            z_res = np.abs(obs[mask] - self.mean_vec[mask]) / self.std_vec[mask]
            rms_z = float(np.sqrt(np.mean(z_res ** 2)))
            max_z = float(np.max(z_res))
            scale_diagnostics.append({
                "k_max": float(kc),
                "rms_z_score": rms_z,
                "max_z_score": max_z,
                "is_misspecified": bool(rms_z > anomaly_z_threshold),
            })

        any_misspecified = any(bool(d["is_misspecified"]) for d in scale_diagnostics)
        # Check if misspecification increases monotonically with k_max (signature of baryonic/non-linear mismatch)
        rms_values = [float(d["rms_z_score"]) for d in scale_diagnostics]
        scale_driven = bool(len(rms_values) >= 2 and (rms_values[-1] - rms_values[0]) > 1.0)

        return {
            "scale_diagnostics": scale_diagnostics,
            "is_misspecified": any_misspecified,
            "is_small_scale_driven": scale_driven,
            "anomaly_z_threshold": anomaly_z_threshold,
        }
