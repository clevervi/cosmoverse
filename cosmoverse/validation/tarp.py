"""
TARP: Tests of Accuracy with Random Points (Resultado 3 — Master Document Section 4.3, 6.6, 13.3).
Reference: Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023), ICML
"Sampling-Based Accuracy Testing of Posterior Estimators for General Inference".

Provides a sufficient and necessary multivariate empirical coverage test using only posterior
samples, acting as a blocking acceptance gate before advancing roadmap stages.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats


class TARPCalibration:
    """
    Tests of Accuracy with Random Points (TARP) for multivariate Bayesian posteriors.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        coverage_tolerance: float = 0.12,
        seed: int = 42,
    ):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.param_dim = len(self.param_names)
        self.coverage_tolerance = coverage_tolerance
        self.seed = seed

        self._lows = np.array([b[0] for b in param_bounds.values()], dtype=np.float64)
        self._spans = np.array([b[1] - b[0] for b in param_bounds.values()], dtype=np.float64) + 1e-12

    def _normalize(self, theta: np.ndarray) -> np.ndarray:
        """Maps parameter values to the unit hypercube [0, 1]^D."""
        return (np.asarray(theta, dtype=np.float64) - self._lows) / self._spans

    def evaluate(
        self,
        true_thetas: np.ndarray,
        posterior_samples: np.ndarray,
        alpha_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes TARP Expected Coverage Probability (ECP) curve and blocking acceptance gate.

        Args:
            true_thetas: (N_trials, D) ground truth parameter vectors
            posterior_samples: (N_trials, N_samples, D) posterior draws for each trial
            alpha_grid: Credibility levels in [0, 1]
        """
        rng = np.random.default_rng(self.seed)
        thetas_arr = np.asarray(true_thetas, dtype=np.float64)
        post_arr = np.asarray(posterior_samples, dtype=np.float64)
        n_trials, n_samples, d = post_arr.shape

        if alpha_grid is None:
            alpha_grid = np.linspace(0.0, 1.0, 21)

        # Normalize to [0, 1]^D
        theta_star_norm = self._normalize(thetas_arr)  # (N_trials, D)
        post_norm = (post_arr - self._lows[np.newaxis, np.newaxis, :]) / self._spans[np.newaxis, np.newaxis, :]

        # Draw random reference points theta_r ~ Uniform([0, 1]^D)
        theta_ref = rng.uniform(0.0, 1.0, size=(n_trials, d))

        # Distance from reference point to true parameter: R_i = ||theta*_i - theta_r,i||_2
        dist_true = np.linalg.norm(theta_star_norm - theta_ref, axis=1)  # (N_trials,)

        # Distances from reference point to posterior samples: (N_trials, N_samples)
        dist_post = np.linalg.norm(post_norm - theta_ref[:, np.newaxis, :], axis=2)

        # Fraction of posterior samples inside the ball B(theta_r, R_i)
        fractions = np.mean(dist_post < dist_true[:, np.newaxis], axis=1)  # (N_trials,)

        # Empirical coverage at each credibility level alpha: P(f <= alpha)
        ecp = np.array([float(np.mean(fractions <= a)) for a in alpha_grid])

        # Specific coverage at 68.27% and 95.45%
        cov_68 = float(np.mean(fractions <= 0.6827))
        cov_95 = float(np.mean(fractions <= 0.9545))

        ks_res = stats.kstest(fractions, "uniform")
        integrated_cal_error = float(np.mean(np.abs(ecp - alpha_grid)))

        passes_68 = abs(cov_68 - 0.6827) <= self.coverage_tolerance
        passes_95 = abs(cov_95 - 0.9545) <= self.coverage_tolerance
        passes_gate = bool(passes_68 and passes_95 and ks_res.pvalue > 0.01)

        return {
            "alpha_grid": alpha_grid,
            "ecp_curve": ecp,
            "coverage_68": cov_68,
            "coverage_95": cov_95,
            "ks_statistic": float(ks_res.statistic),
            "ks_p_value": float(ks_res.pvalue),
            "integrated_calibration_error": integrated_cal_error,
            "passes_acceptance_gate": passes_gate,
        }
