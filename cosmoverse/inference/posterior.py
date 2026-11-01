"""
Posterior analysis, credible intervals, Figure of Merit (FoM), and tension metrics.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PosteriorSummary:
    """
    Analyzes MCMC or SBI posterior parameter chains.
    """

    def __init__(self, samples: np.ndarray, param_names: Optional[List[str]] = None):
        """
        Args:
            samples: (N_samples, N_params) array of posterior draws
            param_names: List of parameter names (e.g. ['omega_m', 'sigma8'])
        """
        self.samples = np.asarray(samples)
        self.n_samples, self.n_params = self.samples.shape
        self.param_names = param_names or [f"param_{i}" for i in range(self.n_params)]

    def credible_intervals(self, level: float = 0.6827) -> Dict[str, Tuple[float, float, float]]:
        """
        Computes median and lower/upper credible interval bounds.
        Returns: {param: (median, lower_err, upper_err)}
        """
        alpha = (1.0 - level) / 2.0
        q_low, q_med, q_high = alpha * 100.0, 50.0, (1.0 - alpha) * 100.0

        intervals = {}
        for idx, name in enumerate(self.param_names):
            vals = self.samples[:, idx]
            low, med, high = np.percentile(vals, [q_low, q_med, q_high])
            intervals[name] = (float(med), float(med - low), float(high - med))
        return intervals

    def figure_of_merit(self, param1: str, param2: str) -> float:
        """
        Computes the Dark Energy Task Force (DETF) Figure of Merit (FoM):
        FoM = 1 / sqrt(det(Cov(param1, param2)))
        """
        idx1 = self.param_names.index(param1)
        idx2 = self.param_names.index(param2)
        sub_samples = self.samples[:, [idx1, idx2]]
        cov = np.cov(sub_samples, rowvar=False)
        det = np.linalg.det(cov)
        return float(1.0 / np.sqrt(np.maximum(det, 1e-18)))

    def tension_metric(self, other_summary: "PosteriorSummary", param: str) -> float:
        """
        Quantifies 1D Gaussian tension T = |mu1 - mu2| / sqrt(sigma1^2 + sigma2^2) in sigmas.
        """
        idx1 = self.param_names.index(param)
        idx2 = other_summary.param_names.index(param)

        mu1 = float(np.mean(self.samples[:, idx1]))
        std1 = float(np.std(self.samples[:, idx1]))

        mu2 = float(np.mean(other_summary.samples[:, idx2]))
        std2 = float(np.std(other_summary.samples[:, idx2]))

        denom = np.sqrt(std1 ** 2 + std2 ** 2)
        return float(abs(mu1 - mu2) / denom)

    def summary(self) -> Dict[str, Any]:
        """Returns statistical overview dictionary."""
        ci68 = self.credible_intervals(0.6827)
        res = {}
        for name in self.param_names:
            idx = self.param_names.index(name)
            med, l_err, u_err = ci68[name]
            res[name] = {
                "mean": float(np.mean(self.samples[:, idx])),
                "std": float(np.std(self.samples[:, idx])),
                "median": med,
                "err_minus": l_err,
                "err_plus": u_err,
            }
        return res
