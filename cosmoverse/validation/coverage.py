"""
Empirical Bayesian posterior coverage analysis.
Checks whether nominal 1-sigma, 2-sigma credible intervals contain true parameters at the nominal rate.
"""

from typing import Dict, List, Optional

import numpy as np


class EmpiricalCoverage:
    """
    Computes empirical coverage curves for cosmological inference algorithms.
    """

    def __init__(self, param_names: List[str]):
        self.param_names = param_names

    def compute_coverage(
        self,
        true_thetas: np.ndarray,
        posterior_samples_list: List[np.ndarray],
        nominal_levels: Optional[List[float]] = None,
    ) -> Dict[str, Dict[float, float]]:
        """
        Args:
            true_thetas: (N_trials, N_params) ground truth values
            posterior_samples_list: List of length N_trials, each (N_samples, N_params)
            nominal_levels: Confidence levels, e.g. [0.50, 0.68, 0.90, 0.95]

        Returns:
            coverage: {param: {nominal_level: empirical_coverage}}
        """
        levels = nominal_levels or [0.50, 0.6827, 0.90, 0.9545]
        n_trials = len(true_thetas)
        results: Dict[str, Dict[float, float]] = {p: {} for p in self.param_names}

        for lvl in levels:
            alpha = (1.0 - lvl) / 2.0
            q_low = alpha * 100.0
            q_high = (1.0 - alpha) * 100.0

            for j, p in enumerate(self.param_names):
                contained_count = 0
                for i in range(n_trials):
                    post_chain = posterior_samples_list[i][:, j]
                    val_true = true_thetas[i, j]
                    low = np.percentile(post_chain, q_low)
                    high = np.percentile(post_chain, q_high)
                    if low <= val_true <= high:
                        contained_count += 1
                results[p][lvl] = contained_count / float(n_trials)

        return results
