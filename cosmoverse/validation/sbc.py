"""
Simulation-Based Calibration (SBC) and Fixed-Parameter Calibration for validating Bayesian inference.
References:
  - Talts et al. (2018) "Validating Bayesian Inference Algorithms with Simulation-Based Calibration"
  - Master Document Section 6.6: Both prior-averaged coverage (SBC/TARP) AND fixed-parameter
    behavior must be verified; SBC/TARP acts as a blocking acceptance gate.
"""

from typing import Any, Callable, Dict, List

import numpy as np
from scipy import stats


class SimulationBasedCalibration:
    """
    Performs Simulation-Based Calibration (SBC) and fixed-parameter diagnostics on an inference engine.
    Ensures posterior distributions are neither overconfident nor underconfident.
    """

    def __init__(
        self,
        sample_prior_fn: Callable[[], np.ndarray],
        simulate_fn: Callable[[np.ndarray], np.ndarray],
        infer_fn: Callable[[np.ndarray, int], np.ndarray],
        param_names: List[str],
        coverage_tolerance: float = 0.12,
    ):
        self.sample_prior_fn = sample_prior_fn
        self.simulate_fn = simulate_fn
        self.infer_fn = infer_fn
        self.param_names = param_names
        self.n_params = len(param_names)
        self.coverage_tolerance = coverage_tolerance

    def run_calibration(
        self,
        n_trials: int = 100,
        n_posterior_samples: int = 200,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs SBC trials, computes rank statistics, 68%/95% empirical coverage,
        and evaluates the blocking acceptance gate.
        """
        ranks: Dict[str, List[int]] = {name: [] for name in self.param_names}
        in_68: Dict[str, int] = dict.fromkeys(self.param_names, 0)
        in_95: Dict[str, int] = dict.fromkeys(self.param_names, 0)

        for trial in range(n_trials):
            if verbose and (trial % 20 == 0 or trial == n_trials - 1):
                print(f"[SBC Trial] {trial+1}/{n_trials}...")

            theta_star = self.sample_prior_fn()
            x_star = self.simulate_fn(theta_star)
            posterior_draws = self.infer_fn(x_star, n_posterior_samples)

            for j, name in enumerate(self.param_names):
                col = posterior_draws[:, j]
                rank = int(np.sum(col < theta_star[j]))
                ranks[name].append(rank)

                p16, p84 = np.percentile(col, [15.865, 84.135])
                p2_5, p97_5 = np.percentile(col, [2.275, 97.725])
                if p16 <= theta_star[j] <= p84:
                    in_68[name] += 1
                if p2_5 <= theta_star[j] <= p97_5:
                    in_95[name] += 1

        metrics: Dict[str, Dict[str, Any]] = {}
        for name in self.param_names:
            normed_ranks = np.array(ranks[name]) / float(n_posterior_samples)
            ks_res = stats.kstest(normed_ranks, "uniform")
            cov68 = in_68[name] / float(n_trials)
            cov95 = in_95[name] / float(n_trials)
            is_cal = bool(ks_res.pvalue > 0.01)
            passes_gate = bool(
                is_cal
                and abs(cov68 - 0.6827) <= self.coverage_tolerance
                and abs(cov95 - 0.9545) <= self.coverage_tolerance
            )
            metrics[name] = {
                "ks_statistic": float(ks_res.statistic),
                "p_value": float(ks_res.pvalue),
                "is_calibrated": is_cal,
                "coverage_68": cov68,
                "coverage_95": cov95,
                "passes_acceptance_gate": passes_gate,
                "mean_rank": float(np.mean(ranks[name])),
                "expected_mean_rank": float(n_posterior_samples / 2.0),
            }

        return metrics

    def run_fixed_parameter_check(
        self,
        fixed_theta: np.ndarray,
        n_repeats: int = 30,
        n_posterior_samples: int = 200,
    ) -> Dict[str, Dict[str, float]]:
        """
        Verifies posterior calibration at a fixed fiducial parameter point theta_0
        across repeated noise/initial-condition realizations (Master Document Section 6.6).
        """
        fixed_arr = np.asarray(fixed_theta, dtype=np.float64)
        biases: Dict[str, List[float]] = {name: [] for name in self.param_names}
        z_scores: Dict[str, List[float]] = {name: [] for name in self.param_names}
        cov68_hits: Dict[str, int] = dict.fromkeys(self.param_names, 0)

        for _ in range(n_repeats):
            x_obs = self.simulate_fn(fixed_arr)
            draws = self.infer_fn(x_obs, n_posterior_samples)
            for j, name in enumerate(self.param_names):
                col = draws[:, j]
                mean_est = float(np.mean(col))
                std_est = float(np.std(col)) + 1e-12
                biases[name].append(mean_est - fixed_arr[j])
                z_scores[name].append((mean_est - fixed_arr[j]) / std_est)
                p16, p84 = np.percentile(col, [15.865, 84.135])
                if p16 <= fixed_arr[j] <= p84:
                    cov68_hits[name] += 1

        report: Dict[str, Dict[str, float]] = {}
        for name in self.param_names:
            report[name] = {
                "mean_bias": float(np.mean(biases[name])),
                "mean_bias_in_sigma": float(np.mean(np.abs(z_scores[name]))),
                "local_coverage_68": cov68_hits[name] / float(n_repeats),
            }
        return report
