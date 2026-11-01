"""
Example 04: Simulation-Based Calibration (SBC) rank statistic validation.
Verifies whether the Bayesian posterior estimator is statistically well-calibrated.
"""

import numpy as np

from cosmoverse.validation import SimulationBasedCalibration


def main():
    print("=" * 60)
    print("COSMOVERSE: Example 04 - Simulation-Based Calibration (SBC)")
    print("=" * 60)

    param_names = ["omega_m", "sigma8"]

    def prior_sampler() -> np.ndarray:
        om = np.random.uniform(0.15, 0.45)
        s8 = np.random.uniform(0.60, 1.00)
        return np.array([om, s8])

    def mock_simulator(theta: np.ndarray) -> np.ndarray:
        # Simple summary statistic: mean signal with Gaussian noise
        noise = np.random.normal(0, 0.05, size=2)
        return theta + noise

    def mock_infer(x_obs: np.ndarray, n_samples: int) -> np.ndarray:
        # Approximate Gaussian posterior
        post = np.random.normal(x_obs, 0.05, size=(n_samples, 2))
        return post

    sbc = SimulationBasedCalibration(
        sample_prior_fn=prior_sampler,
        simulate_fn=mock_simulator,
        infer_fn=mock_infer,
        param_names=param_names,
    )

    print("Running 30 SBC trials...")
    results = sbc.run_calibration(n_trials=30, n_posterior_samples=100, verbose=True)

    print("\nCalibration Results (Kolmogorov-Smirnov Test):")
    for param, metrics in results.items():
        print(f"  Parameter: {param}")
        print(f"    KS Statistic : {metrics['ks_statistic']:.4f}")
        print(f"    p-value      : {metrics['p_value']:.4f}")
        print(f"    Calibrated   : {metrics['is_calibrated']}")


if __name__ == "__main__":
    main()
