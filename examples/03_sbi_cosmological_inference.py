"""
Example 03: Simulation-Based Inference (SBI) for cosmological parameters.
Constrains Omega_m and sigma8 from synthetic observed power spectra.
"""

import numpy as np

from cosmoverse.core import CosmologicalParameters, Cosmology
from cosmoverse.inference import SBIEngine


def fast_forward_simulator(theta: np.ndarray) -> np.ndarray:
    """
    Mock forward simulator: computes power spectrum at 5 key wavenumbers.
    theta = [omega_m, sigma8]
    """
    om, s8 = theta[0], theta[1]
    cosmo = Cosmology(CosmologicalParameters(omega_m=om, sigma8=s8))
    k_eval = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
    pk = np.array([cosmo.linear_matter_power_spectrum(kv) for kv in k_eval])
    # Add 2% measurement noise
    noise = np.random.normal(0, 0.02 * pk)
    return np.log(pk + noise)


def main():
    print("=" * 60)
    print("COSMOVERSE: Example 03 - Simulation-Based Inference (SBI)")
    print("=" * 60)

    bounds = {
        "omega_m": (0.15, 0.45),
        "sigma8": (0.60, 1.00),
    }

    sbi = SBIEngine(simulator_fn=fast_forward_simulator, param_bounds=bounds, seed=42)

    print("Generating simulation bank (40 simulations)...")
    sbi.generate_training_bank(n_simulations=40, verbose=True)

    print("\nFitting conditional normalizing flow...")
    sbi.fit_density_estimator(n_layers=3)

    # True observation
    theta_true = np.array([0.315, 0.811])
    x_obs = fast_forward_simulator(theta_true)
    print(f"\nGround truth parameters: Omega_m={theta_true[0]:.3f}, sigma8={theta_true[1]:.3f}")

    print("Sampling posterior distribution p(theta | x_obs)...")
    posterior = sbi.infer(x_obs, n_posterior_samples=1000)

    summary = posterior.summary()
    print("\nPosterior Parameter Constraints (68% Credible Interval):")
    for name, stats in summary.items():
        print(f"  {name:10}: {stats['median']:.4f} -{stats['err_minus']:.4f} +{stats['err_plus']:.4f}")

    fom = posterior.figure_of_merit("omega_m", "sigma8")
    print(f"\nFigure of Merit (FoM): {fom:.2f}")


if __name__ == "__main__":
    main()
