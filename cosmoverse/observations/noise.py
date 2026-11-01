"""
Observational noise injection: cosmic variance, shot noise, shape noise, and photo-z errors.
"""

from typing import Optional, Tuple

import numpy as np

from cosmoverse.observations.survey import SurveySpecification


def compute_gaussian_covariance_diagonal(
    k_centers: np.ndarray,
    pk: np.ndarray,
    survey: SurveySpecification,
    survey_volume_mpc3: float,
    galaxy_density_mpc3: float = 1e-3,
    delta_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes the diagonal Gaussian covariance for binned power spectrum P(k):
    Var(P(k)) = (2 * (2*pi)^3 / (V_survey * 4*pi * k^2 * Delta_k)) * ( P(k) + 1/n_g )^2
    """
    if delta_k is None:
        delta_k = np.gradient(k_centers)

    # Number of independent Fourier modes in spherical shell:
    # N_k = V_survey * (4*pi * k^2 * delta_k) / (2*pi)^3
    v_survey = survey_volume_mpc3 * survey.f_sky
    n_modes = v_survey * (4.0 * np.pi * (k_centers ** 2) * delta_k) / ((2.0 * np.pi) ** 3)
    n_modes = np.maximum(n_modes, 1.0)

    # Total power = clustering + shot noise
    p_total = pk + (1.0 / galaxy_density_mpc3)
    variance = (2.0 / n_modes) * (p_total ** 2)
    return np.sqrt(variance)


def inject_observational_noise(
    k_centers: np.ndarray,
    pk: np.ndarray,
    survey: SurveySpecification,
    survey_volume_mpc3: float = 1e9,
    galaxy_density_mpc3: float = 1e-3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds Gaussian cosmic variance and shot noise to clean theoretical or simulated P(k).

    Returns:
        noisy_pk: Observed power spectrum with noise
        sigma_pk: 1-sigma uncertainty per bin
    """
    rng = np.random.default_rng(seed)
    sigma_pk = compute_gaussian_covariance_diagonal(
        k_centers, pk, survey, survey_volume_mpc3, galaxy_density_mpc3
    )
    noise = rng.standard_normal(size=len(pk)) * sigma_pk
    noisy_pk = np.maximum(pk + noise, 1e-6)
    return noisy_pk, sigma_pk
