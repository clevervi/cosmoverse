"""
Stage 3 Observational Forward Model (Master Document Section 6.3).
Explicitly bridges periodic physical/emulated fields (Level 1) to realistic survey
observables (Level 2) BEFORE running SBI (Level 3).

Includes:
  1. Survey geometry & 3D window masks
  2. Selection functions & observational incompleteness
  3. Galaxy bias (linear b1 & quadratic b2)
  4. Redshift-Space Distortions (RSD: Kaiser + Finger-of-God) & redshift errors (P0, P2, P4 multipoles)
  5. Poisson shot noise
  6. Instrumental systematics & weak lensing multiplicative shear calibration (1 + m)^2
"""

from typing import Dict, Optional

import numpy as np

from cosmoverse.core.parameters import ObservationalNuisanceParameters
from cosmoverse.observations.noise import inject_observational_noise
from cosmoverse.observations.selection import SelectionFunction
from cosmoverse.observations.survey import SurveySpecification


class GalaxyBiasModel:
    """
    Models galaxy bias mapping dark matter overdensity delta_m to galaxy overdensity delta_g.
    """

    def __init__(self, b1: float = 1.5, b2: float = 0.0):
        self.b1 = b1
        self.b2 = b2

    def apply_to_field(self, delta_m: np.ndarray) -> np.ndarray:
        """
        Applies local Eulerian bias expansion while enforcing zero-mean galaxy overdensity:
        delta_g(x) = b1 * (delta_m(x) - <delta_m>) + 0.5 * b2 * (delta_m(x)^2 - <delta_m^2>)
        """
        dm_zero = delta_m - np.mean(delta_m)
        var_m = float(np.mean(dm_zero ** 2))
        return self.b1 * dm_zero + 0.5 * self.b2 * ((dm_zero ** 2) - var_m)

    def apply_to_pk(self, pk_m: np.ndarray) -> np.ndarray:
        """Linear scale galaxy power spectrum P_gg(k) = b1^2 * P_mm(k)."""
        return (self.b1 ** 2) * pk_m


class RedshiftSpaceDistortionModel:
    """
    Models Redshift-Space Distortions (RSD: Kaiser boost + Finger-of-God velocity dispersion)
    and spectroscopic/photometric redshift errors. Computes Legendre multipoles P0(k), P2(k), P4(k).
    """

    def __init__(
        self,
        b1: float = 1.5,
        growth_rate_f: float = 0.53,
        sigma_fog: float = 3.5,
        sigma_z_comoving: float = 1.5,
    ):
        self.b1 = b1
        self.growth_rate_f = growth_rate_f
        self.sigma_fog = sigma_fog
        self.sigma_z_comoving = sigma_z_comoving

    def anisotropic_pk(self, k: np.ndarray, mu: np.ndarray, pk_matter: np.ndarray) -> np.ndarray:
        """
        Evaluates 2D redshift-space galaxy power spectrum P_g(k, mu):
        P_g(k, mu) = (b1 + f * mu^2)^2 * F_FoG(k, mu) * exp(-(k * mu * sigma_z)^2) * P_m(k)
        where k has shape (N_k, 1) and mu has shape (1, N_mu).
        """
        kaiser = (self.b1 + self.growth_rate_f * (mu ** 2)) ** 2
        k_mu_sq = (k ** 2) * (mu ** 2)
        # Lorentzian Finger-of-God damping + Gaussian redshift error damping
        fog_damping = 1.0 / (1.0 + 0.5 * k_mu_sq * (self.sigma_fog ** 2)) ** 2
        z_err_damping = np.exp(-k_mu_sq * (self.sigma_z_comoving ** 2))
        return kaiser * fog_damping * z_err_damping * pk_matter

    def compute_multipoles(
        self,
        k: np.ndarray,
        pk_matter: np.ndarray,
        shot_noise: float = 0.0,
        n_mu: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Computes the Redshift-Space Multipoles P_0(k) (monopole), P_2(k) (quadrupole),
        and P_4(k) (hexadecapole):
          P_ell(k) = ((2*ell + 1) / 2) * int_{-1}^{1} d(mu) P_g(k, mu) L_ell(mu)
        """
        k_col = np.asarray(k, dtype=np.float64)[:, np.newaxis]
        pk_col = np.asarray(pk_matter, dtype=np.float64)[:, np.newaxis]

        # Gauss-Legendre quadrature on [-1, 1]
        mu_nodes, weights = np.polynomial.legendre.leggauss(n_mu)
        mu_row = mu_nodes[np.newaxis, :]
        w_row = weights[np.newaxis, :]

        pk_2d = self.anisotropic_pk(k_col, mu_row, pk_col)

        # Legendre polynomials L0, L2, L4
        l0 = np.ones_like(mu_row)
        l2 = 0.5 * (3.0 * (mu_row ** 2) - 1.0)
        l4 = 0.125 * (35.0 * (mu_row ** 4) - 30.0 * (mu_row ** 2) + 3.0)

        p0 = 0.5 * np.sum(w_row * pk_2d * l0, axis=1) + shot_noise
        p2 = 2.5 * np.sum(w_row * pk_2d * l2, axis=1)
        p4 = 4.5 * np.sum(w_row * pk_2d * l4, axis=1)

        return {
            "k": np.asarray(k, dtype=np.float64),
            "P0": p0,
            "P2": p2,
            "P4": p4,
        }

    @staticmethod
    def displace_particles_to_redshift_space(
        positions: np.ndarray,
        velocities: np.ndarray,
        scale_a: float,
        hubble_z: float,
        h_param: float,
        box_size: float,
        los_axis: int = 2,
    ) -> np.ndarray:
        """
        Maps real-space particle coordinates x to redshift-space coordinates s along line-of-sight:
        s = x + (v_los / (a * H(a) / h)) * e_los (wrapped periodically in box_size).
        """
        s_pos = np.copy(positions)
        h_scaled = hubble_z / h_param  # km/s / (Mpc/h)
        s_pos[:, los_axis] = np.mod(
            s_pos[:, los_axis] + velocities[:, los_axis] / (scale_a * h_scaled + 1e-12),
            box_size,
        )
        return s_pos


class ObservationalForwardPipeline:
    """
    Stage 3 End-to-End Observational Realism Pipeline (Level 2).
    Connects physical matter power spectra or 3D fields to Stage-IV survey observables
    prior to Stage 4 Bayesian Inference (SBI).
    """

    def __init__(
        self,
        survey: Optional[SurveySpecification] = None,
        nuisance: Optional[ObservationalNuisanceParameters] = None,
        selection: Optional[SelectionFunction] = None,
        growth_rate_f: float = 0.53,
    ):
        self.survey = survey or SurveySpecification.desi_lrg()
        self.nuisance = nuisance or ObservationalNuisanceParameters()
        self.selection = selection or SelectionFunction(completeness_rate=0.95)
        self.growth_rate_f = growth_rate_f

    def apply_3d_survey_mask_and_noise(
        self,
        delta_matter: np.ndarray,
        box_size: float,
        galaxy_density_mpc3: float = 1e-3,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Applies galaxy bias, survey angular/footprint mask, incompleteness, and 3D Poisson shot noise
        to a periodic 3D density field.
        """
        rng = np.random.default_rng(seed)
        bias_model = GalaxyBiasModel(b1=self.nuisance.b1, b2=self.nuisance.b2)
        delta_g = bias_model.apply_to_field(delta_matter)

        # Survey window mask + incompleteness damping
        n = delta_g.shape[0]
        mask_3d = np.ones((n, n, n), dtype=np.float64) * self.selection.completeness_rate
        # Mask out 5% boundary edge to emulate non-periodic survey footprint geometry
        edge = max(1, n // 20)
        mask_3d[:edge, :, :] = 0.0

        # Voxel volume and shot noise std: 1 / sqrt(n_gal * V_voxel)
        v_voxel = (box_size / n) ** 3
        n_per_voxel = max(galaxy_density_mpc3 * v_voxel * self.selection.completeness_rate, 1e-4)
        shot_std = np.sqrt(self.nuisance.shot_noise_scale / n_per_voxel) * 0.05
        shot_field = rng.normal(0.0, shot_std, size=delta_g.shape)

        observed_field = mask_3d * (delta_g + shot_field)
        return observed_field

    def observe_power_spectrum(
        self,
        k: np.ndarray,
        pk_baryon_matter: np.ndarray,
        survey_volume_mpc3: float = 1e9,
        galaxy_density_mpc3: float = 1e-3,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Transforms a baryon-corrected matter power spectrum P_hydro(k) into observed
        redshift-space multipoles (P0, P2, P4) and weak-lensing calibrated power spectrum,
        including completeness, shot noise, and cosmic variance.
        """
        eff_density = max(galaxy_density_mpc3 * self.selection.completeness_rate, 1e-7)
        shot_noise = self.nuisance.shot_noise_scale / eff_density

        rsd = RedshiftSpaceDistortionModel(
            b1=self.nuisance.b1,
            growth_rate_f=self.growth_rate_f,
            sigma_fog=self.nuisance.sigma_fog,
            sigma_z_comoving=3000.0 * self.nuisance.sigma_z,
        )
        multipoles = rsd.compute_multipoles(k, pk_baryon_matter, shot_noise=shot_noise)

        # Inject cosmic variance + shot noise into monopole P0(k)
        p0_noisy, sigma_p0 = inject_observational_noise(
            k_centers=k,
            pk=multipoles["P0"],
            survey=self.survey,
            survey_volume_mpc3=survey_volume_mpc3,
            galaxy_density_mpc3=eff_density,
            seed=seed,
        )

        # Weak lensing shear power calibration: (1 + m)^2 * P_hydro(k)
        pk_lensing_cal = ((1.0 + self.nuisance.shear_m_bias) ** 2) * pk_baryon_matter

        return {
            "k": k,
            "P0_clean": multipoles["P0"],
            "P0_obs": p0_noisy,
            "sigma_P0": sigma_p0,
            "P2_clean": multipoles["P2"],
            "P4_clean": multipoles["P4"],
            "P_lensing_calibrated": pk_lensing_cal,
        }
