"""
Level 1 Cosmological Physics Extension: Dynamic Dark Energy w0-wa CPL
(Chevallier-Polarski-Linder) and Alcock-Paczynski BAO/RSD Distortions.

Models the time-evolving dark energy equation of state w(a) = w0 + wa * (1 - a)
probed by Stage-IV DESI/Euclid/LSST surveys, computing E(z), comoving distance
chi(z), angular diameter distance D_M(z), Hubble distance D_H(z), Alcock-Paczynski
parameters (alpha_parallel, alpha_perp), and growth rate f * sigma8(z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad

from cosmoverse.core.cosmology import C_KM_S, Cosmology
from cosmoverse.core.parameters import CosmologicalParameters


@dataclass(frozen=True)
class CPLDarkEnergyParameters:
    """
    Chevallier-Polarski-Linder (2001, 2003) dynamic dark energy parameters:
      w(z) = w0 + wa * z / (1 + z)
    Default values correspond to the DESI BAO + CMB + SN evolving dark energy benchmark.
    """

    w0: float = -0.82
    wa: float = -0.75

    @classmethod
    def lambda_cdm(cls) -> CPLDarkEnergyParameters:
        return cls(w0=-1.0, wa=0.0)


class DynamicDarkEnergyCosmology:
    """
    Evaluates expansion history, Alcock-Paczynski distortion parameters, and
    linear structure growth under dynamic w0-wa dark energy.
    """

    def __init__(
        self,
        cosmo_params: CosmologicalParameters | None = None,
        de_params: CPLDarkEnergyParameters | None = None,
    ) -> None:
        self.cosmo_params = cosmo_params or CosmologicalParameters.planck2018()
        self.de_params = de_params or CPLDarkEnergyParameters()
        self.fiducial_lcdm = Cosmology(self.cosmo_params)

    def equation_of_state_w(self, z: float | np.ndarray) -> np.ndarray:
        """Evaluates w(z) = w0 + wa * z / (1 + z)."""
        z_arr = np.asarray(z, dtype=np.float64)
        return self.de_params.w0 + self.de_params.wa * (z_arr / (1.0 + z_arr))

    def de_density_evolution(self, z: float | np.ndarray) -> np.ndarray:
        """
        Exact CPL dark energy density evolution:
        rho_DE(z) / rho_DE(0) = (1 + z)^(3*(1 + w0 + wa)) * exp(-3 * wa * z / (1 + z)).
        """
        z_arr = np.asarray(z, dtype=np.float64)
        exponent = 3.0 * (1.0 + self.de_params.w0 + self.de_params.wa)
        return np.power(1.0 + z_arr, exponent) * np.exp(-3.0 * self.de_params.wa * z_arr / (1.0 + z_arr))

    def e_z(self, z: float | np.ndarray) -> np.ndarray:
        """Dimensionless Hubble expansion rate E(z) = H(z) / H0 under w0-wa."""
        z_arr = np.asarray(z, dtype=np.float64)
        om = self.cosmo_params.omega_m
        ode = 1.0 - om
        e2 = om * np.power(1.0 + z_arr, 3.0) + ode * self.de_density_evolution(z_arr)
        return np.sqrt(np.maximum(e2, 1e-12))

    def comoving_distance_mpc(self, z: float) -> float:
        """Line-of-sight comoving distance D_M(z) [Mpc/h] at redshift z."""
        if z <= 0.0:
            return 0.0
        val, _ = quad(lambda zp: 1.0 / float(self.e_z(zp)), 0.0, float(z), epsrel=1e-6)
        return float((C_KM_S / 100.0) * val)

    def alcock_paczynski_parameters(self, z: float = 0.85) -> dict[str, float]:
        """
        Computes Alcock-Paczynski geometric dilation parameters relative to fiducial LambdaCDM:
          alpha_parallel(z) = E_fid(z) / E_w0wa(z)
          alpha_perp(z)     = D_M,w0wa(z) / D_M,fid(z)
          q_iso(z)          = (alpha_parallel * alpha_perp^2)^(1/3)
          epsilon_ap(z)     = (alpha_parallel / alpha_perp)^(1/3) - 1
        """
        e_fid = float(self.fiducial_lcdm.E(z))
        e_cpl = float(self.e_z(z))
        dm_fid = max(float(self.fiducial_lcdm.comoving_distance(z)), 1e-6)
        dm_cpl = max(self.comoving_distance_mpc(z), 1e-6)

        alpha_par = e_fid / e_cpl
        alpha_perp = dm_cpl / dm_fid
        q_iso = float((alpha_par * (alpha_perp**2)) ** (1.0 / 3.0))
        eps_ap = float(((alpha_par / alpha_perp) ** (1.0 / 3.0)) - 1.0)

        return {
            "z": float(z),
            "alpha_parallel": float(alpha_par),
            "alpha_perp": float(alpha_perp),
            "q_iso": q_iso,
            "epsilon_ap": eps_ap,
        }

    def f_sigma8(self, z: float | np.ndarray) -> np.ndarray:
        """
        Evaluates the redshift-space distortion growth rate combination f(z) * sigma8(z)
        using the Linder (2005) growth index gamma(w0, wa) = 0.55 + 0.05 * (1 + w(z=1)).
        """
        z_arr = np.asarray(z, dtype=np.float64)
        e_val = self.e_z(z_arr)
        omega_m_z = (self.cosmo_params.omega_m * np.power(1.0 + z_arr, 3.0)) / (e_val**2)

        w_z1 = float(self.equation_of_state_w(1.0))
        gamma_linder = 0.55 + 0.05 * (1.0 + w_z1)
        f_z = np.power(omega_m_z, gamma_linder)

        # Growth factor D(z) modified by dynamic dark energy equation of state
        d_lcdm = np.asarray([float(self.fiducial_lcdm.growth_factor(float(zv))) for zv in np.atleast_1d(z_arr)])
        de_mod = 1.0 - 0.045 * (1.0 + self.de_params.w0 + 0.35 * self.de_params.wa) / np.power(1.0 + np.atleast_1d(z_arr), 0.7)
        d_cpl = d_lcdm * de_mod
        res = f_z * self.cosmo_params.sigma8 * d_cpl
        return res if np.ndim(z) > 0 else float(res[0])

    def evaluate_desi_w0wa_signatures(self) -> dict[str, Any]:
        """
        Evaluates DESI-like redshift bins z in [0.3, 0.5, 0.8, 1.1, 1.5], comparing
        CPL (w0=-0.82, wa=-0.75) against LambdaCDM (w0=-1, wa=0).
        """
        z_bins = np.array([0.30, 0.50, 0.80, 1.10, 1.50])
        lcdm = DynamicDarkEnergyCosmology(
            cosmo_params=self.cosmo_params,
            de_params=CPLDarkEnergyParameters.lambda_cdm(),
        )
        fs8_cpl = np.asarray(self.f_sigma8(z_bins), dtype=np.float64)
        fs8_lcdm = np.asarray(lcdm.f_sigma8(z_bins), dtype=np.float64)
        ap_085 = self.alcock_paczynski_parameters(z=0.85)

        mean_fs8_shift_pct = float(np.mean(np.abs(fs8_cpl - fs8_lcdm) / fs8_lcdm) * 100.0)
        return {
            "w0": self.de_params.w0,
            "wa": self.de_params.wa,
            "z_bins": z_bins.tolist(),
            "fsigma8_cpl": fs8_cpl.tolist(),
            "fsigma8_lcdm": fs8_lcdm.tolist(),
            "alpha_parallel_z085": ap_085["alpha_parallel"],
            "alpha_perp_z085": ap_085["alpha_perp"],
            "mean_fsigma8_shift_pct": mean_fs8_shift_pct,
            "recovers_lcdm_limit": bool(
                np.allclose(
                    lcdm.de_density_evolution(z_bins),
                    np.ones_like(z_bins),
                    atol=1e-10,
                )
            ),
        }
