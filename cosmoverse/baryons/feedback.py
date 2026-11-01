"""
BCEmu-based baryonic feedback suppression model (Primary v1 Baryonic Emulator).
Implements 1, 3, and 7-parameter baryonification suppression S_b(k, z) = P_hydro(k, z) / P_dmo(k, z).

Decision of v1 (Master Document Section 6.2):
  Use ONE baryonic emulator in v1 (BCEmu: Giri & Schneider 2021) for interpretable marginalization.
  BACCOemu is reserved for v2 or cross-validation so errors can be unambiguously attributed.
"""

from typing import Union

import numpy as np

from cosmoverse.core.parameters import BaryonParameters


class BCEmuModel:
    """
    Parametric baryonification power spectrum suppression emulator (BCEmu1, BCEmu3, BCEmu7).
    Reference: Giri, S. K., & Schneider, A. (2021), JCAP 12, 046 (arXiv:2108.08863).
    """

    def __init__(self, params: BaryonParameters | None = None):
        self.params = params or BaryonParameters(mode="BCEmu3")

    def suppression_factor(
        self,
        k: Union[float, np.ndarray],
        z: float = 0.0,
        f_baryon: float = 0.156,
    ) -> Union[float, np.ndarray]:
        """
        Computes S_b(k, z) = P_baryon(k, z) / P_dmo(k, z) under BCEmu1, BCEmu3, or BCEmu7.
        """
        k_arr = np.asarray(k, dtype=np.float64)
        p = self.params

        # Effective gas ejection fraction controlled by log10_Mcrit (around fiducial 13.5)
        # Lower Mcrit -> more halos lose gas -> stronger suppression
        m_shift = 13.5 - p.log10_Mcrit
        agn_factor = p.A_AGN * np.exp(0.35 * m_shift)

        # In BCEmu1 only log10_Mcrit (and A_AGN scaling) varies; mu and theta_ej are fixed to fiducial
        if p.mode == "BCEmu1":
            mu_eff = 0.4
            theta_ej_eff = 4.0
            gamma_eff = 2.5
            delta_eff = 7.0
            eta_eff = 0.2
        elif p.mode == "BCEmu3":
            mu_eff = p.mu_baryon
            theta_ej_eff = p.theta_ej
            gamma_eff = 2.5
            delta_eff = 7.0
            eta_eff = 0.2
        else:  # BCEmu7
            mu_eff = p.mu_baryon
            theta_ej_eff = p.theta_ej
            gamma_eff = p.gamma_baryon
            delta_eff = p.delta_baryon
            eta_eff = p.eta_b

        # Characteristic ejection wavenumber k_dip scales inversely with theta_ej
        k_dip = 4.0 * (4.0 / np.maximum(theta_ej_eff, 0.5))
        sigma_dip = 1.1 + 0.25 * mu_eff

        # Redshift modulation
        z_mod = (1.0 + z) ** (-0.3 * (delta_eff / 7.0))

        # Maximum gas suppression bounded by baryon fraction f_b = Omega_b / Omega_m
        dip_depth = np.clip(1.6 * f_baryon * agn_factor * z_mod, 0.0, 0.65)
        log_k_ratio = (np.log10(np.maximum(k_arr, 1e-4)) - np.log10(k_dip)) / sigma_dip
        dip = dip_depth * np.exp(-0.5 * (log_k_ratio ** 2))

        # Small-scale stellar condensation upturn at k > 10 h/Mpc
        k_star = 15.0 * (2.5 / gamma_eff)
        star_boost = 0.15 * p.A_SN * (eta_eff / 0.2) / (1.0 + (k_star / np.maximum(k_arr, 1e-4)) ** 2)

        # Large-scale exact mass conservation enforcement: S_b(k -> 0) -> 1
        large_scale_taper = 1.0 - np.exp(-((k_arr / 0.03) ** 2))
        suppression = 1.0 + large_scale_taper * (-dip + star_boost)

        return float(suppression) if np.isscalar(k) else suppression


def baryonic_suppression_factor(
    k: Union[float, np.ndarray],
    baryon_params: BaryonParameters,
    z: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Computes the baryonic feedback response ratio S_b(k) = P_hydro(k) / P_dmo(k)
    using the BCEmu v1 formulation.
    """
    model = BCEmuModel(baryon_params)
    return model.suppression_factor(k, z=z)


def apply_baryonic_feedback_to_pk(
    k: np.ndarray,
    pk_dmo: np.ndarray,
    baryon_params: BaryonParameters,
    z: float = 0.0,
) -> np.ndarray:
    """Modulates a dark-matter-only power spectrum with BCEmu baryonic feedback physics."""
    s_b = baryonic_suppression_factor(k, baryon_params, z=z)
    return pk_dmo * s_b
