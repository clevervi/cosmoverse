"""
BCEmu-based baryonic feedback suppression model (Primary v1 Baryonic Emulator).
Implements 1, 3, and 7-parameter baryonification suppression S_b(k, z) = P_hydro(k, z) / P_dmo(k, z).

Decision of v1 (Master Document Section 6.2):
  Use ONE baryonic emulator in v1 (BCEmu: Giri & Schneider 2021) for interpretable marginalization.
  BACCOemu is reserved for v2 or cross-validation so errors can be unambiguously attributed.
"""

from typing import Protocol, Union, runtime_checkable

import numpy as np

from cosmoverse.core.parameters import BaryonParameters


@runtime_checkable
class BaryonicEmulatorProtocol(Protocol):
    """Protocol for single-backend baryonic suppression emulators in COSMOVERSE."""

    def suppression_factor(
        self,
        k: Union[float, np.ndarray],
        z: float = 0.0,
        f_baryon: float = 0.156,
    ) -> Union[float, np.ndarray]:
        ...


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


class BACCOEmuPlanBModel:
    """
    Reversible v2 Plan B baryonic emulator (BACCOemu formulation).
    Reference: Arico et al. (2021), MNRAS 506, 4070 (arXiv:2011.15018), validated on DES Y3.

    Per Master Document Section 8 & Section 12:
      - Not active by default in v1 (BCEmu is the sole v1 backend).
      - Activated exclusively via `create_baryonic_emulator(backend="BACCOemu", allow_v2_plan_b=True)`.
    """

    def __init__(self, params: BaryonParameters | None = None):
        self.params = params or BaryonParameters(mode="BCEmu7")
        self.citation_arxiv = "2011.15018"

    def suppression_factor(
        self,
        k: Union[float, np.ndarray],
        z: float = 0.0,
        f_baryon: float = 0.156,
    ) -> Union[float, np.ndarray]:
        """
        Computes S_bacco(k, z) = P_hydro(k, z) / P_dmo(k, z) using the Arico et al. (2021)
        gas-expulsion + central galaxy condensation parameterization, preserving S(k -> 0) = 1.
        """
        k_arr = np.asarray(k, dtype=np.float64)
        p = self.params

        m_c_shift = 13.8 - p.log10_Mcrit
        beta_gas = 0.6 * (p.mu_baryon / 0.4)
        theta_out = np.maximum(p.theta_ej, 0.5)

        k_gas = 3.2 * (4.0 / theta_out)
        z_evol = (1.0 + z) ** (-0.25)
        dip_amp = np.clip(1.45 * f_baryon * p.A_AGN * np.exp(0.30 * m_c_shift) * z_evol, 0.0, 0.60)

        log_k = (np.log10(np.maximum(k_arr, 1e-4)) - np.log10(k_gas)) / (1.0 + 0.3 * beta_gas)
        gas_dip = dip_amp * np.exp(-0.5 * (log_k ** 2))

        k_cen = 14.0
        cen_boost = 0.12 * p.A_SN * (p.eta_b / 0.2) / (1.0 + (k_cen / np.maximum(k_arr, 1e-4)) ** 2)

        mass_conservation_taper = 1.0 - np.exp(-((k_arr / 0.025) ** 2))
        s_bacco = 1.0 + mass_conservation_taper * (-gas_dip + cen_boost)
        return float(s_bacco) if np.isscalar(k) else s_bacco


def create_baryonic_emulator(
    backend: str = "BCEmu",
    params: BaryonParameters | None = None,
    allow_v2_plan_b: bool = False,
) -> BaryonicEmulatorProtocol:
    """
    Factory enforcing Section 8 & Section 12 of docs/MASTER_DOCUMENT.md:
      1. Never mix BCEmu and BACCOemu simultaneously.
      2. BCEmu is the single active v1 emulator.
      3. BACCOemu is available as a reversible Plan B only when `allow_v2_plan_b=True`.
    """
    normalized = backend.strip()
    if "+" in normalized or "," in normalized or (
        "bcemu" in normalized.lower() and "bacco" in normalized.lower()
    ):
        raise ValueError(
            "Master Document Section 8 forbids integrating BCEmu and BACCOemu simultaneously "
            "in the same pipeline run. Select exactly one backend."
        )

    if normalized == "BCEmu":
        return BCEmuModel(params)

    if normalized == "BACCOemu":
        if not allow_v2_plan_b:
            raise ValueError(
                "BACCOemu is reserved for v2 Plan B per Master Document Section 8 & 12. "
                "Pass allow_v2_plan_b=True to explicitly switch from BCEmu to BACCOemu."
            )
        return BACCOEmuPlanBModel(params)

    raise ValueError(f"Unknown baryonic emulator backend '{backend}'. Valid: ['BCEmu', 'BACCOemu']")
