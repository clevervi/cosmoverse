"""
Unit tests for MassiveNeutrinoPhysics and HaloMassFunctionModel (`cosmoverse.core.halos_and_neutrinos`).
"""

import numpy as np

from cosmoverse.core.halos_and_neutrinos import (
    HaloMassFunctionModel,
    MassiveNeutrinoPhysics,
)
from cosmoverse.core.parameters import BaryonParameters, CosmologicalParameters


def test_massive_neutrino_free_streaming_suppression():
    k = np.geomspace(1e-3, 2.0, 16)
    massless = CosmologicalParameters(omega_m=0.315, h=0.674, m_nu=0.0)
    assert MassiveNeutrinoPhysics.omega_nu(massless) == 0.0
    s_zero = MassiveNeutrinoPhysics.suppression_factor_S_nu(k, massless)
    assert np.allclose(s_zero, 1.0)

    massive = CosmologicalParameters(omega_m=0.315, h=0.674, m_nu=0.15)
    f_nu = MassiveNeutrinoPhysics.neutrino_fraction_f_nu(massive)
    assert f_nu > 0.0
    s_nu = MassiveNeutrinoPhysics.suppression_factor_S_nu(k, massive, z=0.0)
    # Large scales k -> 0 are unsuppressed; small scales k >> k_fs are suppressed by ~ 8 f_nu
    assert s_nu[0] > 0.99
    assert s_nu[-1] < 1.0 - 6.0 * f_nu


def test_halo_mass_function_and_emulation_gate():
    cosmo_p = CosmologicalParameters(omega_m=0.315, sigma8=0.811, m_nu=0.06)
    baryon_p = BaryonParameters(mode="BCEmu3", A_AGN=1.2)
    hmf = HaloMassFunctionModel(cosmo_params=cosmo_p, baryon_params=baryon_p)

    masses = np.geomspace(1e11, 3e14, 10)
    res = hmf.compute_hmf_dn_dlnm(masses, z=0.0, include_baryonic_depletion=True)
    assert np.all(res["dn_dlnm_hydro"] < res["dn_dlnm_dmo"])
    assert np.all(np.diff(res["dn_dlnm_dmo"]) < 0.0)

    eval_res = hmf.evaluate_emulated_hmf_accuracy()
    assert eval_res["passes_hmf_1pct_gate"] is True
    assert eval_res["hmf_mape_pct"] < 1.0
