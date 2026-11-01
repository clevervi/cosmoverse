"""
Example 02: Matter power spectrum with baryonic feedback and mock DESI observations.
"""

import numpy as np

from cosmoverse.baryons import apply_baryonic_feedback_to_pk
from cosmoverse.core import BaryonParameters, CosmologicalParameters, Cosmology
from cosmoverse.observations import SurveySpecification, inject_observational_noise


def main():
    print("=" * 60)
    print("COSMOVERSE: Example 02 - Linear, Baryonic, and Observed P(k)")
    print("=" * 60)

    cosmo = Cosmology(CosmologicalParameters.planck2018())
    baryons = BaryonParameters(A_AGN=1.5, A_SN=1.0)
    survey = SurveySpecification.desi_lrg()

    k = np.geomspace(0.01, 5.0, 15)

    # 1. Linear theory P(k)
    pk_lin = np.array([cosmo.linear_matter_power_spectrum(kv, z=0.5) for kv in k])

    # 2. Baryonic feedback modulation
    pk_hydro = apply_baryonic_feedback_to_pk(k, pk_lin, baryons, z=0.5)

    # 3. Observational noise injection (DESI volume)
    pk_obs, sigma_pk = inject_observational_noise(
        k, pk_hydro, survey, survey_volume_mpc3=5e8, seed=123
    )

    print(f"{'k [h/Mpc]':>10} | {'P_lin':>12} | {'P_hydro':>12} | {'P_obs':>12} | {'Sigma':>10}")
    print("-" * 65)
    for i in range(len(k)):
        print(f"{k[i]:10.4f} | {pk_lin[i]:12.2f} | {pk_hydro[i]:12.2f} | {pk_obs[i]:12.2f} | {sigma_pk[i]:10.2f}")


if __name__ == "__main__":
    main()
