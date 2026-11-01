"""
Experiment: Generalization and transferability across simulation suites (e.g. CAMELS).
Evaluates emulator predictive power under varying baryonic feedback parameters.
"""

import argparse

import numpy as np

from cosmoverse.baryons import apply_baryonic_feedback_to_pk
from cosmoverse.core import BaryonParameters, CosmologicalParameters, Cosmology
from cosmoverse.validation.benchmarks import camels_latin_hypercube_benchmark


def run_experiment(n_points: int = 5) -> None:
    print(f"=== Running CAMELS Cross-Simulation Transfer Experiment ({n_points} designs) ===")
    lhs_points = camels_latin_hypercube_benchmark(n_samples=n_points, seed=42)

    for i, pt in enumerate(lhs_points):
        cosmo = Cosmology(CosmologicalParameters(omega_m=pt["omega_m"], sigma8=pt["sigma8"], ns=pt["ns"]))
        baryon = BaryonParameters(A_AGN=pt["A_AGN"], A_SN=pt["A_SN"])

        k = np.geomspace(0.01, 10.0, 50)
        pk_dmo = np.array([cosmo.linear_matter_power_spectrum(kv) for kv in k])
        pk_hydro = apply_baryonic_feedback_to_pk(k, pk_dmo, baryon)

        ratio = pk_hydro / pk_dmo
        min_suppression = float(np.min(ratio))
        print(f"[{i+1}/{n_points}] Omega_m={pt['omega_m']:.3f}, A_AGN={pt['A_AGN']:.2f} -> Max Suppression: {(1.0 - min_suppression)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=5)
    args = parser.parse_args()
    run_experiment(args.points)
