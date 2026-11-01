"""
Standard benchmark suites and dataset generators for cosmological emulators.
Supports CAMELS (including 2nd generation 1,192 IllustrisTNG sims, 35 parameters, 50 Mpc/h box)
and Quijote-style consistency checks with explicit (ic_seed, cosmo_id, family_id) metadata.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def generate_latin_hypercube_samples(
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    seed: int = 42,
) -> List[Dict[str, float]]:
    """
    Generates a Latin Hypercube Design (LHD) across parameter space.
    Ensures space-filling properties without clustering or gaps.
    """
    rng = np.random.default_rng(seed)
    param_names = list(bounds.keys())
    d = len(param_names)

    lhs = np.zeros((n_samples, d))
    for j in range(d):
        intervals = np.linspace(0, 1, n_samples + 1)
        sub_samples = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(sub_samples)
        lhs[:, j] = sub_samples

    samples = []
    for i in range(n_samples):
        item = {}
        for j, name in enumerate(param_names):
            low, high = bounds[name]
            item[name] = float(low + lhs[i, j] * (high - low))
        samples.append(item)

    return samples


def camels_latin_hypercube_benchmark(n_samples: int = 25, seed: int = 42) -> List[Dict[str, float]]:
    """
    Standard 5-parameter CAMELS parameter space bounds:
    Omega_m, sigma8, A_SN, A_AGN, ns
    """
    bounds = {
        "omega_m": (0.1, 0.5),
        "sigma8": (0.6, 1.0),
        "A_SN": (0.25, 4.0),
        "A_AGN": (0.25, 4.0),
        "ns": (0.9, 1.05),
    }
    return generate_latin_hypercube_samples(n_samples, bounds, seed=seed)


def camels_gen2_catalog_metadata(
    n_records_per_family: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates synthetic metadata records matching CAMELS 2nd Generation specification
    (1,192 IllustrisTNG simulations in (50 Mpc/h)^3 boxes across 35 parameters)
    paired with N-body DMO counterparts for strict (ic_seed, cosmo_id, family_id)
    partitioning andResultado 4 Cross-Simulation Transfer benchmarks.
    """
    lhs = camels_latin_hypercube_benchmark(n_samples=n_records_per_family, seed=seed)
    records: List[Dict[str, Any]] = []

    for idx, params in enumerate(lhs):
        cosmo_id = f"cosmo_{idx:04d}"
        ic_seed = 10000 + idx
        for family in ("CAMELS-DMO", "CAMELS-IllustrisTNG"):
            rec: Dict[str, Any] = {
                "sim_id": f"{family}_{idx:04d}",
                "family_id": family,
                "cosmo_id": cosmo_id,
                "ic_seed": ic_seed,
                "box_size_mpc_h": 50.0,
                "n_total_camels_gen2_params": 35,
                **params,
            }
            records.append(rec)

    return records
