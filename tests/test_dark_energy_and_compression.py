"""
Unit tests for Level 1 Dynamic w0-wa CPL Dark Energy (`DynamicDarkEnergyCosmology`)
and Level 3 Fisher-Optimal Score / IMNN Summary Compression (`FisherScoreCompressor`).
"""

from __future__ import annotations

import numpy as np

from cosmoverse.core.dark_energy import (
    CPLDarkEnergyParameters,
    DynamicDarkEnergyCosmology,
)
from cosmoverse.inference.compression import FisherScoreCompressor


def test_dynamic_w0wa_dark_energy_and_ap() -> None:
    lcdm = DynamicDarkEnergyCosmology(de_params=CPLDarkEnergyParameters.lambda_cdm())
    ap_lcdm = lcdm.alcock_paczynski_parameters(z=0.85)
    assert np.isclose(ap_lcdm["alpha_parallel"], 1.0, atol=1e-4)
    assert np.isclose(ap_lcdm["alpha_perp"], 1.0, atol=1e-4)

    cpl = DynamicDarkEnergyCosmology(de_params=CPLDarkEnergyParameters(w0=-0.82, wa=-0.75))
    w_z0 = float(cpl.equation_of_state_w(0.0))
    assert np.isclose(w_z0, -0.82)
    sig = cpl.evaluate_desi_w0wa_signatures()
    assert sig["recovers_lcdm_limit"] is True
    assert float(sig["mean_fsigma8_shift_pct"]) > 0.1


def test_fisher_score_compressor() -> None:
    bench = FisherScoreCompressor.benchmark_multiprobe_compression(n_data=96, n_params=6, seed=42)
    assert bench["compression_ratio"] == 16.0
    assert bench["passes_98pct_fisher_preservation_gate"] is True
    assert float(bench["fisher_info_preserved_pct"]) >= 98.0

    theta_fid = np.array([0.30, 0.80])
    mu_fid = np.array([10.0, 20.0, 30.0, 40.0])
    jac = np.array([[1.0, 0.2], [2.0, -0.5], [0.5, 1.5], [-1.0, 2.0]])
    cov = np.eye(4) * 0.25
    comp = FisherScoreCompressor(theta_fid, mu_fid, jac, cov)
    t_fid = comp.compress(mu_fid)
    assert np.allclose(t_fid, theta_fid, atol=1e-6)
