"""
Unit tests for Level 2 CMB Lensing cross-spectra and thermal/kinetic Sunyaev-Zel'dovich
gas calibration of the BCEmu baryonic feedback parameters (`CMBLensingAndSZPipeline`).
"""

from __future__ import annotations

import numpy as np

from cosmoverse.core.parameters import BaryonParameters
from cosmoverse.observations import CMBLensingAndSZPipeline


def test_cmb_lensing_kernel_and_cross_spectra() -> None:
    pipeline = CMBLensingAndSZPipeline()

    z = np.array([0.0, 0.5, 1.0, 2.0])
    w_kappa = pipeline.cmb_lensing_kernel_w_kappa(z)
    assert w_kappa.shape == z.shape
    assert np.all(np.isfinite(w_kappa))
    assert np.all(w_kappa >= 0.0)
    assert float(w_kappa[0]) == 0.0

    spec = pipeline.compute_cmb_lensing_and_tsz_spectra()
    assert spec["ell"].shape == (24,)
    for key in (
        "c_ell_kcmb_kcmb",
        "c_ell_kcmb_g",
        "c_ell_kcmb_kgal",
        "d_ell_yy_1e12",
        "c_ell_gy",
    ):
        values = np.asarray(spec[key], dtype=np.float64)
        assert values.shape == spec["ell"].shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
    assert 0.0 < float(spec["bound_gas_fraction_proxy"]) <= 1.0

    ell_custom = np.array([100.0, 500.0, 1000.0])
    spec_custom = pipeline.compute_cmb_lensing_and_tsz_spectra(ell=ell_custom)
    assert np.allclose(spec_custom["ell"], ell_custom)


def test_agn_feedback_suppresses_cmb_lensing_and_tsz_power() -> None:
    low_agn = CMBLensingAndSZPipeline(
        baryon_params=BaryonParameters(mode="BCEmu3", A_AGN=0.5)
    ).compute_cmb_lensing_and_tsz_spectra()
    high_agn = CMBLensingAndSZPipeline(
        baryon_params=BaryonParameters(mode="BCEmu3", A_AGN=2.5)
    ).compute_cmb_lensing_and_tsz_spectra()

    assert float(np.mean(low_agn["c_ell_kcmb_g"])) > float(np.mean(high_agn["c_ell_kcmb_g"]))
    assert float(np.mean(low_agn["c_ell_kcmb_kgal"])) > float(np.mean(high_agn["c_ell_kcmb_kgal"]))
    assert float(np.mean(low_agn["d_ell_yy_1e12"])) > float(np.mean(high_agn["d_ell_yy_1e12"]))


def test_ksz_profile_and_baryon_calibration_gate() -> None:
    pipeline = CMBLensingAndSZPipeline()
    ksz = pipeline.compute_ksz_optical_depth_profile()
    tau = np.asarray(ksz["tau_ksz"], dtype=np.float64)

    assert tau.shape == ksz["halo_mass_msun_h"].shape == (16,)
    assert np.all(np.isfinite(tau))
    assert np.all(tau > 0.0)
    assert np.all(np.diff(tau) > 0.0)

    low_agn_tau = np.mean(
        CMBLensingAndSZPipeline(
            baryon_params=BaryonParameters(mode="BCEmu3", A_AGN=0.5)
        ).compute_ksz_optical_depth_profile()["tau_ksz"]
    )
    high_agn_tau = np.mean(
        CMBLensingAndSZPipeline(
            baryon_params=BaryonParameters(mode="BCEmu3", A_AGN=2.5)
        ).compute_ksz_optical_depth_profile()["tau_ksz"]
    )
    assert float(low_agn_tau) > float(high_agn_tau)

    gain = pipeline.evaluate_sz_baryon_calibration_gain()
    assert gain["passes_cmb_sz_calibration_gate"] is True
    assert 1.20 < float(gain["tsz_low_to_high_agn_power_ratio"]) < 3.0
    assert 0.0 < float(gain["mean_ksz_optical_depth"]) < 0.10
    assert float(gain["s8_inflation_with_cmb_sz"]) < float(gain["s8_inflation_3x2pt_rsd_only"])
    assert float(gain["residual_s8_bias_sigma"]) < 0.30
