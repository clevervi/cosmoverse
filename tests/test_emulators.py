"""
Unit tests for neural emulators (TUNeS v1 primary, FNO Plan B) and Resultado 1 statistics
(Gaussian P(k) MAPE, 1-point PDF, 3D Peak Counts, 3D Minkowski Functionals).
"""

import numpy as np

from cosmoverse.emulators.fno import FNO3dEmulator
from cosmoverse.emulators.statistics import (
    compute_minkowski_functionals_3d,
    compute_one_point_pdf,
    compute_peak_counts_3d,
    compute_pk_mape,
    evaluate_emulator_result1,
)
from cosmoverse.emulators.training import (
    cross_correlation_coefficient,
    power_spectrum_loss,
)
from cosmoverse.emulators.unet import CosmoUNet, TUNeSEmulator


def test_tunes_emulator_and_result1():
    rng = np.random.default_rng(42)
    base_field = rng.normal(0, 1, size=(16, 16, 16))
    target_field = base_field + 0.01 * (base_field ** 2 - np.mean(base_field ** 2))

    tunes = TUNeSEmulator(in_channels=1, out_channels=1, base_features=8, depth=2, seed=42)
    tunes.calibrate_to_simulation_bank(base_field[np.newaxis, ...], target_field[np.newaxis, ...])
    pred = tunes.forward(base_field, scale_a=1.0)
    assert pred.shape == (16, 16, 16)

    res1 = evaluate_emulator_result1(pred, target_field, box_size=100.0, k_max=1.0)
    assert res1["pk_mape_pct"] < 1.0
    assert res1["passes_pk_threshold"] is True
    assert "V0" in res1["minkowski_mae"]
    assert "V3" in res1["minkowski_mae"]


def test_non_gaussian_statistics_3d():
    rng = np.random.default_rng(123)
    field = rng.normal(0, 1, size=(16, 16, 16))

    centers, pdf = compute_one_point_pdf(field, n_bins=15)
    assert len(centers) == 15
    assert np.all(pdf >= 0.0)

    nu_c, peaks = compute_peak_counts_3d(field, n_bins=8)
    assert len(nu_c) == 8
    assert np.sum(peaks) > 0.0

    mf = compute_minkowski_functionals_3d(field, thresholds=np.array([-1.0, 0.0, 1.0]))
    assert mf["V0"].shape == (3,)
    # V0(nu) is monotonically decreasing with threshold nu
    assert mf["V0"][0] >= mf["V0"][1] >= mf["V0"][2]


def test_fno_3d_emulator_plan_b():
    fno = FNO3dEmulator(in_channels=1, out_channels=1, width=8, modes=4)
    dummy_input = np.random.normal(0, 1, size=(16, 16, 16))
    pred = fno.predict(dummy_input)
    assert pred.shape == (16, 16, 16)


def test_cosmo_unet_alias():
    unet = CosmoUNet()
    dummy_input = np.random.normal(0, 1, size=(16, 16, 16))
    out = unet.forward(dummy_input)
    assert out.shape == (16, 16, 16)


def test_power_spectrum_loss_and_mape():
    pk1 = np.array([100.0, 50.0, 10.0])
    pk2 = np.array([100.5, 49.8, 9.95])
    loss = power_spectrum_loss(pk1, pk2)
    assert loss >= 0.0
    assert power_spectrum_loss(pk1, pk1) == 0.0
    assert compute_pk_mape(pk2, pk1) < 1.0


def test_cross_correlation_coefficient():
    field_a = np.random.normal(0, 1, size=(16, 16, 16))
    k, r_k = cross_correlation_coefficient(field_a, field_a, box_size=50.0, n_bins=5)
    assert len(k) == 5
    assert np.allclose(r_k, 1.0, atol=1e-5)
