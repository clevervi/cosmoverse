"""
Unit tests for the multi-family transfer audit (TNG, SIMBA, Astrid) and the
real CAMELS table loading hook.
"""

from __future__ import annotations

import pathlib

import numpy as np

from cosmoverse.inference.multi_family_transfer import (
    MultiFamilyTransferPipeline,
)


def test_multi_family_transfer_audit() -> None:
    pipeline = MultiFamilyTransferPipeline(
        n_records=600,
        n_test=100,
        seed=42,
    )
    result = pipeline.run()

    assert result["leakage_check"]["is_leakage_free"] is True
    assert result["leakage_check"]["shared_ic_seeds"] == 0
    assert result["leakage_check"]["shared_cosmo_ids"] == 0
    assert result["source_calibration"]["passes_joint_sbi_gate"] is True
    assert set(result["families"]) == {
        "CAMELS-IllustrisTNG",
        "CAMELS-SIMBA",
        "CAMELS-Astrid",
    }
    assert result["passes_multi_family_gate"] is True
    assert result["passes_degradation_bound_all"] is False
    assert any(
        not family_result["passes_family_gate"]
        for family_result in result["families"].values()
    )


def test_real_table_loading_hook(tmp_path: pathlib.Path) -> None:
    k = np.geomspace(0.03, 2.0, 24)
    pk = 1000.0 * (k / 0.1) ** (-1.5)
    table_path = tmp_path / "CAMELS-IllustrisTNG-LH_0_Pk.txt"
    np.savetxt(table_path, np.column_stack([k, pk]))

    pipeline = MultiFamilyTransferPipeline(
        data_dir=str(tmp_path),
        n_records=20,
        n_test=5,
        seed=42,
    )
    theta = np.array([0.315, 0.811, 1.65, 1.0])
    response_fn, source = pipeline._response_function(
        "CAMELS-IllustrisTNG", theta, 0
    )

    assert source == "disk_table"
    response = response_fn(np.array([0.05, 0.2, 1.0]))
    assert response.shape == (3,)
    assert np.all(np.isfinite(response))
    assert np.all(response > 0.0)
