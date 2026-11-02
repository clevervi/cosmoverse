"""
Unit test for leakage-free cross-simulation transfer with the trained flow
(Result 4): source DMO + BCEmu -> target IllustrisTNG-like response.
"""

from __future__ import annotations

from cosmoverse.inference import FlowTransferPipeline


def test_flow_transfer_physical_mismatch_gate() -> None:
    pipeline = FlowTransferPipeline(
        mismatch_strength=1.0,
        n_records=600,
        n_test=100,
        seed=42,
    )
    result = pipeline.run()

    assert result["leakage_check"]["is_leakage_free"] is True
    assert result["leakage_check"]["shared_ic_seeds"] == 0
    assert result["leakage_check"]["shared_cosmo_ids"] == 0
    assert result["source_calibration"]["passes_joint_sbi_gate"] is True
    assert result["post_transfer"]["coverage_95"] >= 0.70
    assert result["degradation"]["delta_tarp_ice"] <= 0.20
    assert result["passes_transfer_gate"] is True
