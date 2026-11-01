"""
Unit tests for BCEmu baryonic feedback (v1 primary), baryonification, and Resultado 2 Bias-Variance Frontier.
"""

import numpy as np
import pytest

from cosmoverse.baryons.baryonification import BaryonificationModel
from cosmoverse.baryons.feedback import (
    BACCOEmuPlanBModel,
    BaryonicEmulatorProtocol,
    BCEmuModel,
    create_baryonic_emulator,
)
from cosmoverse.baryons.frontier import BiasVarianceFrontier
from cosmoverse.core.parameters import BaryonParameters


def test_bcemu_modes_and_suppression():
    for mode in ("BCEmu1", "BCEmu3", "BCEmu7"):
        params = BaryonParameters(mode=mode, log10_Mcrit=13.3, A_AGN=1.5)
        model = BCEmuModel(params)

        # Large scale limit (k -> 0) preserves mass conservation: S_b(k) -> 1.0
        s_large = model.suppression_factor(1e-4, z=0.0)
        assert np.isclose(s_large, 1.0, atol=0.01)

        # Intermediate scale (k ~ 3-5 h/Mpc) experiences AGN gas expulsion dip (< 1.0)
        s_intermediate = model.suppression_factor(4.0, z=0.0)
        assert s_intermediate < 1.0


def test_bias_variance_frontier_result2():
    frontier = BiasVarianceFrontier()
    res = frontier.compute_frontier_curve(
        k_max_values=[0.5, 1.0],
        marginalization_weights=[0.0, 0.5, 1.0],
    )
    assert "frontier_curve" in res
    assert len(res["frontier_curve"]) == 6
    assert res["has_stage4_compatible_regime"] is True

    # Verify that marginalization (w=1.0) reduces bias relative to unmarginalized (w=0.0)
    # while increasing error inflation factor
    pt_unmarg = [p for p in res["frontier_curve"] if p["k_max"] == 0.5 and p["marginalization_weight"] == 0.0][0]
    pt_marg = [p for p in res["frontier_curve"] if p["k_max"] == 0.5 and p["marginalization_weight"] == 1.0][0]
    assert pt_marg["bias_in_stage4_sigma"] < pt_unmarg["bias_in_stage4_sigma"]
    assert pt_marg["error_inflation_factor"] > pt_unmarg["error_inflation_factor"]


def test_baryonification_grid_displacement():
    model = BaryonificationModel(BaryonParameters(A_AGN=1.0))
    grid = np.random.normal(0, 1, size=(16, 16, 16))
    filtered_grid = model.displace_density_field(grid, box_size=50.0)
    assert filtered_grid.shape == (16, 16, 16)


def test_baryonic_emulator_protocol_and_reversible_bacco_plan_b():
    # Default v1 creates BCEmuModel
    emu_v1 = create_baryonic_emulator("BCEmu")
    assert isinstance(emu_v1, BCEmuModel)
    assert isinstance(emu_v1, BaryonicEmulatorProtocol)

    # Mixing BCEmu and BACCOemu simultaneously is strictly forbidden
    with pytest.raises(ValueError, match="forbids integrating BCEmu and BACCOemu"):
        create_baryonic_emulator("BCEmu+BACCOemu")

    # BACCOemu blocked in default v1 unless allow_v2_plan_b=True
    with pytest.raises(ValueError, match="reserved for v2 Plan B"):
        create_baryonic_emulator("BACCOemu", allow_v2_plan_b=False)

    # Reversible Plan B switch works when explicitly enabled
    bacco = create_baryonic_emulator("BACCOemu", allow_v2_plan_b=True)
    assert isinstance(bacco, BACCOEmuPlanBModel)
    assert np.isclose(bacco.suppression_factor(1e-4, z=0.0), 1.0, atol=0.01)
    assert bacco.suppression_factor(4.0, z=0.0) < 1.0
