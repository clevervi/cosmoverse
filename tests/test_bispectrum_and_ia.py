"""
Unit tests for Result 1 Reduced Bispectrum Q(k1, k2, theta) (`ReducedBispectrumModel`)
and Level 2 NLA/TATT Weak Lensing Intrinsic Alignments (`IntrinsicAlignmentModel`).
"""

from __future__ import annotations

import numpy as np

from cosmoverse.emulators.bispectrum import ReducedBispectrumModel
from cosmoverse.observations.intrinsic_alignments import (
    IntrinsicAlignmentModel,
    IntrinsicAlignmentParameters,
)


def test_reduced_bispectrum_and_tunes_accuracy() -> None:
    model = ReducedBispectrumModel()
    bisp = model.compute_angular_bispectrum(k1=0.30, k2=0.60, z=0.0, include_baryons=True)
    assert len(bisp["theta_rad"]) == 24
    assert len(bisp["Q_hydro"]) == 24
    assert np.all(bisp["B_hydro"] < bisp["B_dmo"])
    # Collinear / folded triangles (theta -> 0 or pi) have larger Q than orthogonal (theta ~ pi/2)
    q_arr = np.asarray(bisp["Q_hydro"])
    assert float(q_arr[0]) > float(q_arr[12])

    acc = model.evaluate_tunes_bispectrum_accuracy()
    assert acc["passes_bispectrum_1pct_gate"] is True
    assert float(acc["q_mape_pct"]) < 1.0


def test_nla_tatt_intrinsic_alignments() -> None:
    ia = IntrinsicAlignmentModel(ia_params=IntrinsicAlignmentParameters(a_ia=0.85, b_ta=0.45))
    f_nla = ia.f_nla_amplitude(z=0.62)
    assert f_nla < 0.0  # GI anti-correlation

    ang = ia.compute_angular_ia_contamination()
    assert len(ang["ell"]) == 24
    assert np.all(ang["c_ell_gi"] < 0.0)
    assert np.all(ang["c_ell_ii"] > 0.0)

    eval_res = ia.evaluate_ia_marginalization_impact()
    assert eval_res["passes_stage4_ia_gate"] is True
    assert float(eval_res["marginalized_s8_bias_sigma"]) < 0.30
