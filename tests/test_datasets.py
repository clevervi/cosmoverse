"""
Unit tests for CAMELS Generation 2 (IllustrisTNG 1,192-simulation suite) catalog
and Stage-IV empirical precision calibration (Issue #9).
"""

import pytest

from cosmoverse.baryons.frontier import BiasVarianceFrontier
from cosmoverse.datasets.camels_gen2 import (
    CAMELSGen2Catalog,
    StageIVPrecisionCalibrator,
)
from cosmoverse.validation.transfer import DatasetPartitionProtocol


def test_camels_gen2_catalog_exact_1192_counts():
    catalog = CAMELSGen2Catalog()
    counts = catalog.summary_counts()
    assert counts["LH"] == 1000
    assert counts["CV"] == 27
    assert counts["1P"] == 161
    assert counts["EX"] == 4
    assert counts["total"] == 1192
    assert catalog.box_size_mpc_h == 50.0


def test_camels_gen2_lh_disjoint_partition_zero_leakage():
    catalog = CAMELSGen2Catalog()
    split = catalog.create_disjoint_lh_partition(train_fraction=0.80, seed=42)
    assert split["verification"]["is_leakage_free"] is True
    assert split["verification"]["shared_ic_seeds"] == 0
    assert split["verification"]["shared_cosmo_ids"] == 0
    assert len(split["train"]) > 0
    assert len(split["test"]) > 0


def test_camels_gen2_detects_cv_and_1p_leakage_if_split_naively():
    catalog = CAMELSGen2Catalog()

    # CV simulations share fiducial cosmo_id='0' -> splitting CV across train/test causes leakage
    cv_dicts = [r.to_partition_dict() for r in catalog.get_subset("CV")]
    cv_check = DatasetPartitionProtocol.verify_zero_leakage(cv_dicts[:15], cv_dicts[15:])
    assert cv_check["is_leakage_free"] is False
    assert cv_check["shared_cosmo_ids"] == 1

    # 1P simulations share ic_seed=40000 -> splitting 1P across train/test causes leakage
    one_p_dicts = [r.to_partition_dict() for r in catalog.get_subset("1P")]
    one_p_check = DatasetPartitionProtocol.verify_zero_leakage(
        one_p_dicts[:80], one_p_dicts[80:]
    )
    assert one_p_check["is_leakage_free"] is False
    assert one_p_check["shared_ic_seeds"] == 1


def test_stage_iv_precision_calibrator_and_frontier_integration():
    calibrator = StageIVPrecisionCalibrator()
    euclid = calibrator.calibrate_survey_precision("Euclid-Wide")
    assert euclid["n_cv_simulations"] == 27
    assert euclid["box_size_mpc_h"] == 50.0
    assert 0.009 < euclid["sigma_stage4_S8"] < 0.015
    assert euclid["max_allowed_bias_S8"] == pytest.approx(
        0.3 * euclid["sigma_stage4_S8"]
    )

    frontier = BiasVarianceFrontier(stage4_sigma_s8=euclid["sigma_stage4_S8"])
    report = frontier.compute_frontier_curve()
    assert report["has_stage4_compatible_regime"] is True
