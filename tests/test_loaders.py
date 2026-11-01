"""
Unit tests for leakage-guarded 3D field snapshot loaders (`cosmoverse.datasets.loaders`).
"""

import pathlib

import numpy as np
import pytest

from cosmoverse.datasets.camels_gen2 import CAMELSGen2Catalog
from cosmoverse.datasets.loaders import (
    CAMELSFieldSnapshotDataset,
    build_camels_gen2_lh_dataloaders,
    build_leakage_guarded_dataloaders,
)


def test_build_camels_gen2_lh_dataloaders_and_batch_iteration():
    train_ds, test_ds, ver = build_camels_gen2_lh_dataloaders(
        n_subset=16, train_fraction=0.75, grid_size=16, seed=42
    )
    assert ver["is_leakage_free"] is True
    assert len(train_ds) > 0
    assert len(test_ds) > 0

    batch = next(train_ds.iter_batches(batch_size=3))
    assert batch["delta_ini"].shape == (3, 16, 16, 16)
    assert batch["delta_target"].shape == (3, 16, 16, 16)
    assert batch["theta"].shape == (3, 4)


def test_leakage_guard_blocks_shared_ic_or_cosmology_at_dataloader_boundary():
    catalog = CAMELSGen2Catalog()
    cv_records = catalog.get_subset("CV")  # All share cosmo_id=0
    with pytest.raises(ValueError, match="Data leakage blocked"):
        build_leakage_guarded_dataloaders(cv_records[:10], cv_records[10:20], grid_size=16)

    one_p_records = catalog.get_subset("1P")  # All share ic_seed=40000
    with pytest.raises(ValueError, match="Data leakage blocked"):
        build_leakage_guarded_dataloaders(one_p_records[:10], one_p_records[10:20], grid_size=16)


def test_npz_snapshot_persistence_and_reload(tmp_path: pathlib.Path):
    catalog = CAMELSGen2Catalog()
    lh_small = catalog.get_subset("LH")[:2]
    ds = CAMELSFieldSnapshotDataset(lh_small, grid_size=16)
    saved_files = ds.save_snapshots_to_npz(tmp_path)
    assert len(saved_files) == 2
    assert all(p.exists() for p in saved_files)

    ds_disk = CAMELSFieldSnapshotDataset(lh_small, grid_size=16, snapshot_dir=tmp_path)
    item_mem = ds.load_item(0)
    item_disk = ds_disk.load_item(0)
    assert np.allclose(item_mem["delta_target"], item_disk["delta_target"])
    assert np.allclose(item_mem["theta"], item_disk["theta"])
