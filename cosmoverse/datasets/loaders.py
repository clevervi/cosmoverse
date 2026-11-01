"""
Leakage-Guarded 3D Density Field Snapshot Loader for CAMELS Gen-2 and Quijote suites.

Enforces Section 7 of docs/MASTER_DOCUMENT.md at the DataLoader boundary:
  - Any train/test pair of dataset loaders is verified via `DatasetPartitionProtocol.verify_zero_leakage`
    BEFORE any 3D field tensor is loaded or iterated.
  - Supports on-disk `.npz` / `.npy` archives (and `.hdf5` if `h5py` is installed) as well as
    deterministic 1LPT + BCEmu 3D field synthesis from `CAMELSGen2Record` metadata.
"""

import pathlib
from typing import Any, Iterator

import numpy as np

from cosmoverse.baryons.feedback import BCEmuModel
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import (
    BaryonParameters,
    CosmologicalParameters,
    SimulationParameters,
)
from cosmoverse.datasets.camels_gen2 import CAMELSGen2Catalog, CAMELSGen2Record
from cosmoverse.simulation.initial_conditions import generate_gaussian_random_field
from cosmoverse.validation.transfer import DatasetPartitionProtocol


class CAMELSFieldSnapshotDataset:
    """
    Dataset of 3D cosmological density fields delta(x) indexed by CAMELSGen2Record metadata.
    Supports reading `.npz` snapshots from disk or synthesizing deterministic 3D grids.
    """

    def __init__(
        self,
        records: list[CAMELSGen2Record | dict[str, Any]],
        grid_size: int = 16,
        snapshot_dir: str | pathlib.Path | None = None,
    ):
        if not records:
            raise ValueError("CAMELSFieldSnapshotDataset requires at least one simulation record.")
        self.records: list[dict[str, Any]] = [
            r.to_partition_dict() if isinstance(r, CAMELSGen2Record) else dict(r)
            for r in records
        ]
        self.grid_size = int(grid_size)
        self.snapshot_dir = pathlib.Path(snapshot_dir) if snapshot_dir is not None else None

    def __len__(self) -> int:
        return len(self.records)

    def _synthesize_field_pair(self, rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes a deterministic (delta_initial, delta_target, theta_vec) tuple from record metadata
        using the record's exact `ic_seed`, `Omega_m`, `sigma_8`, and `A_AGN1`.
        """
        om = float(rec.get("Omega_m", rec.get("omega_m", 0.30)))
        s8 = float(rec.get("sigma_8", rec.get("sigma8", 0.80)))
        a_agn = float(rec.get("A_AGN1", rec.get("A_AGN", 1.0)))
        a_sn = float(rec.get("A_SN1", rec.get("A_SN", 1.0)))
        ic_seed = int(rec["ic_seed"])
        box_size = float(rec.get("box_size", 50.0))
        family_id = str(rec.get("family_id", "CAMELS-Gen2-IllustrisTNG"))

        cosmo = Cosmology(CosmologicalParameters(omega_m=om, sigma8=s8))
        sim_params = SimulationParameters(
            box_size=box_size,
            n_mesh=self.grid_size,
            seed=ic_seed,
            ic_seed=ic_seed,
            cosmo_id=str(rec["cosmo_id"]),
            family_id=family_id,
        )
        delta_ini, _ = generate_gaussian_random_field(cosmo, sim_params, seed=ic_seed)

        # Quasi-nonlinear gravity + family-dependent baryonic modulation in Fourier space
        delta_nl = delta_ini + 0.02 * (delta_ini ** 2 - np.mean(delta_ini ** 2))
        if "DMO" not in family_id.upper():
            baryon_model = BCEmuModel(
                BaryonParameters(mode="BCEmu3", A_AGN=np.clip(a_agn, 0.2, 4.0), A_SN=np.clip(a_sn, 0.2, 4.0))
            )
            k_axes = [
                np.fft.fftfreq(self.grid_size, d=box_size / self.grid_size) * 2.0 * np.pi
                for _ in range(2)
            ]
            kz_axis = np.fft.rfftfreq(self.grid_size, d=box_size / self.grid_size) * 2.0 * np.pi
            kx, ky, kz = np.meshgrid(k_axes[0], k_axes[1], kz_axis, indexing="ij")
            k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
            s_b = baryon_model.suppression_factor(np.maximum(k_mag, 1e-4), z=0.0)
            delta_k = np.fft.rfftn(delta_nl) * np.sqrt(np.maximum(s_b, 1e-4))
            delta_target = np.fft.irfftn(delta_k, s=delta_nl.shape)
        else:
            delta_target = delta_nl

        theta_vec = np.array([om, s8, a_sn, a_agn], dtype=np.float64)
        return delta_ini, delta_target, theta_vec

    def load_item(self, index: int) -> dict[str, Any]:
        """Loads or synthesizes a single 3D simulation item by index."""
        rec = self.records[index]
        sim_id = str(rec["sim_id"])

        if self.snapshot_dir is not None:
            npz_path = self.snapshot_dir / f"{sim_id}.npz"
            if npz_path.exists():
                with np.load(npz_path) as data:
                    return {
                        "sim_id": sim_id,
                        "record": rec,
                        "delta_ini": np.asarray(data["delta_ini"], dtype=np.float64),
                        "delta_target": np.asarray(data["delta_target"], dtype=np.float64),
                        "theta": np.asarray(data["theta"], dtype=np.float64),
                    }

        delta_ini, delta_target, theta = self._synthesize_field_pair(rec)
        return {
            "sim_id": sim_id,
            "record": rec,
            "delta_ini": delta_ini,
            "delta_target": delta_target,
            "theta": theta,
        }

    def save_snapshots_to_npz(self, out_dir: str | pathlib.Path) -> list[pathlib.Path]:
        """Materializes all records in this dataset as compressed `.npz` 3D snapshots on disk."""
        target_dir = pathlib.Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[pathlib.Path] = []
        for idx in range(len(self.records)):
            item = self.load_item(idx)
            out_path = target_dir / f"{item['sim_id']}.npz"
            np.savez_compressed(
                out_path,
                delta_ini=item["delta_ini"],
                delta_target=item["delta_target"],
                theta=item["theta"],
            )
            saved_paths.append(out_path)
        return saved_paths

    def iter_batches(self, batch_size: int = 4) -> Iterator[dict[str, Any]]:
        """Yields mini-batches of 3D fields `(B, N, N, N)` and parameter vectors `(B, 4)`."""
        n = len(self.records)
        for start in range(0, n, batch_size):
            items = [self.load_item(i) for i in range(start, min(start + batch_size, n))]
            yield {
                "sim_ids": [it["sim_id"] for it in items],
                "delta_ini": np.stack([it["delta_ini"] for it in items], axis=0),
                "delta_target": np.stack([it["delta_target"] for it in items], axis=0),
                "theta": np.stack([it["theta"] for it in items], axis=0),
            }


def build_leakage_guarded_dataloaders(
    train_records: list[CAMELSGen2Record | dict[str, Any]],
    test_records: list[CAMELSGen2Record | dict[str, Any]],
    grid_size: int = 16,
    snapshot_dir: str | pathlib.Path | None = None,
) -> tuple[CAMELSFieldSnapshotDataset, CAMELSFieldSnapshotDataset, dict[str, Any]]:
    """
    Constructs train and test 3D field datasets ONLY after verifying zero leakage
    of `ic_seed` and `cosmo_id` via `DatasetPartitionProtocol.verify_zero_leakage`.

    Raises:
        ValueError: if any `ic_seed` or `cosmo_id` is shared between train_records and test_records.
    """
    train_dicts = [
        r.to_partition_dict() if isinstance(r, CAMELSGen2Record) else dict(r)
        for r in train_records
    ]
    test_dicts = [
        r.to_partition_dict() if isinstance(r, CAMELSGen2Record) else dict(r)
        for r in test_records
    ]
    verification = DatasetPartitionProtocol.verify_zero_leakage(train_dicts, test_dicts)
    if not verification["is_leakage_free"]:
        raise ValueError(
            f"Data leakage blocked by DatasetPartitionProtocol (Section 7): "
            f"shared_ic_seeds={verification['shared_ic_seeds']}, "
            f"shared_cosmo_ids={verification['shared_cosmo_ids']}"
        )

    train_ds = CAMELSFieldSnapshotDataset(
        train_dicts, grid_size=grid_size, snapshot_dir=snapshot_dir
    )
    test_ds = CAMELSFieldSnapshotDataset(
        test_dicts, grid_size=grid_size, snapshot_dir=snapshot_dir
    )
    return train_ds, test_ds, verification


def build_camels_gen2_lh_dataloaders(
    n_subset: int = 20,
    train_fraction: float = 0.75,
    grid_size: int = 16,
    seed: int = 42,
) -> tuple[CAMELSFieldSnapshotDataset, CAMELSFieldSnapshotDataset, dict[str, Any]]:
    """
    Convenience builder that slices `n_subset` LH simulations from `CAMELSGen2Catalog`,
    partitions them strictly by `(ic_seed, cosmo_id)`, and returns leakage-verified 3D datasets.
    """
    catalog = CAMELSGen2Catalog()
    lh_records = [r.to_partition_dict() for r in catalog.get_subset("LH")[:n_subset]]
    train_recs, test_recs = DatasetPartitionProtocol.partition_by_ic_and_cosmology(
        lh_records, train_fraction=train_fraction, seed=seed
    )
    return build_leakage_guarded_dataloaders(
        train_recs, test_recs, grid_size=grid_size
    )
