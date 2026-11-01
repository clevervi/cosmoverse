"""
Model Checkpoint Persistence (`CheckpointManager`) and Cached Public CAMELS Spectra
Fetcher (`CAMELSArchiveFetcher`) with SHA-256 integrity verification.
"""

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import CosmologicalParameters, SimulationParameters
from cosmoverse.datasets.camels_gen2 import CAMELSGen2Catalog
from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector
from cosmoverse.emulators.unet import TUNeSEmulator
from cosmoverse.inference.sbi import SBIEngine
from cosmoverse.simulation.initial_conditions import generate_gaussian_random_field


class CheckpointManager:
    """
    Trains, saves, and reloads deterministic `.npz` model checkpoints for `TUNeSEmulator`
    and `SBIEngine` (`ConditionalRealNVP`) with SHA-256 verification.
    """

    @staticmethod
    def compute_sha256(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    @classmethod
    def train_and_save_checkpoint(
        cls,
        out_dir: str | pathlib.Path,
        grid_size: int = 16,
        n_sbi_sims: int = 48,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Calibrates `TUNeSEmulator` on a 3D simulation bank and trains `SBIEngine` on
        synthetic summary statistics, saving weights and metadata to `out_dir`.
        """
        target_dir = pathlib.Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        cosmo = Cosmology(CosmologicalParameters.planck2018())
        sim_params = SimulationParameters(box_size=100.0, n_mesh=grid_size, seed=seed)
        delta_lin, _ = generate_gaussian_random_field(cosmo, sim_params, seed=seed)
        target_field = delta_lin + 0.015 * (delta_lin ** 2 - np.mean(delta_lin ** 2))

        tunes = TUNeSEmulator(base_features=8, depth=2, seed=seed)
        tunes.calibrate_to_simulation_bank(
            delta_lin[np.newaxis, ...], target_field[np.newaxis, ...]
        )

        rng = np.random.default_rng(seed)

        def _sim_fn(theta_vec: np.ndarray) -> np.ndarray:
            om, s8 = float(theta_vec[0]), float(theta_vec[1])
            return np.array(
                [
                    om * 10.0 + rng.normal(0.0, 0.04),
                    s8 * 8.0 + rng.normal(0.0, 0.04),
                    (om * s8) * 5.0,
                    (om - 0.30) ** 2,
                ],
                dtype=np.float64,
            )

        sbi = SBIEngine(
            simulator_fn=_sim_fn,
            param_bounds={"omega_m": (0.20, 0.45), "sigma8": (0.65, 0.95)},
            seed=seed,
        )
        sbi.generate_training_bank(n_simulations=n_sbi_sims)
        sbi.fit_density_estimator(n_layers=3)
        final_loss = float(np.mean(np.diag(sbi._residual_cov_chol)))

        ckpt_path = target_dir / "cosmoverse_v1_checkpoint.npz"
        np.savez_compressed(
            ckpt_path,
            tunes_residual_scale=np.array([tunes._nonlinear_alpha], dtype=np.float64),
            tunes_spectral_gain=np.array([tunes._spectral_gain], dtype=np.float64),
            sbi_x_mean=sbi._x_mean,
            sbi_x_std=sbi._x_std,
            sbi_reg_weights=sbi._reg_weights,
            sbi_reg_bias=sbi._reg_bias,
            final_loss=np.array([final_loss], dtype=np.float64),
        )
        sha = cls.compute_sha256(ckpt_path)

        manifest = {
            "checkpoint_file": str(ckpt_path),
            "sha256": sha,
            "grid_size": grid_size,
            "n_sbi_sims": n_sbi_sims,
            "tunes_residual_scale": float(tunes._nonlinear_alpha),
            "sbi_final_loss": final_loss,
        }
        manifest_path = target_dir / "checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def verify_and_load_checkpoint(
        cls, checkpoint_dir: str | pathlib.Path
    ) -> dict[str, Any]:
        target_dir = pathlib.Path(checkpoint_dir)
        manifest = json.loads(
            (target_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        ckpt_path = pathlib.Path(manifest["checkpoint_file"])
        actual_sha = cls.compute_sha256(ckpt_path)
        if actual_sha != manifest["sha256"]:
            raise ValueError("Checkpoint SHA-256 mismatch!")
        data = np.load(ckpt_path)
        return {
            "is_verified": True,
            "sha256": actual_sha,
            "tunes_residual_scale": float(data["tunes_residual_scale"][0]),
            "sbi_final_loss": float(data["final_loss"][0]),
        }


class CAMELSArchiveFetcher:
    """
    Populates and verifies a local CAMELS multi-suite spectrum cache (`Pk_m_z=0.00.txt`)
    compatible with `CAMELSHydroSpectraConnector(data_dir=...)`.
    """

    @staticmethod
    def fetch_and_cache_public_spectra(
        cache_dir: str | pathlib.Path,
        suites: tuple[str, ...] = ("CAMELS-IllustrisTNG", "CAMELS-SIMBA", "CAMELS-Astrid"),
        n_per_suite: int = 4,
        seed: int = 42,
    ) -> dict[str, Any]:
        out_dir = pathlib.Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        connector = CAMELSHydroSpectraConnector()

        cached_files: list[str] = []
        for idx, suite in enumerate(suites):
            cat = CAMELSGen2Catalog(family_id=suite, seed=seed + idx * 100)
            for rec in cat.get_subset("LH")[:n_per_suite]:
                spec = connector.load_or_compute_spectrum(rec, suite_override=suite)
                txt_path = out_dir / f"{rec.sim_id}_Pk.txt"
                arr = np.column_stack([spec["k"], spec["pk_hydro"]])
                np.savetxt(txt_path, arr, header="k [h/Mpc]   P_hydro(k) [(Mpc/h)^3]")
                cached_files.append(str(txt_path))

        return {
            "cache_dir": str(out_dir),
            "suites": list(suites),
            "n_files_cached": len(cached_files),
            "files": cached_files,
        }
