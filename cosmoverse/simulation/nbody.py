"""
High-level Particle-Mesh (PM) N-body simulation orchestrator and power spectrum estimators.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import SimulationParameters
from cosmoverse.simulation.evolution import cic_deposit, step_leapfrog_kdk
from cosmoverse.simulation.initial_conditions import setup_initial_particles


def _as_numpy(array):
    """Converts CuPy arrays to NumPy without importing CuPy on CPU-only runs."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


class ParticleMeshSimulator:
    """
    Particle-Mesh N-body simulator for cosmic structure formation.
    Provides fast, end-to-end forward modeling from cosmological parameters
    to non-linear matter density fields and power spectra.
    """

    def __init__(
        self,
        cosmology: Optional[Cosmology] = None,
        sim_params: Optional[SimulationParameters] = None,
        use_gpu: bool = False,
    ):
        from cosmoverse.acceleration import cupy_available

        if use_gpu and not cupy_available():
            raise RuntimeError(
                "use_gpu=True requires CuPy and a CUDA device"
            )
        self.cosmology = cosmology or Cosmology()
        self.sim_params = sim_params or SimulationParameters()
        self.use_gpu = bool(use_gpu)
        self.positions = None
        self.velocities = None
        self.final_density = None

    def initialize(self, seed: Optional[int] = None):
        """Sets up 1LPT initial conditions at z_initial."""
        self.positions, self.velocities = setup_initial_particles(
            self.cosmology, self.sim_params, seed=seed
        )
        if self.use_gpu:
            import cupy as cp

            self.positions = cp.asarray(self.positions, dtype=cp.float64)
            self.velocities = cp.asarray(self.velocities, dtype=cp.float64)
        return self.positions, self.velocities

    def run(self, seed: Optional[int] = None, verbose: bool = False) -> np.ndarray:
        """
        Runs the full PM simulation from z_initial to z_final.

        Returns:
            final_density: (N_mesh, N_mesh, N_mesh) density contrast delta at z_final
        """
        if self.positions is None:
            self.initialize(seed=seed)

        a_ini = 1.0 / (1.0 + self.sim_params.z_initial)
        a_end = 1.0 / (1.0 + self.sim_params.z_final)

        # Scale factor timeline
        a_steps = np.linspace(a_ini, a_end, self.sim_params.n_steps + 1)

        if self.use_gpu:
            from cosmoverse.simulation.gpu_evolution import (
                cic_deposit_gpu,
                step_leapfrog_kdk_gpu,
            )

            step_fn = step_leapfrog_kdk_gpu
            deposit_fn = cic_deposit_gpu
        else:
            step_fn = step_leapfrog_kdk
            deposit_fn = cic_deposit

        pos = self.positions.copy()
        vel = self.velocities.copy()

        for step in range(self.sim_params.n_steps):
            a_cur = a_steps[step]
            a_next = a_steps[step + 1]
            if verbose:
                z_cur = 1.0 / a_cur - 1.0
                print(f"[PM Step {step+1}/{self.sim_params.n_steps}] a={a_cur:.4f} (z={z_cur:.2f}) -> a={a_next:.4f}")
            pos, vel = step_fn(
                pos, vel, a_cur, a_next, self.cosmology, self.sim_params
            )

        self.positions = pos
        self.velocities = vel
        self.final_density = deposit_fn(
            pos, self.sim_params.box_size, self.sim_params.n_mesh
        )
        return self.final_density

    def run_with_snapshots(
        self,
        snapshot_redshifts: Sequence[float],
        seed: Optional[int] = None,
        include_density: bool = False,
        verbose: bool = False,
    ) -> Dict[float, dict]:
        """
        Runs the PM simulation and records particle positions/velocities at the
        requested snapshot redshifts. The timeline is the union of the initial
        scale factor, the requested snapshots, and the final scale factor.
        """
        if self.positions is None:
            self.initialize(seed=seed)

        a_ini = 1.0 / (1.0 + self.sim_params.z_initial)
        a_end = 1.0 / (1.0 + self.sim_params.z_final)
        a_targets = sorted(
            {
                1.0 / (1.0 + float(z))
                for z in snapshot_redshifts
                if self.sim_params.z_final <= float(z) <= self.sim_params.z_initial
            }
        )
        if not a_targets:
            raise ValueError("No snapshot redshifts lie within [z_final, z_initial]")

        if self.use_gpu:
            from cosmoverse.simulation.gpu_evolution import (
                cic_deposit_gpu,
                step_leapfrog_kdk_gpu,
            )

            step_fn = step_leapfrog_kdk_gpu
            deposit_fn = cic_deposit_gpu
        else:
            step_fn = step_leapfrog_kdk
            deposit_fn = cic_deposit

        a_steps = np.unique(np.concatenate(([a_ini], a_targets, [a_end])))
        pos = self.positions.copy()
        vel = self.velocities.copy()
        snapshots: Dict[float, dict] = {}

        for step in range(len(a_steps) - 1):
            a_cur = float(a_steps[step])
            a_next = float(a_steps[step + 1])

            if np.any(np.isclose(a_cur, a_targets)):
                z_cur = 1.0 / a_cur - 1.0
                snapshots[float(z_cur)] = self._make_snapshot(
                    pos, vel, z_cur, include_density=include_density
                )

            pos, vel = step_fn(
                pos, vel, a_cur, a_next, self.cosmology, self.sim_params
            )
            if verbose:
                print(
                    f"[PM snapshot step {step + 1}/{len(a_steps) - 1}] "
                    f"a={a_cur:.4f} -> {a_next:.4f}"
                )

        if np.any(np.isclose(a_steps[-1], a_targets)):
            z_final = 1.0 / float(a_steps[-1]) - 1.0
            snapshots[float(z_final)] = self._make_snapshot(
                pos, vel, z_final, include_density=include_density
            )

        self.positions = pos
        self.velocities = vel
        self.final_density = deposit_fn(
            pos, self.sim_params.box_size, self.sim_params.n_mesh
        )
        return snapshots

    def _make_snapshot(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        redshift: float,
        include_density: bool,
    ) -> dict:
        """Builds one immutable snapshot payload for light-cone construction."""
        positions_np = _as_numpy(positions)
        velocities_np = _as_numpy(velocities)
        snapshot = {
            "z": float(redshift),
            "positions_mpc_h": np.array(positions_np, copy=True),
            "velocities_km_s": np.array(velocities_np, copy=True),
        }
        if include_density:
            snapshot["density"] = cic_deposit(
                positions_np, self.sim_params.box_size, self.sim_params.n_mesh
            )
        return snapshot

    def compute_power_spectrum(
        self,
        n_bins: int = 25,
        density_field: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the spherically averaged 3D power spectrum P(k) from the density grid.

        Returns:
            k_centers: (n_bins,) bin centers in h/Mpc
            pk: (n_bins,) estimated matter power spectrum in (Mpc/h)^3
        """
        grid = self.final_density if density_field is None else density_field
        if grid is None:
            raise ValueError("No density field available. Run simulation first.")

        return estimate_power_spectrum_3d(
            _as_numpy(grid), self.sim_params.box_size, n_bins=n_bins
        )


def estimate_power_spectrum_3d(
    density_grid: np.ndarray,
    box_size: float,
    n_bins: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherically averaged 3D matter power spectrum estimator P(k) = < |delta(k)|^2 > * V.
    """
    n = density_grid.shape[0]
    volume = box_size ** 3

    # 3D FFT (Hermitian)
    delta_k = np.fft.rfftn(density_grid) * (box_size / n) ** 3
    power_raw = (np.abs(delta_k) ** 2) / volume

    # Frequencies
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2)

    # Binning in spherical shells from fundamental k to Nyquist k
    k_fund = 2.0 * np.pi / box_size
    k_nyq = np.pi * n / box_size
    k_edges = np.geomspace(k_fund, k_nyq, n_bins + 1)

    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    pk_binned = np.zeros(n_bins, dtype=np.float64)

    flat_k = k_mag.ravel()
    flat_pk = power_raw.ravel()

    for b in range(n_bins):
        mask = (flat_k >= k_edges[b]) & (flat_k < k_edges[b + 1])
        if np.any(mask):
            pk_binned[b] = np.mean(flat_pk[mask])
        else:
            pk_binned[b] = 0.0

    return k_centers, pk_binned
