"""
CuPy GPU backend for the Particle-Mesh integrator.

Mirrors `cosmoverse.simulation.evolution` on CUDA arrays. Used by
`ParticleMeshSimulator(use_gpu=True)` for large meshes where cuFFT and the
GPU CIC deposition outperform the NumPy path.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import SimulationParameters


def cic_deposit_gpu(
    positions: cp.ndarray,
    box_size: float,
    n_mesh: int,
) -> cp.ndarray:
    """Cloud-in-Cell deposition on the GPU."""
    dx = box_size / n_mesh
    n_part = positions.shape[0]
    pos_grid = cp.mod(positions, box_size) / dx
    i0 = cp.floor(pos_grid).astype(cp.int64)
    tx = pos_grid - i0
    i1 = cp.mod(i0 + 1, n_mesh)
    i0 = cp.mod(i0, n_mesh)

    w0 = 1.0 - tx
    w1 = tx
    mesh = cp.zeros((n_mesh, n_mesh, n_mesh), dtype=cp.float64)

    for dx_idx, wx in [(0, w0[:, 0]), (1, w1[:, 0])]:
        ix = i1[:, 0] if dx_idx else i0[:, 0]
        for dy_idx, wy in [(0, w0[:, 1]), (1, w1[:, 1])]:
            iy = i1[:, 1] if dy_idx else i0[:, 1]
            wxy = wx * wy
            for dz_idx, wz in [(0, w0[:, 2]), (1, w1[:, 2])]:
                iz = i1[:, 2] if dz_idx else i0[:, 2]
                cp.add.at(mesh, (ix, iy, iz), wxy * wz)

    mean_density = n_part / (n_mesh**3)
    return mesh / (mean_density + 1e-12) - 1.0


def cic_interpolate_gpu(
    forces_grid: cp.ndarray,
    positions: cp.ndarray,
    box_size: float,
) -> cp.ndarray:
    """Cloud-in-Cell interpolation of the force grid on the GPU."""
    n_mesh = forces_grid.shape[1]
    dx = box_size / n_mesh
    pos_grid = cp.mod(positions, box_size) / dx
    i0 = cp.floor(pos_grid).astype(cp.int64)
    tx = pos_grid - i0
    i1 = cp.mod(i0 + 1, n_mesh)
    i0 = cp.mod(i0, n_mesh)

    w0 = 1.0 - tx
    w1 = tx
    particle_forces = cp.zeros_like(positions, dtype=cp.float64)

    for dx_idx, wx in [(0, w0[:, 0]), (1, w1[:, 0])]:
        ix = i1[:, 0] if dx_idx else i0[:, 0]
        for dy_idx, wy in [(0, w0[:, 1]), (1, w1[:, 1])]:
            iy = i1[:, 1] if dy_idx else i0[:, 1]
            wxy = wx * wy
            for dz_idx, wz in [(0, w0[:, 2]), (1, w1[:, 2])]:
                iz = i1[:, 2] if dz_idx else i0[:, 2]
                weight = (wxy * wz)[:, cp.newaxis]
                f_corner = cp.stack(
                    [
                        forces_grid[0, ix, iy, iz],
                        forces_grid[1, ix, iy, iz],
                        forces_grid[2, ix, iy, iz],
                    ],
                    axis=1,
                )
                particle_forces += weight * f_corner

    return particle_forces


def solve_gravitational_acceleration_gpu(
    delta_mesh: cp.ndarray,
    scale_a: float,
    cosmology: Cosmology,
    box_size: float,
) -> cp.ndarray:
    """FFT Poisson solver on the GPU."""
    n = delta_mesh.shape[0]
    kx = cp.fft.fftfreq(n, d=box_size / n) * 2.0 * cp.pi
    ky = cp.fft.fftfreq(n, d=box_size / n) * 2.0 * cp.pi
    kz = cp.fft.rfftfreq(n, d=box_size / n) * 2.0 * cp.pi

    kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = kx_grid**2 + ky_grid**2 + kz_grid**2
    k_sq[0, 0, 0] = 1.0

    h0_scaled = 100.0
    factor = (
        1.5
        * cosmology.params.omega_m
        * (h0_scaled**2)
        / (scale_a + 1e-12)
    )
    delta_k = cp.fft.rfftn(delta_mesh)
    phi_k = -factor * delta_k / k_sq
    phi_k[0, 0, 0] = 0.0

    gx_k = -1j * kx_grid * phi_k
    gy_k = -1j * ky_grid * phi_k
    gz_k = -1j * kz_grid * phi_k

    gx = cp.fft.irfftn(gx_k, s=(n, n, n), axes=(0, 1, 2))
    gy = cp.fft.irfftn(gy_k, s=(n, n, n), axes=(0, 1, 2))
    gz = cp.fft.irfftn(gz_k, s=(n, n, n), axes=(0, 1, 2))
    return cp.stack([gx, gy, gz], axis=0)


def step_leapfrog_kdk_gpu(
    positions: cp.ndarray,
    velocities: cp.ndarray,
    a_start: float,
    a_end: float,
    cosmology: Cosmology,
    sim_params: SimulationParameters,
) -> tuple[cp.ndarray, cp.ndarray]:
    """One KDK leapfrog step on the GPU."""
    box = sim_params.box_size
    n_mesh = sim_params.n_mesh
    da = a_end - a_start
    a_mid = 0.5 * (a_start + a_end)

    h_start = cosmology.H(1.0 / a_start - 1.0) / cosmology.params.h
    h_mid = cosmology.H(1.0 / a_mid - 1.0) / cosmology.params.h
    h_end = cosmology.H(1.0 / a_end - 1.0) / cosmology.params.h

    dt1 = (0.5 * da) / (a_start * h_start)
    dt_drift = da / (a_mid * h_mid)
    dt2 = (0.5 * da) / (a_end * h_end)

    delta = cic_deposit_gpu(positions, box, n_mesh)
    forces = solve_gravitational_acceleration_gpu(
        delta, a_start, cosmology, box
    )
    p_forces = cic_interpolate_gpu(forces, positions, box)
    velocities_half = velocities + (dt1 * (1.0 / a_start)) * p_forces

    positions_new = cp.mod(
        positions + (dt_drift / a_mid) * velocities_half, box
    )

    delta_end = cic_deposit_gpu(positions_new, box, n_mesh)
    forces_end = solve_gravitational_acceleration_gpu(
        delta_end, a_end, cosmology, box
    )
    p_forces_end = cic_interpolate_gpu(forces_end, positions_new, box)
    velocities_new = velocities_half + (dt2 * (1.0 / a_end)) * p_forces_end

    return positions_new, velocities_new


def as_numpy(array):
    """Converts a CuPy array to NumPy, leaving NumPy arrays untouched."""
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
