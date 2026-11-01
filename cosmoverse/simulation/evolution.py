"""
Particle-Mesh (PM) time integration and gravitational solver.
Implements Cloud-in-Cell (CIC) deposition, FFT Poisson solver, and Kick-Drift-Kick (KDK).
"""

from typing import Tuple

import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import SimulationParameters


def cic_deposit(positions: np.ndarray, box_size: float, n_mesh: int) -> np.ndarray:
    """
    Assigns particle mass to a 3D grid using Cloud-in-Cell (CIC) scheme.

    Args:
        positions: (N_part, 3) particle coordinates in [0, box_size)
        box_size: physical box length
        n_mesh: grid resolution

    Returns:
        density_mesh: (N_mesh, N_mesh, N_mesh) density contrast delta = rho / bar{rho} - 1
    """
    dx = box_size / n_mesh
    n_part = positions.shape[0]

    # Grid cell index and fractional offset
    pos_grid = np.mod(positions, box_size) / dx
    i0 = np.floor(pos_grid).astype(np.int64)
    tx = pos_grid - i0
    i1 = np.mod(i0 + 1, n_mesh)
    i0 = np.mod(i0, n_mesh)

    # Weights in each dimension
    w0 = 1.0 - tx
    w1 = tx

    mesh = np.zeros((n_mesh, n_mesh, n_mesh), dtype=np.float64)

    # 8 cell corners
    for dx_idx, wx in [(0, w0[:, 0]), (1, w1[:, 0])]:
        ix = i1[:, 0] if dx_idx else i0[:, 0]
        for dy_idx, wy in [(0, w0[:, 1]), (1, w1[:, 1])]:
            iy = i1[:, 1] if dy_idx else i0[:, 1]
            wxy = wx * wy
            for dz_idx, wz in [(0, w0[:, 2]), (1, w1[:, 2])]:
                iz = i1[:, 2] if dz_idx else i0[:, 2]
                weight = wxy * wz
                np.add.at(mesh, (ix, iy, iz), weight)

    # Convert to overdensity delta = rho / bar{rho} - 1
    mean_density = n_part / (n_mesh ** 3)
    delta_mesh = (mesh / (mean_density + 1e-12)) - 1.0
    return delta_mesh


def cic_interpolate(forces_grid: np.ndarray, positions: np.ndarray, box_size: float) -> np.ndarray:
    """
    Interpolates 3D vector force field on the grid onto particle positions using CIC.

    Args:
        forces_grid: (3, N_mesh, N_mesh, N_mesh) gravitational acceleration field
        positions: (N_part, 3) particle coordinates
        box_size: physical box length

    Returns:
        particle_forces: (N_part, 3) acceleration at particle positions
    """
    n_mesh = forces_grid.shape[1]
    dx = box_size / n_mesh

    pos_grid = np.mod(positions, box_size) / dx
    i0 = np.floor(pos_grid).astype(np.int64)
    tx = pos_grid - i0
    i1 = np.mod(i0 + 1, n_mesh)
    i0 = np.mod(i0, n_mesh)

    w0 = 1.0 - tx
    w1 = tx

    particle_forces = np.zeros_like(positions, dtype=np.float64)

    for dx_idx, wx in [(0, w0[:, 0]), (1, w1[:, 0])]:
        ix = i1[:, 0] if dx_idx else i0[:, 0]
        for dy_idx, wy in [(0, w0[:, 1]), (1, w1[:, 1])]:
            iy = i1[:, 1] if dy_idx else i0[:, 1]
            wxy = wx * wy
            for dz_idx, wz in [(0, w0[:, 2]), (1, w1[:, 2])]:
                iz = i1[:, 2] if dz_idx else i0[:, 2]
                weight = (wxy * wz)[:, np.newaxis]

                f_corner = np.stack([
                    forces_grid[0, ix, iy, iz],
                    forces_grid[1, ix, iy, iz],
                    forces_grid[2, ix, iy, iz],
                ], axis=1)

                particle_forces += weight * f_corner

    return particle_forces


def solve_gravitational_acceleration(
    delta_mesh: np.ndarray,
    scale_a: float,
    cosmology: Cosmology,
    box_size: float,
) -> np.ndarray:
    """
    Solves Poisson's equation for gravitational acceleration:
    nabla^2 phi = (3/2) * Omega_m * (H0^2 / a) * delta
    g = - nabla phi

    Returns:
        forces_grid: (3, N_mesh, N_mesh, N_mesh) acceleration field
    """
    n = delta_mesh.shape[0]
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2
    k_sq[0, 0, 0] = 1.0  # Avoid zero division

    # Source factor in (km/s)^2 / (Mpc/h)
    # H0 in (km/s)/(Mpc/h) = 100
    h0_scaled = 100.0
    factor = 1.5 * cosmology.params.omega_m * (h0_scaled ** 2) / (scale_a + 1e-12)

    delta_k = np.fft.rfftn(delta_mesh)

    # Gravitational potential in Fourier space: phi_k = - factor * delta_k / k^2
    phi_k = - factor * delta_k / k_sq
    phi_k[0, 0, 0] = 0.0

    # Acceleration g = -grad(phi) => g_i(k) = - i * k_i * phi_k
    gx_k = -1j * kx_grid * phi_k
    gy_k = -1j * ky_grid * phi_k
    gz_k = -1j * kz_grid * phi_k

    gx = np.fft.irfftn(gx_k, s=(n, n, n))
    gy = np.fft.irfftn(gy_k, s=(n, n, n))
    gz = np.fft.irfftn(gz_k, s=(n, n, n))

    return np.stack([gx, gy, gz], axis=0)


def step_leapfrog_kdk(
    positions: np.ndarray,
    velocities: np.ndarray,
    a_start: float,
    a_end: float,
    cosmology: Cosmology,
    sim_params: SimulationParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs one Kick-Drift-Kick (KDK) leapfrog integration step from a_start to a_end.
    Positions in [0, box_size) Mpc/h, velocities in km/s.
    """
    box = sim_params.box_size
    n_mesh = sim_params.n_mesh
    da = a_end - a_start
    a_mid = 0.5 * (a_start + a_end)

    # Conversion between da and dt: dt = da / (a * H(a))
    # H(a) / h in (km/s)/(Mpc/h)
    h_start = cosmology.H(1.0 / a_start - 1.0) / cosmology.params.h
    h_mid = cosmology.H(1.0 / a_mid - 1.0) / cosmology.params.h
    h_end = cosmology.H(1.0 / a_end - 1.0) / cosmology.params.h

    dt1 = (0.5 * da) / (a_start * h_start)
    dt_drift = da / (a_mid * h_mid)
    dt2 = (0.5 * da) / (a_end * h_end)

    # 1. First Kick: calculate forces at a_start
    delta = cic_deposit(positions, box, n_mesh)
    forces = solve_gravitational_acceleration(delta, a_start, cosmology, box)
    p_forces = cic_interpolate(forces, positions, box)
    velocities_half = velocities + (dt1 * (1.0 / a_start)) * p_forces

    # 2. Drift: update positions using velocities at half-step
    # v is in km/s, dx is in Mpc/h. 1 Mpc approx 3.08567758e19 km
    # (dt_drift in Mpc/h / (km/s))
    positions_new = np.mod(positions + (dt_drift / a_mid) * velocities_half, box)

    # 3. Second Kick: calculate forces at a_end
    delta_end = cic_deposit(positions_new, box, n_mesh)
    forces_end = solve_gravitational_acceleration(delta_end, a_end, cosmology, box)
    p_forces_end = cic_interpolate(forces_end, positions_new, box)
    velocities_new = velocities_half + (dt2 * (1.0 / a_end)) * p_forces_end

    return positions_new, velocities_new
