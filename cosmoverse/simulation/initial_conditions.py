"""
Initial conditions generator for cosmological N-body simulations.
Generates Gaussian Random Fields (GRF), Zel'dovich (1LPT), and 2LPT displacements.
"""

from typing import Optional, Tuple

import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import SimulationParameters


def generate_gaussian_random_field(
    cosmology: Cosmology,
    sim_params: SimulationParameters,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a 3D Gaussian Random Field (GRF) density perturbation delta(x) at z=0.

    Returns:
        delta_real: Real-space density contrast grid of shape (N, N, N)
        k_magnitude: 3D grid of wavenumber magnitudes
    """
    n = sim_params.n_mesh
    box = sim_params.box_size
    rng = np.random.default_rng(seed if seed is not None else sim_params.seed)

    # Fundamental frequency and frequencies per axis
    2.0 * np.pi / box
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi  # Hermitian rfft

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2
    k_mag = np.sqrt(k_sq)

    # Avoid zero division at k=0
    k_mag_safe = np.copy(k_mag)
    k_mag_safe[0, 0, 0] = 1.0

    # Linear power spectrum at z=0
    pk = cosmology.linear_matter_power_spectrum(k_mag_safe, z=0.0)
    pk[0, 0, 0] = 0.0  # Zero mean density perturbation

    # Volume element for FFT normalization
    # delta(k) variance = V * P(k) / 2 per real/imag component
    volume = box ** 3
    amplitude = np.sqrt(volume * pk / 2.0)

    # Complex standard normal
    real_part = rng.standard_normal((n, n, n // 2 + 1))
    imag_part = rng.standard_normal((n, n, n // 2 + 1))
    delta_k = amplitude * (real_part + 1j * imag_part)

    # Nyquist and k=0 reality conditions
    delta_k[0, 0, 0] = 0.0 + 0.0j

    # Inverse Fourier transform to real space (scaled by grid count)
    delta_real = np.fft.irfftn(
        delta_k, s=(n, n, n), axes=(0, 1, 2)
    ) * (n**3 / volume)

    return delta_real, k_mag


def compute_zeldovich_displacement(
    delta_k: np.ndarray,
    box_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 1LPT (Zel'dovich) displacement field Psi_1 = -grad(phi_1) in Fourier space.
    nabla^2 phi_1 = delta  =>  Psi_1(k) = -i * (k / k^2) * delta(k)
    """
    n = delta_k.shape[0]
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2
    k_sq[0, 0, 0] = 1.0  # Avoid zero division

    # Invert Laplacian
    green = -1.0 / k_sq
    green[0, 0, 0] = 0.0

    # Displacements in Fourier space: Psi_i(k) = i * k_i * green * delta(k)
    psi_x_k = 1j * kx_grid * green * delta_k
    psi_y_k = 1j * ky_grid * green * delta_k
    psi_z_k = 1j * kz_grid * green * delta_k

    # Back to real space
    psi_x = np.fft.irfftn(psi_x_k, s=(n, n, n), axes=(0, 1, 2))
    psi_y = np.fft.irfftn(psi_y_k, s=(n, n, n), axes=(0, 1, 2))
    psi_z = np.fft.irfftn(psi_z_k, s=(n, n, n), axes=(0, 1, 2))

    return psi_x, psi_y, psi_z


def setup_initial_particles(
    cosmology: Cosmology,
    sim_params: SimulationParameters,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates initial particle positions and peculiar velocities using 1LPT (Zel'dovich).

    Returns:
        positions: (N_particles^3, 3) particle coordinates in Mpc/h in [0, BoxSize)
        velocities: (N_particles^3, 3) peculiar velocities in km/s
    """
    n = sim_params.n_particles
    box = sim_params.box_size
    z_ini = sim_params.z_initial

    # Generate linear density field at z=0
    delta_real, _ = generate_gaussian_random_field(cosmology, sim_params, seed=seed)
    delta_k = np.fft.rfftn(delta_real)

    # 1LPT displacement at z=0
    psi_x, psi_y, psi_z = compute_zeldovich_displacement(delta_k, box)

    # Regular unperturbed Lagrangian grid q
    grid_coords = np.linspace(0.0, box, n, endpoint=False) + 0.5 * (box / n)
    qx, qy, qz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")

    # Linear growth factor and growth rate at z_ini
    d_ini = cosmology.growth_factor(z_ini)
    f_ini = cosmology.growth_rate(z_ini)
    a_ini = 1.0 / (1.0 + z_ini)
    h_ini = cosmology.H(z_ini)  # in km/s/Mpc

    # Particle positions: x = q + D(z) * Psi (periodic boundary)
    px = np.mod(qx + d_ini * psi_x, box)
    py = np.mod(qy + d_ini * psi_y, box)
    pz = np.mod(qz + d_ini * psi_z, box)

    # Particle velocities: v = a * H(a) * f(a) * D(a) * Psi
    # v in km/s (since Psi is in Mpc/h, H is in km/s/(Mpc/h) if we adjust for h)
    # H0 = 100 * h => H(z) / h is in km/s/(Mpc/h)
    h_scaled = h_ini / cosmology.params.h
    vx = a_ini * h_scaled * f_ini * d_ini * psi_x
    vy = a_ini * h_scaled * f_ini * d_ini * psi_y
    vz = a_ini * h_scaled * f_ini * d_ini * psi_z

    positions = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=1)
    velocities = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1)

    return positions, velocities
