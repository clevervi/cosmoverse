"""
Baryonification framework: particle and field displacement model.
Alters Dark Matter Only (DMO) density fields to match full hydrodynamic simulation profiles.
"""

from typing import Optional

import numpy as np

from cosmoverse.core.parameters import BaryonParameters


class BaryonificationModel:
    """
    Implements the baryonification displacement field method.
    Re-distributes mass from halo centers into extended halos via AGN gas ejection.
    Reference: Schneider et al. (2015), Arico et al. (2020).
    """

    def __init__(self, params: Optional[BaryonParameters] = None):
        self.params = params or BaryonParameters()

    def displace_density_field(
        self,
        delta_grid: np.ndarray,
        box_size: float,
    ) -> np.ndarray:
        """
        Applies a field-level baryonic filter in Fourier space.
        Modulates overdense halo peaks and redistributes gas into circumgalactic medium.
        """
        n = delta_grid.shape[0]
        kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
        ky = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
        kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi

        kx_g, ky_g, kz_g = np.meshgrid(kx, ky, kz, indexing="ij")
        k_mag = np.sqrt(kx_g ** 2 + ky_g ** 2 + kz_g ** 2)

        # High-k suppression filter corresponding to baryon feedback
        k_scale = 3.0  # h / Mpc
        # Filter factor: dips at non-linear scales
        filter_transfer = 1.0 - 0.2 * (self.params.A_AGN / 1.0) / (1.0 + (k_scale / np.maximum(k_mag, 1e-4)) ** 2)

        delta_k = np.fft.rfftn(delta_grid)
        delta_baryon_k = delta_k * filter_transfer

        delta_baryon = np.fft.irfftn(
        delta_baryon_k, s=(n, n, n), axes=(0, 1, 2)
    )
        return delta_baryon
