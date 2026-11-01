"""
Neural emulator models:
  - Primary v1 Architecture: TUNeSEmulator (Temporal U-Net)
  - Plan B / Alternative: FNO3dEmulator (Fourier Neural Operator / CosmoRGNet candidate)
  - Summary Statistics (Resultado 1): Gaussian P(k) MAPE, 1-point PDF, 3D Peak Counts, 3D Minkowski Functionals
"""

from cosmoverse.emulators.fno import FNO3dEmulator, SpectralConv3dNumPy
from cosmoverse.emulators.statistics import (
    compute_minkowski_functionals_3d,
    compute_one_point_pdf,
    compute_peak_counts_3d,
    compute_pk_mape,
    evaluate_emulator_result1,
)
from cosmoverse.emulators.training import (
    EmulatorTrainer,
    cross_correlation_coefficient,
    power_spectrum_loss,
)
from cosmoverse.emulators.unet import CosmoUNet, ResidualBlock3D, TUNeSEmulator

__all__ = [
    "TUNeSEmulator",
    "CosmoUNet",
    "ResidualBlock3D",
    "FNO3dEmulator",
    "SpectralConv3dNumPy",
    "power_spectrum_loss",
    "cross_correlation_coefficient",
    "EmulatorTrainer",
    "compute_pk_mape",
    "compute_one_point_pdf",
    "compute_peak_counts_3d",
    "compute_minkowski_functionals_3d",
    "evaluate_emulator_result1",
]
