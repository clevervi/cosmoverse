"""
Fourier Neural Operator (FNO) for field-level cosmological emulation.
Maps linear initial density fields or parameters to non-linear cosmological fields.
Reference: Li et al. (2020) "Fourier Neural Operator for Parametric Partial Differential Equations".
"""

import numpy as np


class SpectralConv3dNumPy:
    """
    Reference Spectral Convolution 3D layer for Fourier Neural Operators.
    Operates in Fourier space by parameterizing Fourier modes up to a cutoff frequency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        modes3: int = 12,
        seed: int = 42,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        rng = np.random.default_rng(seed)
        scale = 1.0 / (in_channels * out_channels)

        # 8 octants for 3D modes
        self.weights1 = scale * (
            rng.standard_normal((in_channels, out_channels, modes1, modes2, modes3))
            + 1j * rng.standard_normal((in_channels, out_channels, modes1, modes2, modes3))
        )
        self.weights2 = scale * (
            rng.standard_normal((in_channels, out_channels, modes1, modes2, modes3))
            + 1j * rng.standard_normal((in_channels, out_channels, modes1, modes2, modes3))
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass:
        x: (batch, in_channels, N, N, N)
        """
        b, c, n1, n2, n3 = x.shape
        x_ft = np.fft.rfftn(x, axes=(-3, -2, -1))

        out_ft = np.zeros(
            (b, self.out_channels, n1, n2, n3 // 2 + 1), dtype=np.complex128
        )

        m1, m2, m3 = min(self.modes1, n1 // 2), min(self.modes2, n2 // 2), min(self.modes3, n3 // 2 + 1)

        # Low frequency mode multiplication
        # Einstein summation: out_ft[b, o, x, y, z] = sum_i x_ft[b, i, x, y, z] * weights[i, o, x, y, z]
        sub_x = x_ft[:, :, :m1, :m2, :m3]
        w1 = self.weights1[:, :, :m1, :m2, :m3]
        out_ft[:, :, :m1, :m2, :m3] = np.einsum("bixyz,ioxyz->boxyz", sub_x, w1)

        # Inverse FFT back to physical domain
        x_out = np.fft.irfftn(out_ft, s=(n1, n2, n3), axes=(-3, -2, -1))
        return x_out


class FNO3dEmulator:
    """
    3D Fourier Neural Operator emulator for cosmological simulations.
    Can predict 3D cosmic web density fields from linear initial perturbations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 16,
        modes: int = 8,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes

        # Spectral conv layer
        self.spectral_layer = SpectralConv3dNumPy(
            in_channels=width,
            out_channels=width,
            modes1=modes,
            modes2=modes,
            modes3=modes,
        )

    def predict(self, initial_delta: np.ndarray) -> np.ndarray:
        """
        Predicts non-linear density field delta from initial Gaussian field.
        Input: (N, N, N) or (Batch, 1, N, N, N).
        """
        if initial_delta.ndim == 3:
            x = initial_delta[np.newaxis, np.newaxis, ...]
        elif initial_delta.ndim == 4:
            x = initial_delta[:, np.newaxis, ...]
        else:
            x = initial_delta

        b, _, n1, n2, n3 = x.shape
        # Lifting: project input to higher channel width
        lifted = np.repeat(x, self.width, axis=1)

        # Spectral convolution + linear skip
        conv_out = self.spectral_layer.forward(lifted)
        activated = np.maximum(0.0, conv_out + lifted)  # ReLU / GELU approximation

        # Projection back to single density channel
        out = np.mean(activated, axis=1, keepdims=True)
        return np.squeeze(out)
