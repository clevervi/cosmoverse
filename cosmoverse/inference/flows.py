"""
Conditional Normalizing Flows for Simulation-Based Inference (SBI).
Provides RealNVP and affine coupling transformations for density estimation.
"""

from typing import Optional, Tuple

import numpy as np


class AffineCouplingLayer:
    """
    Affine coupling layer for conditional normalizing flows:
    y1 = x1
    y2 = x2 * exp(s(x1, context)) + t(x1, context)
    """

    def __init__(self, dim: int, context_dim: int, seed: int = 42):
        self.dim = dim
        self.split_dim = dim // 2
        self.context_dim = context_dim
        rng = np.random.default_rng(seed)

        # Small random weights for scale and translation networks
        input_size = self.split_dim + context_dim
        output_size = dim - self.split_dim

        self.w_s = rng.standard_normal((input_size, output_size)) * 0.05
        self.b_s = np.zeros(output_size)
        self.w_t = rng.standard_normal((input_size, output_size)) * 0.05
        self.b_t = np.zeros(output_size)

    def forward(
        self, x: np.ndarray, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms base distribution sample z -> theta given context x.
        Returns: (transformed_y, log_abs_det_jacobian)
        """
        x1 = x[:, : self.split_dim]
        x2 = x[:, self.split_dim :]

        inp = np.concatenate([x1, context], axis=1)
        scale = np.tanh(inp @ self.w_s + self.b_s)
        shift = inp @ self.w_t + self.b_t

        y1 = x1
        y2 = x2 * np.exp(scale) + shift
        y = np.concatenate([y1, y2], axis=1)

        log_det = np.sum(scale, axis=1)
        return y, log_det

    def inverse(
        self, y: np.ndarray, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms parameter theta -> base standard normal z given context x.
        """
        y1 = y[:, : self.split_dim]
        y2 = y[:, self.split_dim :]

        inp = np.concatenate([y1, context], axis=1)
        scale = np.tanh(inp @ self.w_s + self.b_s)
        shift = inp @ self.w_t + self.b_t

        x1 = y1
        x2 = (y2 - shift) * np.exp(-scale)
        x = np.concatenate([x1, x2], axis=1)

        log_det = -np.sum(scale, axis=1)
        return x, log_det


class ConditionalRealNVP:
    """
    Multi-layer Conditional RealNVP Normalizing Flow.
    """

    def __init__(self, param_dim: int, context_dim: int, n_layers: int = 4, seed: int = 42):
        self.param_dim = param_dim
        self.context_dim = context_dim
        self.layers = [
            AffineCouplingLayer(param_dim, context_dim, seed=seed + i)
            for i in range(n_layers)
        ]

    def sample(self, context: np.ndarray, n_samples: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Draws posterior samples theta ~ q(theta | context).
        """
        rng = np.random.default_rng(seed)
        # Base standard normal
        z = rng.standard_normal((n_samples, self.param_dim))
        ctx_batch = np.tile(context, (n_samples, 1)) if context.ndim == 1 else context

        curr = z
        for layer in self.layers:
            curr, _ = layer.forward(curr, ctx_batch)
            # Permute dimensions between layers
            curr = curr[:, ::-1]

        return curr
