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

    def __init__(
        self,
        dim: int,
        context_dim: int,
        seed: int = 42,
        scale_bound: float = 1.0,
    ):
        if scale_bound <= 0.0:
            raise ValueError("scale_bound must be positive")
        self.dim = dim
        self.split_dim = dim // 2
        self.context_dim = context_dim
        self.scale_bound = float(scale_bound)
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
        scale = self.scale_bound * np.tanh(inp @ self.w_s + self.b_s)
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
        scale = self.scale_bound * np.tanh(inp @ self.w_s + self.b_s)
        shift = inp @ self.w_t + self.b_t

        x1 = y1
        x2 = (y2 - shift) * np.exp(-scale)
        x = np.concatenate([x1, x2], axis=1)

        log_det = -np.sum(scale, axis=1)
        return x, log_det

    def inverse_with_cache(
        self, y: np.ndarray, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        """Inverse transform that also returns the cache needed for backprop."""
        y1 = y[:, : self.split_dim]
        y2 = y[:, self.split_dim :]
        inp = np.concatenate([y1, context], axis=1)
        scale = self.scale_bound * np.tanh(inp @ self.w_s + self.b_s)
        shift = inp @ self.w_t + self.b_t

        x1 = y1
        x2 = (y2 - shift) * np.exp(-scale)
        x = np.concatenate([x1, x2], axis=1)
        log_det = -np.sum(scale, axis=1)
        cache = (y1, y2, inp, scale, shift, x2)
        return x, log_det, cache

    def backward_inverse(
        self,
        grad_x: np.ndarray,
        grad_log_det: float,
        cache: tuple[np.ndarray, ...],
    ) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Backpropagates through `inverse_with_cache`.
        Returns the gradient w.r.t. the layer input and the parameter gradients.
        """
        y1, y2, inp, scale, shift, _ = cache
        grad_x1 = grad_x[:, : self.split_dim]
        grad_x2 = grad_x[:, self.split_dim :]
        exp_neg = np.exp(-scale)

        grad_y2 = grad_x2 * exp_neg
        grad_scale = grad_x2 * (-(y2 - shift) * exp_neg) - grad_log_det
        grad_shift = grad_x2 * (-exp_neg)

        tanh_scale = scale / self.scale_bound
        grad_pre_scale = (
            grad_scale * self.scale_bound * (1.0 - tanh_scale**2)
        )
        grad_inp = grad_pre_scale @ self.w_s.T + grad_shift @ self.w_t.T
        grad_y1 = grad_x1 + grad_inp[:, : self.split_dim]
        grad_y = np.concatenate([grad_y1, grad_y2], axis=1)

        return grad_y, {
            "w_s": inp.T @ grad_pre_scale,
            "b_s": np.sum(grad_pre_scale, axis=0),
            "w_t": inp.T @ grad_shift,
            "b_t": np.sum(grad_shift, axis=0),
        }


class ConditionalRealNVP:
    """
    Multi-layer Conditional RealNVP Normalizing Flow.
    """

    def __init__(
        self,
        param_dim: int,
        context_dim: int,
        n_layers: int = 4,
        seed: int = 42,
        scale_bound: float = 1.0,
    ):
        self.param_dim = param_dim
        self.context_dim = context_dim
        self.scale_bound = float(scale_bound)
        self.layers = [
            AffineCouplingLayer(
                param_dim,
                context_dim,
                seed=seed + i,
                scale_bound=self.scale_bound,
            )
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

    @staticmethod
    def _expand_context(context: np.ndarray, n_samples: int) -> np.ndarray:
        ctx = np.asarray(context, dtype=np.float64)
        if ctx.ndim == 1:
            ctx = np.tile(ctx, (n_samples, 1))
        if ctx.shape[0] != n_samples:
            raise ValueError("context batch size does not match theta")
        return ctx

    def inverse(
        self,
        theta: np.ndarray,
        context: np.ndarray,
        return_cache: bool = False,
    ):
        """
        Maps parameter space back to the standard-normal base.
        Optionally returns the per-layer cache needed for backpropagation.
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        ctx = self._expand_context(context, theta_arr.shape[0])
        curr = theta_arr
        total_log_det = np.zeros(theta_arr.shape[0], dtype=np.float64)
        caches = []

        for layer in reversed(self.layers):
            curr = curr[:, ::-1]
            curr, log_det, cache = layer.inverse_with_cache(curr, ctx)
            total_log_det += log_det
            caches.append(cache)

        if return_cache:
            return curr, total_log_det, caches
        return curr, total_log_det

    def log_prob(self, theta: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Per-sample log density log q(theta | context)."""
        z, log_det_inverse = self.inverse(theta, context)
        log_base = -0.5 * np.sum(z**2, axis=1) - 0.5 * self.param_dim * np.log(
            2.0 * np.pi
        )
        return log_base + log_det_inverse

    def backward_log_prob(
        self, theta: np.ndarray, context: np.ndarray
    ) -> list[dict[str, np.ndarray]]:
        """
        Gradients of -mean(log q(theta | context)) w.r.t. every layer parameter.
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        ctx = self._expand_context(context, theta_arr.shape[0])
        z, _, caches = self.inverse(theta_arr, ctx, return_cache=True)

        grad = z
        grad_log_det = -1.0
        grads: list[dict[str, np.ndarray]] = [{} for _ in self.layers]

        for index, (layer, cache) in enumerate(
            zip(self.layers, reversed(caches), strict=True)
        ):
            grad, layer_grads = layer.backward_inverse(grad, grad_log_det, cache)
            grad = grad[:, ::-1]
            grads[index] = layer_grads

        n_samples = max(theta_arr.shape[0], 1)
        return [
            {
                key: value / float(n_samples)
                for key, value in layer_grads.items()
            }
            for layer_grads in grads
        ]

    def fit(
        self,
        theta: np.ndarray,
        context: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, list[float] | int]:
        """
        Trains the conditional flow by maximizing the log-likelihood with Adam.
        `theta` and `context` must be pre-standardized.
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        ctx = self._expand_context(context, theta_arr.shape[0])
        rng = np.random.default_rng(seed)

        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        moments: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        for layer_index, layer in enumerate(self.layers):
            for name in ("w_s", "b_s", "w_t", "b_t"):
                value = getattr(layer, name)
                moments[(layer_index, name)] = (
                    np.zeros_like(value),
                    np.zeros_like(value),
                )

        loss_history: list[float] = []
        step = 0

        for epoch in range(int(n_epochs)):
            permutation = rng.permutation(theta_arr.shape[0])
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, theta_arr.shape[0], int(batch_size)):
                indices = permutation[start : start + int(batch_size)]
                if indices.size < 2:
                    continue

                grads = self.backward_log_prob(
                    theta_arr[indices], ctx[indices]
                )
                step += 1
                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step

                for layer_index, layer in enumerate(self.layers):
                    for name in ("w_s", "b_s", "w_t", "b_t"):
                        grad = grads[layer_index][name]
                        first, second = moments[(layer_index, name)]
                        first = beta1 * first + (1.0 - beta1) * grad
                        second = beta2 * second + (1.0 - beta2) * (grad**2)
                        moments[(layer_index, name)] = (first, second)
                        update = (
                            learning_rate
                            * (first / bias1)
                            / (np.sqrt(second / bias2) + epsilon)
                        )
                        setattr(layer, name, getattr(layer, name) - update)

                epoch_loss += float(
                    -np.mean(
                        self.log_prob(theta_arr[indices], ctx[indices])
                    )
                )
                n_batches += 1

            loss_history.append(epoch_loss / max(n_batches, 1))
            if verbose and (epoch % 20 == 0 or epoch == int(n_epochs) - 1):
                print(
                    f"[Flow training] epoch {epoch + 1}/{n_epochs} "
                    f"loss={loss_history[-1]:.5f}"
                )

        return {"loss_history": loss_history, "n_epochs": int(n_epochs)}
