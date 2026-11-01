"""
Training routines, cosmological physics loss functions, and validation metrics for emulators.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def power_spectrum_loss(
    pred_pk: np.ndarray,
    true_pk: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Log-space Mean Squared Error between predicted and true matter power spectra.
    L_pk = 1/N * sum ( log(P_pred(k)) - log(P_true(k)) )^2
    """
    safe_pred = np.maximum(pred_pk, 1e-12)
    safe_true = np.maximum(true_pk, 1e-12)
    diff = np.log(safe_pred) - np.log(safe_true)

    if weights is not None:
        return float(np.mean(weights * (diff ** 2)))
    return float(np.mean(diff ** 2))


def cross_correlation_coefficient(
    field_a: np.ndarray,
    field_b: np.ndarray,
    box_size: float,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes cross-correlation coefficient r(k) = P_ab(k) / sqrt( P_aa(k) * P_bb(k) ).
    r(k) close to 1.0 indicates perfect phase and amplitude recovery.
    """
    n = field_a.shape[0]
    volume = box_size ** 3
    dx3 = (box_size / n) ** 3

    ft_a = np.fft.rfftn(field_a) * dx3
    ft_b = np.fft.rfftn(field_b) * dx3

    p_aa = (np.abs(ft_a) ** 2) / volume
    p_bb = (np.abs(ft_b) ** 2) / volume
    p_ab = (np.real(ft_a * np.conj(ft_b))) / volume

    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi

    kx_g, ky_g, kz_g = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(kx_g ** 2 + ky_g ** 2 + kz_g ** 2).ravel()

    k_edges = np.geomspace(2.0 * np.pi / box_size, np.pi * n / box_size, n_bins + 1)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    r_k = np.zeros(n_bins, dtype=np.float64)

    p_aa_flat = p_aa.ravel()
    p_bb_flat = p_bb.ravel()
    p_ab_flat = p_ab.ravel()

    for b in range(n_bins):
        mask = (k_mag >= k_edges[b]) & (k_mag < k_edges[b + 1])
        if np.any(mask):
            denom = np.sqrt(np.mean(p_aa_flat[mask]) * np.mean(p_bb_flat[mask]))
            if denom > 1e-12:
                r_k[b] = np.mean(p_ab_flat[mask]) / denom
            else:
                r_k[b] = 0.0

    return k_centers, r_k


class EmulatorTrainer:
    """
    Manages emulator training cycles, loss logging, and checkpointing.
    """
    def __init__(
        self,
        model: Any,
        learning_rate: float = 1e-3,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        loss_fn: Optional[Callable] = None,
    ) -> float:
        """Runs a training epoch and records loss."""
        l_fn = loss_fn or (lambda y_hat, y: float(np.mean((y_hat - y) ** 2)))
        # Emulate forward pass
        preds = self.model.forward(inputs) if hasattr(self.model, "forward") else self.model.predict(inputs)
        loss = l_fn(preds, targets)
        self.history["train_loss"].append(loss)
        return loss

    def evaluate(
        self,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
        loss_fn: Optional[Callable] = None,
    ) -> float:
        """Evaluates model performance on validation set."""
        l_fn = loss_fn or (lambda y_hat, y: float(np.mean((y_hat - y) ** 2)))
        preds = self.model.forward(val_inputs) if hasattr(self.model, "forward") else self.model.predict(val_inputs)
        loss = l_fn(preds, val_targets)
        self.history["val_loss"].append(loss)
        return loss
