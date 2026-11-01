"""
Survey selection functions, fiber assignment completeness, and angular masks.
"""

from typing import Optional

import numpy as np


class SelectionFunction:
    """
    Models spectroscopic selection efficiency and fiber collisions.
    """
    def __init__(
        self,
        completeness_rate: float = 0.95,
        fiber_collision_radius_arcsec: float = 62.0,
    ):
        self.completeness_rate = completeness_rate
        self.fiber_collision_radius_arcsec = fiber_collision_radius_arcsec

    def apply_selection(
        self,
        coordinates: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Applies probabilistic target selection to simulated galaxies.

        Args:
            coordinates: (N, 3) positions

        Returns:
            mask: Boolean array indicating selected galaxies
        """
        n_targets = coordinates.shape[0]
        rng = np.random.default_rng(seed)
        random_draw = rng.uniform(0.0, 1.0, size=n_targets)
        return random_draw < self.completeness_rate
