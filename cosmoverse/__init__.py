"""
COSMOVERSE: Cosmological Modeling and Inference Engine
======================================================

An open framework for computational cosmology bridging differentiable simulation,
neural emulation, and Bayesian parameter inference.

Copyright (c) 2026 clevervi (adriandariov@gmail.com)
Licensed under the MIT License.
"""

__version__ = "3.5.0"
__author__ = "clevervi (Adrian Dario V.)"
__email__ = "adriandariov@gmail.com"

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import (
    BaryonParameters,
    CosmologicalParameters,
    SimulationParameters,
)

__all__ = [
    "__version__",
    "Cosmology",
    "CosmologicalParameters",
    "BaryonParameters",
    "SimulationParameters",
]
