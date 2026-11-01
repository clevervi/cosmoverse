"""
Baryonic physics, BCEmu feedback (v1 Primary), baryonification, and Bias-Variance Frontier (Resultado 2).
"""

from cosmoverse.baryons.baryonification import BaryonificationModel
from cosmoverse.baryons.feedback import (
    BACCOEmuPlanBModel,
    BaryonicEmulatorProtocol,
    BCEmuModel,
    apply_baryonic_feedback_to_pk,
    baryonic_suppression_factor,
    create_baryonic_emulator,
)
from cosmoverse.baryons.frontier import BiasVarianceFrontier

__all__ = [
    "BaryonicEmulatorProtocol",
    "BCEmuModel",
    "BACCOEmuPlanBModel",
    "create_baryonic_emulator",
    "baryonic_suppression_factor",
    "apply_baryonic_feedback_to_pk",
    "BaryonificationModel",
    "BiasVarianceFrontier",
]
