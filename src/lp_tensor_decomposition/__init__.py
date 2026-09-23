"""Lovitz--Petrov tensor decomposition."""

from .contractions import alternating_contraction
from .decompose import LPDecompositionResult, decompose_lp
from .tensor_utils import cp_tensor, relative_residual

__all__ = [
    "LPDecompositionResult",
    "alternating_contraction",
    "cp_tensor",
    "decompose_lp",
    "relative_residual",
]
