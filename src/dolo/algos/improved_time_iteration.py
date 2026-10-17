"""Improved Time Iteration (Compatibility module for Newton-Krylov solver)"""

from .newton import (
    newton,
    improved_time_iteration,
    Operator,
    smooth,
    smooth_nodiff,
    ssmul,
    ssmul_inplace,
    d_filt_dx,
    neumann_solve,
    invert_jac,
    radius_jac,
    euler_residuals,
    golden_section_search,
)
from .results import ImprovedTimeIterationResult, NewtonResult

__all__ = [
    "newton",
    "improved_time_iteration",
    "Operator",
    "smooth",
    "smooth_nodiff",
    "ssmul",
    "ssmul_inplace",
    "d_filt_dx",
    "neumann_solve",
    "invert_jac",
    "radius_jac",
    "euler_residuals",
    "golden_section_search",
    "ImprovedTimeIterationResult",
    "NewtonResult",
]
