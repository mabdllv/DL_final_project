"""Shared inverse-problem evaluation utilities.

The package is intentionally split into method-agnostic experiment code and
small method modules under :mod:`inverse.methods`.  All methods should consume
the same saved case files so comparisons use identical validation fields and
identical corruptions.
"""

from .cases import CaseConfig, create_case_file, load_case_file
from .checkpoint import LoadedScoreCheckpoint, load_score_checkpoint
from .operators import LinearOperator, build_operator

__all__ = [
    "CaseConfig",
    "LinearOperator",
    "LoadedScoreCheckpoint",
    "build_operator",
    "create_case_file",
    "load_case_file",
    "load_score_checkpoint",
]
