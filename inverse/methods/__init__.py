"""Sampler implementations.

Each method module exposes:

    sample(checkpoint, operator, y_norm, params) -> x_hat_norm

The returned tensor must be normalized model-space velocity with shape
``[B, C, H, W]``.
"""

from .registry import get_sampler

__all__ = ["get_sampler"]
