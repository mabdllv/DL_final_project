from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SamplerParams:
    steps: int = 256
    seed: int = 0
    guidance_scale: float = 1.0
    measurement_sigma: float = 0.0
    guidance_start: float = 1.0
    guidance_end: float = 0.0
    gradient_clip: float = 0.0
    div_weight: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
