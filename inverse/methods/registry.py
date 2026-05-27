from __future__ import annotations

from collections.abc import Callable

from . import ddnm, ddrm, dps, repaint, unconditional


def get_sampler(name: str) -> Callable:
    samplers = {
        "unconditional": unconditional.sample,
        "dps": dps.sample,
        "ddnm": ddnm.sample,
        "repaint": repaint.sample,
        "ddrm": ddrm.sample,
    }
    try:
        return samplers[name]
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}. Available: {', '.join(sorted(samplers))}") from exc
