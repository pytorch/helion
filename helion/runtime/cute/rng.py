"""Host staging for explicit dynamic CuTe RNG seeds."""

from __future__ import annotations

import torch


def stage_explicit_seed(seed: int, like: torch.Tensor) -> torch.Tensor:
    """Allocate and fill on the caller's current device stream for every call."""
    result = torch.empty((1,), dtype=torch.int64, device=like.device)
    result.fill_(seed)
    return result
