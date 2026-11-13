"""A reduction whose synthetic axis has no input coordinate dependency."""

from __future__ import annotations

from typing import cast

import torch

import helion
import helion.language as hl

BROADCAST_PHYSICAL_CONFIGS = ((8, 4, 1), (16, 4, 4), (64, 32, 1))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def broadcast_count(
    x: torch.Tensor,
    copies: hl.constexpr,
    reexpand: hl.constexpr = 0,  # pyrefly: ignore [bad-function-definition]
) -> torch.Tensor:
    output = torch.empty_like(x, dtype=torch.int64)
    for tile in hl.tile(x.shape):
        predicate = x[tile].reshape(-1, 1) > 0
        repeated = predicate.expand(predicate.shape[0], cast("int", copies))
        if reexpand == 2:
            expanded = repeated.expand(1, predicate.shape[0], cast("int", copies))
            reduced = expanded.sum(2).reshape(-1)
        else:
            if reexpand == 1:
                repeated = repeated.expand(predicate.shape[0], cast("int", copies))
            padded = repeated[:, :, None]
            reduced = padded.sum(1).reshape(-1)
        output[tile] = reduced
    return output
