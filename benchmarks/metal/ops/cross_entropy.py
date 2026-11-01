"""cross_entropy: mean over rows of ``logsumexp(x) - x[target]``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from . import REFERENCE_ROWS
from helion._testing import import_example

cross_entropy = import_example("cross_entropy").cross_entropy

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (cross_entropy,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    n, v = case.params["N"], case.params["V"]
    return {
        "logits": torch.randn(n, v, generator=gen),
        "target": torch.randint(v, (n,), generator=gen),
    }


def reference(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, logits.shape[0], REFERENCE_ROWS):
        rows = logits[start : start + REFERENCE_ROWS].double()
        picked = rows.gather(1, target[start : start + REFERENCE_ROWS, None])
        total += (torch.logsumexp(rows, dim=1) - picked.squeeze(1)).sum()
    return total / logits.shape[0]


def eager(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def helion(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits, target)


def mlx(logits: mx.array, target: mx.array) -> mx.array:
    import mlx.nn

    return mlx.nn.losses.cross_entropy(logits, target, reduction="mean")
