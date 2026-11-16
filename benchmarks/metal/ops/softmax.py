"""softmax: fp16 row softmax over dim 1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from helion._testing import import_example

_examples_softmax = import_example("softmax")
softmax_fwd_bwd = _examples_softmax.softmax_fwd_bwd
softmax_two_pass = _examples_softmax.softmax_two_pass

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (softmax_two_pass,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    m, n = case.params["M"], case.params["N"]
    return {"x": torch.randn(m, n, generator=gen, dtype=torch.float16)}


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.double(), dim=1)


def eager(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=1)


def helion(x: torch.Tensor) -> torch.Tensor:
    return softmax_fwd_bwd(x)


def mlx(x: mx.array) -> mx.array:
    import mlx.core as mx

    # precise=True accumulates in fp32, as ATen and Helion do.
    return mx.softmax(x, axis=1, precise=True)
