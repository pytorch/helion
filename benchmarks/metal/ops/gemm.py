"""gemm: fp16 ``a @ w`` with a row-major [M, K] and a column-major [K, N]."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from helion._testing import import_example

_examples_matmul = import_example("matmul")
matmul = _examples_matmul.matmul
matmul_autograd = _examples_matmul.matmul_autograd

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (matmul,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    m, n, k = case.params["M"], case.params["N"], case.params["K"]
    # TritonBench's _scaled_randn; w is stored [N, K] and viewed as [K, N],
    # its default "tn" layout.
    return {
        "a": ((torch.randn(m, k, generator=gen) + 1) / k).half(),
        "w": ((torch.randn(n, k, generator=gen) + 1) / k).half().T,
    }


def reference(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return a.double() @ w.double()


def eager(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, w)


def helion(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return matmul_autograd(a, w)


def mlx(a: mx.array, w: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.matmul(a, w)
