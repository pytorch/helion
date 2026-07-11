from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from examples.grouped_gemm import grouped_gemm_jagged_persistent_example

if TYPE_CHECKING:
    import torch


def grouped_gemm_jagged_persistent_tritonbench(
    tb_op: object,
    group_A: list[torch.Tensor],
    group_B: list[torch.Tensor],
    w: torch.Tensor | None = None,
    split: torch.Tensor | None = None,
) -> Callable[[], torch.Tensor]:
    """Adapter for the persistent grouped GEMM jagged demo kernel in TritonBench."""
    B_shared = group_B[0]

    def inner() -> torch.Tensor:
        return grouped_gemm_jagged_persistent_example(group_A, B_shared)

    return inner
