"""helion.inductor's lowering of aten.mm, which follows the backend's plan
(HelionChoices.transpose_matmul()): the transposed product."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import torch
from torch._inductor import ir
from torch._inductor.lowering import lowerings
from torch._inductor.lowering import user_lowerings
from torch._inductor.virtualized import V

from ..choices import HelionChoices

if TYPE_CHECKING:
    from collections.abc import Callable

aten = torch.ops.aten


def register_mm() -> None:
    """Lower mat1 @ mat2 as the backend's plan asks (HelionChoices), else as
    before: the user_lowerings entry this replaces, a user's, or Inductor's
    own lowering. The transposed product writes an N-contiguous inner buffer,
    whose transpose is the row-major result itself:

        buf0 = empty_strided_cuda((1024, 16384), (1, 1024), torch.bfloat16)
        helion_fused_mm_0(reinterpret_tensor(arg1_1, (1024, 4096), (4096, 1), 0),
                          reinterpret_tensor(arg0_1, (4096, 16384), (1, 4096), 0),
                          buf0)  # permute(mm(b.T, a.T))
        return (reinterpret_tensor(buf0, (16384, 1024), (1024, 1), 0), )
    """
    previous: Callable[..., Any] | None = user_lowerings.get(aten.mm.default)
    if getattr(previous, "_helion_transposed_mm", False):
        return  # registered already

    def lower_mm(mat1: ir.IRNode, mat2: ir.IRNode) -> Any:  # noqa: ANN401
        choices = V.choices
        if not (
            isinstance(choices, HelionChoices) and choices.transpose_matmul(mat1, mat2)
        ):
            return None if previous is None else previous(mat1, mat2)
        permute = lowerings[aten.permute]
        product = lowerings[aten.mm](permute(mat2, [1, 0]), permute(mat1, [1, 0]))
        product.realize()
        product = ir.ExternKernel.require_stride_order(product, [0, 1])
        return permute(product, [1, 0])

    lower_mm._helion_transposed_mm = True  # pyrefly: ignore [missing-attribute]
    user_lowerings[aten.mm.default] = lower_mm
