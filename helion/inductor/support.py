"""What Helion kernels support, which helion.inductor asks both while lowering
(HelionChoices) and while generating code (HelionScheduling)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.runtime_utils import last_power_of_2
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    import sympy

MATMUL_DTYPES = OrderedSet([torch.float16, torch.bfloat16, torch.float32])


def python_wrapper() -> bool:
    return not (V.graph.cpp_wrapper or V.graph.fx_wrapper)


# Inductor's numerics modes, which Helion kernels do not implement yet.
NUMERICS_MODES = (
    "strict_reduction",
    "batch_invariant",
    "deterministic",
    "eager_numerics.division_rounding",
    "eager_numerics.disable_ftz",
    "eager_numerics.use_pytorch_libdevice",
    "emulate_precision_casts",
)


def numerics_modes() -> list[str]:
    """The NUMERICS_MODES that are on."""
    return [
        mode
        for mode in NUMERICS_MODES
        if functools.reduce(getattr, mode.split("."), config)
    ]


def names_row_dim(rnumel: sympy.Expr) -> bool:
    """Whether HelionKernel.row_dim() can name the dim of a row of `rnumel`
    elements, so that its reduction can be persistent. Not for a row of fewer
    than 16 that is not a power of two: Helion sizes a factory's dim of under
    16 by its value, so hl.zeros([tile_x, 12]).size(1) is 12 rather than the
    reduction dim of in0[tile_x, :], and hl.arange(12) beside the slice fails
    with "Attempting to broadcast a dimension of length 12". Such a row keeps
    a loop over tile_r, of one block."""
    if not rnumel.is_Integer:
        return False
    size = int(rnumel)
    return size >= 16 or size == next_power_of_2(size)


def mix_order_split(rows: int, device: torch.device) -> int | None:
    """The rows of each split of a mix-order reduction over `rows` rows
    (HelionScheduling.codegen_mix_order_reduction()), whose kernel tiles the
    splits and loops over the rows of each: a power of two from 16 to 128 that
    divides the rows, the largest of those that leaves 8 splits per SM if one
    does, as Inductor's Triton kernel picks; None if no such power divides
    them. A split that divides the rows gives the kernel a [splits, split,
    cols] view of each [rows, cols] tensor, in0[tile_y, tile_x, :]."""
    splits = DeviceProperties.create(device).multi_processor_count * 8
    split = min(max(last_power_of_2(rows // splits), 16), 128)
    while split >= 16:
        if rows % split == 0:
            return split
        split //= 2
    return None
