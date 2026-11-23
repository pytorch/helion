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


def names_rdim(rnumel: sympy.Expr) -> bool:
    """Whether HelionKernel.rdim_code() can name an rdim of `rnumel` elements,
    so that its reduction can be persistent. Not for an rdim of fewer
    than 16 that is not a power of two: Helion sizes a factory's dim of under
    16 by its value, so hl.zeros([tile_x, 12]).size(1) is 12 rather than the
    reduction dim of in0[tile_x, :], and hl.arange(12) beside the slice fails
    with "Attempting to broadcast a dimension of length 12". Such an rdim keeps
    a loop over tile_r, of one block."""
    if not rnumel.is_Integer:
        return False
    size = int(rnumel)
    return size >= 16 or size == next_power_of_2(size)


def mix_order_split(xnumel: int, device: torch.device) -> int | None:
    """The x indices of each split of a mix-order reduction over `xnumel` of
    them (HelionScheduling.codegen_mix_order_reduction()), whose kernel tiles
    the splits and loops over the x indices of each: a power of two from 16 to
    128 that divides xnumel, the largest of those that leaves 8 splits per SM
    if one does, as Inductor's Triton kernel picks; None if no such power
    divides it. A split that divides xnumel gives the kernel a [splits, split,
    r] view of each [x, r] tensor, in0[tile_y, tile_x, :]."""
    splits = DeviceProperties.create(device).multi_processor_count * 8
    split = min(max(last_power_of_2(xnumel // splits), 16), 128)
    while split >= 16:
        if xnumel % split == 0:
            return split
        split //= 2
    return None
