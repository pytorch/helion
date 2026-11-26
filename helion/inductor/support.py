"""What Helion kernels support, which helion.inductor asks both while lowering
(HelionChoices) and while generating code (HelionScheduling)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.ir import ComputedBuffer
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.runtime_utils import last_power_of_2
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    from collections.abc import Iterable

    import sympy
    from torch._inductor.scheduler import BaseSchedulerNode

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


def reduced_precision_reductions() -> dict[torch.dtype, tuple[bool, bool]]:
    """torch.backends.cuda.matmul's flag pair for each 16-bit dtype:
    (allow_bf16_reduced_precision_reduction, its _split_k) for bf16, whether
    cuBLAS may reduce a split K in that dtype and whether it may split K."""
    return {
        torch.bfloat16: torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        torch.float16: torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
    }


# Helion's Settings.split_k_reduction, from the most to the least permissive.
SPLIT_K_REDUCTIONS = ("output", "accumulator", "none")


def split_k_reduction(dtypes: Iterable[torch.dtype]) -> str:
    """The Settings.split_k_reduction of a kernel whose dots multiply 16-bit
    operands of `dtypes`, the strictest over them: (True, True) -> "output",
    (False, True) -> "accumulator", (False, False) -> "none"."""
    flags = reduced_precision_reductions()
    return SPLIT_K_REDUCTIONS[
        max(
            (
                0 if reduced else 1 if split_k else 2
                for reduced, split_k in map(flags.__getitem__, dtypes)
            ),
            default=0,
        )
    ]


def dot_rows_on_x(node: BaseSchedulerNode) -> OrderedSet[bool]:
    """For each native matmul in `node`, whether its dot's rows are the x loop
    var rather than y: the var whose coefficient in the output's store index
    is the output's row stride (an N-contiguous output, as lowering/mm.py's
    transposed product writes, puts them on x)."""
    rows_on_x: OrderedSet[bool] = OrderedSet()
    for snode in node.get_nodes():
        if not (
            isinstance(snode, SchedulerNode)
            and isinstance(snode.node, ComputedBuffer)
            and snode.is_native_matmul()
        ):
            continue
        body = snode._body
        pointwise = list(body.var_ranges)[: len(snode._sizes[0])]
        (store,) = body.memory_usage[MemoryUsageType.STORE_REDUCTION]
        index = body.indexing_exprs[store.index_name]
        row_stride = snode.node.get_layout().stride[-2]
        rows_on_x.add(len(pointwise) >= 2 and index.coeff(pointwise[-1]) == row_stride)
    return rows_on_x
