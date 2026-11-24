"""Reduce over a tensor's dims in the order its memory holds them.

Inductor lowers a reduction over several dims in the order of their dim
numbers, and splits a large one into chunks of the flat index of that order
(Reduction.create(), before any loop reordering). A sum over N, H and W of a
channels-last tensor whose H and W are swapped in memory (strides [125440, 1,
160, 4480] for [128, 160, 28, 28], the residual add that better-benchmark's
sum_9683f50f2dcf returns from a ConvNeXtV2 block) then reads each chunk of 128
(n, h, w) positions across W, the outer of the two in memory:

    for tile_r in hl.tile(128):
        tmp0 = in0[160*(((tile_r.index + 128*(tile_x.index // 160)) // 28) % 28)
                   + 4480*((tile_r.index + 128*(tile_x.index // 160)) % 28)
                   + 125440*((tile_r.index + 128*(tile_x.index // 160)) // 784)
                   + tile_x.index % 160]

whose divisions and remainders keep the loads from being vectors, and whose
access the add that writes the tensor cannot share, so the add keeps a kernel
of its own (reindex_onto_reduction() declines it). Reduced over N, W and H
instead, the same values are a [100352, 160] view, read as tiles,

    for tile_r in hl.tile(128):
        tmp0 = in0[tile_y, tile_r, tile_x]

and the add's loops follow them into one kernel that stores the sum and
reduces it. The reduction's result is the same up to the order of its adds:
only the reduced dims trade places, and no kept dim moves."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import torch
from torch._inductor import config as inductor_config
from torch._inductor import lowering as inductor_lowering
from torch._inductor.ir import TensorBox
from torch._inductor.virtualized import V
from torch.fx.operator_schemas import normalize_function

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

aten = torch.ops.aten

# The reductions whose results the order of the reduced dims does not change
# (but for the order of the adds of a sum, a mean or a variance). Not argmax
# and argmin, whose indices count the reduced elements in dim order.
REDUCTIONS = (
    aten.sum.dim_IntList,
    aten.mean.dim,
    aten.amax.default,
    aten.amin.default,
    aten.any.dims,
    aten.var.correction,
    aten.var_mean.correction,
)


def memory_order_permutation(
    value: torch.Tensor,
    dims: Sequence[int] | None,
    *,
    empty_reduces_all: bool = True,
) -> list[int] | None:
    """The permutation of `value`'s dims that puts its reduced `dims` (all of
    them for None, and for [] where `empty_reduces_all`) in the order of their
    strides, largest first, at the positions the reduced dims hold, every other
    dim in place: [0, 1, 3, 2] for the [128, 160, 28, 28] reduced over [0, 2,
    3] with strides [125440, 1, 160, 4480]. None when they are in that order
    already (dims of one element do not count), when a size or stride is not a
    constant, for [] where it reduces nothing (aten.any.dims keeps every dim
    for an empty list, where sum, mean, amax, amin and var reduce them all),
    and for a 0-dim tensor, which torch.sum(x, 0) reduces too (and
    `d % value.dim()` would divide by zero)."""
    if (dims is not None and not dims and not empty_reduces_all) or (
        value.dim() == 0
        or not all(isinstance(n, int) for n in (*value.shape, *value.stride()))
    ):
        return None
    reduced = sorted({d % value.dim() for d in dims} if dims else range(value.dim()))
    positions = [d for d in reduced if value.shape[d] != 1]
    ordered = sorted(positions, key=lambda d: -value.stride(d))
    if ordered == positions:
        return None
    permutation = list(range(value.dim()))
    for position, dim in zip(positions, ordered, strict=True):
        permutation[position] = dim
    return permutation


def reduced_input_permutation(op: torch._ops.OpOverload, x: object) -> list[int] | None:
    """memory_order_permutation() of the reduction V.graph.current_node
    lowers, with cuda_backend "helion" on a CUDA device, for an input whose
    layout is known at lowering time: a graph input, or a graph output, whose
    strides Inductor gives the eager ones (the node's "val"). None for any
    other input, an intermediate whose layout Inductor picks later, and
    wherever the op is not the node's."""
    node = V.graph.current_node
    if not (
        inductor_config.cuda_backend == "helion"
        and isinstance(x, TensorBox)
        and (device := x.get_device()) is not None
        and device.type == "cuda"
        and node is not None
        and node.target is op
    ):
        return None
    normalized = normalize_function(
        op, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )
    if normalized is None:
        return None
    source = normalized.kwargs.get("self", normalized.kwargs.get("input"))
    dims = normalized.kwargs.get("dim")
    if isinstance(dims, int):
        dims = [dims]
    if not (
        isinstance(source, torch.fx.Node)
        and isinstance(value := source.meta.get("val"), torch.Tensor)
        and (dims is None or isinstance(dims, (list, tuple)))
        and (
            source.op == "placeholder"
            or any(user.op == "output" for user in source.users)
        )
    ):
        return None
    return memory_order_permutation(
        value, dims, empty_reduces_all=op is not aten.any.dims
    )


def reduce_in_memory_order(
    op: torch._ops.OpOverload, replaced: Callable[..., Any] | None
) -> Callable[..., Any]:
    """The user lowering of `op`: the replaced entry's (else Inductor's)
    lowering of `op` over the input permuted by reduced_input_permutation(),
    or of the input as it is (the replaced entry's, else None)."""

    def lowering(x: object, *args: object, **kwargs: object) -> object:
        permutation = reduced_input_permutation(op, x)
        if permutation is not None:
            x = inductor_lowering.permute(x, permutation)
        out = replaced(x, *args, **kwargs) if replaced is not None else None
        if out is None and permutation is not None:
            out = inductor_lowering.lowerings[op](x, *args, **kwargs)
        return out

    return lowering


def register_reduction_order_lowerings() -> None:
    """Register reduce_in_memory_order() for every op of REDUCTIONS, once:
    registering again (helion.inductor._register() runs again in tests) keeps
    the one wrapper, where wrapping it would permute the input twice, which
    undoes a swap of two dims (the [0, 1, 3, 2] of an H/W-swapped tensor)."""
    for op in REDUCTIONS:
        current = inductor_lowering.user_lowerings.get(op)
        if getattr(current, _REDUCTION_ORDER_WRAPPER, False):
            continue
        lowering = reduce_in_memory_order(op, current)
        setattr(lowering, _REDUCTION_ORDER_WRAPPER, True)
        inductor_lowering.user_lowerings[op] = lowering


# Marks the user lowerings register_reduction_order_lowerings() installed.
_REDUCTION_ORDER_WRAPPER = "_helion_reduction_order"
