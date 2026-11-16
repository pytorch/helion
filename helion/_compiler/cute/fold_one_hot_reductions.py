"""Fold the sum of a one-hot selection into closed form.

Role: device-IR pass (``canonicalize_traced_graphs_pass``).

The cross-entropy (NLL) backward builds its gradient from a one-hot row and
sums it::

    grad = torch.where(target[:, None] == hl.arange(n)[None, :], hit, 0.0)
    grad_sum = torch.sum(grad * scale[:, None], dim=-1, keepdim=True)

Every element of the row is ``0.0 * scale`` except the one at ``target``,
``hit * scale``, when ``0 <= target < n``.  Lowered as written, the sum is a
sweep over the whole row that loads nothing: a separate pass of compares,
selects and adds before the pass that needs its value, about a fifth of the
instructions of the CE backward kernels of better-benchmark's BERT and BART
repros.  This pass replaces it with::

    where(hit_in_range, hit * scale + 0.0 * scale, 0.0 * scale) + 0.0

which is the sum bit for bit, case by case of the sweep's terms:

- In range (``0 <= target < n``), one term is ``hit * scale`` and every
  other one ``0.0 * scale``.  Adding a zero to a nonzero value is exact, so
  the sum is the hit term, rounded once as the sweep's product is.
- Out of range (``ignore_index``'s ``-100``, or ``n`` and above), no term
  is hit: every one is ``0.0 * scale``, as is the closed form (the zero and
  NaN cases below).
- When every term is a zero (a zero ``scale`` or hit, ``+0`` or ``-0``, or
  no hit beside a finite ``scale``), the trailing ``+ 0.0`` gives the sum
  the positive sign that eager ``torch.sum``, the sweep's ``+0``
  accumulators and its identity-padded combines give it.  The sign of a
  zero sum is a reduction-order effect, and one order differs: with a
  thread per element of a row of whole warps, the sweep's warp reduction
  of ``-0`` terms (a missing label beside a negative ``scale``) returns
  ``-0``, where the closed form returns eager's ``+0``.
- An infinite or NaN ``scale`` makes ``0.0 * scale`` a NaN, which the
  closed form adds as the sweep adds its terms, so both are NaN.
- ``hit``, the miss and ``scale`` share the sum's dtype, in which both
  forms compute the terms, so the hit term rounds the same in both.

Only a zero miss value qualifies (any other one sums rounding errors), and
``target``, ``hit`` and ``scale`` must be uniform along the reduced dim; the
iota must count ``0, 1, ..., n - 1``, distinct in its dtype, along the whole
reduced dim (a reduction block of static extent ``n``: along a tiled dim the
sum covers one tile).  The range check compares ``target`` in the dtype the
``==`` promotes it to.  A sum that drops the dim computes on operands that
lack it (``scale`` for ``scale[:, None]``, 0-d values).
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from ..fx_side_effects import is_impure_node
from ..merge_reduction_dims import _source_meta
from ..utils import is_full_slice
from .analysis.match_common import binary_pair
from .analysis.match_common import is_call
from .analysis.match_common import node_tensor
from .analysis.match_common import scalar_value

if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch.fx.node import Argument

    from ..device_ir import DeviceIR

_SUM = torch.ops.aten.sum.dim_IntList
_MUL = torch.ops.aten.mul.Tensor
_WHERE = torch.ops.aten.where.self
_EQ = torch.ops.aten.eq.Tensor
_IOTA = torch.ops.prims.iota.default
_SCALAR = torch.ops.aten.scalar_tensor.default


def _uniform_along(node: object, dim: int, rank: int) -> bool:
    """Whether ``node`` broadcasts along ``dim`` of a rank-``rank`` operand: a
    0-d tensor, or a tensor of that rank with size 1 there."""
    value = node_tensor(node)
    if value is None:
        return False
    if value.ndim == 0:
        return True
    shape = value.shape
    return value.ndim == rank and isinstance(shape[dim], int) and shape[dim] == 1


def _zero_scalar(node: object) -> bool:
    return is_call(node, _SCALAR) and scalar_value(node) == 0


def _reduced_extent(size: object) -> int | None:
    """The static extent of the reduced dim of size ``size`` when it is a
    whole-row reduction dim (a reduction block), else ``None``: along a tiled
    dim the sum covers one tile, whose partial last tile the summand's mask
    zeroes, and its iota counts within the tile."""
    if not isinstance(size, torch.SymInt):
        return None
    env = CompileEnvironment.current()
    block_id = env.get_block_id(size)
    if block_id is None:
        return None
    info = env.block_sizes[block_id]
    if not info.reduction or not isinstance(info.size, int):
        return None
    return info.size


def _counting_iota(node: object, dim: int, rank: int, size: object) -> int | None:
    """The static length of the ``iota`` (``0, 1, ...``) that ``node`` views
    along ``dim`` alone (size 1 along the other dims), counting the whole
    reduced dim of size ``size``; ``None`` for anything else."""
    view = binary_pair(node, (view_ops.subscript,))
    if view is None:
        return None
    iota, indices = view
    if (
        not is_call(iota, _IOTA)
        or not isinstance(indices, (list, tuple))
        or len(indices) != rank
        or not all(
            is_full_slice(index) if position == dim else index is None
            for position, index in enumerate(indices)
        )
    ):
        return None
    assert isinstance(iota, torch.fx.Node)
    extent = _reduced_extent(size)
    if extent is None:
        return None
    assert isinstance(size, torch.SymInt)
    length = iota.args[0] if len(iota.args) == 1 else None
    if isinstance(length, torch.fx.Node):
        # ``hl.arange(x.size(1))`` counts to the row's reduction dim.
        length = length.meta.get("val")
    if isinstance(length, torch.SymInt):
        known = CompileEnvironment.current().known_equal(length, size)
        length = extent if known else None
    value = node_tensor(iota)
    if (
        length != extent
        or iota.kwargs.get("start", 0) != 0
        or iota.kwargs.get("step", 1) != 1
        or value is None
        or value.dtype.is_floating_point
        or value.dtype is torch.bool
        or value.ndim != 1
        # The values must be distinct: an iota past its dtype's range wraps.
        or extent - 1 > torch.iinfo(value.dtype).max
    ):
        return None
    return extent


def _one_hot_product(
    summand: torch.fx.Node, dim: int, rank: int, size: object
) -> (
    tuple[
        int,
        torch.dtype,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node | None,
    ]
    | None
):
    """``(n, dtype, target, hit, miss, scale)`` of ``where(target == iota(n),
    hit, miss) * scale`` (``scale`` ``None`` without the product), with the
    dtype the ``==`` compares in."""
    selection, scale = summand, None
    product = binary_pair(summand, (_MUL,))
    if product is not None:
        for candidate, other in (product, product[::-1]):
            if is_call(candidate, _WHERE) and _uniform_along(other, dim, rank):
                selection, scale = candidate, other
                break
        else:
            return None
    if not is_call(selection, _WHERE):
        return None
    assert isinstance(selection, torch.fx.Node)
    if len(selection.args) != 3 or selection.kwargs:
        return None
    condition, hit, miss = selection.args
    equal = binary_pair(condition, (_EQ,))
    if equal is None:
        return None
    for target, iota in (equal, equal[::-1]):
        target_value = node_tensor(target)
        if (
            target_value is not None
            and not target_value.dtype.is_floating_point
            and target_value.dtype is not torch.bool
            and _uniform_along(target, dim, rank)
            and (extent := _counting_iota(iota, dim, rank, size)) is not None
        ):
            break
    else:
        return None
    if not _zero_scalar(miss) or not _uniform_along(hit, dim, rank):
        return None
    assert isinstance(target, torch.fx.Node)
    assert isinstance(hit, torch.fx.Node) and isinstance(miss, torch.fx.Node)
    dtype = torch.result_type(
        cast("torch.Tensor", target_value), cast("torch.Tensor", node_tensor(iota))
    )
    return extent, dtype, target, hit, miss, cast("torch.fx.Node | None", scale)


def _without_dim(
    node: torch.fx.Node | None, dim: int, rank: int
) -> torch.fx.Node | None:
    """``node`` with the reduced dim dropped: a 0-d tensor as is, the source
    of a view that inserts the dim (``x[:, None]``), else ``None``."""
    value = node_tensor(node)
    if value is None:
        return None
    if value.ndim == 0:
        return node
    view = binary_pair(node, (view_ops.subscript,))
    if view is None:
        return None
    source, indices = view
    source_value = node_tensor(source)
    if (
        source_value is None
        or source_value.ndim != rank - 1
        or not isinstance(indices, (list, tuple))
        or len(indices) != rank
        or not all(
            index is None if position == dim else is_full_slice(index)
            for position, index in enumerate(indices)
        )
    ):
        return None
    return cast("torch.fx.Node", source)


def _fold(graph: torch.fx.Graph, reduction: torch.fx.Node) -> int | None:
    """Fold ``reduction`` if it sums a one-hot row; the block id of the
    reduction dim it summed over, ``None`` when it does not fold."""
    if (
        not is_call(reduction, _SUM)
        or len(reduction.args) not in (2, 3)
        or set(reduction.kwargs) - {"keepdim", "dtype"}
    ):
        return None
    summand, dims = reduction.args[:2]
    keepdim = (
        reduction.args[2]
        if len(reduction.args) == 3
        else reduction.kwargs.get("keepdim")
    )
    result_value, summand_value = node_tensor(reduction), node_tensor(summand)
    if (
        keepdim not in (True, False, None)
        or result_value is None
        or summand_value is None
        or not result_value.dtype.is_floating_point
        or summand_value.dtype is not result_value.dtype
        # The sum ``canonicalize_reductions`` makes of an ``hl.reduce``
        # names the dtype it sums in.
        or reduction.kwargs.get("dtype", result_value.dtype) is not result_value.dtype
        or not isinstance(dims, (list, tuple))
        or len(dims) != 1
        or not isinstance(dims[0], int)
    ):
        return None
    rank = summand_value.ndim
    if rank == 0:
        return None
    dim = dims[0] % rank
    assert isinstance(summand, torch.fx.Node)
    # ``hl.reduce`` masks the padding of what it sums to the identity.
    masked = binary_pair(summand, (_tracing_ops._mask_to,))
    if masked is not None:
        if masked[1] != 0:
            return None
        summand = cast("torch.fx.Node", masked[0])
    matched = _one_hot_product(summand, dim, rank, summand_value.shape[dim])
    if matched is None:
        return None
    extent, compare_dtype, target, hit, miss, scale = matched
    if keepdim is not True:
        # The sum drops the reduced dim: compute on the operands' sources,
        # which lack it (``scale`` for ``scale[:, None]``).
        sources = [
            _without_dim(operand, dim, rank) for operand in (target, hit, miss, scale)
        ]
        if any(source is None for source in sources[:3]) or (
            scale is not None and sources[3] is None
        ):
            return None
        target, hit, miss = cast("list[torch.fx.Node]", sources[:3])
        scale = sources[3]
    values = [
        node_tensor(operand)
        for operand in ((hit, miss) if scale is None else (hit, miss, scale))
    ]
    if any(value is None or value.dtype is not result_value.dtype for value in values):
        return None
    target_value = cast("torch.Tensor", node_tensor(target))
    hit_value, miss_value, *scale_value = cast("list[torch.Tensor]", values)
    hit_shape, miss_shape = hit_value.shape, miss_value.shape
    if scale_value:
        hit_shape = torch.broadcast_shapes(hit_shape, scale_value[0].shape)
        miss_shape = torch.broadcast_shapes(miss_shape, scale_value[0].shape)
    sum_shape = torch.broadcast_shapes(hit_shape, miss_shape)
    shape = torch.broadcast_shapes(target_value.shape, sum_shape)
    env = CompileEnvironment.current()
    if len(shape) != result_value.ndim or not all(
        starmap(env.known_equal, zip(shape, result_value.shape, strict=True))
    ):
        return None

    def emit(
        function: OpOverload,
        args: tuple[Argument, ...],
        value: torch.Tensor,
        **kwargs: Argument,
    ) -> torch.fx.Node:
        node = graph.call_function(function, args, kwargs)
        node.meta = {
            **_source_meta(reduction.meta),
            "val": value,
            "original_aten": function,
        }
        return node

    dtype = result_value.dtype
    with graph.inserting_before(reduction):
        if target_value.dtype is not compare_dtype:
            # The ``==`` compares in its promoted dtype (int8 labels beside an
            # int64 iota as int64): a narrower label dtype would wrap the
            # extent (``int8(1000)`` is -24).
            target_value = target_value.new_empty(
                target_value.shape, dtype=compare_dtype
            )
            target = emit(
                torch.ops.prims.convert_element_type.default,
                (target, compare_dtype),
                target_value,
            )
        flag = target_value.new_empty(target_value.shape, dtype=torch.bool)
        # ``extent - 1`` fits the iota's dtype (``_counting_iota``) and hence
        # the compare dtype; ``extent`` itself may not.
        in_range = emit(
            torch.ops.aten.bitwise_and.Tensor,
            (
                emit(torch.ops.aten.ge.Scalar, (target, 0), flag),
                emit(torch.ops.aten.le.Scalar, (target, extent - 1), flag),
            ),
            flag,
        )
        hit_term, miss_term = hit, miss
        if scale_value:
            hit_term = emit(
                _MUL, (hit, scale), result_value.new_empty(hit_shape, dtype=dtype)
            )
            miss_term = emit(
                _MUL, (miss, scale), result_value.new_empty(miss_shape, dtype=dtype)
            )
        if extent > 1:
            # The row holds misses beside the hit: their NaN, if any, too.
            hit_term = emit(
                torch.ops.aten.add.Tensor,
                (hit_term, miss_term),
                result_value.new_empty(sum_shape, dtype=dtype),
            )
        selected = emit(
            _WHERE,
            (in_range, hit_term, miss_term),
            result_value.new_empty(shape, dtype=dtype),
        )
        # In the sum's dtype, as the traced scalars are: a float32 zero would
        # promote what follows and skip the rounding a half sum takes.
        zero = emit(
            _SCALAR,
            (0.0,),
            result_value.new_empty((), dtype=dtype),
            dtype=dtype,
            layout=torch.strided,
            device=result_value.device,
        )
        folded = emit(
            torch.ops.aten.add.Tensor,
            (selected, zero),
            result_value.new_empty(result_value.shape, dtype=dtype),
        )
    reduction.replace_all_uses_with(folded)
    graph.erase_node(reduction)
    block_id = env.get_block_id(summand_value.shape[dim])
    assert block_id is not None
    return block_id


def fold_one_hot_reductions(device_ir: DeviceIR) -> int:
    """Fold every matching sum of every graph; the number folded.

    The dims they summed over become ``retirable_reduction_block_ids``: a sum
    that was its dim's only user leaves the dim unused, and the lowering
    retires it, as it does merged dims, so that it gets no thread axis and
    no reduction knobs.  ``folded_one_hot_sums`` counts the folds for the
    seed heuristics."""
    block_ids = []
    for graph_info in device_ir.graphs:
        folded = [
            block_id
            for node in list(graph_info.graph.nodes)
            if (block_id := _fold(graph_info.graph, node)) is not None
        ]
        if folded:
            # The summands may have fed nothing but the sums.
            graph_info.graph.eliminate_dead_code(is_impure_node=is_impure_node)
        block_ids.extend(folded)
    device_ir.retirable_reduction_block_ids.update(block_ids)
    device_ir.folded_one_hot_sums += len(block_ids)
    return len(block_ids)
