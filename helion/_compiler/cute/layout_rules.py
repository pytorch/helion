"""Per-operation rules for preferred CuTe TV layouts.

Each rule inspects a torch.fx.Node and its metadata to determine the
ideal thread-value layout.  These preferences are later reconciled by
the propagation pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import reduce_ops
from ..compile_environment import CompileEnvironment
from .layout import LayoutConstraint
from .layout import LayoutTag
from .layout import ThreadLayout

if TYPE_CHECKING:
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

    SymIntLike = torch.SymInt | int

_SYMBOLIC_STRIDE_SORT_KEY = 2**62


def _clamp_threads(size: int, num_threads: int) -> int:
    """Clamp *num_threads* to the largest value <= size that divides *size*.

    num_threads is always a power of 2, so halving finds the answer quickly.
    """
    num_threads = min(num_threads, size)
    while num_threads > 1 and size % num_threads != 0:
        num_threads //= 2
    return num_threads


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preferred_constraint_for_node(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> LayoutConstraint | None:
    """Return a LayoutConstraint for *node*, or None if unconstrained.

    This only fills in the *preferred* field; the propagation pass
    resolves the final *layout* field later.
    """
    if node.op != "call_function":
        return None

    if node.target is memory_ops.load:
        return _constraint_for_load(node, tile_strategy)
    if node.target is memory_ops.store:
        return _constraint_for_store(node, tile_strategy)
    if node.target is reduce_ops._reduce:
        return _constraint_for_reduce(node, tile_strategy)
    if _tracing_ops.is_for_loop_target(node.target):
        return None  # control flow -- handled at graph level
    # Pointwise / aten ops: unconstrained, inherit from inputs
    return None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _constraint_for_load(
    node: torch.fx.Node,
    tile_strategy: TileStrategyDispatch,
) -> LayoutConstraint | None:
    """Loads prefer coalesced access: threads along the contiguous dim."""
    layout = _layout_from_tensor_strides(
        node, tile_strategy=tile_strategy, tag=LayoutTag.COALESCED
    )
    if layout is None:
        return None
    return LayoutConstraint(preferred=layout)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _constraint_for_store(
    node: torch.fx.Node,
    tile_strategy: TileStrategyDispatch,
) -> LayoutConstraint | None:
    """Stores also prefer coalesced access."""
    layout = _layout_from_tensor_strides(
        node, tile_strategy=tile_strategy, tag=LayoutTag.COALESCED
    )
    if layout is None:
        return None
    return LayoutConstraint(preferred=layout)


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def _constraint_for_reduce(
    node: torch.fx.Node,
    tile_strategy: TileStrategyDispatch,
) -> LayoutConstraint | None:
    """Reductions prefer threads along the reduction axis.

    node.args for _reduce:
      (combine_graph_id, input_tensor, dim, ...)
    """
    if len(node.args) < 3:
        return None
    dim = node.args[2]
    if not isinstance(dim, int):
        return None

    # We need the shape of the *input* to the reduction, which is args[1]
    input_node = node.args[1]
    if isinstance(input_node, torch.fx.Node):
        input_val = input_node.meta.get("val")
    elif isinstance(input_node, torch.Tensor):
        input_val = input_node
    else:
        return None

    if not isinstance(input_val, torch.Tensor):
        return None

    ndim = input_val.ndim
    if ndim == 0:
        return None

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        return None

    shape = input_val.shape
    # Compute total elements and try to put threads along reduction dim
    reduce_size = shape[dim]
    env = CompileEnvironment.current()
    num_threads = _reduction_num_threads(env, reduce_size, tile_strategy)

    # Zero-sized reduction dim: nothing to lay out — skip.
    if isinstance(reduce_size, int) and reduce_size == 0:
        return None

    # Clamp threads to largest divisor of reduce_size
    if isinstance(reduce_size, int) and isinstance(num_threads, int):
        num_threads = _clamp_threads(reduce_size, num_threads)

    # Build a simple layout: threads along dim, values along other dims
    # For a 2D (M, K) reduce(dim=1):
    #   thr along K (reduction), val along M
    if ndim == 1:
        layout = ThreadLayout.make_1d(
            reduce_size, num_threads=num_threads, tag=LayoutTag.REDUCTION
        )
    elif ndim == 2:
        if dim == 1:
            # Threads along cols (reduction), values along rows
            layout = ThreadLayout.make_row_major(
                shape[0], shape[1], num_threads=num_threads, tag=LayoutTag.REDUCTION
            )
        else:
            # Threads along rows (reduction), values along cols
            layout = ThreadLayout.make_col_major(
                shape[0], shape[1], num_threads=num_threads, tag=LayoutTag.REDUCTION
            )
    else:
        # Higher rank: fall back to simple 1D over reduction dim
        layout = ThreadLayout.make_1d(
            reduce_size, num_threads=num_threads, tag=LayoutTag.REDUCTION
        )

    return LayoutConstraint(preferred=layout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _total_threads_from_strategy(
    tile_strategy: TileStrategyDispatch,
) -> int:
    """Compute total thread count from the tile strategy's block dims.

    This is the product of the CUDA thread block dimensions and represents
    the actual number of threads that will execute each instruction.
    """
    dims = tile_strategy.thread_block_dims()
    return dims[0] * dims[1] * dims[2]


def _reduction_num_threads(
    env: CompileEnvironment,
    reduce_size: SymIntLike,
    tile_strategy: TileStrategyDispatch,
) -> int:
    """Get thread count for a reduction from the tile strategy.

    If a ``PersistentReductionStrategy`` or ``LoopedReductionStrategy``
    exists for the reduction's block dimension, use its pre-computed
    thread count.  Otherwise fall back to the total thread count.
    """
    from ..reduction_strategy import ReductionStrategy

    block_id = env.get_block_id(reduce_size)
    if block_id is not None:
        strategy = tile_strategy.block_id_to_strategy.get((block_id,))
        if isinstance(strategy, ReductionStrategy):
            tc = strategy._reduction_thread_count()
            if tc > 0:
                return tc
    return _total_threads_from_strategy(tile_strategy)


def _resolve_subscript_val(k: object) -> object:
    """Resolve a subscript element to its runtime value.

    In the FX graph, subscript elements are often torch.fx.Node objects.
    We need to look at their ``meta["val"]`` to get the actual
    SymInt / Tensor value for analysis.
    """
    if isinstance(k, torch.fx.Node):
        return k.meta.get("val")
    return k


def _is_active_tile_dim(val: object, env: CompileEnvironment) -> bool:
    """Return True if *val* represents an active tile dimension.

    Active tile dimensions are either:
    - A tensor (e.g. from ``tile.index``)
    - A SymInt that maps to a block_id
    """
    if isinstance(val, torch.Tensor):
        return True
    if isinstance(val, torch.SymInt):
        return env.get_block_id(val) is not None
    return False


def _layout_from_tensor_strides(
    node: torch.fx.Node,
    *,
    tile_strategy: TileStrategyDispatch,
    tag: LayoutTag,
) -> ThreadLayout | None:
    """Build a TV layout that coalesces access for the tensor at node.args[0].

    Uses the tensor's strides from the fake value to determine which
    dimension is contiguous.
    """
    # For load: args = (tensor_node, subscript_list, ...)
    # For store: args = (tensor_node, subscript_list, value_node, ...)
    if len(node.args) < 2:
        return None

    # The fake tensor value is attached to args[0]
    tensor_arg = node.args[0]
    if isinstance(tensor_arg, torch.fx.Node):
        fake_tensor = tensor_arg.meta.get("val")
    elif isinstance(tensor_arg, torch.Tensor):
        fake_tensor = tensor_arg
    else:
        return None

    if not isinstance(fake_tensor, torch.Tensor) or fake_tensor.ndim == 0:
        return None

    # The subscript is a list of FX nodes (or literals).
    subscript = node.args[1]
    if not isinstance(subscript, (list, tuple)):
        return None

    env = CompileEnvironment.current()
    strides = fake_tensor.stride()
    shape = fake_tensor.shape

    # Walk subscript, resolve each element, and collect active tile dims
    # with their tensor dimension index, size, and stride.
    dim_info: list[tuple[int, SymIntLike, SymIntLike]] = []
    tensor_dim = 0
    for k_raw in subscript:
        k = _resolve_subscript_val(k_raw)
        if k is None:
            # None adds a dimension (broadcasting) but doesn't consume a tensor dim
            continue
        if tensor_dim >= len(strides):
            break
        if _is_active_tile_dim(k, env):
            dim_info.append((tensor_dim, shape[tensor_dim], strides[tensor_dim]))
        tensor_dim += 1

    if not dim_info:
        return None

    # Sort by stride to find the contiguous dim (smallest stride = best for
    # coalescing).  Symbolic strides are unknown at compile time but the true
    # contiguous dimension always has a concrete stride of 1, so we push
    # symbolic strides to the end by giving them a large sort key.
    dim_info.sort(
        key=lambda x: x[2] if isinstance(x[2], int) else _SYMBOLIC_STRIDE_SORT_KEY
    )

    num_threads: SymIntLike = _total_threads_from_strategy(tile_strategy)

    if len(dim_info) == 1:
        _, size, _ = dim_info[0]
        if isinstance(size, int) and size == 0:
            return None
        if isinstance(size, int) and isinstance(num_threads, int):
            num_threads = _clamp_threads(size, num_threads)
        return ThreadLayout.make_1d(size, num_threads=num_threads, tag=tag)

    if len(dim_info) >= 2:
        # 2D+ case: threads along the smallest-stride (contiguous) dim
        contiguous_tensor_dim, contiguous_size, _ = dim_info[0]
        other_tensor_dim, other_size, _ = dim_info[1]

        if isinstance(contiguous_size, int) and contiguous_size == 0:
            return None
        if isinstance(contiguous_size, int) and isinstance(num_threads, int):
            num_threads = _clamp_threads(contiguous_size, num_threads)

        if contiguous_tensor_dim > other_tensor_dim:
            # Contiguous dim is a later tensor dimension → row-major
            return ThreadLayout.make_row_major(
                other_size, contiguous_size, num_threads=num_threads, tag=tag
            )
        # Contiguous dim is an earlier tensor dimension → col-major
        return ThreadLayout.make_col_major(
            contiguous_size, other_size, num_threads=num_threads, tag=tag
        )

    return None
