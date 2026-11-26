"""Partition a complete, thread-invariant output sweep after an FP32 row sum."""

from __future__ import annotations

import ast
from math import prod
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..reduction_strategy import LoopedReductionStrategy
from ..runtime_tensor_guards import tensor_pairs_with_proven_disjoint_storage
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import names_read
from ._ast_pass_utils import parse_expr
from ._ast_pass_utils import same_expr
from .parallel_row_sum import parallel_sum
from .thread_budget import linear_thread_index
from .warp_row_partition import prove_row_sweep
from .warp_row_partition import pure_row_expression
from .warp_row_partition import sweep_tensor_base

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState

KEY = "cute_single_row_publication"

# The scalar proof accepts arithmetic, numeric casts and rsqrt. Use the
# pointwise part of a decomposed mean as well (its actual reduction must
# independently be an FP32 sum). An unknown math op is not a seed witness.
_ROW_POINTWISE = frozenset(
    (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.neg.default,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.mean.dim,
        torch.ops.prims.convert_element_type.default,
    )
)


def row_pointwise_witness(graph: torch.fx.Graph) -> bool:
    return all(
        not isinstance(node.meta.get("lowering"), PointwiseLowering)
        or node.target in _ROW_POINTWISE
        for node in graph.nodes
    )


def publication_roots(
    env: CompileEnvironment, ir: DeviceIR
) -> dict[int, tuple[int, int]]:
    """A single logical row, one full-slice output and a single FP32 statistic."""
    facts: dict[int, tuple[int, int]] = {}
    if (
        len(ir.root_ids) < 2
        or ir.host_function is None
        or env.index_dtype is not torch.int32
    ):
        return facts
    with ir.host_function:
        for root, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True):
            if (
                len(grid) != 1
                or type(env.block_sizes[grid[0]].size) is not int
                or env.block_sizes[grid[0]].size != 1
            ):
                continue
            graph = ir.graphs[root].graph
            if not row_pointwise_witness(graph):
                continue
            ns = list(graph.nodes)
            stores = [n for n in ns if n.target is memory_ops.store]
            reds = [
                (n, r)
                for n in ns
                if isinstance(r := n.meta.get("lowering"), ReductionLowering)
            ]
            if len(reds) != 1 or len(stores) != 1 or stores[0] is not ns[-2]:
                continue
            ((node, red),) = reds
            target, indices = stores[0].args[:2]
            if (
                not isinstance(target, torch.fx.Node)
                or not isinstance(indices, (tuple, list))
                or len(indices) != 2
                or indices[-1] != slice(None)
            ):
                continue
            # An in-place output is already known to share its input. Distinct
            # host views still need the address-span proof at code generation.
            if any(
                n.target in (memory_ops.load, view_ops.subscript)
                and isinstance(src := n.args[0], torch.fx.Node)
                and (
                    src is target
                    or (
                        target.target is _tracing_ops._host_tensor
                        and src.target is target.target
                        and src.args == target.args
                    )
                )
                for n in ns
            ):
                continue
            stat = node.meta.get("val")
            arr = target.meta.get("val")
            size = env.block_sizes[red.block_index].size
            if not (
                red.reduction_type == "sum"
                and isinstance(stat, torch.Tensor)
                and stat.dtype is torch.float32
                and isinstance(arr, torch.Tensor)
                and arr.ndim == 2
                and isinstance(size, (int, torch.SymInt))
            ):
                continue
            stride = arr.stride(-1)
            if type(stride) is int and stride != 1:
                continue
            n = env.static_kernel_tensor_extent(size)
            if not (
                n
                and n >= 32
                and n % 32 == 0
                and env.static_kernel_tensor_extent(arr.shape[0]) == 1
                and env.static_kernel_tensor_extent(arr.shape[-1]) == n
            ):
                continue
            # Only the output must be disjoint from each input. Distinct
            # read-only inputs can alias freely, so never compare them.
            if target.target is _tracing_ops._host_tensor:
                direct = {
                    arg
                    for load in ns
                    if load.target is memory_ops.load
                    and isinstance(arg := load.args[0], torch.fx.Node)
                    and arg.target is _tracing_ops._host_tensor
                    and isinstance(arg.meta.get("val"), torch.Tensor)
                }
                if any(
                    not tensor_pairs_with_proven_disjoint_storage(
                        env,
                        {target: arr, arg: arg.meta["val"]},
                        allow_unbound=True,
                    )
                    for arg in direct
                ):
                    continue
            facts[root] = (red.block_index, n)
    return facts


def partition_row_publication(
    cg: GenerateAST, grid: DeviceGridState, body: list[ast.AST]
) -> None:
    fn = cg.device_function
    if not fn.config.get(KEY) or (root := cg.current_root_graph_info) is None:
        return
    env = CompileEnvironment.current()
    pair = env.config_spec.cute_single_row_publication_roots.get(root.graph_id)
    if pair is None or len(body) < 4:
        return
    # Ordinary literal-stride metadata is not a tensor data read. Any
    # remaining subscript (gather, register view, tensor[index], etc.) lies
    # outside the pointer-load alias inventory, so keep the original writers.
    for stmt in body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Subscript) and not (
                isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "stride"
                and isinstance(layout := sub.value.value, ast.Attribute)
                and layout.attr == "layout"
                and isinstance(layout.value, ast.Name)
                and isinstance(sub.slice, ast.Constant)
                and type(sub.slice.value) is int
            ):
                return
    axis, n = pair
    strategy = fn.tile_strategy.block_id_to_strategy.get((axis,))
    dims = fn.tile_strategy.thread_block_dims()
    threads = prod(dims)
    needed, setup, spliced = grid._live_lane_setup(body)
    if (
        not isinstance(strategy, LoopedReductionStrategy)
        or strategy._thread_count != 1
        or not 32 <= threads <= 1024
        or n % threads
        or grid.outer_prefix
        or grid.outer_suffix
        or grid.body_finalizers
        or grid.vec_lane_wrappers
        or grid.deferred_vector_ops
        or setup
        or spliced
        or any(v in needed for v, _ in grid.lane_loops)
    ):
        return
    # A native bind can materialize a static symbolic extent before the
    # accumulator. It remains a uniform statement outside the rewritten sweep.
    # Do not skip an intrinsic, memory effect, or a thread-dependent value.
    prefix: set[str] = set()
    for stmt in body:
        if not (
            isinstance(stmt, ast.Assign)
            and (name := assigned_name(stmt))
            and isinstance(stmt.value, ast.Constant)
            and type(stmt.value.value) in (int, float)
        ):
            break
        if name in prefix or any(name == arg.name for arg in fn.arguments):
            return
        prefix.add(name)
    region = body[len(prefix) :]
    if len(region) < 4:
        return
    reduce, publish = region[1], region[-1]
    if not isinstance(reduce, ast.For) or not isinstance(publish, ast.For):
        return
    reads: set[str] = set()
    sw = prove_row_sweep(
        region[0],
        reduce,
        region[2],
        strategy,
        n,
        reads,
        # The original serial loops must cover the whole row. Both replacement
        # walks use the independent physical thread coordinate, so the old
        # inner chunk need not itself fill a warp.
        chunk_multiple=1,
    )
    if (
        sw is None
        or not same_expr(reduce.iter, publish.iter)
        or not same_expr(reduce.target, publish.target)
        or len(publish.body) != 1
        or publish.orelse
    ):
        return
    inner = publish.body[0]
    if not (
        isinstance(inner, ast.For)
        and same_expr(inner.iter, sw.inner.iter)
        and same_expr(inner.target, sw.inner.target)
        and not inner.orelse
        and len(inner.body) >= 2
        and same_expr(inner.body[0], sw.index)
    ):
        return
    # The statistic may read only uniform kernel arguments and the current
    # logical coordinate. No value left over from another lane or phase can
    # affect the value published by its new owner.
    uniform = {a.name for a in fn.arguments} | {"cute", "cutlass"} | prefix
    kindex = strategy.index_var(axis)
    acc = assigned_name(region[0])
    statistic = assigned_name(region[2])
    if acc is None or statistic is None:
        return

    def pure_assigns(
        seq: Sequence[ast.AST], defined: set[str], protected: set[str]
    ) -> bool:
        for stmt in seq:
            if (
                not isinstance(stmt, ast.Assign)
                or not (lhs := assigned_name(stmt))
                or lhs in protected
                or not pure_row_expression(stmt.value, reads)
                or not names_read(stmt.value) <= defined
            ):
                return False
            defined.add(lhs)
        return True

    if not pure_assigns(sw.inner.body[1:], uniform | {kindex, acc}, uniform | {kindex}):
        return
    known = uniform | {statistic}
    if not pure_assigns(region[3:-1], known, uniform | {statistic}):
        return
    # An update to a value defined before the publication loop can carry
    # state between columns. Each new owner must see the same immutable inputs.
    entry = known | {kindex}
    defined = entry.copy()
    if not pure_assigns(inner.body[1:-1], defined, entry):
        return
    terminal = inner.body[-1]
    if not (
        isinstance(terminal, ast.Expr)
        and isinstance(store := terminal.value, ast.Call)
        and (dst := sweep_tensor_base(store, "store"))
        and isinstance(store.func, ast.Attribute)
        and len(store.args) == 1
        and not store.keywords
        and pure_row_expression(store.args[0], reads)
    ):
        return
    if not names_read(store.args[0]) <= defined:
        return
    # Exact materialized [0,k] address, with a guarded unit trailing stride.
    # This rules out broadcasting, scatter, and internally overlapping stores.
    expected = parse_expr(
        f"{dst}.iterator + cutlass.Int32(0) * cutlass.Int32({dst}.layout.stride[0]) + cutlass.Int32({kindex}) * cutlass.Int32({dst}.layout.stride[1])"
    )
    if (
        not same_expr(expected, store.func.value)
        or fn.proven_tensor_stride_values().get((dst, 1)) != 1
    ):
        return
    disjoint = fn.proven_disjoint_tensor_pairs()
    if any(frozenset((dst, src)) not in disjoint for src in reads):
        return
    tid = fn.new_var("publication_thread")
    step = fn.new_var("publication_step")
    # Keep the signed index cast and the plain literal strides in the
    # established, source-keyed publication lowering.
    coordinate = linear_thread_index(
        {axis: size for axis, size in enumerate(dims) if axis == 0 or size > 1},
        lambda x: f"cutlass.Int32({x})",
        cast_strides=False,
    )
    coordinate_setup, loop = ast.parse(
        f"{tid} = {coordinate}\nfor {step} in range({n // threads}):\n    {kindex} = {tid} + cutlass.Int32({step}) * {threads}"
    ).body
    assert isinstance(loop, ast.For)
    loop.body.extend(inner.body[1:])
    if fn.config.get("cute_parallel_row_sum"):
        parallel = parallel_sum(
            fn, coordinate_setup, loop, acc, statistic, sw.inner.body[1:], threads
        )
        if parallel is None:
            return
        # Only replace the K coordinate and the final FP32 addition tree.
        # In particular the original per-element operations and all later
        # casts, normalization, weighting and a single store stay in place.
        if fn.config.get("cute_packed_row_publication"):
            from .packed_row_publication import install_paired_sweeps
            from .packed_row_publication import paired_sweeps

            plans = paired_sweeps(
                fn,
                parallel,
                region[3:-1],
                loop,
                acc,
                statistic,
                dst,
                sw.inner.body[1:],
                n,
                threads,
                kindex,
                tid,
            )
            if plans is not None:
                install_paired_sweeps(fn, grid, [*parallel, *region[3:-1], loop], plans)
        body[len(prefix) + 1 : len(prefix) + 3] = parallel
        body[-1:] = [loop]
    else:
        body[-1:] = [coordinate_setup, loop]
    fn.cute_state.lanes.single_row_publication_dims.add(dims)


def validate_publication_launch(
    fn: DeviceFunction, planned: tuple[int, int, int] | None
) -> None:
    """The full set of owners must match the literal block in the host launcher."""
    state = fn.cute_state.lanes
    if (
        fn.config.get("cute_packed_row_publication")
        and not state.packed_row_publication
    ):
        raise exc.InvalidConfig(
            "packed row publication requires matching BF16 input, statistic and output sweeps"
        )
    if not state.single_row_publication_dims:
        raise exc.InvalidConfig(f"{KEY} needs a proved independent full-row store")
    if (
        planned is None
        or state.single_row_publication_dims != {planned}
        or state.launch_block != planned
    ):
        raise exc.InvalidConfig(f"{KEY} needs the proved full CTA")
