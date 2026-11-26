"""Distribute independent, ordered BF16 column sums across a shared block."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import _compat
from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import ReductionLowering
from ..runtime_tensor_guards import tensor_pairs_with_proven_disjoint_storage
from ._ast_pass_utils import parse_expr
from .autotune.config_keys import ORDERED_COLUMN_PAIRS_KEY
from .loop_nesting import boundary_only_grid_blocks
from .ordered_column_region import ordered_column_region
from .thread_budget import linear_thread_index
from .thread_budget import tile_loop_thread_count

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState


@dataclass(frozen=True)
class _ColumnSum:
    source: torch.Tensor
    destination: torch.Tensor
    column: int
    reduction: int
    slots: int
    width: int


def has_column_sum_structure(env: CompileEnvironment, ir: DeviceIR) -> bool:
    """Keep field order stable when a bound shape disallows the optimization.

    Do not inspect strides, slot counts or pointwise users here; they can
    change across members of a multi-shape search. A later exact match owns
    eligibility for each member and explicit unsupported requests.
    """
    capability = env.config_spec.target_device_capability
    if capability is None or capability[0] != 10 or _compat.is_hip():
        return False
    for rid in ir.root_ids:
        nodes = list(ir.graphs[rid].graph.nodes)
        stores = [n for n in nodes if n.target is memory_ops.store]
        if len(stores) != 1 or stores[0] is not nodes[-2]:
            continue
        dest = stores[0].args[0]
        if (
            not isinstance(dest, torch.fx.Node)
            or not isinstance(out := dest.meta.get("val"), torch.Tensor)
            or out.ndim not in (1, 2)
            or out.dtype != torch.bfloat16
        ):
            continue
        if any(
            n.target is torch.ops.aten.sum.dim_IntList
            and len(n.args) >= 2
            and n.args[1] in ([out.ndim - 1], [-2])
            and isinstance(v := n.meta.get("val"), torch.Tensor)
            and v.ndim == out.ndim
            and v.dtype == torch.float32
            for n in nodes
        ):
            return True
    return False


def _match(
    env: CompileEnvironment, graph: torch.fx.Graph, grid: list[int]
) -> _ColumnSum | None:
    nodes = list(graph.nodes)
    stores = [n for n in nodes if n.target is memory_ops.store]
    if len(stores) != 1 or stores[0] is not nodes[-2]:
        return None
    store = stores[0]
    dst, output_indices, rounded, *rest = store.args
    if rest not in ([], [None]) or store.kwargs:
        return None

    def operation(n: object, op: object, count: int) -> bool:
        return (
            isinstance(n, torch.fx.Node)
            and n.target is op
            and len(n.args) == count
            and not n.kwargs
        )

    cvt = torch.ops.prims.convert_element_type.default
    if not (
        isinstance(rounded, torch.fx.Node)
        and operation(rounded, cvt, 2)
        and rounded.args[1] is torch.bfloat16
        and isinstance(red := rounded.args[0], torch.fx.Node)
        and operation(red, torch.ops.aten.sum.dim_IntList, 2)
        and isinstance(lower := red.meta.get("lowering"), ReductionLowering)
        and lower.reduction_type == "sum"
        and isinstance(widen := red.args[0], torch.fx.Node)
        and operation(widen, cvt, 2)
        and widen.args[1] is torch.float32
        and isinstance(load := widen.args[0], torch.fx.Node)
        and load.target is memory_ops.load
        and not load.kwargs
        and all(arg is None for arg in load.args[2:])
    ):
        return None
    src, indices = load.args[:2]
    if not (
        isinstance(src, torch.fx.Node)
        and isinstance(dst, torch.fx.Node)
        and src.target is dst.target is _tracing_ops._host_tensor
        and src is not dst
        and isinstance(x := src.meta.get("val"), torch.Tensor)
        and isinstance(y := dst.meta.get("val"), torch.Tensor)
        and x.dtype is y.dtype is torch.bfloat16
        and isinstance(indices, (list, tuple))
        and isinstance(output_indices, (list, tuple))
        and len(indices) == x.ndim
        and len(output_indices) == y.ndim == x.ndim - 1
        and len(grid) == y.ndim
        and y.ndim in (1, 2)
        and indices[-2] == slice(None)
        and [*indices[:-2], indices[-1]] == list(output_indices)
        and red.args[1] in ([x.ndim - 2], [-2])
        and all(
            isinstance(n, torch.fx.Node)
            and n.target is _tracing_ops._get_symnode
            and isinstance(sym := n.meta.get("val"), torch.SymInt)
            and env.get_block_id(sym) == axis
            for n, axis in zip(output_indices, grid, strict=True)
        )
    ):
        return None
    # The leading literal singleton (if present) is elided by scalar indexing.
    # A retained batch or an indirect/scattered coordinate needs its own proof.
    if any(
        env.static_kernel_tensor_extent(size) != 1
        for size in (*x.shape[:-2], *y.shape[:-1])
    ):
        return None
    if set(nodes) != {
        src,
        dst,
        load,
        widen,
        red,
        rounded,
        store,
        nodes[-1],
        *output_indices,
    }:
        return None
    width = env.static_kernel_tensor_extent(x.shape[-1])
    slots = env.static_kernel_tensor_extent(x.shape[-2])
    if not (
        width
        and 1 <= width < 2**31
        and slots
        and 2 <= slots <= 16
        and width == env.static_kernel_tensor_extent(y.shape[-1])
        and all(env.static_kernel_tensor_extent(t.stride(-1)) == 1 for t in (x, y))
        and isinstance(
            col_extent := env.block_sizes[grid[-1]].size, (int, torch.SymInt)
        )
        and env.static_kernel_tensor_extent(col_extent) == width
        and isinstance(
            red_extent := env.block_sizes[lower.block_index].size, (int, torch.SymInt)
        )
        and env.static_kernel_tensor_extent(red_extent) == slots
    ):
        return None
    if not tensor_pairs_with_proven_disjoint_storage(
        env, {src: x, dst: y}, allow_unbound=True
    ):
        return None
    return _ColumnSum(x, y, grid[-1], lower.block_index, slots, width)


def column_sum_roots(
    env: CompileEnvironment, ir: DeviceIR
) -> dict[int, tuple[int, int, int, int]]:
    capability = env.config_spec.target_device_capability
    if (
        ir.host_function is None
        or env.index_dtype is not torch.int32
        or capability is None
        or capability[0] != 10
        or _compat.is_hip()
    ):
        return {}
    with ir.host_function:
        return {
            rid: (fact.column, fact.reduction, fact.slots, fact.width)
            for rid, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True)
            if (fact := _match(env, ir.graphs[rid].graph, grid)) is not None
        }


def lower_ordered_columns(
    cg: GenerateAST, grid: DeviceGridState, body: list[ast.AST]
) -> bool:
    fn = cg.device_function
    if (
        not fn.config.get(ORDERED_COLUMN_PAIRS_KEY)
        or (root := cg.current_root_graph_info) is None
    ):
        return False
    env = CompileEnvironment.current()
    if root.graph_id not in env.config_spec.cute_ordered_column_sum_roots:
        return False
    ir = cg.host_function.device_ir
    ids = ir.grid_block_ids[ir.root_ids.index(root.graph_id)]
    fact = _match(env, ir.graphs[root.graph_id].graph, ids)
    if (
        fact is None
        or grid.outer_suffix
        or grid.body_finalizers
        or grid.vec_lane_wrappers
        or grid.deferred_vector_ops
    ):
        return False
    dims = fn.tile_strategy.thread_block_dims()
    threads = prod(dims)
    chunk = fn.resolved_block_size(fact.column)
    if (
        not isinstance(chunk, int)
        or not 1 <= threads <= 1024
        or chunk % (2 * threads)
        or chunk > fact.width
    ):
        return False
    x = fn.tensor_arg(fact.source).name
    y = fn.tensor_arg(fact.destination).name
    if not ordered_column_region(body, grid, fn, fact, x, y, chunk):
        return False
    if frozenset((x, y)) not in fn.proven_disjoint_tensor_pairs():
        return False
    si, so = fact.source.ndim - 1, fact.destination.ndim - 1
    # The typed root has only this immutable direct load/store. Keep the
    # emitted source range/masks and precise scalar arithmetic as a fallback.
    if any(
        not isinstance(s, ast.Assign) or not isinstance(s.value, ast.Name)
        for s in grid.outer_prefix
    ):
        return False
    base = grid.strategy.grid_origin_var(fact.column)
    coordinate = linear_thread_index(
        {i: size for i, size in enumerate(dims) if i == 0 or size > 1},
        lambda v: f"cutlass.Int32({v})",
        cast_strides=False,
    )
    tid, col, slot, lo, hi, word, x0, x1, step = [
        fn.new_var("column_sum_" + suffix)
        for suffix in (
            "thread",
            "column",
            "slot",
            "low",
            "high",
            "word",
            "x",
            "y",
            "step",
        )
    ]
    steps = chunk // (2 * threads)
    # This is a bijection from (sweep, CUDA-linear thread, halfword)
    # to the complete [base,base+chunk) output interval.
    inner = ast.parse(f"""
{col} = {base} + {tid} * 2{f" + cutlass.Int32({step}) * {2 * threads}" if steps > 1 else ""}
{lo} = cutlass.Float32(0)
{hi} = cutlass.Float32(0)
for {slot} in cutlass.range_constexpr({fact.slots}):
    {word} = cute.recast_ptr({x}.iterator + {col} + cutlass.Int32({slot}) * cutlass.Int32({x}.layout.stride[{si - 1}]), dtype=cutlass.Uint32).load()
    {x0}, {x1} = _cute_column_unpack_bf16x2({word})
    {lo}, {hi} = cute.arch.add_packed_f32x2(({lo}, {hi}), ({x0}, {x1}), rnd="rn")
cute.recast_ptr({y}.iterator + {col}, dtype=cutlass.Uint32).store(_cute_column_pack_bf16x2({lo}, {hi}))
""").body
    fast = ast.parse(f"{tid} = {coordinate}").body
    if steps > 1:
        sweep = cast(
            "ast.For", ast.parse(f"for {step} in range({steps}): pass").body[0]
        )
        sweep.body = inner
        fast.append(sweep)
    else:
        fast.extend(inner)
    # Check without overflow. The aligned complete-tile branch is only legal
    # if both byte addresses also stay word-aligned at every expert row.
    guard = parse_expr(
        f"cutlass.const_expr({x}.iterator.alignment >= 4 and {y}.iterator.alignment >= 4) and {x}.layout.stride[{si}] == 1 and {y}.layout.stride[{so}] == 1 and {x}.layout.stride[{si - 1}] % 2 == 0 and {base} >= 0 and {base} <= {fact.width - chunk} and {base} % 2 == 0"
    )
    if (
        fn.config.get("cute_cooperative_row_norm")
        and (tile_symbol := grid.strategy.block_size_var(fact.column)) is not None
    ):
        from .cooperative_row_normalization import PublishedPairs

        fn.cute_state.lanes.cooperative_pair_producer = PublishedPairs(
            root.graph_id,
            x,
            y,
            fact.source.ndim,
            fact.width,
            chunk,
            base,
            tile_symbol,
            ast.dump(guard),
            tid,
            col,
            lo,
            hi,
        )
    fallback = cast("list[ast.stmt]", grid.wrap_body(body))
    body[:] = [ast.If(test=guard, body=fast, orelse=fallback)]
    fn.cute_state.lanes.ordered_column_dims.add(dims)
    return True


def validate_column_launch(
    fn: DeviceFunction, dims: tuple[int, int, int] | None
) -> None:
    if (
        dims is None
        or fn.cute_state.lanes.launch_block != dims
        or fn.cute_state.lanes.ordered_column_dims != {dims}
    ):
        raise exc.InvalidConfig(
            f"{ORDERED_COLUMN_PAIRS_KEY} requires a complete, compatible physical column tile"
        )


def add_column_seed(
    env: CompileEnvironment, ir: DeviceIR, configs: list[Config]
) -> list[Config]:
    facts = env.config_spec.cute_ordered_column_sum_roots
    if not facts or ir.host_function is None:
        return configs
    with ir.host_function:
        inactive = boundary_only_grid_blocks(env, ir, ir.graphs)
        for seed in reversed(configs):
            sizes = seed.get("block_sizes")
            threads = seed.get("num_threads")
            if not isinstance(sizes, list) or not isinstance(threads, list):
                continue
            total = tile_loop_thread_count(
                env, ir, ir.graphs, seed, inactive_block_ids=inactive
            )
            # A scalar column must have at least one extra lane to exploit. The
            # final launcher check remains authoritative for shared physical axes.
            if not 32 <= total <= 1024:
                continue
            if any(
                type(n := env.config_spec.block_sizes.config_get(sizes, col, 0)) is int
                and 2 * total <= n <= width
                and n % (2 * total) == 0
                and env.config_spec.num_threads.config_get(threads, red, 0) == 1
                for col, red, _, width in facts.values()
            ):
                return [
                    *configs,
                    Config.from_dict(seed.config | {ORDERED_COLUMN_PAIRS_KEY: True}),
                ]
    return configs
