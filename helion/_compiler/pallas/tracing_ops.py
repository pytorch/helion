"""Pallas-backend codegen for ops defined in ``helion.language._tracing_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``_tracing_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.codegen.simd import constant_repr

from ...exc import BackendUnsupported
from ...exc import InvalidConfig
from ...language import _decorators
from ...language._tracing_ops import _and
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _for_loop_step
from ...language._tracing_ops import _host_tensor
from ...language._tracing_ops import _if
from ...language._tracing_ops import _mask_to
from ...language._tracing_ops import _new_var
from ...language._tracing_ops import _not
from ...language._tracing_ops import _phi
from ...language._tracing_ops import _pre_broadcast_tile
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_sympy_expr
from ..device_function import find_block_size_symbols
from ..host_function import HostFunction
from .dma import DmaResources
from .dma import DmaTransfer
from .dma import IndirectDmaTransfer
from .dma import ScheduledDmaTransfer
from .dma import allocate_dma_resources
from .dma import allocate_indirect_dma_resources
from .dma import async_copy_statements
from .dma import indirect_group_statements
from .dma import is_tpu_dma_aligned_shape
from .tensorcore_plan import DmaAccessPlan
from .tensorcore_plan import build_dma_access_candidates

log = logging.getLogger(__name__)

_PALLAS_LOOP_LOAD_COUNT_META = "_helion_pallas_loop_load_count"

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..generate_ast import ResidentPrepLowering
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import LoopDimInfo
    from ..tile_strategy import TileStrategy
    from .compact_worklist import ResidentPrepHoist
    from .dma import DmaDirection
    from .plan_tiling import ContiguousRangeIndexPattern


@_decorators.codegen(_not, "pallas")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "jnp.logical_not({lhs})",
        lhs=state.ast_arg(0),
    )


def _loop_carried_indices(state: CodegenState, n_args: int) -> set[int]:
    """Return the set of arg indices that are loop-carried (not read-only).

    Uses ``_phi`` nodes in the parent graph: each ``_phi(init_val, getitem)``
    identifies ``init_val`` as loop-carried.  The ``_for_loop`` FX node's
    ``args[3]`` list gives the ordered args; matching by identity finds the
    loop-carried indices.
    """
    fx_node = state.fx_node
    assert fx_node is not None
    # Collect names of loop-carried initial values from _phi users
    carried_names: set[str] = set()
    for user in fx_node.users:
        for phi_user in user.users:
            if (
                phi_user.op == "call_function"
                and phi_user.target is _phi
                and len(phi_user.args) >= 1
                and hasattr(phi_user.args[0], "name")
            ):
                # pyrefly: ignore [bad-argument-type]
                carried_names.add(phi_user.args[0].name)

    # Match against the _for_loop's arg list
    loop_args = fx_node.args[3]
    assert isinstance(loop_args, list)
    carried: set[int] = set()
    for i, arg in enumerate(loop_args):
        if hasattr(arg, "name") and arg.name in carried_names:
            carried.add(i)
    return carried


def _proxy_loop_parts(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _dependent_tile_end_expr(state: CodegenState, loop_dim_index: int) -> str | None:
    """Render a supported enclosing-``Tile.end`` bound from its provenance.

    Accepts ``tile.end`` and ``min(<host expr>, tile.end)`` on the traced
    SymInt, for kernels with no worklist plan.  Returns ``None`` for any other
    bound, leaving the caller to fall back.
    ``compact_worklist._ordered_source_end`` recognizes the same two forms on
    the source AST; extend the two together.
    """
    from ..variable_origin import TileEndOrigin

    graph_info = state.get_graph(state.proxy_arg(0))
    block_ids = getattr(graph_info, "block_ids", ())
    if loop_dim_index >= len(block_ids):
        return None

    ends = _proxy_loop_parts(state.proxy_arg(2))
    if loop_dim_index >= len(ends):
        return None
    end = ends[loop_dim_index]
    if not isinstance(end, torch.SymInt):
        return None
    expr = _symint_sympy_expr(end)

    tile_ends: list[tuple[sympy.Symbol, TileEndOrigin]] = []
    for symbol in expr.free_symbols:
        if not isinstance(symbol, sympy.Symbol):
            return None
        origin_info = HostFunction.current().expr_to_origin.get(symbol)
        if origin_info is not None and isinstance(origin_info.origin, TileEndOrigin):
            tile_ends.append((symbol, origin_info.origin))
    if len(tile_ends) != 1:
        return None

    tile_end_symbol, tile_end_origin = tile_ends[0]
    if (
        tile_end_origin.block_id in block_ids
        or not state.codegen.active_device_loops.get(tile_end_origin.block_id)
    ):
        return None
    tile_end_expr = tile_end_origin.host_str()
    if expr == tile_end_symbol:
        return tile_end_expr
    if (
        expr.func is not sympy.Min
        or tile_end_symbol not in expr.args
        or len(expr.args) != 2
    ):
        return None
    source_end = next(arg for arg in expr.args if arg != tile_end_symbol)
    if not isinstance(source_end, sympy.Expr):
        return None
    for symbol in source_end.free_symbols:
        origin_info = HostFunction.current().expr_to_origin.get(symbol)
        if origin_info is None or not origin_info.origin.is_host():
            return None
    return CompileEnvironment.current().backend.minimum_expr(
        state.sympy_expr(source_end), tile_end_expr
    )


def _has_supported_dependent_tile_end(state: CodegenState) -> bool:
    """Whether this loop has one supported enclosing-``Tile.end`` bound."""
    graph_info = state.get_graph(state.proxy_arg(0))
    block_ids = getattr(graph_info, "block_ids", ())
    return len(block_ids) == 1 and _dependent_tile_end_expr(state, 0) is not None


def _has_dynamic_unroll_bound(state: CodegenState) -> bool:
    bounds = [
        *_proxy_loop_parts(state.proxy_arg(1)),
        *_proxy_loop_parts(state.proxy_arg(2)),
    ]
    return any(isinstance(bound, (torch.SymInt, torch.Tensor)) for bound in bounds)


def _raise_unsupported_dynamic_unroll() -> None:
    raise InvalidConfig(
        "pallas_loop_type='unroll' requires static inner-loop bounds, an "
        "enclosing Tile.end bound, or pallas_worklist_grouping in (1, 2)."
    )


def _uses_buffered_static_unroll(state: CodegenState) -> bool:
    """Whether a static loop requested the existing depth-two load route."""
    if state.config.get("pallas_loop_type", "unroll") != "unroll":
        return False
    counts = state.config.get("pallas_load_buffer_count", ())
    return isinstance(counts, (list, tuple)) and 2 in counts


def _extract_subscript_vals(subscript: object) -> list[object]:
    """Extract meta values from a subscript argument in an FX graph.

    The subscript is typically a list of FX nodes whose ``meta["val"]``
    contain SymInts or other types representing the tile indices.
    """
    if not isinstance(subscript, (list, tuple)):
        return []
    result: list[object] = []
    for item in subscript:
        if isinstance(item, torch.fx.Node):
            result.append(item.meta.get("val", item))
        else:
            result.append(item)
    return result


@_decorators.codegen(_for_loop, "pallas")
def _(state: CodegenState) -> object:
    """Emit inner device loops for Pallas/TPU.

    Worklist flattening is handled before the ordinary loop-type dispatch: its
    compact tile is already represented by the work-item grid, while its ordered
    loop uses the configured resident or streaming lowering.
    """
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "unroll")
    if CompileEnvironment.current().compact_worklist_plan is not None:
        if _is_compact_ordered_inner_loop(state):
            if pallas_loop_type == "unroll":
                return _codegen_resident_cache(state)
            if pallas_loop_type == "emit_pipeline":
                return _codegen_emit_pipeline(state)
            assert pallas_loop_type == "fori_loop"
        if _is_compact_tile_loop(state):
            plan = CompileEnvironment.current().compact_worklist_plan
            assert plan is not None
            if plan.grouping == 2:
                return _codegen_grouped_compact_tile(state)
        return _codegen_fori_loop(state)
    if pallas_loop_type == "emit_pipeline":
        return _codegen_emit_pipeline(state)
    if pallas_loop_type == "fori_loop":
        return _codegen_fori_loop(state)
    if _has_supported_dependent_tile_end(state):
        return _codegen_dynamic_unroll(state)
    if _has_dynamic_unroll_bound(state):
        _raise_unsupported_dynamic_unroll()
    if _uses_buffered_static_unroll(state):
        return _codegen_fori_loop(state, static_unroll=True)
    # unroll: fall through to common codegen path
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


@_decorators.codegen(_for_loop_step, "pallas")
def _(state: CodegenState) -> None:
    """Emit inner stepped device loops for Pallas/TPU."""
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "unroll")
    if CompileEnvironment.current().compact_worklist_plan is not None:
        if _is_compact_ordered_inner_loop(state):
            if pallas_loop_type == "unroll":
                _codegen_resident_cache(state)
                return None
            if pallas_loop_type == "emit_pipeline":
                _codegen_emit_pipeline(state)
                return None
            assert pallas_loop_type == "fori_loop"
        if _is_compact_tile_loop(state):
            plan = CompileEnvironment.current().compact_worklist_plan
            assert plan is not None
            if plan.grouping == 2:
                _codegen_grouped_compact_tile(state)
                return None
        _codegen_fori_loop(state)
        return None
    if pallas_loop_type == "emit_pipeline":
        _codegen_emit_pipeline(state)
        return None
    if pallas_loop_type == "fori_loop":
        _codegen_fori_loop(state)
        return None
    if _has_supported_dependent_tile_end(state):
        _codegen_dynamic_unroll(state)
        return None
    if _has_dynamic_unroll_bound(state):
        _raise_unsupported_dynamic_unroll()
    if _uses_buffered_static_unroll(state):
        _codegen_fori_loop(state, static_unroll=True)
        return None
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


def _codegen_resident_cache(state: CodegenState) -> object:
    """Range-keyed resident-window lowering for the compact-worklist ordered loop.

    The ordered operand is held in a per-range resident ``C``-row window
    keyed on ``range_start`` (``C`` is the compile-threaded physical window).
    Optional prep-cache descriptors are installed by the dynamic resident loop.

    Ranges longer than ``C`` are NOT handled in-kernel: the torch launcher raises
    (``runtime.pallas.launcher._compact_raise_if_range_exceeds_window``), while
    JAX export keeps this as a caller precondition. There is no in-kernel streamed
    ``else``.
    """
    decision = CompileEnvironment.current().compact_worklist_resident_cache_decision
    assert decision is not None and decision.active
    return _codegen_dynamic_unroll(state)


def _resident_prep_fallback(reason: str) -> None:
    log.debug(
        "compact_worklist resident prep: skipping prep cache hoist (%s); "
        "using resident-only lowering",
        reason,
    )


def _active_resident_prep_hoists(
    state: CodegenState,
    block_ids: list[int],
) -> tuple[ResidentPrepHoist, ...]:
    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    if plan is None or plan.ordered_axis is None:
        return ()
    if not (len(block_ids) == 1 and block_ids[0] == plan.ordered_axis.block_id):
        return ()
    graph_info = state.get_graph(state.proxy_arg(0))
    return tuple(
        hoist
        for hoist in env.compact_worklist_resident_prep_hoists
        if hoist.graph_id == graph_info.graph_id
    )


def _prepare_resident_prep_lowerings(
    state: CodegenState,
    block_ids: list[int],
    all_tensor_info: Sequence[tuple[torch.Tensor, object, object]],
) -> list[ResidentPrepLowering]:
    """Register prep-cache scratch and return lowering descriptors for this graph."""
    from ..generate_ast import ResidentPrepLowering
    from .compact_worklist import elide_installed_prep_load_masks
    from .compact_worklist import metadata_ref_for_field

    env = CompileEnvironment.current()
    decision = env.compact_worklist_resident_cache_decision
    if decision is None or not decision.active:
        return []
    plan = env.compact_worklist_plan
    assert plan is not None
    active_hoists = _active_resident_prep_hoists(state, block_ids)
    if not active_hoists:
        return []
    if not decision.prep_key_fields:
        _resident_prep_fallback("prep cache has no semantic key fields")
        return []
    for field in decision.prep_key_fields:
        metadata_ref_for_field(plan, field)
    metadata_ref_for_field(plan, "range_len")

    graph_info = state.get_graph(state.proxy_arg(0))
    cache_key = (
        graph_info.graph_id,
        decision.physical_window,
        active_hoists,
    )
    common_statements = state.codegen.grouped_compact_common_statements
    if common_statements is not None:
        cached = state.codegen.grouped_resident_prep_lowering_cache.get(cache_key)
        if cached is not None:
            return cached

    load_nodes = {node.name: node for node in graph_info.graph.nodes}
    needed_hosts = {hoist.host_arg for hoist in active_hoists}
    resident_windows: dict[str, tuple[str, torch.Tensor]] = {}
    for fake, _sub_meta, _direction in all_tensor_info:
        ta = state.device_function.tensor_arg(fake)
        host = ta.host_str()
        if host in needed_hosts:
            resident_windows[host] = (ta.name, fake)
    missing = sorted(needed_hosts - resident_windows.keys())
    if missing:
        _resident_prep_fallback(
            f"prep operands have no resident window refs: {missing}"
        )
        return []

    prepared: list[tuple[ResidentPrepHoist, torch.Tensor, str, tuple[int, ...]]] = []
    for hoist in active_hoists:
        load_node = load_nodes.get(hoist.load_node_name)
        load_val = load_node.meta.get("val") if load_node is not None else None
        if not isinstance(load_val, torch.Tensor):
            _resident_prep_fallback(
                f"load node {hoist.load_node_name!r} has no tensor metadata"
            )
            return []
        resident_window_name = resident_windows[hoist.host_arg][0]
        win_shape = (
            decision.physical_window,
            *(int(s) for s in load_val.shape[1:]),
        )
        cache_shape = tuple(win_shape[p] for p in hoist.perm)
        prepared.append((hoist, load_val, resident_window_name, cache_shape))

    lowerings: list[ResidentPrepLowering] = []
    for hoist, load_val, resident_window_name, cache_shape in prepared:
        cache_name = state.device_function.register_scratch(
            cache_shape,
            load_val.dtype,
            name_hint=f"{resident_window_name}_prep",
        )
        lowerings.append(
            ResidentPrepLowering(
                hoist=hoist,
                resident_window_name=resident_window_name,
                cache_name=cache_name,
                # The transpose refill zero-fills the padded tail (see
                # _emit_resident_prep_refill); a per-tile mask with this fill on the
                # load is therefore redundant.
                tail_fill_value=0.0,
            )
        )
    # Fallback paths above return before declaring any tail-fill guarantees.
    load_tail_fills = {
        lowering.hoist.load_node_name: lowering.tail_fill_value
        for lowering in lowerings
    }
    elide_installed_prep_load_masks(graph_info.graph, load_tail_fills)
    if common_statements is not None:
        state.codegen.grouped_resident_prep_lowering_cache[cache_key] = lowerings
    return lowerings


def _emit_resident_prep_refill(
    state: CodegenState,
    block_ids: list[int],
    lowerings: list[ResidentPrepLowering],
) -> None:
    """Emit once-per-prep-key cache refill for active descriptors."""
    from ..generate_ast import ResidentPrepLowering
    from .compact_worklist import metadata_ref_for_field

    if not lowerings:
        return
    assert all(isinstance(lowering, ResidentPrepLowering) for lowering in lowerings)
    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    assert plan is not None and plan.ordered_axis is not None
    decision = env.compact_worklist_resident_cache_decision
    assert decision is not None and decision.active
    blk = state.device_function.block_size_var(block_ids[0])
    assert blk is not None
    prep_key_refs = tuple(
        metadata_ref_for_field(plan, field) for field in decision.prep_key_fields
    )
    assert prep_key_refs
    prep_key_changed = " | ".join(
        f"({ref}[_wid] != {ref}[jnp.maximum(_wid - 1, 0)])" for ref in prep_key_refs
    )
    range_len_ref = metadata_ref_for_field(plan, "range_len")
    num_ordered_tiles = f"(({range_len_ref}[_wid] + {blk} - 1) // {blk})"

    def _stmt(src: str) -> ast.stmt:
        return cast("ast.stmt", statement_from_string(src))

    refill_full_stmts: list[ast.stmt] = []
    refill_tail_stmts: list[ast.stmt] = []
    for lowering in lowerings:
        assert isinstance(lowering, ResidentPrepLowering)
        # Generated fill literals currently support finite values only.
        tail_fill = lowering.tail_fill_value
        assert -float("inf") < tail_fill < float("inf"), (
            "resident prep refill supports only finite numeric tail_fill_value"
        )
        perm = lowering.hoist.perm
        rank = len(perm)
        win_elts = [f"pl.ds(_rc_ordered_tile * {blk}, {blk})"]
        win_elts.extend(":" for _ in range(rank - 1))
        resident_read = f"{lowering.resident_window_name}[{', '.join(win_elts)}]"
        cache_write = ", ".join(win_elts[p] for p in perm)
        ordered_axis_after_perm = perm.index(0)
        mask_dims = [blk if i == ordered_axis_after_perm else "1" for i in range(rank)]
        mask_shape = f"({mask_dims[0]},)" if rank == 1 else f"({', '.join(mask_dims)})"
        full_src_var = state.device_function.new_var("_rc_prep_src")
        refill_full_stmts.extend(
            [
                _stmt(f"{full_src_var} = jnp.transpose({resident_read}, {list(perm)})"),
                _stmt(f"{lowering.cache_name}[{cache_write}] = {full_src_var}"),
            ]
        )
        tail_src_var = state.device_function.new_var("_rc_prep_src")
        mask_var = state.device_function.new_var("_rc_prep_mask")
        refill_tail_stmts.extend(
            [
                _stmt(f"{tail_src_var} = jnp.transpose({resident_read}, {list(perm)})"),
                _stmt(
                    f"{mask_var} = (_rc_ordered_tile * {blk} + "
                    f"jax.lax.broadcasted_iota(jnp.int32, {mask_shape}, "
                    f"{ordered_axis_after_perm})) < {range_len_ref}[_wid]"
                ),
                _stmt(
                    f"{lowering.cache_name}[{cache_write}] = "
                    f"jnp.where({mask_var}, {tail_src_var}, "
                    f"jnp.full_like({tail_src_var}, {tail_fill}))"
                ),
            ]
        )

    refill_fn = _stmt(
        f"@pl.when((_wid == 0) | ({prep_key_changed}))\n"
        f"def _rc_prep_refill():\n"
        f"    pass"
    )
    assert isinstance(refill_fn, ast.FunctionDef)

    body_fn = _stmt("def _rc_prep_refill_body(_rc_ordered_tile, _rc_carry):\n    pass")
    assert isinstance(body_fn, ast.FunctionDef)
    body_fn.body = [
        *refill_full_stmts,
        _stmt("return _rc_carry"),
    ]

    tail_fn = _stmt(
        f"@pl.when(_rc_full_ordered_tiles < {num_ordered_tiles})\n"
        f"def _rc_prep_refill_tail():\n"
        f"    pass"
    )
    assert isinstance(tail_fn, ast.FunctionDef)
    tail_fn.body = [
        _stmt("_rc_ordered_tile = _rc_full_ordered_tiles"),
        *refill_tail_stmts,
    ]

    refill_fn.body = [
        _stmt(f"_rc_full_ordered_tiles = {range_len_ref}[_wid] // {blk}"),
        body_fn,
        _stmt("jax.lax.fori_loop(0, _rc_full_ordered_tiles, _rc_prep_refill_body, ())"),
        tail_fn,
    ]
    state.add_statement(refill_fn)


def _emit_resident_prep_refill_once(
    state: CodegenState,
    block_ids: list[int],
    lowerings: list[ResidentPrepLowering],
) -> None:
    if not lowerings:
        return
    refill_key = tuple(
        (
            lowering.hoist.graph_id,
            lowering.hoist.prep_node_name,
            lowering.cache_name,
        )
        for lowering in lowerings
    )
    common_statements = state.codegen.grouped_compact_common_statements
    if (
        common_statements is not None
        and refill_key in state.codegen.grouped_resident_prep_refill_cache
    ):
        return
    if common_statements is None:
        _emit_resident_prep_refill(state, block_ids, lowerings)
        return
    with state.codegen.set_statements(common_statements):
        _emit_resident_prep_refill(state, block_ids, lowerings)
    state.codegen.grouped_resident_prep_refill_cache[refill_key] = "emitted"


def _classify_loop_tensors(
    graph_info: object,
    state: object,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
    dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
]:
    """Classify tensors accessed in an inner loop body into loaded/stored.

    Returns (loaded_tensors, stored_tensors) dicts keyed by id(fake_tensor).
    """
    from ...language.memory_ops import load as _load_op
    from ...language.memory_ops import store as _store_op

    host_tensor_nodes: dict[torch.fx.Node, torch.Tensor] = {}
    for node in graph_info.graph.nodes:  # type: ignore[union-attr]
        if node.op == "call_function" and node.target is _host_tensor:
            if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
                host_tensor_nodes[node] = node.meta["val"]

    loaded_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]] = {}
    stored_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]] = {}

    for node in graph_info.graph.nodes:  # type: ignore[union-attr]
        if node.op != "call_function":
            continue
        if node.target is _load_op:
            tensor_node = node.args[0]
            subscript = node.args[1]
            if (
                isinstance(tensor_node, torch.fx.Node)
                and tensor_node in host_tensor_nodes
            ):
                fake = host_tensor_nodes[tensor_node]
                key = id(fake)
                if key not in loaded_tensors:
                    sub_vals = _extract_subscript_vals(subscript)
                    loaded_tensors[key] = (fake, node, sub_vals)
                    node.meta[_PALLAS_LOOP_LOAD_COUNT_META] = 1
                else:
                    first_load = loaded_tensors[key][1]
                    first_load.meta[_PALLAS_LOOP_LOAD_COUNT_META] = (
                        int(first_load.meta[_PALLAS_LOOP_LOAD_COUNT_META]) + 1
                    )
        elif node.target is _store_op:
            tensor_node = node.args[0]
            subscript = node.args[1]
            if (
                isinstance(tensor_node, torch.fx.Node)
                and tensor_node in host_tensor_nodes
            ):
                fake = host_tensor_nodes[tensor_node]
                key = id(fake)
                if key not in stored_tensors:
                    sub_vals = _extract_subscript_vals(subscript)
                    stored_tensors[key] = (fake, tensor_node, sub_vals)

    return loaded_tensors, stored_tensors


def _dma_plan_for_node(node: torch.fx.Node) -> DmaAccessPlan | None:
    from .tensorcore_plan import TENSORCORE_PLAN_META

    plan = node.meta.get(TENSORCORE_PLAN_META)
    return plan if isinstance(plan, DmaAccessPlan) else None


def _collect_indirect_accesses(
    graphs: list[GraphInfo],
    graph_info: GraphInfo,
    block_ids: list[int],
    block_extents: dict[int, int],
    active_block_ids: set[int],
) -> tuple[list[IndirectDmaTransfer], set[int]]:
    """Return indirect gathers and paired writebacks owned by one scheduler."""
    from ..device_function import DeviceFunction
    from .memory_access import MemoryAccessKind
    from .plan_tiling import ArbitraryIndexPattern
    from .plan_tiling import TileBeginWithOffsetPattern
    from .plan_tiling import TilePattern

    admitted: list[IndirectDmaTransfer] = []
    metadata_ids: set[int] = set()

    optional_dma = (
        DeviceFunction.current().config.get("pallas_indirect_access_mode", "one_hot")
        == "dma"
    )
    if not optional_dma:
        return [], set()
    for candidate in build_dma_access_candidates(graphs):
        if candidate.graph_id != graph_info.graph_id:
            continue
        specs = [candidate.load]
        if candidate.store is not None:
            specs.append(candidate.store)
        plans = [_dma_plan_for_node(spec.node) for spec in specs]
        if not any(plan is not None for plan in plans):
            continue
        if any(plan is None for plan in plans):
            raise InvalidConfig(
                "indirect DMA requires the paired state load and store to use "
                "the same DMA schedule"
            )
        for spec, plan in zip(specs, plans, strict=True):
            assert plan is not None
            node = spec.node
            tensor = plan.access.tensor
            extent = block_extents.get(plan.spec.index_block_id)
            if plan.spec.index_block_id not in block_ids or extent is None:
                raise InvalidConfig(
                    "indirect DMA address metadata is not owned by the active scheduler"
                )
            if extent % plan.group_count != 0:
                raise InvalidConfig(
                    f"indirect DMA block size {plan.group_count} does not divide "
                    f"scheduler extent {extent}"
                )

            index_access = plan.spec.index_access
            patterns = index_access.patterns
            sub_meta = _extract_subscript_vals(index_access.subscript)
            if len(patterns) != len(sub_meta) or any(
                not (
                    isinstance(pattern, TilePattern)
                    or (
                        isinstance(pattern, TileBeginWithOffsetPattern)
                        and pattern.block_id in active_block_ids
                    )
                    or (
                        isinstance(pattern, ArbitraryIndexPattern)
                        and isinstance(index, (int, torch.SymInt))
                    )
                )
                for pattern, index in zip(patterns, sub_meta, strict=True)
            ):
                raise InvalidConfig(
                    "indirect DMA address metadata cannot be sliced by the active "
                    "scheduler"
                )

            admitted.append(
                IndirectDmaTransfer(
                    tensor=tensor,
                    subscript=tuple(_extract_subscript_vals(node.args[1])),
                    direction=(
                        "load" if plan.access.kind is MemoryAccessKind.LOAD else "store"
                    ),
                    plan=plan,
                )
            )
        metadata_ids.update(candidate.metadata_tensor_ids)

    return admitted, metadata_ids


def _collect_fori_indirect_accesses(
    graph_info: object,
    block_ids: list[int],
    state: CodegenState,
) -> tuple[list[IndirectDmaTransfer], set[int]]:
    """Admit static indirect accesses owned by one fori scheduler."""
    from ..device_ir import ForLoopGraphInfo
    from ..tile_strategy import DeviceLoopState
    from ..tile_strategy import EmitPipelineLoopState
    from ..tile_strategy import ForiLoopState

    env = CompileEnvironment.current()
    if state.config.get("pallas_indirect_access_mode", "one_hot") != "dma":
        return [], set()
    nested_scheduler = any(
        isinstance(loop, (DeviceLoopState, EmitPipelineLoopState, ForiLoopState))
        for loops in state.codegen.active_device_loops.values()
        for loop in loops
    )
    if (
        not isinstance(graph_info, ForLoopGraphInfo)
        or len(block_ids) != 1
        or env.compact_worklist_plan is not None
        or nested_scheduler
    ):
        return [], set()

    steps = state.proxy_arg(4) if len(state.proxy_args) > 4 else None
    step = steps[0] if isinstance(steps, (list, tuple)) else steps
    if step is not None and sympy.sympify(step) not in (
        sympy.Integer(0),
        sympy.Integer(1),
    ):
        return [], set()
    begin, end = _get_loop_begin_and_end(state, 0)
    try:
        extent = int(end) - int(begin)
    except (TypeError, ValueError):
        return [], set()

    active_block_ids = {
        block_id
        for block_id, loops in state.codegen.active_device_loops.items()
        if loops
    } | set(block_ids)
    accesses, metadata_ids = _collect_indirect_accesses(
        list(state.codegen.codegen_graphs),
        graph_info,
        block_ids,
        {block_ids[0]: extent},
        active_block_ids,
    )
    from ..device_function import DeviceFunction
    from ..device_function import PallasMemorySpace

    device_fn = DeviceFunction.current()
    for access in accesses:
        device_fn.pallas_memory_space[id(access.tensor)] = PallasMemorySpace.HBM
    return accesses, metadata_ids


def plan_grid_indirect_accesses(graphs: list[GraphInfo]) -> None:
    """Bind root-grid indirect accesses to immediate-wait DMA scratch."""
    from ..device_function import DeviceFunction
    from ..device_function import PallasMemorySpace
    from ..device_ir import RootGraphInfo

    env = CompileEnvironment.current()
    device_fn = DeviceFunction.current()
    if env.compact_worklist_plan is not None or env.settings.pallas_interpret:
        return
    device_ir = HostFunction.current().device_ir
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    for root_id, block_ids in zip(
        device_ir.root_ids, device_ir.grid_block_ids, strict=True
    ):
        graph_info = graph_by_id.get(root_id)
        if not isinstance(graph_info, RootGraphInfo):
            continue
        block_extents: dict[int, int] = {}
        for block_id in block_ids:
            size = env.block_sizes[block_id].size
            if not isinstance(size, (int, torch.SymInt)):
                break
            extent = env.try_concretize_symint(size)
            if not isinstance(extent, int) or extent <= 0:
                break
            block_extents[block_id] = extent
        if len(block_extents) != len(block_ids):
            continue
        accesses, _ = _collect_indirect_accesses(
            graphs,
            graph_info,
            block_ids,
            block_extents,
            set(block_ids),
        )
        load_resources_by_storage: dict[int, DmaResources] = {}
        for access in accesses:
            node = access.plan.access.node
            storage_id = id(access.tensor.untyped_storage())
            resources = allocate_indirect_dma_resources(
                device_fn,
                access,
                buffer_count=1,
                load_resources=load_resources_by_storage.get(storage_id),
            )
            if access.direction == "load":
                load_resources_by_storage[storage_id] = resources
            device_fn.pallas_grid_dma_bindings[node] = resources
            device_fn.pallas_memory_space[id(access.tensor)] = PallasMemorySpace.HBM


def _tensor_dim_subscripts(subscript_meta: Sequence[object]) -> list[object]:
    """Drop rank-expanding ``None`` entries from a tensor subscript."""
    return [index for index in subscript_meta if index is not None]


def _subscript_at_dim(subscripts: Sequence[object], dim: int) -> object:
    return subscripts[dim] if dim < len(subscripts) else slice(None)


def _get_dim_block_ids(
    subscript_meta: list[object],
    env: CompileEnvironment,
) -> dict[int, int]:
    """Map tensor dimension index -> block_id from subscript metadata."""
    dim_to_bid: dict[int, int] = {}
    if not isinstance(subscript_meta, (list, tuple)):
        return dim_to_bid
    for dim_idx, idx in enumerate(_tensor_dim_subscripts(subscript_meta)):
        if isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None:
                dim_to_bid[dim_idx] = bid
        elif isinstance(idx, slice) and idx == slice(None):
            pass
    return dim_to_bid


def _contiguous_range_patterns(
    loaded_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
) -> dict[int, dict[int, ContiguousRangeIndexPattern]]:
    """Return direct HBM range patterns keyed by tensor and tensor dimension."""
    from .plan_tiling import ContiguousRangeIndexPattern
    from .plan_tiling import NonePattern

    result: dict[int, dict[int, ContiguousRangeIndexPattern]] = {}
    for fake, load_node, _subscript in loaded_tensors.values():
        tensor_dim = 0
        ranges: dict[int, ContiguousRangeIndexPattern] = {}
        for pattern in load_node.meta.get("indexing_patterns", ()):
            if isinstance(pattern, NonePattern):
                continue
            if isinstance(pattern, ContiguousRangeIndexPattern):
                ranges[tensor_dim] = pattern
            tensor_dim += 1
        if ranges:
            result[id(fake)] = ranges
    return result


def _contiguous_range_base_expr(
    value: object,
    *,
    state: CodegenState,
    block_ids: list[int],
    begin_exprs: list[str],
    iter_step_exprs: list[str],
    iteration_indices: list[str],
    captured_exprs: dict[torch.fx.Node, str] | None = None,
) -> str | None:
    """Render a supported scalar address expression for one loop iteration."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, torch.SymInt):
        return state.device_function.literal_expr(value)
    if not isinstance(value, torch.fx.Node):
        return None
    if value.op == "placeholder":
        return captured_exprs.get(value) if captured_exprs is not None else None
    if value.op != "call_function":
        return None

    if value.target is _new_var and value.args:
        return _contiguous_range_base_expr(
            value.args[0],
            state=state,
            block_ids=block_ids,
            begin_exprs=begin_exprs,
            iter_step_exprs=iter_step_exprs,
            iteration_indices=iteration_indices,
            captured_exprs=captured_exprs,
        )

    from ...language import memory_ops
    from ...language.tile_ops import tile_begin

    if value.target is tile_begin:
        symbolic_value = value.meta.get("val")
        if not isinstance(symbolic_value, torch.SymInt):
            return None
        block_id = CompileEnvironment.current().get_block_id(symbolic_value)
        if block_id is None or block_id not in block_ids:
            return None
        position = block_ids.index(block_id)
        return (
            f"({begin_exprs[position]}) + ({iteration_indices[position]}) * "
            f"({iter_step_exprs[position]})"
        )

    binary_operators = {
        operator.add: "+",
        operator.floordiv: "//",
        operator.mod: "%",
        operator.mul: "*",
        operator.sub: "-",
        torch.ops.aten.add.Scalar: "+",
        torch.ops.aten.add.Tensor: "+",
        torch.ops.aten.mul.Scalar: "*",
        torch.ops.aten.mul.Tensor: "*",
        torch.ops.aten.sub.Scalar: "-",
        torch.ops.aten.sub.Tensor: "-",
    }
    if value.target in binary_operators and len(value.args) >= 2:
        if (
            value.target
            in (
                torch.ops.aten.add.Scalar,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Scalar,
                torch.ops.aten.sub.Tensor,
            )
            and value.kwargs.get("alpha", 1) != 1
        ):
            return None
        lhs = _contiguous_range_base_expr(
            value.args[0],
            state=state,
            block_ids=block_ids,
            begin_exprs=begin_exprs,
            iter_step_exprs=iter_step_exprs,
            iteration_indices=iteration_indices,
            captured_exprs=captured_exprs,
        )
        rhs = _contiguous_range_base_expr(
            value.args[1],
            state=state,
            block_ids=block_ids,
            begin_exprs=begin_exprs,
            iter_step_exprs=iter_step_exprs,
            iteration_indices=iteration_indices,
            captured_exprs=captured_exprs,
        )
        if lhs is None or rhs is None:
            return None
        return f"({lhs}) {binary_operators[value.target]} ({rhs})"

    if value.target is memory_ops.load:
        tensor_node, subscript = value.args[:2]
        if not isinstance(tensor_node, torch.fx.Node):
            return None
        tensor = tensor_node.meta.get("val")
        if not isinstance(tensor, torch.Tensor):
            return None
        if not isinstance(subscript, (list, tuple)) or len(subscript) != 1:
            return None
        index = _contiguous_range_base_expr(
            subscript[0],
            state=state,
            block_ids=block_ids,
            begin_exprs=begin_exprs,
            iter_step_exprs=iter_step_exprs,
            iteration_indices=iteration_indices,
            captured_exprs=captured_exprs,
        )
        if index is None:
            return None
        name = state.device_function.tensor_arg(tensor).name
        return f"{name}[{index}]"

    return None


def _find_strategy(
    state: CodegenState,
    block_ids: list[int],
) -> TileStrategy:
    """Find the tile strategy for the given block_ids."""
    strategy = state.device_function.tile_strategy.block_id_to_strategy.get(
        tuple(block_ids)
    )
    if strategy is None:
        for (
            key_tuple,
            candidate,
        ) in state.device_function.tile_strategy.block_id_to_strategy.items():
            if set(block_ids).issubset(set(key_tuple)):
                strategy = candidate
                break
    assert strategy is not None, f"No strategy found for block_ids {block_ids}"
    return strategy


def _compact_axis_kind(state: CodegenState, loop_dim_index: int) -> str | None:
    """Classify a loop dim under ``compact_worklist`` as the compact tile, the
    ordered inner (carried) tile, or neither.

    Single source of truth shared by the loop-type dispatch
    (:func:`_is_compact_ordered_inner_loop`) and the begin/end remap
    (:func:`_compact_worklist_bounds`), so the two can never disagree on which
    axis a loop is.
    """
    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    if plan is None:
        return None
    from ..device_ir import ForLoopGraphInfo

    graph_info = state.get_graph(state.proxy_arg(0))
    if not isinstance(graph_info, ForLoopGraphInfo):
        return None
    block_ids = graph_info.block_ids
    if loop_dim_index >= len(block_ids):
        return None
    block_id = block_ids[loop_dim_index]
    if block_id == plan.compact_axis.block_id:
        return "compact"
    ordered = plan.ordered_axis
    if ordered is not None and block_id == ordered.block_id:
        return "ordered"
    return None


def _is_compact_ordered_inner_loop(state: CodegenState) -> bool:
    """True if this ``_for_loop`` is the compact-worklist ordered inner tile.

    The ordered tile is the carried inner reduction selected by
    ``pallas_loop_type``. The owner grid and single-iteration compact tile stay
    on the fori path.
    """
    from ..device_ir import ForLoopGraphInfo

    graph_info = state.get_graph(state.proxy_arg(0))
    if not isinstance(graph_info, ForLoopGraphInfo):
        return False
    return any(
        _compact_axis_kind(state, i) == "ordered"
        for i in range(len(graph_info.block_ids))
    )


def _is_compact_tile_loop(state: CodegenState) -> bool:
    """True for the parallel compact-tile loop represented by the work item."""
    from ..device_ir import ForLoopGraphInfo

    plan = CompileEnvironment.current().compact_worklist_plan
    if plan is None:
        return False
    graph_info = state.get_graph(state.proxy_arg(0))
    if not isinstance(graph_info, ForLoopGraphInfo):
        return False
    if graph_info.block_ids != [plan.compact_axis.block_id]:
        return False
    args = state.proxy_args[-1]
    return isinstance(args, list) and not _loop_carried_indices(state, len(args))


@contextlib.contextmanager
def _compact_block_variant(state: CodegenState, factor: int) -> Iterator[None]:
    """Temporarily codegen the compact loop with ``factor * base_block``."""
    if factor == 1:
        yield
        return

    from ...runtime.config import Config
    from ..tile_dispatch import TileStrategyDispatch

    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    assert plan is not None and factor == 2
    block_id = plan.compact_axis.block_id
    fn = state.device_function

    block_sizes = [*fn.config.block_sizes]
    block_index = env.config_spec.block_sizes.block_id_to_index(block_id)
    block_sizes[block_index] *= factor
    variant_config = Config.from_dict({**fn.config.config, "block_sizes": block_sizes})

    original_config = fn.config
    original_tile_strategy = fn.tile_strategy
    original_block_cache = fn.block_size_var_cache
    original_expr_cache = fn.expr_to_var_info

    variant_block_cache = original_block_cache.copy()
    variant_block_cache.pop((block_id,), None)
    variant_expr_cache = {}
    for expr, info in original_expr_cache.items():
        block_mapping, _ = find_block_size_symbols(expr)
        if block_id not in block_mapping.values():
            variant_expr_cache[expr] = info

    fn.config = variant_config
    fn.block_size_var_cache = variant_block_cache
    fn.expr_to_var_info = variant_expr_cache
    try:
        fn.tile_strategy = TileStrategyDispatch(fn, variant_config)
        yield
    finally:
        fn.config = original_config
        fn.tile_strategy = original_tile_strategy
        fn.block_size_var_cache = original_block_cache
        fn.expr_to_var_info = original_expr_cache


def _compact_output_initializers(state: CodegenState) -> list[ast.stmt]:
    """Zero the full max-sized output window before the one-tile body."""
    from ..device_function import TensorArg

    plan = CompileEnvironment.current().compact_worklist_plan
    assert plan is not None and plan.grouping == 2
    output_hosts = {
        policy.arg_name
        for policy in plan.tensor_policies
        if policy.kind == "compact_exact_store"
    }
    statements: list[ast.stmt] = []
    for arg in state.device_function.arguments:
        if not isinstance(arg, TensorArg) or arg.host_str() not in output_hosts:
            continue
        indices = ", ".join(":" for _ in range(arg.fake_value.ndim))
        statements.append(
            statement_from_string(
                f"{arg.name}[{indices}] = jnp.zeros_like({arg.name}[{indices}])"
            )
        )
    assert len(statements) == len(output_hosts)
    return statements


def _codegen_grouped_compact_tile(state: CodegenState) -> None:
    """Emit static base-block and double-block compact-body variants."""
    from .compact_worklist import compact_ref_names

    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    assert plan is not None and plan.grouping == 2
    assert _is_compact_tile_loop(state)

    codegen = state.codegen
    common_statements: list[ast.AST] = []
    branch_defs: list[ast.FunctionDef] = []
    extent_ref = f"{compact_ref_names(plan)[1]}_ref"

    counter_names = (
        "atomic_op_index",
        "device_load_index",
        "device_load_cache_modifier_index",
        "device_store_index",
        "device_store_cache_modifier_index",
        "device_memory_op_index",
    )
    initial_counters = {
        name: getattr(state.device_function, name) for name in counter_names
    }
    final_counters: dict[str, int] | None = None
    previous_common = codegen.grouped_compact_common_statements
    assert previous_common is None
    assert not codegen.grouped_resident_prep_lowering_cache
    assert not codegen.grouped_resident_prep_refill_cache
    assert not codegen.grouped_fori_dma_resource_cache
    codegen.grouped_compact_common_statements = common_statements
    try:
        for factor in (1, 2):
            for name, value in initial_counters.items():
                setattr(state.device_function, name, value)
            branch_body: list[ast.AST] = []
            with (
                codegen.set_statements(branch_body),
                _compact_block_variant(state, factor),
            ):
                result = _codegen_fori_loop(state)
            assert result is None
            if factor == 1:
                branch_body[:0] = _compact_output_initializers(state)
                final_counters = {
                    name: getattr(state.device_function, name) for name in counter_names
                }

            fn_name = state.device_function.new_var(f"_compact_group_{factor}")
            comparison = "<=" if factor == 1 else ">"
            fn_def = statement_from_string(
                f"@pl.when({extent_ref}[_wid] {comparison} "
                f"{env.compact_worklist_block})\n"
                f"def {fn_name}():\n"
                f"    pass"
            )
            assert isinstance(fn_def, ast.FunctionDef)
            fn_def.body = cast("list[ast.stmt]", branch_body) or [ast.Pass()]
            branch_defs.append(fn_def)
    finally:
        codegen.grouped_compact_common_statements = previous_common
        codegen.grouped_resident_prep_lowering_cache.clear()
        codegen.grouped_resident_prep_refill_cache.clear()
        codegen.grouped_fori_dma_resource_cache.clear()
        for name, value in (final_counters or initial_counters).items():
            setattr(state.device_function, name, value)

    for statement in common_statements:
        state.add_statement(statement)
    for fn_def in branch_defs:
        state.add_statement(fn_def)


def _compact_worklist_bounds(
    state: CodegenState, loop_dim_index: int
) -> tuple[str, str] | None:
    """Metadata-ref begin/end for a compact/ordered tile, else None."""
    kind = _compact_axis_kind(state, loop_dim_index)
    if kind is None:
        return None
    from ..pallas.compact_worklist import compact_ref_names
    from ..pallas.compact_worklist import ordered_ref_names

    plan = CompileEnvironment.current().compact_worklist_plan
    assert plan is not None
    ref_names = compact_ref_names if kind == "compact" else ordered_ref_names
    begin_ref, extent_ref = (f"{n}_ref" for n in ref_names(plan))
    begin = f"{begin_ref}[_wid]"
    end = f"{begin} + {extent_ref}[_wid]"
    if kind == "ordered":
        # The source range above still spans the whole reused window; these
        # narrow only what this work item computes.  Resident-window reads take
        # the local offset as (absolute offset - range_start), so a begin that
        # no longer coincides with the window base needs nothing extra.
        compact_begin, compact_extent = (
            f"{name}_ref" for name in compact_ref_names(plan)
        )
        if plan.ordered_begin_window is not None:
            window_start = f"{compact_begin}[_wid] - {plan.ordered_begin_window}"
            begin = f"jnp.maximum({begin}, {window_start})"
        if plan.ordered_end_clamped_to_compact:
            compact_end = f"{compact_begin}[_wid] + {compact_extent}[_wid]"
            end = f"jnp.minimum({end}, {compact_end})"
    return begin, end


def _get_loop_begin_and_end(
    state: CodegenState, loop_dim_index: int
) -> tuple[str, str]:
    """Extract the begin and end values from the _for_loop state args.

    Under worklist flattening the compact tile's begin/end are remapped to the
    per-work-item metadata refs: begin =
    ``tile_starts_ref[_wid]``, end =
    ``tile_starts_ref[_wid] + tile_extents_ref[_wid]`` (and likewise the ordered
    axis -> ``range_start_ref``/``range_len_ref``).  Every downstream consumer
    (trip count, offset, masks) then composes unchanged: ``_codegen_fori_loop``
    runs a single iteration for the compact tile (``tile_extent <= BLOCK``),
    while the ordered axis uses the same remapped bounds for its resident-window
    fori loop or streamed pipeline grid.
    """
    remap = _compact_worklist_bounds(state, loop_dim_index)
    if remap is not None:
        return remap
    dependent_end = _dependent_tile_end_expr(state, loop_dim_index)
    ast_begins = state.ast_args[1]
    ast_ends = state.ast_args[2]
    begins = list(ast_begins) if isinstance(ast_begins, (list, tuple)) else [ast_begins]
    ends = list(ast_ends) if isinstance(ast_ends, (list, tuple)) else [ast_ends]

    def _to_str(value: object) -> str:
        if isinstance(value, ast.AST):
            return ast.unparse(value)
        return str(value)

    return _to_str(begins[loop_dim_index]), (
        dependent_end if dependent_end is not None else _to_str(ends[loop_dim_index])
    )


def _get_loop_numel(state: CodegenState, loop_dim_index: int) -> str:
    begin, end = _get_loop_begin_and_end(state, loop_dim_index)
    return f"(({end}) - ({begin}))"


def _loop_dim_infos(
    state: CodegenState,
    block_ids: list[int],
    env: CompileEnvironment,
) -> dict[int, LoopDimInfo]:
    """Per-dim bounds for an inner device loop, shared by every loop lowering.

    ``tile.end``/``tile.count`` on an enclosing tile read ``end_var_name`` back
    out of here, so all three lowerings must publish the same bounds they
    generate code against; building them in one place keeps them from drifting.
    """
    from ..tile_strategy import LoopDimInfo

    infos: dict[int, LoopDimInfo] = {}
    for i, block_id in enumerate(block_ids):
        block_size = env.block_sizes[block_id]
        begin_expr, end_expr = _get_loop_begin_and_end(state, i)
        infos[block_id] = LoopDimInfo(
            begin_var_name=begin_expr,
            end_var_name=end_expr,
            # No SymPy numel exists when the block size has no static size.
            end_expr=block_size.numel if block_size.size is not None else None,
        )
    return infos


def _is_static_int(expr: str) -> bool:
    """True if a begin/end expression string is a compile-time integer constant.

    Used to decide whether a tile loop's ``[begin, end)`` extent is statically
    known. When it is not (data-dependent bounds — a jagged ``hl.tile(start,
    end)`` or even ``hl.tile(0, dynamic_end)``), the final tile may be a partial
    sub-range of the backing tensor, so an output store must clamp its extent.
    """
    try:
        int(expr)
    except (TypeError, ValueError):
        return False
    return True


def _compute_grid_and_block_sizes(
    state: CodegenState,
    block_ids: list[int],
    env: CompileEnvironment,
    aligned_dim: dict[int, int] | None = None,
) -> tuple[list[str], list[str]]:
    """Compute grid dimensions and block size vars for the given block_ids."""
    aligned_dim = aligned_dim or {}
    grid_parts: list[str] = []
    block_size_vars: list[str] = []
    for i, block_id in enumerate(block_ids):
        block_size_var = state.device_function.block_size_var(block_id)
        assert block_size_var is not None
        block_size_vars.append(block_size_var)
        block_value = state.device_function.resolved_block_size(block_id)
        if block_value is not None:
            state.device_function.constexpr_arg(block_size_var, block_value)
        if block_id in aligned_dim:
            # Aligned-enclosing span: ceil(end/S)*S - floor(begin/S)*S.
            begin, end = _get_loop_begin_and_end(state, i)
            sublane = aligned_dim[block_id]
            a_start = f"(({begin}) - ({begin}) % {sublane})"
            a_end = f"((({end}) + {sublane} - 1) // {sublane} * {sublane})"
            numel_expr = f"({a_end} - {a_start})"
        else:
            numel_expr = _get_loop_numel(state, i)
        grid_parts.append(
            env.backend.cdiv_expr(numel_expr, block_size_var, is_device=True)
        )
    return grid_parts, block_size_vars


def _pallas_loop_begin_and_step_exprs(
    state: CodegenState,
    block_ids: list[int],
    block_size_vars: list[str],
    aligned_dim: dict[int, int] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return begin, per-iteration step, and slice-size expressions for loop dims."""
    aligned_dim = aligned_dim or {}
    steps = state.proxy_arg(4) if len(state.proxy_args) > 4 else None

    if not isinstance(steps, (list, tuple)):
        steps = [steps] * len(block_ids)

    begin_exprs: list[str] = []
    iter_step_exprs: list[str] = []
    slice_size_exprs: list[str] = []

    for i in range(len(block_ids)):
        step = steps[i]
        begin_expr, _ = _get_loop_begin_and_end(state, i)
        if block_ids[i] in aligned_dim:
            # Align the tile begin DOWN to the sublane (aligned-enclosing).
            sublane = aligned_dim[block_ids[i]]
            begin_expr = f"(({begin_expr}) - ({begin_expr}) % {sublane})"
        if step is None or sympy.sympify(step) in (
            sympy.Integer(0),
            sympy.Integer(1),
        ):
            iter_step_expr = block_size_vars[i]
            slice_size_expr = block_size_vars[i]
        else:
            iter_step_expr = state.sympy_expr(sympy.sympify(step))
            slice_size_expr = "1"
        begin_exprs.append(begin_expr)
        iter_step_exprs.append(iter_step_expr)
        slice_size_exprs.append(slice_size_expr)

    return begin_exprs, iter_step_exprs, slice_size_exprs


def _pipeline_begin_alignment(
    begin_expr: str,
    state: CodegenState,
) -> int | None:
    """Return a proven divisor of ``begin_expr``, or ``None``.

    A nested-tile inner loop's begin is its outer loop's ``offset_var``, which is
    a multiple of the outer block size when that outer loop begins at 0 — so it
    needs no extra pad. Returns that block size in that case.
    """
    for block_id, loops in state.codegen.active_device_loops.items():
        if not loops:
            continue
        if state.codegen.offset_var(block_id) != begin_expr:
            continue
        info = loops[-1].block_id_to_info.get(block_id)
        # Only sound when the outer loop begins at 0, so the offset is a clean
        # multiple of the outer block size.
        if info is None or info.begin_expr not in (0, sympy.Integer(0)):
            return None
        outer_bs = state.device_function.resolved_block_size(block_id)
        return outer_bs if isinstance(outer_bs, int) else None
    return None


def _fixed_loop_extent(state: CodegenState, loop_dim_index: int) -> int | None:
    """Return a direct ``end = begin + constant`` loop extent, if present.

    Device-loop bounds that come from scalar tensor arithmetic are no longer
    symbolic integers by codegen time. Their generated names therefore do not
    retain enough information for SymPy simplification, but the enclosing
    ``_for_loop`` FX node still records the original relationship. Keep this
    matcher deliberately narrow: it proves only a direct, positive constant
    addition (or the equivalent subtraction on the begin).
    """
    node = state.fx_node
    if node is None or len(node.args) < 3:
        return None
    raw_begins, raw_ends = node.args[1:3]
    begins = list(raw_begins) if isinstance(raw_begins, (list, tuple)) else [raw_begins]
    ends = list(raw_ends) if isinstance(raw_ends, (list, tuple)) else [raw_ends]
    if loop_dim_index >= len(begins) or loop_dim_index >= len(ends):
        return None
    begin = begins[loop_dim_index]
    end = ends[loop_dim_index]

    def _positive_int(value: object) -> int | None:
        if isinstance(value, (int, sympy.Integer)) and int(value) > 0:
            return int(value)
        return None

    if isinstance(begin, (int, sympy.Integer)) and isinstance(
        end, (int, sympy.Integer)
    ):
        return _positive_int(int(end) - int(begin))

    if isinstance(end, torch.fx.Node) and end.op == "call_function":
        if (
            end.target
            in (operator.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
            and len(end.args) >= 2
            and end.kwargs.get("alpha", 1) == 1
        ):
            if end.args[0] is begin:
                return _positive_int(end.args[1])
            if end.args[1] is begin:
                return _positive_int(end.args[0])
    if isinstance(begin, torch.fx.Node) and begin.op == "call_function":
        if (
            begin.target
            in (operator.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
            and len(begin.args) >= 2
            and begin.kwargs.get("alpha", 1) == 1
        ):
            if begin.args[0] is end:
                return _positive_int(begin.args[1])
    return None


def _hoist_initial_dma_before_pure_outer_compute(
    state: CodegenState,
    statements: list[ast.stmt],
    loop_local_names: list[str],
) -> bool:
    """Move an independent initial DMA to the start of the current root body.

    A buffered inner loop normally starts its first load immediately before
    entering the loop. If the address depends only on kernel arguments and
    constants, starting it before preceding elementwise root computation
    exposes useful DMA/compute overlap. The intervening work may itself contain
    communication or stores; moving a read-only DMA across it is safe as long
    as none of that work produces a value or mutates a ref read by the DMA.
    """
    from ..ast_read_writes import ReadWrites

    outer = state.codegen.statements_stack[-1]
    dma_reads = set(ReadWrites.from_list(statements).reads)
    # A nested loop's initial DMA can depend on an enclosing device-loop
    # induction variable. That variable is an argument of the not-yet-emitted
    # loop body, so it is unavailable in ``outer`` even though no assignment in
    # ``outer`` produces it. Keep the prime inside that loop body in this case.
    if dma_reads.intersection(loop_local_names):
        return False
    for statement in outer:
        rw = ReadWrites.from_ast(statement)
        if (set(rw.writes) | set(rw.inplace_writes)) & dma_reads:
            return False
    grid_state = state.codegen.current_grid_state
    if (
        state.device_function.pallas_hoisted_direct_dma_copy_names
        and grid_state is not None
    ):
        grid_state.outer_prefix.extend(statements)
    else:
        outer[:0] = statements
    return True


def _compute_pipeline_or_dma_extra_pad(
    begin_expr: str,
    bid: int,
    env: CompileEnvironment,
    state: CodegenState,
    loop_dim_index: int | None = None,
) -> int:
    """Return extra host-side padding for a pipeline/DMA dim with a non-zero begin.

    When ``pl.ds(offset, block_size)`` reads from a tensor whose loop starts
    at a non-zero begin, the last block can overshoot the tensor boundary
    beyond what ``(-shape) % block_size`` accounts for.  The worst case is
    ``block_size - 1`` extra elements when the begin is data-dependent, but a
    begin that is provably a multiple of ``block_size`` (e.g. an outer tile's
    aligned offset) needs no extra padding at all.
    """
    if begin_expr == "0":
        return 0
    bs_val = state.device_function.resolved_block_size(bid)
    if not isinstance(bs_val, int):
        return 0
    if loop_dim_index is not None:
        extent = _fixed_loop_extent(state, loop_dim_index)
        if extent is not None and extent % bs_val == 0:
            return 0
    alignment = _pipeline_begin_alignment(begin_expr, state)
    if alignment is not None and alignment % bs_val == 0:
        return 0
    return bs_val - 1


def _active_loop_begin_expr(state: CodegenState, block_id: int) -> str:
    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return "0"
    info = loops[-1].block_id_to_info.get(block_id)
    if info is None:
        return "0"
    if info.begin_expr is not None:
        return str(info.begin_expr)
    return info.begin_var_name or "0"


def _scratch_read(state: CodegenState, sname: str) -> str:
    """Read expression for a scratch buffer, slicing if padded for TPU."""
    sl = state.device_function.scratch_read_slice(sname)
    return f"{sname}[{sl}]" if sl else f"{sname}[...]"


def _scratch_write_stmt(state: CodegenState, sname: str, val: ast.AST) -> ast.AST:
    """Write statement for a scratch buffer, slicing if padded for TPU.

    Always dereferences source refs with [...] or slice to avoid
    "Cannot store a Ref into another Ref" errors.
    """
    sl = state.device_function.scratch_read_slice(sname)
    idx = sl or "..."
    # Always dereference source -- it may be a scratch ref
    if isinstance(val, ast.Name):
        src_sl = state.device_function.scratch_read_slice(val.id)
        val = expr_from_string(f"{val.id}[{src_sl}]" if src_sl else f"{val.id}[...]")
    return statement_from_string(f"{sname}[{idx}] = {{val}}", val=val)


def _resolve_dim_size(
    s: object,
    env: CompileEnvironment,
    config: Config,
) -> int | None:
    """Resolve a tensor-dim size to a concrete int from ``config``, else ``None``.

    Handles a single tile dim via ``resolve_block_id`` and ``reshape``-merged
    dims (a sympy product/sum/power of block symbols) by substituting each block
    size. The ``int(s)`` fallback would otherwise return the full-extent size
    hint and over-size loop-carried scratch.
    """
    bid = env.resolve_block_id(s)
    if bid is not None:
        bs = env.block_sizes[bid].from_config(config)
        return bs if isinstance(bs, int) else None

    if isinstance(s, int):
        return s
    expr = _symint_sympy_expr(s) if isinstance(s, torch.SymInt) else s
    if not isinstance(expr, sympy.Expr):
        return None
    if expr.is_Integer:
        return int(expr)

    block_mapping, non_block_symbols = find_block_size_symbols(expr)
    if non_block_symbols:
        return None
    subs: dict[sympy.Symbol, sympy.Integer] = {}
    for symbol, block_id in block_mapping.items():
        bs = env.block_sizes[block_id].from_config(config)
        if not isinstance(bs, int):
            return None
        subs[symbol] = sympy.Integer(bs)
    resolved = expr.xreplace(subs)
    return int(resolved) if resolved.is_Integer else None


def _resolve_shape(
    proxy: torch.Tensor,
    env: CompileEnvironment,
    config: Config,
) -> tuple[int, ...]:
    """Resolve symbolic tile sizes to concrete block sizes from config."""
    resolved = []
    for s in proxy.shape:
        size = _resolve_dim_size(s, env, config)
        if size is not None:
            resolved.append(size)
        else:
            resolved.append(int(s))
    return tuple(resolved)


def _setup_loop_carried_state(
    state: CodegenState,
    args: list[ast.AST],
    proxy_args: list[object],
    env: CompileEnvironment,
) -> tuple[list[str], list[object], set[int]]:
    """Set up scratch VMEM buffers for loop-carried state.

    Returns (scratch_names, result_vars, carried) where:
    - scratch_names[i] is the scratch buffer name for arg i (empty if not carried)
    - result_vars contains (result_name, scratch_name) tuples for carried tensors
    - carried is the set of carried arg indices
    """
    carried = _loop_carried_indices(state, len(args))
    scratch_names: list[str] = []
    result_vars: list[object] = []

    for i, (arg_ast, proxy) in enumerate(zip(args, proxy_args, strict=True)):
        if i not in carried:
            scratch_names.append("")
            continue
        if isinstance(proxy, torch.Tensor):
            assert isinstance(arg_ast, ast.Name)
            shape = _resolve_shape(proxy, env, state.config)
            dtype = proxy.dtype
            scratch_name = state.device_function.register_scratch(
                shape, dtype, name_hint=f"scratch_{i}"
            )
            # Initialize scratch with the arg value.
            state.add_statement(_scratch_write_stmt(state, scratch_name, arg_ast))
            scratch_names.append(scratch_name)

            # Result will be read after loop
            result_name = state.device_function.new_var(f"state_{i}")
            result_vars.append((result_name, scratch_name))
        else:
            scratch_names.append("")
            result_vars.append(arg_ast)

    return scratch_names, result_vars, carried


def _emit_nonlocal_scratch_declarations(
    state: CodegenState,
    body_stmts: list[ast.AST],
) -> None:
    """Insert ``nonlocal <scratch>`` at the top of the closure body.

    Without ``nonlocal``, an assignment like ``scratch = scratch[...]`` inside
    a fori_loop/emit_pipeline closure makes ``scratch`` local to the entire
    function, causing an UnboundLocalError on the RHS read.

    We emit nonlocal for *all* VMEM scratch args, not just the current loop's
    carried state, because an outer loop body may contain ``scratch = scratch[...]``
    from a nested inner loop's ``_read_final_loop_state``.
    """
    names = [
        s.name for s in state.device_function._scratch_args if s.scratch_type == "vmem"
    ]
    if names:
        body_stmts.insert(0, ast.Nonlocal(names=names))


def _remap_args_to_scratch(
    args: list[ast.AST],
    scratch_names: list[str],
    state: CodegenState,
) -> list[ast.AST]:
    """Remap loop args to scratch reads for loop-carried state."""
    body_args = [*args]
    for i, sname in enumerate(scratch_names):
        if sname:
            body_args[i] = expr_from_string(_scratch_read(state, sname))
    return body_args


def _write_back_loop_carried(
    state: CodegenState,
    scratch_names: list[str],
    carried: set[int],
    graph_results: object,
) -> None:
    """Write updated loop-carried values back to scratch after body codegen."""
    if isinstance(graph_results, list):
        scratch_output_names = [
            s for i, s in enumerate(scratch_names) if s and i in carried
        ]
        for sname, result in zip(scratch_output_names, graph_results, strict=True):
            if isinstance(result, ast.AST):
                state.codegen.add_statement(_scratch_write_stmt(state, sname, result))


def _read_final_loop_state(
    state: CodegenState,
    result_vars: list[object],
) -> list[ast.AST] | None:
    """After loop: read final loop-carried state from scratch."""
    if not result_vars:
        return None
    final_results: list[ast.AST] = []
    for rv in result_vars:
        if isinstance(rv, tuple):
            result_name, scratch_name = rv
            state.add_statement(
                statement_from_string(
                    f"{result_name} = {_scratch_read(state, scratch_name)}"
                )
            )
            final_results.append(expr_from_string(result_name))
        else:
            assert isinstance(rv, ast.AST)
            final_results.append(rv)
    return final_results


def _emit_inner_loop_offset_indices(
    state: CodegenState,
    strategy: object,
    block_ids: list[int],
    block_size_vars: list[str],
    begin_exprs: list[str],
    iter_step_exprs: list[str],
    loop_index_exprs: list[str],
    env: CompileEnvironment,
    body_stmts: list[ast.AST],
) -> None:
    """Emit ``offset_<bid> = …`` and ``indices_<bid> = …`` at the inner-loop
    body prologue, using the canonical names from ``strategy``.

    Used by ``_codegen_emit_pipeline`` and ``_codegen_fori_loop`` so kernel
    code that references ``tile.index`` (lowered to ``indices_<bid>``) or
    ``pl.ds`` offsets (``offset_<bid>``) sees defined symbols regardless of
    whether the inner block is divisible.  Both vars are allocated
    ``dce=True``, so unused emissions are pruned downstream.

    Args:
        loop_index_exprs: Per-block-id expression for the inner-loop iteration
            index (``_helion_compat_pipeline_indices[i]`` for emit_pipeline; the fori_loop
            variable like ``_j`` for fori_loop).  Combined with ``begin_exprs``
            and ``iter_step_exprs`` to form the absolute start of the tile.
    """
    for i, bid in enumerate(block_ids):
        offset_name = strategy.offset_var(bid)  # type: ignore[attr-defined]
        index_name = strategy.index_var(bid)  # type: ignore[attr-defined]
        idx_expr = env.backend.loop_index_expr(
            offset_name, block_size_vars[i], env.index_type(), axis=0
        )
        body_stmts.extend(
            [
                statement_from_string(
                    f"{offset_name} = ({begin_exprs[i]}) + "
                    f"({loop_index_exprs[i]}) * ({iter_step_exprs[i]})"
                ),
                statement_from_string(f"{index_name} = {idx_expr}"),
            ]
        )


def _setup_inner_loop_masks(
    state: CodegenState,
    strategy: object,
    block_ids: list[int],
    block_size_vars: list[str],
    env: CompileEnvironment,
    body_stmts: list[ast.AST],
    offset_expr_fn: Callable[[int, str], str],
    aligned_dim: dict[int, int] | None = None,
) -> bool:
    """Set up mask variables for inner-loop block_ids.

    Args:
        offset_expr_fn: Given (block_id_index, block_size_var), returns a string
            expression for the per-element offset (e.g. "_j * bs + jnp.arange(bs)").

    Returns True if any mask requires explicit indices.
    """
    aligned_dim = aligned_dim or {}
    needs_explicit = False
    if hasattr(strategy, "_setup_mask"):
        for i, bid in enumerate(block_ids):
            offset_var = state.device_function.new_var(f"offset_{bid}")
            if bid in aligned_dim:
                # Two-sided validity mask for an aligned-enclosing dynamic row
                # tile: the load over-reads [a_start, begin) and [end, a_end), so
                # mask both ends.  The relative offset is measured from a_start.
                sublane = aligned_dim[bid]
                begin, end = _get_loop_begin_and_end(state, i)
                a_start = f"(({begin}) - ({begin}) % {sublane})"
                mask_var = strategy.fn.new_var(f"mask_{bid}", dce=True)  # pyrefly: ignore[missing-attribute]
                strategy.mask_vars[bid] = mask_var  # pyrefly: ignore[missing-attribute]
                needs_explicit = True
                body_stmts.extend(
                    [
                        statement_from_string(
                            f"{offset_var} = {offset_expr_fn(i, block_size_vars[i])}"
                        ),
                        statement_from_string(
                            f"{mask_var} = (({a_start}) + ({offset_var}) "
                            f">= ({begin})) & (({a_start}) + ({offset_var}) "
                            f"< ({end}))"
                        ),
                    ]
                )
                continue
            block_value = state.device_function.resolved_block_size(bid)
            assert isinstance(block_value, int)
            numel_expr = _get_loop_numel(state, i)
            mask_stmt = strategy._setup_mask(
                state, bid, block_value, offset_var, numel_expr
            )
            if mask_stmt is not None:
                needs_explicit = True
                body_stmts.extend(
                    [
                        statement_from_string(
                            f"{offset_var} = {offset_expr_fn(i, block_size_vars[i])}"
                        ),
                        mask_stmt,
                    ]
                )
    return needs_explicit


PRE_BROADCAST_SIZE = 128


def _apply_pre_broadcast_transform(
    state: CodegenState,
    graph: torch.fx.Graph,
    carried: set[int],
    proxy_args: list[object],
    scratch_names: list[str],
    args: list[ast.AST],
    block_ids: list[int],
    env: CompileEnvironment,
) -> None:
    """Shared pre-broadcast transform for emit_pipeline and fori_loop codegen.

    On TPU, implicit broadcast an array of (block, 1) is significantly
    slower than pre-expanding them to (block, 128) and using explicit
    jnp.tile at the point of use. This is because TPU hardware can execute
    element-wise ops on same-shaped tiles much more efficiently than ops that
    require implicit broadcast across the trailing dimension.

    This transform detects loop-carried scratch buffers that participate in
    such broadcasts (via subscript[..., None] followed by an op with a
    wider-dimensioned sibling), appends a trailing PRE_BROADCAST_SIZE (128)
    dimension to their scratch shapes, and rewrites the FX graph so that:

    - The subscript[..., None] unsqueezes become identity (the trailing dim
      is already present in the scratch).
    - A _pre_broadcast_tile op is inserted where the narrow (128-wide)
      value needs to match a wider dimension (e.g. head_dim=256), generating
      jnp.tile(tensor, block_size // 128) in the output code.
    - Lower-rank values (e.g. reduction results) get an unsqueeze to [..., 1]
      so JAX broadcasting against the [..., 128] scratch still works.

    The transform is gated by the pallas_pre_broadcast config flag and only
    applies when all broadcast target dimensions are multiples of 128.
    """
    candidates = _find_pre_broadcast_candidates(
        graph, carried, proxy_args, env, state.config
    )
    if not candidates:
        return
    pre_broadcast_nodes = _compute_pre_broadcast_nodes(graph, candidates, proxy_args)
    placeholders = list(graph.find_nodes(op="placeholder"))
    for i, proxy in enumerate(proxy_args):
        if (
            i in carried
            and isinstance(proxy, torch.Tensor)
            and i < len(placeholders)
            and placeholders[i].name in pre_broadcast_nodes
            and i not in candidates
        ):
            candidates[i] = placeholders[i]
    _apply_pre_broadcast_to_scratch(state, candidates, scratch_names, args)
    _rewrite_outer_subscripts_for_pre_broadcast(state.fx_node, candidates, state.config)
    _annotate_pre_broadcast(graph, pre_broadcast_nodes, block_ids, env, state.config)


def _find_pre_broadcast_candidates(
    graph: torch.fx.Graph,
    carried: set[int],
    proxy_args: list[object],
    env: CompileEnvironment,
    config: Config,
) -> dict[int, torch.fx.Node]:
    """Find loop-carried tensor args that are broadcast via subscript[..., None].

    Returns a dict mapping carried arg index to the placeholder node.
    """
    from ...language.view_ops import subscript as _subscript_op

    placeholders = list(graph.find_nodes(op="placeholder"))
    candidates: dict[int, torch.fx.Node] = {}
    for i, proxy in enumerate(proxy_args):
        if i not in carried:
            continue
        if not isinstance(proxy, torch.Tensor):
            continue
        if i >= len(placeholders):
            continue
        ph = placeholders[i]
        if _placeholder_has_broadcast_usage(
            ph, _subscript_op, len(proxy.shape), env, config
        ):
            candidates[i] = ph
    return candidates


def _dim_concrete_size(
    dim: int | torch.SymInt,
    env: CompileEnvironment,
    config: Config,
) -> int | None:
    """Resolve a dimension size to a concrete int.

    For SymInts that correspond to block size variables, reads the configured
    block size via ``BlockSizeInfo.from_config``.
    """
    if isinstance(dim, int):
        return dim
    block_id = env.get_block_id(dim)
    if block_id is not None and block_id < len(env.block_sizes):
        val = env.block_sizes[block_id].from_config(config)
        if isinstance(val, int):
            return val
    return None


def _placeholder_has_broadcast_usage(
    ph: torch.fx.Node,
    subscript_op: object,
    orig_rank: int,
    env: CompileEnvironment,
    config: Config,
) -> bool:
    """Check if placeholder feeds into subscript[..., None] that is then broadcast.

    First finds unsqueeze nodes (subscript[..., None]) reachable from the
    placeholder through same-rank ops.  Then checks whether any unsqueeze
    result is consumed by an op whose sibling arg has a wider last dimension,
    confirming an actual broadcast.  All broadcast target dimensions must be
    multiples of PRE_BROADCAST_SIZE for the optimization to be valid.
    """
    unsqueeze_nodes: list[torch.fx.Node] = []
    worklist = [ph]
    visited: set[str] = set()
    while worklist:
        node = worklist.pop()
        if node.name in visited:
            continue
        visited.add(node.name)
        for user in node.users:
            if user.op == "call_function" and user.target is subscript_op:
                idx = user.args[1] if len(user.args) > 1 else None
                if isinstance(idx, (list, tuple)) and len(idx) > 0 and idx[-1] is None:
                    unsqueeze_nodes.append(user)
            if user.op == "call_function":
                user_val = user.meta.get("val", None)
                if (
                    isinstance(user_val, torch.Tensor)
                    and len(user_val.shape) == orig_rank
                ):
                    worklist.append(user)

    if not unsqueeze_nodes:
        return False

    found_broadcast = False
    for unsq in unsqueeze_nodes:
        for user in unsq.users:
            if user.op != "call_function":
                continue
            for arg in user.args:
                if not isinstance(arg, torch.fx.Node) or arg is unsq:
                    continue
                arg_val = arg.meta.get("val", None)
                if not isinstance(arg_val, torch.Tensor) or len(arg_val.shape) < 1:
                    continue
                arg_last = arg_val.shape[-1]
                if isinstance(arg_last, int) and arg_last == 1:
                    continue
                size = _dim_concrete_size(arg_last, env, config)
                if size is not None and size % PRE_BROADCAST_SIZE != 0:
                    return False
                found_broadcast = True
    return found_broadcast


def _compute_pre_broadcast_nodes(
    graph: torch.fx.Graph,
    candidates: dict[int, torch.fx.Node],
    proxy_args: list[object],
) -> set[str]:
    """Compute the set of FX node names whose runtime shape becomes [.., PRE_BROADCAST_SIZE].

    Starts from candidate placeholders and propagates through _new_var copies,
    subscript unsqueezes, and element-wise ops whose FX shape has the same rank
    as the candidate (because at runtime the trailing PRE_BROADCAST_SIZE dimension
    is carried along).
    """
    from collections import deque

    from ...language.view_ops import subscript as _subscript_op

    pre_broadcast_nodes: set[str] = set()
    placeholders = list(graph.find_nodes(op="placeholder"))

    candidate_ranks: set[int] = set()
    for arg_idx in candidates:
        proxy = proxy_args[arg_idx]
        if isinstance(proxy, torch.Tensor):
            candidate_ranks.add(len(proxy.shape))

    node_by_name: dict[str, torch.fx.Node] = {n.name: n for n in graph.nodes}

    def _is_forward_candidate(node: torch.fx.Node) -> bool:
        if node.name in pre_broadcast_nodes or node.op != "call_function":
            return False
        if node.target is _new_var and len(node.args) >= 1:
            arg0 = node.args[0]
            if isinstance(arg0, torch.fx.Node) and arg0.name in pre_broadcast_nodes:
                return True
        if node.target is _subscript_op and len(node.args) >= 2:
            base = node.args[0]
            idx = node.args[1]
            if (
                isinstance(base, torch.fx.Node)
                and base.name in pre_broadcast_nodes
                and isinstance(idx, (list, tuple))
                and len(idx) > 0
                and idx[-1] is None
            ):
                return True
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor) and len(val.shape) in candidate_ranks:
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and arg.name in pre_broadcast_nodes:
                    return True
        return False

    # Forward pass: propagate from candidate placeholders through users
    worklist: deque[torch.fx.Node] = deque()
    for arg_idx in candidates:
        ph = placeholders[arg_idx]
        pre_broadcast_nodes.add(ph.name)
        worklist.append(ph)

    while worklist:
        node = worklist.popleft()
        for user in node.users:
            if _is_forward_candidate(user):
                pre_broadcast_nodes.add(user.name)
                worklist.append(user)

    # Backward pass: propagate back through _new_var (loop-carried copies)
    # to find placeholder sources that should also be pre-broadcast.
    backward_worklist: deque[torch.fx.Node] = deque(
        node_by_name[name]
        for name in pre_broadcast_nodes
        if node_by_name[name].op == "call_function"
    )
    while backward_worklist:
        node = backward_worklist.popleft()
        for arg in node.args:
            if not isinstance(arg, torch.fx.Node) or arg.name in pre_broadcast_nodes:
                continue
            if arg.op != "call_function" or arg.target is not _new_var:
                continue
            a_val = arg.meta.get("val", None)
            if (
                not isinstance(a_val, torch.Tensor)
                or len(a_val.shape) not in candidate_ranks
            ):
                continue
            pre_broadcast_nodes.add(arg.name)
            backward_worklist.append(arg)
            # Follow _new_var chain to its placeholder source
            src = arg.args[0]
            if (
                isinstance(src, torch.fx.Node)
                and src.op == "placeholder"
                and src.name not in pre_broadcast_nodes
            ):
                src_val = src.meta.get("val", None)
                if (
                    isinstance(src_val, torch.Tensor)
                    and len(src_val.shape) in candidate_ranks
                ):
                    pre_broadcast_nodes.add(src.name)

    return pre_broadcast_nodes


def _apply_pre_broadcast_to_scratch(
    state: CodegenState,
    candidates: dict[int, torch.fx.Node],
    scratch_names: list[str],
    args: list[ast.AST],
) -> set[str]:
    """Modify scratch shapes for pre-broadcast candidates.

    Appends PRE_BROADCAST_SIZE to the scratch shape (e.g. (a,b) → (a,b,128)).
    For scratches NOT from hl.full/hl.zeros (where the init was already emitted
    without the extra dim), rewrites the existing init statement to broadcast.
    Returns the set of scratch names that were modified.
    """
    modified_scratches: set[str] = set()
    for arg_idx in candidates:
        sname = scratch_names[arg_idx]
        if not sname:
            continue
        for sa in state.device_function._scratch_args:
            if sa.name == sname:
                sa.shape = (*sa.shape, PRE_BROADCAST_SIZE)
                modified_scratches.add(sname)
                # If scratch != arg, the init `scratch[...] = arg[...]` was
                # emitted without the trailing dim. Rewrite it to broadcast.
                arg_ast = args[arg_idx]
                if isinstance(arg_ast, ast.Name) and arg_ast.id != sname:
                    _rewrite_scratch_init_for_pre_broadcast(state, sname, arg_ast.id)
                break
    return modified_scratches


def _rewrite_scratch_init_for_pre_broadcast(
    state: CodegenState,
    scratch_name: str,
    arg_name: str,
) -> None:
    """Find and rewrite `scratch[...] = arg[...]` to broadcast the N-D arg to (N+1)-D."""
    stmts = state.codegen.statements_stack[-1]
    replacement = statement_from_string(
        f"{scratch_name}[...] = jnp.broadcast_to("
        f"{arg_name}[..., None], {scratch_name}.shape)"
    )
    for i, stmt in enumerate(stmts):
        src = ast.unparse(stmt) if isinstance(stmt, ast.AST) else str(stmt)
        if f"{scratch_name}[" in src and f"{arg_name}[" in src:
            stmts[i] = replacement
            return
    stmts.append(replacement)


def _rewrite_outer_subscripts_for_pre_broadcast(
    for_loop_node: torch.fx.Node | None,
    candidates: dict[int, torch.fx.Node],
    config: object,
) -> None:
    """Rewrite outer-scope subscript[..., None] to identity for pre-broadcast results.

    After pre-broadcast, loop-carried values read from scratch have an extra
    trailing PRE_BROADCAST_SIZE dim. The outer graph's subscript(val, [..., None])
    would add yet another dim. Instead, rewrite to identity slicing.
    """
    from torch._inductor.virtualized import V

    from ...language.view_ops import subscript as _subscript_op
    from ..inductor_lowering import FakeGraphLowering
    from ..inductor_lowering import compile_lock
    from ..inductor_lowering import prepare_node_lowering

    if for_loop_node is None:
        return

    # The _for_loop result is a tuple. Each result index i corresponds
    # to proxy_args[i]. candidates maps arg index → inner placeholder.
    # Track which result indices are pre-broadcast.
    pre_broadcast_result_indices = set(candidates.keys())

    # Find getitem nodes that extract pre-broadcast results
    pre_broadcast_outer_nodes: set[str] = set()
    for user in for_loop_node.users:
        if user.op == "call_function" and user.target is operator.getitem:
            idx = user.args[1]
            if isinstance(idx, int) and idx in pre_broadcast_result_indices:
                pre_broadcast_outer_nodes.add(user.name)
                # Follow through _phi nodes
                pre_broadcast_outer_nodes.update(
                    phi_user.name for phi_user in user.users
                )

    # Rewrite subscript[:, :, None] → [:, :] for pre-broadcast outer nodes
    reshaped: list[torch.fx.Node] = []
    reshaped_bases: set[str] = set()
    outer_graph = for_loop_node.graph
    for node in outer_graph.nodes:
        if node.op != "call_function" or node.target is not _subscript_op:
            continue
        base = node.args[0]
        idx = node.args[1]
        if (
            isinstance(base, torch.fx.Node)
            and base.name in pre_broadcast_outer_nodes
            and isinstance(idx, (list, tuple))
            and len(idx) > 0
            and idx[-1] is None
        ):
            new_idx = [i for i in idx if i is not None]
            node.args = (base, new_idx)
            base_val = base.meta.get("val", None)
            if isinstance(base_val, torch.Tensor):
                if base.name not in reshaped_bases:
                    new_val = base_val.new_empty([*base_val.shape, PRE_BROADCAST_SIZE])
                    base.meta["val"] = new_val
                    reshaped_bases.add(base.name)
                    reshaped.append(base)
                node.meta["val"] = base.meta["val"].new_empty(
                    list(base.meta["val"].shape)
                )
                reshaped.append(node)

    # Insert _pre_broadcast_tile where pre-broadcast outer nodes feed wider-dim ops.
    # First, propagate pre-broadcast status transitively through indirect consumers.
    # After rewriting subscript[:, :, None] → subscript[:, :], downstream nodes
    # (e.g. add, rsqrt) may still have stale meta shapes (u0, u1, 1) from trace
    # time. We identify them by checking if any arg is pre-broadcast — if so,
    # the node is also pre-broadcast (its real last dim is PRE_BROADCAST_SIZE).
    all_pre_broadcast_outer: set[str] = set(pre_broadcast_outer_nodes)
    all_pre_broadcast_outer.update(node.name for node in reshaped)
    for node in outer_graph.nodes:
        if node.op != "call_function" or node.name in all_pre_broadcast_outer:
            continue
        node_val = node.meta.get("val", None)
        if not isinstance(node_val, torch.Tensor) or len(node_val.shape) < 2:
            continue
        last_dim = node_val.shape[-1]
        if isinstance(last_dim, torch.SymInt):
            continue
        last_dim_int = int(last_dim)
        if last_dim_int > PRE_BROADCAST_SIZE:
            continue
        has_pre_broadcast_arg = False
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.name in all_pre_broadcast_outer:
                arg_val = arg.meta.get("val", None)
                if isinstance(arg_val, torch.Tensor) and len(arg_val.shape) >= 2:
                    arg_last = arg_val.shape[-1]
                    if isinstance(arg_last, int) and arg_last == PRE_BROADCAST_SIZE:
                        has_pre_broadcast_arg = True
                        break
        if has_pre_broadcast_arg:
            new_shape = [*node_val.shape[:-1], PRE_BROADCAST_SIZE]
            node.meta["val"] = node_val.new_empty(new_shape)
            all_pre_broadcast_outer.add(node.name)
            reshaped.append(node)

    new_nodes: list[torch.fx.Node] = []
    for node in list(outer_graph.nodes):
        if node.op != "call_function" or node.name in all_pre_broadcast_outer:
            continue
        node_val = node.meta.get("val", None)
        if not isinstance(node_val, torch.Tensor) or len(node_val.shape) < 2:
            continue
        last_dim = node_val.shape[-1]
        last_dim_is_sym = isinstance(last_dim, torch.SymInt)
        if not last_dim_is_sym and int(last_dim) <= PRE_BROADCAST_SIZE:
            continue
        args_list = list(node.args)
        changed = False
        for ai, arg in enumerate(args_list):
            if not isinstance(arg, torch.fx.Node):
                continue
            if arg.name not in all_pre_broadcast_outer:
                continue
            arg_val = arg.meta.get("val", None)
            if not isinstance(arg_val, torch.Tensor):
                continue
            if not (
                isinstance(arg_val.shape[-1], int)
                and arg_val.shape[-1] == PRE_BROADCAST_SIZE
            ):
                continue
            with outer_graph.inserting_before(node):
                tiled = outer_graph.call_function(
                    _pre_broadcast_tile,
                    args=(arg, last_dim),
                )
            tiled.meta = {
                **arg.meta,
                "val": arg_val.new_empty([*arg_val.shape[:-1], last_dim]),
            }
            new_nodes.append(tiled)
            args_list[ai] = tiled
            changed = True
        if changed:
            node.args = tuple(args_list)

    # Re-prepare lowerings for modified outer nodes
    all_to_prepare = reshaped + new_nodes
    if all_to_prepare:
        with compile_lock:
            graph_lowering = FakeGraphLowering()
            with V.set_graph_handler(graph_lowering):
                for node in all_to_prepare:
                    if node.op == "call_function":
                        with node.meta["location"]:
                            prepare_node_lowering(graph_lowering, node)


def _annotate_pre_broadcast(
    graph: torch.fx.Graph,
    pre_broadcast_nodes: set[str],
    inner_block_ids: list[int],
    env: CompileEnvironment,
    config: object,
) -> None:
    """FX graph rewrite for pre-broadcast optimization.

    Appends PRE_BROADCAST_SIZE to pre-broadcast node meta shapes, rewrites
    subscript unsqueezes to identity, inserts _pre_broadcast_tile for
    wider-dim consumers, inserts unsqueezes for lower-rank non-pre-broadcast
    values feeding pre-broadcast ops, and re-prepares lowerings for all
    affected nodes.
    """
    from ...language.view_ops import subscript as _subscript_op

    new_nodes: list[torch.fx.Node] = []
    reshaped_nodes: list[torch.fx.Node] = []

    def _node_val(n: torch.fx.Node) -> torch.Tensor | None:
        v = n.meta.get("val", None)
        return v if isinstance(v, torch.Tensor) else None

    # --- Step 1: append PRE_BROADCAST_SIZE to meta shapes for pre-broadcast nodes ---
    # Skip nodes that already have PRE_BROADCAST_SIZE as last dim (subscript
    # unsqueezes with shape [..., 1] will be handled in Step 2).
    for node in graph.nodes:
        if node.name not in pre_broadcast_nodes:
            continue
        val = _node_val(node)
        if val is None:
            continue
        if isinstance(val.shape[-1], int) and val.shape[-1] == PRE_BROADCAST_SIZE:
            continue
        if isinstance(val.shape[-1], int) and val.shape[-1] == 1:
            continue
        new_val = val.new_empty([*val.shape, PRE_BROADCAST_SIZE])
        node.meta["val"] = new_val
        reshaped_nodes.append(node)

    # --- Step 2: rewrite subscript(base, [:, :, None]) → subscript(base, [:, :]) ---
    # The subscript was an unsqueeze from 2D→3D. Now the base is already 3D,
    # so we change it to an identity slice. Also update the subscript's meta
    # shape from [a,b,1] to [a,b,PRE_BROADCAST_SIZE] to match the base.
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not _subscript_op:
            continue
        if node.name not in pre_broadcast_nodes:
            continue
        base = node.args[0]
        idx = node.args[1]
        if (
            isinstance(base, torch.fx.Node)
            and base.name in pre_broadcast_nodes
            and isinstance(idx, (list, tuple))
            and len(idx) > 0
            and idx[-1] is None
        ):
            new_idx = [i for i in idx if i is not None]
            node.args = (base, new_idx)
            base_val = _node_val(base)
            if base_val is not None:
                node.meta["val"] = base_val.new_empty(list(base_val.shape))

    # --- Step 3: insert _pre_broadcast_tile where pre-broadcast values feed wider-dim ops ---
    for node in list(graph.nodes):
        if node.op != "call_function" or node.name in pre_broadcast_nodes:
            continue
        node_val = _node_val(node)
        if node_val is None or len(node_val.shape) < 2:
            continue
        last_dim = node_val.shape[-1]
        last_dim_is_sym = isinstance(last_dim, torch.SymInt)
        if not last_dim_is_sym and int(last_dim) <= PRE_BROADCAST_SIZE:
            continue
        args_list = list(node.args)
        changed = False
        for ai, arg in enumerate(args_list):
            if not isinstance(arg, torch.fx.Node):
                continue
            if arg.name not in pre_broadcast_nodes:
                continue
            arg_val = _node_val(arg)
            if arg_val is None:
                continue
            if not (
                isinstance(arg_val.shape[-1], int)
                and arg_val.shape[-1] == PRE_BROADCAST_SIZE
            ):
                continue
            with graph.inserting_before(node):
                tiled = graph.call_function(
                    _pre_broadcast_tile,
                    args=(arg, last_dim),
                )
            tiled.meta = {
                **arg.meta,
                "val": arg_val.new_empty([*arg_val.shape[:-1], last_dim]),
            }
            new_nodes.append(tiled)
            args_list[ai] = tiled
            changed = True
        if changed:
            node.args = tuple(args_list)

    # --- Step 4: insert unsqueeze for lower-rank non-pre-broadcast values ---
    # Reductions produce rank R-1. Pre-broadcast nodes now have rank R+1
    # (with trailing 128). We unsqueeze to [..., 1] so JAX broadcast works:
    # [..., 128] op [..., 1].
    for node in list(graph.nodes):
        if node.op != "call_function" or node.name in pre_broadcast_nodes:
            continue
        node_val = _node_val(node)
        if node_val is None:
            continue
        node_rank = len(node_val.shape)
        # Check if any pre-broadcast consumer/sibling has a higher rank
        needs_unsqueeze = False
        for u in node.users:
            u_val = _node_val(u)
            if (
                u.name in pre_broadcast_nodes
                and u_val is not None
                and len(u_val.shape) > node_rank
            ):
                needs_unsqueeze = True
                break
            for ua in u.args:
                if isinstance(ua, torch.fx.Node) and ua.name in pre_broadcast_nodes:
                    ua_val = _node_val(ua)
                    if ua_val is not None and len(ua_val.shape) > node_rank:
                        needs_unsqueeze = True
                        break
            if needs_unsqueeze:
                break
        if not needs_unsqueeze:
            continue
        with graph.inserting_after(node):
            unsq = graph.call_function(
                torch.ops.aten.unsqueeze.default,
                args=(node, node_rank),
            )
        unsq.meta = {**node.meta, "val": node_val.new_empty([*node_val.shape, 1])}
        new_nodes.append(unsq)
        for user in list(node.users):
            if user is unsq:
                continue
            if user.name in pre_broadcast_nodes or any(
                isinstance(ua, torch.fx.Node) and ua.name in pre_broadcast_nodes
                for ua in user.args
            ):
                user.replace_input_with(node, unsq)

    # --- Step 5: annotate all pre-broadcast nodes ---
    for node in graph.nodes:
        if node.name in pre_broadcast_nodes:
            node.meta["pre_broadcast"] = True

    # --- Step 6: re-prepare lowerings for all affected nodes ---
    from torch._inductor.virtualized import V

    from ..inductor_lowering import FakeGraphLowering
    from ..inductor_lowering import compile_lock
    from ..inductor_lowering import prepare_node_lowering

    all_affected = new_nodes + reshaped_nodes
    with compile_lock:
        graph_lowering = FakeGraphLowering()
        with V.set_graph_handler(graph_lowering):
            for node in all_affected:
                if hasattr(node, "_erased") and node._erased:
                    continue
                if node.op == "call_function":
                    with node.meta["location"]:
                        prepare_node_lowering(graph_lowering, node)


@_decorators.codegen(_pre_broadcast_tile, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor_ast = state.ast_arg(0)
    target_size = state.proxy_arg(1)
    if isinstance(target_size, torch.SymInt):
        target_expr = state.sympy_expr(_symint_sympy_expr(target_size))
        block_id = CompileEnvironment.current().get_block_id(target_size)
        bs_var = (
            state.device_function.block_size_var(block_id)
            if block_id is not None
            else None
        )
        if bs_var:
            return expr_from_string(
                f"jnp.tile({{tensor}}, {bs_var} // {PRE_BROADCAST_SIZE})",
                tensor=tensor_ast,
            )
        return expr_from_string(
            f"jnp.tile({{tensor}}, {target_expr} // {PRE_BROADCAST_SIZE})",
            tensor=tensor_ast,
        )
    assert isinstance(target_size, int)
    factor = target_size // PRE_BROADCAST_SIZE
    if factor <= 1:
        return tensor_ast
    return expr_from_string(
        f"jnp.tile({{tensor}}, {factor})",
        tensor=tensor_ast,
    )


def _lane_tile(
    state: CodegenState, fake: torch.Tensor, dim_to_bid: dict[int, int]
) -> int | None:
    """Access extent on ``fake``'s last dim: its block size if tiled, else width."""
    lane_bid = dim_to_bid.get(fake.ndim - 1)
    if lane_bid is not None:
        rb = state.device_function.resolved_block_size(lane_bid)
        if isinstance(rb, int):
            return rb
    last = fake.shape[fake.ndim - 1]
    return last if isinstance(last, int) else None


def _loop_dim_is_dynamic(state: CodegenState, i: int) -> bool:
    """Whether loop dim ``i`` has a runtime begin and end (a fully-dynamic jagged
    tile).  The tracing-time form of is_dynamic_bound_tile, used here because the
    device loops are not registered yet.
    """
    begins, ends = state.proxy_args[1], state.proxy_args[2]
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return False
    begin = begins[i] if i < len(begins) else 0
    end = ends[i] if i < len(ends) else 0
    return not isinstance(begin, (int, torch.SymInt)) and not isinstance(
        end, (int, torch.SymInt)
    )


def _aligned_dim(
    state: CodegenState,
    env: CompileEnvironment,
    block_ids: list[int],
    loaded_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
    stored_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
) -> dict[int, int]:
    """Jagged row tiles (runtime end) that read an aligned-enclosing window.

    Maps each to the sublane S its range rounds down to.  A row aligns when a
    tensor it slices must land on a sublane tile (bf16, or f32 spanning more than
    one lane tile), or when a per-row map store carries its shared boundary.
    A DIRECT row that does not carry is omitted: it reads at the exact offset.  S
    is the largest float-tensor sublane (bf16 forces 16); tiles that carry are
    also registered for the store fold/save.
    """
    from .backend import SliceAddressing
    from .backend import _slice_addressing
    from helion._compiler.pallas.ordered_carry import CarryBoundaryTile
    from helion._compiler.pallas.ordered_carry import is_row_map_axis
    from helion._compiler.pallas.ordered_carry import needs_ordered_carry

    sublanes = [
        env.backend.sublane_tiling(t.dtype)  # pyrefly: ignore[missing-attribute]
        for t in HostFunction.current().tensor_to_origin
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    ]
    if not sublanes:
        return {}

    # Strictest addressing each row needs over the tensors it slices.
    addressing: dict[int, SliceAddressing] = {}
    for fake, _node, sub_meta in (*loaded_tensors.values(), *stored_tensors.values()):
        if not (isinstance(fake, torch.Tensor) and fake.is_floating_point()):
            continue
        dim_to_bid = _get_dim_block_ids(sub_meta, env)
        lane_block = _lane_tile(state, fake, dim_to_bid)
        for dim, dim_bid in dim_to_bid.items():
            if _slice_addressing(fake, dim, lane_block) is SliceAddressing.ALIGNED:
                addressing[dim_bid] = SliceAddressing.ALIGNED
            else:
                addressing.setdefault(dim_bid, SliceAddressing.DIRECT)

    sublane = max(sublanes)
    aligned_dim: dict[int, int] = {}
    for i, bid in enumerate(block_ids):
        if not _loop_dim_is_dynamic(state, i):
            continue  # static begin or end: not a fully-dynamic jagged tile
        carry = needs_ordered_carry(state, bid)
        direct = addressing.get(bid, SliceAddressing.ALIGNED) is SliceAddressing.DIRECT
        if direct and not carry:
            continue  # reads any offset; a plain clamped slice suffices
        if not carry and not is_row_map_axis(state, bid):
            # ALIGNED but not a map axis: a bf16 reduction over the row.  Its
            # dense bf16 output store can't be proven aligned for Mosaic (E2003),
            # so reject it cleanly here instead.  f32 reductions are DIRECT and
            # already skipped above.
            raise NotImplementedError(
                "Pallas: bf16 reduction over a jagged row is not supported yet "
                "(its dense bf16 output store cannot be proven sublane-aligned)."
            )
        aligned_dim[bid] = sublane
        if carry:
            begin, end = _get_loop_begin_and_end(state, i)
            state.device_function.carry_tiles[bid] = CarryBoundaryTile(
                block_id=bid,
                begin_var=begin,
                end_var=end,
                sublane=sublane,
            )
    return aligned_dim


def _codegen_emit_pipeline(state: CodegenState) -> object:
    """Emit inner device loops using pltpu.emit_pipeline.

    Handles both simple load->compute->store pipelines and loops with
    loop-carried state (accumulators, running max/sum) by converting
    the state into scratch VMEM buffers.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import codegen_call_with_graph
    from ..tile_strategy import EmitPipelineLoopState

    graph_info = state.get_graph(state.proxy_arg(0))
    assert isinstance(graph_info, ForLoopGraphInfo)
    assert isinstance(state.codegen, GenerateAST)

    block_ids = graph_info.block_ids
    env = CompileEnvironment.current()

    args = state.ast_args[-1]
    assert isinstance(args, list)
    assert all(isinstance(x, ast.AST) for x in args)

    # Check if we have loop-carried state (accumulators etc.)
    proxy_args = state.proxy_args[-1]
    assert isinstance(proxy_args, list)
    has_loop_state = len(args) > 0

    loaded_tensors, stored_tensors = _classify_loop_tensors(graph_info, state)

    aligned_dim = _aligned_dim(state, env, block_ids, loaded_tensors, stored_tensors)

    grid_parts, block_size_vars = _compute_grid_and_block_sizes(
        state, block_ids, env, aligned_dim
    )
    begin_exprs, iter_step_exprs, slice_size_exprs = _pallas_loop_begin_and_step_exprs(
        state, block_ids, block_size_vars, aligned_dim
    )
    # Loop end expressions (used to clamp store extents for data-dependent begins).
    end_exprs = [_get_loop_begin_and_end(state, i)[1] for i in range(len(block_ids))]

    # Pipelined tensors flow through emit_pipeline's per-iter Buffered
    # BlockSpec; the rest stay on the outer pallas_call BlockSpec
    # (escape clause `bs == as`) and are closure-read from the body.
    all_tensor_info, _vmem_shapes, pipelined_tensor_ids = _classify_pipelined_tensors(
        loaded_tensors, stored_tensors, block_ids, slice_size_exprs, env, state
    )

    # Build in_specs and out_specs
    in_tensors: list[tuple[torch.Tensor, str]] = []
    out_tensors: list[tuple[torch.Tensor, str]] = []
    in_specs: list[str] = []
    out_specs: list[str] = []
    body_params: list[str] = []
    pipeline_in_args: list[str] = []
    pipeline_out_args: list[str] = []

    # Map outer grid block_ids to program_id variable names.
    # Compute program_ids before emit_pipeline so the BlockSpec lambda
    # captures them as closure variables (like the reference pattern).
    # Use pid_info ordering (which reflects loop_order) rather than
    # grid_block_ids (which is logical order), so that program_id(g)
    # correctly maps to the block_id at grid dimension g.
    from ..device_function import DeviceFunction as _DF

    _bid_to_pid_var: dict[int, str] = {}
    device_fn = _DF.current()
    if device_fn.pid is not None:
        for g, pid in enumerate(device_fn.pid.pid_info):
            pid_var = f"_outer_pid_{g}"
            state.add_statement(
                statement_from_string(f"{pid_var} = pl.program_id({g})")
            )
            _bid_to_pid_var[pid.block_id] = pid_var

    def _make_block_spec(
        fake: torch.Tensor, subscript_meta: list[object], is_store: bool = False
    ) -> str:
        """Build a BlockSpec string for a tensor accessed in the pipeline body.

        Encodes BOTH outer grid dims (via pl.program_id) and inner pipeline
        dims into the BlockSpec lambda, so the full HBM tensor can be passed
        without pre-slicing.
        """
        from helion._compiler.pallas.ordered_carry import is_dynamic_bound_tile

        dim_to_bid = _get_dim_block_ids(subscript_meta, env)
        tensor_subscripts = _tensor_dim_subscripts(subscript_meta)
        shape = fake.shape
        block_shape_parts: list[str] = []
        lambda_parts: list[str] = []
        lambda_params: list[str] = []

        for i, _bid in enumerate(block_ids):
            param = f"_j{i}" if len(block_ids) > 1 else "_j"
            lambda_params.append(param)

        for dim_idx in range(len(shape)):
            bid = dim_to_bid.get(dim_idx)
            if bid is not None and bid in block_ids:
                # Inner pipeline dim -- tiled by pipeline grid
                bid_idx = block_ids.index(bid)
                slice_size_expr = slice_size_exprs[bid_idx]
                begin_expr = begin_exprs[bid_idx]
                iter_step_expr = iter_step_exprs[bid_idx]
                from ...language.memory_ops import _record_pad_info

                extra_pad = _compute_pipeline_or_dma_extra_pad(
                    begin_expr, bid, env, state, bid_idx
                )
                _record_pad_info(state, fake, dim_idx, bid, extra_pad)
                begin_is_zero = begin_expr == "0"
                end_expr = end_exprs[bid_idx]
                dim_size = shape[dim_idx]
                # Whether this loop spans the ENTIRE backing tensor dim, i.e.
                # ``[0, dim_size)`` with a compile-time-constant extent. Only
                # then is a full-block store safe: a partial final tile overruns
                # past ``dim_size`` into padding, which the host-side pad handles.
                # For any sub-range -- a jagged ``hl.tile(start, end)``, a
                # ``hl.tile(0, dynamic_end)``, or even a static ``hl.tile(0, k)``
                # with ``k < dim_size`` -- a full-block store would overrun into
                # live rows of the tensor, so the extent must be clamped.
                covers_full_dim = (
                    begin_is_zero
                    and _is_static_int(end_expr)
                    and isinstance(dim_size, int)
                    and int(end_expr) == dim_size
                )
                # Loads need a dynamic ``pl.ds`` only for a non-zero begin (a
                # block-aligned index can't express an arbitrary start; the
                # over-read past ``end`` is zeroed by the inner-loop mask).
                # Stores need it whenever they target a sub-range (not the full
                # dim), so the extent can be clamped and a partial final tile
                # does not overrun live rows. Full-dim, from-zero loops keep the
                # original block-index codegen (no change).
                if not begin_is_zero or (is_store and not covers_full_dim):
                    # Dynamic ``pl.ds`` at the true element offset, with a
                    # ``pl.BoundedSlice`` block shape (required for ds-style
                    # index maps). Lifts the "emit_pipeline fails on unaligned
                    # dims" limitation so data-dependent tile loops can pipeline.
                    block_shape_parts.append(f"pl.BoundedSlice({slice_size_expr})")
                    start_expr = (
                        f"({begin_expr}) + ({lambda_params[bid_idx]}) "
                        f"* ({iter_step_expr})"
                    )
                    if is_store and bid not in state.device_function.carry_tiles:
                        # Clamp the store extent to min(block, end - offset) so a
                        # short final tile writes only its valid rows
                        # [begin, end) instead of overrunning into the next
                        # sub-range (which would corrupt it under cross-iteration
                        # double-buffering, and is wasteful for large blocks).
                        #
                        # For ordered carry tiles (`bid in [...].carry_tiles`),
                        # clamping is skipped: fixed sublane-aligned windows are
                        # required for carry propagation, and zeroing/masking of
                        # unowned rows is safely handled by the ordered carry logic.
                        size_expr = (
                            f"jnp.minimum({slice_size_expr}, "
                            f"({end_exprs[bid_idx]}) - ({start_expr}))"
                        )
                    else:
                        size_expr = slice_size_expr
                    if bid in state.device_function.carry_tiles:
                        sublane = state.device_function.carry_tiles[bid].sublane
                        start_expr = f"pl.multiple_of({start_expr}, {sublane})"
                    lambda_parts.append(f"pl.ds({start_expr}, {size_expr})")
                else:
                    # Static, from-zero loop: a block-aligned index is exact.
                    # Identical to the pre-existing codegen.
                    block_shape_parts.append(slice_size_expr)
                    if iter_step_expr == slice_size_expr:
                        lambda_parts.append(lambda_params[bid_idx])
                    else:
                        lambda_parts.append(
                            f"(({begin_expr}) + ({lambda_params[bid_idx]}) * ({iter_step_expr})) // ({slice_size_expr})"
                        )
            elif bid is not None and bid in _bid_to_pid_var:
                # Outer grid dim -- select via captured program_id variable
                pid_var = _bid_to_pid_var[bid]
                bs_var = state.device_function.block_size_var(bid)
                if bs_var:
                    block_shape_parts.append(bs_var)
                    from ...language.memory_ops import _record_pad_info

                    extra_pad = _compute_pipeline_or_dma_extra_pad(
                        _active_loop_begin_expr(state, bid), bid, env, state
                    )
                    _record_pad_info(state, fake, dim_idx, bid, extra_pad)
                else:
                    block_shape_parts.append(str(int(shape[dim_idx])))
                lambda_parts.append(pid_var)
            elif bid is not None and is_dynamic_bound_tile(state, bid):
                # Jagged row tile from an inner pipeline.  pl.multiple_of is
                # assume_multiple: it suppresses the tiled-row alignment check.
                # Safe because the begin is rounded to the sublane in the dim's
                # own loop, and a DIRECT f32 single-lane-tile row reads
                # contiguously.  Always emitted, as sibling loops reference the
                # same jagged dim.  Must precede the outer-non-grid branch.
                block_m = state.device_function.block_size_var(bid)
                offset_v = state.codegen.offset_var(bid)
                sublane = env.backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
                block_shape_parts.append(f"pl.BoundedSlice({block_m})")
                lambda_parts.append(
                    f"pl.ds(pl.multiple_of({offset_v}, {sublane}), {block_m})"
                )
            elif bid is not None and state.codegen.active_device_loops.get(bid):
                # Outer non-grid device loop -- the HBM ref is pre-sliced via
                # ``.at[pl.ds(offset, bs)]`` (see _make_hbm_slice), so the
                # BlockSpec sees an already-sliced ref of size ``bs`` along
                # this dim. Use the full sliced size with a constant index.
                bs_var = state.device_function.block_size_var(bid)
                if bs_var:
                    block_shape_parts.append(bs_var)
                else:
                    block_shape_parts.append(str(int(shape[dim_idx])))
                lambda_parts.append("0")
            else:
                idx_meta = _subscript_at_dim(tensor_subscripts, dim_idx)
                from helion._utils import is_scalar_index

                if is_scalar_index(idx_meta):
                    block_shape_parts.append("1")
                    if isinstance(idx_meta, torch.Tensor):
                        var_name = state.device_function.tensor_arg(idx_meta).name
                        lambda_parts.append(var_name)
                    else:
                        lambda_parts.append(
                            state.device_function.literal_expr(idx_meta)
                        )
                else:
                    block_shape_parts.append(str(int(shape[dim_idx])))
                    lambda_parts.append("0")

        block_shape_str = ", ".join(block_shape_parts)
        lambda_body = ", ".join(lambda_parts)
        lambda_param_str = ", ".join(lambda_params)
        return (
            f"pl.BlockSpec(({block_shape_str},), "
            f"lambda {lambda_param_str}: ({lambda_body},), "
            f"pipeline_mode=pl.Buffered(buffer_count=2))"
        )

    def _make_load_block_spec(fake: torch.Tensor, subscript_meta: list[object]) -> str:
        """BlockSpec for a pipelined input (full-block ``pl.ds``; mask zeroes over-read)."""
        return _make_block_spec(fake, subscript_meta, is_store=False)

    def _make_store_block_spec(fake: torch.Tensor, subscript_meta: list[object]) -> str:
        """BlockSpec for a pipelined output (clamped ``pl.ds`` extent on dynamic bounds)."""
        return _make_block_spec(fake, subscript_meta, is_store=True)

    def _make_hbm_slice(
        fake: torch.Tensor, hbm_name: str, subscript_meta: list[object]
    ) -> str:
        """Slice the HBM ref for outer non-grid device loop dims.

        Outer grid dims are handled by BlockSpec via captured ``program_id``,
        and inner pipeline dims are handled by BlockSpec via the iteration
        lambda — so this only adds ``pl.ds(offset, bs)`` slices for outer
        device loops whose offset is a closure variable in this scope.

        Dynamic (jagged) row tiles are skipped here: they are addressed by an
        aligned-enclosing BoundedSlice in the BlockSpec (their group start is
        not block-aligned, so a plain ``pl.ds`` pre-slice would be unaddressable
        for bf16).
        """
        from helion._compiler.pallas.ordered_carry import is_dynamic_bound_tile

        dim_to_bid = _get_dim_block_ids(subscript_meta, env)
        shape = fake.shape
        parts: list[str] = []
        needs_slice = False
        for dim_idx in range(len(shape)):
            bid = dim_to_bid.get(dim_idx)
            if (
                bid is not None
                and bid not in block_ids
                and bid not in _bid_to_pid_var
                and state.codegen.active_device_loops.get(bid)
                and not is_dynamic_bound_tile(state, bid)
            ):
                offset = state.codegen.offset_var(bid)
                bs_var = state.device_function.block_size_var(bid)
                if bs_var:
                    parts.append(f"pl.ds({offset}, {bs_var})")
                    needs_slice = True
                else:
                    parts.append(":")
            else:
                parts.append(":")
        if not needs_slice:
            return hbm_name
        return f"{hbm_name}.at[{', '.join(parts)}]"

    # --- Handle loop-carried state as scratch VMEM buffers ---
    scratch_names: list[str] = []
    result_vars: list[object] = []
    carried: set[int] = set()
    if has_loop_state:
        scratch_names, result_vars, carried = _setup_loop_carried_state(
            state, args, proxy_args, env
        )

    # --- Pre-broadcast transform: append PRE_BROADCAST_SIZE to scratch shapes
    #     to avoid costly implicit broadcasts on TPU. ---
    if state.config.get("pallas_pre_broadcast", False) and has_loop_state:
        _apply_pre_broadcast_transform(
            state,
            graph_info.graph,
            carried,
            proxy_args,
            scratch_names,
            args,
            block_ids,
            env,
        )

    from ..device_function import PallasMemorySpace

    for fake, _tensor_node, _sub_meta in loaded_tensors.values():
        if id(fake) in pipelined_tensor_ids:
            state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
    for fake, _tensor_node, _sub_meta in stored_tensors.values():
        if id(fake) in pipelined_tensor_ids:
            state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM

    for key, (fake, _tensor_node, sub_meta) in loaded_tensors.items():
        if key in stored_tensors:
            continue  # Handle as output instead
        if id(fake) not in pipelined_tensor_ids:
            continue
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        in_tensors.append((fake, hbm_name))
        in_specs.append(_make_load_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        pipeline_in_args.append(_make_hbm_slice(fake, hbm_name, sub_meta))

    for fake, _tensor_node, sub_meta in stored_tensors.values():
        if id(fake) not in pipelined_tensor_ids:
            continue
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        out_tensors.append((fake, hbm_name))
        out_specs.append(_make_store_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        pipeline_out_args.append(_make_hbm_slice(fake, hbm_name, sub_meta))

    # Build the body function
    body_fn_name = state.device_function.new_var("_pipeline_body")
    body_stmts: list[ast.AST] = [
        # JAX commit 6cc8faf8 (https://github.com/jax-ml/jax/commit/6cc8faf8) introduced
        # a PipelineStep object to wrap emit_pipeline's indices, which originally were
        # just a tuple.
        # TODO(cota): Eventually remove _helion_compat_pipeline_indices once older JAX
        # versions without PipelineStep are no longer supported.
        statement_from_string(
            "_helion_compat_pipeline_indices = _pipeline_indices "
            "if isinstance(_pipeline_indices, (tuple, list)) "
            "else _pipeline_indices.index"
        )
    ]

    # Build block_id_to_info for the pipeline state
    block_id_to_info = _loop_dim_infos(state, block_ids, env)

    strategy = _find_strategy(state, block_ids)
    # Emit offset_<bid>/indices_<bid> at the body prologue.
    _emit_inner_loop_offset_indices(
        state,
        strategy,
        block_ids,
        block_size_vars,
        begin_exprs,
        iter_step_exprs,
        [f"_helion_compat_pipeline_indices[{i}]" for i in range(len(block_ids))],
        env,
        body_stmts,
    )
    # Set up mask variables for inner-loop block_ids (non-divisible bounds).
    _setup_inner_loop_masks(
        state,
        strategy,
        block_ids,
        block_size_vars,
        env,
        body_stmts,
        # emit_pipeline passes indices as a single tuple arg
        offset_expr_fn=lambda i, bs: (
            f"_helion_compat_pipeline_indices[{i}] * {bs} + jnp.arange({bs})"
        ),
        aligned_dim=aligned_dim,
    )

    # Emit absolute offset assignments inside the pipeline body so any
    # non-pipelined tensors (those left on their outer BlockSpec) can be
    # sliced via pl.ds against a VMEM ref whose extent is the whole
    # outer-block window.  Pipelined tensors ignore these offsets and
    # use the ``:`` full-slice inside their VMEM scratches.
    from ...language.distributed_ops import make_async_remote_copy

    uses_remote_copy = any(
        node.op == "call_function" and node.target is make_async_remote_copy
        for node in graph_info.graph.nodes
    )

    # Remote HBM refs have no BlockSpec to apply an inner-loop tile offset.
    # Materialize absolute offsets even when every ordinary load/store tensor
    # is streamed, so distributed_ops can address the correct HBM tile.
    any_non_pipelined = (
        len(pipelined_tensor_ids) < len(all_tensor_info) or uses_remote_copy
    )
    if any_non_pipelined:
        _needs_explicit_indices = True
        for i, bid in enumerate(block_ids):
            offset_name = strategy.offset_var(bid)
            body_stmts.append(
                statement_from_string(
                    f"{offset_name} = ({begin_exprs[i]}) + "
                    f"(_helion_compat_pipeline_indices[{i}]) * ({iter_step_exprs[i]})"
                )
            )

    # Build tensor_to_dma_scratch mapping
    tensor_to_dma_scratch: dict[str, str] = {}
    idx = 0
    for _fake, hbm_name in in_tensors:
        tensor_to_dma_scratch[hbm_name] = body_params[idx]
        idx += 1
    for _fake, hbm_name in out_tensors:
        tensor_to_dma_scratch[hbm_name] = body_params[idx]
        idx += 1

    # Create the pipeline loop state
    pipeline_state = EmitPipelineLoopState(
        strategy=strategy,  # pyrefly: ignore[bad-argument-type]
        block_id_to_info=block_id_to_info,
        body_fn_name=body_fn_name,
        inner_statements=body_stmts,
        _tensor_to_dma_scratch=tensor_to_dma_scratch,
    )

    # For loop-carried state, remap args to scratch reads inside the body
    body_args = (
        _remap_args_to_scratch(args, scratch_names, state)
        if has_loop_state
        else [*args]
    )

    # Generate body code within the pipeline context
    with state.codegen.add_emit_pipeline_loop(pipeline_state):
        graph_results = codegen_call_with_graph(
            state.codegen, graph_info.graph, body_args
        )

        # Write updated loop-carried values back to scratch
        if has_loop_state:
            _write_back_loop_carried(state, scratch_names, carried, graph_results)

    _emit_nonlocal_scratch_declarations(state, body_stmts)

    all_body_params = body_params
    # emit_pipeline passes indices as a single tuple argument; the prologue
    # always references _pipeline_indices, so the body always takes it.
    fn_args = "_pipeline_indices, " + ", ".join(all_body_params)
    fn_def = statement_from_string(f"def {body_fn_name}({fn_args}): pass")
    assert isinstance(fn_def, ast.FunctionDef)
    fn_def.body = body_stmts or [ast.Pass()]  # pyrefly: ignore[bad-assignment]

    # Build the emit_pipeline call
    grid_str = ", ".join(grid_parts)
    in_specs_str = ", ".join(in_specs) if in_specs else ""
    out_specs_str = ", ".join(out_specs) if out_specs else ""

    spec_parts: list[str] = []
    if in_specs:
        spec_parts.append(f"in_specs=[{in_specs_str}]")
    if out_specs:
        spec_parts.append(f"out_specs=[{out_specs_str}]")
    spec_parts.append("_explicit_indices=True")
    specs_str = ", ".join(spec_parts)

    all_pipeline_args = pipeline_in_args + pipeline_out_args
    call_args_str = ", ".join(all_pipeline_args)

    if specs_str:
        pipeline_call_str = (
            f"pltpu.emit_pipeline({body_fn_name}, grid=({grid_str},), {specs_str})"
            f"({call_args_str})"
        )
    else:
        pipeline_call_str = (
            f"pltpu.emit_pipeline({body_fn_name}, grid=({grid_str},))({call_args_str})"
        )

    # Emit the function def and pipeline call into the current scope
    state.add_statement(fn_def)
    state.add_statement(statement_from_string(pipeline_call_str))

    # After pipeline: read final loop-carried state from scratch
    if has_loop_state:
        return _read_final_loop_state(state, result_vars)
    return None


def _is_supported_contiguous_row_slab_dma(
    fake: torch.Tensor,
    sub_meta: list[object],
    block_ids: list[int],
    vmem_shape: tuple[int, ...],
    env: CompileEnvironment,
    state: CodegenState,
) -> bool:
    """Whether an otherwise-unaligned load is a contiguous row-slab DMA.

    ``_check_dma_alignment`` is conservative for shapes like
    ``[TOKEN_BLOCK, H=4, D=128]`` because the second-to-last logical dim is not a
    multiple of 8.  TPU7x accepts HBM copies from arbitrary dynamic row offsets
    for row-slab layouts: rows are page-addressed and the full suffix is
    contiguous with an aligned lane dimension.

    Keep this exception narrow: load-only caller, one dynamic-begin/end
    current-loop row tile, only scalar-selected prefix dims, full-slice suffix,
    no gathers/scatters/stores.
    """
    if not fake.is_floating_point():
        return False
    if fake.ndim < 2:
        return False
    if not fake.is_contiguous():
        return False
    if len(vmem_shape) != fake.ndim:
        return False

    dim_to_bid = _get_dim_block_ids(sub_meta, env)
    tensor_subscripts = _tensor_dim_subscripts(sub_meta)
    inner_dims = [dim for dim, bid in dim_to_bid.items() if bid in block_ids]
    if len(inner_dims) != 1:
        return False
    row_dim = inner_dims[0]
    if row_dim == fake.ndim - 1:
        return False
    row_bid = dim_to_bid[row_dim]
    if not _loop_dim_is_dynamic(state, block_ids.index(row_bid)):
        return False

    from helion._utils import is_scalar_index

    for dim_idx in range(row_dim):
        idx_meta = _subscript_at_dim(tensor_subscripts, dim_idx)
        if vmem_shape[dim_idx] != 1:
            return False
        if dim_idx in dim_to_bid:
            continue
        if not is_scalar_index(idx_meta):
            return False

    for dim_idx in range(row_dim + 1, fake.ndim):
        idx_meta = _subscript_at_dim(tensor_subscripts, dim_idx)
        if idx_meta != slice(None):
            return False
        if dim_idx in dim_to_bid:
            return False
        dim_size = fake.shape[dim_idx]
        if not isinstance(dim_size, int) or vmem_shape[dim_idx] != dim_size:
            return False

    lane_dim = fake.shape[-1]
    return isinstance(lane_dim, int) and lane_dim % 128 == 0


def _can_stream_inner_tile(
    fake: torch.Tensor,
    sub_meta: list[object],
    direction: str,
    block_ids: list[int],
    vmem_shape: tuple[int, ...],
    env: CompileEnvironment,
    state: CodegenState,
) -> bool:
    """Return whether a loop-local tensor should use the inner streaming path."""
    if is_tpu_dma_aligned_shape(vmem_shape, fake.dtype):
        return True
    if direction != "load":
        return False
    return _is_supported_contiguous_row_slab_dma(
        fake, sub_meta, block_ids, vmem_shape, env, state
    )


def _compute_vmem_shapes(
    all_tensor_info: list[tuple[torch.Tensor, list[object], str]],
    block_ids: list[int],
    slice_size_exprs: list[str],
    env: CompileEnvironment,
    state: CodegenState,
    contiguous_ranges: dict[int, dict[int, ContiguousRangeIndexPattern]],
) -> list[tuple[int, ...]]:
    """Compute VMEM buffer shapes for each tensor in the fori_loop body."""
    vmem_shapes: list[tuple[int, ...]] = []
    for fake, sub_meta, _direction in all_tensor_info:
        dim_to_bid = _get_dim_block_ids(sub_meta, env)
        tensor_subscripts = _tensor_dim_subscripts(sub_meta)
        range_dims = contiguous_ranges.get(id(fake), {})
        parts: list[int] = []
        for dim_idx in range(len(fake.shape)):
            bid = dim_to_bid.get(dim_idx)
            if dim_idx in range_dims:
                parts.append(range_dims[dim_idx].length)
            elif bid is not None and bid in block_ids:
                bid_idx = block_ids.index(bid)
                block_value_sym = sympy.sympify(slice_size_exprs[bid_idx])
                if isinstance(block_value_sym, sympy.Integer):
                    parts.append(int(block_value_sym))
                else:
                    block_value = state.device_function.resolved_block_size(
                        block_ids[bid_idx]
                    )
                    assert isinstance(block_value, int)
                    parts.append(block_value)
            elif bid is not None:
                outer_block_value = state.device_function.resolved_block_size(bid)
                if isinstance(outer_block_value, int):
                    parts.append(outer_block_value)
                else:
                    parts.append(int(fake.shape[dim_idx]))
            else:
                idx_meta = _subscript_at_dim(tensor_subscripts, dim_idx)
                from helion._utils import is_scalar_index

                parts.append(
                    1 if is_scalar_index(idx_meta) else int(fake.shape[dim_idx])
                )
        vmem_shapes.append(tuple(parts))
    return vmem_shapes


def _runtime_vmem_shape_sources(
    fake: torch.Tensor,
    sub_meta: list[object],
    block_ids: list[int],
    env: CompileEnvironment,
) -> tuple[tuple[torch.Tensor, int] | None, ...]:
    """Map untiled full-slice scratch dimensions back to runtime inputs.

    Tiling analysis uses concrete hint values for alignment and planning. The
    launch-time scratch allocation must still follow untiled runtime dimensions
    when ``static_shapes=False``.
    """
    from helion._utils import is_scalar_index

    dim_to_bid = _get_dim_block_ids(sub_meta, env)
    tensor_subscripts = _tensor_dim_subscripts(sub_meta)
    result: list[tuple[torch.Tensor, int] | None] = []
    for dim in range(fake.ndim):
        bid = dim_to_bid.get(dim)
        idx_meta = _subscript_at_dim(tensor_subscripts, dim)
        if bid is None and not is_scalar_index(idx_meta):
            result.append((fake, dim))
        else:
            result.append(None)
    return tuple(result)


def _classify_pipelined_tensors(
    loaded_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
    stored_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
    block_ids: list[int],
    slice_size_exprs: list[str],
    env: CompileEnvironment,
    state: CodegenState,
) -> tuple[
    list[tuple[torch.Tensor, list[object], str]], list[tuple[int, ...]], set[int]
]:
    """Build (all_tensor_info, vmem_shapes, pipelined_ids) for an inner loop.

    A tensor is eligible for the inner-DMA path (HBM ref + small VMEM scratch
    in fori_loop, or ``pl.Buffered`` BlockSpec in emit_pipeline) when:

    * Its inner-block ``vmem_shape`` passes the standard TPU DMA alignment check,
      or it is a load-only contiguous row-slab layout covered by
      ``_is_supported_contiguous_row_slab_dma``.
    * It is not also accessed at outer scope (i.e. in a root graph,
      between/before/after inner loops).  Pipelining replaces the tensor's
      outer BlockSpec with ``pltpu.HBM`` so the inner loop's BlockSpec can
      handle slicing; reads/writes at outer scope would then have to
      ``pl.ds`` an HBM ref, which Pallas rejects with "Loads are only
      allowed on VMEM and SMEM references."  Pallas lowers atomics as
      load-compute-store on the same ref, so outer-scope atomics count as
      memory accesses too.

    Tensors that fail any check stay on their outer BlockSpec and are
    closure-read from the body.
    """
    from ...language.atomic_ops import ATOMIC_OPS
    from ...language.memory_ops import load as _load_op
    from ...language.memory_ops import store as _store_op

    outer_access_targets = ATOMIC_OPS | {_load_op, _store_op}

    all_tensor_info = _resident_loop_tensor_info(loaded_tensors, stored_tensors)
    contiguous_ranges = _contiguous_range_patterns(loaded_tensors)
    vmem_shapes = _compute_vmem_shapes(
        all_tensor_info,
        block_ids,
        slice_size_exprs,
        env,
        state,
        contiguous_ranges,
    )
    device_ir = HostFunction.current().device_ir

    # Walk all root graphs (outer pallas_call body) for load/store/atomic
    # nodes; any tensor accessed there is read/written outside the inner
    # loop and must keep its outer BlockSpec.
    outer_access_tensor_ids: set[int] = set()
    for root_id in device_ir.root_ids:
        root_graph = device_ir.graphs[root_id].graph
        for node in root_graph.nodes:
            if node.op != "call_function" or node.target not in outer_access_targets:
                continue
            tensor_node = node.args[0]
            if not isinstance(tensor_node, torch.fx.Node):
                continue
            val = tensor_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                outer_access_tensor_ids.add(id(val))

    # Pallas lowers atomics as direct load/compute/store operations on their
    # tensor ref. Such a tensor cannot simultaneously be remapped to the
    # loop-local DMA scratch used by ordinary loads.
    atomic_storages: set[int] = set()
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in ATOMIC_OPS:
                continue
            tensor_node = node.args[0]
            if not isinstance(tensor_node, torch.fx.Node):
                continue
            val = tensor_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                atomic_storages.add(id(val.untyped_storage()))

    pipelined_ids: set[int] = set()
    for (fake, sub_meta, direction), vmem_shape in zip(
        all_tensor_info, vmem_shapes, strict=True
    ):
        if direction == "load":
            first_load = loaded_tensors[id(fake)][1]
            if int(first_load.meta.get(_PALLAS_LOOP_LOAD_COUNT_META, 1)) > 1:
                # Tensor-level prefetching is keyed by input tensor, not load
                # site. Dynamic ranges can remain in HBM so each load site
                # stages its own exact window. Ordinary tiled loads must keep
                # their outer BlockSpec: raw HBM load-site staging is defined
                # only for dynamic ranges and remote-copy operands.
                if id(fake) in contiguous_ranges:
                    from ..device_function import PallasMemorySpace

                    state.device_function.pallas_memory_space[id(fake)] = (
                        PallasMemorySpace.HBM
                    )
                    continue
        dim_to_bid = _get_dim_block_ids(sub_meta, env)
        if state.device_function.is_pallas_remote_copy_operand(fake) and not set(
            dim_to_bid.values()
        ).intersection(block_ids):
            # A loop-invariant remote-copy operand must keep one stable address.
            # Streaming it would select a different VMEM pipeline generation
            # on each iteration. Ordinary tensors retain the existing pipeline
            # classification below.
            continue
        if not _can_stream_inner_tile(
            fake, sub_meta, direction, block_ids, vmem_shape, env, state
        ):
            continue
        if id(fake) in outer_access_tensor_ids:
            continue
        if id(fake.untyped_storage()) in atomic_storages:
            continue
        if range_patterns := contiguous_ranges.get(id(fake)):
            can_render_ranges = all(
                _contiguous_range_base_expr(
                    pattern.base,
                    state=state,
                    block_ids=block_ids,
                    begin_exprs=["0"] * len(block_ids),
                    iter_step_exprs=["1"] * len(block_ids),
                    iteration_indices=["0"] * len(block_ids),
                )
                is not None
                for pattern in range_patterns.values()
            )
            if not can_render_ranges:
                # The load-site lowering can still stage this window, but the
                # loop prefetcher cannot safely synthesize its next address.
                continue
        pipelined_ids.add(id(fake))
    return all_tensor_info, vmem_shapes, pipelined_ids


def _resident_loop_tensor_info(
    loaded_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
    stored_tensors: dict[int, tuple[torch.Tensor, torch.fx.Node, list[object]]],
) -> list[tuple[torch.Tensor, list[object], str]]:
    """Tensor access records needed by optional resident prep lowering."""
    result = [
        (fake, sub_meta, "load")
        for key, (fake, _tensor_node, sub_meta) in loaded_tensors.items()
        if key not in stored_tensors
    ]
    result.extend(
        (fake, sub_meta, "store")
        for fake, _tensor_node, sub_meta in stored_tensors.values()
    )
    return result


def _codegen_dynamic_unroll(state: CodegenState) -> object:
    """Run the ordinary resident unroll body with a dynamic trip count."""
    from ..device_ir import ForLoopGraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import codegen_call_with_graph
    from ..tile_strategy import ForiLoopState

    graph_info = state.get_graph(state.proxy_arg(0))
    assert isinstance(graph_info, ForLoopGraphInfo)
    assert isinstance(state.codegen, GenerateAST)
    block_ids = graph_info.block_ids
    if len(block_ids) != 1:
        raise InvalidConfig(
            "dynamic pallas unroll currently supports one inner tile dimension"
        )

    args = state.ast_args[-1]
    assert isinstance(args, list)
    assert all(isinstance(arg, ast.AST) for arg in args)

    env = CompileEnvironment.current()
    grid_parts, block_size_vars = _compute_grid_and_block_sizes(state, block_ids, env)
    begin_exprs, iter_step_exprs, _ = _pallas_loop_begin_and_step_exprs(
        state, block_ids, block_size_vars
    )
    strategy = _find_strategy(state, block_ids)
    loop_var = state.device_function.new_var("_j")
    body_stmts: list[ast.AST] = []
    _emit_inner_loop_offset_indices(
        state,
        strategy,
        block_ids,
        block_size_vars,
        begin_exprs,
        iter_step_exprs,
        [loop_var],
        env,
        body_stmts,
    )
    _setup_inner_loop_masks(
        state,
        strategy,
        block_ids,
        block_size_vars,
        env,
        body_stmts,
        offset_expr_fn=lambda _i, bs: f"{loop_var} * {bs} + jnp.arange({bs})",
    )

    body_fn_name = state.device_function.new_var("_dynamic_unroll_body")
    fori_state = ForiLoopState(
        strategy=strategy,  # pyrefly: ignore[bad-argument-type]
        block_id_to_info=_loop_dim_infos(state, block_ids, env),
        body_fn_name=body_fn_name,
        loop_var_name=loop_var,
        inner_statements=body_stmts,
    )

    loaded_tensors, stored_tensors = _classify_loop_tensors(graph_info, state)
    resident_prep_lowerings = _prepare_resident_prep_lowerings(
        state,
        block_ids,
        _resident_loop_tensor_info(loaded_tensors, stored_tensors),
    )
    _emit_resident_prep_refill_once(state, block_ids, resident_prep_lowerings)

    carried = sorted(_loop_carried_indices(state, len(args)))
    # Uniquely named: a nested dynamic-unroll body would otherwise shadow the
    # enclosing loop's carry tuple and silently rebind reads of it.
    carry_var = state.device_function.new_var("_carry")
    body_args = [*args]
    for carry_index, arg_index in enumerate(carried):
        body_args[arg_index] = expr_from_string(f"{carry_var}[{carry_index}]")

    with state.codegen.add_fori_loop(fori_state):
        with state.codegen.resident_prep_lowering_scope(resident_prep_lowerings):
            graph_results = codegen_call_with_graph(
                state.codegen,
                graph_info.graph,
                body_args,
            )
        assert len(graph_results) == len(carried)
        assert all(isinstance(result, ast.AST) for result in graph_results)
        if graph_results:
            ast_results = cast("list[ast.AST]", graph_results)
            return_values = ", ".join(ast.unparse(result) for result in ast_results)
            if len(graph_results) == 1:
                return_values += ","
            state.codegen.add_statement(
                statement_from_string(f"return ({return_values})")
            )
        else:
            state.codegen.add_statement(statement_from_string(f"return {carry_var}"))

    _emit_nonlocal_scratch_declarations(state, body_stmts)
    body_fn = statement_from_string(
        f"def {body_fn_name}({loop_var}, {carry_var}):\n    pass"
    )
    assert isinstance(body_fn, ast.FunctionDef)
    body_fn.body = cast("list[ast.stmt]", body_stmts)
    state.add_statement(body_fn)

    initial_values = ", ".join(ast.unparse(args[index]) for index in carried)
    if len(carried) == 1:
        initial_values += ","
    initial_carry = f"({initial_values})"
    if not carried:
        state.add_statement(
            statement_from_string(
                f"jax.lax.fori_loop(0, {grid_parts[0]}, {body_fn_name}, ())"
            )
        )
        return None

    result_var = state.device_function.new_var("_dynamic_unroll_result")
    state.add_statement(
        statement_from_string(
            f"{result_var} = jax.lax.fori_loop(0, {grid_parts[0]}, "
            f"{body_fn_name}, {initial_carry})"
        )
    )
    return [expr_from_string(f"{result_var}[{i}]") for i in range(len(carried))]


def _codegen_fori_loop(state: CodegenState, *, static_unroll: bool = False) -> object:
    """Emit inner device loops using jax.lax.fori_loop.

    Tensors admitted by the existing streaming classifier use explicit DMA.
    Selected read-only input tensors use two DMA buffers; all other routes keep
    their existing single-buffered lowering. ``static_unroll`` retains that DMA
    schedule while using Python ``range`` so JAX traces a straight-line program.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import LiftTensorArgs
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import codegen_call_with_graph
    from ..tile_strategy import ForiLoopState

    graph_info = state.get_graph(state.proxy_arg(0))
    assert isinstance(graph_info, ForLoopGraphInfo)
    assert isinstance(state.codegen, GenerateAST)

    block_ids = graph_info.block_ids
    env = CompileEnvironment.current()

    args = state.ast_args[-1]
    assert isinstance(args, list)
    assert all(isinstance(x, ast.AST) for x in args)

    proxy_args = state.proxy_args[-1]
    assert isinstance(proxy_args, list)
    has_loop_state = len(args) > 0

    grid_parts, block_size_vars = _compute_grid_and_block_sizes(state, block_ids, env)

    loaded_tensors, stored_tensors = _classify_loop_tensors(graph_info, state)
    placeholders = list(graph_info.graph.find_nodes(op="placeholder"))
    placeholder_exprs = {
        placeholder: ast.unparse(arg)
        for placeholder, arg in zip(placeholders, args, strict=True)
    }
    captured_scalar_exprs: dict[int, str] = {}
    scalar_index_nodes: dict[int, torch.fx.Node] = {}
    for _fake, load_node, _subscript in loaded_tensors.values():
        load_indices = load_node.args[1]
        if not isinstance(load_indices, (list, tuple)):
            continue
        for index_node in load_indices:
            if not isinstance(index_node, torch.fx.Node):
                continue
            index_value = index_node.meta.get("val")
            if not isinstance(index_value, torch.Tensor) or index_value.ndim != 0:
                continue
            scalar_index_nodes[id(index_value)] = index_node
            source = index_node
            seen: set[torch.fx.Node] = set()
            while (
                source not in seen
                and source.op == "call_function"
                and source.target is _new_var
                and source.args
                and isinstance(source.args[0], torch.fx.Node)
            ):
                seen.add(source)
                source = source.args[0]
            if source in placeholder_exprs:
                captured_scalar_exprs[id(index_value)] = placeholder_exprs[source]
    contiguous_ranges = _contiguous_range_patterns(loaded_tensors)
    begin_exprs, iter_step_exprs, slice_size_exprs = _pallas_loop_begin_and_step_exprs(
        state, block_ids, block_size_vars
    )
    indirect_accesses, dma_metadata_ids = _collect_fori_indirect_accesses(
        graph_info, block_ids, state
    )

    # --- Handle loop-carried state as scratch VMEM buffers ---
    scratch_names: list[str] = []
    result_vars: list[object] = []
    carried: set[int] = set()
    if has_loop_state:
        scratch_names, result_vars, carried = _setup_loop_carried_state(
            state, args, proxy_args, env
        )

    # --- Pre-broadcast transform (same as emit_pipeline) ---
    if state.config.get("pallas_pre_broadcast", False) and has_loop_state:
        _apply_pre_broadcast_transform(
            state,
            graph_info.graph,
            carried,
            proxy_args,
            scratch_names,
            args,
            block_ids,
            env,
        )

    # Pipelined tensors get HBM refs (no outer BlockSpec) + VMEM scratch +
    # semaphore; the rest keep their outer BlockSpec and are accessed via
    # pl.ds() in the body.  Mixing both paths inside a single fori_loop
    # avoids forcing every tensor onto the non-DMA path when a lone
    # non-pipelined tensor is present (which would load full outer-block
    # tiles into VMEM and may OOM at large shapes).
    all_tensor_info, vmem_shapes, pipelined_tensor_ids = _classify_pipelined_tensors(
        loaded_tensors, stored_tensors, block_ids, slice_size_exprs, env, state
    )
    # Indirect addresses must be available before the graph body so the
    # scheduler can form the next iteration's HBM Refs. Keep their producer
    # tensors on the enclosing BlockSpec instead of streaming them here.
    pipelined_tensor_ids -= dma_metadata_ids | {
        id(access.tensor) for access in indirect_accesses
    }

    # Compact worklist: the compact-tile aligned_load and exact_store tensors use
    # max-sized window BlockSpecs, which Pallas double-buffers across the
    # work-item grid.  Keep them OUT of the manual make_async_copy DMA path: with
    # a single straight-line compact tile there is no inner loop to overlap, so a
    # DMA start()/wait() would run fully serial (load -> wait -> compute -> store
    # -> wait, measured ~1.8x slower).  Excluding them lets the BlockSpec pipeline
    # hide the latency.
    _compact_plan = env.compact_worklist_plan
    if _compact_plan is not None and not carried:
        _compact_names = {
            p.arg_name
            for p in _compact_plan.tensor_policies
            if p.kind in ("compact_aligned_load", "compact_exact_store")
        }
        _fid_to_fake = {id(f): f for f, _s, _d in all_tensor_info}
        pipelined_tensor_ids = {
            fid
            for fid in pipelined_tensor_ids
            if state.device_function.tensor_arg(_fid_to_fake[fid]).host_str()
            not in _compact_names
        }

    # Resident caching: only active resident ordered operands are held in a
    # per-range resident window and read at the local ordered-tile offset (see
    # codegen._is_ordered_aligned_load).  Keep just those OFF the streamed
    # make_async_copy path.  Gate on the SAME decision the window is built from,
    # so an inactive residency decision leaves every operand streaming.
    if _compact_plan is not None:
        _decision = env.compact_worklist_resident_cache_decision
        _ordered_names = (
            set(_decision.resident_operands)
            if _decision is not None and _decision.active
            else set()
        )
        if _ordered_names:
            _fid_to_fake_o = {id(f): f for f, _s, _d in all_tensor_info}
            pipelined_tensor_ids = {
                fid
                for fid in pipelined_tensor_ids
                if state.device_function.tensor_arg(_fid_to_fake_o[fid]).host_str()
                not in _ordered_names
            }

    from ..device_function import PallasMemorySpace

    tensor_to_dma_scratch: dict[str, str] = {}
    tensor_to_sem: dict[str, str] = {}
    prefetched_load_tensors: set[str] = set()
    prefetched_loads: list[ScheduledDmaTransfer] = []
    immediate_loads: list[ScheduledDmaTransfer] = []
    dma_stores: list[ScheduledDmaTransfer] = []
    scheduled_by_hbm_name: dict[str, ScheduledDmaTransfer] = {}
    memory_op_to_dma_scratch: dict[torch.fx.Node, DmaResources] = {}
    refilled_load_tensors: set[str] = set()
    refilled_loads: list[tuple[ScheduledDmaTransfer, torch.fx.Node, torch.fx.Node]] = []
    # compact_worklist shares this lowering but keeps its compact/resident routes.
    # Ordinary fori loops and explicitly buffered static unrolls honor the
    # per-input depth; other callers keep the historical single-buffer route.
    load_buffer_counts_active = state.config.get("pallas_loop_type") in (
        "fori_loop",
        "unroll",
    )

    input_tensors = cast(
        "list[torch.Tensor]",
        LiftTensorArgs(dict(HostFunction.current().params.arguments)).get_tensor_args(),
    )
    input_slots_by_id: dict[int, list[int]] = {}
    input_slots_by_storage: dict[int, list[int]] = {}
    for input_slot, input_tensor in enumerate(input_tensors):
        input_slots_by_id.setdefault(id(input_tensor), []).append(input_slot)
        input_slots_by_storage.setdefault(
            id(input_tensor.untyped_storage()), []
        ).append(input_slot)
    stored_tensor_storages = {
        id(fake.untyped_storage()) for fake, _node, _sub_meta in stored_tensors.values()
    }
    dma_transfers: list[tuple[DmaTransfer, tuple[int, ...]]] = [
        (transfer, transfer.plan.transfer_shape) for transfer in indirect_accesses
    ]
    for (fake, sub_meta, direction), vmem_shape in zip(
        all_tensor_info, vmem_shapes, strict=True
    ):
        if id(fake) not in pipelined_tensor_ids:
            continue
        dma_transfers.append(
            (
                DmaTransfer(
                    tensor=fake,
                    subscript=tuple(sub_meta),
                    direction=cast("DmaDirection", direction),
                ),
                vmem_shape,
            )
        )

    indirect_load_resources_by_storage: dict[int, DmaResources] = {}
    indirect_store_storages = {
        id(transfer.tensor.untyped_storage())
        for transfer in indirect_accesses
        if transfer.direction == "store"
    }
    for transfer, vmem_shape in dma_transfers:
        fake = transfer.tensor
        storage_id = id(fake.untyped_storage())
        input_slots = input_slots_by_id.get(
            id(fake), input_slots_by_storage.get(storage_id)
        )
        load_buffer_count = (
            state.config.pallas_load_buffer_count[input_slots[0]]
            if load_buffer_counts_active
            and transfer.direction == "load"
            and storage_id not in stored_tensor_storages
            and storage_id not in indirect_store_storages
            and input_slots is not None
            and len(input_slots) == 1
            else 1
        )
        assert load_buffer_count in (1, 2)
        uses_load_prefetch = load_buffer_count == 2
        state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
        hbm_name = state.device_function.tensor_arg(fake).name
        resource_key = (
            graph_info.graph_id,
            hbm_name,
            transfer.direction,
            vmem_shape,
            load_buffer_count,
        )
        resource_cache = (
            state.codegen.grouped_fori_dma_resource_cache
            if not isinstance(transfer, IndirectDmaTransfer)
            and state.codegen.grouped_compact_common_statements is not None
            and _is_compact_ordered_inner_loop(state)
            else None
        )
        cached_resource = (
            resource_cache.get(resource_key) if resource_cache is not None else None
        )
        if isinstance(transfer, IndirectDmaTransfer):
            resources = allocate_indirect_dma_resources(
                state.device_function,
                transfer,
                buffer_count=load_buffer_count,
                load_resources=indirect_load_resources_by_storage.get(storage_id),
            )
        elif cached_resource is None:
            scratch_hint = hbm_name.replace("_hbm", "") + "_buf"
            sem_hint = hbm_name.replace("_hbm", "") + "_sem"
            shape_sources = (
                _runtime_vmem_shape_sources(
                    fake, list(transfer.subscript), block_ids, env
                )
                if state.device_function.is_pallas_remote_copy_operand(fake)
                else None
            )
            resources = allocate_dma_resources(
                state.device_function,
                transfer,
                vmem_shape=vmem_shape,
                buffer_count=load_buffer_count,
                scratch_hint=scratch_hint,
                semaphore_hint=sem_hint,
                shape_sources=shape_sources,
            )
            if resource_cache is not None:
                resource_cache[resource_key] = resources
        else:
            resources = cached_resource
        if isinstance(transfer, IndirectDmaTransfer) and transfer.direction == "load":
            indirect_load_resources_by_storage[storage_id] = resources
        scheduled = ScheduledDmaTransfer(transfer, resources)

        if not isinstance(transfer, IndirectDmaTransfer):
            tensor_to_dma_scratch[hbm_name] = resources.scratch
            tensor_to_sem[hbm_name] = resources.semaphore
            scheduled_by_hbm_name[hbm_name] = scheduled
            if uses_load_prefetch:
                prefetched_load_tensors.add(hbm_name)
        else:
            memory_op_to_dma_scratch[transfer.plan.access.node] = resources

        if transfer.direction == "store":
            dma_stores.append(scheduled)
        elif uses_load_prefetch:
            prefetched_loads.append(scheduled)
        elif isinstance(transfer, IndirectDmaTransfer):
            immediate_loads.append(scheduled)
        elif (
            load_buffer_counts_active
            and transfer.direction == "load"
            and len(block_ids) == 1
            and id(fake) in contiguous_ranges
        ):
            load_node = loaded_tensors[id(fake)][1]
            users = list(load_node.users)
            if (
                len(users) == 1
                and users[0].op == "call_function"
                and users[0].target
                in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default)
            ):
                refilled_load_tensors.add(hbm_name)
                refilled_loads.append((scheduled, load_node, users[0]))

    # ``all_tensor_info`` represents a read-modify-write tensor only by its
    # store record so that its load and store share one VMEM buffer. Build
    # contiguous immediate loads from the original load sites while indirect
    # loads continue to use their access-specific plans above.
    for fake, _tensor_node, sub_meta in loaded_tensors.values():
        hbm_name = state.device_function.tensor_arg(fake).name
        scheduled = scheduled_by_hbm_name.get(hbm_name)
        if (
            scheduled is None
            or hbm_name in prefetched_load_tensors
            or hbm_name in refilled_load_tensors
        ):
            continue
        immediate_loads.append(
            ScheduledDmaTransfer(
                DmaTransfer(
                    tensor=fake,
                    subscript=tuple(sub_meta),
                    direction="load",
                ),
                scheduled.resources,
            )
        )

    # Build the body function
    body_stmts: list[ast.AST] = []

    strategy = _find_strategy(state, block_ids)

    # NOTE: FlattenedTileStrategy with multi-dim inner loops is not handled
    # yet.  The nested fori_loop emission assumes NDTileStrategy where each
    # dimension has its own block size and grid extent.

    # Create one loop variable per dimension for nested fori_loops.
    # Each dimension gets its own fori_loop; the innermost wraps body_stmts.
    if len(block_ids) == 1:
        loop_vars = [state.device_function.new_var("_j")]
    else:
        loop_vars = [
            state.device_function.new_var(f"_j{i}") for i in range(len(block_ids))
        ]
    dim_idx_exprs: list[str] = loop_vars

    # Build block_id_to_info
    block_id_to_info = _loop_dim_infos(state, block_ids, env)

    # Emit offset_<bid>/indices_<bid> at the body prologue.
    _emit_inner_loop_offset_indices(
        state,
        strategy,
        block_ids,
        block_size_vars,
        begin_exprs,
        iter_step_exprs,
        dim_idx_exprs,
        env,
        body_stmts,
    )
    # Set up mask variables for inner-loop block_ids (non-divisible bounds).
    _setup_inner_loop_masks(
        state,
        strategy,
        block_ids,
        block_size_vars,
        env,
        body_stmts,
        # fori_loop has direct access to the loop variable
        offset_expr_fn=lambda i, bs: f"{dim_idx_exprs[i]} * {bs} + jnp.arange({bs})",
    )

    fori_state = ForiLoopState(
        strategy=strategy,  # pyrefly: ignore[bad-argument-type]
        block_id_to_info=block_id_to_info,
        body_fn_name="_fori_body_0",
        loop_var_name=loop_vars[-1],
        static_unroll=static_unroll,
        inner_statements=body_stmts,
        _tensor_to_dma_scratch=tensor_to_dma_scratch,
        _tensor_to_sem=tensor_to_sem,
        _prefetched_load_tensors=prefetched_load_tensors,
        _memory_op_to_dma_scratch=memory_op_to_dma_scratch,
    )
    resident_prep_lowerings = _prepare_resident_prep_lowerings(
        state, block_ids, all_tensor_info
    )
    if resident_prep_lowerings:
        assert len(grid_parts) == 1
        _emit_resident_prep_refill_once(state, block_ids, resident_prep_lowerings)

    def _build_dma_slices(
        fake: torch.Tensor,
        vmem_name: str,
        hbm_name: str,
        subscript_meta: list[object],
        *,
        clamp: bool,
        iteration_indices: list[str],
        stage_expr: str | None = None,
        hbm_part_overrides: dict[int, str] | None = None,
        resident_source: bool = False,
        indexing_patterns: Sequence[object] | None = None,
    ) -> tuple[str, str]:
        """Build (vmem_ref, hbm_ref) ref slices for a DMA copy with loop variable.

        The HBM ref is sliced to this iteration's tile.  With ``clamp=True``
        (ragged stores) a leading tiled dim is trimmed to its live extent
        ``min(block_size, end - offset)`` and the VMEM side sliced to match, so
        only live rows are written instead of overrunning adjacent regions
        packed in the same tensor; with ``clamp=False`` (loads, dense stores)
        the VMEM side stays the bare buffer.
        """
        from helion._compiler.pallas.ordered_carry import is_dynamic_bound_tile

        assert len(iteration_indices) == len(block_ids)
        dim_to_bid = _get_dim_block_ids(subscript_meta, env)
        tensor_subscripts = _tensor_dim_subscripts(subscript_meta)
        shape = fake.shape
        hbm_parts: list[str] = []
        vmem_parts: list[str] = []
        hbm_needs_slice = False
        vmem_needs_slice = False
        for dim_idx in range(len(shape)):
            if hbm_part_overrides and dim_idx in hbm_part_overrides:
                hbm_parts.append(hbm_part_overrides[dim_idx])
                hbm_needs_slice = True
                vmem_parts.append(":")
                continue
            bid = dim_to_bid.get(dim_idx)
            range_pattern = contiguous_ranges.get(id(fake), {}).get(dim_idx)
            if range_pattern is not None:
                base_expr = _contiguous_range_base_expr(
                    range_pattern.base,
                    state=state,
                    block_ids=block_ids,
                    begin_exprs=begin_exprs,
                    iter_step_exprs=iter_step_exprs,
                    iteration_indices=iteration_indices,
                    captured_exprs=placeholder_exprs,
                )
                if base_expr is None:
                    raise RuntimeError(
                        "Pallas could not render a planned contiguous HBM range"
                    )
                hbm_parts.append(
                    "pl.ds(pl.multiple_of("
                    f"{base_expr}, {range_pattern.alignment}), "
                    f"{range_pattern.length})"
                )
                vmem_parts.append(":")
                hbm_needs_slice = True
            elif bid is not None and bid in block_ids:
                bid_idx = block_ids.index(bid)
                begin_expr = begin_exprs[bid_idx]
                iter_step_expr = iter_step_exprs[bid_idx]
                slice_size_expr = slice_size_exprs[bid_idx]
                dim_idx_expr = iteration_indices[bid_idx]
                offset_expr = f"({begin_expr}) + ({dim_idx_expr}) * ({iter_step_expr})"
                # Mosaic requires the lane (/128) and sublane (/8) VMEM dims to
                # stay tile-aligned, so only clamp dims outside the last two; a
                # ragged store on a last-two dim can't clamp and is rejected.
                if clamp and dim_idx < len(shape) - 2:
                    end_expr = _get_loop_begin_and_end(state, bid_idx)[1]
                    slice_size_expr = f"jnp.minimum({slice_size_expr}, ({end_expr}) - ({offset_expr}))"
                    vmem_parts.append(f"pl.ds(0, {slice_size_expr})")
                    vmem_needs_slice = True
                elif clamp and is_dynamic_bound_tile(state, bid):
                    raise NotImplementedError(
                        "Pallas: a ragged (data-dependent) store whose tiled "
                        "dimension is one of the last two (lane/sublane) "
                        "dimensions is not supported. Mosaic tile alignment "
                        "forbids clamping there, so a partial tile would "
                        "silently overrun adjacent rows. Move the ragged "
                        "dimension to a leading position, e.g. "
                        "[tokens, heads, head_dim]."
                    )
                else:
                    vmem_parts.append(":")
                hbm_parts.append(f"pl.ds({offset_expr}, {slice_size_expr})")
                hbm_needs_slice = True
                from ...language.memory_ops import _record_pad_info

                extra_pad = _compute_pipeline_or_dma_extra_pad(
                    begin_expr, bid, env, state, bid_idx
                )
                _record_pad_info(state, fake, dim_idx, bid, extra_pad)
            elif bid is not None and bid not in block_ids:
                # Outer grid dim: use grid offset
                grid_loops = state.codegen.active_device_loops.get(bid)
                if grid_loops:
                    if resident_source:
                        idx_meta = _subscript_at_dim(tensor_subscripts, dim_idx)
                        pattern = _subscript_at_dim(indexing_patterns or [], dim_idx)
                        from .plan_tiling import ArbitraryIndexPattern
                        from .plan_tiling import TileBeginWithOffsetPattern

                        if isinstance(pattern, ArbitraryIndexPattern):
                            hbm_parts.append(
                                state.device_function.literal_expr(idx_meta)
                            )
                        elif isinstance(pattern, TileBeginWithOffsetPattern):
                            dim_tilings = (
                                state.device_function.pallas_tensor_dim_tilings[
                                    id(fake)
                                ]
                            )
                            if dim_tilings[dim_idx].can_tile:
                                hbm_parts.append(
                                    state.device_function.literal_expr(pattern.offset)
                                )
                            else:
                                offset = state.codegen.offset_var(bid)
                                if pattern.offset != 0:
                                    offset += (
                                        " + "
                                        + state.device_function.literal_expr(
                                            pattern.offset
                                        )
                                    )
                                hbm_parts.append(offset)
                        else:
                            hbm_parts.append(":")
                        hbm_needs_slice = True
                        vmem_parts.append(":")
                        continue
                    offset = state.codegen.offset_var(bid)
                    bs_var = state.device_function.block_size_var(bid)
                    if bs_var:
                        hbm_parts.append(f"pl.ds({offset}, {bs_var})")
                        hbm_needs_slice = True
                        from ...language.memory_ops import _record_pad_info

                        extra_pad = _compute_pipeline_or_dma_extra_pad(
                            _active_loop_begin_expr(state, bid), bid, env, state
                        )
                        _record_pad_info(state, fake, dim_idx, bid, extra_pad)
                    else:
                        hbm_parts.append(":")
                else:
                    hbm_parts.append(":")
                vmem_parts.append(":")
            else:
                idx_meta = _subscript_at_dim(tensor_subscripts, dim_idx)
                from helion._utils import is_scalar_index

                if is_scalar_index(idx_meta):
                    offset_expr = None
                    if isinstance(idx_meta, torch.Tensor):
                        index_node = scalar_index_nodes.get(id(idx_meta))
                        if index_node is not None:
                            offset_expr = _contiguous_range_base_expr(
                                index_node,
                                state=state,
                                block_ids=block_ids,
                                begin_exprs=begin_exprs,
                                iter_step_exprs=iter_step_exprs,
                                iteration_indices=iteration_indices,
                                captured_exprs=placeholder_exprs,
                            )
                        if offset_expr is None:
                            offset_expr = captured_scalar_exprs.get(id(idx_meta))
                    if offset_expr is None:
                        offset_expr = state.device_function.literal_expr(idx_meta)
                    hbm_parts.append(
                        offset_expr if resident_source else f"pl.ds({offset_expr}, 1)"
                    )
                    hbm_needs_slice = True
                elif isinstance(idx_meta, slice) and idx_meta != slice(None):
                    start = 0 if idx_meta.start is None else idx_meta.start
                    stop = shape[dim_idx] if idx_meta.stop is None else idx_meta.stop
                    if not isinstance(start, int) or not isinstance(stop, int):
                        raise NotImplementedError(
                            "Pallas DMA requires concrete bounded slice extents"
                        )
                    hbm_parts.append(f"pl.ds({start}, {stop - start})")
                else:
                    hbm_parts.append(":")
                vmem_parts.append(":")
        # ``.at[]`` (Ref transform), not ``[]`` which would materialize a
        # dynamically-shaped array; make_async_copy operates on Refs.  Each side
        # falls back to the bare ref when it has no slices.
        hbm = f"{hbm_name}.at[{', '.join(hbm_parts)}]" if hbm_needs_slice else hbm_name
        vmem_base = (
            f"{vmem_name}.at[{stage_expr}]" if stage_expr is not None else vmem_name
        )
        vmem = (
            f"{vmem_base}.at[{', '.join(vmem_parts)}]"
            if vmem_needs_slice
            else vmem_base
        )
        return vmem, hbm

    def _dma_copy_statements(
        transfer: DmaTransfer,
        resources: DmaResources,
        iteration_indices: list[str],
        stage_expr: str | None,
        methods: tuple[str, ...],
    ) -> list[ast.stmt]:
        fake = transfer.tensor
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_ref, hbm_ref = _build_dma_slices(
            fake,
            resources.scratch,
            hbm_name,
            list(transfer.subscript),
            clamp=transfer.direction == "store",
            iteration_indices=iteration_indices,
            stage_expr=stage_expr,
        )
        source, destination = (
            (vmem_ref, hbm_ref)
            if transfer.direction == "store"
            else (hbm_ref, vmem_ref)
        )
        return async_copy_statements(
            state,
            source,
            destination,
            resources.semaphore_ref(stage_expr),
            methods,
            "_copy_out" if transfer.direction == "store" else "_copy",
        )

    def _dma_transfer_statements(
        scheduled: ScheduledDmaTransfer,
        iteration_indices: list[str],
        stage_expr: str | None,
        methods: tuple[str, ...],
    ) -> list[ast.stmt]:
        """Emit one transfer through the common fori scheduling policy."""
        transfer = scheduled.transfer
        resources = scheduled.resources
        if not isinstance(transfer, IndirectDmaTransfer):
            return _dma_copy_statements(
                transfer,
                resources,
                iteration_indices,
                stage_expr,
                methods,
            )

        plan = transfer.plan
        index_access = plan.spec.index_access
        metadata_fake = index_access.tensor
        metadata_patterns = list(index_access.patterns)
        hbm_name = state.device_function.tensor_arg(transfer.tensor).name
        metadata_name = state.device_function.tensor_arg(metadata_fake).name
        result: list[ast.stmt] = []
        if "start" in methods:
            index_name = state.device_function.new_var("_dma_indices")
            _, metadata_ref = _build_dma_slices(
                metadata_fake,
                "_unused_metadata_scratch",
                metadata_name,
                _extract_subscript_vals(index_access.subscript),
                clamp=False,
                iteration_indices=iteration_indices,
                resident_source=True,
                indexing_patterns=metadata_patterns,
            )
            result.append(statement_from_string(f"{index_name} = {metadata_ref}[...]"))
            _, member_hbm = _build_dma_slices(
                transfer.tensor,
                "_unused_group_scratch",
                hbm_name,
                list(transfer.subscript),
                clamp=False,
                iteration_indices=iteration_indices,
                hbm_part_overrides={0: "{index}"},
            )
        else:
            index_name = ""
            member_hbm = ""
        if "wait" in methods:
            _, aggregate_hbm = _build_dma_slices(
                transfer.tensor,
                "_unused_group_scratch",
                hbm_name,
                list(transfer.subscript),
                clamp=False,
                iteration_indices=iteration_indices,
                hbm_part_overrides={0: f"pl.ds(0, {plan.group_count})"},
            )
        else:
            aggregate_hbm = ""
        result.extend(
            indirect_group_statements(
                state,
                group_count=plan.group_count,
                index_name=index_name,
                member_hbm=member_hbm,
                aggregate_hbm=aggregate_hbm,
                scratch_ref=resources.scratch_ref(stage_expr),
                semaphore_ref=resources.semaphore_ref(stage_expr),
                direction=transfer.direction,
                methods=methods,
            )
        )
        return result

    def _guarded_statements(
        condition: str, name_hint: str, statements: list[ast.stmt]
    ) -> ast.FunctionDef:
        fn_name = state.device_function.new_var(name_hint)
        fn_def = statement_from_string(
            f"@pl.when({condition})\ndef {fn_name}():\n    pass"
        )
        assert isinstance(fn_def, ast.FunctionDef)
        fn_def.body = statements or [ast.Pass()]
        return fn_def

    prime_statements: list[ast.stmt] = []
    body_prefetch: ast.FunctionDef | None = None
    body_current_stage_waits: list[ast.stmt] = []
    if prefetched_loads or refilled_loads:
        num_iterations = state.device_function.new_var("_num_iterations")
        prime_statements.append(
            statement_from_string(f"{num_iterations} = {grid_parts[-1]}")
        )
        grid_parts[-1] = num_iterations
        prime_indices = [*loop_vars]
        prime_indices[-1] = "0"
        prime_starts: list[ast.stmt] = []
        for transfer in prefetched_loads:
            prime_starts.extend(
                _dma_transfer_statements(transfer, prime_indices, "0", ("start",))
            )
        for transfer, _load_node, _consumer in refilled_loads:
            prime_starts.extend(
                _dma_transfer_statements(
                    transfer,
                    prime_indices,
                    None,
                    ("start",),
                )
            )
        fixed_extent = _fixed_loop_extent(state, len(block_ids) - 1)
        prime_was_hoisted = fixed_extent is not None and (
            _hoist_initial_dma_before_pure_outer_compute(state, prime_starts, loop_vars)
        )
        if not prime_was_hoisted:
            prime_statements.append(
                _guarded_statements(
                    f"{num_iterations} > 0", "_prime_fori_loads", prime_starts
                )
            )

        stage_loop_var = loop_vars[-1]
        next_iteration = f"({stage_loop_var} + 1)"
        next_indices = [*loop_vars]
        next_indices[-1] = next_iteration
        if prefetched_loads:
            next_stage = f"{next_iteration} % 2"
            next_starts: list[ast.stmt] = []
            for transfer in prefetched_loads:
                next_starts.extend(
                    _dma_transfer_statements(
                        transfer, next_indices, next_stage, ("start",)
                    )
                )
            body_prefetch = _guarded_statements(
                f"{next_iteration} < {num_iterations}",
                "_prefetch_fori_loads",
                next_starts,
            )

            current_stage = f"{stage_loop_var} % 2"
            for transfer in prefetched_loads:
                body_current_stage_waits.extend(
                    _dma_transfer_statements(
                        transfer, loop_vars, current_stage, ("wait",)
                    )
                )

        for transfer, load_node, consumer in refilled_loads:
            state.codegen.add_pre_node_statements(
                load_node,
                _dma_transfer_statements(transfer, loop_vars, None, ("wait",)),
            )
            refill_starts = _dma_transfer_statements(
                transfer,
                next_indices,
                None,
                ("start",),
            )
            state.codegen.add_post_node_statement(
                consumer,
                _guarded_statements(
                    f"{next_iteration} < {num_iterations}",
                    "_refill_fori_load",
                    refill_starts,
                ),
            )

    # For loop-carried state, remap args to scratch reads inside the body
    body_args = (
        _remap_args_to_scratch(args, scratch_names, state)
        if has_loop_state
        else [*args]
    )

    # Generate body code within the fori_loop context
    with state.codegen.add_fori_loop(fori_state):
        # Non-DMA tensors keep their outer BlockSpec (whole-shape VMEM ref)
        # and need an absolute offset for ``pl.ds()`` indexing in the body.
        # DMA copies build their own absolute slice via _build_dma_slices,
        # so this offset is dead when every tensor is DMA'd.
        if len(tensor_to_dma_scratch) < len(all_tensor_info):
            for i, bid in enumerate(block_ids):
                offset_name = strategy.offset_var(bid)
                state.codegen.add_statement(
                    statement_from_string(
                        f"{offset_name} = ({begin_exprs[i]}) + ({dim_idx_exprs[i]}) * ({iter_step_exprs[i]})"
                    )
                )

        if body_prefetch is not None:
            state.codegen.add_statement(body_prefetch)

        for transfer in immediate_loads:
            for statement in _dma_transfer_statements(
                transfer, loop_vars, None, ("start", "wait")
            ):
                state.codegen.add_statement(statement)

        for statement in body_current_stage_waits:
            state.codegen.add_statement(statement)

        with state.codegen.resident_prep_lowering_scope(resident_prep_lowerings):
            graph_results = codegen_call_with_graph(
                state.codegen, graph_info.graph, body_args
            )

        if has_loop_state:
            _write_back_loop_carried(state, scratch_names, carried, graph_results)

        for transfer in dma_stores:
            for statement in _dma_transfer_statements(
                transfer, loop_vars, None, ("start", "wait")
            ):
                state.codegen.add_statement(statement)

    for drain in fori_state._remote_send_drains:
        if not drain.waits_deferred:
            continue
        assert drain.pending_counter is not None
        for slot, reference in enumerate(drain.references):
            send_copy = state.device_function.new_var("_send_drain", dce=False)
            send_index = state.device_function.new_var("_send_index", dce=False)
            send_wait_body = state.device_function.new_var("_send_wait_body", dce=False)
            fori_state.outer_suffix.extend(
                (
                    statement_from_string(
                        f"{send_copy} = pltpu.make_async_copy("
                        f"{{reference}}, {{reference}}, {drain.semaphore}.at[{slot}])",
                        reference=reference,
                    ),
                    statement_from_string(
                        f"def {send_wait_body}({send_index}, _):\n"
                        f"    {send_copy}.wait()"
                    ),
                    statement_from_string(
                        f"jax.lax.fori_loop(0, "
                        f"{drain.pending_counter}[{slot}], "
                        f"{send_wait_body}, None)"
                    ),
                )
            )

    for drain in fori_state._remote_recv_drains.values():
        if not drain.waits_deferred:
            continue
        assert drain.starts_per_iteration > 0 or drain.dynamic_start_counter is not None
        recv_copy = state.device_function.new_var("_recv_drain", dce=False)
        recv_index = state.device_function.new_var("_recv_index", dce=False)
        loop_iterations = " * ".join(f"({part})" for part in grid_parts)
        static_receive_count = (
            "0"
            if drain.starts_per_iteration == 0
            else loop_iterations
            if drain.starts_per_iteration == 1
            else f"{drain.starts_per_iteration} * ({loop_iterations})"
        )
        receive_count = static_receive_count
        if drain.dynamic_start_counter is not None:
            dynamic_receive_count = f"{drain.dynamic_start_counter}[0]"
            receive_count = (
                dynamic_receive_count
                if static_receive_count == "0"
                else f"({static_receive_count}) + {dynamic_receive_count}"
            )
        fori_state.outer_suffix.append(
            statement_from_string(
                f"{recv_copy} = pltpu.make_async_copy({{reference}}, "
                f"{{reference}}, {drain.semaphore})",
                reference=drain.reference,
            )
        )
        if drain.dynamic_start_counter is None:
            fori_state.outer_suffix.append(
                statement_from_string(
                    f"for {recv_index} in range({receive_count}):\n"
                    f"    {recv_copy}.wait()"
                )
            )
        else:
            recv_wait_body = state.device_function.new_var("_recv_wait_body", dce=False)
            fori_state.outer_suffix.extend(
                (
                    statement_from_string(
                        f"def {recv_wait_body}({recv_index}, _):\n"
                        f"    {recv_copy}.wait()"
                    ),
                    statement_from_string(
                        f"jax.lax.fori_loop(0, {receive_count}, {recv_wait_body}, None)"
                    ),
                )
            )

    # Compact-worklist outer tile: it IS the grid (exactly one static compact
    # body per work item), so emit the body
    # straight-line with the loop var bound to 0 -- no fori_loop wrapper (which
    # would add control-flow overhead and block pipelining for the common
    # no-ordered-axis case). The ordered inner tile does not reach here: it lowers
    # separately through the loop type selected by the _for_loop Pallas dispatch.
    if _is_compact_tile_loop(state):
        assert not carried
        # No nonlocal declarations: the body runs at kernel scope (scratch refs
        # are kernel params, directly assignable -- nonlocal would be invalid).
        state.add_statement(statement_from_string(f"{loop_vars[0]} = 0"))
        for stmt in body_stmts or [ast.Pass()]:
            state.add_statement(stmt)
        for statement in fori_state.outer_suffix:
            state.add_statement(statement)
        return None

    if not static_unroll:
        _emit_nonlocal_scratch_declarations(state, body_stmts)

    # Emit nested fori_loop calls — one per dimension.
    # Build inside-out: innermost function wraps body_stmts, each outer
    # function wraps the inner fori_loop call.
    # Note: loops are emitted in block_ids order (not loop_order).
    # loop_order is a config knob for the outer grid strategy (NDTileStrategy),
    # not for inner device loops.  For element-wise ops iteration order does
    # not affect correctness; for loop-carried state the user's source order
    # (block_ids order) is the correct semantic order.
    current_body = body_stmts or [ast.Pass()]  # pyrefly: ignore[bad-assignment]
    if static_unroll:
        for dim in reversed(range(len(loop_vars))):
            loop = statement_from_string(
                f"for {loop_vars[dim]} in range({grid_parts[dim]}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body = current_body  # pyrefly: ignore[bad-assignment]
            if dim == len(loop_vars) - 1:
                current_body = [*prime_statements, loop]
            else:
                current_body = [loop]
        for statement in current_body:
            state.add_statement(statement)
        for statement in fori_state.outer_suffix:
            state.add_statement(statement)
        if has_loop_state:
            return _read_final_loop_state(state, result_vars)
        return None

    for dim in reversed(range(len(loop_vars))):
        fn_name = state.device_function.new_var(f"_fori_body_{dim}")
        fn_def = statement_from_string(f"def {fn_name}({loop_vars[dim]}, _): pass")
        assert isinstance(fn_def, ast.FunctionDef)
        fn_def.body = current_body  # pyrefly: ignore[bad-assignment]
        fori_call = statement_from_string(
            f"jax.lax.fori_loop(0, {grid_parts[dim]}, {fn_name}, None)"
        )
        call_prefix = prime_statements if dim == len(loop_vars) - 1 else []
        if dim == 0:
            # Outermost: emit function def and fori_loop call into the kernel
            state.add_statement(fn_def)
            for statement in call_prefix:
                state.add_statement(statement)
            state.add_statement(fori_call)
        else:
            # Inner: wrap in the next outer function's body
            current_body = [fn_def, *call_prefix, fori_call]

    for statement in fori_state.outer_suffix:
        state.add_statement(statement)

    # After fori_loop: read final loop-carried state from scratch
    if has_loop_state:
        return _read_final_loop_state(state, result_vars)
    return None


def _is_static_unroll_predicate(state: CodegenState) -> bool:
    """Whether this predicate is resolved by a surrounding Python tile loop."""
    test = state.proxy_arg(0)
    if isinstance(test, (bool, int)):
        return True
    if not isinstance(test, torch.SymBool):
        return False

    from ..tile_strategy import ForiLoopState
    from ..variable_origin import BlockSizeOrigin
    from ..variable_origin import GridOrigin

    static_block_ids: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, ForiLoopState) and loop.static_unroll:
                static_block_ids.update(loop.block_ids)
    if not static_block_ids:
        return False

    expr = test._sympy_()
    if not isinstance(expr, sympy.Basic):
        return False
    origins = HostFunction.current().expr_to_origin
    for symbol in expr.free_symbols:
        origin_info = origins.get(symbol)
        if origin_info is None:
            return False
        origin = origin_info.origin
        base_type = origin.base_type()
        if issubclass(base_type, BlockSizeOrigin):
            continue
        if not issubclass(base_type, GridOrigin):
            return False
        block_id = getattr(origin, "block_id", None)
        if block_id not in static_block_ids:
            return False
    return True


@_decorators.codegen(_if, "pallas")
def _(state: CodegenState) -> list[object]:
    """Emit dynamic if-conditions for Pallas/TPU using ``lax.cond``.

    JAX's tracing model does not support Python ``if`` on traced values.
    We use ``lax.cond(pred, true_fn, false_fn)`` which requires a scalar
    predicate. Tensor-derived predicates (from tensor loads) are unsupported
    because TPU block shapes make them vectors at runtime.
    """
    from ..device_ir import ElseGraphInfo
    from ..device_ir import IfGraphInfo
    from ..inductor_lowering import codegen_call_with_graph

    graph_info = state.get_graph(state.proxy_arg(1))
    assert isinstance(graph_info, IfGraphInfo)

    test = state.ast_arg(0)
    if_args = state.ast_args[3]
    else_args = state.ast_args[4]
    assert isinstance(if_args, list)
    assert isinstance(else_args, list)
    assert all(isinstance(x, ast.AST) for x in if_args)
    assert all(isinstance(x, ast.AST) for x in else_args)

    from ..generate_ast import GenerateAST

    assert isinstance(state.codegen, GenerateAST)

    if graph_info.predicate_is_tensor:
        raise BackendUnsupported(
            "pallas",
            "if-statements with tensor-derived predicates. "
            "lax.cond requires a scalar predicate, but tensor loads produce "
            "vectors on TPU due to hardware tiling constraints. "
            "Use a scalar kernel argument for the condition instead.",
        )

    if_body_stmts: list[ast.AST] = []
    with state.codegen.set_statements(if_body_stmts):
        if_outputs = codegen_call_with_graph(
            state.codegen, graph_info.graph, [*if_args]
        )

    assert graph_info.else_branch is not None
    else_graph = state.get_graph(graph_info.else_branch)
    assert isinstance(else_graph, ElseGraphInfo)
    else_body_stmts: list[ast.AST] = []
    with state.codegen.set_statements(else_body_stmts):
        else_outputs = codegen_call_with_graph(
            state.codegen, else_graph.graph, [*else_args]
        )

    if_return_names, else_return_names = graph_info.get_branches_return_names(
        state, if_outputs, else_outputs
    )

    if (
        _is_static_unroll_predicate(state)
        and not if_return_names
        and not else_return_names
    ):
        if_node = create(
            ast.If,
            test=test,
            body=if_body_stmts or [ast.Pass()],
            orelse=else_body_stmts or [ast.Pass()],
        )
        state.add_statement(if_node)
        return []

    if_arg_ids = {arg.id for arg in if_args}
    union_args = if_args + [a for a in else_args if a.id not in if_arg_ids]
    arg_list_with_defaults = ", ".join(f"{n.id}={n.id}" for n in union_args)
    if_return_names_str = ""

    if if_return_names:
        if_return_names_str = ", ".join(if_return_names)
        if_return_stmt = statement_from_string(f"return {if_return_names_str}")
        if_body_stmts.append(if_return_stmt)

    if else_return_names:
        else_return_names_str = ", ".join(else_return_names)
        else_return_stmt = statement_from_string(f"return {else_return_names_str}")
        else_body_stmts.append(else_return_stmt)

    if_fn_name = state.device_function.new_var("_if_branch")
    else_fn_name = state.device_function.new_var("_else_branch")

    if_fn_def = statement_from_string(
        f"def {if_fn_name}({arg_list_with_defaults}): pass"
    )
    assert isinstance(if_fn_def, ast.FunctionDef)
    if_fn_def.body = if_body_stmts or [ast.Pass()]  # pyrefly: ignore[bad-assignment]

    else_fn_def = statement_from_string(
        f"def {else_fn_name}({arg_list_with_defaults}): pass"
    )
    assert isinstance(else_fn_def, ast.FunctionDef)
    else_fn_def.body = else_body_stmts or [  # pyrefly: ignore[bad-assignment]
        ast.Pass()
    ]

    state.add_statement(if_fn_def)
    state.add_statement(else_fn_def)

    if (
        if_return_names
    ):  # can also use else_return_names, they will by phi-ed so they will be the same
        state.add_statement(
            statement_from_string(
                f"{if_return_names_str} = lax.cond({{test}}, {if_fn_name}, {else_fn_name})",
                test=test,
            )
        )
    else:
        state.add_statement(
            statement_from_string(
                f"lax.cond({{test}}, {if_fn_name}, {else_fn_name})", test=test
            )
        )

    return cast(
        "list[object]",
        [expr_from_string(n) for n in if_return_names]
        + [expr_from_string(n) for n in else_return_names],
    )


@_decorators.codegen(_and, "pallas")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-return]
    return expr_from_string("{lhs} & {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1))


@_decorators.codegen(_mask_to, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    env = CompileEnvironment.current()
    backend = env.backend
    for dim, size in enumerate(input_sizes):
        if (index := env.resolve_block_id(size)) is not None and (
            mask_var := state.codegen.mask_var(index)
        ) is not None:
            expand = state.tile_strategy.expand_str(input_sizes, dim)
            if env.is_jagged_tile(index):
                mask_shape = env.jagged_tile_mask_shapes[index]
                expand = state.tile_strategy.jagged_tile_expand_str(
                    mask_shape, input_sizes
                )
            # Cast bool mask to float before expanding — Mosaic cannot
            # reshape bool vectors (e.g. vector<32xi1> → vector<32x1xi1>).
            expr = f"({mask_var}.astype(jnp.float32){expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return state.ast_arg(0)
    # Combine float masks via multiplication (equivalent to bool AND).
    mask_expr = " * ".join(mask_exprs)
    if len(mask_exprs) < len(input_sizes):
        mask_expr = backend.broadcast_to_expr(
            mask_expr, state.tile_strategy.shape_str(input_sizes)
        )
    # Ensure the masked value literal matches the tensor dtype
    input_dtype = tensor.dtype
    other_typed = expr_from_string(
        backend.full_expr([], constant_repr(other), input_dtype)
    )
    return expr_from_string(
        backend.where_expr(mask_expr, "{expr}", "{other}"),
        expr=state.ast_arg(0),
        other=other_typed,
    )
