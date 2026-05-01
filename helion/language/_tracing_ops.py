from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.utils import triton_type
from torch.fx import has_side_effect
from torch.fx.experimental.sym_node import SymNode

from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.dtype_utils import cast_ast
from .._compiler.host_function import HostFunction
from .._compiler.variable_origin import BlockSizeOrigin
from ..exc import BackendUnsupported
from ..exc import NotInsideKernel
from . import _decorators
from .tile_proxy import Tile

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.tile_strategy import TileStrategy
    from ..runtime.config import Config

    _T = TypeVar("_T", bound=object)

"""
This file contains "fake" ops that cannot appear in user program but
are generated while compiling the user program. These ops are used to
generate code for certain constructs.
"""

_symbolic_types = (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)


def is_for_loop_target(target: object) -> bool:
    return target in (_for_loop, _for_loop_step)


@_decorators.api()
def _get_symnode(debug_name: str) -> int:
    """FX requires a torch.SymInt to come from an op. This is a fake op is added lazily to work around this."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_get_symnode, "common")
def _(state: CodegenState) -> ast.AST:
    # pyrefly: ignore [missing-attribute]
    val = state.fx_node.meta["val"]

    # Handle the case where val is a regular integer (e.g., from reduction_loops config)
    if isinstance(val, int):
        return expr_from_string(str(val))

    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    sym_expr = getattr(getattr(val, "node", None), "_expr", None)
    if not isinstance(sym_expr, sympy.Expr):
        sym_expr = val._sympy_()
    origin_info = HostFunction.current().expr_to_origin.get(sym_expr)

    if origin_info is not None and isinstance(origin_info.origin, BlockSizeOrigin):
        block_size_var = state.device_function.block_size_var(
            origin_info.origin.block_id
        )
        if block_size_var is None:
            return expr_from_string("1")
        return expr_from_string(block_size_var)
    return state.codegen.lift_symnode(
        expr_from_string(state.sympy_expr(sym_expr)),
        sym_expr,
        dce=True,
        prefix="symnode",
    )


@_decorators.codegen(_get_symnode, "cute")
def _(state: CodegenState) -> ast.AST:
    # pyrefly: ignore [missing-attribute]
    val = state.fx_node.meta["val"]
    if isinstance(val, int):
        return expr_from_string(str(val))

    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    sym_expr = getattr(getattr(val, "node", None), "_expr", None)
    if not isinstance(sym_expr, sympy.Expr):
        sym_expr = val._sympy_()
    origin_info = HostFunction.current().expr_to_origin.get(sym_expr)
    if origin_info is not None and isinstance(origin_info.origin, BlockSizeOrigin):
        block_size_var = state.device_function.block_size_var(
            origin_info.origin.block_id
        )
        if block_size_var is None:
            return expr_from_string("1")
        return expr_from_string(block_size_var)
    return state.codegen.lift_symnode(
        expr_from_string(state.sympy_expr(sym_expr)),
        sym_expr,
        dce=True,
        prefix="symnode",
    )


@_decorators.api()
def _host_tensor(debug_name: str) -> torch.Tensor:
    """Source of a tensor that was allocated on the host and must be passed to the kernel as an arg."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_host_tensor, "common")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_host_tensor")  # should be unused


@_decorators.api()
def _constant_tensor(value: float, dtype: torch.dtype) -> torch.Tensor:
    """
    Source of a constant scalar tensor created inside a kernel.
    This is generated when torch.tensor(val) is called inside a kernel.
    """
    raise AssertionError("this should never be called")


@_decorators.codegen(_constant_tensor, "common")
def _(state: CodegenState) -> ast.AST:
    value = state.proxy_arg(0)
    dtype = state.proxy_arg(1)
    assert isinstance(value, (int, float, bool))
    assert isinstance(dtype, torch.dtype)
    return expr_from_string(
        CompileEnvironment.current().backend.full_expr([], constant_repr(value), dtype)
    )


@has_side_effect
@_decorators.api()
def _for_loop(
    graph_id: int,
    begin: list[int],
    end: list[int],
    args: list[object],
) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


@has_side_effect
@_decorators.api()
def _for_loop_step(
    graph_id: int,
    begin: list[int],
    end: list[int],
    args: list[object],
    step: list[int | None],
) -> list[object]:
    """Stepped ``for`` loops mapped into FX."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop_step, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


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

    When ``pallas_loop_type="emit_pipeline"``, generates ``pltpu.emit_pipeline``
    calls with automatic DMA pipelining.  When ``pallas_loop_type="fori_loop"``,
    generates ``jax.lax.fori_loop`` with explicit ``pltpu.make_async_copy`` DMA.
    Otherwise falls through to the common ``ForLoopGraphInfo.codegen`` path.
    """
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "unroll")
    if pallas_loop_type == "emit_pipeline":
        return _codegen_emit_pipeline(state)
    if pallas_loop_type == "fori_loop":
        return _codegen_fori_loop(state)
    # unroll: fall through to common codegen path
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


@_decorators.codegen(_for_loop_step, "pallas")
def _(state: CodegenState) -> None:
    """Emit inner stepped device loops for Pallas/TPU."""
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "unroll")
    if pallas_loop_type == "emit_pipeline":
        _codegen_emit_pipeline(state)
        return None
    if pallas_loop_type == "fori_loop":
        _codegen_fori_loop(state)
        return None
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


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
    from .memory_ops import load as _load_op
    from .memory_ops import store as _store_op

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
                    loaded_tensors[key] = (fake, tensor_node, sub_vals)
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


def _get_dim_block_ids(
    subscript_meta: list[object],
    env: CompileEnvironment,
) -> dict[int, int]:
    """Map tensor dimension index -> block_id from subscript metadata."""
    dim_to_bid: dict[int, int] = {}
    if not isinstance(subscript_meta, (list, tuple)):
        return dim_to_bid
    for dim_idx, idx in enumerate(subscript_meta):
        if isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None:
                dim_to_bid[dim_idx] = bid
        elif isinstance(idx, slice) and idx == slice(None):
            pass
    return dim_to_bid


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


def _compute_grid_and_block_sizes(
    state: CodegenState,
    block_ids: list[int],
    env: CompileEnvironment,
) -> tuple[list[str], list[str]]:
    """Compute grid dimensions and block size vars for the given block_ids."""
    grid_parts: list[str] = []
    block_size_vars: list[str] = []
    for block_id in block_ids:
        block_size_var = state.device_function.block_size_var(block_id)
        assert block_size_var is not None
        block_size_vars.append(block_size_var)
        block_value = env.block_sizes[block_id].from_config(state.config)
        if block_value is not None:
            state.device_function.constexpr_arg(block_size_var, block_value)
        numel_expr = state.sympy_expr(env.block_sizes[block_id].numel)
        grid_parts.append(
            env.backend.cdiv_expr(numel_expr, block_size_var, is_device=True)
        )
    return grid_parts, block_size_vars


def _pallas_loop_begin_and_step_exprs(
    state: CodegenState,
    block_ids: list[int],
    block_size_vars: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return begin, per-iteration step, and slice-size expressions for loop dims."""
    begins = state.proxy_arg(1)
    steps = state.proxy_arg(4) if len(state.proxy_args) > 4 else None

    if not isinstance(begins, (list, tuple)):
        begins = [begins]
    if not isinstance(steps, (list, tuple)):
        steps = [steps] * len(block_ids)

    begin_exprs: list[str] = []
    iter_step_exprs: list[str] = []
    slice_size_exprs: list[str] = []

    for i in range(len(block_ids)):
        begin = begins[i]
        step = steps[i]
        begin_expr = state.sympy_expr(sympy.sympify(begin))
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


def _resolve_shape(
    proxy: torch.Tensor,
    env: CompileEnvironment,
    config: Config,
) -> tuple[int, ...]:
    """Resolve symbolic tile sizes to concrete block sizes from config."""
    resolved = []
    for s in proxy.shape:
        bid = env.resolve_block_id(s)
        if bid is not None:
            bs = env.block_sizes[bid].from_config(config)
            assert isinstance(bs, int)
            resolved.append(bs)
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
            # Reuse existing scratch if the init value is already in one
            # (e.g. from hl.full / hl.zeros). Otherwise allocate new.
            existing = any(
                s.name == arg_ast.id for s in state.device_function._scratch_args
            )
            if existing:
                scratch_name = arg_ast.id
            else:
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
            index (``_pipeline_indices[i]`` for emit_pipeline; the fori_loop
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
) -> bool:
    """Set up mask variables for inner-loop block_ids.

    Args:
        offset_expr_fn: Given (block_id_index, block_size_var), returns a string
            expression for the per-element offset (e.g. "_j * bs + jnp.arange(bs)").

    Returns True if any mask requires explicit indices.
    """
    needs_explicit = False
    if hasattr(strategy, "_setup_mask"):
        for i, bid in enumerate(block_ids):
            block_value = env.block_sizes[bid].from_config(state.config)
            assert isinstance(block_value, int)
            numel_expr = state.sympy_expr(env.block_sizes[bid].numel)
            offset_var = state.device_function.new_var(f"offset_{bid}")
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
    from .view_ops import subscript as _subscript_op

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

    from .view_ops import subscript as _subscript_op

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

    from .._compiler.inductor_lowering import FakeGraphLowering
    from .._compiler.inductor_lowering import compile_lock
    from .._compiler.inductor_lowering import prepare_node_lowering
    from .view_ops import subscript as _subscript_op

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

    # Re-prepare lowerings for modified outer nodes
    if reshaped:
        with compile_lock:
            graph_lowering = FakeGraphLowering()
            with V.set_graph_handler(graph_lowering):
                for node in reshaped:
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
    from .view_ops import subscript as _subscript_op

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

    from .._compiler.inductor_lowering import FakeGraphLowering
    from .._compiler.inductor_lowering import compile_lock
    from .._compiler.inductor_lowering import prepare_node_lowering

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


@_decorators.api()
def _pre_broadcast_tile(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """Tile a pre-broadcast tensor along its last dim to match target_size."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_pre_broadcast_tile)
def _(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    new_shape = [*tensor.shape[:-1], target_size]
    return tensor.new_empty(new_shape)


@_decorators.codegen(_pre_broadcast_tile, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor_ast = state.ast_arg(0)
    target_size = state.proxy_arg(1)
    if isinstance(target_size, torch.SymInt):
        target_expr = state.sympy_expr(target_size._sympy_())
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


def _codegen_emit_pipeline(state: CodegenState) -> object:
    """Emit inner device loops using pltpu.emit_pipeline.

    Handles both simple load->compute->store pipelines and loops with
    loop-carried state (accumulators, running max/sum) by converting
    the state into scratch VMEM buffers.
    """
    from .._compiler.device_ir import ForLoopGraphInfo
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.inductor_lowering import codegen_call_with_graph
    from .._compiler.tile_strategy import EmitPipelineLoopState
    from .._compiler.tile_strategy import LoopDimInfo

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

    grid_parts, block_size_vars = _compute_grid_and_block_sizes(state, block_ids, env)

    loaded_tensors, stored_tensors = _classify_loop_tensors(graph_info, state)
    begin_exprs, iter_step_exprs, slice_size_exprs = _pallas_loop_begin_and_step_exprs(
        state, block_ids, block_size_vars
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
    from .._compiler.device_function import DeviceFunction as _DF

    _bid_to_pid_var: dict[int, str] = {}
    device_fn = _DF.current()
    if device_fn.pid is not None:
        for g, pid in enumerate(device_fn.pid.pid_info):
            pid_var = f"_outer_pid_{g}"
            state.add_statement(
                statement_from_string(f"{pid_var} = pl.program_id({g})")
            )
            _bid_to_pid_var[pid.block_id] = pid_var

    def _make_block_spec(fake: torch.Tensor, subscript_meta: list[object]) -> str:
        """Build a BlockSpec string for a tensor accessed in the pipeline body.

        Encodes BOTH outer grid dims (via pl.program_id) and inner pipeline
        dims into the BlockSpec lambda, so the full HBM tensor can be passed
        without pre-slicing.
        """
        dim_to_bid = _get_dim_block_ids(subscript_meta, env)
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
                block_shape_parts.append(slice_size_expr)
                from .memory_ops import _record_pad_info

                _record_pad_info(state, fake, dim_idx, bid)
                if begin_expr == "0" and iter_step_expr == slice_size_expr:
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
                else:
                    block_shape_parts.append(str(int(shape[dim_idx])))
                lambda_parts.append(pid_var)
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

    def _make_hbm_slice(
        fake: torch.Tensor, hbm_name: str, subscript_meta: list[object]
    ) -> str:
        """Build an HBM ref slicing expression for outer grid dims."""
        dim_to_bid = _get_dim_block_ids(subscript_meta, env)
        shape = fake.shape
        parts: list[str] = []
        needs_slice = False
        for dim_idx in range(len(shape)):
            bid = dim_to_bid.get(dim_idx)
            if bid is not None and bid not in block_ids:
                grid_loops = state.codegen.active_device_loops.get(bid)
                if grid_loops:
                    offset = state.codegen.offset_var(bid)
                    bs_var = state.device_function.block_size_var(bid)
                    if bs_var:
                        parts.append(f"pl.ds({offset}, {bs_var})")
                        needs_slice = True
                    else:
                        parts.append(":")
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

    # Record which tensors are in the pipeline body (need HBM refs)
    from .._compiler.device_function import PallasMemorySpace

    for fake, _tensor_node, _sub_meta in loaded_tensors.values():
        state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
    for fake, _tensor_node, _sub_meta in stored_tensors.values():
        state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM

    # Process loaded tensors (inputs to pipeline)
    for key, (fake, _tensor_node, sub_meta) in loaded_tensors.items():
        if key in stored_tensors:
            continue  # Handle as output instead
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        in_tensors.append((fake, hbm_name))
        in_specs.append(_make_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        # Pass full HBM ref -- BlockSpec lambda handles outer grid indexing
        pipeline_in_args.append(hbm_name)

    # Process stored tensors (outputs of pipeline, may also be read)
    for fake, _tensor_node, sub_meta in stored_tensors.values():
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        out_tensors.append((fake, hbm_name))
        out_specs.append(_make_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        # Pass full HBM ref -- BlockSpec lambda handles outer grid indexing
        pipeline_out_args.append(hbm_name)

    # Build the body function
    body_fn_name = state.device_function.new_var("_pipeline_body")
    body_stmts: list[ast.AST] = []

    # Build block_id_to_info for the pipeline state
    block_id_to_info: dict[int, LoopDimInfo] = {}
    for block_id in block_ids:
        block_id_to_info[block_id] = LoopDimInfo(
            end_var_name=None,
            end_expr=env.block_sizes[block_id].numel,
        )

    strategy = _find_strategy(state, block_ids)
    # Emit offset_<bid>/indices_<bid> at the body prologue.
    _emit_inner_loop_offset_indices(
        state,
        strategy,
        block_ids,
        block_size_vars,
        begin_exprs,
        iter_step_exprs,
        [f"_pipeline_indices[{i}]" for i in range(len(block_ids))],
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
            f"_pipeline_indices[{i}] * {bs} + jnp.arange({bs})"
        ),
    )
    # Build tensor_to_vmem mapping
    tensor_to_vmem: dict[str, str] = {}
    idx = 0
    for _fake, hbm_name in in_tensors:
        tensor_to_vmem[hbm_name] = body_params[idx]
        idx += 1
    for _fake, hbm_name in out_tensors:
        tensor_to_vmem[hbm_name] = body_params[idx]
        idx += 1

    # Create the pipeline loop state
    pipeline_state = EmitPipelineLoopState(
        strategy=strategy,  # pyrefly: ignore[bad-argument-type]
        block_id_to_info=block_id_to_info,
        body_fn_name=body_fn_name,
        inner_statements=body_stmts,
    )
    pipeline_state._tensor_to_vmem = tensor_to_vmem  # type: ignore[attr-defined]

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


def _check_dma_alignment(vmem_shape: tuple[int, ...]) -> bool:
    """Check if a VMEM buffer shape satisfies TPU DMA alignment.

    DMA requires last dim % 128 == 0 and second-to-last dim % 8 == 0
    for 2D+ tensors. Note that these rules are currently optimized for
    bf16 sublanes; they are overly conservative for f32 (no constraint)
    and too lenient for 1D (which should be % 1024).

    These rules differ from outer BlockSpec constraints where 1D is
    dtype-dependent: 128 * (32 / bitwidth(dtype)). Unlike outer BlockSpecs,
    emit_pipeline/fori_loop inner DMA does NOT have a ``block == tensor_dim``
    exception.
    """
    if len(vmem_shape) >= 2:
        return vmem_shape[-1] % 128 == 0 and vmem_shape[-2] % 8 == 0
    if len(vmem_shape) == 1:
        return vmem_shape[0] % 128 == 0
    return True


def _compute_vmem_shapes(
    all_tensor_info: list[tuple[torch.Tensor, list[object], str]],
    block_ids: list[int],
    slice_size_exprs: list[str],
    env: CompileEnvironment,
    state: CodegenState,
) -> list[tuple[int, ...]]:
    """Compute VMEM buffer shapes for each tensor in the fori_loop body."""
    vmem_shapes: list[tuple[int, ...]] = []
    for fake, sub_meta, _direction in all_tensor_info:
        dim_to_bid = _get_dim_block_ids(sub_meta, env)
        parts: list[int] = []
        for dim_idx in range(len(fake.shape)):
            bid = dim_to_bid.get(dim_idx)
            if bid is not None and bid in block_ids:
                bid_idx = block_ids.index(bid)
                block_value_sym = sympy.sympify(slice_size_exprs[bid_idx])
                if isinstance(block_value_sym, sympy.Integer):
                    parts.append(int(block_value_sym))
                else:
                    block_value = env.block_sizes[block_ids[bid_idx]].from_config(
                        state.config
                    )
                    assert isinstance(block_value, int)
                    parts.append(block_value)
            elif bid is not None:
                outer_block_value = env.block_sizes[bid].from_config(state.config)
                if isinstance(outer_block_value, int):
                    parts.append(outer_block_value)
                else:
                    parts.append(int(fake.shape[dim_idx]))
            else:
                parts.append(int(fake.shape[dim_idx]))
        vmem_shapes.append(tuple(parts))
    return vmem_shapes


def _codegen_fori_loop(state: CodegenState) -> object:
    """Emit inner device loops using jax.lax.fori_loop.

    When inner block shapes satisfy TPU DMA alignment, uses
    ``pltpu.make_async_copy`` for double-buffered DMA pipelining.
    Otherwise, falls back to direct ``pl.ds`` slicing on HBM refs
    (no DMA, no alignment requirement).
    """
    from .._compiler.device_ir import ForLoopGraphInfo
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.inductor_lowering import codegen_call_with_graph
    from .._compiler.tile_strategy import ForiLoopState
    from .._compiler.tile_strategy import LoopDimInfo

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
    begin_exprs, iter_step_exprs, slice_size_exprs = _pallas_loop_begin_and_step_exprs(
        state, block_ids, block_size_vars
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

    # Collect all tensors: load-only first, then stored (which may also be read)
    all_tensor_info: list[tuple[torch.Tensor, list[object], str]] = []
    for key, (fake, _tensor_node, sub_meta) in loaded_tensors.items():
        if key not in stored_tensors:
            all_tensor_info.append((fake, sub_meta, "load"))
    for fake, _tensor_node, sub_meta in stored_tensors.values():
        all_tensor_info.append((fake, sub_meta, "store"))

    # Compute VMEM shapes and check DMA alignment
    vmem_shapes = _compute_vmem_shapes(
        all_tensor_info, block_ids, slice_size_exprs, env, state
    )
    use_dma = all(_check_dma_alignment(shape) for shape in vmem_shapes)

    # With DMA: tensors need HBM refs (no outer BlockSpec) because DMA
    # handles all slicing, and we register VMEM scratch buffers +
    # semaphores for each tensor.  Without DMA: outer BlockSpecs handle
    # outer dims and inner dims use pl.ds(), so neither is needed.
    tensor_to_vmem: dict[str, str] = {}
    tensor_to_sem: dict[str, str] = {}
    if use_dma:
        from .._compiler.device_function import PallasMemorySpace

        for fake, _tensor_node, _sub_meta in loaded_tensors.values():
            state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
        for fake, _tensor_node, _sub_meta in stored_tensors.values():
            state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
        for (fake, _sub_meta, _direction), vmem_shape in zip(
            all_tensor_info, vmem_shapes, strict=True
        ):
            hbm_name = state.device_function.tensor_arg(fake).name
            vmem_name = state.device_function.register_scratch(
                vmem_shape,
                fake.dtype,
                name_hint=hbm_name.replace("_hbm", "") + "_buf",
            )
            sem_name = state.device_function.register_dma_semaphore(
                name_hint=hbm_name.replace("_hbm", "") + "_sem",
            )
            tensor_to_vmem[hbm_name] = vmem_name
            tensor_to_sem[hbm_name] = sem_name

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
    block_id_to_info: dict[int, LoopDimInfo] = {}
    for block_id in block_ids:
        block_id_to_info[block_id] = LoopDimInfo(
            end_var_name=None,
            end_expr=env.block_sizes[block_id].numel,
        )

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

    # Create ForiLoopState (body_fn_name and loop_var_name are currently
    # unused by consumers but stored for debugging; use outermost values)
    fori_state = ForiLoopState(
        strategy=strategy,  # pyrefly: ignore[bad-argument-type]
        block_id_to_info=block_id_to_info,
        body_fn_name="_fori_body_0",
        loop_var_name=loop_vars[0],
        use_dma=use_dma,
        inner_statements=body_stmts,
        _tensor_to_vmem=tensor_to_vmem,
        _tensor_to_sem=tensor_to_sem,
    )

    def _build_hbm_dma_slice(
        fake: torch.Tensor, hbm_name: str, subscript_meta: list[object]
    ) -> str:
        """Build an HBM ref slicing expression for DMA with loop variable."""
        dim_to_bid = _get_dim_block_ids(subscript_meta, env)
        shape = fake.shape
        parts: list[str] = []
        needs_slice = False
        for dim_idx in range(len(shape)):
            bid = dim_to_bid.get(dim_idx)
            if bid is not None and bid in block_ids:
                bid_idx = block_ids.index(bid)
                begin_expr = begin_exprs[bid_idx]
                iter_step_expr = iter_step_exprs[bid_idx]
                slice_size_expr = slice_size_exprs[bid_idx]
                dim_idx_expr = dim_idx_exprs[bid_idx]
                parts.append(
                    f"pl.ds(({begin_expr}) + ({dim_idx_expr}) * ({iter_step_expr}), {slice_size_expr})"
                )
                needs_slice = True
                from .memory_ops import _record_pad_info

                _record_pad_info(state, fake, dim_idx, bid)
            elif bid is not None and bid not in block_ids:
                # Outer grid dim: use grid offset
                grid_loops = state.codegen.active_device_loops.get(bid)
                if grid_loops:
                    offset = state.codegen.offset_var(bid)
                    bs_var = state.device_function.block_size_var(bid)
                    if bs_var:
                        parts.append(f"pl.ds({offset}, {bs_var})")
                        needs_slice = True
                    else:
                        parts.append(":")
                else:
                    parts.append(":")
            else:
                parts.append(":")
        if not needs_slice:
            return hbm_name
        return f"{hbm_name}.at[{', '.join(parts)}]"

    # For loop-carried state, remap args to scratch reads inside the body
    body_args = (
        _remap_args_to_scratch(args, scratch_names, state)
        if has_loop_state
        else [*args]
    )

    # Generate body code within the fori_loop context
    with state.codegen.add_fori_loop(fori_state):
        if use_dma:
            # Emit DMA read copies at start of body
            for fake, _tensor_node, sub_meta in loaded_tensors.values():
                hbm_name = state.device_function.tensor_arg(fake).name
                vmem_name = tensor_to_vmem[hbm_name]
                sem_name = tensor_to_sem[hbm_name]
                src_slice = _build_hbm_dma_slice(fake, hbm_name, sub_meta)
                copy_var = state.device_function.new_var("_copy")
                state.codegen.add_statement(
                    statement_from_string(
                        f"{copy_var} = pltpu.make_async_copy({src_slice}, {vmem_name}, {sem_name})"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(f"{copy_var}.start()")
                )
                state.codegen.add_statement(statement_from_string(f"{copy_var}.wait()"))

        # Codegen the user's body (loads/stores remapped via _tensor_to_vmem
        # when using DMA, or accessing HBM refs directly when not)
        graph_results = codegen_call_with_graph(
            state.codegen, graph_info.graph, body_args
        )

        # Write updated loop-carried values back to scratch
        if has_loop_state:
            _write_back_loop_carried(state, scratch_names, carried, graph_results)

        if use_dma:
            # Emit DMA write copies at end of body for stored tensors
            for fake, _tensor_node, sub_meta in stored_tensors.values():
                hbm_name = state.device_function.tensor_arg(fake).name
                vmem_name = tensor_to_vmem[hbm_name]
                sem_name = tensor_to_sem[hbm_name]
                dst_slice = _build_hbm_dma_slice(fake, hbm_name, sub_meta)
                copy_out_var = state.device_function.new_var("_copy_out")
                state.codegen.add_statement(
                    statement_from_string(
                        f"{copy_out_var} = pltpu.make_async_copy({vmem_name}, {dst_slice}, {sem_name})"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(f"{copy_out_var}.start()")
                )
                state.codegen.add_statement(
                    statement_from_string(f"{copy_out_var}.wait()")
                )

    # Emit nested fori_loop calls — one per dimension.
    # Build inside-out: innermost function wraps body_stmts, each outer
    # function wraps the inner fori_loop call.
    # Note: loops are emitted in block_ids order (not loop_order).
    # loop_order is a config knob for the outer grid strategy (NDTileStrategy),
    # not for inner device loops.  For element-wise ops iteration order does
    # not affect correctness; for loop-carried state the user's source order
    # (block_ids order) is the correct semantic order.
    current_body = body_stmts or [ast.Pass()]  # pyrefly: ignore[bad-assignment]
    for dim in reversed(range(len(loop_vars))):
        fn_name = state.device_function.new_var(f"_fori_body_{dim}")
        fn_def = statement_from_string(f"def {fn_name}({loop_vars[dim]}, _): pass")
        assert isinstance(fn_def, ast.FunctionDef)
        fn_def.body = current_body  # pyrefly: ignore[bad-assignment]
        fori_call = statement_from_string(
            f"jax.lax.fori_loop(0, {grid_parts[dim]}, {fn_name}, None)"
        )
        if dim == 0:
            # Outermost: emit function def and fori_loop call into the kernel
            state.add_statement(fn_def)
            state.add_statement(fori_call)
        else:
            # Inner: wrap in the next outer function's body
            current_body = [fn_def, fori_call]

    # After fori_loop: read final loop-carried state from scratch
    if has_loop_state:
        return _read_final_loop_state(state, result_vars)
    return None


@has_side_effect
@_decorators.api()
def _while_loop(
    cond_graph_id: int,
    body_graph_id: int,
    args: list[object],
    orelse_graph_id: int | None = None,
) -> list[object]:
    """Represent a while loop in FX since FX lacks native control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_while_loop, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(1)).codegen(state)


@has_side_effect
@_decorators.api()
def _if(
    test: object,
    if_graph_id: int,
    else_graph_id: int,
    if_args: list[object],
    else_args: list[object],
) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_if, "common")
def _(state: CodegenState) -> list[object]:
    return state.get_graph(state.proxy_arg(1)).codegen(state)


@_decorators.codegen(_if, "pallas")
def _(state: CodegenState) -> list[object]:
    """Emit dynamic if-conditions for Pallas/TPU using ``lax.cond``.

    JAX's tracing model does not support Python ``if`` on traced values.
    We use ``lax.cond(pred, true_fn, false_fn)`` which requires a scalar
    predicate. Tensor-derived predicates (from tensor loads) are unsupported
    because TPU block shapes make them vectors at runtime.
    """
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.device_ir import ElseGraphInfo
    from .._compiler.device_ir import IfGraphInfo
    from .._compiler.inductor_lowering import codegen_call_with_graph

    graph_info = state.get_graph(state.proxy_arg(1))
    assert isinstance(graph_info, IfGraphInfo)

    test = state.ast_arg(0)
    if_args = state.ast_args[3]
    else_args = state.ast_args[4]
    assert isinstance(if_args, list)
    assert isinstance(else_args, list)
    assert all(isinstance(x, ast.AST) for x in if_args)
    assert all(isinstance(x, ast.AST) for x in else_args)

    from .._compiler.generate_ast import GenerateAST

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

    assert graph_info.if_arg_names is not None
    assert graph_info.else_arg_names is not None
    assert graph_info.branches_outputs is not None

    arg_node_name_to_ast_name = {
        graph_info.if_arg_names[i]: if_args[i].id for i in range(len(if_args))
    } | {graph_info.else_arg_names[i]: else_args[i].id for i in range(len(else_args))}

    if_return_names = [
        cast("ast.Name", if_outputs[o]).id
        if isinstance(o, int)
        else arg_node_name_to_ast_name[o]
        for (o, _) in graph_info.branches_outputs
    ]
    else_return_names = [
        cast("ast.Name", else_outputs[o]).id
        if isinstance(o, int)
        else arg_node_name_to_ast_name[o]
        for (_, o) in graph_info.branches_outputs
    ]

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


@_decorators.codegen(_if, "cute")
def _(state: CodegenState) -> list[object]:
    """Emit dynamic if-conditions for the CuTe DSL backend.

    CuTe DSL forbids referencing a variable after a dynamic if/else when the
    variable is first defined inside the branches. Pre-declare any such output
    in the outer scope before emitting the if so both branches reassign it.
    """
    from .._compiler.ast_extension import create
    from .._compiler.device_ir import ElseGraphInfo
    from .._compiler.device_ir import IfGraphInfo
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.inductor_lowering import codegen_call_with_graph

    graph_info = state.get_graph(state.proxy_arg(1))
    assert isinstance(graph_info, IfGraphInfo)
    assert isinstance(state.codegen, GenerateAST)

    test = state.ast_arg(0)
    if_args = state.ast_args[3]
    else_args = state.ast_args[4]
    assert isinstance(if_args, list)
    assert isinstance(else_args, list)
    assert all(isinstance(x, ast.AST) for x in if_args)
    assert all(isinstance(x, ast.AST) for x in else_args)

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

    # Pre-declare any variable that is first defined inside both branches in the
    # outer scope so CuTe DSL can resolve it after the if/else. The phi pass
    # later renames the else-branch's name to match the if-branch's name, so we
    # use the if-branch output name as the canonical pre-declared name.
    if graph_info.branches_outputs is not None:
        if_output_node = graph_info.graph.find_nodes(op="output")[0]
        if_graph_outputs = cast("tuple[object, ...]", if_output_node.args[0])
        backend = CompileEnvironment.current().backend
        for if_entry, else_entry in graph_info.branches_outputs:
            if not (isinstance(if_entry, int) and isinstance(else_entry, int)):
                continue
            if_name_node = if_outputs[if_entry]
            assert isinstance(if_name_node, ast.Name)
            fx_out = if_graph_outputs[if_entry]
            if not isinstance(fx_out, torch.fx.Node):
                continue
            val = fx_out.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            dtype_str = backend.dtype_str(val.dtype)
            state.add_statement(
                statement_from_string(f"{if_name_node.id} = {dtype_str}(0)")
            )

    if not if_body_stmts:
        if_body_stmts.append(ast.Pass())
    if not else_body_stmts:
        else_body_stmts.append(ast.Pass())
    if_ast_node = create(ast.If, test=test, body=if_body_stmts, orelse=else_body_stmts)
    state.add_statement(if_ast_node)

    return if_outputs + else_outputs


# Note we can't DCE phi nodes because there may be a loop carry dependency not captured in the outer graph
@has_side_effect
@_decorators.api(allow_host_tensor=True)
def _phi(lhs: object, rhs: object) -> object:
    """Combine values from different branches of a control flow."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_phi)
def _(lhs: object, rhs: object) -> object:
    if isinstance(lhs, Tile):
        assert isinstance(rhs, Tile)
        assert lhs.block_id == rhs.block_id
        return lhs
    assert isinstance(lhs, torch.Tensor), lhs
    assert isinstance(rhs, torch.Tensor), rhs
    assert lhs.size() == rhs.size()
    assert lhs.dtype == rhs.dtype
    assert lhs.device == rhs.device
    return torch.empty_like(lhs)


@_decorators.codegen(_phi, "common")
def _(state: CodegenState) -> ast.Name:
    lhs = state.ast_arg(0)
    assert isinstance(lhs, ast.Name), lhs
    rhs = state.ast_arg(1)
    assert isinstance(rhs, ast.Name), rhs
    state.device_function.merge_variable_names(lhs.id, rhs.id)
    return lhs


@_decorators.get_masked_value(_phi)
def _(node: torch.fx.Node) -> float | bool | None:
    lhs, rhs = node.args
    assert isinstance(lhs, torch.fx.Node)
    assert isinstance(rhs, torch.fx.Node)

    from .._compiler.node_masking import cached_masked_value

    lval = cached_masked_value(lhs)
    if lval is not None:
        rval = cached_masked_value(rhs)
        if lval == rval:
            return lval
    return None


@_decorators.api()
def _inductor_lowering_extra(args: list[object]) -> torch.Tensor:
    """
    When we have an inductor lowering that results in multiple inductor
    buffers, we insert this fake op in the graph to represent intermediate
    values.
    """
    raise AssertionError("this should never be called")


@_decorators.api()
def _and(left: object, right: object) -> object:
    raise NotInsideKernel


@_decorators.codegen(_and, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-return]
    return expr_from_string(
        "{lhs} and {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )


@_decorators.codegen(_and, "pallas")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-return]
    return expr_from_string("{lhs} & {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1))


@_decorators.register_fake(_and)
def _(left: object, right: object) -> object:
    if not isinstance(left, _symbolic_types):
        if not left:
            return left
        return right
    if not isinstance(right, _symbolic_types):
        if not right:
            return right
        return left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool) and isinstance(right, torch.SymBool):
        return torch.SymBool(
            SymNode(
                sympy.And(left._sympy_(), right._sympy_()),
                env.shape_env,
                bool,
                hint=None,
            )
        )
    # TODO(jansel): should match the type of the input
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.api()
def _or(left: object, right: object) -> object:
    raise NotInsideKernel


@_decorators.register_fake(_or)
def _(left: object, right: object) -> object:
    if not isinstance(left, _symbolic_types):
        if left:
            return left
        return right
    if not isinstance(right, _symbolic_types):
        if right:
            return right
        return left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool) and isinstance(right, torch.SymBool):
        return torch.SymBool(
            SymNode(
                sympy.Or(left._sympy_(), right._sympy_()),
                env.shape_env,
                bool,
                hint=None,
            )
        )
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.codegen(_or, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-return]
    return expr_from_string(
        "{lhs} or {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )


@_decorators.api()
def _not(left: object) -> object:
    raise NotInsideKernel


@_decorators.register_fake(_not)
def _(left: object) -> object:
    if not isinstance(left, _symbolic_types):
        return not left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool):
        return torch.SymBool(
            SymNode(sympy.Not(left._sympy_()), env.shape_env, bool, hint=None)
        )
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.codegen(_not, "common")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "not {lhs}",
        lhs=state.ast_arg(0),
    )


@_decorators.codegen(_not, "pallas")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "jnp.logical_not({lhs})",
        lhs=state.ast_arg(0),
    )


@_decorators.api()
def _mask_to(tensor: torch.Tensor, other: float | bool, /) -> torch.Tensor:
    """
    Set the masked out values of a given tile to a specific value.
    This operation is automatically generated by the compiler when doing a
    dot or reduction operation, and should not need to be called directly
    by users.

    Args:
        tensor: The tensor to apply the mask to.
        other: The value to set the masked out elements to.

    Returns:
        torch.Tensor: A tensor with the masked out elements set to `other`.
    """
    raise NotInsideKernel


@_decorators.register_fake(_mask_to)
def _(tensor: torch.Tensor, other: float) -> torch.Tensor:
    return torch.empty_like(tensor)


@_decorators.codegen(_mask_to, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    env = CompileEnvironment.current()
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

            expr = f"({mask_var}{expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = "&".join(mask_exprs)
    if len(mask_exprs) < len(input_sizes):
        mask_expr = f"tl.broadcast_to({mask_expr}, {state.tile_strategy.shape_str(input_sizes)})"
    # Ensure the masked value literal matches the tensor dtype to avoid unintended upcasts
    input_dtype = tensor.dtype
    other_typed = expr_from_string(
        f"tl.full([], {constant_repr(other)}, {triton_type(input_dtype)})"
    )
    return expr_from_string(
        f"tl.where({mask_expr}, {{expr}}, {{other}})",
        expr=state.ast_arg(0),
        other=other_typed,
    )


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
            expr = f"({mask_var}{expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = "&".join(mask_exprs)
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


@_decorators.codegen(_mask_to, "metal")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    for size in input_sizes:
        if (
            index := CompileEnvironment.current().resolve_block_id(size)
        ) is not None and (mask_var := state.codegen.mask_var(index)) is not None:
            if mask_var not in mask_exprs:
                mask_exprs.append(mask_var)
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = " and ".join(mask_exprs)
    input_dtype = tensor.dtype
    other_typed = CompileEnvironment.current().backend.cast_ast(
        expr_from_string(constant_repr(other)),
        input_dtype,
    )
    return expr_from_string(
        "({expr} if {mask} else {other})",
        expr=state.ast_arg(0),
        mask=expr_from_string(mask_expr),
        other=other_typed,
    )


@_decorators.codegen(_mask_to, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))

    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    for dim, size in enumerate(input_sizes):
        if (
            index := CompileEnvironment.current().resolve_block_id(size)
        ) is not None and (mask_var := state.codegen.mask_var(index)) is not None:
            expand = state.tile_strategy.expand_str(input_sizes, dim)
            expr = f"({mask_var}{expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = " and ".join(mask_exprs)
    input_dtype = tensor.dtype
    expr_typed = cast_ast(state.ast_arg(0), input_dtype)
    other_typed = CompileEnvironment.current().backend.cast_ast(
        expr_from_string(constant_repr(other)),
        input_dtype,
    )
    return expr_from_string(
        "({expr} if {mask} else {other})",
        expr=expr_typed,
        mask=expr_from_string(mask_expr),
        other=other_typed,
    )


@_decorators.get_masked_value(_mask_to)
def _(node: torch.fx.Node) -> float | bool:
    value = node.args[1]
    assert isinstance(value, (int, float, bool))
    return value


@_decorators.api(allow_host_tensor=True)
def _new_var(value: _T, /) -> _T:
    """
    Create a shallow copy of a value that is assigned a fresh variable in codegen.

    This is used to ensure phi() node handling works properly when a value is renamed
    without mutation in a loop.  We need to copy the inputs to a loop so that phi nodes
    are handled properly.  Phi nodes will merge variable names from outside the loop,
    but the old value of those variables could have usages.
    """
    raise NotInsideKernel


@_decorators.register_fake(_new_var)
def _(value: _T) -> _T:
    if isinstance(value, torch.Tensor):
        # pyrefly: ignore [bad-return]
        return torch.empty_like(value)
    if isinstance(value, torch.SymInt):
        # pyrefly: ignore [bad-return]
        return CompileEnvironment.current().create_unbacked_symint()
    if isinstance(value, (int, float, bool)) or value is None:
        # pyrefly: ignore [bad-return]
        return value
    raise NotImplementedError(f"Unsupported type for _new_var: {type(value)}")


@_decorators.codegen(_new_var, "common")
def _(state: CodegenState) -> ast.AST:
    value = state.ast_arg(0)
    assert isinstance(value, ast.AST)
    varname = state.codegen.tmpvar(
        prefix=value.id if isinstance(value, ast.Name) else "new_var"
    )
    state.add_statement(statement_from_string(f"{varname} = {{expr}}", expr=value))
    return create(ast.Name, id=varname, ctx=ast.Load())


@_decorators.get_masked_value(_new_var)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    (arg,) = node.args
    assert isinstance(arg, torch.fx.Node)
    return cached_masked_value(arg)
