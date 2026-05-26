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
    When ``pallas_loop_type="outer_grid"`` AND the matmul-pattern eligibility
    check passes, the inner K loop is lifted into the outer ``pl.pallas_call``
    grid (matching ``examples/pallas_perf/matmul_pallas.py``); otherwise
    falls back to ``emit_pipeline``.  Otherwise falls through to the common
    ``ForLoopGraphInfo.codegen`` path.
    """
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "unroll")
    if pallas_loop_type == "outer_grid":
        return _codegen_outer_grid_or_fallback(state)
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
    if pallas_loop_type == "outer_grid":
        _codegen_outer_grid_or_fallback(state)
        return None
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


def _get_loop_begin_and_end(
    state: CodegenState, loop_dim_index: int
) -> tuple[str, str]:
    """Extract the begin and end values from the _for_loop state args."""
    ast_begins = state.ast_args[1]
    ast_ends = state.ast_args[2]
    begins = list(ast_begins) if isinstance(ast_begins, (list, tuple)) else [ast_begins]
    ends = list(ast_ends) if isinstance(ast_ends, (list, tuple)) else [ast_ends]

    def _to_str(value: object) -> str:
        if isinstance(value, ast.AST):
            return ast.unparse(value)
        return str(value)

    return _to_str(begins[loop_dim_index]), _to_str(ends[loop_dim_index])


def _get_loop_numel(state: CodegenState, loop_dim_index: int) -> str:
    begin, end = _get_loop_begin_and_end(state, loop_dim_index)
    return f"(({end}) - ({begin}))"


def _compute_grid_and_block_sizes(
    state: CodegenState,
    block_ids: list[int],
    env: CompileEnvironment,
) -> tuple[list[str], list[str]]:
    """Compute grid dimensions and block size vars for the given block_ids."""
    grid_parts: list[str] = []
    block_size_vars: list[str] = []
    for i, block_id in enumerate(block_ids):
        block_size_var = state.device_function.block_size_var(block_id)
        assert block_size_var is not None
        block_size_vars.append(block_size_var)
        block_value = state.device_function.resolved_block_size(block_id)
        if block_value is not None:
            state.device_function.constexpr_arg(block_size_var, block_value)
        numel_expr = _get_loop_numel(state, i)
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
    steps = state.proxy_arg(4) if len(state.proxy_args) > 4 else None

    if not isinstance(steps, (list, tuple)):
        steps = [steps] * len(block_ids)

    begin_exprs: list[str] = []
    iter_step_exprs: list[str] = []
    slice_size_exprs: list[str] = []

    for i in range(len(block_ids)):
        step = steps[i]
        begin_expr, _ = _get_loop_begin_and_end(state, i)
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


def _compute_pipeline_or_dma_extra_pad(
    begin_expr: str,
    bid: int,
    env: CompileEnvironment,
    state: CodegenState,
) -> int:
    """Return extra host-side padding for a pipeline/DMA dim with a non-zero begin.

    When ``pl.ds(offset, block_size)`` reads from a tensor whose loop starts
    at a non-zero begin, the last block can overshoot the tensor boundary
    beyond what ``(-shape) % block_size`` accounts for.  The worst case is
    ``block_size - 1`` extra elements when the begin is data-dependent.

    # TODO(dunfanlu): if begin isn't "0" but is another constexpr int,
    # we should be able to use a smaller padding than bs-1?
    """
    if begin_expr == "0":
        return 0
    bs_val = state.device_function.resolved_block_size(bid)
    if isinstance(bs_val, int):
        return bs_val - 1
    return 0


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
                if _try_inplace_accumulator_rewrite(state, sname, result):
                    continue
                state.codegen.add_statement(_scratch_write_stmt(state, sname, result))


def _try_inplace_accumulator_rewrite(
    state: CodegenState,
    sname: str,
    result: ast.AST,
) -> bool:
    """Rewrite ``result = scratch[...] + RHS; scratch[...] = result`` to
    ``scratch[...] += RHS``.

    The hand-written reference Pallas matmul accumulates with
    ``acc_ref[...] += pl.dot(x_val, y_val)`` directly into the VMEM scratch.
    Helion's default lowering emits the equivalent value-flow as a separate
    ``acc_N = scratch[...] + dot(...)`` followed by ``scratch[...] =
    acc_N[...]``, which the Mosaic backend then materializes through an
    extra register binding per K iteration. Recognising the pattern here
    lets the inner pipeline body stay in place on the VMEM ref between
    iterations.

    Returns True when the rewrite fires; the caller then skips emitting the
    standard write-back statement. The rewrite is intentionally narrow: it
    only matches when the result Name's most recent binding is
    ``LHS + RHS`` and the LHS chain (through trivial ``a = b`` renames)
    terminates at a Subscript read of *sname* with the same slice the
    write-back would use. Anything else falls back to the unchanged
    behaviour.
    """
    if not isinstance(result, ast.Name):
        return False

    stmts = state.codegen.statements_stack[-1]
    sl = state.device_function.scratch_read_slice(sname)
    scratch_idx = sl or "..."

    # Locate the assignment that binds `result`, walking through trivial
    # Name-to-Name renames added by ``codegen_call_with_graph``'s placeholder
    # copy step.
    name_to_find = result.id
    binding_index: int | None = None
    binding_stmt: ast.Assign | None = None
    i = len(stmts) - 1
    while i >= 0:
        stmt = stmts[i]
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name_to_find
        ):
            value = stmt.value
            if isinstance(value, ast.Name):
                name_to_find = value.id
                i -= 1
                continue
            binding_index = i
            binding_stmt = stmt
            break
        i -= 1

    if binding_stmt is None or binding_index is None:
        return False

    value = binding_stmt.value
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return False

    lhs_chain = _lhs_chain_to_scratch_read(
        stmts, binding_index, value.left, sname, scratch_idx
    )
    if lhs_chain is None:
        return False

    # `result` must not be referenced anywhere else in the body; otherwise
    # dropping its binding would leave a dangling reference. The write-back
    # statement we are replacing is the only consumer in the matmul pattern.
    if _name_referenced_outside(stmts, result.id, skip_index=binding_index):
        return False

    rhs = value.right
    # Drop the binding for `result` and any intermediate Name aliases that
    # only existed to feed the LHS of the addition. ``lhs_chain`` contains
    # the indices (descending) of the trivial ``a = b`` and
    # ``a = sname[...]`` statements that the LHS resolves through; each can
    # be dropped only when the bound name is not referenced elsewhere.
    drop_indices = [binding_index]
    for idx, name in lhs_chain:
        if _name_referenced_outside(
            stmts, name, skip_index=binding_index, additional_skips=drop_indices
        ):
            break
        drop_indices.append(idx)
    for idx in sorted(drop_indices, reverse=True):
        stmts.pop(idx)
    state.codegen.add_statement(
        statement_from_string(f"{sname}[{scratch_idx}] += {{rhs}}", rhs=rhs)
    )
    return True


def _lhs_chain_to_scratch_read(
    stmts: list[ast.AST],
    before_index: int,
    expr: ast.AST,
    sname: str,
    scratch_idx: str,
) -> list[tuple[int, str]] | None:
    """If *expr* ultimately reads ``sname[scratch_idx]`` return the chain
    of intermediate ``a = b`` / ``a = sname[...]`` statements, else None.

    The chain is a list of ``(statement_index, bound_name)`` pairs in
    descending order of *statement_index* (most recent first). The caller
    may use the indices to drop the now-redundant intermediate bindings
    when fusing the read/compute/store into ``sname[...] += rhs``.
    """
    target_repr = f"{sname}[{scratch_idx}]"
    chain: list[tuple[int, str]] = []
    current = expr
    j = before_index - 1
    while True:
        if isinstance(current, ast.Subscript):
            if ast.unparse(current) == target_repr:
                return chain
            return None
        if not isinstance(current, ast.Name):
            return None
        lookup_name = current.id
        found_binding = False
        while j >= 0:
            stmt = stmts[j]
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == lookup_name
            ):
                chain.append((j, lookup_name))
                current = stmt.value
                j -= 1
                found_binding = True
                break
            j -= 1
        if not found_binding:
            return None


def _name_referenced_outside(
    stmts: list[ast.AST],
    name: str,
    *,
    skip_index: int,
    additional_skips: list[int] | None = None,
) -> bool:
    """Return True if *name* is read in any stmt except *skip_index* and
    the optional *additional_skips* list.

    Only Load-context references count: a Store-context occurrence (the
    LHS of the binding we plan to drop) is not a dependency.
    """
    skips: set[int] = {skip_index}
    if additional_skips:
        skips.update(additional_skips)
    for idx, stmt in enumerate(stmts):
        if idx in skips:
            continue
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Name)
                and node.id == name
                and isinstance(node.ctx, ast.Load)
            ):
                return True
    return False


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
            block_value = state.device_function.resolved_block_size(bid)
            assert isinstance(block_value, int)
            numel_expr = _get_loop_numel(state, i)
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


def _outer_pid_block_is_singleton(
    env: CompileEnvironment,
    config: object,
    pid: object,
) -> bool:
    """Return True when the outer-grid axis described by ``pid`` resolves to
    a configured block size of 1.

    The outer-grid lift rewrite (``_apply_outer_grid_rewrites``) assumes the
    loop-carried accumulator is a multi-row / multi-column matmul tile.  When
    the configured block size for an outer axis is 1 (e.g. ``bm == 1`` on the
    bf16 1×K×N row of the matrix), the scratch holds a degenerate 1-row or
    1-column tile and the rewrite emits silently-wrong outputs whenever the
    inner K loop has > 1 iteration.  Treating singleton outer-grid blocks as
    ineligible falls back to ``_codegen_emit_pipeline``, which handles the
    degenerate shape correctly.
    """
    block_id = pid.block_id  # pyrefly: ignore[missing-attribute]
    if not (0 <= block_id < len(env.block_sizes)):
        return False
    block_size_info = env.block_sizes[block_id]
    value = block_size_info.from_config(config)  # type: ignore[arg-type]
    return isinstance(value, int) and value == 1


def _codegen_outer_grid_or_fallback(state: CodegenState) -> object:
    """Lift the inner K loop into the outer ``pl.pallas_call`` grid when the
    matmul-pattern eligibility check passes; otherwise fall back to
    ``_codegen_emit_pipeline``.

    The hand-written reference matmul (``examples/pallas_perf/matmul_pallas.py``)
    uses a 3-axis outer grid ``(grid_m, grid_n, grid_k)`` with
    ``dimension_semantics = ("parallel", "parallel", "arbitrary")``, a
    ``pl.when(pl.program_id(2) == 0)`` accumulator init guard, and a
    ``pl.when(pl.program_id(2) == nsteps - 1)`` accumulator store guard.
    The inner loop is a single ``scratch[...] += pl.dot(x_tile, y_tile)``
    statement; ``X``/``Y`` outer in_specs are ``(bm, bk)`` / ``(bk, bn)``
    indexed by ``(i, k)`` / ``(k, j)`` rather than ``(i, 0)`` / ``(0, j)``
    VMEM strips, which lets Mosaic pipeline across the 8 grid iterations.

    Helion's default ``emit_pipeline`` path keeps K inside
    ``pltpu.emit_pipeline`` (a serial ``lax.fori_loop`` with double-buffered
    DMA but no cross-iteration compute pipelining); the outer-grid form
    matches the hand-written reference exactly.

    Eligibility:
      * Single inner block_id (the K reduction axis).
      * Block size info marks that block_id as ``reduction=True``.
      * Inner loop has loop-carried state (an accumulator).
      * The outer grid currently has at least one non-reduction axis
        (i.e. matmul-like; pure-reduction kernels stay on emit_pipeline).

    On any miss, falls back transparently to ``_codegen_emit_pipeline`` so
    the autotuner doesn't have to know about which inner loops qualify.
    """
    from .._compiler.device_function import DeviceFunction as _DF
    from .._compiler.device_function import PallasOuterGridLiftedAxis
    from .._compiler.device_ir import ForLoopGraphInfo

    graph_info = state.get_graph(state.proxy_arg(0))
    assert isinstance(graph_info, ForLoopGraphInfo)
    block_ids = graph_info.block_ids
    env = CompileEnvironment.current()
    device_fn = _DF.current()

    eligible = False
    if (
        len(block_ids) == 1
        and device_fn.pid is not None
        and len(device_fn.pid.pid_info) >= 1
    ):
        # Loop-carried state: the for_loop call passes the carried args as
        # the last entry in ``ast_args`` / ``proxy_args``.  An accumulator
        # pattern needs at least one carried tensor.  This is the semantic
        # marker for "this inner loop is a reduction" — Helion's tile-loop
        # block sizes don't carry an explicit ``reduction=True`` flag for
        # user-written ``for tile_k in hl.tile(k)`` (only
        # ``allocate_reduction_dimension`` sets it, and the lowering of
        # ``torch.addmm`` does not), so we infer reduction-ness from the
        # presence of accumulator state.
        last_args = state.ast_args[-1] if isinstance(state.ast_args, list) else None
        if isinstance(last_args, list) and len(last_args) > 0:
            # Refuse to lift if any outer-grid block_id is also a reduction
            # (some other helion pattern); the outer-grid restructure only
            # makes sense when M/N really are independent.
            outer_has_reduction = any(
                env.block_sizes[pid.block_id].reduction
                for pid in device_fn.pid.pid_info
            )
            # Refuse to lift when any outer-grid axis resolves to a
            # configured block size of 1.  The body rewrite that lifts the
            # inner K loop into the outer grid assumes the loop-carried
            # accumulator is a 2D matmul tile; when ``bm == 1`` (or any
            # outer-tile dim == 1) the scratch holds a 1-row / 1-col
            # broadcast outer product instead and the rewrite produces
            # silently-wrong outputs whenever the K loop has > 1 iteration
            # (NaN / mismatch up to 4.5e6 vs CPU f32 reference; the
            # autotuner's accuracy check skips these configs, but forced
            # configs are unprotected).  Fall back to ``emit_pipeline``
            # which handles singleton dims correctly.
            outer_has_singleton_block = any(
                _outer_pid_block_is_singleton(env, state.config, pid)
                for pid in device_fn.pid.pid_info
            )
            if not outer_has_reduction and not outer_has_singleton_block:
                eligible = True

    if not eligible:
        return _codegen_emit_pipeline(state)

    # Record the body-list position before emit_pipeline runs so the
    # post-codegen rewrite (``_apply_outer_grid_rewrites``) can locate
    # the scratch init / pipeline call / scratch read it needs to wrap
    # in ``pl.when`` guards.  Only ``idx_before`` is captured; the matching
    # ``idx_after`` would be invalidated by DCE in the apply pass, which
    # searches for the pipeline call by content instead.
    body_list = state.codegen.statements_stack[-1]
    idx_before = len(body_list)
    result = _codegen_emit_pipeline(state)

    # Determine the K loop's outer grid dim index: append after the
    # existing outer-grid PIDs.  ``_compute_reduction_grid_dims`` will
    # then flag it as a reduction so the launcher marks it ``"arbitrary"``.
    block_id = block_ids[0]
    nsteps_expr = env.backend.cdiv_expr(
        _get_loop_numel(state, 0),
        state.device_function.block_size_var(block_id) or "1",
        is_device=False,
    )

    # The matched scratch is the loop-carried scratch.  ``_codegen_emit_pipeline``
    # registered exactly one scratch for the carried accumulator; pick the
    # most-recently registered VMEM scratch (the only one created in this
    # pipeline emission frame).
    scratch_name = ""
    for sa in reversed(device_fn._scratch_args):
        if sa.scratch_type == "vmem":
            scratch_name = sa.name
            break
    if not scratch_name:
        # Fallback: no scratch was registered; the rewrite would have nothing
        # to wrap, so leave as emit_pipeline form.
        return result

    device_fn.pallas_outer_grid_lifted.append(
        PallasOuterGridLiftedAxis(
            block_id=block_id,
            nsteps_expr=nsteps_expr,
            scratch_name=scratch_name,
        )
    )

    # Extend ``pid_info`` with the lifted K axis BEFORE the host wrapper
    # (``codegen_function_call``) runs so the host's
    # ``grid=(grid_m, grid_n, grid_k)`` expression and
    # ``_block_spec_info``'s ``block_id → grid_dim`` mapping both include
    # the new axis.  The K axis is placed at ``len(pid_info)``; the body
    # rewrite emits ``_outer_pid_K = pl.program_id(<that index>)`` to
    # match.  Mirror the matmul reference: K stays ``"arbitrary"`` via the
    # existing ``_compute_reduction_grid_dims`` flow once we set the K
    # block_size's ``reduction`` flag.
    from .._compiler.program_id import PIDInfo

    k_grid_dim = len(device_fn.pid.pid_info)  # type: ignore[union-attr]
    k_pid_var = device_fn.new_var(f"_outer_pid_{k_grid_dim}")
    block_size_var = device_fn.block_size_var(block_id) or "1"
    k_numel = env.block_sizes[block_id].numel
    device_fn.pid.pid_info.append(  # type: ignore[union-attr]
        PIDInfo(
            pid_var=k_pid_var,
            block_size_var=block_size_var,
            numel=k_numel,
            block_id=block_id,
        )
    )
    # ``_compute_reduction_grid_dims`` reads
    # ``device_fn.pallas_outer_grid_lifted`` (set immediately below) and
    # adds the lifted K grid dim to the reduction list, so the launcher
    # marks it ``"arbitrary"`` in ``dimension_semantics``.  Avoid mutating
    # ``env.block_sizes[block_id].reduction`` because the BlockSizeInfo is
    # shared across re-compilations and other ``pallas_loop_type`` configs
    # would silently inherit the flag flip.

    # Strip the K block_id from the inner-pipeline tracking so the
    # launcher emits a regular outer ``(bm, bk) / (bk, bn)`` BlockSpec for
    # X / Y instead of an HBM ref + VMEM strip pair.  This is what makes
    # the 3-axis grid actually carry K through the outer pallas_call: with
    # X / Y now bound by ``_block_spec_info``'s ``(0, k_grid_dim)`` /
    # ``(k_grid_dim, 1)`` mapping, Mosaic gets the same outer specs the
    # hand-written matmul reference uses.  Walk the live emit_pipeline
    # body output (the just-emitted ``def _pipeline_body(...)`` plus the
    # ``pltpu.emit_pipeline(...)(x, y)`` call) to discover which outer
    # refs are bound to the pipeline body's params (e.g. ``x`` <- ``x_vmem``)
    # and reset their pipeline state.
    _reset_pipeline_state_for_lifted_axis(device_fn, body_list, idx_before)

    # Tag the position of the emit_pipeline call so the post-codegen rewrite
    # can identify exactly which call to replace.  ``_apply_outer_grid_rewrites``
    # searches the live body for the pipeline call (DCE may rearrange
    # earlier setup statements between capture and now).
    if not hasattr(device_fn, "_pallas_outer_grid_pending_rewrites"):
        device_fn._pallas_outer_grid_pending_rewrites = []  # type: ignore[attr-defined]
    device_fn._pallas_outer_grid_pending_rewrites.append(  # type: ignore[attr-defined]
        {
            "body_list": body_list,
            "scratch_name": scratch_name,
            "block_id": block_id,
            "nsteps_expr": nsteps_expr,
            "k_grid_dim": k_grid_dim,
            "k_pid_var": k_pid_var,
        }
    )
    return result


def _reset_pipeline_state_for_lifted_axis(
    device_fn: object,
    body_list: list[ast.AST],
    idx_before: int,
) -> None:
    """Reset HBM / VMEM-strip tags on tensors bound through the pipeline
    body that's about to be inlined into the outer-grid form.

    The just-emitted ``_codegen_emit_pipeline`` call appends:
      def _pipeline_body(_pipeline_indices, x_vmem, y_vmem): ...
      pltpu.emit_pipeline(_pipeline_body, ...)(x, y)

    The pipeline call's args (``x``, ``y``) are the outer ref names whose
    pipeline tagging we want to clear so the launcher emits a regular
    BlockSpec for them on the now-3-axis outer grid.
    """
    from .._compiler.device_function import DeviceFunction
    from .._compiler.device_function import PallasMemorySpace
    from .._compiler.device_function import TensorArg

    assert isinstance(device_fn, DeviceFunction)
    referenced_names: set[str] = set()
    for stmt in body_list[idx_before:]:
        if _is_emit_pipeline_call_stmt(stmt):
            assert isinstance(stmt, ast.Expr)
            call = stmt.value
            assert isinstance(call, ast.Call)
            for arg in call.args:
                if isinstance(arg, ast.Name):
                    referenced_names.add(arg.id)
    for arg in device_fn._tensor_args.values():
        if not isinstance(arg, TensorArg):
            continue
        if arg.name in referenced_names:
            tid = id(arg.fake_value)
            if device_fn.pallas_memory_space.get(tid) == PallasMemorySpace.HBM:
                del device_fn.pallas_memory_space[tid]
            device_fn.pallas_pipeline_vmem_strip.discard(tid)


def _apply_outer_grid_rewrites(
    device_fn: object,
    pending: list[dict[str, object]],
) -> None:
    """Rewrite ``DeviceFunction.body`` for each lifted outer-grid axis.

    For every entry in *pending*, replaces the
    ``scratch_init / pltpu.emit_pipeline(...)(args) / scratch_read``
    triple plus the subsequent convert+store statements with a
    matmul_pallas.py-style 3-axis outer-grid form:

      _outer_pid_K = pl.program_id(<K_grid_dim>)
      @pl.when(_outer_pid_K == 0)
      def _init():
          <init statements>

      <inline pipeline body using outer refs and _outer_pid_K>

      @pl.when(_outer_pid_K == <nsteps - 1>)
      def _store():
          <post-pipeline statements: scratch read, convert, store>

    The rewrite is a pure AST surgery — no new tensors / scratches /
    block sizes are introduced.  ``pid_info`` is also extended with a
    PIDInfo for the K block_id so that ``pid.codegen_grid()`` returns the
    3-axis grid tuple at host-wrapper emission time and
    ``_compute_block_spec_info`` maps the X/Y K dim to the new grid axis.
    Any pending entry whose scaffolding doesn't match the expected
    matmul shape is left alone (the body stays in emit_pipeline form),
    so the eligibility check in ``_codegen_outer_grid_or_fallback`` can
    stay coarse and the rewrite stays the strict gate.
    """
    from .._compiler.device_function import DeviceFunction
    from .._compiler.device_function import PallasOuterGridLiftedAxis

    assert isinstance(device_fn, DeviceFunction)

    for entry in pending:
        # ``body_list`` is the live device_function.body reference captured
        # at codegen time, but DCE / scratch-arg insertion may have removed
        # early ``offset_N / indices_N / begin_N`` setup statements between
        # capture and now, so the ``idx_before`` snapshot from
        # ``_codegen_outer_grid_or_fallback`` is unreliable.  Search the
        # body by content for the pipeline call instead — there's exactly
        # one emit_pipeline call per pending entry by construction (each
        # entry corresponds to a single matched inner loop).
        body_list = cast("list[ast.AST]", entry["body_list"])
        scratch_name = cast("str", entry["scratch_name"])
        block_id = cast("int", entry["block_id"])

        pipeline_call_idx = -1
        pipeline_body_idx = -1
        for i, stmt in enumerate(body_list):
            if _is_emit_pipeline_call_stmt(stmt):
                pipeline_call_idx = i
            elif (
                isinstance(stmt, ast.FunctionDef)
                and stmt.name.startswith("_pipeline_body")
                and pipeline_body_idx == -1
            ):
                pipeline_body_idx = i
        if pipeline_call_idx < 0 or pipeline_body_idx < 0:
            # Body shape didn't match (e.g. fori_loop fallback or another
            # pass already rewrote this body); skip.
            continue

        # Walk back from the pipeline body def to find the scratch init
        # ``scratch_N[...] = acc[...]`` and the surrounding init statements
        # (outer accumulator assignment ``acc = jnp.full(...)``).  The
        # scratch init line is the start of the init block we want to
        # guard with ``pl.when(_outer_pid_K == 0)``.
        init_start = 0
        scratch_init_idx = -1
        for i in range(pipeline_body_idx - 1, -1, -1):
            stmt = body_list[i]
            if _is_scratch_init_for(stmt, scratch_name):
                scratch_init_idx = i
                break
        if scratch_init_idx < 0:
            # No scratch init found — bail out and leave the body as-is.
            continue
        # Walk back from scratch_init_idx to swallow any contiguous run of
        # outer-pre-init statements (e.g. ``acc = jnp.full(...)``,
        # ``_outer_pid_N = pl.program_id(N)`` etc.).  We stop at the first
        # statement that doesn't look like outer-init / pid-setup.
        init_start = scratch_init_idx
        while init_start > 0 and _looks_like_outer_init_stmt(body_list[init_start - 1]):
            init_start -= 1

        init_end_exclusive = pipeline_body_idx

        # Post-pipeline statements: from pipeline_call_idx + 1 through end
        # of the body (or until the next outer-grid pid setup, if multiple
        # matmuls are stacked; the simple "to-end" form works for the
        # headline since the kernel has a single matmul).
        post_start = pipeline_call_idx + 1
        post_end_exclusive = len(body_list)

        # ``_outer_pid_N = pl.program_id(N)`` setup statements stay outside
        # the pl.when guards (cheap and used by both branches).  The rest
        # of init_start..init_end_exclusive becomes the guarded init body.
        guard_init_stmts: list[ast.AST] = []
        keep_outside_stmts: list[ast.AST] = []
        for stmt in body_list[init_start:init_end_exclusive]:
            if _is_outer_pid_setup_stmt(stmt):
                keep_outside_stmts.append(stmt)
            else:
                guard_init_stmts.append(stmt)

        # K's PIDInfo was already appended in ``_codegen_outer_grid_or_fallback``
        # so the host grid is already 3-axis; pick up the saved indices.
        k_grid_dim = cast("int", entry["k_grid_dim"])
        k_pid_var = cast("str", entry["k_pid_var"])

        # Compute device-side ``_K_NSTEPS_<i> = pl.num_programs(<k_grid_dim>)``
        # so the store guard's ``_outer_pid_K == _K_NSTEPS - 1`` expression
        # works with both compile-time and runtime grid sizes.
        nsteps_var = device_fn.new_var(f"_k_nsteps_{k_grid_dim}")

        # Extract the pipeline body's parameter names and arguments from
        # the ``pltpu.emit_pipeline(_pipeline_body, ...)(x, y)`` call so we
        # can substitute ``x_vmem -> x`` etc. when inlining.
        pipeline_body_fn = body_list[pipeline_body_idx]
        assert isinstance(pipeline_body_fn, ast.FunctionDef)
        body_param_names = [arg.arg for arg in pipeline_body_fn.args.args]
        pipeline_call_args = _extract_emit_pipeline_call_args(
            body_list[pipeline_call_idx]
        )
        if (
            len(body_param_names) < 1
            or len(pipeline_call_args) != len(body_param_names) - 1
        ):
            # Unexpected shape — skip rewrite (leave body as emit_pipeline).
            continue
        # body_param_names[0] is ``_pipeline_indices``; remaining are tensor
        # refs.  Replace tensor params with the outer ref names; replace
        # ``_pipeline_indices[0]`` with the K pid var.
        param_renames: dict[str, str] = {}
        for i, param_name in enumerate(body_param_names[1:]):
            param_renames[param_name] = pipeline_call_args[i]
        pipeline_indices_name = body_param_names[0]
        inner_body_stmts = _rename_and_dereference_body(
            list(pipeline_body_fn.body),
            param_renames,
            pipeline_indices_name,
            k_pid_var,
        )

        # Build the guarded init / inlined body / guarded store.
        is_first_k = f"{k_pid_var} == 0"
        is_last_k = f"{k_pid_var} == ({nsteps_var}) - 1"

        new_section: list[ast.AST] = []
        new_section.extend(keep_outside_stmts)
        new_section.extend(
            [
                statement_from_string(f"{k_pid_var} = pl.program_id({k_grid_dim})"),
                statement_from_string(f"{nsteps_var} = pl.num_programs({k_grid_dim})"),
                _wrap_in_pl_when("_init", is_first_k, guard_init_stmts),
            ]
        )
        new_section.extend(inner_body_stmts)
        new_section.append(
            _wrap_in_pl_when(
                "_store", is_last_k, body_list[post_start:post_end_exclusive]
            )
        )

        # Splice ``new_section`` in place of the old [init_start, end) range.
        del body_list[init_start:post_end_exclusive]
        for offset, stmt in enumerate(new_section):
            body_list.insert(init_start + offset, stmt)

        # Record on the device function (also flushed into the public list
        # populated by the codegen so the launcher can pick it up).
        device_fn.pallas_outer_grid_lifted[-1] = PallasOuterGridLiftedAxis(
            block_id=block_id,
            nsteps_expr=cast("str", entry["nsteps_expr"]),
            scratch_name=scratch_name,
        )

        # Strip the K block_id from any pipeline-related sets so the launcher
        # treats X/Y as ordinary outer-grid tiles (proper BlockSpec) rather
        # than HBM refs / VMEM strips.  This is what makes the launcher emit
        # the 3-axis ``(bm, bk)/(bk, bn)`` block specs that Mosaic can
        # pipeline across the outer grid.
        _restore_outer_grid_tensor_specs(
            device_fn, body_list[init_start : init_start + len(new_section)]
        )


def _is_emit_pipeline_call_stmt(stmt: ast.AST) -> bool:
    """True iff *stmt* is an ``Expr(call=pltpu.emit_pipeline(...)(...))`` stmt."""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    inner = call.func
    if not isinstance(inner, ast.Call):
        return False
    fn = inner.func
    return (
        isinstance(fn, ast.Attribute)
        and fn.attr == "emit_pipeline"
        and isinstance(fn.value, ast.Name)
        and fn.value.id == "pltpu"
    )


def _is_scratch_init_for(stmt: ast.AST, scratch_name: str) -> bool:
    """True iff *stmt* assigns ``<scratch_name>[<slice>] = <expr>``.

    The slice is either ``...`` or ``[:, :]`` style; both are accepted.
    """
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    target = stmt.targets[0]
    if not isinstance(target, ast.Subscript):
        return False
    return isinstance(target.value, ast.Name) and target.value.id == scratch_name


def _looks_like_outer_init_stmt(stmt: ast.AST) -> bool:
    """True iff *stmt* looks like an outer-pre-init or pid-setup statement.

    Conservative: only accepts simple ``<name> = <expr>`` assignments where
    the RHS is a ``Call`` or ``Subscript`` (i.e. ``acc = jnp.full(...)``,
    ``_outer_pid_N = pl.program_id(N)``, ``offset_N = ...``).
    """
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        return isinstance(target, ast.Name)
    return False


def _is_outer_pid_setup_stmt(stmt: ast.AST) -> bool:
    """True iff *stmt* assigns ``_outer_pid_N = pl.program_id(N)``."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    target = stmt.targets[0]
    if not isinstance(target, ast.Name) or not target.id.startswith("_outer_pid_"):
        return False
    value = stmt.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "program_id"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "pl"
    )


def _drop_dead_outer_pid_reads(body_list: list[ast.AST]) -> None:
    """DCE top-level ``_outer_pid_N = pl.program_id(N)`` setup statements
    whose LHS name is never read elsewhere in *body_list*.

    The ``outer_grid`` body rewrite and ``emit_pipeline`` codegen both
    emit one ``_outer_pid_N`` setup per outer-grid axis (M / N / lifted
    K), even when the body only references a subset of them.  For the
    matmul headline the ``outer_grid`` path uses only ``_outer_pid_K``
    (inside the ``@pl.when`` guards); the ``emit_pipeline`` strip path
    uses none of the M / N pids inside its BlockSpec lambdas (the
    lambda emits ``0`` for strip-tagged outer dims).  Leaving the dead
    reads in place costs ~5% on the headline (hand-edit ablation: the
    matmul_pallas.py mirror went from 125.5 us to 132.9 us when the
    same two dead reads were inserted).

    The pass is a strict use-def DCE on flat top-level statements.
    Reference detection uses ``ast.walk`` so nested function bodies
    (e.g. ``@pl.when(...) def _init():`` / ``def _pipeline_body``) and
    BlockSpec lambdas are inspected — a pid var read from inside any of
    those keeps its setup alive.  The setup statement itself only
    writes its LHS (the call to ``pl.program_id`` is closed over), so
    excluding the LHS of setup statements from the "uses" set is the
    only subtlety.
    """
    used_pid_names: set[str] = set()
    setup_indices: list[int] = []
    setup_lhs_names: list[str] = []

    for idx, stmt in enumerate(body_list):
        if _is_outer_pid_setup_stmt(stmt):
            assert isinstance(stmt, ast.Assign)
            target = stmt.targets[0]
            assert isinstance(target, ast.Name)
            setup_indices.append(idx)
            setup_lhs_names.append(target.id)
        else:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id.startswith("_outer_pid_")
                ):
                    used_pid_names.add(node.id)

    # Drop dead setups in reverse order so earlier indices stay valid.
    for idx, lhs_name in zip(
        reversed(setup_indices), reversed(setup_lhs_names), strict=True
    ):
        if lhs_name not in used_pid_names:
            del body_list[idx]


def _extract_emit_pipeline_call_args(stmt: ast.AST) -> list[str]:
    """Return the call args of ``pltpu.emit_pipeline(...)(x, y)`` as strings."""
    assert isinstance(stmt, ast.Expr)
    call = stmt.value
    assert isinstance(call, ast.Call)
    return [ast.unparse(a) for a in call.args]


def _rename_and_dereference_body(
    body: list[ast.AST],
    param_renames: dict[str, str],
    pipeline_indices_name: str,
    k_pid_var: str,
) -> list[ast.AST]:
    """Rename pipeline body parameters to outer ref names and rewrite the
    ``_pipeline_indices[0]`` access to ``k_pid_var``.

    Drops the ``nonlocal`` declaration the closure body carries — the body
    is being inlined into the function scope where the names are already
    visible.
    """

    class _Renamer(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            # _pipeline_indices[<i>] → k_pid_var (single-axis lift only)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == pipeline_indices_name
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 0
            ):
                return ast.copy_location(ast.Name(id=k_pid_var, ctx=ast.Load()), node)
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in param_renames:
                new_id = param_renames[node.id]
                return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
            return node

    renamer = _Renamer()
    out: list[ast.AST] = []
    for stmt in body:
        if isinstance(stmt, ast.Nonlocal):
            continue
        new_stmt = renamer.visit(stmt)
        ast.fix_missing_locations(new_stmt)
        out.append(new_stmt)
    return out


def _wrap_in_pl_when(
    fn_name: str,
    cond_expr: str,
    body_stmts: list[ast.AST],
) -> ast.AST:
    """Build ``@pl.when(<cond_expr>); def <fn_name>(): <body_stmts>``.

    Matches the hand-written ``matmul_pallas.py`` form exactly.  Empty
    bodies get a ``pass`` so the FunctionDef stays valid.
    """
    fn_def = statement_from_string(f"def {fn_name}(): pass")
    assert isinstance(fn_def, ast.FunctionDef)
    fn_def.body = list(body_stmts) if body_stmts else [ast.Pass()]  # pyrefly: ignore[bad-assignment]
    decorator_expr = expr_from_string(f"pl.when({cond_expr})")
    assert isinstance(decorator_expr, ast.expr)
    fn_def.decorator_list = [decorator_expr]
    return fn_def


def _restore_outer_grid_tensor_specs(
    device_fn: object,
    section: list[ast.AST],
) -> None:
    """When K is lifted into the outer grid, X and Y need ordinary outer
    BlockSpecs (``(bm, bk)`` indexed by ``(i, k)`` / ``(bk, bn)`` indexed by
    ``(k, j)``) rather than the HBM ref + VMEM strip pair that
    ``_codegen_emit_pipeline`` registered.  Strip the relevant entries so
    ``backend.build_launcher_args`` emits a regular outer ``BlockSpec`` for
    each of them at host-wrapper time.

    The simple heuristic walks the rewritten section, collects names that
    appear as the outer ref in a load (``<name>[:, :]``), and clears the
    pipeline-related state for any tensor whose ``tensor_arg`` name matches.
    """
    from .._compiler.device_function import DeviceFunction
    from .._compiler.device_function import PallasMemorySpace
    from .._compiler.device_function import TensorArg

    assert isinstance(device_fn, DeviceFunction)
    # Names that appear as ``<name>[:, :]`` or ``<name>[...]`` reads in the
    # rewritten body.
    referenced_names: set[str] = set()
    for stmt in section:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                referenced_names.add(node.value.id)

    # Reset memory_space / pipeline strip for any tensor whose arg name is
    # referenced.  This makes ``_pallas_build_pipeline_specs`` emit a real
    # BlockSpec (via ``_pallas_make_block_spec``) for each one.
    for arg in device_fn._tensor_args.values():
        if not isinstance(arg, TensorArg):
            continue
        if arg.name in referenced_names:
            tid = id(arg.fake_value)
            if device_fn.pallas_memory_space.get(tid) == PallasMemorySpace.HBM:
                del device_fn.pallas_memory_space[tid]
            device_fn.pallas_pipeline_vmem_strip.discard(tid)


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

    # Decide which pipelined tensors should keep an outer VMEM-strip
    # BlockSpec ``(outer_block, full_inner)`` instead of an HBM ref.  The
    # strip lets the inner ``pltpu.emit_pipeline`` slice from VMEM, which
    # the Deep Replan pod probe showed cuts per-K HBM DMA cost ~1.9x at
    # bf16 block 128.  We opt in only when the combined strip working set
    # of the candidate tensors fits the strip budget (separate from the
    # ~65 MB vmem capacity check the launcher already enforces) so that
    # large-block configs don't accidentally OOM the TPU.
    pipeline_vmem_strip_ids = _select_pipeline_vmem_strip_ids(
        all_tensor_info,
        pipelined_tensor_ids,
        block_ids,
        _bid_to_pid_var,
        env,
        state,
    )

    def _make_block_spec(fake: torch.Tensor, subscript_meta: list[object]) -> str:
        """Build a BlockSpec string for a tensor accessed in the pipeline body.

        Encodes BOTH outer grid dims (via pl.program_id) and inner pipeline
        dims into the BlockSpec lambda, so the full HBM tensor can be passed
        without pre-slicing.

        For tensors whose outer pallas_call BlockSpec is a VMEM strip
        (id in ``pipeline_vmem_strip_ids``), the inner BlockSpec uses
        ``0`` for outer-grid dims because the outer ref is already
        pre-sliced for the outer grid coordinate — indexing with the outer
        pid would go out of bounds in that dim.
        """
        is_vmem_strip = id(fake) in pipeline_vmem_strip_ids
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

                extra_pad = _compute_pipeline_or_dma_extra_pad(
                    begin_expr, bid, env, state
                )
                _record_pad_info(state, fake, dim_idx, bid, extra_pad)
                if begin_expr == "0" and iter_step_expr == slice_size_expr:
                    lambda_parts.append(lambda_params[bid_idx])
                else:
                    lambda_parts.append(
                        f"(({begin_expr}) + ({lambda_params[bid_idx]}) * ({iter_step_expr})) // ({slice_size_expr})"
                    )
            elif bid is not None and bid in _bid_to_pid_var:
                # Outer grid dim -- select via captured program_id variable
                # (HBM-ref path) or constant 0 when the outer pallas_call
                # already pre-sliced this dim via a VMEM-strip BlockSpec.
                pid_var = _bid_to_pid_var[bid]
                bs_var = state.device_function.block_size_var(bid)
                if bs_var:
                    block_shape_parts.append(bs_var)
                else:
                    block_shape_parts.append(str(int(shape[dim_idx])))
                lambda_parts.append("0" if is_vmem_strip else pid_var)
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

    from .._compiler.device_function import PallasMemorySpace

    for fake, _tensor_node, _sub_meta in loaded_tensors.values():
        if id(fake) in pipelined_tensor_ids:
            state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
            if id(fake) in pipeline_vmem_strip_ids:
                state.device_function.pallas_pipeline_vmem_strip.add(id(fake))
    for fake, _tensor_node, _sub_meta in stored_tensors.values():
        if id(fake) in pipelined_tensor_ids:
            state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
            if id(fake) in pipeline_vmem_strip_ids:
                state.device_function.pallas_pipeline_vmem_strip.add(id(fake))

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
        in_specs.append(_make_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        pipeline_in_args.append(hbm_name)

    for fake, _tensor_node, sub_meta in stored_tensors.values():
        if id(fake) not in pipelined_tensor_ids:
            continue
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        out_tensors.append((fake, hbm_name))
        out_specs.append(_make_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        pipeline_out_args.append(hbm_name)

    # Build the body function
    body_fn_name = state.device_function.new_var("_pipeline_body")
    body_stmts: list[ast.AST] = []

    # Build block_id_to_info for the pipeline state
    block_id_to_info: dict[int, LoopDimInfo] = {}
    for block_id in block_ids:
        block_size = env.block_sizes[block_id]
        # when the block_size.size is None, we cannot form a SymPy expr for the numel
        sympy_end_expr = block_size.numel if block_size.size is not None else None
        block_id_to_info[block_id] = LoopDimInfo(
            end_var_name=None,
            end_expr=sympy_end_expr,
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

    # Emit absolute offset assignments inside the pipeline body so any
    # non-pipelined tensors (those left on their outer BlockSpec) can be
    # sliced via pl.ds against a VMEM ref whose extent is the whole
    # outer-block window.  Pipelined tensors ignore these offsets and
    # use the ``:`` full-slice inside their VMEM scratches.
    any_non_pipelined = len(pipelined_tensor_ids) < len(all_tensor_info)
    if any_non_pipelined:
        _needs_explicit_indices = True
        for i, bid in enumerate(block_ids):
            offset_name = strategy.offset_var(bid)
            body_stmts.append(
                statement_from_string(
                    f"{offset_name} = ({begin_exprs[i]}) + "
                    f"(_pipeline_indices[{i}]) * ({iter_step_exprs[i]})"
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
                parts.append(int(fake.shape[dim_idx]))
        vmem_shapes.append(tuple(parts))
    return vmem_shapes


def _select_pipeline_vmem_strip_ids(
    all_tensor_info: list[tuple[torch.Tensor, list[object], str]],
    pipelined_tensor_ids: set[int],
    block_ids: list[int],
    _bid_to_pid_var: dict[int, str],
    env: CompileEnvironment,
    state: CodegenState,
) -> set[int]:
    """Return pipelined tensor ids whose outer BlockSpec should be a VMEM strip.

    A pipelined tensor is strip-eligible when each of its dimensions is
    either tiled by the inner pipeline (block_id ∈ ``block_ids`` — kept at
    full extent in the strip) or by an outer-grid loop (block_id in
    ``_bid_to_pid_var`` — sliced to the outer block size in the strip).
    Other dims (no block_id) are kept at full extent.  The strip footprint
    is the product of those resolved sizes times ``dtype.itemsize``.

    All strip-eligible candidates opt in together when the combined
    footprint fits ``_OUTER_VMEM_STRIP_BUDGET_BYTES``; otherwise none do
    and the launcher falls back to HBM refs for every pipelined arg.

    Anything that depends on a still-symbolic shape is rejected — there is
    no safe way to estimate its strip size at codegen time, and HBM refs
    are the conservative choice.
    """
    from ..runtime import _OUTER_VMEM_STRIP_BUDGET_BYTES

    if not pipelined_tensor_ids:
        return set()

    candidate_ids: list[int] = []
    candidate_bytes: list[int] = []

    for fake, sub_meta, _direction in all_tensor_info:
        if id(fake) not in pipelined_tensor_ids:
            continue
        dim_to_bid = _get_dim_block_ids(sub_meta, env)
        strip_numel = 1
        has_outer_dim = False
        ok = True
        for dim_idx in range(len(fake.shape)):
            bid = dim_to_bid.get(dim_idx)
            if bid is not None and bid in block_ids:
                # Inner pipeline dim — kept at full extent in the strip.
                size = fake.shape[dim_idx]
                if not isinstance(size, int):
                    ok = False
                    break
                strip_numel *= size
            elif bid is not None and bid in _bid_to_pid_var:
                # Outer-grid dim — strip width = outer block size.
                resolved = state.device_function.resolved_block_size(bid)
                if not isinstance(resolved, int):
                    ok = False
                    break
                dim_size = fake.shape[dim_idx]
                if not isinstance(dim_size, int):
                    ok = False
                    break
                strip_numel *= min(resolved, dim_size)
                has_outer_dim = True
            else:
                size = fake.shape[dim_idx]
                if not isinstance(size, int):
                    ok = False
                    break
                strip_numel *= size
        if not ok:
            # Pessimise: even one unresolved candidate disables the strip
            # path for the whole pallas_call so we never partially commit
            # to a strip pattern we can't size-check.
            return set()
        if not has_outer_dim:
            # No outer-grid dim varies for this tensor; a full HBM strip
            # would just be the whole tensor in VMEM, with no per-grid-iter
            # reuse benefit.  Skip — keep the existing HBM ref for it.
            continue
        candidate_ids.append(id(fake))
        candidate_bytes.append(strip_numel * fake.element_size())

    if not candidate_ids:
        return set()
    # Double-buffered Buffered(buffer_count=2) BlockSpecs are double-VMEM
    # at runtime; budget the worst-case strip footprint accordingly.
    total = sum(candidate_bytes) * 2
    if total > _OUTER_VMEM_STRIP_BUDGET_BYTES:
        return set()
    return set(candidate_ids)


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

    * Its inner-block ``vmem_shape`` passes ``_check_dma_alignment`` -- a TPU
      DMA hardware constraint.
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
    from .atomic_ops import ATOMIC_OPS
    from .memory_ops import load as _load_op
    from .memory_ops import store as _store_op

    outer_access_targets = ATOMIC_OPS | {_load_op, _store_op}

    all_tensor_info: list[tuple[torch.Tensor, list[object], str]] = []
    for key, (fake, _tensor_node, sub_meta) in loaded_tensors.items():
        if key not in stored_tensors:
            all_tensor_info.append((fake, sub_meta, "load"))
    for fake, _tensor_node, sub_meta in stored_tensors.values():
        all_tensor_info.append((fake, sub_meta, "store"))
    vmem_shapes = _compute_vmem_shapes(
        all_tensor_info, block_ids, slice_size_exprs, env, state
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

    pipelined_ids: set[int] = set()
    for (fake, _sub_meta, _direction), vmem_shape in zip(
        all_tensor_info, vmem_shapes, strict=True
    ):
        if not _check_dma_alignment(vmem_shape):
            continue
        if id(fake) in outer_access_tensor_ids:
            continue
        pipelined_ids.add(id(fake))
    return all_tensor_info, vmem_shapes, pipelined_ids


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

    # Pipelined tensors get HBM refs (no outer BlockSpec) + VMEM scratch +
    # semaphore; the rest keep their outer BlockSpec and are accessed via
    # pl.ds() in the body.  Mixing both paths inside a single fori_loop
    # avoids forcing every tensor onto the non-DMA path when a lone
    # non-pipelined tensor is present (which would load full outer-block
    # tiles into VMEM and may OOM at large shapes).
    all_tensor_info, vmem_shapes, pipelined_tensor_ids = _classify_pipelined_tensors(
        loaded_tensors, stored_tensors, block_ids, slice_size_exprs, env, state
    )

    from .._compiler.device_function import PallasMemorySpace

    tensor_to_dma_scratch: dict[str, str] = {}
    tensor_to_sem: dict[str, str] = {}
    for (fake, _sub_meta, _direction), vmem_shape in zip(
        all_tensor_info, vmem_shapes, strict=True
    ):
        if id(fake) not in pipelined_tensor_ids:
            continue
        state.device_function.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.register_scratch(
            vmem_shape,
            fake.dtype,
            name_hint=hbm_name.replace("_hbm", "") + "_buf",
        )
        sem_name = state.device_function.register_dma_semaphore(
            name_hint=hbm_name.replace("_hbm", "") + "_sem",
        )
        tensor_to_dma_scratch[hbm_name] = vmem_name
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
        block_size = env.block_sizes[block_id]
        # when the block_size.size is None, we cannot form a SymPy expr for the numel
        sympy_end_expr = block_size.numel if block_size.size is not None else None
        block_id_to_info[block_id] = LoopDimInfo(
            end_var_name=None,
            end_expr=sympy_end_expr,
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
        inner_statements=body_stmts,
        _tensor_to_dma_scratch=tensor_to_dma_scratch,
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

                extra_pad = _compute_pipeline_or_dma_extra_pad(
                    begin_expr, bid, env, state
                )
                _record_pad_info(state, fake, dim_idx, bid, extra_pad)
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
        # Non-DMA tensors keep their outer BlockSpec (whole-shape VMEM ref)
        # and need an absolute offset for ``pl.ds()`` indexing in the body.
        # DMA copies build their own absolute slice via _build_hbm_dma_slice,
        # so this offset is dead when every tensor is DMA'd.
        if len(tensor_to_dma_scratch) < len(all_tensor_info):
            for i, bid in enumerate(block_ids):
                offset_name = strategy.offset_var(bid)
                state.codegen.add_statement(
                    statement_from_string(
                        f"{offset_name} = ({begin_exprs[i]}) + ({dim_idx_exprs[i]}) * ({iter_step_exprs[i]})"
                    )
                )

        for fake, _tensor_node, sub_meta in loaded_tensors.values():
            hbm_name = state.device_function.tensor_arg(fake).name
            if hbm_name not in tensor_to_dma_scratch:
                continue
            vmem_name = tensor_to_dma_scratch[hbm_name]
            sem_name = tensor_to_sem[hbm_name]
            src_slice = _build_hbm_dma_slice(fake, hbm_name, sub_meta)
            copy_var = state.device_function.new_var("_copy")
            state.codegen.add_statement(
                statement_from_string(
                    f"{copy_var} = pltpu.make_async_copy({src_slice}, {vmem_name}, {sem_name})"
                )
            )
            state.codegen.add_statement(statement_from_string(f"{copy_var}.start()"))
            state.codegen.add_statement(statement_from_string(f"{copy_var}.wait()"))

        graph_results = codegen_call_with_graph(
            state.codegen, graph_info.graph, body_args
        )

        if has_loop_state:
            _write_back_loop_carried(state, scratch_names, carried, graph_results)

        for fake, _tensor_node, sub_meta in stored_tensors.values():
            hbm_name = state.device_function.tensor_arg(fake).name
            if hbm_name not in tensor_to_dma_scratch:
                continue
            vmem_name = tensor_to_dma_scratch[hbm_name]
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
            state.codegen.add_statement(statement_from_string(f"{copy_out_var}.wait()"))

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

    if_return_names, else_return_names = graph_info.get_branches_return_names(
        state, if_outputs, else_outputs
    )

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

    if_return_names, else_return_names = graph_info.get_branches_return_names(
        state, if_outputs, else_outputs
    )
    return cast(
        "list[object]",
        [expr_from_string(n) for n in if_return_names]
        + [expr_from_string(n) for n in else_return_names],
    )


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
