from __future__ import annotations

import ast
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
    pallas_loop_type = config.get("pallas_loop_type", "default")
    if pallas_loop_type == "emit_pipeline":
        return _codegen_emit_pipeline(state)
    if pallas_loop_type == "fori_loop":
        return _codegen_fori_loop(state)
    # default: fall through to common codegen path
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


@_decorators.codegen(_for_loop_step, "pallas")
def _(state: CodegenState) -> None:
    """Emit inner stepped device loops for Pallas/TPU."""
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "default")
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

    # Record which tensors are in the pipeline body (need HBM refs)
    for fake, _tensor_node, _sub_meta in loaded_tensors.values():
        state.device_function.pallas_pipeline_tensor_ids.add(id(fake))
    for fake, _tensor_node, _sub_meta in stored_tensors.values():
        state.device_function.pallas_pipeline_tensor_ids.add(id(fake))

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
    # Set up mask variables for inner-loop block_ids.
    _needs_explicit_indices = _setup_inner_loop_masks(
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
    if _needs_explicit_indices:
        # emit_pipeline passes indices as a single tuple argument
        fn_args = "_pipeline_indices, " + ", ".join(all_body_params)
    else:
        fn_args = ", ".join(all_body_params)
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
    if _needs_explicit_indices:
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


def _check_dma_alignment(vmem_shapes: list[tuple[int, ...]]) -> bool:
    """Check if all VMEM buffer shapes satisfy TPU DMA alignment.

    DMA requires last dim % 128 == 0 and second-to-last dim % 8 == 0
    for 2D+ tensors. Note that these rules are currently optimized for
    bf16 sublanes; they are overly conservative for f32 (no constraint)
    and too lenient for 1D (which should be % 1024).

    These rules differ from outer BlockSpec constraints where 1D is
    dtype-dependent: 128 * (32 / bitwidth(dtype)). Unlike outer BlockSpecs,
    emit_pipeline/fori_loop inner DMA does NOT have a ``block == tensor_dim``
    exception.
    """
    for shape in vmem_shapes:
        if len(shape) >= 2:
            if shape[-1] % 128 != 0:
                return False
            if shape[-2] % 8 != 0:
                return False
        elif len(shape) == 1:
            if shape[0] % 128 != 0:
                return False
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
    use_dma = _check_dma_alignment(vmem_shapes)

    # With DMA: tensors need HBM refs (no outer BlockSpec) because DMA
    # handles all slicing, and we register VMEM scratch buffers +
    # semaphores for each tensor.  Without DMA: outer BlockSpecs handle
    # outer dims and inner dims use pl.ds(), so neither is needed.
    tensor_to_vmem: dict[str, str] = {}
    tensor_to_sem: dict[str, str] = {}
    if use_dma:
        for fake, _tensor_node, _sub_meta in loaded_tensors.values():
            state.device_function.pallas_pipeline_tensor_ids.add(id(fake))
        for fake, _tensor_node, _sub_meta in stored_tensors.values():
            state.device_function.pallas_pipeline_tensor_ids.add(id(fake))
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

    # Set up mask variables for inner-loop block_ids
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
        else:
            # No DMA: emit offset assignments so pl.ds() indexing works
            for i, bid in enumerate(block_ids):
                offset_name = strategy.offset_var(bid)
                state.codegen.add_statement(
                    statement_from_string(
                        f"{offset_name} = ({begin_exprs[i]}) + ({dim_idx_exprs[i]}) * ({iter_step_exprs[i]})"
                    )
                )

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

    return [expr_from_string(n) for n in if_return_names] + [
        expr_from_string(n) for n in else_return_names
    ]


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
