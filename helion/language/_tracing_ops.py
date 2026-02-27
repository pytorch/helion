from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeVar

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
from .._compiler.host_function import HostFunction
from .._compiler.variable_origin import BlockSizeOrigin
from ..exc import BackendUnsupported
from ..exc import NotInsideKernel
from . import _decorators
from .tile_proxy import Tile

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T", bound=object)

"""
This file contains "fake" ops that cannot appear in user program but
are generated while compiling the user program. These ops are used to
generate code for certain constructs.
"""

_symbolic_types = (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)


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
    graph_id: int, begin: list[int], end: list[int], args: list[object]
) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-index]
    return HostFunction.current().device_ir.graphs[state.proxy_arg(0)].codegen(state)


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
def _(state: CodegenState) -> None:
    """Emit inner device loops for Pallas/TPU.

    When ``pallas_loop_type="emit_pipeline"``, generates ``pltpu.emit_pipeline``
    calls with automatic DMA pipelining.  When ``pallas_loop_type="fori_loop"``,
    generates ``jax.lax.fori_loop`` with explicit ``pltpu.make_async_copy`` DMA.
    Otherwise falls through to the common ``ForLoopGraphInfo.codegen`` path.
    """
    config = state.config
    pallas_loop_type = config.get("pallas_loop_type", "default")
    if pallas_loop_type == "emit_pipeline":
        _codegen_emit_pipeline(state)
        return
    if pallas_loop_type == "fori_loop":
        _codegen_fori_loop(state)
        return
    # default: fall through to common codegen path
    # pyrefly: ignore [bad-index]
    HostFunction.current().device_ir.graphs[state.proxy_arg(0)].codegen(state)


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
) -> object:
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


def _codegen_emit_pipeline(state: CodegenState) -> None:
    """Emit inner device loops using pltpu.emit_pipeline."""
    from .._compiler.device_ir import ForLoopGraphInfo
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.inductor_lowering import codegen_call_with_graph
    from .._compiler.tile_strategy import EmitPipelineLoopState
    from .._compiler.tile_strategy import LoopDimInfo

    # pyrefly: ignore [bad-index]
    graph_info = HostFunction.current().device_ir.graphs[state.proxy_arg(0)]
    assert isinstance(graph_info, ForLoopGraphInfo)
    assert isinstance(state.codegen, GenerateAST)

    block_ids = graph_info.block_ids
    env = CompileEnvironment.current()

    args = state.ast_args[-1]
    assert isinstance(args, list)
    assert all(isinstance(x, ast.AST) for x in args)

    grid_parts, block_size_vars = _compute_grid_and_block_sizes(state, block_ids, env)

    loaded_tensors, stored_tensors = _classify_loop_tensors(graph_info, state)

    # Build in_specs and out_specs
    in_tensors: list[tuple[torch.Tensor, str]] = []
    out_tensors: list[tuple[torch.Tensor, str]] = []
    in_specs: list[str] = []
    out_specs: list[str] = []
    body_params: list[str] = []
    pipeline_in_args: list[str] = []
    pipeline_out_args: list[str] = []

    def _make_block_spec(fake: torch.Tensor, subscript_meta: list[object]) -> str:
        """Build a BlockSpec string for a tensor accessed in the pipeline body."""
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
                bid_idx = block_ids.index(bid)
                bs_var = block_size_vars[bid_idx]
                block_shape_parts.append(bs_var)
                lambda_parts.append(lambda_params[bid_idx])
            elif bid is not None:
                bs_var = state.device_function.block_size_var(bid)
                if bs_var:
                    block_shape_parts.append(bs_var)
                else:
                    block_shape_parts.append(str(int(shape[dim_idx])))
                lambda_parts.append("0")
            else:
                block_shape_parts.append(str(int(shape[dim_idx])))
                lambda_parts.append("0")

        block_shape_str = ", ".join(block_shape_parts)
        lambda_body = ", ".join(lambda_parts)
        lambda_param_str = ", ".join(lambda_params)
        return f"pl.BlockSpec(({block_shape_str},), lambda {lambda_param_str}: ({lambda_body},))"

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
        hbm_slice = _make_hbm_slice(fake, hbm_name, sub_meta)
        pipeline_in_args.append(hbm_slice)

    # Process stored tensors (outputs of pipeline, may also be read)
    for fake, _tensor_node, sub_meta in stored_tensors.values():
        hbm_name = state.device_function.tensor_arg(fake).name
        vmem_name = state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        out_tensors.append((fake, hbm_name))
        out_specs.append(_make_block_spec(fake, sub_meta))
        body_params.append(vmem_name)
        hbm_slice = _make_hbm_slice(fake, hbm_name, sub_meta)
        pipeline_out_args.append(hbm_slice)

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

    # Generate body code within the pipeline context
    with state.codegen.add_emit_pipeline_loop(pipeline_state):
        codegen_call_with_graph(state.codegen, graph_info.graph, [*args])

    # Build the function def for the body
    fn_args = ", ".join(body_params)
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


def _codegen_fori_loop(state: CodegenState) -> None:
    """Emit inner device loops using jax.lax.fori_loop + pltpu.make_async_copy."""
    from .._compiler.device_ir import ForLoopGraphInfo
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.inductor_lowering import codegen_call_with_graph
    from .._compiler.tile_strategy import ForiLoopState
    from .._compiler.tile_strategy import LoopDimInfo

    # pyrefly: ignore [bad-index]
    graph_info = HostFunction.current().device_ir.graphs[state.proxy_arg(0)]
    assert isinstance(graph_info, ForLoopGraphInfo)
    assert isinstance(state.codegen, GenerateAST)

    block_ids = graph_info.block_ids
    env = CompileEnvironment.current()

    args = state.ast_args[-1]
    assert isinstance(args, list)
    assert all(isinstance(x, ast.AST) for x in args)

    grid_parts, block_size_vars = _compute_grid_and_block_sizes(state, block_ids, env)

    loaded_tensors, stored_tensors = _classify_loop_tensors(graph_info, state)

    # For each tensor, register VMEM scratch buffer + DMA semaphore
    tensor_to_vmem: dict[str, str] = {}
    tensor_to_sem: dict[str, str] = {}

    # Collect all tensors: load-only first, then stored (which may also be read)
    all_tensor_info: list[tuple[torch.Tensor, list[object], str]] = []
    for key, (fake, _tensor_node, sub_meta) in loaded_tensors.items():
        if key not in stored_tensors:
            all_tensor_info.append((fake, sub_meta, "load"))
    for fake, _tensor_node, sub_meta in stored_tensors.values():
        all_tensor_info.append((fake, sub_meta, "store"))

    for fake, sub_meta, _direction in all_tensor_info:
        hbm_name = state.device_function.tensor_arg(fake).name
        # Compute VMEM buffer shape (block-sized for pipeline dims, full for others)
        dim_to_bid = _get_dim_block_ids(sub_meta, env)
        vmem_shape_parts: list[int] = []
        for dim_idx in range(len(fake.shape)):
            bid = dim_to_bid.get(dim_idx)
            if bid is not None and bid in block_ids:
                bid_idx = block_ids.index(bid)
                block_value = env.block_sizes[block_ids[bid_idx]].from_config(
                    state.config
                )
                assert isinstance(block_value, int), (
                    f"Block size for block_id {bid} must be a concrete int"
                )
                vmem_shape_parts.append(block_value)
            elif bid is not None:
                outer_block_value = env.block_sizes[bid].from_config(state.config)
                if isinstance(outer_block_value, int):
                    vmem_shape_parts.append(outer_block_value)
                else:
                    vmem_shape_parts.append(int(fake.shape[dim_idx]))
            else:
                vmem_shape_parts.append(int(fake.shape[dim_idx]))

        vmem_name = state.device_function.register_scratch(
            tuple(vmem_shape_parts),
            fake.dtype,
            name_hint=hbm_name.replace("_hbm", "") + "_buf",
        )
        sem_name = state.device_function.register_dma_semaphore(
            name_hint=hbm_name.replace("_hbm", "") + "_sem",
        )
        tensor_to_vmem[hbm_name] = vmem_name
        tensor_to_sem[hbm_name] = sem_name

    # Build the body function
    body_fn_name = state.device_function.new_var("_fori_body")
    loop_var = state.device_function.new_var("_j")
    body_stmts: list[ast.AST] = []

    # Build block_id_to_info
    block_id_to_info: dict[int, LoopDimInfo] = {}
    for block_id in block_ids:
        block_id_to_info[block_id] = LoopDimInfo(
            end_var_name=None,
            end_expr=env.block_sizes[block_id].numel,
        )

    strategy = _find_strategy(state, block_ids)

    # Create ForiLoopState
    fori_state = ForiLoopState(
        strategy=strategy,  # pyrefly: ignore[bad-argument-type]
        block_id_to_info=block_id_to_info,
        body_fn_name=body_fn_name,
        loop_var_name=loop_var,
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
                # Pipeline dim: use loop_var * block_size
                bid_idx = block_ids.index(bid)
                bs_var = block_size_vars[bid_idx]
                parts.append(f"pl.ds({loop_var} * {bs_var}, {bs_var})")
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

    # Generate body code within the fori_loop context
    with state.codegen.add_fori_loop(fori_state):
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
            state.codegen.add_statement(statement_from_string(f"{copy_var}.start()"))
            state.codegen.add_statement(statement_from_string(f"{copy_var}.wait()"))

        # Codegen the user's body (loads/stores remapped via _tensor_to_vmem)
        codegen_call_with_graph(state.codegen, graph_info.graph, [*args])

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
            state.codegen.add_statement(statement_from_string(f"{copy_out_var}.wait()"))

    # Emit the function def and fori_loop call
    fn_def = statement_from_string(f"def {body_fn_name}({loop_var}, _): pass")
    assert isinstance(fn_def, ast.FunctionDef)
    fn_def.body = body_stmts or [ast.Pass()]  # pyrefly: ignore[bad-assignment]

    # Compute n_tiles
    if len(grid_parts) == 1:
        n_tiles_expr = grid_parts[0]
    else:
        n_tiles_expr = " * ".join(f"({p})" for p in grid_parts)

    state.add_statement(fn_def)
    state.add_statement(
        statement_from_string(
            f"jax.lax.fori_loop(0, {n_tiles_expr}, {body_fn_name}, None)"
        )
    )


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
    # pyrefly: ignore [bad-index]
    return HostFunction.current().device_ir.graphs[state.proxy_arg(1)].codegen(state)


@has_side_effect
@_decorators.api()
def _if(test: object, graph_id: int, args: list[object]) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_if, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-index]
    return HostFunction.current().device_ir.graphs[state.proxy_arg(1)].codegen(state)


@_decorators.codegen(_if, "pallas")
def _(state: CodegenState) -> None:
    """Emit dynamic if-conditions for Pallas/TPU using ``lax.cond``.

    JAX's tracing model does not support Python ``if`` on traced values.
    We use ``lax.cond(pred, true_fn, false_fn)`` which requires a scalar
    predicate. Tensor-derived predicates (from tensor loads) are unsupported
    because TPU block shapes make them vectors at runtime.
    """
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.device_ir import IfGraphInfo
    from .._compiler.inductor_lowering import codegen_call_with_graph

    # pyrefly: ignore[bad-index]
    graph_info = HostFunction.current().device_ir.graphs[state.proxy_arg(1)]
    assert isinstance(graph_info, IfGraphInfo)

    test = state.ast_arg(0)
    args = state.ast_args[2]
    assert isinstance(args, list)
    assert all(isinstance(x, ast.AST) for x in args)

    from .._compiler.generate_ast import GenerateAST

    assert isinstance(state.codegen, GenerateAST)

    assert state.fx_node is not None
    if state.fx_node.meta.get("predicate_is_tensor", False):
        raise BackendUnsupported(
            "pallas",
            "if-statements with tensor-derived predicates. "
            "lax.cond requires a scalar predicate, but tensor loads produce "
            "vectors on TPU due to hardware tiling constraints. "
            "Use a scalar kernel argument for the condition instead.",
        )

    branch_fn_name = state.device_function.new_var("_cond_branch")

    body_stmts: list[ast.AST] = []
    with state.codegen.set_statements(body_stmts):
        codegen_call_with_graph(state.codegen, graph_info.graph, [*args])

    fn_def = statement_from_string(f"def {branch_fn_name}(): pass")
    assert isinstance(fn_def, ast.FunctionDef)
    fn_def.body = body_stmts or [ast.Pass()]  # pyrefly: ignore[bad-assignment]
    state.add_statement(fn_def)

    state.add_statement(
        statement_from_string(
            f"lax.cond({{test}}, {branch_fn_name}, lambda: None)",
            test=test,
        )
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
