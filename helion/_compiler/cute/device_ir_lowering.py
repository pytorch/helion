"""CuTe-specific device IR lowering pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from ..device_ir_lowering import DeviceIRLowering
from ..device_ir_lowering import register_load_store_tunables

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


class CuteDeviceIRLowering(DeviceIRLowering):
    """CuTe-specific overrides for the device IR lowering pipeline.

    Adds half-precision atomic promotion in ``transform()`` and
    CuTe-specific tunable registration (vector widths, indexed
    reductions) in ``register()``.
    """

    def transform(self, device_ir: DeviceIR, func: HostFunction) -> None:
        super().transform(device_ir, func)
        _promote_cute_half_atomic_outputs(device_ir)

    def register(self, device_ir: DeviceIR) -> None:
        super().register(device_ir)
        register_load_store_tunables(device_ir)
        _register_cute_vector_widths(device_ir)
        _track_cute_indexed_reductions(device_ir)


def _promote_cute_half_atomic_outputs(device_ir: DeviceIR) -> None:
    """Promote float16 atomic output tensors to float32.

    CuTe's SMEM atomics don't support float16, so we promote the
    allocation to float32 and cast back on return.  Must run before
    ``prepare_graph_lowerings`` reads ``node.meta["val"].dtype``.
    """
    from ..host_function import HostFunction

    promotions = collect_cute_half_atomic_output_promotions(device_ir.graphs)
    if promotions:
        host_fn = HostFunction.current()
        rewrite_cute_half_atomic_output_allocations(host_fn, promotions)
        promote_cute_root_graph_host_tensors(device_ir.graphs, promotions)


def collect_cute_half_atomic_output_promotions(
    graph_infos: list[GraphInfo],
) -> dict[str, torch.dtype]:
    from ...language import atomic_add
    from ...language._tracing_ops import _host_tensor
    from ..host_function import HostFunction
    from ..variable_origin import ArgumentOrigin

    promotions: dict[str, torch.dtype] = {}
    host_fn = HostFunction.current()
    host_tensor_nodes: dict[str, list[torch.fx.Node]] = {}

    for graph_info in graph_infos:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target is _host_tensor:
                target_name = node.args[0]
                if isinstance(target_name, str):
                    host_tensor_nodes.setdefault(target_name, []).append(node)

    def is_promotable_target(node: torch.fx.Node) -> bool:
        target_val = node.meta.get("val")
        if (
            not isinstance(target_val, torch.Tensor)
            or target_val.dtype != torch.float16
        ):
            return False
        origin = host_fn.tensor_to_origin.get(target_val)
        if origin is None or isinstance(origin, ArgumentOrigin):
            return False
        if not node.users:
            return False
        for user in node.users:
            if user.op != "call_function" or user.target is not atomic_add:
                return False
            if len(user.args) < 3 or user.args[0] is not node or len(user.users) != 0:
                return False
            value_node = user.args[2]
            if not isinstance(value_node, torch.fx.Node):
                return False
            value_val = value_node.meta.get("val")
            if not isinstance(value_val, torch.Tensor) or value_val.dtype not in (
                torch.float16,
                torch.float32,
            ):
                return False
        return True

    for target_name, nodes in host_tensor_nodes.items():
        if all(is_promotable_target(node) for node in nodes):
            promotions[target_name] = torch.float16

    return promotions


def rewrite_cute_half_atomic_output_allocations(
    host_fn: HostFunction,
    promotions: dict[str, torch.dtype],
) -> None:
    import ast

    from ..ast_extension import create
    from ..ast_extension import expr_from_string

    torch_factory_names = {
        "empty",
        "empty_like",
        "full",
        "full_like",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }

    def dtype_expr(dtype: torch.dtype) -> ast.expr:
        expr = expr_from_string(f"torch.{str(dtype).split('.', 1)[1]}")
        assert isinstance(expr, ast.expr)
        return expr

    def is_torch_factory_call(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in torch_factory_names
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
        )

    def rewrite_allocation_dtype(call: ast.Call) -> None:
        dtype = dtype_expr(torch.float32)
        for kwarg in call.keywords:
            if kwarg.arg == "dtype":
                kwarg.value = dtype
                return
        if is_torch_factory_call(call):
            call.keywords.append(create(ast.keyword, arg="dtype", value=dtype))

    def rewrite_return_expr(expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.Name) and expr.id in promotions:
            cast_expr = expr_from_string(
                "{value}.to({dtype})",
                value=expr,
                dtype=dtype_expr(promotions[expr.id]),
            )
            assert isinstance(cast_expr, ast.expr)
            return cast_expr
        if isinstance(expr, ast.Tuple):
            return create(
                ast.Tuple,
                elts=[rewrite_return_expr(elt) for elt in expr.elts],
                ctx=expr.ctx,
            )
        if isinstance(expr, ast.List):
            return create(
                ast.List,
                elts=[rewrite_return_expr(elt) for elt in expr.elts],
                ctx=expr.ctx,
            )
        return expr

    for stmt in ast.walk(ast.Module(body=host_fn.body, type_ignores=[])):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in promotions
            and isinstance(stmt.value, ast.Call)
        ):
            rewrite_allocation_dtype(stmt.value)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            stmt.value = rewrite_return_expr(stmt.value)


def promote_cute_root_graph_host_tensors(
    graph_infos: list[GraphInfo],
    promotions: dict[str, torch.dtype],
) -> None:
    from ...language._tracing_ops import _host_tensor
    from ..host_function import HostFunction

    host_fn = HostFunction.current()
    for graph_info in graph_infos:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not _host_tensor:
                continue
            target_name = node.args[0]
            if not isinstance(target_name, str) or target_name not in promotions:
                continue
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                promoted_value = value.to(dtype=torch.float32)
                if origin := host_fn.tensor_to_origin.get(value):
                    host_fn.tensor_to_origin[promoted_value] = origin
                node.meta["val"] = promoted_value


def _register_cute_vector_widths(device_ir: DeviceIR) -> None:
    """Register CuteVectorWidthSpec entries for tile and reduction blocks.

    For CuTe kernels without reduction dims, registers vector width
    specs for tile blocks (enables vec-aware lane loops in
    ``CuteNDTileStrategy``).  When reduction dims are present,
    registers vector width specs for each rollable reduction dim.

    The reduction-dim slot must stay at index 0 of
    ``cute_vector_widths`` to match the ``CuteReductionTileHeuristic``
    seed and user-facing API.
    """
    from ...autotuner.config_spec import CuteVectorWidthSpec

    env = CompileEnvironment.current()
    rdims = [bs for bs in env.block_sizes if bs.reduction]

    if not rdims:
        # Register vector widths for non-reduction tile blocks
        already_registered = set(env.config_spec.cute_vector_widths.valid_block_ids())
        tile_blocks = [bs for bs in env.block_sizes if not bs.reduction]
        for tile_bs in tile_blocks:
            if tile_bs.block_id in already_registered:
                continue
            if not isinstance(tile_bs.size, (int, torch.SymInt)):
                continue
            try:
                size_hint_val = int(tile_bs.size_hint())
            except (TypeError, ValueError, AttributeError, AssertionError):
                continue
            env.config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(
                    block_id=tile_bs.block_id,
                    size_hint=size_hint_val,
                )
            )
    else:
        # Register vector widths for each rollable reduction dim.
        # register_rollable_reductions() already ran via super().register()
        # and populated config_spec.reduction_loops.  We create matching
        # CuteVectorWidthSpec entries.
        for spec in env.config_spec.reduction_loops:
            env.config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(
                    block_id=spec.block_id,
                    size_hint=spec.size_hint,
                )
            )


def _track_cute_indexed_reductions(device_ir: DeviceIR) -> None:
    """Track which reduction dims are used for argmin/argmax.

    CuTe can only combine these via ``cute.arch.warp_reduction``
    (32 threads max), so the autotuner must keep their persistent
    thread count and looped chunk size within a single warp.
    """
    env = CompileEnvironment.current()
    rdims = [bs for bs in env.block_sizes if bs.reduction]
    if not rdims:
        return

    num_original_graphs = len(device_ir.graphs)
    # Roller analysis may have added sub-graphs; scan only originals.
    # Count original graphs by finding the first ReductionLoopGraphInfo.
    from ..device_ir import ReductionLoopGraphInfo

    for i, g in enumerate(device_ir.graphs):
        if isinstance(g, ReductionLoopGraphInfo):
            num_original_graphs = i
            break

    indexed_blocks: set[int] = set()
    indexed_targets = {
        torch.ops.aten.argmin.default,
        torch.ops.aten.argmax.default,
    }
    for graph_info in device_ir.graphs[:num_original_graphs]:
        for node in graph_info.graph.nodes:
            if getattr(node, "target", None) not in indexed_targets:
                continue
            args = node.args or ()
            if not args:
                continue
            val = getattr(args[0], "meta", {}).get("val")
            if val is None:
                continue
            dim_arg = args[1] if len(args) >= 2 else -1
            dim_indices = (
                [int(cast("int", d)) for d in dim_arg]
                if isinstance(dim_arg, list)
                else [int(cast("int", dim_arg))]
            )
            for dim_idx in dim_indices:
                if dim_idx < 0:
                    dim_idx += val.ndim
                if 0 <= dim_idx < val.ndim:
                    reduce_dim = val.size(dim_idx)
                    block_id = env.resolve_block_id(reduce_dim)
                    if block_id is not None:
                        indexed_blocks.add(block_id)
    env.config_spec.cute_indexed_reduction_block_ids = indexed_blocks
