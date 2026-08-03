"""Pallas view codegen and resident Ref composition.

Per-config planning keeps a direct VMEM load as a Ref while all of its users
are registered, address-preserving view operations. A view is materialized only
at its value boundary. Narrowing subscripts therefore lower to ``pl.ds`` on the
current Ref, including across nested control-flow captures and static reshapes.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import NoReturn
from typing import cast

import sympy
import torch

from ... import exc
from ...language import _decorators
from ...language import _tracing_ops
from ...language.memory_ops import load
from ...language.tile_ops import tile_index
from ...language.view_ops import join
from ...language.view_ops import split
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr

if TYPE_CHECKING:
    import ast

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState


_RESIDENT_PLAN_KEY = "pallas_resident_ref_plan"


class _ResidentVariant(NamedTuple):
    worklist_factor: int
    shape: tuple[int, ...]
    squeezed_dims: tuple[int, ...]
    live_guard: tuple[int, int] | None


class _ResidentSelector(NamedTuple):
    kind: str
    logical_dim: int
    local_block_id: int
    begin: int | torch.SymInt
    width: int
    mask: bool


class _ResidentPlan(NamedTuple):
    materialize: bool
    variants: tuple[_ResidentVariant, ...]
    selector: _ResidentSelector | None


class _PlanningContext(NamedTuple):
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]]
    parents: dict[int, torch.fx.Node]
    graph_infos: dict[torch.fx.Graph, GraphInfo]
    config: Config
    placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node]
    rejections: dict[torch.fx.Node, str]


def _node_value(value: object) -> object:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _narrowed_dims(indices: object) -> list[int]:
    if not isinstance(indices, (list, tuple)):
        return []
    return [
        dim
        for dim, index in enumerate(indices)
        if index is not None and index != slice(None)
    ]


def _resident_plan(node: torch.fx.Node) -> _ResidentPlan | None:
    value = node.meta.get(_RESIDENT_PLAN_KEY)
    return value if isinstance(value, _ResidentPlan) else None


def _where(node: torch.fx.Node) -> str:
    location = node.meta.get("location")
    filename = getattr(location, "filename", None)
    lineno = getattr(location, "lineno", None)
    return (
        f"{filename}:{lineno}"
        if filename is not None and lineno is not None
        else f"<{node.name}>"
    )


def _current_variant(state: CodegenState, plan: _ResidentPlan) -> _ResidentVariant:
    variants = plan.variants
    if len(variants) == 1:
        return variants[0]

    env = CompileEnvironment.current()
    worklist = env.compact_worklist_plan
    assert worklist is not None and worklist.grouping == 2
    block_id = worklist.compact_axis.block_id
    current = state.device_function.resolved_block_size(block_id)
    assert isinstance(current, int)
    factor = current // env.compact_worklist_block
    return next(variant for variant in variants if variant.worklist_factor == factor)


def maybe_materialize_resident_ref(node: torch.fx.Node, result: ast.AST) -> ast.AST:
    """Materialize a registered reshape-family Ref at its value boundary."""
    plan = _resident_plan(node)
    assert plan is not None
    if not plan.materialize:
        return result
    assert all(variant.live_guard is None for variant in plan.variants)
    return expr_from_string("{result}[...]", result=result)


def _capture_edges(
    graphs: list[GraphInfo],
) -> tuple[
    dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
    dict[int, torch.fx.Node],
    dict[torch.fx.Node, torch.fx.Node],
]:
    """Build parent and capture links on the current per-config graph copies."""
    parent_args: dict[int, tuple[torch.fx.Node, int]] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or not node.args:
                continue
            if _tracing_ops.is_for_loop_target(node.target) and isinstance(
                node.args[0], int
            ):
                parent_args[node.args[0]] = (node, 3)
            elif node.target is _tracing_ops._if and len(node.args) >= 5:
                if isinstance(node.args[1], int):
                    parent_args[node.args[1]] = (node, 3)
                if isinstance(node.args[2], int):
                    parent_args[node.args[2]] = (node, 4)

    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]] = {}
    for info in graphs:
        entry = parent_args.get(info.graph_id)
        if entry is None:
            continue
        parent, arg_index = entry
        outer_args = parent.args[arg_index]
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        if not isinstance(outer_args, (list, tuple)):
            continue
        for outer, placeholder in zip(outer_args, placeholders, strict=True):
            if isinstance(outer, torch.fx.Node):
                captures.setdefault(outer, []).append((placeholder, parent))
    placeholder_to_outer = {
        placeholder: outer
        for outer, edges in captures.items()
        for placeholder, _parent in edges
    }
    parents = {graph_id: parent for graph_id, (parent, _arg) in parent_args.items()}
    return captures, parents, placeholder_to_outer


def _resolve_node(
    node: object, placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node | None:
    while isinstance(node, torch.fx.Node):
        if node.target is _tracing_ops._new_var and node.args:
            node = node.args[0]
        elif node.op == "placeholder" and node in placeholder_to_outer:
            node = placeholder_to_outer[node]
        else:
            return node
    return None


def _mutated_storage_ids(graphs: list[GraphInfo]) -> set[int]:
    from ...language.atomic_ops import ATOMIC_OPS
    from ...language.memory_ops import store

    mutated_storages: set[int] = set()
    for info in graphs:
        for node in info.graph.nodes:
            if node.op == "call_function" and node.target in ATOMIC_OPS | {store}:
                value = _node_value(node.args[0])
                if isinstance(value, torch.Tensor):
                    mutated_storages.add(id(value.untyped_storage()))
    return mutated_storages


def _loop_bounds(
    info: GraphInfo,
    parents: dict[int, torch.fx.Node],
    block_id: int,
) -> tuple[object, object] | None:
    from ..device_ir import ForLoopGraphInfo

    entry = parents.get(info.graph_id)
    if (
        not isinstance(info, ForLoopGraphInfo)
        or entry is None
        or block_id not in info.block_ids
    ):
        return None
    parent = entry
    position = info.block_ids.index(block_id)
    begins, ends = parent.args[1:3]
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    return _node_value(begins[position]), _node_value(ends[position])


def _enclosing_loop_bounds(
    info: GraphInfo,
    parents: dict[int, torch.fx.Node],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    block_id: int,
) -> tuple[object, object] | None:
    while True:
        if (bounds := _loop_bounds(info, parents, block_id)) is not None:
            return bounds
        parent = parents.get(info.graph_id)
        outer = graph_infos.get(parent.graph) if parent is not None else None
        if outer is None:
            return None
        info = outer
    return None


def _variant_block_size(block_id: int, config: Config, factor: int) -> int | None:
    env = CompileEnvironment.current()
    value = env.block_sizes[block_id].from_config(config)
    if not isinstance(value, int):
        return None
    plan = env.compact_worklist_plan
    if (
        factor == 2
        and plan is not None
        and plan.grouping == 2
        and block_id == plan.compact_axis.block_id
    ):
        value *= 2
    return value


def _concrete_size(size: object, config: Config, factor: int) -> int | None:
    if isinstance(size, int):
        return size
    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(size)
    if block_id is not None:
        return _variant_block_size(block_id, config, factor)
    if isinstance(size, torch.SymInt):
        value = env.try_concretize_symint(size)
        return value if isinstance(value, int) else None
    return None


def _physical_shape(
    value: torch.Tensor, config: Config, factor: int
) -> tuple[int, ...] | None:
    shape = tuple(_concrete_size(size, config, factor) for size in value.shape)
    return cast("tuple[int, ...]", shape) if None not in shape else None


def _root_variants(
    producer: torch.fx.Node,
    graph_info: GraphInfo,
    context: _PlanningContext,
) -> tuple[_ResidentVariant, ...] | None:
    from .backend import SliceAddressing
    from .backend import _slice_addressing
    from .plan_tiling import IndirectGatherPattern

    tensor = _node_value(producer.args[0])
    value = producer.meta.get("val")
    indices = producer.args[1]
    if (
        not isinstance(tensor, torch.Tensor)
        or not isinstance(value, torch.Tensor)
        or tensor.ndim != value.ndim
        or producer.args[2] is not None
        or not isinstance(indices, (list, tuple))
        or any(
            isinstance(pattern, IndirectGatherPattern)
            for pattern in producer.meta.get("indexing_patterns") or ()
        )
    ):
        return None

    selected = _narrowed_dims(indices)
    if len(selected) != 1:
        return None
    dim = selected[0]
    outer_block_id = CompileEnvironment.current().resolve_block_id(
        _node_value(indices[dim])
    )
    if outer_block_id is None:
        return None

    full_loop = False
    bounds = _loop_bounds(graph_info, context.parents, outer_block_id)
    if bounds is not None:
        start, end = bounds
        env = CompileEnvironment.current()
        full_loop = (
            isinstance(start, (int, torch.SymInt))
            and isinstance(end, (int, torch.SymInt))
            and env.known_equal(start, 0)
            and env.known_equal(end, tensor.shape[dim])
        )

    worklist = CompileEnvironment.current().compact_worklist_plan
    factors = (1, 2) if worklist is not None and worklist.grouping == 2 else (1,)
    variants: list[_ResidentVariant] = []
    for factor in factors:
        shape = _physical_shape(value, context.config, factor)
        if (
            shape is None
            or _slice_addressing(value, dim, shape[-1]) is not SliceAddressing.DIRECT
        ):
            return None
        backing = _concrete_size(tensor.shape[dim], context.config, factor)
        full = full_loop and backing is not None and backing % shape[dim] == 0
        validity = None if full else (dim, outer_block_id)
        variants.append(_ResidentVariant(factor, shape, (), validity))
    return tuple(variants)


def _logical_to_physical(
    logical_dim: int, squeezed_dims: tuple[int, ...], rank: int
) -> int:
    visible = [dim for dim in range(rank) if dim not in squeezed_dims]
    assert logical_dim < len(visible), "logical dimension is outside resident Ref rank"
    return visible[logical_dim]


def _tile_run(index: torch.fx.Node) -> int | None:
    from ..indexing_strategy import subscript_tile_info

    env = CompileEnvironment.current()
    info = subscript_tile_info(env, index)
    if info is not None:
        return info.block_id if env.known_equal(info.offset, 0) else None
    if index.target is tile_index and index.args:
        return env.resolve_block_id(_node_value(index.args[0]))
    return None


def _config_error(node: torch.fx.Node, reason: str) -> NoReturn:
    raise exc.InvalidConfig(f"resident Ref view at {node.name}: {reason}")


def _selector(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    variants: tuple[_ResidentVariant, ...],
    context: _PlanningContext,
) -> tuple[tuple[_ResidentVariant, ...], _ResidentSelector] | None:
    """Validate one contiguous Ref selector and return its output state."""
    from ..device_ir import IfGraphInfo
    from ..type_info import _detect_outer_block_bound
    from ..variable_origin import TileBeginOrigin
    from .backend import SliceAddressing
    from .backend import _slice_addressing
    from .plan_tiling import _maybe_get_symbol_origin

    indices = node.args[1]
    input_node = node.args[0]
    input_value = (
        input_node.meta.get("val") if isinstance(input_node, torch.fx.Node) else None
    )
    if not isinstance(indices, (list, tuple)) or not isinstance(
        input_value, torch.Tensor
    ):
        return None
    if any(index is None for index in indices):
        return None
    selected = _narrowed_dims(indices)
    if len(selected) != 1:
        return None
    logical_dim = selected[0]
    index = indices[logical_dim]
    env = CompileEnvironment.current()

    outer_block_id = -1
    squeeze = False
    static_loop_extent = -1
    selector: _ResidentSelector

    def exact_outer_live(value: torch.SymInt) -> int | None:
        if env.get_block_id(value) is not None:
            return None
        return _detect_outer_block_bound(value, env)

    source = (
        _resolve_node(index, context.placeholder_to_outer)
        if isinstance(index, torch.fx.Node)
        else None
    )
    if source is not None and source.target is torch.ops.prims.iota.default:
        if not isinstance(graph_info, IfGraphInfo) or not source.args:
            return None
        width = _node_value(source.args[0])
        start = _node_value(source.kwargs.get("start"))
        parent = context.parents.get(graph_info.graph_id)
        predicate = _node_value(parent.args[0]) if parent is not None else None
        if (
            not isinstance(width, int)
            or width < 1
            or source.kwargs.get("step") != 1
            or not isinstance(start, torch.SymInt)
        ):
            return None
        live = start + width
        detected_outer = exact_outer_live(live)
        outer_block_id = detected_outer if detected_outer is not None else -1
        live_expr = _symint_expr(live)
        if (
            outer_block_id < 0
            or not isinstance(predicate, torch.SymBool)
            or live_expr is None
            or predicate._sympy_() != sympy.Ge(live_expr, width)
        ):
            return None
        selector = _ResidentSelector("tail", logical_dim, -1, start, width, False)
    elif source is not None:
        value = source.meta.get("val")
        squeeze = isinstance(value, torch.SymInt)
        if squeeze:
            origin = _maybe_get_symbol_origin(value)
            if origin is None or not isinstance(origin.origin, TileBeginOrigin):
                return None
            local_block_id = origin.origin.block_id
        else:
            detected_local = _tile_run(source)
            local_block_id = detected_local if detected_local is not None else -1
        bounds = _enclosing_loop_bounds(
            graph_info, context.parents, context.graph_infos, local_block_id
        )
        if bounds is None:
            return None
        start, end = bounds
        if not isinstance(start, (int, torch.SymInt)) or not env.known_equal(start, 0):
            return None
        if isinstance(end, int):
            if end < 1:
                return None
            static_loop_extent = end
        elif not isinstance(end, torch.SymInt):
            return None
        else:
            detected_outer = exact_outer_live(end)
            outer_block_id = detected_outer if detected_outer is not None else -1
            if outer_block_id < 0:
                return None
        selector = _ResidentSelector(
            "tile", logical_dim, local_block_id, -1, -1, not squeeze
        )
    elif isinstance(index, int):
        if index < 0:
            return None
        squeeze = True
        selector = _ResidentSelector("static", logical_dim, -1, index, 1, False)
    elif isinstance(index, slice):
        if (
            index.step not in (None, 1)
            or not isinstance(index.start, int)
            or not isinstance(index.stop, int)
            or index.start < 0
            or index.stop <= index.start
        ):
            return None
        selector = _ResidentSelector(
            "static", logical_dim, -1, index.start, index.stop - index.start, False
        )
    else:
        return None

    output: list[_ResidentVariant] = []
    for variant in variants:
        physical_dim = _logical_to_physical(
            logical_dim, variant.squeezed_dims, len(variant.shape)
        )
        lane_block = variant.shape[-1]
        if (
            _slice_addressing(input_value, logical_dim, lane_block)
            is not SliceAddressing.DIRECT
        ):
            return None
        width = (
            _variant_block_size(
                selector.local_block_id, context.config, variant.worklist_factor
            )
            if selector.kind == "tile"
            else selector.width
        )
        if not isinstance(width, int) or width < 1:
            _config_error(node, "the selector has no concrete positive width")
        if squeeze and width != 1:
            _config_error(node, f"a scalar selector cannot use width {width}")
        if width > variant.shape[physical_dim]:
            _config_error(
                node,
                f"the run is {width} wide but the block holds only "
                f"{variant.shape[physical_dim]} rows for this config",
            )
        if selector.kind == "static":
            assert isinstance(selector.begin, int)
            if selector.begin + width > variant.shape[physical_dim]:
                _config_error(node, "the static run reaches past the resident Ref")
            if variant.live_guard is not None and variant.live_guard[0] == physical_dim:
                return None
        elif selector.kind == "tile":
            if variant.shape[physical_dim] % width != 0:
                _config_error(node, "the selector width does not divide the Ref")
            if (
                static_loop_extent >= 0
                and static_loop_extent != variant.shape[physical_dim]
            ):
                _config_error(node, "the inner loop does not cover exactly the Ref")

        remaining_guard = (
            None
            if variant.live_guard == (physical_dim, outer_block_id)
            else variant.live_guard
        )
        new_shape = list(variant.shape)
        new_shape[physical_dim] = width
        new_squeezed_dims = (
            tuple(sorted((*variant.squeezed_dims, physical_dim)))
            if squeeze
            else variant.squeezed_dims
        )
        output.append(
            _ResidentVariant(
                variant.worklist_factor,
                tuple(new_shape),
                new_squeezed_dims,
                remaining_guard,
            )
        )

    return tuple(output), selector


def _reshape_variants(
    node: torch.fx.Node,
    config: Config,
    variants: tuple[_ResidentVariant, ...],
) -> tuple[_ResidentVariant, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype is torch.bool:
        return None
    output: list[_ResidentVariant] = []
    for variant in variants:
        new_shape = _physical_shape(value, config, variant.worklist_factor)
        if (
            variant.live_guard is not None
            or len(variant.shape) < 2
            or new_shape is None
            or prod(variant.shape) != prod(new_shape)
            or variant.shape[-2:] != new_shape[-2:]
        ):
            return None
        output.append(_ResidentVariant(variant.worklist_factor, new_shape, (), None))
    return tuple(output)


def _registered_transform(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    variants: tuple[_ResidentVariant, ...],
    context: _PlanningContext,
) -> tuple[tuple[_ResidentVariant, ...], _ResidentSelector | None] | None:
    from ..aten_lowering import AtenLowering
    from ..inductor_lowering import APIFuncLowering

    lowering = node.meta.get("lowering")
    if (
        isinstance(lowering, APIFuncLowering)
        and lowering.api_func is subscript
        and "pallas_ref" in lowering.api_func._codegen
    ):
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)):
            return None
        if not _narrowed_dims(indices):
            output = _reshape_variants(node, context.config, variants)
            return (output, None) if output is not None else None
        return _selector(node, graph_info, variants, context)
    # Registration is the opt-in contract for address-preserving Aten views.
    if isinstance(lowering, AtenLowering) and "pallas_ref" in lowering.codegen_impls:
        output = _reshape_variants(node, context.config, variants)
        return (output, None) if output is not None else None
    return None


def _effective_users(
    node: torch.fx.Node,
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
) -> list[tuple[torch.fx.Node, tuple[torch.fx.Node, ...]]]:
    results: list[tuple[torch.fx.Node, tuple[torch.fx.Node, ...]]] = []
    seen: set[torch.fx.Node] = set()
    stack: list[tuple[torch.fx.Node, tuple[torch.fx.Node, ...]]] = [(node, ())]
    while stack:
        current, transports = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        edges = captures.get(current, ())
        parent_calls = {parent for _placeholder, parent in edges}
        for user in current.users:
            if user in parent_calls or user.target == torch.ops.aten.sym_size.int:
                continue
            if user.target is _tracing_ops._new_var:
                stack.append((user, (*transports, user)))
            else:
                results.append((user, transports))
        for placeholder, _parent in edges:
            stack.append((placeholder, (*transports, placeholder)))
    return results


def _mark_rejected_descendants(
    producer: torch.fx.Node,
    context: _PlanningContext,
    reason: str,
) -> None:
    seen: set[torch.fx.Node] = set()
    stack = [producer]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.target is subscript and _narrowed_dims(node.args[1]):
            context.rejections.setdefault(node, reason)
        for user, _transports in _effective_users(node, context.captures):
            stack.append(user)


def _analyze_resident_chain(
    node: torch.fx.Node,
    node_variants: tuple[_ResidentVariant, ...],
    selector: _ResidentSelector | None,
    *,
    context: _PlanningContext,
    annotations: dict[torch.fx.Node, _ResidentPlan],
    root: bool = False,
) -> bool:
    users = _effective_users(node, context.captures)
    planned: dict[
        torch.fx.Node,
        tuple[
            tuple[torch.fx.Node, ...],
            tuple[_ResidentVariant, ...],
            _ResidentSelector | None,
        ],
    ] = {}
    unsupported = False
    for user, transports in users:
        user_info = context.graph_infos.get(user.graph)
        if user_info is None:
            unsupported = True
            continue
        result = _registered_transform(
            user,
            user_info,
            node_variants,
            context,
        )
        if result is None:
            unsupported = True
        else:
            planned[user] = (transports, *result)

    if root and (unsupported or not planned):
        if planned and unsupported:
            blockers = [
                _where(user) for user, _transports in users if user not in planned
            ]
            reason = "the block is also consumed whole at " + ", ".join(blockers[:3])
            for user in planned:
                _mark_rejected_descendants(user, context, reason)
        return False
    if not root and (
        (selector is not None and selector.mask) or unsupported or not planned
    ):
        if any(variant.live_guard is not None for variant in node_variants):
            return False
        annotations[node] = _ResidentPlan(True, node_variants, selector)
        return True

    annotations[node] = _ResidentPlan(False, node_variants, selector)
    for user, (transports, output, user_spec) in planned.items():
        transport_plan = _ResidentPlan(False, node_variants, None)
        for transport in transports:
            annotations[transport] = transport_plan
        if not _analyze_resident_chain(
            user,
            output,
            user_spec,
            context=context,
            annotations=annotations,
        ):
            return False
    return True


def plan_resident_ref_views(graphs: list[GraphInfo], config: Config) -> None:
    """Mark conservative resident Ref transform chains on config graph copies."""
    captures, parents, placeholder_to_outer = _capture_edges(graphs)
    mutated_storages = _mutated_storage_ids(graphs)
    context = _PlanningContext(
        captures,
        parents,
        {info.graph: info for info in graphs},
        config,
        placeholder_to_outer,
        {},
    )

    for info in graphs:
        for producer in info.graph.find_nodes(op="call_function", target=load):
            tensor = _node_value(producer.args[0])
            if (
                isinstance(tensor, torch.Tensor)
                and id(tensor.untyped_storage()) in mutated_storages
            ):
                _mark_rejected_descendants(
                    producer,
                    context,
                    "the tensor it reads is also written on device",
                )
                continue

            variants = _root_variants(producer, info, context)
            if variants is None:
                continue

            annotations: dict[torch.fx.Node, _ResidentPlan] = {}
            if _analyze_resident_chain(
                producer,
                variants,
                None,
                context=context,
                annotations=annotations,
                root=True,
            ):
                for node, resident_plan in annotations.items():
                    node.meta[_RESIDENT_PLAN_KEY] = resident_plan

    for info in graphs:
        for node in info.graph.find_nodes(op="call_function", target=subscript):
            if _resident_plan(node) is not None or not _narrowed_dims(node.args[1]):
                continue
            reason = context.rejections.get(node, "it could not be planned")
            raise exc.InvalidIndexingType(
                f"Pallas narrowing subscript {node.name} requires a resident Ref, "
                f"but {str(reason).rstrip('.')}."
            )


@_decorators.codegen(subscript, "pallas_ref")
def _resident_subscript(state: CodegenState) -> ast.AST:
    assert state.fx_node is not None
    plan = _resident_plan(state.fx_node)
    assert plan is not None
    variant = _current_variant(state, plan)
    if plan.selector is None:
        shape = ", ".join(str(size) for size in variant.shape)
        result = expr_from_string(
            f"{{base}}.reshape(({shape},))", base=state.ast_arg(0)
        )
        return maybe_materialize_resident_ref(state.fx_node, result)

    kind, logical_dim, local_block_id, begin, width, mask = plan.selector
    input_node = state.fx_node.args[0]
    assert isinstance(input_node, torch.fx.Node)
    input_plan = _resident_plan(input_node)
    assert input_plan is not None
    input_variant = _current_variant(state, input_plan)
    physical_dim = _logical_to_physical(
        logical_dim, input_variant.squeezed_dims, len(input_variant.shape)
    )
    if kind == "tile":
        begin_expr = state.codegen.offset_var(local_block_id)
        width_value = state.device_function.resolved_block_size(local_block_id)
        assert isinstance(width_value, int)
    elif kind == "tail":
        dynamic_begin = state.device_function.literal_expr(begin)
        begin_expr = (
            f"jnp.clip({dynamic_begin}, 0, {input_variant.shape[physical_dim] - width})"
        )
        width_value = width
    else:
        begin_expr = str(begin)
        width_value = width

    parts = [":" for _ in input_variant.shape]
    parts[physical_dim] = f"pl.ds({begin_expr}, {width_value})"
    accessor = "" if plan.materialize else ".at"
    result = expr_from_string(
        f"{{base}}{accessor}[{', '.join(parts)}]", base=state.ast_arg(0)
    )
    if not plan.materialize:
        return result

    assert variant.live_guard is None
    squeezed_dims = variant.squeezed_dims
    if squeezed_dims:
        squeeze_parts = [
            "0" if dim in squeezed_dims else ":" for dim in range(len(variant.shape))
        ]
        result = expr_from_string(
            f"{{result}}[{', '.join(squeeze_parts)}]", result=result
        )
    if mask:
        mask_var = state.codegen.mask_var(local_block_id)
        if mask_var is None:
            return result
        value = state.fx_node.meta.get("val")
        assert isinstance(value, torch.Tensor)
        expand = state.tile_strategy.expand_str([*value.shape], logical_dim)
        dtype = CompileEnvironment.current().backend.dtype_str(value.dtype)
        result = expr_from_string(
            f"{{result}} * ({mask_var}.astype({dtype}){expand})", result=result
        )
    return result


@_decorators.codegen(subscript, "pallas")
def _(state: CodegenState) -> ast.AST:
    assert state.fx_node is not None
    if _resident_plan(state.fx_node) is not None:
        # pyrefly: ignore [missing-attribute]
        return subscript._codegen["pallas_ref"](state)
    # pyrefly: ignore [missing-attribute]
    return subscript._codegen["common"](state)


@_decorators.codegen(split, "pallas")
def _(state: CodegenState) -> list[ast.AST]:
    tensor = state.ast_arg(0)
    return [
        expr_from_string("{tensor}[..., 0]", tensor=tensor),
        expr_from_string("{tensor}[..., 1]", tensor=tensor),
    ]


@_decorators.codegen(join, "pallas")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "jnp.stack(jnp.broadcast_arrays({tensor0}, {tensor1}), axis=-1)",
        tensor0=state.ast_arg(0),
        tensor1=state.ast_arg(1),
    )
