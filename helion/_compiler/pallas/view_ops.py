"""Pallas view codegen and resident Ref composition.

Per-config planning keeps a direct VMEM load as a Ref while all of its users
are registered, address-preserving view operations. A view is materialized only
at its value boundary. Narrowing subscripts therefore lower to ``pl.ds`` on the
current Ref, including across nested control-flow captures and static reshapes.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING
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


_RESIDENT_REF = "pallas_resident_ref"
_RESIDENT_REJECTION = "pallas_resident_ref_rejection"

# Factor, physical shape, hidden singleton dimensions, and outstanding
# (physical dimension, outer block id) logical-validity obligations.
_ResidentVariant = tuple[
    int, tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]
]
# kind, logical dim, local block, begin, width, mask
_ResidentSpec = tuple[str, int, int, int | torch.SymInt, int, bool]
# Mode ("ref" preserves the Ref; "read" materializes it), worklist variants,
# and the transform spec.
_ResidentInfo = tuple[str, tuple[_ResidentVariant, ...], _ResidentSpec | None]
_PlannedTransform = tuple[
    torch.fx.Node,
    tuple[torch.fx.Node, ...],
    tuple[_ResidentVariant, ...],
    _ResidentSpec | None,
    bool,
]


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


def _resident_info(node: torch.fx.Node) -> _ResidentInfo | None:
    value = node.meta.get(_RESIDENT_REF)
    return value if isinstance(value, tuple) else None


def _where(node: torch.fx.Node) -> str:
    location = node.meta.get("location")
    filename = getattr(location, "filename", None)
    lineno = getattr(location, "lineno", None)
    return (
        f"{filename}:{lineno}"
        if filename is not None and lineno is not None
        else f"<{node.name}>"
    )


def _current_variant(state: CodegenState, info: _ResidentInfo) -> _ResidentVariant:
    variants = info[1]
    if len(variants) == 1:
        return variants[0]

    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    assert plan is not None and plan.grouping == 2
    block_id = plan.compact_axis.block_id
    current = state.device_function.resolved_block_size(block_id)
    assert isinstance(current, int)
    factor = current // env.compact_worklist_block
    return next(variant for variant in variants if variant[0] == factor)


def maybe_materialize_resident_ref(node: torch.fx.Node, result: ast.AST) -> ast.AST:
    """Materialize a registered reshape-family Ref at its value boundary."""
    info = _resident_info(node)
    assert info is not None
    if info[0] == "ref":
        return result
    assert all(not variant[3] for variant in info[1])
    return expr_from_string("{result}[...]", result=result)


def _capture_edges(
    graphs: list[GraphInfo],
) -> tuple[
    dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
    dict[int, tuple[torch.fx.Node, int]],
    dict[torch.fx.Node, torch.fx.Node],
]:
    """Build parent and capture links on the current per-config graph copies."""
    parents: dict[int, tuple[torch.fx.Node, int]] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or not node.args:
                continue
            if _tracing_ops.is_for_loop_target(node.target) and isinstance(
                node.args[0], int
            ):
                parents[node.args[0]] = (node, 3)
            elif node.target is _tracing_ops._if and len(node.args) >= 5:
                if isinstance(node.args[1], int):
                    parents[node.args[1]] = (node, 3)
                if isinstance(node.args[2], int):
                    parents[node.args[2]] = (node, 4)

    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]] = {}
    for info in graphs:
        entry = parents.get(info.graph_id)
        if entry is None:
            continue
        parent, arg_index = entry
        outer_args = parent.args[arg_index] if arg_index < len(parent.args) else None
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        if not isinstance(outer_args, (list, tuple)) or len(outer_args) != len(
            placeholders
        ):
            continue
        for outer, placeholder in zip(outer_args, placeholders, strict=True):
            if isinstance(outer, torch.fx.Node):
                captures.setdefault(outer, []).append((placeholder, parent))
    placeholder_to_outer = {
        placeholder: outer
        for outer, edges in captures.items()
        for placeholder, _parent in edges
    }
    return captures, parents, placeholder_to_outer


def _resolve_node(
    node: object, placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node | None:
    seen: set[torch.fx.Node] = set()
    while isinstance(node, torch.fx.Node) and node not in seen:
        seen.add(node)
        if node.target is _tracing_ops._new_var and node.args:
            node = node.args[0]
        elif node.op == "placeholder" and node in placeholder_to_outer:
            node = placeholder_to_outer[node]
        else:
            return node
    return None


def _mutated_tensor_ids(graphs: list[GraphInfo]) -> set[int]:
    from ...language.atomic_ops import ATOMIC_OPS
    from ...language.memory_ops import store

    mutated: set[int] = set()
    for info in graphs:
        for node in info.graph.nodes:
            if node.op == "call_function" and node.target in ATOMIC_OPS | {store}:
                value = _node_value(node.args[0])
                if isinstance(value, torch.Tensor):
                    mutated.add(id(value))
    return mutated


def _loop_bounds(
    info: GraphInfo,
    parents: dict[int, tuple[torch.fx.Node, int]],
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
    parent = entry[0]
    position = info.block_ids.index(block_id)
    begins, ends = parent.args[1:3]
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    return _node_value(begins[position]), _node_value(ends[position])


def _enclosing_loop_bounds(
    info: GraphInfo,
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    block_id: int,
) -> tuple[object, object] | None:
    seen: set[int] = set()
    while id(info) not in seen:
        seen.add(id(info))
        if (bounds := _loop_bounds(info, parents, block_id)) is not None:
            return bounds
        entry = parents.get(info.graph_id)
        parent = entry[0] if entry is not None else None
        outer = graph_infos.get(parent.graph) if parent is not None else None
        if outer is None:
            return None
        info = outer
    return None


def _variant_factors() -> tuple[int, ...]:
    plan = CompileEnvironment.current().compact_worklist_plan
    return (1, 2) if plan is not None and plan.grouping == 2 else (1,)


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
    parents: dict[int, tuple[torch.fx.Node, int]],
    config: Config,
    mutated: set[int],
) -> tuple[_ResidentVariant, ...] | None:
    from .backend import SliceAddressing
    from .backend import _slice_addressing
    from .plan_tiling import IndirectGatherPattern

    tensor = _node_value(producer.args[0])
    value = producer.meta.get("val")
    indices = producer.args[1]
    if isinstance(tensor, torch.Tensor) and id(tensor) in mutated:
        producer.meta[_RESIDENT_REJECTION] = (
            "the tensor it reads is also written on device"
        )
        return None
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
    bounds = _loop_bounds(graph_info, parents, outer_block_id)
    if bounds is not None:
        start, end = bounds
        env = CompileEnvironment.current()
        full_loop = (
            isinstance(start, (int, torch.SymInt))
            and isinstance(end, (int, torch.SymInt))
            and env.known_equal(start, 0)
            and env.known_equal(end, tensor.shape[dim])
        )

    variants: list[_ResidentVariant] = []
    for factor in _variant_factors():
        shape = _physical_shape(value, config, factor)
        if (
            shape is None
            or _slice_addressing(value, dim, shape[-1]) is not SliceAddressing.DIRECT
        ):
            return None
        backing = _concrete_size(tensor.shape[dim], config, factor)
        full = full_loop and backing is not None and backing % shape[dim] == 0
        validity = () if full else ((dim, outer_block_id),)
        variants.append((factor, shape, (), validity))
    return tuple(variants)


def _logical_to_physical(logical_dim: int, hidden: tuple[int, ...], rank: int) -> int:
    logical = -1
    for physical in range(rank):
        if physical in hidden:
            continue
        logical += 1
        if logical == logical_dim:
            return physical
    raise AssertionError("logical dimension is outside resident Ref rank")


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
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    config: Config,
    variants: tuple[_ResidentVariant, ...],
    placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node],
) -> tuple[tuple[_ResidentVariant, ...], _ResidentSpec, bool] | None:
    """Validate one contiguous Ref selector and return its output state."""
    from ..device_ir import IfGraphInfo
    from ..host_function import HostFunction
    from ..type_info import _detect_outer_block_bound
    from ..variable_origin import TileBeginOrigin
    from .backend import SliceAddressing
    from .backend import _slice_addressing

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

    kind = ""
    local_block_id = -1
    outer_block_id = -1
    selector_begin: int | torch.SymInt = -1
    static_width = -1
    squeeze = False
    requires_mask = False
    static_loop_extent = -1

    def exact_outer_live(value: torch.SymInt) -> int | None:
        if env.get_block_id(value) is not None:
            return None
        return _detect_outer_block_bound(value, env)

    source = (
        _resolve_node(index, placeholder_to_outer)
        if isinstance(index, torch.fx.Node)
        else None
    )
    if source is not None and source.target is torch.ops.prims.iota.default:
        if not isinstance(graph_info, IfGraphInfo) or not source.args:
            return None
        width = _node_value(source.args[0])
        start = _node_value(source.kwargs.get("start"))
        entry = parents.get(graph_info.graph_id)
        parent = entry[0] if entry is not None else None
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
        kind = "tail"
        selector_begin = start
        static_width = width
    elif source is not None:
        value = source.meta.get("val")
        squeeze = isinstance(value, torch.SymInt)
        if squeeze:
            expr = _symint_expr(value)
            origin = (
                HostFunction.current().expr_to_origin.get(expr)
                if expr is not None
                else None
            )
            if origin is None or not isinstance(origin.origin, TileBeginOrigin):
                return None
            local_block_id = origin.origin.block_id
        else:
            detected_local = _tile_run(source)
            local_block_id = detected_local if detected_local is not None else -1
        bounds = _enclosing_loop_bounds(
            graph_info, parents, graph_infos, local_block_id
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
        kind = "tile"
        requires_mask = not squeeze
    elif isinstance(index, int):
        if index < 0:
            return None
        kind = "static"
        selector_begin = index
        static_width = 1
        squeeze = True
    elif isinstance(index, slice):
        if (
            index.step not in (None, 1)
            or not isinstance(index.start, int)
            or not isinstance(index.stop, int)
            or index.start < 0
            or index.stop <= index.start
        ):
            return None
        kind = "static"
        selector_begin = index.start
        static_width = index.stop - index.start
    else:
        return None

    output: list[_ResidentVariant] = []
    for factor, shape, hidden, validity in variants:
        physical_dim = _logical_to_physical(logical_dim, hidden, len(shape))
        lane_block = shape[-1]
        if (
            _slice_addressing(input_value, logical_dim, lane_block)
            is not SliceAddressing.DIRECT
        ):
            return None
        width = (
            _variant_block_size(local_block_id, config, factor)
            if kind == "tile"
            else static_width
        )
        if not isinstance(width, int) or width < 1:
            _config_error(node, "the selector has no concrete positive width")
        if squeeze and width != 1:
            _config_error(node, f"a scalar selector cannot use width {width}")
        if width > shape[physical_dim]:
            _config_error(
                node,
                f"the run is {width} wide but the block holds only "
                f"{shape[physical_dim]} rows for this config",
            )
        if kind == "static":
            assert isinstance(selector_begin, int)
            if selector_begin + width > shape[physical_dim]:
                _config_error(node, "the static run reaches past the resident Ref")
            if any(dim == physical_dim for dim, _block in validity):
                return None
        elif kind == "tile":
            if shape[physical_dim] % width != 0:
                _config_error(node, "the selector width does not divide the Ref")
            if static_loop_extent >= 0 and static_loop_extent != shape[physical_dim]:
                _config_error(node, "the inner loop does not cover exactly the Ref")

        remaining = tuple(
            guard
            for guard in validity
            if not (guard[0] == physical_dim and guard[1] == outer_block_id)
        )
        new_shape = list(shape)
        new_shape[physical_dim] = width
        new_hidden = tuple(sorted((*hidden, physical_dim))) if squeeze else hidden
        output.append((factor, tuple(new_shape), new_hidden, remaining))

    spec: _ResidentSpec = (
        kind,
        logical_dim,
        local_block_id,
        selector_begin,
        static_width,
        requires_mask,
    )
    return tuple(output), spec, requires_mask


def _reshape_variants(
    node: torch.fx.Node,
    config: Config,
    variants: tuple[_ResidentVariant, ...],
) -> tuple[_ResidentVariant, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype is torch.bool:
        return None
    output: list[_ResidentVariant] = []
    for factor, shape, _hidden, validity in variants:
        new_shape = _physical_shape(value, config, factor)
        if (
            validity
            or len(shape) < 2
            or new_shape is None
            or prod(shape) != prod(new_shape)
            or shape[-2:] != new_shape[-2:]
        ):
            return None
        output.append((factor, new_shape, (), ()))
    return tuple(output)


def _registered_transform(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    config: Config,
    variants: tuple[_ResidentVariant, ...],
    placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node],
) -> tuple[tuple[_ResidentVariant, ...], _ResidentSpec | None, bool] | None:
    from ..aten_lowering import AtenLowering
    from ..aten_lowering import reshape_lowering
    from ..aten_lowering import squeeze_lowering
    from ..aten_lowering import unsqueeze_lowering
    from ..aten_lowering import view_lowering
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
            output = _reshape_variants(node, config, variants)
            return (output, None, False) if output is not None else None
        return _selector(
            node,
            graph_info,
            parents,
            graph_infos,
            config,
            variants,
            placeholder_to_outer,
        )
    if isinstance(lowering, AtenLowering) and "pallas_ref" in lowering.codegen_impls:
        if not any(
            lowering is candidate
            for candidate in (
                reshape_lowering,
                squeeze_lowering,
                unsqueeze_lowering,
                view_lowering,
            )
        ):
            return None
        output = _reshape_variants(node, config, variants)
        return (output, None, False) if output is not None else None
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
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
    reason: str,
) -> None:
    seen: set[torch.fx.Node] = set()
    stack = [producer]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for user, _transports in _effective_users(node, captures):
            if user.target is subscript and _narrowed_dims(user.args[1]):
                user.meta.setdefault(_RESIDENT_REJECTION, reason)
            stack.append(user)


def _analyze_resident_chain(
    node: torch.fx.Node,
    node_variants: tuple[_ResidentVariant, ...],
    spec: _ResidentSpec | None,
    *,
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    config: Config,
    placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node],
    annotations: dict[torch.fx.Node, _ResidentInfo],
    root: bool = False,
    must_read: bool = False,
) -> bool:
    users = _effective_users(node, captures)
    planned: list[_PlannedTransform] = []
    unsupported = False
    for user, transports in users:
        user_info = graph_infos.get(user.graph)
        if user_info is None:
            unsupported = True
            continue
        result = _registered_transform(
            user,
            user_info,
            parents,
            graph_infos,
            config,
            node_variants,
            placeholder_to_outer,
        )
        if result is None:
            unsupported = True
        else:
            planned.append((user, transports, *result))

    if root and (unsupported or not planned):
        if planned and unsupported:
            blockers = [
                _where(user)
                for user, _transports in users
                if all(user is not item[0] for item in planned)
            ]
            reason = "the block is also consumed whole at " + ", ".join(blockers[:3])
            for user, _transports, *_rest in planned:
                user.meta.setdefault(_RESIDENT_REJECTION, reason)
        return False
    if not root and (must_read or unsupported or not planned):
        if any(variant[3] for variant in node_variants):
            return False
        annotations[node] = ("read", node_variants, spec)
        return True

    annotations[node] = ("ref", node_variants, spec)
    for user, transports, output, user_spec, user_must_read in planned:
        transport_info: _ResidentInfo = ("ref", node_variants, None)
        for transport in transports:
            annotations[transport] = transport_info
        if not _analyze_resident_chain(
            user,
            output,
            user_spec,
            captures=captures,
            parents=parents,
            graph_infos=graph_infos,
            config=config,
            placeholder_to_outer=placeholder_to_outer,
            annotations=annotations,
            must_read=user_must_read,
        ):
            return False
    return True


def plan_resident_ref_views(graphs: list[GraphInfo], config: Config) -> None:
    """Mark conservative resident Ref transform chains on config graph copies."""
    captures, parents, placeholder_to_outer = _capture_edges(graphs)
    graph_infos = {info.graph: info for info in graphs}
    mutated = _mutated_tensor_ids(graphs)
    for info in graphs:
        for node in info.graph.nodes:
            node.meta.pop(_RESIDENT_REF, None)
            node.meta.pop(_RESIDENT_REJECTION, None)

    for info in graphs:
        for producer in info.graph.nodes:
            if producer.op != "call_function" or producer.target is not load:
                continue
            variants = _root_variants(producer, info, parents, config, mutated)
            if variants is None:
                reason = producer.meta.get(_RESIDENT_REJECTION)
                if isinstance(reason, str):
                    _mark_rejected_descendants(producer, captures, reason)
                continue

            annotations: dict[torch.fx.Node, _ResidentInfo] = {}
            if _analyze_resident_chain(
                producer,
                variants,
                None,
                captures=captures,
                parents=parents,
                graph_infos=graph_infos,
                config=config,
                placeholder_to_outer=placeholder_to_outer,
                annotations=annotations,
                root=True,
            ):
                for node, resident_info in annotations.items():
                    node.meta[_RESIDENT_REF] = resident_info

    for info in graphs:
        for node in info.graph.nodes:
            if (
                node.op != "call_function"
                or node.target is not subscript
                or _resident_info(node) is not None
            ):
                continue
            if not _narrowed_dims(node.args[1]):
                continue
            reason = node.meta.get(_RESIDENT_REJECTION, "it could not be planned")
            raise exc.InvalidIndexingType(
                f"Pallas narrowing subscript {node.name} requires a resident Ref, "
                f"but {str(reason).rstrip('.')}."
            )


@_decorators.codegen(subscript, "pallas_ref")
def _resident_subscript(state: CodegenState) -> ast.AST:
    assert state.fx_node is not None
    info = _resident_info(state.fx_node)
    assert info is not None
    variant = _current_variant(state, info)
    if info[2] is None:
        shape = ", ".join(str(size) for size in variant[1])
        result = expr_from_string(
            f"{{base}}.reshape(({shape},))", base=state.ast_arg(0)
        )
        return maybe_materialize_resident_ref(state.fx_node, result)

    kind, logical_dim, local_block_id, begin, width, mask = info[2]
    input_node = state.fx_node.args[0]
    assert isinstance(input_node, torch.fx.Node)
    input_info = _resident_info(input_node)
    assert input_info is not None
    input_variant = _current_variant(state, input_info)
    physical_dim = _logical_to_physical(
        logical_dim, input_variant[2], len(input_variant[1])
    )
    if kind == "tile":
        begin_expr = state.codegen.offset_var(local_block_id)
        width_value = state.device_function.resolved_block_size(local_block_id)
        assert isinstance(width_value, int)
    elif kind == "tail":
        dynamic_begin = state.device_function.literal_expr(begin)
        begin_expr = (
            f"jnp.clip({dynamic_begin}, 0, {input_variant[1][physical_dim] - width})"
        )
        width_value = width
    else:
        begin_expr = str(begin)
        width_value = width

    parts = [":" for _ in input_variant[1]]
    parts[physical_dim] = f"pl.ds({begin_expr}, {width_value})"
    accessor = ".at" if info[0] == "ref" else ""
    result = expr_from_string(
        f"{{base}}{accessor}[{', '.join(parts)}]", base=state.ast_arg(0)
    )
    if info[0] == "ref":
        return result

    assert not variant[3]
    hidden = variant[2]
    if hidden:
        squeeze_parts = [
            "0" if dim in hidden else ":" for dim in range(len(variant[1]))
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
    if _resident_info(state.fx_node) is not None:
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
