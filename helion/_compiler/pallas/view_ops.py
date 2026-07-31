"""Pallas view codegen and resident Ref composition."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING
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

# Factor, physical shape, hidden singleton dimensions, and outstanding
# (physical dimension, outer block id) logical-validity obligations.
_ResidentVariant = tuple[
    int, tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]
]
# kind, logical dim, local block, static begin, width, mask
_ResidentSpec = tuple[str, int, int, int, int, bool]
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


def _resident_info(node: torch.fx.Node) -> _ResidentInfo | None:
    value = node.meta.get(_RESIDENT_REF)
    return value if isinstance(value, tuple) else None


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


def _contains(value: object, target: torch.fx.Node) -> bool:
    if value is target:
        return True
    return isinstance(value, (list, tuple)) and any(
        _contains(item, target) for item in value
    )


def _node_signature(node: torch.fx.Node) -> tuple[object, ...]:
    return (
        node.name,
        node.op,
        node.target,
        tuple(n.name for n in node.all_input_nodes),
    )


def _capture_edges(
    graphs: list[GraphInfo],
) -> tuple[
    dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
    dict[int, torch.fx.Node],
]:
    """Build read-only capture edges on the current config graph copies."""
    from ..device_ir import ElseGraphInfo
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import IfGraphInfo
    from ..host_function import HostFunction

    original_infos = {
        info.graph_id: info for info in HostFunction.current().device_ir.graphs
    }
    original_graph_ids = {info.graph: info.graph_id for info in original_infos.values()}
    current_graph_ids = {info.graph: info.graph_id for info in graphs}

    parents: dict[int, torch.fx.Node] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                _tracing_ops.is_for_loop_target(node.target)
                and node.args
                and isinstance(node.args[0], int)
            ):
                parents[node.args[0]] = node
            elif node.target is _tracing_ops._if and len(node.args) >= 3:
                if isinstance(node.args[1], int):
                    parents[node.args[1]] = node
                if isinstance(node.args[2], int):
                    parents[node.args[2]] = node

    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]] = {}
    for info in graphs:
        if not isinstance(info, (ForLoopGraphInfo, IfGraphInfo, ElseGraphInfo)):
            continue
        original = original_infos.get(info.graph_id)
        parent = parents.get(info.graph_id)
        if original is None or parent is None or type(original) is not type(info):
            continue
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        if isinstance(info, ForLoopGraphInfo):
            capture_args = parent.args[3]
            noncapture = (*parent.args[:3], *parent.args[4:])
        elif isinstance(info, IfGraphInfo):
            capture_args = parent.args[3]
            noncapture = parent.args[:3]
        else:
            capture_args = parent.args[4]
            noncapture = parent.args[:3]
        if not isinstance(capture_args, (list, tuple)) or not (
            len(original.node_args) == len(capture_args) == len(placeholders)
        ):
            continue
        for original_outer, outer, placeholder in zip(
            original.node_args, capture_args, placeholders, strict=True
        ):
            if (
                isinstance(outer, torch.fx.Node)
                and original_graph_ids.get(original_outer.graph)
                == current_graph_ids.get(outer.graph)
                and _node_signature(original_outer) == _node_signature(outer)
                and not _contains(noncapture, outer)
            ):
                captures.setdefault(outer, []).append((placeholder, parent))
    return captures, parents


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
    parent: torch.fx.Node | None,
    config: Config,
) -> tuple[_ResidentVariant, ...] | None:
    from ..device_ir import ForLoopGraphInfo
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

    selected = [dim for dim, index in enumerate(indices) if index != slice(None)]
    if len(selected) != 1:
        return None
    dim = selected[0]
    outer_block_id = CompileEnvironment.current().resolve_block_id(
        _node_value(indices[dim])
    )
    if outer_block_id is None:
        return None

    full_loop = False
    if (
        isinstance(graph_info, ForLoopGraphInfo)
        and outer_block_id in graph_info.block_ids
        and parent is not None
    ):
        position = graph_info.block_ids.index(outer_block_id)
        starts, ends = parent.args[1:3]
        if isinstance(starts, (list, tuple)) and isinstance(ends, (list, tuple)):
            start = _node_value(starts[position])
            end = _node_value(ends[position])
            start_zero = isinstance(start, (int, torch.SymInt)) and (
                CompileEnvironment.current().known_equal(start, 0)
            )
            full_loop = (
                start_zero
                and isinstance(end, (int, torch.SymInt))
                and CompileEnvironment.current().known_equal(end, tensor.shape[dim])
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


def _selector(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    parent: torch.fx.Node | None,
    config: Config,
    variants: tuple[_ResidentVariant, ...],
) -> tuple[tuple[_ResidentVariant, ...], _ResidentSpec, bool] | None:
    """Validate one contiguous Ref selector and return its output state."""
    from ..device_ir import ForLoopGraphInfo
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
    selected = [dim for dim, index in enumerate(indices) if index != slice(None)]
    if len(selected) != 1:
        return None
    logical_dim = selected[0]
    index = indices[logical_dim]
    env = CompileEnvironment.current()

    kind = ""
    local_block_id = -1
    outer_block_id = -1
    static_begin = -1
    static_width = -1
    squeeze = False
    requires_mask = False
    static_loop_extent = -1

    def exact_outer_live(value: torch.SymInt) -> int | None:
        if env.get_block_id(value) is not None:
            return None
        return _detect_outer_block_bound(value, env)

    if isinstance(graph_info, IfGraphInfo) and isinstance(index, torch.fx.Node):
        if index.target is not torch.ops.prims.iota.default or not index.args:
            return None
        width = _node_value(index.args[0])
        start = _node_value(index.kwargs.get("start"))
        predicate = _node_value(parent.args[0]) if parent is not None else None
        if (
            not isinstance(width, int)
            or width < 1
            or index.kwargs.get("step") != 1
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
        static_width = width
    elif isinstance(graph_info, ForLoopGraphInfo) and isinstance(index, torch.fx.Node):
        value = index.meta.get("val")
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
        elif index.target is tile_index and index.args:
            detected_local = env.resolve_block_id(_node_value(index.args[0]))
            local_block_id = detected_local if detected_local is not None else -1
        else:
            return None
        if local_block_id not in graph_info.block_ids or parent is None:
            return None
        position = graph_info.block_ids.index(local_block_id)
        starts, ends = parent.args[1:3]
        start = (
            _node_value(starts[position]) if isinstance(starts, (list, tuple)) else None
        )
        if (
            not isinstance(starts, (list, tuple))
            or not isinstance(ends, (list, tuple))
            or not isinstance(start, (int, torch.SymInt))
            or not env.known_equal(start, 0)
        ):
            return None
        end = _node_value(ends[position])
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
        static_begin = index
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
        static_begin = index.start
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
        if not isinstance(width, int) or width < 1 or (squeeze and width != 1):
            return None
        if width > shape[physical_dim]:
            return None
        if kind == "static":
            if static_begin + width > shape[physical_dim] or any(
                dim == physical_dim for dim, _block in validity
            ):
                return None
        elif kind == "tile":
            if shape[physical_dim] % width != 0 or (
                static_loop_extent >= 0 and static_loop_extent != shape[physical_dim]
            ):
                return None

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
        static_begin,
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
    parent: torch.fx.Node | None,
    config: Config,
    variants: tuple[_ResidentVariant, ...],
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
        return _selector(node, graph_info, parent, config, variants)
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


def discover_resident_ref_subviews(graphs: list[GraphInfo], config: Config) -> None:
    """Mark conservative resident Ref transform chains on config graph copies."""
    captures, parents = _capture_edges(graphs)
    graph_infos = {info.graph: info for info in graphs}
    for info in graphs:
        for node in info.graph.nodes:
            node.meta.pop(_RESIDENT_REF, None)

    for info in graphs:
        for producer in info.graph.nodes:
            if producer.op != "call_function" or producer.target is not load:
                continue
            variants = _root_variants(
                producer, info, parents.get(info.graph_id), config
            )
            if variants is None:
                continue

            annotations: dict[torch.fx.Node, _ResidentInfo] = {}

            def analyze(
                node: torch.fx.Node,
                node_variants: tuple[_ResidentVariant, ...],
                spec: _ResidentSpec | None,
                *,
                root: bool = False,
                must_read: bool = False,
                _annotations: dict[torch.fx.Node, _ResidentInfo] = annotations,
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
                        parents.get(user_info.graph_id),
                        config,
                        node_variants,
                    )
                    if result is None:
                        unsupported = True
                    else:
                        planned.append((user, transports, *result))

                if root and (unsupported or not planned):
                    return False
                if not root and (must_read or unsupported or not planned):
                    if any(variant[3] for variant in node_variants):
                        return False
                    _annotations[node] = ("read", node_variants, spec)
                    return True

                _annotations[node] = ("ref", node_variants, spec)
                for user, transports, output, user_spec, user_must_read in planned:
                    transport_info: _ResidentInfo = ("ref", node_variants, None)
                    for transport in transports:
                        _annotations[transport] = transport_info
                    if not analyze(
                        user,
                        output,
                        user_spec,
                        must_read=user_must_read,
                    ):
                        return False
                return True

            if analyze(producer, variants, None, root=True):
                for node, resident_info in annotations.items():
                    node.meta[_RESIDENT_REF] = resident_info


@_decorators.codegen(subscript, "pallas_ref")
def _resident_subscript(state: CodegenState) -> ast.AST:
    assert state.fx_node is not None
    info = _resident_info(state.fx_node)
    assert info is not None and info[2] is not None
    variant = _current_variant(state, info)
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
        indices = state.fx_node.args[1]
        assert isinstance(indices, (list, tuple))
        index = indices[logical_dim]
        assert isinstance(index, torch.fx.Node)
        begin_expr = state.device_function.literal_expr(
            _node_value(index.kwargs.get("start"))
        )
        width_value = width
    else:
        begin_expr = str(begin)
        width_value = width

    parts = [":" for _ in input_variant[1]]
    parts[physical_dim] = f"pl.ds({begin_expr}, {width_value})"
    result = expr_from_string(f"{{base}}.at[{', '.join(parts)}]", base=state.ast_arg(0))
    if info[0] == "ref":
        return result

    assert not variant[3]
    result = expr_from_string("{result}[...]", result=result)
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
            raise exc.InvalidConfig("fixed resident Ref subview requires a loop mask")
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
