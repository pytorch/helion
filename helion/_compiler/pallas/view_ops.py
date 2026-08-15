"""Pallas view codegen and resident Ref composition.

Mosaic cannot dynamically slice a materialized VMEM value, so a narrowing
subscript must instead read from the ``Ref`` produced by its outer tile load.
The per-config planner starts at eligible direct loads and walks forward through
an explicit set of address-preserving views. ``_new_var`` and control-flow
captures transport the same Ref without changing its address interpretation.

Planning is transactional: annotations are committed only after the complete
reachable chain is valid. An unsupported value consumer normally terminates a
chain by materializing the current view, but it is an error when another
narrowing subscript occurs beyond that boundary. Failures are recorded where
planning stops and reported as ``BackendUnsupported`` only by the final
completeness check. Pallas autotuning skips that config while preserving the
specific diagnostic. Codegen consumes committed plans; it does not decide
whether a view is eligible.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING
from typing import NamedTuple
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


RESIDENT_PLAN_META = "pallas_resident_ref_plan"
STATIC_BASIC_VALUE_SUBSCRIPT_META = "pallas_static_basic_value_subscript"

# These Aten operations may preserve the address interpretation of a resident
# Ref. Membership only permits planning; ``_reshape_variants`` still proves the
# physical shape/layout invariant for every config. Keep this list explicit so
# adding ordinary Pallas codegen cannot accidentally opt an operation into Ref
# composition.
_RESIDENT_REF_ATEN_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    }
)


class _ResidentVariant(NamedTuple):
    """Physical Ref state for one compact-worklist branch."""

    worklist_factor: int
    shape: tuple[int, ...]
    squeezed_dims: tuple[int, ...]
    # (physical dimension, source block id) while that dimension may contain tail
    # padding. Selecting it through the matching live extent discharges the guard.
    live_guard: tuple[int, int] | None


class _ResidentSelector(NamedTuple):
    """A selector proven to be one contiguous, directly addressable Ref slice.

    Tile selectors derive their begin/width from ``local_block_id``; tail and
    static selectors carry concrete ``begin`` and ``width`` values.
    """

    kind: str
    logical_dim: int
    local_block_id: int
    begin: int | torch.SymInt
    width: int
    mask: bool


class _ResidentPlan(NamedTuple):
    """Planner result consumed by load/view/subscript codegen."""

    materialize: bool
    variants: tuple[_ResidentVariant, ...]
    selector: _ResidentSelector | None


class _ResidentTransform(NamedTuple):
    variants: tuple[_ResidentVariant, ...]
    selector: _ResidentSelector | None


class _PlanningContext(NamedTuple):
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]]
    parents: dict[int, torch.fx.Node]
    graph_infos: dict[torch.fx.Graph, GraphInfo]
    config: Config
    placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node]
    failures: dict[torch.fx.Node, exc.BackendUnsupported]


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


class _StaticIndexRange(NamedTuple):
    """A compile-time contiguous tensor index represented as a Python slice."""

    start: int
    length: int


def _static_index(
    value: object, seen: set[torch.fx.Node] | None = None
) -> int | _StaticIndexRange | None:
    """Evaluate a scalar or contiguous compile-time tensor index.

    Mosaic does not implement general advanced indexing on materialized VMEM
    values. Helion's ``hl.arange`` commonly traces as an iota, however, and an
    iota shifted by a constant is exactly a basic Python slice. Recognize that
    deliberately small subset so Pallas codegen can retain basic indexing
    semantics without introducing a gather.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, torch.fx.Node):
        return None
    seen = set() if seen is None else seen
    if value in seen:
        return None
    seen.add(value)
    if value.op != "call_function":
        return None
    if value.target is torch.ops.prims.iota.default:
        length = value.args[0]
        start = value.kwargs.get("start", 0)
        step = value.kwargs.get("step", 1)
        if not isinstance(length, int) or isinstance(length, bool):
            return None
        if not isinstance(start, int) or isinstance(start, bool):
            return None
        if not isinstance(step, int) or isinstance(step, bool):
            return None
        if length < 0 or start < 0 or step != 1:
            return None
        return _StaticIndexRange(start=start, length=length)

    shift_arithmetic = {
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
    }
    if value.target not in shift_arithmetic or value.kwargs.get("alpha", 1) != 1:
        return None
    if len(value.args) != 2:
        return None
    lhs = _static_index(value.args[0], seen)
    rhs = _static_index(value.args[1], seen)
    if value.target in {torch.ops.aten.add.Scalar, torch.ops.aten.add.Tensor}:
        if isinstance(lhs, _StaticIndexRange) and isinstance(rhs, int):
            start = lhs.start + rhs
            return _StaticIndexRange(start, lhs.length) if start >= 0 else None
        if isinstance(lhs, int) and isinstance(rhs, _StaticIndexRange):
            start = lhs + rhs.start
            return _StaticIndexRange(start, rhs.length) if start >= 0 else None
        if isinstance(lhs, int) and isinstance(rhs, int):
            return lhs + rhs
        return None
    if isinstance(lhs, _StaticIndexRange) and isinstance(rhs, int):
        start = lhs.start - rhs
        return _StaticIndexRange(start, lhs.length) if start >= 0 else None
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs - rhs
    return None


def _is_static_basic_value_subscript(node: torch.fx.Node, config: Config) -> bool:
    """Whether static narrowing can use ordinary JAX basic indexing."""
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)):
        return False
    if not all(
        index is None or index == slice(None) or _static_index(index) is not None
        for index in indices
    ):
        return False

    input_value = _node_value(node.args[0])
    if not isinstance(input_value, torch.Tensor):
        return False
    tensor_dim = 0
    for index in indices:
        if index is None:
            continue
        if tensor_dim >= input_value.ndim:
            return False
        if index != slice(None):
            static_index = _static_index(index)
            assert static_index is not None
            size = _concrete_size(input_value.shape[tensor_dim], config, 1)
            if size is None:
                return False
            if isinstance(static_index, _StaticIndexRange):
                if static_index.start + static_index.length > size:
                    return False
            elif not -size <= static_index < size:
                return False
        tensor_dim += 1

    source = node.args[0]
    seen: set[torch.fx.Node] = set()
    while isinstance(source, torch.fx.Node) and source not in seen:
        seen.add(source)
        if source.op != "call_function":
            return False
        if source.target in _RESIDENT_REF_ATEN_VIEW_TARGETS:
            source = source.args[0]
            continue
        # An earlier narrowing subscript may still carry a resident Ref and its
        # boundary-mask invariants. A root load, by contrast, can materialize as
        # an ordinary value before applying this compile-time basic index.
        return source.target is not subscript
    return False


def _resident_plan(node: torch.fx.Node) -> _ResidentPlan | None:
    # Codegen repair may reload this module after planning, so an existing plan
    # can be an instance of the pre-reload NamedTuple class. The private metadata
    # key is the stable type boundary across that reload.
    return cast("_ResidentPlan | None", node.meta.get(RESIDENT_PLAN_META))


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
    """Select the physical shape emitted by the active grouped-worklist branch."""
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
    variant = next(
        (variant for variant in variants if variant.worklist_factor == factor), None
    )
    assert variant is not None, f"no resident Ref variant for worklist factor {factor}"
    return variant


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
    # TODO(jansel): Reuse NodeArgsGraphInfo.node_args for capture edges once kwargs()
    # remaps those nodes into per-config graph copies (see the TODO in device_ir.py).
    # Until then, reconstruct the links from the copied parent calls below.
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
        # Capture arity is a GraphInfo invariant; strict zip surfaces malformed IR.
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
    seen: set[torch.fx.Node] = set()
    while isinstance(node, torch.fx.Node):
        assert node not in seen, "cycle while resolving a resident Ref index"
        seen.add(node)
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
    seen: set[torch.fx.Graph] = set()
    while True:
        if (bounds := _loop_bounds(info, parents, block_id)) is not None:
            return bounds
        assert info.graph not in seen, "cycle while resolving an enclosing tile loop"
        seen.add(info.graph)
        parent = parents.get(info.graph_id)
        outer = graph_infos.get(parent.graph) if parent is not None else None
        if outer is None:
            return None
        info = outer


def _variant_block_size(block_id: int, config: Config, factor: int) -> int | None:
    """Resolve a block size, including a grouped compact branch's doubled tile."""
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
) -> tuple[_ResidentVariant, ...]:
    """Prove that a direct load can remain a Ref and seed its physical states."""
    from .backend import SliceAddressing
    from .backend import _slice_addressing
    from .tensorcore_plan import TENSORCORE_PLAN_META
    from .tensorcore_plan import OneHotGatherPlan

    tensor = _node_value(producer.args[0])
    value = producer.meta.get("val")
    indices = producer.args[1]
    location = _where(producer)
    if not isinstance(tensor, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise exc.BackendUnsupported(
            "pallas", f"the source load at {location} has no tensor value metadata"
        )
    if tensor.ndim != value.ndim:
        raise exc.BackendUnsupported(
            "pallas", f"the source load at {location} changes the tensor rank"
        )
    if producer.args[2] is not None:
        raise exc.BackendUnsupported(
            "pallas", f"the source load at {location} has an explicit mask"
        )
    if not isinstance(indices, (list, tuple)):
        raise exc.BackendUnsupported(
            "pallas", f"the source load at {location} has unsupported indices"
        )
    if isinstance(producer.meta.get(TENSORCORE_PLAN_META), OneHotGatherPlan):
        raise exc.BackendUnsupported(
            "pallas", f"the source load at {location} uses an indirect gather"
        )

    selected = _narrowed_dims(indices)
    if len(selected) != 1:
        raise exc.BackendUnsupported(
            "pallas",
            f"the source load at {location} must tile exactly one dimension, "
            f"but it tiles {len(selected)}",
        )
    dim = selected[0]
    outer_block_id = CompileEnvironment.current().resolve_block_id(
        _node_value(indices[dim])
    )
    if outer_block_id is None:
        raise exc.BackendUnsupported(
            "pallas", f"the source load at {location} is not indexed by a tile"
        )

    # A full source loop with evenly-sized blocks has no tail padding. Otherwise
    # the loaded Ref remains guarded until a selector proves it uses the exact live
    # extent of this source block.
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

    # Grouping=2 emits two static branches: one base block for a short work item
    # and one double block for a full grouped item. Both must satisfy the plan.
    worklist = CompileEnvironment.current().compact_worklist_plan
    factors = (1, 2) if worklist is not None and worklist.grouping == 2 else (1,)
    variants: list[_ResidentVariant] = []
    for factor in factors:
        shape = _physical_shape(value, context.config, factor)
        if shape is None:
            raise exc.BackendUnsupported(
                "pallas",
                f"the source load at {location} has no concrete physical shape "
                "for this config",
            )
        if _slice_addressing(value, dim, shape[-1]) is not SliceAddressing.DIRECT:
            raise exc.BackendUnsupported(
                "pallas",
                f"the source load at {location} does not have direct VMEM "
                "addressing for this config",
            )
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
    """Return the block id for an unshifted vector tile index, if recognized."""
    from ..indexing_strategy import subscript_tile_info

    env = CompileEnvironment.current()
    info = subscript_tile_info(env, index)
    if info is not None:
        return info.block_id if env.known_equal(info.offset, 0) else None
    if index.target is tile_index and index.args:
        return env.resolve_block_id(_node_value(index.args[0]))
    return None


def _apply_selector_to_variants(
    location: str,
    input_value: torch.Tensor,
    variants: tuple[_ResidentVariant, ...],
    selector: _ResidentSelector,
    *,
    squeeze: bool,
    outer_block_id: int,
    static_loop_extent: int,
    config: Config,
) -> _ResidentTransform:
    """Validate and apply a logical selector to every physical Ref variant."""
    from .backend import SliceAddressing
    from .backend import _slice_addressing

    logical_dim = selector.logical_dim
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
            raise exc.BackendUnsupported(
                "pallas",
                f"the selector at {location} does not have direct VMEM "
                "addressing for this config",
            )
        width = (
            _variant_block_size(
                selector.local_block_id, config, variant.worklist_factor
            )
            if selector.kind == "tile"
            else selector.width
        )
        if not isinstance(width, int) or width < 1:
            raise exc.BackendUnsupported(
                "pallas",
                f"the selector at {location} has no concrete positive width "
                "for this config",
            )
        if squeeze and width != 1:
            raise exc.BackendUnsupported(
                "pallas",
                f"the scalar selector at {location} cannot use width {width} "
                "for this config",
            )
        if width > variant.shape[physical_dim]:
            raise exc.BackendUnsupported(
                "pallas",
                f"the run is {width} wide but the block holds only "
                f"{variant.shape[physical_dim]} rows for this config at {location}",
            )
        if selector.kind == "static":
            assert isinstance(selector.begin, int)
            if selector.begin + width > variant.shape[physical_dim]:
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the static run at {location} reaches past the resident "
                    "Ref for this config",
                )
            if variant.live_guard is not None and variant.live_guard[0] == physical_dim:
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the static selector at {location} may address padding in "
                    "the source tile for this config",
                )
        elif selector.kind == "tile":
            if variant.shape[physical_dim] % width != 0:
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the selector width at {location} does not divide the Ref "
                    "for this config",
                )
            if (
                static_loop_extent >= 0
                and static_loop_extent != variant.shape[physical_dim]
            ):
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the inner loop at {location} does not cover exactly the Ref "
                    "for this config",
                )

        # Only selection through the matching source live extent proves that the
        # guarded physical dimension no longer contains addressable padding.
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

    return _ResidentTransform(tuple(output), selector)


def _selector(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    variants: tuple[_ResidentVariant, ...],
    context: _PlanningContext,
) -> _ResidentTransform:
    """Validate one contiguous Ref selector and return its output state.

    Dynamic selectors are accepted only when an enclosing zero-based tile loop or
    an exact tail predicate proves both their offset provenance and live extent.
    """
    from ..device_ir import IfGraphInfo
    from ..type_info import _detect_outer_block_bound
    from ..variable_origin import TileBeginOrigin
    from .plan_tiling import _maybe_get_symbol_origin

    location = _where(node)
    indices = node.args[1]
    input_node = node.args[0]
    input_value = (
        input_node.meta.get("val") if isinstance(input_node, torch.fx.Node) else None
    )
    if not isinstance(indices, (list, tuple)) or not isinstance(
        input_value, torch.Tensor
    ):
        raise exc.BackendUnsupported(
            "pallas",
            f"the selector at {location} has no tensor indexing metadata",
        )
    if any(index is None for index in indices):
        raise exc.BackendUnsupported(
            "pallas",
            f"the selector at {location} adds a dimension; resident Ref "
            "narrowing does not support None",
        )
    selected = _narrowed_dims(indices)
    if len(selected) != 1:
        raise exc.BackendUnsupported(
            "pallas",
            f"the selector at {location} must narrow exactly one dimension, "
            f"but it narrows {len(selected)}",
        )
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
    # A dynamic tail is safe only in the branch whose predicate proves that the
    # requested fixed-width iota is live. Merely sharing the source block symbol is
    # insufficient because the final source block may be shorter.
    if source is not None and source.target is torch.ops.prims.iota.default:
        if not isinstance(graph_info, IfGraphInfo) or not source.args:
            raise exc.BackendUnsupported(
                "pallas",
                f"the dynamic tail selector at {location} is not inside its "
                "matching conditional",
            )
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
            raise exc.BackendUnsupported(
                "pallas",
                f"the dynamic tail selector at {location} is not a contiguous "
                "positive iota",
            )
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
            raise exc.BackendUnsupported(
                "pallas",
                f"the dynamic tail selector at {location} is not guarded by "
                "its exact live-width predicate",
            )
        selector = _ResidentSelector("tail", logical_dim, -1, start, width, False)
    elif source is not None:
        # Ordinary dynamic runs must partition the resident block from zero. A
        # symbolic end must be the exact live extent of the source tile so it can
        # also discharge that source's tail-padding guard below.
        value = source.meta.get("val")
        squeeze = isinstance(value, torch.SymInt)
        if squeeze:
            origin = _maybe_get_symbol_origin(value)
            if origin is None or not isinstance(origin.origin, TileBeginOrigin):
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the scalar selector at {location} is not a tile begin",
                )
            local_block_id = origin.origin.block_id
        else:
            detected_local = _tile_run(source)
            if detected_local is None:
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the selector at {location} is not an unshifted tile run",
                )
            local_block_id = detected_local
        bounds = _enclosing_loop_bounds(
            graph_info, context.parents, context.graph_infos, local_block_id
        )
        if bounds is None:
            raise exc.BackendUnsupported(
                "pallas",
                f"the selector at {location} is not driven by an enclosing tile loop",
            )
        start, end = bounds
        if not isinstance(start, (int, torch.SymInt)) or not env.known_equal(start, 0):
            raise exc.BackendUnsupported(
                "pallas",
                f"the selector's tile loop at {location} must start at zero",
            )
        if isinstance(end, int):
            if end < 1:
                raise exc.BackendUnsupported(
                    "pallas", f"the selector's tile loop at {location} is empty"
                )
            static_loop_extent = end
        elif not isinstance(end, torch.SymInt):
            raise exc.BackendUnsupported(
                "pallas",
                f"the selector's tile loop at {location} has an unsupported end",
            )
        else:
            detected_outer = exact_outer_live(end)
            outer_block_id = detected_outer if detected_outer is not None else -1
            if outer_block_id < 0:
                raise exc.BackendUnsupported(
                    "pallas",
                    f"the selector's live extent at {location} does not match "
                    "the source tile",
                )
        selector = _ResidentSelector(
            "tile", logical_dim, local_block_id, -1, -1, not squeeze
        )
    elif isinstance(index, int):
        if index < 0:
            raise exc.BackendUnsupported(
                "pallas",
                f"the static selector at {location} has a negative index",
            )
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
            raise exc.BackendUnsupported(
                "pallas",
                f"the static selector at {location} is not a positive contiguous slice",
            )
        selector = _ResidentSelector(
            "static", logical_dim, -1, index.start, index.stop - index.start, False
        )
    else:
        raise exc.BackendUnsupported(
            "pallas",
            f"the selector at {location} has unsupported index {index!r}",
        )

    return _apply_selector_to_variants(
        location,
        input_value,
        variants,
        selector,
        squeeze=squeeze,
        outer_block_id=outer_block_id,
        static_loop_extent=static_loop_extent,
        config=context.config,
    )


def _reshape_variants(
    node: torch.fx.Node,
    config: Config,
    variants: tuple[_ResidentVariant, ...],
) -> tuple[_ResidentVariant, ...]:
    """Prove a view changes shape without changing the Ref's TPU lane layout.

    Mosaic can reinterpret major dimensions, but the physical element count and
    two hardware-tiled minor dimensions must remain identical. A partially-live
    Ref is rejected because reshaping would lose which physical dimension carries
    its tail-padding guard.
    """
    value = node.meta.get("val")
    target = str(node.target)
    location = _where(node)
    if not isinstance(value, torch.Tensor):
        raise exc.BackendUnsupported(
            "pallas", f"{target} at {location} has no tensor value metadata"
        )
    if value.dtype is torch.bool:
        raise exc.BackendUnsupported(
            "pallas",
            f"{target} at {location} cannot preserve a boolean resident Ref",
        )
    output: list[_ResidentVariant] = []
    for variant in variants:
        new_shape = _physical_shape(value, config, variant.worklist_factor)
        if variant.live_guard is not None:
            raise exc.BackendUnsupported(
                "pallas",
                f"{target} at {location} cannot reshape a partially live resident "
                "Ref for this config",
            )
        if len(variant.shape) < 2:
            raise exc.BackendUnsupported(
                "pallas",
                f"{target} at {location} requires a resident Ref with at least "
                "two physical dimensions",
            )
        if new_shape is None:
            raise exc.BackendUnsupported(
                "pallas",
                f"{target} at {location} has no concrete physical shape for this "
                "config",
            )
        if prod(variant.shape) != prod(new_shape):
            raise exc.BackendUnsupported(
                "pallas",
                f"{target} at {location} changes the resident block's physical "
                "element count for this config",
            )
        if variant.shape[-2:] != new_shape[-2:]:
            raise exc.BackendUnsupported(
                "pallas",
                f"{target} at {location} changes the resident block's two minor "
                "dimensions for this config",
            )
        output.append(_ResidentVariant(variant.worklist_factor, new_shape, (), None))
    return tuple(output)


def _registered_transform(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    variants: tuple[_ResidentVariant, ...],
    context: _PlanningContext,
) -> _ResidentTransform:
    if node.target is subscript:
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)):
            raise exc.BackendUnsupported(
                "pallas",
                f"hl.subscript at {_where(node)} has unsupported indices",
            )
        if not _narrowed_dims(indices):
            output = _reshape_variants(node, context.config, variants)
            return _ResidentTransform(output, None)
        return _selector(node, graph_info, variants, context)

    if node.target in _RESIDENT_REF_ATEN_VIEW_TARGETS:
        output = _reshape_variants(node, context.config, variants)
        return _ResidentTransform(output, None)

    supported = ", ".join(
        [
            *(sorted(str(target) for target in _RESIDENT_REF_ATEN_VIEW_TARGETS)),
            "hl.subscript",
        ]
    )
    raise exc.BackendUnsupported(
        "pallas",
        f"{node.target} at {_where(node)} is not an address-preserving resident "
        f"Ref transform; expected one of {supported}",
    )


def _effective_users(
    node: torch.fx.Node,
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
) -> list[tuple[torch.fx.Node, tuple[torch.fx.Node, ...]]]:
    """Return semantic users while traversing identity-only graph transports.

    ``_new_var`` and control-flow placeholders carry the same value into another
    scope, so they are returned separately as transports to annotate. Parent calls
    and shape-only reads are not value consumers of the resident Ref.
    """
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


def _record_descendant_failure(
    producer: torch.fx.Node,
    context: _PlanningContext,
    failure: exc.BackendUnsupported,
) -> None:
    seen: set[torch.fx.Node] = set()
    stack = [producer]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.target is subscript and _narrowed_dims(node.args[1]):
            # Keep the first failure found below this node. Later transactional
            # rollback errors are broader and must not hide the local diagnostic.
            context.failures.setdefault(node, failure)
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
) -> None:
    """Plan every semantic user, materializing only at a valid value boundary.

    The root load itself cannot serve both ordinary value users and resident Ref
    users. A non-root view may end the Ref chain by materializing, provided it is
    fully live and no later narrowing selector depends on retaining the Ref.
    """
    users = _effective_users(node, context.captures)
    planned: dict[
        torch.fx.Node,
        tuple[
            tuple[torch.fx.Node, ...],
            tuple[_ResidentVariant, ...],
            _ResidentSelector | None,
        ],
    ] = {}
    unsupported: list[tuple[torch.fx.Node, exc.BackendUnsupported]] = []
    for user, transports in users:
        user_info = context.graph_infos.get(user.graph)
        if user_info is None:
            failure = exc.BackendUnsupported(
                "pallas",
                f"the resident Ref crosses an unknown graph at {_where(user)}",
            )
            unsupported.append((user, failure))
            _record_descendant_failure(user, context, failure)
            continue
        try:
            result = _registered_transform(
                user,
                user_info,
                node_variants,
                context,
            )
        except exc.BackendUnsupported as failure:
            unsupported.append((user, failure))
            _record_descendant_failure(user, context, failure)
        else:
            planned[user] = (transports, result.variants, result.selector)

    # A root load has one codegen representation. If any branch needs its ordinary
    # value, it cannot simultaneously remain a Ref for a narrowing branch.
    if root and (unsupported or not planned):
        if planned and unsupported:
            blockers = [_where(user) for user, _failure in unsupported]
            reason = "the block is also consumed whole at " + ", ".join(blockers[:3])
            failure = exc.BackendUnsupported("pallas", reason)
            for user in planned:
                _record_descendant_failure(user, context, failure)
            raise failure
        if unsupported:
            raise unsupported[0][1]
        raise exc.BackendUnsupported(
            "pallas",
            f"the resident load at {_where(node)} has no address-preserving users",
        )
    # Past the root, an unsupported or terminal user is normally a legal boundary:
    # materialize this view and let ordinary Pallas codegen continue from its value.
    if not root and (
        (selector is not None and selector.mask) or unsupported or not planned
    ):
        masked_boundary = selector is not None and selector.mask and bool(planned)
        if masked_boundary:
            # A mask changes values rather than addresses. It is applied only when
            # materializing, so an address-only Ref chain cannot cross this selector.
            failure = exc.BackendUnsupported(
                "pallas",
                f"the masked selection at {_where(node)} must materialize before "
                "another narrowing subscript",
            )
            for user in planned:
                _record_descendant_failure(user, context, failure)

        live_guards = [
            variant.live_guard
            for variant in node_variants
            if variant.live_guard is not None
        ]
        # Materializing copies the full physical Ref. It is unsafe until every tail
        # guard has been discharged, because padding would become an ordinary value.
        if live_guards:
            failure = exc.BackendUnsupported(
                "pallas",
                f"the resident Ref at {_where(node)} may contain padding and "
                "cannot be materialized for this config",
            )
            _record_descendant_failure(node, context, failure)
            raise failure

        if planned and unsupported and not masked_boundary:
            # Ordinary users force materialization here; record why any deeper
            # narrowing users can no longer receive a resident Ref.
            blockers = [_where(user) for user, _failure in unsupported]
            reason = (
                f"the resident view at {_where(node)} must materialize because "
                "it is also consumed at " + ", ".join(blockers[:3])
            )
            failure = exc.BackendUnsupported("pallas", reason)
            for user in planned:
                _record_descendant_failure(user, context, failure)
        annotations[node] = _ResidentPlan(True, node_variants, selector)
        return

    annotations[node] = _ResidentPlan(False, node_variants, selector)
    for user, (transports, output, user_spec) in planned.items():
        # Identity transports retain the input physical state; the semantic user
        # receives the transformed state returned by _registered_transform.
        transport_plan = _ResidentPlan(False, node_variants, None)
        for transport in transports:
            annotations[transport] = transport_plan
        _analyze_resident_chain(
            user,
            output,
            user_spec,
            context=context,
            annotations=annotations,
        )


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
            # Keeping a Ref defers the physical read until its consumer. A device
            # write through an alias could therefore change program semantics.
            if (
                isinstance(tensor, torch.Tensor)
                and id(tensor.untyped_storage()) in mutated_storages
            ):
                _record_descendant_failure(
                    producer,
                    context,
                    exc.BackendUnsupported(
                        "pallas", "the tensor it reads is also written on device"
                    ),
                )
                continue

            annotations: dict[torch.fx.Node, _ResidentPlan] = {}
            try:
                variants = _root_variants(producer, info, context)
                _analyze_resident_chain(
                    producer,
                    variants,
                    None,
                    context=context,
                    annotations=annotations,
                    root=True,
                )
            except exc.BackendUnsupported as failure:
                # Planning is transactional. Carry a child failure to every
                # narrowing node that loses its tentative annotation.
                _record_descendant_failure(producer, context, failure)
            else:
                for node, resident_plan in annotations.items():
                    node.meta[RESIDENT_PLAN_META] = resident_plan

    # Narrowing is accepted during tracing before its load provenance and config
    # are known. The load-rooted walk may also stop at a legitimate value boundary,
    # so only this completeness pass decides whether a recorded failure is fatal.
    for info in graphs:
        for node in info.graph.find_nodes(op="call_function", target=subscript):
            node.meta.pop(STATIC_BASIC_VALUE_SUBSCRIPT_META, None)
            if _resident_plan(node) is not None or not _narrowed_dims(node.args[1]):
                continue
            if _is_static_basic_value_subscript(node, config):
                node.meta[STATIC_BASIC_VALUE_SUBSCRIPT_META] = True
                continue
            failure = context.failures.get(node)
            if failure is None:
                failure = exc.BackendUnsupported(
                    "pallas",
                    f"narrowing subscript {node.name} requires a resident Ref, but "
                    "its input is not derived from an eligible direct Pallas load",
                )
            raise failure


def _codegen_resident_subscript(state: CodegenState) -> ast.AST:
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
    # ``Ref.at[...]`` composes an address transform; ``Ref[...]`` performs the
    # load. The planner decides which representation downstream users require.
    accessor = "" if plan.materialize else ".at"
    result = expr_from_string(
        f"{{base}}{accessor}[{', '.join(parts)}]", base=state.ast_arg(0)
    )
    if not plan.materialize:
        return result

    assert variant.live_guard is None
    # Scalar selectors are tracked as hidden physical dimensions while the value
    # remains a Ref. Drop them only after the boundary load has materialized it.
    squeezed_dims = variant.squeezed_dims
    if squeezed_dims:
        squeeze_parts = [
            "0" if dim in squeezed_dims else ":" for dim in range(len(variant.shape))
        ]
        result = expr_from_string(
            f"{{result}}[{', '.join(squeeze_parts)}]", result=result
        )
    if mask:
        # Like squeezing, masking changes the loaded value and therefore belongs
        # after materialization rather than in the address-only Ref transform.
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
        return _codegen_resident_subscript(state)
    indices = state.fx_node.args[1]
    if state.fx_node.meta.get(STATIC_BASIC_VALUE_SUBSCRIPT_META):
        assert isinstance(indices, (list, tuple))
        placeholders: dict[str, ast.AST] = {"base": state.ast_arg(0)}
        rendered: list[str] = []
        for index in indices:
            if index is None:
                rendered.append("None")
            elif index == slice(None):
                rendered.append(":")
            else:
                static_index = _static_index(index)
                assert static_index is not None
                if isinstance(static_index, _StaticIndexRange):
                    rendered.append(
                        f"{static_index.start}:{static_index.start + static_index.length}"
                    )
                else:
                    rendered.append(repr(static_index))
        return expr_from_string(
            f"{{base}}[{', '.join(rendered)}]",
            **placeholders,
        )
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
