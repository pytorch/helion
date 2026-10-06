"""Current-call scalar control frames, lexical epochs and logical domains.

Structural, epoch and bound readonly/domain proofs are separate transactional
steps. The emitter still proves physical ownership and performs publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import scan_ops
from ..device_ir import ElseGraphInfo
from ..device_ir import GraphInfo
from ..device_ir import HelperFunctionGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import WhileConditionGraphInfo
from ..device_ir import WhileLoopGraphInfo
from ..device_ir import control_flow_parent_entries
from .scan_helper import additive_scan_helper
from .uniform_control import _carry_map
from .uniform_control import _outputs
from .uniform_control import _signature

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy

    from ..compile_environment import CompileEnvironment
    from .gather_domains import GatherDomainFacts


@dataclass(frozen=True)
class UniformRegionFrame:
    graph: GraphInfo
    call: Node | None
    parent_graph_id: int | None
    role: str
    captures: tuple[Node, ...]
    placeholders: tuple[Node, ...]
    outputs: tuple[Node, ...]
    carry_map: tuple[tuple[int, int], ...]
    local_targets: frozenset[Node]


@dataclass(frozen=True)
class CompletedLocalRead:
    node: Node
    allocation: Node
    frame_graph_id: int
    logical_extent: int


@dataclass(frozen=True)
class UniformRegionTree:
    frames: tuple[UniformRegionFrame, ...]
    auxiliary_graph_ids: tuple[int, ...]
    indexed_reads: tuple[CompletedLocalRead, ...] = ()
    scan_calls: tuple[Node, ...] = ()
    output_stores: frozenset[Node] = frozenset()
    requirements: tuple[str, ...] = (
        "scalar_CTA_owner_and_shared_predicate_publication",
        "readonly_captures_and_logical_domains_in_actual_parent_context",
        "initialize_update_final_read_epoch_transfer_and_retirement",
        "transactional_arm_join_and_simultaneous_scalar_commits",
        "zero_trip_entry_and_backedge_read_WAR_barriers",
        "complete_lowering_and_configuration_resource_checks",
    )


def uniform_region_tree(graphs: Sequence[GraphInfo]) -> UniformRegionTree:
    """Validate one actual rooted if/while tree without publishing cached facts.

    Sequential calls and nested calls use the same frame representation. Loops
    with vector carries, non-scalar joins, captured atomic mutation, graph reuse
    and unsupported control kinds remain outside this initial structural slice.
    """

    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise exc.InvalidConfig(f"uniform region tree: {reason}")

    roots = [g for g in graphs if isinstance(g, RootGraphInfo)]
    require(len(roots) == 1, "one root required")
    require(
        all(g.graph_id == i for i, g in enumerate(graphs))
        and len({id(g.graph) for g in graphs}) == len(graphs),
        "current indexed unique graphs required",
    )
    entries: dict[int, tuple[Node, int, str, type[GraphInfo]]] = {}
    calls: list[Node] = []
    owners = {g.graph: g for g in graphs}
    for info in graphs:
        seen: set[Node] = set()
        for node in info.graph.nodes:
            require(node.graph is info.graph, "node ownership")
            require(all(v in seen for v in node.all_input_nodes), "dominating operands")
            seen.add(node)
            require(
                not _tracing_ops.is_for_loop_target(node.target), "for is outside slice"
            )
            if node.target not in (_tracing_ops._if, _tracing_ops._while_loop):
                continue
            require(node.op == "call_function" and not node.kwargs, "control ABI")
            calls.append(node)
            if node.target is _tracing_ops._if:
                require(len(node.args) == 5, "if ABI")
                predicate = node.args[0]
                require(
                    isinstance(predicate, Node)
                    and _signature(predicate) == (torch.bool, ()),
                    "scalar Boolean predicate required",
                )
                children = [
                    (1, 3, "if_true", IfGraphInfo),
                    (2, 4, "if_false", ElseGraphInfo),
                ]
            else:
                require(len(node.args) == 4 and node.args[3] is None, "while ABI")
                children = [
                    (0, 2, "while_condition", WhileConditionGraphInfo),
                    (1, 2, "while_body", WhileLoopGraphInfo),
                ]
            for graph_slot, capture_slot, role, kind in children:
                child = node.args[graph_slot]
                require(
                    type(child) is int and 0 <= child < len(graphs), "child graph ID"
                )
                child = cast("int", child)
                require(
                    child not in entries and isinstance(graphs[child], kind),
                    "unique typed child",
                )
                entries[child] = (node, capture_slot, role, kind)
    # The existing utility covers if/for only. Confirm its actual if entries,
    # then add while condition/body entries from the same current call ABI.
    require(
        all(
            gid in entries and entries[gid][:2] == pair
            for gid, pair in control_flow_parent_entries(graphs).items()
        ),
        "parent helper disagreement",
    )
    frames: list[UniformRegionFrame] = []
    active: set[int] = set()
    visited: set[int] = set()

    def visit(info: GraphInfo) -> None:
        require(info.graph_id not in active | visited, "recursive or reused graph")
        active.add(info.graph_id)
        call = None
        parent = None
        role = "root"
        captures: tuple[Node, ...] = ()
        placeholders = tuple(info.graph.find_nodes(op="placeholder"))
        carry_map: tuple[tuple[int, int], ...] = ()
        outputs = () if info is roots[0] else _outputs(info)
        if info is not roots[0]:
            require(info.graph_id in entries, "unreachable child")
            call, slot, role, _ = entries[info.graph_id]
            parent = owners[call.graph].graph_id
            require(parent in active, "actual ancestor required")
            raw = call.args[slot]
            require(
                isinstance(raw, (tuple, list))
                and all(isinstance(v, Node) for v in raw),
                "capture node sequence",
            )
            captures = tuple(cast("Sequence[Node]", raw))
            require(
                len(captures) == len(placeholders)
                and len(set(captures)) == len(captures),
                "capture arity/identity",
            )
            positions = {n: i for i, n in enumerate(call.graph.nodes)}
            for value, placeholder in zip(captures, placeholders, strict=True):
                require(
                    isinstance(value, Node)
                    and value.graph is call.graph
                    and positions[value] < positions[call],
                    "current dominating capture",
                )
                require(
                    _signature(value) == _signature(placeholder),
                    "capture physical type/shape",
                )
            if role == "while_condition":
                require(
                    len(outputs) == 1 and _signature(outputs[0]) == (torch.bool, ()),
                    "condition output",
                )
                require(
                    not any(n in calls for n in info.graph.nodes),
                    "condition control unsupported",
                )
            if role == "while_body":
                require(
                    cast("WhileLoopGraphInfo", info).cond_graph_id == call.args[0],
                    "paired condition identity",
                )
                require(
                    all(_signature(v)[1] == () for v in outputs), "scalar carries only"
                )
                carry_map = _carry_map(call, captures, outputs)
        targets: set[Node] = set()
        for node in info.graph.nodes:
            if node.target in atomic_ops.ATOMIC_OPS:
                target = node.args[0]
                require(
                    role != "while_condition"
                    and node.target is atomic_ops.atomic_add
                    and isinstance(target, Node)
                    and target.graph is info.graph
                    and target.target is creation_ops.full,
                    "fresh lexical atomic target only",
                )
                targets.add(cast("Node", target))
        frames.append(
            UniformRegionFrame(
                info,
                call,
                parent,
                role,
                captures,
                placeholders,
                outputs,
                carry_map,
                frozenset(targets),
            )
        )
        for node in info.graph.nodes:
            for gid, (caller, _, _, _) in entries.items():
                if caller is node:
                    visit(graphs[gid])
        active.remove(info.graph_id)
        visited.add(info.graph_id)

    visit(roots[0])
    scan_calls: list[Node] = []
    scan_helpers: set[int] = set()
    for frame in frames:
        for node in frame.graph.graph.nodes:
            if node.target is not scan_ops._associative_scan:
                continue
            require(
                node.op == "call_function"
                and len(node.args) == 5
                and not node.kwargs
                and type(node.args[0]) is int
                and 0 <= node.args[0] < len(graphs)
                and isinstance(node.args[1], Node)
                and node.args[1].graph is frame.graph.graph
                and type(node.args[2]) is int
                and node.args[3] is False
                and node.args[4] is False,
                "current forward single-tensor scan call ABI",
            )
            source = cast("Node", node.args[1])
            signature = _signature(source)
            require(
                bool(signature[1])
                and _signature(node) == signature
                and -len(signature[1]) <= node.args[2] < len(signature[1]),
                "scan input/output signature and axis",
            )
            helper_id = cast("int", node.args[0])
            helper = graphs[helper_id]
            require(
                helper_id not in visited
                and isinstance(helper, HelperFunctionGraphInfo)
                and additive_scan_helper(helper),
                "called closed additive scan helper",
            )
            lhs, rhs, add, output = helper.graph.nodes
            require(
                add.op == "call_function"
                and set(add.kwargs) <= {"alpha"}
                and type(add.kwargs.get("alpha", 1)) in (int, float)
                and not lhs.args
                and not rhs.args
                and not lhs.kwargs
                and not rhs.kwargs
                and not output.kwargs
                and all(
                    _signature(value) == (signature[0], (1,))
                    for value in (lhs, rhs, add)
                ),
                "closed typed scalar scan helper",
            )
            scan_calls.append(node)
            scan_helpers.add(helper_id)
    auxiliary = tuple(g.graph_id for g in graphs if g.graph_id not in visited)
    require(
        all(
            (gid in scan_helpers or isinstance(graphs[gid], ReductionLoopGraphInfo))
            and not any(n in calls for n in graphs[gid].graph.nodes)
            for gid in auxiliary
        ),
        "unreachable control graph",
    )
    by_id = {f.graph.graph_id: f for f in frames}
    for call in calls:
        if call.target is not _tracing_ops._if:
            continue
        left, right = (by_id[cast("int", gid)] for gid in call.args[1:3])
        info = cast("IfGraphInfo", left.graph)
        require(
            info.branches_outputs is not None
            and info.if_arg_names is not None
            and info.else_arg_names is not None,
            "initialized branch metadata",
        )
        outer: dict[str, Node] = {}
        for names, frame in [
            (cast("list[str]", info.if_arg_names), left),
            (cast("list[str]", info.else_arg_names), right),
        ]:
            require(len(names) == len(frame.captures), "capture-name arity")
            for name, value in zip(names, frame.captures, strict=True):
                require(
                    name not in outer or outer[name] is value,
                    "ambiguous unchanged capture",
                )
                outer[name] = value
        join_signatures: list[tuple[torch.dtype, tuple[int, ...]]] = []
        for pair in cast("list[tuple[int | str, ...]]", info.branches_outputs):
            require(len(pair) == 2, "two branch outcomes required")
            values: list[Node] = []
            for slot, frame in zip(pair, [left, right], strict=True):
                if type(slot) is int:
                    require(0 <= slot < len(frame.outputs), "branch output index")
                    values.append(frame.outputs[slot])
                else:
                    require(
                        isinstance(slot, str) and slot in outer,
                        "initialized unchanged output",
                    )
                    values.append(outer[cast("str", slot)])
            require(
                _signature(values[0]) == _signature(values[1])
                and _signature(values[0])[1] == (),
                "compatible scalar join required",
            )
            join_signatures.append(_signature(values[0]))
        call_values = call.meta.get("val")
        require(
            isinstance(call_values, (list, tuple))
            and len(call_values) == 2 * len(join_signatures),
            "branch call result arity",
        )
        for value, signature in zip(
            cast("Sequence[object]", call_values),
            [*join_signatures, *join_signatures],
            strict=True,
        ):
            require(
                isinstance(value, torch.Tensor)
                and (value.dtype, tuple(value.shape)) == signature,
                "branch call result signature",
            )
        joins = cast("list[tuple[int | str, ...]]", info.branches_outputs)
        count = len(joins)
        positions = {n: i for i, n in enumerate(call.graph.nodes)}
        for item in call.users:
            require(
                item.graph is call.graph
                and item in positions
                and item.target is operator.getitem
                and len(item.args) == 2
                and not item.kwargs
                and item.args[0] is call
                and type(item.args[1]) is int
                and 0 <= item.args[1] < 2 * count
                and positions[item] > positions[call],
                "actual branch projection",
            )
            index = cast("int", item.args[1])
            pair = joins[index % count]
            slot = pair[0]
            value = (
                left.outputs[slot] if type(slot) is int else outer[cast("str", slot)]
            )
            require(
                _signature(item) == _signature(value), "branch projection signature"
            )
            for phi in item.users:
                require(
                    phi.graph is call.graph
                    and phi in positions
                    and phi.target is _tracing_ops._phi
                    and len(phi.args) == 2
                    and not phi.kwargs
                    and positions[phi] > positions[item],
                    "branch phi required",
                )
                indices = []
                for projection in phi.args:
                    require(
                        isinstance(projection, Node)
                        and projection.target is operator.getitem
                        and projection.graph is call.graph
                        and projection in positions
                        and len(projection.args) == 2
                        and projection.args[0] is call
                        and type(projection.args[1]) is int
                        and not projection.kwargs
                        and positions[projection] < positions[phi],
                        "actual paired branch projection",
                    )
                    indices.append(cast("int", cast("Node", projection).args[1]))
                require(
                    sorted(indices) == [index % count, index % count + count],
                    "paired branch phi halves",
                )
                require(_signature(phi) == _signature(value), "branch phi signature")
    return UniformRegionTree(tuple(frames), auxiliary, scan_calls=tuple(scan_calls))


def uniform_local_regions(graphs: Sequence[GraphInfo]) -> UniformRegionTree:
    """Prove lexical completed epochs on a current-call scalar-control tree.

    A capture is a final read at its actual owner call, even when unused. Mutable
    targets cannot be captured for mutation or escape as a selected scalar join.
    Logical domains and bound host disjointness remain a transactional next step.
    """
    from ...language import memory_ops
    from ...language import scan_ops
    from ...language import view_ops
    from ..aten_lowering import alias_lowering
    from ..aten_lowering import view_dtype_lowering
    from ..inductor_lowering import APIFuncLowering
    from ..inductor_lowering import PointwiseLowering
    from ..inductor_lowering import ReductionLowering
    from ..inductor_lowering import SympyExprLowering
    from .local_atomic import local_indexed_load

    tree = uniform_region_tree(graphs)
    frames = {f.graph.graph: f for f in tree.frames}
    allocations = frozenset(a for f in tree.frames for a in f.local_targets)

    def require(ok: bool, why: str) -> None:
        if not ok:
            raise exc.InvalidConfig(f"uniform local regions: {why}")

    require(bool(allocations) and len(tree.frames) > 1, "fresh control epochs required")
    captures = {
        placeholder: entry
        for f in tree.frames
        for placeholder, entry in zip(f.placeholders, f.captures, strict=True)
    }

    def origin(node: Node) -> Node:
        while node in captures or node.target in (
            _tracing_ops._new_var,
            torch.ops.aten.alias.default,
        ):
            node = captures[node] if node in captures else cast("Node", node.args[0])
        return node

    def same_tensor_metadata(source: object, result: object) -> bool:
        return (
            isinstance(source, torch.Tensor)
            and isinstance(result, torch.Tensor)
            and all(type(size) is int and size > 0 for size in source.shape)
            and all(type(size) is int and size > 0 for size in result.shape)
            and tuple(source.shape) == tuple(result.shape)
            and source.layout == result.layout == torch.strided
            and all(type(stride) is int for stride in source.stride())
            and all(type(stride) is int for stride in result.stride())
            and source.stride() == result.stride()
            and type(source.storage_offset()) is int
            and type(result.storage_offset()) is int
            and cast("int", source.storage_offset())
            == cast("int", result.storage_offset())
            and source.device == result.device
        )

    def pure_bitcast(node: Node) -> bool:
        if (
            node.target is not torch.ops.aten.view.dtype
            or node.meta.get("lowering") is not view_dtype_lowering
            or len(node.args) != 2
            or node.kwargs
            or not isinstance(node.args[0], Node)
            or node.args[0].graph is not node.graph
            or not isinstance(node.args[1], torch.dtype)
        ):
            return False
        source = node.args[0].meta.get("val")
        result = node.meta.get("val")
        numeric = (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
        return (
            isinstance(source, torch.Tensor)
            and isinstance(result, torch.Tensor)
            and source.dtype in numeric
            and result.dtype in numeric
            and result.dtype == node.args[1]
            and source.dtype.itemsize == result.dtype.itemsize
            and same_tensor_metadata(source, result)
            and origin(node.args[0]).target is not _tracing_ops._host_tensor
        )

    def pure_alias(node: Node) -> bool:
        if (
            node.target is not torch.ops.aten.alias.default
            or node.meta.get("lowering") is not alias_lowering
            or len(node.args) != 1
            or node.kwargs
            or not isinstance(node.args[0], Node)
            or node.args[0].graph is not node.graph
        ):
            return False
        source = node.args[0].meta.get("val")
        result = node.meta.get("val")
        return (
            same_tensor_metadata(source, result)
            and cast("torch.Tensor", source).dtype == cast("torch.Tensor", result).dtype
            and origin(node.args[0]).target is not _tracing_ops._host_tensor
        )

    def pure_subscript(node: Node) -> bool:
        lowering = node.meta.get("lowering")
        if (
            node.target is not view_ops.subscript
            or not isinstance(lowering, APIFuncLowering)
            or lowering.api_func is not view_ops.subscript
            or len(node.args) != 2
            or node.kwargs
            or not isinstance(node.args[0], Node)
            or node.args[0].graph is not node.graph
            or not isinstance(node.args[1], (tuple, list))
        ):
            return False
        source = node.args[0].meta.get("val")
        result = node.meta.get("val")
        indices = node.args[1]
        if (
            not isinstance(source, torch.Tensor)
            or not isinstance(result, torch.Tensor)
            or source.dtype != result.dtype
            or source.device != result.device
            or source.layout != torch.strided
            or result.layout != torch.strided
            or not all(type(size) is int and size > 0 for size in source.shape)
            or not all(type(size) is int and size > 0 for size in result.shape)
            or any(
                index is not None
                and (not isinstance(index, slice) or index != slice(None))
                for index in indices
            )
            or sum(index is not None for index in indices) != source.ndim
            or origin(node.args[0]).target is _tracing_ops._host_tensor
        ):
            return False
        axis = iter(source.shape)
        expected = tuple(1 if index is None else next(axis) for index in indices)
        return tuple(result.shape) == expected

    def through_view(node: Node) -> bool:
        # A readonly view becomes an owner-local value recipe, not a new
        # mutable target. Follow the same current-call aliases as origin().
        while node in captures or node.target is _tracing_ops._new_var:
            node = captures[node] if node in captures else cast("Node", node.args[0])
        return node.target in (
            torch.ops.aten.view.dtype,
            torch.ops.aten.alias.default,
            view_ops.subscript,
        )

    stores = frozenset(
        node
        for frame in tree.frames
        for node in frame.graph.graph.nodes
        if node.target is memory_ops.store
    )
    # Preserve the existing root-only path. A new nested output store uses one
    # approved set for both operation admission and completed-local final reads.
    output_stores = (
        stores
        if any(frames[node.graph].role != "root" for node in stores)
        else frozenset()
    )
    for node in output_stores:
        require(
            len(node.args) == 4
            and not node.kwargs
            and not node.users
            and node.meta.get("val") is None,
            "write-only store ABI",
        )
        target = node.args[0]
        require(
            isinstance(target, Node)
            and target.graph is node.graph
            and target.target is _tracing_ops._host_tensor
            and len(target.args) == 1
            and isinstance(target.args[0], str)
            and not target.kwargs
            and isinstance(target.meta.get("val"), torch.Tensor)
            and target.meta["val"].layout == torch.strided,
            "direct frame-local output pointer required",
        )
        require(
            isinstance(node.args[1], (tuple, list))
            and target not in node.args[1]
            and node.args[2] is not target
            and node.args[3] is not target,
            "output pointer is not an index, value or mask",
        )
    output_values = {
        id(cast("Node", node.args[0]).meta["val"]) for node in output_stores
    }
    for frame in tree.frames:
        for node in frame.graph.graph.nodes:
            if (
                node.target is _tracing_ops._host_tensor
                and id(node.meta.get("val")) in output_values
            ):
                require(
                    all(
                        user in output_stores and user.args[0] is node
                        for user in node.users
                    ),
                    "output pointers are write-only and cannot be captured",
                )

    indexed_reads: list[tuple[Node, Node, int]] = []
    for frame in tree.frames:
        for node in frame.graph.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            require(node.op == "call_function", "functional graph nodes required")
            if node.target is memory_ops.load:
                require(
                    len(node.args) == 4
                    and not node.kwargs
                    and isinstance(node.args[0], Node),
                    "load ABI",
                )
                source = origin(cast("Node", node.args[0]))
                if source in allocations:
                    require(local_indexed_load(node), "completed local load ABI")
                    result = node.meta.get("val")
                    require(
                        isinstance(result, torch.Tensor)
                        and result.dtype == _signature(source)[0]
                        and result.ndim <= 1,
                        "completed local read scalar/vector result type",
                    )
                    indexed_reads.append((node, source, frame.graph.graph_id))
                else:
                    require(
                        node.args[0].target is _tracing_ops._host_tensor,
                        "direct readonly host or completed local loads only",
                    )
            elif node.target is memory_ops.store:
                require(
                    frame.role == "root" or node in output_stores,
                    "approved output store required",
                )
                require(
                    not through_view(cast("Node", node.args[0])),
                    "view and alias values are readonly",
                )
            elif node.target is torch.ops.aten.view.dtype:
                require(pure_bitcast(node), "same-width pure bitcast metadata")
            elif node.target is torch.ops.aten.alias.default:
                require(pure_alias(node), "typed readonly alias metadata")
            elif node.target is view_ops.subscript:
                require(
                    pure_subscript(node), "readonly singleton-axis subscript metadata"
                )
            elif node.target in (
                _tracing_ops._if,
                _tracing_ops._while_loop,
                _tracing_ops._new_var,
                _tracing_ops._phi,
                _tracing_ops._host_tensor,
                _tracing_ops._constant_tensor,
                _tracing_ops._get_symnode,
                _tracing_ops._mask_to,
                creation_ops.full,
                atomic_ops.atomic_add,
                scan_ops._associative_scan,
            ):
                pass
            elif node.target is operator.getitem:
                require(
                    isinstance(node.args[0], Node)
                    and node.args[0].target
                    in (_tracing_ops._if, _tracing_ops._while_loop),
                    "control projections only",
                )
            elif isinstance(node.target, torch._ops.OpOverload):
                require(
                    not node.target._schema.is_mutable
                    and torch.Tag.nondeterministic_seeded not in node.target.tags
                    and torch.Tag.nondeterministic_bitwise not in node.target.tags
                    and (
                        torch.Tag.pointwise in node.target.tags
                        or isinstance(
                            node.meta.get("lowering"),
                            (PointwiseLowering, ReductionLowering),
                        )
                        or node.target
                        in (
                            torch.ops.prims.iota.default,
                            torch.ops.aten.scalar_tensor.default,
                            torch.ops.aten.full.default,
                        )
                    ),
                    "established pure tensor lowering required",
                )
            else:
                require(
                    isinstance(node.meta.get("lowering"), SympyExprLowering),
                    "opaque operation",
                )

    for allocation in allocations:
        shape, initial, dtype, _device = allocation.args
        fake = allocation.meta.get("val")
        require(
            isinstance(fake, torch.Tensor)
            and fake.ndim == 1
            and isinstance(shape, (tuple, list))
            and len(shape) == 1
            and type(shape[0]) is int
            and shape[0] > 0
            and dtype in (torch.int32, torch.float32)
            and isinstance(initial, (int, float)),
            "constant one-dimensional local target",
        )
        positions = {n: i for i, n in enumerate(allocation.graph.nodes)}
        updates: list[int] = []
        reads: list[int] = []

        def position(
            user: Node,
            allocation: Node = allocation,
            positions: dict[Node, int] = positions,
        ) -> int:
            while user.graph is not allocation.graph:
                frame = frames[user.graph]
                require(frame.call is not None, "read requires lexical ancestor owner")
                user = cast("Node", frame.call)
            return positions[user]

        for frame in tree.frames:
            for alias in frame.graph.graph.nodes:
                if origin(alias) is not allocation:
                    continue
                for user in alias.users:
                    if user.target is _tracing_ops._new_var:
                        continue
                    if user.op == "output":
                        # Tree validation permits only scalar selected joins/carries.
                        require(frame.role in ("if_true", "if_false"), "buffer escape")
                        continue
                    if user.target in (_tracing_ops._if, _tracing_ops._while_loop):
                        captures_at_call = (
                            user.args[3:5]
                            if user.target is _tracing_ops._if
                            else [user.args[2]]
                        )
                        require(
                            any(alias in seq for seq in captures_at_call),
                            "read-only capture required",
                        )
                        require(
                            user.target is not _tracing_ops._if
                            or user.args[0] is not alias,
                            "scalar predicate required",
                        )
                        reads.append(position(user))
                    elif user.target is atomic_ops.atomic_add and user.args[0] is alias:
                        require(
                            user.graph is allocation.graph,
                            "no mutation through parent captures",
                        )
                        require(
                            user.args[3] == "relaxed"
                            and (not user.users or dtype == torch.int32),
                            "local atomic type/order",
                        )
                        require(
                            user.args[2] is not alias and alias not in user.args[1],
                            "self-dependent mutation",
                        )
                        updates.append(position(user))
                    elif user.target is atomic_ops.atomic_add:
                        require(
                            user.args[0] in allocations,
                            "fresh local destination required",
                        )
                        reads.append(position(user))
                    elif user.target is memory_ops.load:
                        require(
                            user.args[0] is alias and local_indexed_load(user),
                            "completed source read only",
                        )
                        reads.append(position(user))
                    elif (
                        user.target
                        in (scan_ops._associative_scan, _tracing_ops._mask_to)
                        or pure_bitcast(user)
                        or pure_alias(user)
                        or pure_subscript(user)
                        or (
                            isinstance(user.target, torch._ops.OpOverload)
                            and not user.target._schema.is_mutable
                            and isinstance(
                                user.meta.get("lowering"),
                                (PointwiseLowering, ReductionLowering),
                            )
                        )
                        or user in output_stores
                        and (
                            user.args[2] is alias
                            or user.args[3] is alias
                            or alias in user.args[1]
                        )
                        or user.target is memory_ops.store
                        and user.args[2] is alias
                        and frame.role == "root"
                    ):
                        reads.append(position(user))
                    else:
                        require(False, "alias, indexed read or escaping local buffer")
        require(
            bool(updates) and positions[allocation] < min(updates),
            "initialization before updates",
        )
        require(
            all(read > max(updates) for read in reads),
            "updates before every final read/capture",
        )
    return replace(
        tree,
        output_stores=output_stores,
        indexed_reads=tuple(
            CompletedLocalRead(
                node, source, graph_id, cast("list[int]", source.args[0])[0]
            )
            for node, source, graph_id in indexed_reads
        ),
    )


def uniform_output_stores_are_writeonly(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    tree: UniformRegionTree,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Bind the approved stores to fresh or cache-specialized disjoint outputs.

    Pointer metadata alone is not an input-alias proof. Check every other
    public tensor, including inputs with no load in this region, and every
    distinct output. Repeated stores through one exact tensor retain order.
    """
    from ...language import memory_ops
    from .memory_ops import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    from .memory_ops import runtime_tensors_are_proven_disjoint
    from .register_loads import host_load_is_readonly

    if not tree.output_stores:
        return True
    targets = {
        id(value): value
        for node in tree.output_stores
        for value in (cast("Node", node.args[0]).meta["val"],)
    }
    for target in targets.values():
        storage = target.untyped_storage()
        fresh = storage in env._symbolically_exact_layout_storages
        if not fresh:
            source = env.tensor_input_source(target)
            specialization = env.runtime_input_specializations.get(
                _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
            )
            if (
                source is None
                or specialization is None
                or source not in specialization.sources
                or (
                    not allow_unbound
                    and _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
                    not in env.bound_runtime_input_specialization_results
                )
            ):
                return False
        # Fake storage inequality is only sufficient for compiler allocations.
        others = {id(value): value for value in env.input_sources}
        others.update(targets)
        for other in others.values():
            if other is target:
                continue
            other_storage = other.untyped_storage()
            if storage == other_storage:
                return False
            if (
                not fresh
                and other_storage not in env._symbolically_exact_layout_storages
                and not runtime_tensors_are_proven_disjoint(
                    env, target, other, allow_unbound=allow_unbound
                )
            ):
                return False
    local_reads = {read.node for read in tree.indexed_reads}
    return all(
        host_load_is_readonly(node, env, graphs, allow_unbound=allow_unbound)
        for frame in tree.frames
        for node in frame.graph.graph.nodes
        if node.target is memory_ops.load and node not in local_reads
    )


def uniform_region_domains(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    incoming: GatherDomainFacts,
    *,
    allow_unbound: bool = False,
) -> GatherDomainFacts:
    """Transactional logical shape transfer; no mutable entry range inheritance."""
    import operator

    from ...language import memory_ops
    from ...language import scan_ops
    from ..inductor_lowering import ReductionLowering
    from .computed_fragment import _fragment_logical_shape
    from .gather_domains import GatherDomainFacts
    from .register_loads import host_load_is_readonly

    try:
        tree = uniform_local_regions(graphs)
        if not uniform_output_stores_are_writeonly(
            env, graphs, tree, allow_unbound=allow_unbound
        ):
            return incoming
        local_reads = {read.node for read in tree.indexed_reads}
        trial = GatherDomainFacts(
            dict(incoming.shapes),
            dict(incoming.readonly_ranges),
            set(incoming.resident_whiles),
        )
        frames = {f.graph.graph_id: f for f in tree.frames}
        children = {}
        for f in tree.frames:
            if f.call is not None:
                children.setdefault(f.call, []).append(f)

        def require(ok: bool, why: str) -> None:
            if not ok:
                raise exc.InvalidConfig(f"uniform region domains: {why}")

        def shape(node: Node) -> tuple[sympy.Expr, ...] | None:
            result = _fragment_logical_shape(
                env,
                node,
                scalar_indexed_loads=True,
                tensor_indexed_loads=True,
                all_axis_reductions=True,
                proven_domains=trial.shapes,
            )
            if (
                result is None
                and isinstance(node.meta.get("lowering"), ReductionLowering)
                and _signature(node)[1] == ()
            ):
                source = node.args[0]
                # The reduction emitter supplies the neutral guard over this
                # established logical input, including physical padding.
                if isinstance(source, Node) and shape(source) is not None:
                    result = ()
            return result

        def scalar_result(node: Node) -> None:
            require(_signature(node)[1] == (), "scalar join/carry")
            trial.shapes[node] = ()
            trial.readonly_ranges.pop(node, None)

        def visit(frame: UniformRegionFrame) -> None:
            for placeholder, entry in zip(
                frame.placeholders, frame.captures, strict=True
            ):
                logical = shape(entry)
                require(logical is not None, f"capture domain {entry.name}")
                trial.shapes[placeholder] = cast("tuple[sympy.Expr, ...]", logical)
                # This slice needs no inferred ranges. Closed index expressions
                # retain the normal local proof, but mutable entry values do not.
                trial.readonly_ranges.pop(placeholder, None)
            for node in frame.graph.graph.nodes:
                if node in children:
                    for child in children[node]:
                        visit(child)
                    if node.target is _tracing_ops._while_loop:
                        body = frames[cast("int", node.args[1])]
                        for index, slot in body.carry_map:
                            require(
                                shape(body.outputs[index]) == ()
                                and shape(body.captures[slot]) == (),
                                "scalar backedge domain",
                            )
                        require(
                            shape(frames[cast("int", node.args[0])].outputs[0]) == (),
                            "scalar condition domain",
                        )
                        trial.resident_whiles.add(node)
                    else:
                        require(
                            shape(cast("Node", node.args[0])) == (),
                            "scalar branch domain",
                        )
                        info = cast(
                            "IfGraphInfo", frames[cast("int", node.args[1])].graph
                        )
                        for pair in cast(
                            "list[tuple[int | str, ...]]", info.branches_outputs
                        ):
                            for ordinal, index in enumerate(pair):
                                if type(index) is int:
                                    require(
                                        shape(
                                            frames[
                                                cast("int", node.args[1 + ordinal])
                                            ].outputs[index]
                                        )
                                        == (),
                                        "branch result domain",
                                    )
                    for item in node.users:
                        require(item.target is operator.getitem, "control projection")
                        scalar_result(item)
                        for phi in item.users:
                            if phi.target is _tracing_ops._phi:
                                scalar_result(phi)
                    continue
                if node.target is memory_ops.load and node not in local_reads:
                    require(
                        host_load_is_readonly(
                            node, env, graphs, allow_unbound=allow_unbound
                        ),
                        "readonly host storage",
                    )
                if (
                    node.op in ("placeholder", "output")
                    or node.target is _tracing_ops._host_tensor
                ):
                    continue
                if not isinstance(node.meta.get("val"), torch.Tensor):
                    continue
                if node.target is atomic_ops.atomic_add:
                    # Returned integer tickets have exactly their contribution
                    # domain; target capacity is not an output extent.
                    source = node.args[2]
                    logical = shape(source) if isinstance(source, Node) else ()
                elif node.target is scan_ops._associative_scan:
                    logical = shape(node.args[1])
                else:
                    logical = shape(node)
                require(logical is not None, f"logical domain {node.name}")
                trial.shapes[node] = cast("tuple[sympy.Expr, ...]", logical)

        visit(next(f for f in tree.frames if f.role == "root"))
        return trial
    except exc.InvalidConfig:
        return incoming
