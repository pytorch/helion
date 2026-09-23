"""Typed signed fields and their current, vectorized byte-load carriers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ..host_function import HostFunction
from .promote_output_axis import _access_tensor
from .promote_output_axis import _fresh_tensors
from .promote_output_axis import _ordinary_operation

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState

KEY = "cute_signed_bitfield_bf16"
_SIGNED_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
_CASTS = (torch.ops.aten._to_copy.default, torch.ops.prims.convert_element_type.default)
_SHIFTS = (
    torch.ops.aten.__rshift__.Scalar,
    torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
    torch.ops.aten.bitwise_right_shift.Tensor,
)
_ANDS = (torch.ops.aten.bitwise_and.Scalar, torch.ops.aten.bitwise_and.Tensor)
_SUBS = (torch.ops.aten.sub.Scalar, torch.ops.aten.sub.Tensor)
_GES = (torch.ops.aten.ge.Scalar, torch.ops.aten.ge.Tensor)


@dataclass(frozen=True)
class SignedByteField:
    load: Node
    bit_offset: int
    bit_width: int
    nodes: frozenset[Node]
    signed: bool = True

    @property
    def bounds(self) -> tuple[int, int]:
        if self.signed:
            return -(1 << (self.bit_width - 1)), (1 << (self.bit_width - 1)) - 1
        return 0, (1 << self.bit_width) - 1


def _dtype(node: object) -> torch.dtype | None:
    value = node.meta.get("val") if isinstance(node, Node) else None
    return value.dtype if isinstance(value, torch.Tensor) else None


def _fits(dtype: torch.dtype | None, low: int, high: int) -> bool:
    return dtype in _SIGNED_DTYPES and (
        torch.iinfo(dtype).min <= low <= high <= torch.iinfo(dtype).max
    )


def _cast_input(node: Node) -> Node | None:
    if node.target not in _CASTS or not node.args or not isinstance(node.args[0], Node):
        return None
    if node.target is torch.ops.prims.convert_element_type.default:
        if len(node.args) != 2 or node.kwargs or node.args[1] is not _dtype(node):
            return None
    elif (
        len(node.args) != 1
        or set(node.kwargs) != {"dtype"}
        or node.kwargs["dtype"] is not _dtype(node)
    ):
        return None
    return node.args[0]


def _binary(node: Node, targets: tuple[object, ...]) -> tuple[Node, int] | None:
    if (
        node.target not in targets
        or len(node.args) != 2
        or not isinstance(node.args[0], Node)
        or type(node.args[1]) is not int
        or node.kwargs not in ({}, {"alpha": 1})
        or (node.kwargs and node.target not in _SUBS)
    ):
        return None
    return node.args[0], node.args[1]


def _integer_field(node: Node) -> SignedByteField | None:
    dtype = _dtype(node)
    if dtype not in _SIGNED_DTYPES or node.op != "call_function":
        return None
    if node.target is memory_ops.load:
        tensor = _access_tensor(node)
        if tensor is not None and tensor.dtype is torch.int8 and dtype is torch.int8:
            return SignedByteField(node, 0, 8, frozenset((node,)))
        return None
    if (source := _cast_input(node)) is not None:
        field = _integer_field(source)
        if field is not None and _fits(dtype, *field.bounds):
            return replace(field, nodes=field.nodes | {node})
        return None
    if (shift := _binary(node, _SHIFTS)) is not None:
        source, offset = shift
        field = _integer_field(source)
        if (
            field is not None
            and field.signed
            and dtype is _dtype(source)
            and 0 <= offset < field.bit_width
        ):
            return replace(
                field,
                bit_offset=field.bit_offset + offset,
                bit_width=field.bit_width - offset,
                nodes=field.nodes | {node},
            )
        return None
    if (masked := _binary(node, _ANDS)) is not None:
        source, mask = masked
        field = _integer_field(source)
        width = mask.bit_length()
        if (
            field is not None
            and 1 <= width <= field.bit_width
            and mask == (1 << width) - 1
            and dtype is _dtype(source)
            and _fits(dtype, 0, mask)
        ):
            return replace(
                field, bit_width=width, signed=False, nodes=field.nodes | {node}
            )
        return None
    if (
        node.target is torch.ops.aten.where.self
        and len(node.args) == 3
        and not node.kwargs
    ):
        condition, negative, raw = node.args
        if (
            not isinstance(condition, Node)
            or not isinstance(negative, Node)
            or not isinstance(raw, Node)
        ):
            return None
        field = _integer_field(raw)
        compare = _binary(condition, _GES)
        subtract = _binary(negative, _SUBS)
        if (
            field is not None
            and not field.signed
            and compare == (raw, 1 << (field.bit_width - 1))
            and subtract == (raw, 1 << field.bit_width)
            and _dtype(condition) is torch.bool
            and dtype is _dtype(raw) is _dtype(negative)
            # The *whole* subtraction, including the unselected branch,
            # must have the same non-wrapping signed-integer semantics.
            and _fits(dtype, -(1 << field.bit_width), (1 << field.bit_width) - 1)
        ):
            return replace(
                field, signed=True, nodes=field.nodes | {condition, negative, node}
            )
    return None


def match_signed_byte_field(value: object) -> SignedByteField | None:
    """Recheck actual FX values; copied-node metadata is never a proof."""
    if not isinstance(value, Node) or _dtype(value) is not torch.bfloat16:
        return None
    source = _cast_input(value)
    field = _integer_field(source) if source is not None else None
    if (
        field is None
        or not field.signed
        or any(node.graph is not value.graph for node in field.nodes)
    ):
        return None
    return replace(field, nodes=field.nodes | {value})


def has_signed_byte_field(graph: torch.fx.Graph) -> bool:
    return any(
        node.target is memory_ops.store
        and len(node.args) >= 3
        and match_signed_byte_field(node.args[2]) is not None
        for node in graph.nodes
    )


def mask_implies(output_mask: str | None, input_mask: str | None) -> bool:
    """Exact conjunction containment, with no range/name-based simplification."""

    def terms(mask: str | None) -> set[str]:
        if mask is None:
            return set()

        def flatten(node: ast.AST) -> set[str]:
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                return set().union(*(flatten(value) for value in node.values))
            return {ast.dump(node)}

        return flatten(ast.parse(mask, mode="eval").body)

    return terms(input_mask).issubset(terms(output_mask))


@dataclass(frozen=True)
class SignedBytePacket:
    load: Node
    env: object
    strategy: object
    block_id: int
    width: int
    tensor: torch.Tensor
    carrier: str
    definition: ast.Assign
    lane_body: list[ast.AST]
    vloop: ast.For
    statements: list[ast.AST]
    loop_ids: tuple[tuple[int, tuple[int, ...]], ...]
    lane_index: str
    load_mask: str | None


def _loop_ids(state: CodegenState) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (block, tuple(map(id, stack)))
        for block, stack in state.codegen.active_device_loops.items()
        if stack
    )


@dataclass(frozen=True)
class SignedByteSite:
    """Lowering-site facts for one load or store.

    Grid-owned vector operations are emitted only when the root body is
    wrapped. By then the statement stack, branch path, and active loops no
    longer describe the original site, so callers capture these facts while
    lowering and hand them to the deferred packet record and lookup.
    """

    statements: list[ast.AST]
    loop_ids: tuple[tuple[int, tuple[int, ...]], ...]
    in_branch: bool
    grid_only: bool


def signed_byte_site(state: CodegenState) -> SignedByteSite:
    from ..tile_strategy import DeviceGridState

    return SignedByteSite(
        state.codegen.statements_stack[-1],
        _loop_ids(state),
        bool(state.codegen._cute_branch_path),
        all(
            isinstance(loop, DeviceGridState)
            for stack in state.codegen.active_device_loops.values()
            for loop in stack
        ),
    )


class _TileUnrollStrategy(Protocol):
    _cute_lane_body_by_block: dict[int, list[ast.AST]]
    _cute_lane_vloop_by_block: dict[int, ast.For]
    _cute_lane_axis_pos_by_block: dict[int, int]


def _lane_context(
    strategy: object, block_id: int
) -> tuple[list[ast.AST], ast.For, int]:
    # The existing tile_unroll admission supplies this protocol. The common
    # strategy base also serves scalar/reduction paths without these fields.
    context = cast("_TileUnrollStrategy", strategy)
    return (
        context._cute_lane_body_by_block[block_id],
        context._cute_lane_vloop_by_block[block_id],
        context._cute_lane_axis_pos_by_block[block_id],
    )


def record_signed_byte_packet(
    state: CodegenState,
    strategy: object,
    block_id: int,
    tensor: torch.Tensor,
    width: int,
    index_exprs: list[str],
    carrier: str,
    definition: ast.stmt | None,
    mask: str | None,
    site: SignedByteSite | None = None,
) -> None:
    """Retain only the already admitted load, in this codegen invocation."""
    node = state.fx_node
    if site is None:
        site = signed_byte_site(state)
    if (
        state.config.config.get(KEY) is not True
        or tensor.dtype is not torch.int8
        or node is None
        or width not in (2, 4, 8)
        or site.in_branch
        or not site.grid_only
    ):
        return
    lane_body, vloop, lane_pos = _lane_context(strategy, block_id)
    records = state.device_function.cute_state.signed_byte_packets
    if definition is None:
        # Cache reuse retains the original definition object; never recognize
        # a scalar AST expression or invent a carrier from an identifier.
        matches = [
            packet.definition
            for packet in records.values()
            if packet.carrier == carrier
            and packet.lane_body is lane_body
            and packet.vloop is vloop
        ]
        if not matches or any(item is not matches[0] for item in matches):
            return
        definition = matches[0]
    if not isinstance(definition, ast.Assign):
        return
    records[node] = SignedBytePacket(
        node,
        state.env,
        strategy,
        block_id,
        width,
        tensor,
        carrier,
        definition,
        lane_body,
        vloop,
        site.statements,
        site.loop_ids,
        index_exprs[lane_pos],
        mask,
    )


def _no_alias_effects(state: CodegenState, graph: torch.fx.Graph) -> bool:
    from ..compile_environment import CompileEnvironment
    from .memory_ops import runtime_tensors_are_proven_disjoint

    if not all(
        _ordinary_operation(node)
        and node.target
        not in (
            _tracing_ops._for_loop,
            _tracing_ops._for_loop_step,
            _tracing_ops._if,
            _tracing_ops._phi,
        )
        for node in graph.nodes
    ):
        return False
    fresh = {
        tensor.untyped_storage()._cdata
        for tensor in _fresh_tensors(HostFunction.current())
    }
    loads = [node for node in graph.nodes if node.target is memory_ops.load]
    stores = [node for node in graph.nodes if node.target is memory_ops.store]
    env = CompileEnvironment.current()
    for store in stores:
        output = _access_tensor(store)
        if output is None:
            return False
        for load in loads:
            source = _access_tensor(load)
            if source is None:
                return False
            output_storage = output.untyped_storage()._cdata
            if (
                output_storage in fresh
                and output_storage != source.untyped_storage()._cdata
            ) or runtime_tensors_are_proven_disjoint(env, output, source):
                continue
            return False
    return True


def packed_store_value(
    state: CodegenState,
    strategy: object,
    block_id: int,
    width: int,
    tensor: torch.Tensor,
    index_exprs: list[str],
    mask: str | None,
    site: SignedByteSite | None = None,
) -> str | None:
    """Use the packet only at the original vector flush, with identical lanes."""
    node = state.fx_node
    if site is None:
        site = signed_byte_site(state)
    if (
        state.config.config.get(KEY) is not True
        or tensor.dtype is not torch.bfloat16
        or node is None
        or site.in_branch
        or (field := match_signed_byte_field(node.args[2])) is None
        or (
            packet := state.device_function.cute_state.signed_byte_packets.get(
                field.load
            )
        )
        is None
    ):
        return None
    lane_body, vloop, lane_pos = _lane_context(strategy, block_id)
    if (
        packet.env is not state.env
        or packet.strategy is not strategy
        or packet.load.graph is not node.graph
        or not field.nodes.issubset(state.env)
        or packet.block_id != block_id
        or packet.width != width
        or packet.lane_body is not lane_body
        or packet.vloop is not vloop
        or packet.statements is not site.statements
        or packet.loop_ids != site.loop_ids
        or packet.lane_index != index_exprs[lane_pos]
        or not mask_implies(mask, packet.load_mask)
        or packet.definition not in packet.lane_body
        or packet.vloop not in packet.lane_body
        or packet.lane_body.index(packet.definition)
        >= packet.lane_body.index(packet.vloop)
    ):
        return None
    # The compiler allocated the name, but also reject a mutated/reused record.
    writes = [
        item
        for statement in packet.lane_body
        for item in ast.walk(statement)
        if isinstance(item, ast.Name)
        and isinstance(item.ctx, ast.Store)
        and item.id == packet.carrier
    ]
    if (
        len(packet.definition.targets) != 1
        or not isinstance(packet.definition.targets[0], ast.Name)
        or writes != [packet.definition.targets[0]]
        or not _no_alias_effects(state, node.graph)
    ):
        return None
    return (
        f"_cute_signed_bitfield_to_bf16_packed({packet.carrier}, "
        f"{field.bit_offset}, {field.bit_width}, {width})"
    )
