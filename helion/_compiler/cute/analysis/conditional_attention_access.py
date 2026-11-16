"""Native operand/address facts for a structured conditional attention root.

These facts retain original load nodes, host views and row expressions. They
do not replace a clamped gather by TMA, prove host input contents, or admit an
emitter. A scheduling consumer still owns aliasing, output addresses, scalar
prelude effects and publication/retirement of its physical panels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ....language import _tracing_ops
from ....language import memory_ops
from ....language import view_ops
from .native_s2f6_operand import match_native_s2f6_operand

if TYPE_CHECKING:
    from .conditional_attention_recurrence import ConditionalAttentionRecurrence
    from .native_s2f6_match import NativeS2F6Decode


@dataclass(frozen=True)
class SourceLoad:
    node: Node
    host: Node
    indices: tuple[object, ...]
    # This is normalized compiler metadata. The unchanged host node owns the
    # real pointer/view origin; its physical storage offset is not inferred
    # from the fake tensor's (potentially rebased) offset.
    shape: tuple[object, ...]
    strides: tuple[object, ...]
    metadata_storage_offset: object
    dtype: torch.dtype


@dataclass(frozen=True)
class NativeGatherPanel:
    value: Node
    decode: NativeS2F6Decode
    raw: SourceLoad
    scale: SourceLoad
    physical_page: Node
    head: Node
    safe_row: Node
    shifted_row: Node
    row_shift: Node
    row_mask: Node
    channels: Node
    page_size: int
    depth: int
    # No old floating decoder compensation or positive-zero addition.
    group_size: int = 32
    post_scale: int = 0
    positive_zero: bool = False


@dataclass(frozen=True)
class ConditionalAttentionAccess:
    query: SourceLoad
    key: NativeGatherPanel
    value: NativeGatherPanel
    page_table: SourceLoad
    # The complete original expressions are retained, not re-derived from
    # tensor names, a benchmark shape, or assumed contiguous coordinates.
    request: object
    logical_page: Node


class _Refused(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Refused


class _AccessProof:
    def __init__(self, recurrence: ConditionalAttentionRecurrence) -> None:
        self.recurrence = recurrence
        self.captures = {
            capture.placeholder: capture.source
            for region in recurrence.regions.regions
            for capture in region.captures
        }

    def require_no_attention_state(self, value: object) -> None:
        if not isinstance(value, Node):
            raise _Refused
        # The emitter updates physical attention state rather than rebinding
        # FX placeholders. Inspect dependencies BEFORE resolving captures to
        # initial values, or a changing maximum becomes its initial -inf.
        forbidden = {
            *(carry.placeholder for carry in self.recurrence.regions.carries),
            self.recurrence.qk,
            self.recurrence.pv,
            self.recurrence.probability,
        }
        pending = [value]
        seen: set[Node] = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            _require(node not in forbidden)
            if node in self.captures:
                pending.append(self.captures[node])
            else:
                pending.extend(node.all_input_nodes)

    def resolve(self, value: object) -> Node:
        if not isinstance(value, Node):
            raise _Refused
        seen: set[Node] = set()
        while True:
            _require(value not in seen)
            seen.add(value)
            if value in self.captures:
                value = self.captures[value]
            elif value.target is _tracing_ops._new_var:
                _require(len(value.args) == 1 and not value.kwargs)
                source = value.args[0]
                if not isinstance(source, Node):
                    raise _Refused
                value = source
            else:
                return value

    def call(self, value: object, target: object, count: int) -> tuple:
        node = self.resolve(value)
        _require(
            node.op == "call_function"
            and node.target is target
            and len(node.args) == count
            and not node.kwargs
        )
        return node.args

    def tensor(self, value: object, dtype: torch.dtype) -> torch.Tensor:
        fake = self.resolve(value).meta.get("val")
        if not isinstance(fake, torch.Tensor) or fake.dtype != dtype:
            raise _Refused
        return fake

    def shape(self, value: object, dtype: torch.dtype) -> tuple[int, ...]:
        fake = self.tensor(value, dtype)
        _require(all(type(size) is int for size in fake.shape))
        return tuple(fake.shape)

    def integer_scalar(self, value: object) -> None:
        fake = self.resolve(value).meta.get("val")
        _require(
            type(fake) is int
            or isinstance(fake, torch.SymInt)
            or isinstance(fake, torch.Tensor)
            and fake.dtype in (torch.int32, torch.int64)
            and fake.ndim == 0
        )

    def load(self, value: object, dtype: torch.dtype, rank: int) -> SourceLoad:
        node = self.resolve(value)
        host, indices, extra_mask, other = self.call(node, memory_ops.load, 4)
        _require(extra_mask is None and other is None)
        _require(isinstance(indices, (tuple, list)) and len(indices) == rank)
        host = self.resolve(host)
        _require(host.target is _tracing_ops._host_tensor)
        fake = self.tensor(host, dtype)
        _require(fake.ndim == rank)
        self.tensor(node, dtype)
        return SourceLoad(
            node,
            host,
            tuple(indices),
            tuple(fake.shape),
            tuple(fake.stride()),
            fake.storage_offset(),
            dtype,
        )

    def expand(self, value: object, column: bool) -> Node:
        source, indices = self.call(value, view_ops.subscript, 2)
        _require(indices == ([slice(None), None] if column else [None, slice(None)]))
        return self.resolve(source)

    def iota(self, value: object, count: int) -> Node:
        node = self.resolve(value)
        _require(node.target is torch.ops.prims.iota.default)
        _require(node.args == (count,))
        start, step = node.kwargs.get("start"), node.kwargs.get("step")
        _require(
            type(start) is int
            and start == 0
            and type(step) is int
            and step == 1
            and node.kwargs.get("requires_grad", False) is False
        )
        _require(self.shape(node, torch.int32) == (count,))
        return node

    def panel(self, value: object) -> NativeGatherPanel:
        panel = self.resolve(value)
        mask, decoded, zero = self.call(panel, torch.ops.aten.where.self, 3)
        zero = self.resolve(zero)
        _require(zero.target is torch.ops.aten.scalar_tensor.default)
        _require(len(zero.args) == 1)
        zero_value = zero.args[0]
        if type(zero_value) is not float:
            raise _Refused
        _require(zero_value == 0.0 and math.copysign(1.0, zero_value) == 1.0)
        _require(self.shape(zero, torch.bfloat16) == ())
        decoded = self.resolve(decoded)
        proof = match_native_s2f6_operand(decoded, resolve=self.resolve)
        if proof is None:
            raise _Refused
        raw, scale = (
            self.load(proof.raw, torch.int8, 4),
            self.load(proof.scale, torch.uint8, 4),
        )
        panel_shape = self.shape(panel, torch.bfloat16)
        _require(len(panel_shape) == 2)
        rows, depth = panel_shape
        _require(rows > 0 and depth > 0 and depth % 32 == 0)
        _require(self.shape(proof.raw, torch.int8) == (rows, depth))
        _require(self.shape(proof.scale, torch.uint8) == (rows, depth))
        _require(raw.shape[1:] == (rows, raw.shape[2], depth))
        _require(scale.shape == (*raw.shape[:3], depth // 32))
        page = self.resolve(raw.indices[0])
        head = self.resolve(raw.indices[2])
        _require(self.resolve(scale.indices[0]) is page)
        _require(self.resolve(scale.indices[2]) is head)
        _require(self.shape(page, torch.int32) == ())
        self.integer_scalar(head)
        safe_row = self.expand(raw.indices[1], column=True)
        _require(self.expand(scale.indices[1], column=True) is safe_row)
        channels = self.iota(self.expand(raw.indices[3], column=False), depth)
        scale_index = self.resolve(scale.indices[3])
        _require(
            scale_index.target is torch.ops.aten.div.Tensor_mode
            and scale_index.kwargs == {"rounding_mode": "floor"}
            and len(scale_index.args) == 2
            and type(scale_index.args[1]) is int
            and scale_index.args[1] == 32
        )
        _require(self.expand(scale_index.args[0], column=False) is channels)
        lower, upper_bound = self.call(safe_row, torch.ops.aten.clamp_max.default, 2)
        shifted_row, lower_bound = self.call(lower, torch.ops.aten.clamp_min.default, 2)
        _require(type(lower_bound) is int and lower_bound == 0)
        _require(type(upper_bound) is int and upper_bound == rows - 1)
        shifted_row = self.resolve(shifted_row)
        row_iota, shift = self.call(shifted_row, torch.ops.aten.add.Tensor, 2)
        self.iota(row_iota, rows)
        shift = self.resolve(shift)
        _require(self.shape(shift, torch.int32) == ())
        row_mask = self.expand(mask, column=True)
        _require(self.shape(row_mask, torch.bool) == (rows,))
        ge, lt = self.call(row_mask, torch.ops.aten.bitwise_and.Tensor, 2)
        ge_value, ge_bound = self.call(ge, torch.ops.aten.ge.Scalar, 2)
        lt_value, lt_bound = self.call(lt, torch.ops.aten.lt.Scalar, 2)
        _require(self.resolve(ge_value) is self.resolve(lt_value) is shifted_row)
        _require(type(ge_bound) is int and ge_bound == 0)
        _require(type(lt_bound) is int and lt_bound == rows)
        return NativeGatherPanel(
            panel,
            proof,
            raw,
            scale,
            page,
            head,
            safe_row,
            shifted_row,
            shift,
            row_mask,
            channels,
            rows,
            depth,
        )

    def prove(self) -> ConditionalAttentionAccess:
        recurrence = self.recurrence
        query, transposed = recurrence.qk.args[:2]
        # Page coordinates may depend on the iteration, but the query,
        # visibility and panel operands cannot read physical attention state.
        # Re-emitting a QK/PV/probability dependency through the generic
        # fragment printer would compute a second, differently rounded value.
        for operand in (query, transposed, recurrence.pv.args[1], recurrence.visible):
            self.require_no_attention_state(operand)
        key, axes = self.call(transposed, torch.ops.aten.permute.default, 2)
        _require(axes == [1, 0])
        key_panel, value_panel = self.panel(key), self.panel(recurrence.pv.args[1])
        for name in (
            "physical_page",
            "head",
            "safe_row",
            "shifted_row",
            "row_shift",
            "row_mask",
            "channels",
            "page_size",
            "depth",
        ):
            _require(getattr(key_panel, name) == getattr(value_panel, name))
        query_load = self.load(query, torch.bfloat16, 1)
        # Query staging runs once before the page loop; only an already
        # captured outer load has the matching execution region.
        _require(query_load.node.graph is not recurrence.loop.graph)
        query_shape = self.shape(query, torch.bfloat16)
        _require(len(query_shape) == 2)
        query_rows, query_depth = query_shape
        _require(query_rows > 0 and query_depth == key_panel.depth)
        _require(
            self.shape(recurrence.qk, torch.float32)
            == (query_rows, key_panel.page_size)
        )
        _require(
            self.shape(recurrence.pv, torch.float32) == (query_rows, value_panel.depth)
        )
        page_table = self.load(key_panel.physical_page, torch.int32, 2)
        logical_page = self.resolve(page_table.indices[1])
        _require(self.shape(logical_page, torch.int32) == ())
        return ConditionalAttentionAccess(
            query_load,
            key_panel,
            value_panel,
            page_table,
            page_table.indices[0],
            logical_page,
        )


def prove_conditional_attention_access(
    recurrence: ConditionalAttentionRecurrence,
) -> ConditionalAttentionAccess | None:
    """Prove paired native panels without changing any address or mask."""
    try:
        return _AccessProof(recurrence).prove()
    except _Refused:
        return None
