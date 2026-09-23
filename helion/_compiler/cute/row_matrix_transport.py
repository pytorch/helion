"""Typed and geometric admission for a two-row matrix transport schedule.

The matrix instructions use the published m8n8 and m16n8k16 lane maps. All
addresses below are byte offsets, independent of an example, tensor name,
pointer value or generated source. The only aliased publication is the final
warp-collective matrix store: its four replicas have identical ordered MMA
inputs and the same pure typed epilogue. Global stores have unique owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...language import memory_ops
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
from .memory_ops import tensor_has_specialized_base_alignment
from .row_resident import RowResidentPlan
from .row_resident import RowStore
from .row_resident import _call
from .row_resident import _direct_load
from .row_resident import _value

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction

ROW_MATRIX_MODE = "warp_rows2_matrix"
ROW_MATRIX_BARRIERS = (
    "first_inputs_ready",
    "first_readers_retired",
    "middle_scatter_ready",
    "middle_global_ready",
    "second_inputs_ready",
    "second_readers_retired",
    "final_scatter_ready",
    "final_global_ready",
)


@dataclass(frozen=True)
class RowMatrixLayout:
    """One complete static N/K tile, eight column warps, two logical rows."""

    columns: int
    reduction: int

    @classmethod
    def create(cls, columns: int, reduction: int) -> RowMatrixLayout | None:
        if (
            type(columns) is not int
            or type(reduction) is not int
            or columns % 64
            or columns // 64 not in (1, 2, 4)
            or reduction % 32
            or reduction // 32 not in (1, 2, 4, 8)
        ):
            return None
        return cls(columns, reduction)

    @property
    def column_packets(self) -> int:
        # One m16n8 result per warp and packet; 8 warps cover 64 columns.
        return self.columns // 64

    @property
    def weight_packets(self) -> int:
        # Each of 256 threads transfers eight contiguous FP16 values.
        return self.columns * self.reduction // (256 * 8)

    @staticmethod
    def compact_offset(row: int, column: int, width: int) -> int:
        # Width is a power of two >=32. Move the row bit into byte bit4;
        # preserve the row, complete 16-byte packet and low address bits.
        return 2 * (row * width + column) ^ (row * 16)

    def weight_offset(self, reduction: int, column: int) -> int:
        # The row bits belong to K, not to a fixed byte stride. This couples
        # the 128-bit producer to the compact transposed matrix consumer.
        return 2 * (reduction * self.columns + column) ^ ((reduction & 7) * 16)

    @staticmethod
    def linear_matrix_address(thread: int, width: int) -> int:
        # The first eight lanes supply one aligned m8n8 row address each.
        # Other lanes repeat valid addresses but do not supply row pointers.
        warp, lane = divmod(thread, 32)
        flat = warp * 64 + (lane % 8) * 8
        row, column = divmod(flat, width)
        return RowMatrixLayout.compact_offset(row, column, width)

    def a_matrix_address(self, thread: int, packet: int) -> int:
        lane = thread % 32
        return self.compact_offset(
            lane % 2, packet * 32 + lane // 8 * 8, self.reduction
        )

    def b_matrix_address(
        self, thread: int, reduction_packet: int, column_packet: int
    ) -> int:
        warp, lane = divmod(thread, 32)
        return self.weight_offset(
            reduction_packet * 32 + lane, warp * 8 + column_packet * 64
        )

    def result_coordinate(self, thread: int, packet: int, half: int) -> tuple[int, int]:
        warp, lane = divmod(thread, 32)
        return lane // 4 % 2, warp * 8 + packet * 64 + 2 * (lane % 4) + half

    def middle_offset(self, row: int, column: int) -> int:
        # Pair the two 32-column halves within each 64-column packet. This
        # is a bit permutation followed by an invertible row-bit XOR.
        permuted = column // 64 * 64 + (column % 32) * 2 + column // 32 % 2
        return 2 * (row * self.columns + permuted) ^ (row * 4)

    def result_matrix_address(self, thread: int) -> int:
        warp, lane = divmod(thread, 32)
        return self.compact_offset(
            lane % 2, warp * 8 + lane // 8 % self.column_packets * 64, self.columns
        )


@dataclass(frozen=True)
class RowMatrixPlan:
    row: RowResidentPlan
    layouts: tuple[RowMatrixLayout, RowMatrixLayout]
    materialized: RowStore
    product: RowStore
    product_input: str
    materialized_is_left_operand: bool
    # Exact FX identity, including the existing FP16 materialization boundary.
    materialized_value: torch.fx.Node
    product_value: torch.fx.Node

    @property
    def weight_bytes(self) -> int:
        return max(2 * p.columns * p.reduction for p in self.layouts)

    @property
    def shared_bytes(self) -> int:
        # Weight arena is retired before the middle/final redistribution.
        # A occupies a disjoint compact arena throughout each contraction.
        return self.weight_bytes + 4 * max(self.row.reductions)


def _stored_values(host: HostFunction, index: int) -> dict[str, torch.fx.Node]:
    ir = host.device_ir
    root = ir.graphs[ir.root_ids[index]]
    result = {}
    for node in root.graph.nodes:
        if node.target is memory_ops.store:
            source, _, value, _ = node.args
            assert isinstance(source, torch.fx.Node)
            assert isinstance(source.args[0], str) and isinstance(value, torch.fx.Node)
            result[source.args[0]] = value
    return result


def prove_row_matrix_transport(
    env: CompileEnvironment,
    host: HostFunction,
    row: RowResidentPlan,
    *,
    binding: bool = False,
) -> RowMatrixPlan | None:
    """Refine an already complete row proof; never admit partial graph work."""
    capability = env.config_spec.target_device_capability
    if capability is None or capability[0] < 9:
        return None  # stmatrix requires sm90+, independently of warp MMA.
    first = RowMatrixLayout.create(row.columns[0], row.reductions[0])
    second = RowMatrixLayout.create(row.columns[1], row.reductions[1])
    if (
        first is None
        or second is None
        or row.columns[0] != row.reductions[1]
        or len(row.stages[1].stores) != 2
    ):
        return None
    values = _stored_values(host, 1)
    # Initial post-materialization family: an exact FP16 product of the
    # stored FP16 carrier and one direct, same-coordinate FP16 input. This
    # selects a dataflow/precision boundary, not a particular activation.
    for materialized, product in (
        row.stages[1].stores,
        tuple(reversed(row.stages[1].stores)),
    ):
        base = values[materialized.name]
        value = values[product.name]
        if materialized.chain.auxiliary_tensor_loads:
            continue
        base_tensor = _value(base)
        if base_tensor is None or base_tensor.dtype is not torch.float16:
            continue
        if _call(value, torch.ops.prims.convert_element_type.default, 2):
            if value.args[1] is not torch.float16:
                continue
            value = value.args[0]
        value_tensor = _value(value)
        if (
            not _call(value, torch.ops.aten.mul.Tensor, 2)
            or value_tensor is None
            or value_tensor.dtype is not torch.float16
        ):
            continue
        assert isinstance(value, torch.fx.Node)
        lhs, rhs = value.args
        if lhs is base:
            auxiliary, base_left = rhs, True
        elif rhs is base:
            auxiliary, base_left = lhs, False
        else:
            continue
        stage = row.stages[1]
        loaded = _direct_load(env, auxiliary, (stage.row_axis, stage.column_axis))
        if loaded is None:
            continue
        name, tensor = loaded
        if (
            name not in host.params.arguments
            or host.params.arguments[name] is not tensor
        ):
            continue
        # The packed product adds a vector read of the auxiliary input. Reuse
        # the same dispatch-key alignment facts as the contraction operands;
        # a contiguous offset view need not have an aligned base pointer.
        if binding:
            actual = env.runtime_arg_values_by_name.get(name)
            source = env.tensor_input_source(tensor)
            specialization = env.runtime_input_specializations.get(
                _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
            )
            if (
                not isinstance(actual, torch.Tensor)
                or actual.data_ptr() % 4
                or source is None
                or specialization is None
                or source not in specialization.sources
            ):
                continue
        elif not tensor_has_specialized_base_alignment(env, tensor, 4):
            continue
        plan = RowMatrixPlan(
            row, (first, second), materialized, product, name, base_left, base, value
        )
        if plan.shared_bytes > 48 * 1024 or plan.weight_bytes % 1024:
            continue
        return plan
    return None
