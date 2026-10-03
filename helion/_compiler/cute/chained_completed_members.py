"""Non-authorizing completed-member store candidates and native ownership.

No stage completion, storage omission, lifetime, synchronization or publication
is implied. A later stage-owned binding must prove those independently. Maps
are evaluated from the original full-M128 CuTe FP32 load/identity partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from . import chained_matmul as chain
from .chained_register_islands import _freeze
from .chained_store_expression import _node_facts
from .chained_tmem_segments import validate_tmem_segment
from .contraction_region import _domain

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan


@dataclass(frozen=True)
class CompletedMemberMap:
    full_shape: tuple[int, int]
    logical_shape: tuple[int, int]
    offset: int
    width: int
    transpose: bool
    input_dtype: torch.dtype
    # Full-group register-slot order, including slots outside this member.
    coordinates: tuple[tuple[tuple[int, int], ...], ...]
    tmem_offsets: tuple[tuple[int, ...], ...]
    selected_slots: tuple[tuple[int, ...], ...]

    @property
    def same_store_order(self) -> bool:
        if (
            not len(self.coordinates)
            == len(self.tmem_offsets)
            == len(self.selected_slots)
            == 128
        ):
            return False
        m, n = self.logical_shape
        for thread, slots in enumerate(self.selected_slots):
            actual = []
            for slot in slots:
                row, column = self.coordinates[thread][slot]
                column -= self.offset
                logical = (column, row) if self.transpose else (row, column)
                actual.append(logical[0] * n + logical[1])
            if actual != list(range(thread, m * n, 128)):
                return False
        return True


@lru_cache(maxsize=64)
def _native_map(
    full_shape: tuple[int, int],
    logical_shape: tuple[int, int],
    offset: int,
    width: int,
    transpose: bool,
    dtype: torch.dtype,
    operand_source: str,
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    kind = cutlass.BFloat16 if dtype is torch.bfloat16 else cutlass.Float16
    coordinates, addresses, selected = [], [], []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                kind,
                kind,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                full_shape,
                tcgen05.OperandSource.SMEM
                if operand_source == "SMEM"
                else tcgen05.OperandSource.TMEM,
            )
            layout = mma.make_fragment_C(mma.partition_shape_C(full_shape)).layout
            accumulator = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem), layout
            )
            # Same M128 load operation as chained_result_transport.load_operation.
            repetition = min(32, full_shape[1] & -full_shape[1])
            copy = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(repetition)),
                    cutlass.Float32,
                ),
                accumulator,
            )
            identity = mma.get_slice(0).partition_C(
                cute.make_identity_tensor(full_shape)
            )
            source_values = int(cute.size(copy.layout_src_tv.shape[1]))
            destination_values = int(cute.size(copy.layout_dst_tv.shape[1]))
            atom_threads = int(cute.size(copy.thr_id))
            if int(copy.layout_src_tv.stride[0]) != 0:
                raise ValueError("native source is not warp-uniform")
            atom_source = {
                int(copy.layout_src_tv((0, i))): i for i in range(source_values)
            }
            atom_slots = tuple(
                tuple(
                    atom_source[int(copy.layout_dst_tv((lane, i)))]
                    for i in range(destination_values)
                )
                for lane in range(atom_threads)
            )
            for thread in range(128):
                local = copy.get_slice(thread)
                source = local.partition_S(accumulator)
                source_identity = local.partition_S(identity)
                coords = local.partition_D(identity)
                count = int(cute.size(coords))
                if count * source_values != int(cute.size(source)) * destination_values:
                    raise ValueError("native source and identity fragment differ")
                points = tuple(
                    (int(coords[i][0]), int(coords[i][1])) for i in range(count)
                )
                lane = thread % atom_threads
                # The TMEM source is warp-uniform and includes every lane's
                # words. Resolve the actual atom's source/destination TV maps;
                # source and destination fragment sizes are intentionally unequal.
                expected, origins = [], []
                for slot, point in enumerate(points):
                    source_slot = (
                        atom_slots[lane][slot % destination_values]
                        + (slot // destination_values) * source_values
                    )
                    source_point = tuple(int(x) for x in source_identity[source_slot])
                    if source_point != point:
                        raise ValueError(
                            "source register order disagrees with identity"
                        )
                    address = int(layout(((point[0], point[1]), 0, 0)))
                    expected.append(address)
                    origins.append(address - int(source.layout(source_slot)))
                if len(set(origins)) != 1:
                    raise ValueError("source partition changes its native origin")
                slots = []
                for i, (row, column) in enumerate(points):
                    if not offset <= column < offset + width:
                        continue
                    logical = (
                        (column - offset, row) if transpose else (row, column - offset)
                    )
                    if logical[0] < logical_shape[0] and logical[1] < logical_shape[1]:
                        slots.append(i)
                coordinates.append(points)
                addresses.append(tuple(expected))
                selected.append(tuple(slots))
        if not module.operation.verify():
            raise ValueError("invalid native completed-member map")
    return tuple(coordinates), tuple(addresses), tuple(selected)


def plan_completed_member_map(
    full_shape: tuple[int, int],
    logical_shape: tuple[int, int],
    offset: int,
    width: int,
    transpose: bool,
    dtype: torch.dtype,
    *,
    operand_source: str = "SMEM",
) -> CompletedMemberMap | None:
    """Full native slot map only when original scalar owner/order is unchanged."""
    try:
        validate_tmem_segment(full_shape, offset, width)
    except ValueError:
        return None
    if (
        type(full_shape) is not tuple
        or type(logical_shape) is not tuple
        or type(transpose) is not bool
        or len(logical_shape) != 2
        or any(type(x) is not int or x <= 0 for x in logical_shape)
        or dtype not in (torch.bfloat16, torch.float16)
        or type(operand_source) is not str
        or operand_source not in ("SMEM", "TMEM")
    ):
        return None
    physical = logical_shape[::-1] if transpose else logical_shape
    if physical[0] > 128 or physical[1] > width:
        return None
    result = CompletedMemberMap(
        full_shape,
        logical_shape,
        offset,
        width,
        transpose,
        dtype,
        *_native_map(
            full_shape, logical_shape, offset, width, transpose, dtype, operand_source
        ),
    )
    cells = [
        result.coordinates[t][i]
        for t, slots in enumerate(result.selected_slots)
        for i in slots
    ]
    if len(set(cells)) != math.prod(logical_shape) or not result.same_store_order:
        return None
    return result


def _plan_facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    region = plan.region
    assert region is not None
    return (
        plan.dots,
        plan.shapes,
        plan.dtype,
        plan.threads,
        plan.strategy,
        tuple(
            (
                g.stages,
                tuple((x.logical, x.transpose, x.native_rows) for x in g.geometries),
            )
            for g in plan.contraction_groups or ()
        ),
        plan.warp_mma_stages,
        region,
        tuple(region.graph.nodes),
        region.nodes,
        region.graph_id,
        region.stores,
        region.live_outs,
        tuple(_freeze(vars(spec)) for spec in region.contractions),
        tuple(
            (c.input_index, c.output_index, c.input, c.output) for c in region.carries
        ),
        _node_facts(tuple(region.graph.nodes)),
    )


def _pointwise(node: Node) -> bool:
    if node.target in (
        _tracing_ops._new_var,
        torch.ops.prims.convert_element_type.default,
    ):
        return True
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
        and chain._pointwise_inputs(node) is not None
    )


@dataclass(frozen=True)
class CompletedMemberCandidate:
    plan: ChainedMatmulPlan
    group: ContractionGroup
    member: int
    node: Node
    store: Node
    descendants: tuple[Node, ...]
    store_inputs: tuple[Node, ...]
    native: CompletedMemberMap
    shapes: tuple[tuple[Node, tuple[int, ...]], ...]
    facts: tuple[object, ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.plan,
            self.group,
            self.member,
            self.node,
            self.store,
            self.descendants,
            self.store_inputs,
            _freeze(vars(self.native)),
            self.shapes,
            self.facts,
        )

    def matches(
        self, plan: ChainedMatmulPlan, shapes: Mapping[Node, tuple[int, ...]]
    ) -> bool:
        return (
            plan is self.plan
            and self._selection == self._fields()
            and plan.region is not None
            and self.facts == _plan_facts(plan)
            and all(shapes.get(n) == s for n, s in self.shapes)
        )


def plan_completed_member_store(
    plan: ChainedMatmulPlan,
    group: ContractionGroup,
    member: int,
    store: Node,
    shapes: Mapping[Node, tuple[int, ...]],
) -> CompletedMemberCandidate | None:
    """Discover exclusive original pointwise uses; grant no execution authority."""
    region = plan.region
    if (
        region is None
        or tuple(region.graph.nodes) != region.nodes
        or plan.strategy != "tcgen05_tmem"
        or group not in (plan.contraction_groups or ())
        or not group.stages
        or group.stages[-1] != len(plan.dots) - 1
        or group.stages != tuple(range(group.stages[0], group.stages[-1] + 1))
        or type(member) is not int
        or member not in group.stages
        or any(index in plan.warp_mma_stages for index in group.stages)
        or store not in region.stores
        or store.target is not memory_ops.store
        or store.graph is not region.graph
    ):
        return None
    resolved = []
    for item in region.nodes:
        value = item.meta.get("val")
        if isinstance(value, torch.Tensor):
            shape = shapes.get(item)
            if (
                type(shape) is not tuple
                or len(shape) != value.ndim
                or any(type(x) is not int or x < 0 for x in shape)
                or any(
                    type(old) is int and old != new
                    for old, new in zip(_domain(value), shape, strict=True)
                )
            ):
                return None
            resolved.append((item, shape))
    node = plan.dots[member]
    geometry = group.geometries[group.stages.index(member)]
    shape = shapes[node]
    if (
        node.meta["val"].dtype is not torch.float32
        or shape != geometry.logical[:2]
        or geometry.logical != plan.shapes[member]
        or shapes.get(cast("Node", store.args[2])) != shape
        or any(node is c.output for c in region.carries)
    ):
        return None
    descendants: set[Node] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current in descendants:
            continue
        descendants.add(current)
        if not current.users:
            return None
        for user in current.users:
            if user is store:
                continue
            if (
                user.graph is not region.graph
                or not _pointwise(user)
                or shapes.get(user) != shape
            ):
                return None
            pending.append(user)
    if cast("Node", store.args[2]) not in descendants:
        return None
    # Reading C through an index or mask is not this bounded value-only slice.
    pending = [x for x in store.all_input_nodes if x is not store.args[2]]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        if current in descendants:
            return None
        visited.add(current)
        pending.extend(current.all_input_nodes)
    native = plan_completed_member_map(
        group.physical[:2],
        cast("tuple[int, int]", shape),
        group.offsets[group.stages.index(member)],
        geometry.physical[1],
        geometry.transpose,
        plan.operand_dtype(member),
    )
    if native is None:
        return None
    result = CompletedMemberCandidate(
        plan,
        group,
        member,
        node,
        store,
        tuple(n for n in region.nodes if n in descendants),
        tuple(store.all_input_nodes),
        native,
        tuple(resolved),
        _plan_facts(plan),
        (),
    )
    return CompletedMemberCandidate(
        plan,
        group,
        member,
        node,
        store,
        result.descendants,
        result.store_inputs,
        native,
        result.shapes,
        result.facts,
        result._fields(),
    )
