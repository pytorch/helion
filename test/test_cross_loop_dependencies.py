from __future__ import annotations

import dataclasses
from typing import Literal

import torch

import helion
from helion._compiler.cross_loop_scheduler import CROSS_LOOP_NUM_WORKERS_CONFIG
from helion._compiler.cross_loop_scheduler import AccessProgramPoint
from helion._compiler.cross_loop_scheduler import CountedEventContribution
from helion._compiler.cross_loop_scheduler import CountedEventPlan
from helion._compiler.cross_loop_scheduler import CountedEventUse
from helion._compiler.cross_loop_scheduler import InstantiatedEventContribution
from helion._compiler.cross_loop_scheduler import InstantiatedEventGraph
from helion._compiler.cross_loop_scheduler import InstantiatedEventUse
from helion._compiler.cross_loop_scheduler import InstantiatedKeyedEvent
from helion._compiler.cross_loop_scheduler import InstantiatedTaskFamily
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.cross_loop_scheduler import _compress_task_to_key
from helion._compiler.cross_loop_scheduler import build_baseline_worker_schedule
from helion._compiler.cross_loop_scheduler import build_cross_loop_schedule
from helion._compiler.cross_loop_scheduler import canonicalize_ready_events
from helion._compiler.cross_loop_scheduler import canonicalize_task_readiness
from helion._compiler.cross_loop_scheduler import choose_counted_events
from helion._compiler.cross_loop_scheduler import derive_local_triggers
from helion._compiler.cross_loop_scheduler import instantiate_event_graph
from helion._compiler.cross_loop_scheduler import lower_counted_events
from helion._compiler.cross_loop_scheduler import order_local_contributors_by_key
from helion._compiler.cross_loop_scheduler import validate_worker_schedule
from helion._compiler.tile_dependency import AllocationRegion
from helion._compiler.tile_dependency import EventContribution
from helion._compiler.tile_dependency import KeyedEvent
from helion._compiler.tile_dependency import LogicalTaskAxis
from helion._compiler.tile_dependency import TaskFamily
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import TileDependencyGraph
from helion._compiler.tile_dependency import TileDependencyKind
from helion._compiler.tile_dependency import allocation_regions_may_overlap
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._compiler.tile_dependency import predecessor_task_ids
from helion._compiler.tile_dependency import prove_uniform_task_partition
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion.autotuner.config_fragment import IntegerFragment
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def grouped_affine_chain(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    group_size: int,
    reverse_groups: hl.constexpr,
) -> torch.Tensor:
    m, hidden = x.size()
    _, twice_intermediate = w13.size()
    intermediate = twice_intermediate // 2
    _, out_features = w2.size()
    hl.specialize(group_size)
    groups = intermediate // group_size
    gate_up = torch.empty((m, twice_intermediate), dtype=x.dtype, device=x.device)
    activation = torch.empty((m, intermediate), dtype=x.dtype, device=x.device)
    activation_scale = torch.empty((m, groups), dtype=torch.float32, device=x.device)
    out = torch.empty((m, out_features), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, twice_intermediate]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden, block_size=32):
            acc = torch.addmm(acc, x[tile_m, tile_k], w13[tile_k, tile_n])
        gate_up[tile_m, tile_n] = acc.to(x.dtype)

    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, group_size]):
        if reverse_groups:
            source_group = groups - 1 - tile_i.id
            source_i = source_group * group_size + hl.arange(group_size)
        else:
            source_i = tile_i
        gate = gate_up[tile_m, source_i].to(torch.float32)
        up = gate_up[tile_m, source_i + intermediate].to(torch.float32)
        activated = gate * up
        map_scale = torch.amax(torch.abs(activated), dim=-1) + 1
        activation[tile_m, tile_i] = activated.to(x.dtype)
        activation_scale[tile_m, tile_i.id] = map_scale

    for tile_m, tile_n in hl.tile([m, out_features]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate, block_size=group_size):
            values = activation[tile_m, tile_k].to(torch.float32)
            consumer_scale = activation_scale[tile_m, tile_k.id].to(torch.float32)
            acc = torch.addmm(
                acc,
                values * consumer_scale[:, None],
                w2[tile_k, tile_n].to(torch.float32),
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def cartesian_affine_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for tile_batch, tile_width in hl.tile([batch, width]):
        tmp[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        out[tile_batch, tile_width] = tmp[tile_batch, tile_width] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def size_one_view_chain(x: torch.Tensor) -> torch.Tensor:
    heads, width = x.size()
    tmp = torch.empty_like(x)
    viewed = tmp.unsqueeze(0)
    out = torch.empty_like(viewed)

    for tile_head in hl.tile(heads):
        tmp[tile_head, :] = x[tile_head, :] + 1
    for tile_batch, tile_head, tile_width in hl.tile([1, heads, width]):
        out[tile_batch, tile_head, tile_width] = (
            viewed[tile_batch, tile_head, tile_width] * 2
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def three_way_affine_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    output_width = width // 3
    tmp = torch.empty_like(x)
    out = torch.empty((batch, output_width), dtype=x.dtype, device=x.device)

    for tile_batch, tile_width in hl.tile([batch, width]):
        tmp[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, output_width]):
        out[tile_batch, tile_width] = (
            tmp[tile_batch, tile_width]
            + tmp[tile_batch, tile_width + output_width]
            + tmp[tile_batch, tile_width + 2 * output_width]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def counted_event_chain(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.size()
    assert rows == 8
    assert columns == 4
    tmp = torch.empty_like(x)
    partial = torch.empty((rows // 2, columns), dtype=x.dtype, device=x.device)
    reduced = torch.empty((columns,), dtype=x.dtype, device=x.device)
    out = torch.empty((1,), dtype=x.dtype, device=x.device)

    for producer_row, producer_column in hl.tile([rows, columns], block_size=[1, 1]):
        tmp[producer_row, producer_column] = x[producer_row, producer_column] + 1
    for partial_row, partial_column in hl.tile([rows, columns], block_size=[2, 1]):
        partial[partial_row.id, partial_column] = torch.sum(
            tmp[partial_row, partial_column], dim=0
        )
    for final_row, final_column in hl.tile(
        [rows // 2, columns], block_size=[rows // 2, 1]
    ):
        reduced[final_column] = torch.sum(partial[final_row, final_column], dim=0)
    for output_index in hl.tile(1, block_size=1):
        out[output_index] = torch.sum(reduced[:], dim=-1)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def cartesian_affine_join(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    out = torch.empty_like(x)

    for tile_batch, tile_width in hl.tile([batch, width]):
        left[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        right[tile_batch, tile_width] = x[tile_batch, tile_width] - 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        out[tile_batch, tile_width] = (
            left[tile_batch, tile_width] + right[tile_batch, tile_width]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def singleton_root_join(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for tile_batch, tile_width in hl.tile([batch, width]):
        left[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        right[tile_batch, tile_width] = x[tile_batch, tile_width] - 1
    for tile_batch in hl.tile(batch, block_size=1):
        out[tile_batch] = torch.sum(
            left[tile_batch, :].to(torch.float32)
            + right[tile_batch, :].to(torch.float32),
            dim=-1,
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def streamed_singleton_reduction(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for producer_batch, producer_width in hl.tile([batch, width]):
        tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch in hl.tile(batch, block_size=1):
        acc = hl.zeros([consumer_batch], dtype=torch.float32)
        for reduction_width in hl.tile(width, block_size=16):
            acc = acc + torch.sum(
                tmp[consumer_batch, reduction_width].to(torch.float32), dim=-1
            )
        out[consumer_batch] = acc + tmp[consumer_batch, 0].to(torch.float32)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def nested_store_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_batch in hl.tile(batch, block_size=1):
        for producer_width in hl.tile(width, block_size=16):
            tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch, consumer_width in hl.tile([batch, width], block_size=[1, 16]):
        out[consumer_batch, consumer_width] = tmp[consumer_batch, consumer_width] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def offset_affine_chain(x: torch.Tensor) -> torch.Tensor:
    width = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty((width - 32,), dtype=x.dtype, device=x.device)

    for producer_tile in hl.tile(32, width):
        tmp[producer_tile] = x[producer_tile] + 1
    for consumer_tile in hl.tile(width - 32):
        out[consumer_tile] = tmp[consumer_tile + 32] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def partial_prefix_continuation(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty((width - 32,), dtype=x.dtype, device=x.device)

    for producer_tile in hl.tile(width):
        tmp[producer_tile] = x[producer_tile] + 1
    for consumer_tile in hl.tile(width - 32):
        out[consumer_tile] = tmp[consumer_tile] * 2
    return tmp, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def partial_prefix_in_place_chain(x: torch.Tensor) -> torch.Tensor:
    width = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_tile in hl.tile(width):
        tmp[producer_tile] = x[producer_tile] + 1
    for prefix_tile in hl.tile(width - 32):
        tmp[prefix_tile] = tmp[prefix_tile] * 2
    for output_tile in hl.tile(width):
        out[output_tile] = tmp[output_tile]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def multi_producer_join(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    left = torch.empty_like(x)
    right = torch.empty_like(y)
    out = torch.empty_like(x)

    for tile in hl.tile(x.size(0)):
        left[tile] = x[tile] + 1
    for tile in hl.tile(y.size(0)):
        right[tile] = y[tile] * 2
    for tile in hl.tile(x.size(0)):
        out[tile] = left[tile] + right[tile]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def coalesced_multi_producer_join(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    heads, width = x.shape
    splits = 4
    left = torch.empty_like(x)
    right = torch.empty_like(y)
    out = torch.empty((splits, heads, width), dtype=x.dtype, device=x.device)

    for tile_head, tile_width in hl.tile([heads, width], block_size=[1, 1]):
        left[tile_head, tile_width] = x[tile_head, tile_width] + 1
    for tile_head in hl.tile(heads, block_size=1):
        right[tile_head] = y[tile_head] * 2
    for tile_split, tile_head, tile_width in hl.tile(
        [splits, heads, width], block_size=[1, 1, width]
    ):
        out[tile_split, tile_head, tile_width] = (
            left[tile_head, tile_width]
            + right[tile_head][:, None]
            + tile_split.index[:, None, None]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def coalesced_single_producer_fanout(x: torch.Tensor) -> torch.Tensor:
    heads, width = x.shape
    splits = 4
    tmp = torch.empty_like(x)
    out = torch.empty((splits, heads, width), dtype=x.dtype, device=x.device)

    for tile_head, tile_width in hl.tile([heads, width], block_size=[1, 1]):
        tmp[tile_head, tile_width] = x[tile_head, tile_width] + 1
    for tile_split, tile_head, tile_width in hl.tile(
        [splits, heads, width], block_size=[1, 1, width]
    ):
        out[tile_split, tile_head, tile_width] = (
            tmp[tile_head, tile_width] + tile_split.index[:, None, None]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def direct_nested_continuation(x: torch.Tensor) -> torch.Tensor:
    width = x.size(0)
    tmp = torch.empty_like(x)
    reduced = torch.empty((width // 2,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(reduced)

    for producer_tile in hl.tile(width, block_size=1):
        tmp[producer_tile] = x[producer_tile] + 1
    for reduced_tile in hl.tile(width, block_size=2):
        reduced[reduced_tile.id] = torch.sum(tmp[reduced_tile], dim=-1)
    for output_tile in hl.tile(width // 2, block_size=1):
        out[output_tile] = reduced[output_tile] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def specialized_quotient_chain(
    x: torch.Tensor,
    numerator: int,
    denominator: int,
) -> torch.Tensor:
    hl.specialize(numerator)
    hl.specialize(denominator)
    width = numerator // denominator
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_tile in hl.tile(width, block_size=1):
        tmp[producer_tile] = x[producer_tile] + 1
    for consumer_tile in hl.tile(width, block_size=1):
        out[consumer_tile] = tmp[consumer_tile] * 2
    return out


def _access(
    access_id: int,
    *,
    root: int,
    allocation_id: int = 0,
    kind: Literal["load", "store"],
    shape: tuple[int, ...] = (128,),
    strides: tuple[int, ...] = (1,),
    block_ids: tuple[int | None, ...] = (0,),
    scales: tuple[int, ...] = (1,),
    offsets: tuple[int | None, ...] = (0,),
    scalar: tuple[bool, ...] | None = None,
    full_slice: tuple[bool, ...] | None = None,
    masked: bool = False,
    tensor_name: str = "tmp",
    storage_offset: int = 0,
    layout_is_static: bool = True,
) -> TileAccess:
    return TileAccess(
        access_id=access_id,
        memory_op_index=access_id,
        graph_id=root,
        root=root,
        allocation_id=allocation_id,
        kind=kind,
        tensor_name=tensor_name,
        tensor_shape=shape,
        tensor_strides=strides,
        storage_offset=storage_offset,
        subscript_dims=tuple(range(len(block_ids))),
        subscript_affine_block_ids=block_ids,
        subscript_index_scales=scales,
        subscript_offsets=offsets,
        subscript_is_scalar=scalar or tuple(False for _ in block_ids),
        has_explicit_mask=masked,
        subscript_is_full_slice=full_slice or tuple(False for _ in block_ids),
        layout_is_static=layout_is_static,
    )


class TestCrossLoopDependencies(TestCase):
    @skipIfNotCUDA()
    def test_device_ir_scopes_preserve_nested_producer_and_consumer_axes(
        self,
    ) -> None:
        x = torch.empty((2, 64), device=DEVICE, dtype=torch.float32)

        producer_graph = nested_store_chain.bind((x,)).host_function.device_ir
        assert producer_graph.tile_dependency_graph is not None
        producer_store = next(
            access
            for access in producer_graph.tile_dependency_graph.accesses
            if access.root == 0 and access.kind == "store"
        )
        (producer_scope,) = producer_graph.tile_dependency_graph.scopes_for_access(
            producer_store.access_id
        )
        self.assertEqual(producer_scope.kind, "loop")
        self.assertEqual(len(producer_scope.callsite_path), 1)
        self.assertEqual(
            producer_scope.logical_axis_order,
            (
                *producer_graph.task_families[0].logical_axis_order,
                *producer_scope.local_axis_order,
            ),
        )
        self.assertTrue(producer_scope.guaranteed)
        self.assertTrue(producer_scope.segmentable)

        consumer_graph = streamed_singleton_reduction.bind((x,)).host_function.device_ir
        assert consumer_graph.tile_dependency_graph is not None
        consumer_load = next(
            access
            for access in consumer_graph.tile_dependency_graph.accesses
            if access.root == 1
            and access.kind == "load"
            and any(
                scope.kind == "loop"
                for scope in consumer_graph.tile_dependency_graph.scopes_for_access(
                    access.access_id
                )
            )
        )
        (consumer_scope,) = consumer_graph.tile_dependency_graph.scopes_for_access(
            consumer_load.access_id
        )
        self.assertEqual(consumer_scope.kind, "loop")
        self.assertEqual(len(consumer_scope.callsite_path), 1)
        self.assertEqual(
            consumer_scope.logical_axis_order,
            (
                *consumer_graph.task_families[1].logical_axis_order,
                *consumer_scope.local_axis_order,
            ),
        )
        self.assertTrue(consumer_scope.guaranteed)
        self.assertTrue(consumer_scope.segmentable)

    def test_semantic_event_graph_composes_arbitrary_chain_depth(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(20,)),
                _access(2, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(3, root=2, allocation_id=1, kind="load", block_ids=(30,)),
                _access(4, root=2, allocation_id=2, kind="store", block_ids=(30,)),
                _access(5, root=3, allocation_id=2, kind="load", block_ids=(40,)),
            ),
            [[10], [20], [30], [40]],
        )

        self.assertEqual(len(graph.task_families), 4)
        self.assertEqual(
            tuple(event.producer_root for event in graph.events),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(use.consumer_root for use in graph.waits),
            (1, 2, 3),
        )
        configured = instantiate_event_graph(
            graph,
            tuple(
                InstantiatedTaskFamily(
                    logical_axis_order=(block_id,),
                    physical_axis_order=(block_id,),
                    axis_counts_items=((block_id, 4),),
                    block_sizes_items=((block_id, 1),),
                )
                for block_id in (10, 20, 30, 40)
            ),
        )
        self.assertEqual(len(configured.events), 3)
        self.assertEqual(
            tuple(event.expected_arrivals for event in configured.events),
            ((1, 1, 1, 1),) * 3,
        )
        canonical = canonicalize_ready_events(configured)
        baseline = build_baseline_worker_schedule(
            canonical.task_families, worker_count=4
        )
        local_triggers = derive_local_triggers(canonical, baseline)
        self.assertEqual(
            tuple(
                canonical.event(trigger.event_index)
                .uses[trigger.use_index]
                .consumer_root
                for trigger in local_triggers
            ),
            (1, 2, 3),
        )
        local_tasks = frozenset(
            (root, task)
            for root in (1, 2, 3)
            for task in range(canonical.task_families[root].task_count)
        )
        validate_worker_schedule(
            canonical,
            baseline.without_tasks(local_tasks),
            local_triggers,
        )

    def test_semantic_event_graph_represents_diamond_without_path_matching(
        self,
    ) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=0, allocation_id=1, kind="store", block_ids=(10,)),
                _access(2, root=1, allocation_id=0, kind="load", block_ids=(20,)),
                _access(3, root=1, allocation_id=2, kind="store", block_ids=(20,)),
                _access(4, root=2, allocation_id=1, kind="load", block_ids=(30,)),
                _access(5, root=2, allocation_id=3, kind="store", block_ids=(30,)),
                _access(6, root=3, allocation_id=2, kind="load", block_ids=(40,)),
                _access(7, root=3, allocation_id=3, kind="load", block_ids=(40,)),
            ),
            [[10], [20], [30], [40]],
        )

        root_zero_events = graph.events_contributed_by(0)
        self.assertEqual(len(root_zero_events), 1)
        self.assertEqual(
            {
                use.consumer_root
                for use in graph.uses_for_event(root_zero_events[0].event_id)
            },
            {1, 2},
        )
        self.assertEqual(
            {
                graph.event(use.event_id).producer_root
                for use in graph.waits_for_root(3)
            },
            {1, 2},
        )
        configured = instantiate_event_graph(
            graph,
            tuple(
                InstantiatedTaskFamily(
                    logical_axis_order=(block_id,),
                    physical_axis_order=(block_id,),
                    axis_counts_items=((block_id, 4),),
                    block_sizes_items=((block_id, 1),),
                )
                for block_id in (10, 20, 30, 40)
            ),
        )
        root_zero_event = configured.events_contributed_by(0)[0]
        self.assertEqual(
            {use.consumer_root for use in root_zero_event.uses},
            {1, 2},
        )
        canonical = canonicalize_ready_events(configured)
        local_triggers = derive_local_triggers(
            canonical,
            build_baseline_worker_schedule(canonical.task_families, worker_count=4),
        )
        self.assertEqual(
            {
                canonical.event(trigger.event_index)
                .uses[trigger.use_index]
                .consumer_root
                for trigger in local_triggers
            },
            {3},
        )

    def test_exact_and_family_done_dependencies_can_share_one_consumer(
        self,
    ) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    block_ids=(10,),
                    layout_is_static=False,
                ),
                _access(1, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    block_ids=(30,),
                    layout_is_static=False,
                ),
                _access(3, root=2, allocation_id=1, kind="load", block_ids=(30,)),
            ),
            [[10], [20], [30]],
        )

        required_events = tuple(
            graph.event(use.event_id) for use in graph.waits_for_root(2)
        )
        self.assertEqual(len(required_events), 2)
        self.assertEqual(
            {(event.producer_root, event.is_family_done) for event in required_events},
            {(0, True), (1, False)},
        )
        configured = instantiate_event_graph(
            graph,
            tuple(
                InstantiatedTaskFamily(
                    logical_axis_order=(block_id,),
                    physical_axis_order=(block_id,),
                    axis_counts_items=((block_id, 4),),
                    block_sizes_items=((block_id, 1),),
                )
                for block_id in (10, 20, 30)
            ),
        )
        configured_uses = configured.uses_for_root(2)
        self.assertEqual(len(configured_uses), 2)
        family_event = next(
            event for event in configured.events if event.is_family_done
        )
        self.assertEqual(family_event.family_done_root, 0)
        self.assertEqual(family_event.expected_arrivals, (4,))

    def test_baseline_worker_schedule_preserves_source_order(self) -> None:
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 3),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 5),), ((20, 1),)),
        )

        schedule = build_baseline_worker_schedule(task_families, worker_count=4)

        self.assertEqual(schedule.placement(0, 0), (0, 0))
        self.assertEqual(schedule.placement(0, 2), (2, 0))
        self.assertEqual(schedule.placement(1, 0), (0, 1))
        self.assertEqual(schedule.placement(1, 4), (0, 2))
        self.assertEqual(schedule.task_at(3, 0), None)
        self.assertEqual(schedule.task_at(3, 1), (1, 3))

    def test_baseline_worker_schedule_preserves_physical_traversal(self) -> None:
        task_families = (
            InstantiatedTaskFamily(
                (10,),
                (10,),
                ((10, 4),),
                ((10, 1),),
                logical_task_by_physical_task=(0, 2, 1, 3),
            ),
        )

        schedule = build_baseline_worker_schedule(task_families, worker_count=2)

        self.assertEqual(schedule.placement(0, 0), (0, 0))
        self.assertEqual(schedule.placement(0, 2), (1, 0))
        self.assertEqual(schedule.placement(0, 1), (0, 1))
        self.assertEqual(schedule.placement(0, 3), (1, 1))

    def test_worker_schedule_segment_supports_multiple_rounds(self) -> None:
        segment = WorkerScheduleSegment(
            root=2,
            task_begin=10,
            task_count=3,
            task_step=2,
            worker_begin=2,
            worker_count=2,
            schedule_begin=0,
        )

        self.assertEqual(segment.placement(10), (2, 0))
        self.assertEqual(segment.placement(12), (3, 0))
        self.assertEqual(segment.placement(14), (2, 1))
        self.assertEqual(segment.placement(11), None)
        self.assertEqual(segment.task_at(2, 1), 14)

        periodic = WorkerScheduleSegment(
            root=3,
            task_begin=0,
            task_count=6,
            task_step=1,
            task_period=3,
            task_period_step=10,
            worker_begin=0,
            worker_count=2,
            schedule_begin=0,
        )
        self.assertEqual(
            tuple(periodic.task_for_offset(offset) for offset in range(6)),
            (0, 1, 2, 10, 11, 12),
        )
        self.assertEqual(periodic.placement(11), (0, 2))

        rectangular = WorkerScheduleSegment(
            root=4,
            task_begin=0,
            task_count=6,
            task_step=1,
            worker_begin=0,
            worker_count=4,
            schedule_begin=0,
            schedule_period=2,
            schedule_period_step=4,
        )
        self.assertEqual(
            tuple(rectangular.schedule_for_offset(offset) for offset in range(6)),
            (0, 1, 4, 5, 8, 9),
        )
        self.assertEqual(rectangular.task_at(1, 2), 5)

    def test_worker_schedule_removes_local_tasks_without_closing_positions(
        self,
    ) -> None:
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 3),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 3),), ((20, 1),)),
        )
        baseline = build_baseline_worker_schedule(task_families, worker_count=2)

        schedule = baseline.without_tasks(frozenset(((0, 1), (1, 0), (1, 2))))

        self.assertEqual(schedule.placement(0, 0), (0, 0))
        self.assertEqual(schedule.placement(0, 1), None)
        self.assertEqual(schedule.placement(0, 2), (0, 1))
        self.assertEqual(schedule.placement(1, 0), None)
        self.assertEqual(schedule.placement(1, 1), (1, 2))
        self.assertEqual(schedule.placement(1, 2), None)

    def test_local_contributors_are_ordered_by_ready_key(self) -> None:
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 4),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 2),), ((20, 1),)),
        )
        event_graph = InstantiatedEventGraph(
            task_families=task_families,
            events=(
                InstantiatedKeyedEvent(
                    event_id=0,
                    source_event_ids=(0,),
                    key_count=2,
                    contributions=(
                        InstantiatedEventContribution(
                            producer_root=0,
                            keys_by_task=(
                                frozenset((0,)),
                                frozenset((1,)),
                                frozenset((0,)),
                                frozenset((1,)),
                            ),
                        ),
                    ),
                    uses=(
                        InstantiatedEventUse(
                            consumer_root=1,
                            consumer_access_id=None,
                            placement="root_entry",
                            required_keys_by_task=(
                                frozenset((0,)),
                                frozenset((1,)),
                            ),
                        ),
                    ),
                ),
            ),
        )
        baseline = build_baseline_worker_schedule(task_families, worker_count=2)
        triggers = derive_local_triggers(event_graph, baseline)

        schedule = order_local_contributors_by_key(
            event_graph,
            baseline,
            triggers,
        )

        self.assertEqual(schedule.placement(0, 0), (0, 0))
        self.assertEqual(schedule.placement(0, 2), (1, 0))
        self.assertEqual(schedule.placement(0, 1), (0, 1))
        self.assertEqual(schedule.placement(0, 3), (1, 1))

    def test_worker_schedule_detects_dependency_order_cycle(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 1),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 1),), ((20, 1),)),
        )
        event_graph = instantiate_event_graph(graph, task_families)

        validate_worker_schedule(
            event_graph,
            build_baseline_worker_schedule(task_families, worker_count=1),
        )
        reversed_schedule = WorkerSchedule(
            worker_count=1,
            segments=(
                WorkerScheduleSegment(1, 0, 1, 0, 1, 0),
                WorkerScheduleSegment(0, 0, 1, 0, 1, 1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "dependency/order cycle"):
            validate_worker_schedule(event_graph, reversed_schedule)

    def test_counted_event_supports_independent_consumer_uses(self) -> None:
        event = CountedEventPlan(
            contributors=(
                CountedEventContribution(
                    source_event_ids=(3,),
                    producer_root=0,
                    task_to_key=(0, 0),
                ),
            ),
            uses=(
                CountedEventUse(consumer_root=1, key_by_task=(0,)),
                CountedEventUse(consumer_root=2, key_by_task=(0, 0)),
            ),
        )

        self.assertEqual(event.key_count, 1)
        self.assertEqual(event.expected_arrivals, 2)
        self.assertIsNone(event.local_use)
        self.assertEqual(tuple(use.consumer_root for use in event.uses), (1, 2))

    def test_counted_event_selection_keeps_independent_direct_uses(self) -> None:
        task_families = tuple(
            InstantiatedTaskFamily((axis,), (axis,), ((axis, 2),), ((axis, 1),))
            for axis in (10, 20, 30)
        )
        event_graph = InstantiatedEventGraph(
            task_families=task_families,
            events=(
                InstantiatedKeyedEvent(
                    event_id=0,
                    source_event_ids=(0,),
                    key_count=2,
                    contributions=(
                        InstantiatedEventContribution(
                            producer_root=0,
                            keys_by_task=(frozenset((0,)), frozenset((1,))),
                        ),
                    ),
                    uses=tuple(
                        InstantiatedEventUse(
                            consumer_root=root,
                            consumer_access_id=None,
                            placement="root_entry",
                            required_keys_by_task=(
                                frozenset((0,)),
                                frozenset((1,)),
                            ),
                        )
                        for root in (1, 2)
                    ),
                ),
            ),
        )
        dependency_graph = TileDependencyGraph((), (), (), (), ())

        (selected,) = choose_counted_events(
            event_graph,
            (),
            dependency_graph,
            excluded_direct_roots=frozenset((1,)),
        )

        self.assertEqual(selected.key_count, 2)
        self.assertEqual(tuple(use.consumer_root for use in selected.uses), (2,))

    def test_counted_event_lowering_is_derived_from_the_semantic_graph(self) -> None:
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 4),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 2),), ((20, 2),)),
        )
        event_graph = InstantiatedEventGraph(
            task_families=task_families,
            events=(
                InstantiatedKeyedEvent(
                    event_id=0,
                    source_event_ids=(3,),
                    key_count=2,
                    contributions=(
                        InstantiatedEventContribution(
                            producer_root=0,
                            keys_by_task=(
                                frozenset((0,)),
                                frozenset((0,)),
                                frozenset((1,)),
                                frozenset((1,)),
                            ),
                        ),
                    ),
                    uses=(
                        InstantiatedEventUse(
                            consumer_root=1,
                            consumer_access_id=None,
                            placement="root_entry",
                            required_keys_by_task=(
                                frozenset((0,)),
                                frozenset((1,)),
                            ),
                        ),
                    ),
                ),
            ),
        )
        baseline = build_baseline_worker_schedule(task_families, worker_count=4)
        triggers = derive_local_triggers(event_graph, baseline)

        (lowered,) = lower_counted_events(event_graph, triggers)

        self.assertEqual(lowered.single_contributor.task_to_key, (0, 0, 1, 1))
        self.assertEqual(lowered.uses[0].key_by_task, (0, 1))
        self.assertEqual(lowered.expected_arrivals, 2)
        self.assertEqual(lowered.local_trigger_use, 0)

    def test_ready_event_canonicalization_derives_fanin(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(4,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(4,),
                    block_ids=(20,),
                ),
            ),
            [[10], [20]],
        )
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 4),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 2),), ((20, 2),)),
        )

        event_graph = canonicalize_ready_events(
            instantiate_event_graph(graph, task_families)
        )
        event = event_graph.events[-1]

        self.assertEqual(event.key_count, 2)
        self.assertEqual(event.expected_arrivals, (2, 2))
        self.assertEqual(
            event.contributions[0].keys_by_task,
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(
            event.uses[0].required_keys_by_task,
            (frozenset((0,)), frozenset((1,))),
        )

    def test_ready_event_canonicalization_supports_overlapping_keys(self) -> None:
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 3),), ((10, 1),)),
            InstantiatedTaskFamily((20,), (20,), ((20, 2),), ((20, 1),)),
        )
        event_graph = InstantiatedEventGraph(
            task_families=task_families,
            events=(
                InstantiatedKeyedEvent(
                    event_id=0,
                    source_event_ids=(0,),
                    key_count=3,
                    contributions=(
                        InstantiatedEventContribution(
                            producer_root=0,
                            keys_by_task=(
                                frozenset((0,)),
                                frozenset((1,)),
                                frozenset((2,)),
                            ),
                        ),
                    ),
                    uses=(
                        InstantiatedEventUse(
                            consumer_root=1,
                            consumer_access_id=None,
                            placement="root_entry",
                            required_keys_by_task=(
                                frozenset((0, 1)),
                                frozenset((1, 2)),
                            ),
                        ),
                    ),
                ),
            ),
        )

        event = canonicalize_ready_events(event_graph).events[-1]

        self.assertEqual(event.key_count, 2)
        self.assertEqual(event.expected_arrivals, (2, 2))
        self.assertEqual(
            event.contributions[0].keys_by_task,
            (
                frozenset((0,)),
                frozenset((0, 1)),
                frozenset((1,)),
            ),
        )

    def test_noninjective_regions_are_not_coordinate_disjoint(self) -> None:
        for layout, left_interval, right_interval, second_dimension in (
            (((2, 1), (0, 1), 0), (0, 1), (0, 1), (0, 1)),
            (((2, 2), (1, 1), 0), (0, 2), (1, 3), (0, 2)),
        ):
            with self.subTest(layout=layout):
                left = AllocationRegion(
                    left_interval,
                    False,
                    layout,
                    ((0, 1), second_dimension),
                    True,
                )
                right = AllocationRegion(
                    right_interval,
                    False,
                    layout,
                    ((1, 2), second_dimension),
                    True,
                )

                self.assertTrue(allocation_regions_may_overlap(left, right))

    def test_nonstatic_layout_falls_back_to_root_readiness(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    block_ids=(10,),
                    layout_is_static=False,
                ),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertEqual(plan.events[0].granularity, "root")
        self.assertTrue(plan.events[0].is_family_done)
        self.assertEqual(plan.events[0].contributors, (EventContribution(0),))

    def test_multi_contributor_event_has_no_unique_producer(self) -> None:
        event = KeyedEvent(
            event_id=0,
            contributors=(EventContribution(0), EventContribution(1)),
            granularity="task",
        )

        with self.assertRaisesRegex(ValueError, "no unique producer root"):
            _ = event.producer_root

    def test_multidimensional_storage_offset_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(4, 4),
                    strides=(4, 1),
                    block_ids=(10, 11),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(3, 3),
                    strides=(4, 1),
                    block_ids=(20, 21),
                    storage_offset=5,
                ),
            ),
            [[10, 11], [20, 21]],
        )

        self.assertEqual(plan.events[0].granularity, "root")

    def test_one_dimensional_storage_offset_remains_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(128,),
                    strides=(1,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(64,),
                    strides=(1,),
                    block_ids=(20,),
                    storage_offset=32,
                ),
            ),
            [[10], [20]],
        )

        predecessor_map = plan.edges[0].readiness[0].predecessor_map
        assert predecessor_map is not None
        self.assertEqual(predecessor_map.axes[0].consumer_offset, 32)

    def test_repeated_task_to_key_map_uses_one_periodic_segment(self) -> None:
        segments = _compress_task_to_key(tuple(range(8)) * 4)

        self.assertEqual(len(segments), 1)
        self.assertEqual(
            dataclasses.asdict(segments[0]),
            {
                "task_begin": 0,
                "task_count": 32,
                "tasks_per_key": 1,
                "first_key": 0,
                "key_stride": 1,
                "key_period": 8,
            },
        )

    def test_source_phase_boundary_satisfies_allocation_dependency(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            task_families=(
                TaskFamily((LogicalTaskAxis(10, None),)),
                TaskFamily((LogicalTaskAxis(20, None),)),
            ),
            root_phases=(0, 1),
        )

        self.assertEqual(plan.edges, ())

    def test_edge_retains_every_alias_of_the_allocation(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    tensor_name="base",
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    tensor_name="producer_view",
                    block_ids=(1,),
                ),
                _access(
                    2,
                    root=1,
                    kind="store",
                    tensor_name="producer_view",
                    block_ids=(1,),
                ),
                _access(
                    3,
                    root=2,
                    kind="load",
                    tensor_name="consumer_view",
                    block_ids=(2,),
                ),
            ),
            [[0], [1], [2]],
        )

        edge = plan.edges_between(1, 2)[0]
        self.assertEqual(edge.allocation_id, 0)
        self.assertEqual(
            edge.tensor_names,
            frozenset(("base", "producer_view", "consumer_view")),
        )

    def test_identity_mapping_is_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertEqual(len(plan.edges), 1)
        edge = plan.edges[0]
        self.assertTrue(edge.is_raw_only)
        self.assertTrue(edge.is_task_ready)
        self.assertEqual(
            [(event.producer_root, event.granularity) for event in plan.events],
            [(0, "task")],
        )
        self.assertEqual(len(plan.waits), 1)
        self.assertEqual(plan.waits[0].consumer_access_id, 1)
        self.assertEqual(plan.waits[0].event_id, plan.events[0].event_id)
        self.assertEqual(plan.waits[0].placement, "root_entry")
        predecessor_map = edge.readiness[0].predecessor_map
        assert predecessor_map is not None
        self.assertEqual(
            [
                (axis.producer_block_id, axis.consumer_block_id)
                for axis in predecessor_map.axes
            ],
            [(10, 20)],
        )

    def test_aligned_in_place_update_is_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
                _access(2, root=1, kind="store", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertEqual(
            edge.kinds,
            frozenset(
                (
                    TileDependencyKind.READ_AFTER_WRITE,
                    TileDependencyKind.WRITE_AFTER_WRITE,
                )
            ),
        )
        self.assertTrue(edge.is_task_ready)
        self.assertEqual(
            [requirement.consumer_access_id for requirement in edge.readiness],
            [1, 2],
        )
        self.assertEqual(
            [(event.producer_root, event.granularity) for event in plan.events],
            [(0, "task")],
        )

    def test_aligned_write_after_read_is_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="load", block_ids=(10,)),
                _access(1, root=1, kind="store", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertEqual(
            edge.kinds,
            frozenset((TileDependencyKind.WRITE_AFTER_READ,)),
        )
        self.assertTrue(edge.is_task_ready)
        self.assertEqual(plan.events[0].granularity, "task")

    def test_unproven_write_hazard_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    kind="store",
                    block_ids=(20,),
                    scales=(-1,),
                ),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertFalse(edge.is_task_ready)
        self.assertEqual(edge.readiness[0].granularity, "root")
        self.assertEqual(plan.events[0].granularity, "root")

    def test_reversed_mapping_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    scales=(-1,),
                ),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertFalse(edge.is_task_ready)
        self.assertEqual(edge.readiness[0].granularity, "root")
        self.assertIsNone(edge.readiness[0].predecessor_map)
        self.assertEqual(plan.events[0].granularity, "root")
        self.assertEqual(plan.waits[0].consumer_access_id, 1)
        self.assertEqual(plan.waits[0].placement, "access")

    def test_batch_axis_is_part_of_task_mapping(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(2, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(2, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21]],
        )

        edge = plan.edges[0]
        self.assertTrue(edge.is_task_ready)
        predecessor_map = edge.readiness[0].predecessor_map
        assert predecessor_map is not None
        self.assertEqual(
            [
                (axis.producer_block_id, axis.consumer_block_id)
                for axis in predecessor_map.axes
            ],
            [(10, 20), (11, 21)],
        )
        self.assertEqual(
            predecessor_task_ids(
                predecessor_map,
                consumer_coordinates={20: 1, 21: 2},
                block_sizes={10: 1, 11: 1, 20: 1, 21: 1},
                producer_axis_order=(11, 10),
                producer_axis_counts={10: 2, 11: 4},
            ),
            frozenset((6,)),
        )

    def test_size_one_view_dimensions_are_normalized(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(32, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(1, 32, 128),
                    strides=(4096, 128, 1),
                    block_ids=(None, 20, 21),
                    scales=(1, 1, 1),
                    offsets=(0, 0, 0),
                    scalar=(True, False, False),
                    full_slice=(True, False, False),
                ),
            ),
            [[10, 11], [20, 21]],
        )

        edge = plan.edges[0]
        self.assertTrue(edge.is_task_ready)
        predecessor_map = edge.readiness[0].predecessor_map
        assert predecessor_map is not None
        self.assertEqual(
            [
                (axis.producer_block_id, axis.consumer_block_id)
                for axis in predecessor_map.axes
            ],
            [(10, 20), (11, 21)],
        )

    def test_nontrivial_reshape_still_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(32, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(4096,),
                    strides=(1,),
                    block_ids=(20,),
                    scales=(1,),
                    offsets=(0,),
                ),
            ),
            [[10, 11], [20]],
        )

        self.assertFalse(plan.edges[0].is_task_ready)
        self.assertEqual(plan.edges[0].readiness[0].granularity, "root")

    def test_unequal_tiles_map_to_every_overlapping_producer(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        predecessor_map = plan.edges[0].readiness[0].predecessor_map
        assert predecessor_map is not None

        self.assertEqual(
            predecessor_task_ids(
                predecessor_map,
                consumer_coordinates={20: 1},
                block_sizes={10: 16, 20: 64},
                producer_axis_order=(10,),
                producer_axis_counts={10: 8},
            ),
            frozenset((4, 5, 6, 7)),
        )

    def test_uniform_partition_uses_coordinates_not_flattened_pid_runs(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    kind="load",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 128),
                ),
            ),
            [[10, 11], [20, 21]],
        )
        predecessor_maps = tuple(
            requirement.predecessor_map
            for requirement in plan.edges[0].readiness
            if requirement.predecessor_map is not None
        )

        partition = prove_uniform_task_partition(
            predecessor_maps,
            consumer_axis_order=(21, 20),
            consumer_axis_counts={20: 2, 21: 4},
            producer_axis_order=(11, 10),
            producer_axis_counts={10: 2, 11: 16},
            block_sizes={10: 1, 11: 16, 20: 1, 21: 32},
        )

        assert partition is not None
        self.assertEqual(partition.producer_tasks, 32)
        self.assertEqual(partition.consumer_tasks, 8)
        self.assertEqual(partition.fanin, 4)
        self.assertEqual(
            [
                (axis.producer_block_id, axis.consumer_block_id, axis.scale)
                for axis in partition.outer_axes
            ],
            [(10, 20, 1)],
        )
        self.assertEqual(partition.partition_producer_block_id, 11)
        self.assertEqual(partition.partition_consumer_block_id, 21)
        self.assertEqual(partition.partition_consumer_stride, 2)
        self.assertEqual(
            [(segment.begin, segment.length) for segment in partition.segments],
            [(0, 2), (8, 2)],
        )

    def test_uniform_partition_accepts_non_power_of_two_fanin(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", shape=(96,), block_ids=(10,)),
                _access(1, root=1, kind="load", shape=(96,), block_ids=(20,)),
            ),
            [[10], [20]],
        )
        predecessor_map = plan.edges[0].readiness[0].predecessor_map
        assert predecessor_map is not None

        partition = prove_uniform_task_partition(
            (predecessor_map,),
            consumer_axis_order=(20,),
            consumer_axis_counts={20: 2},
            producer_axis_order=(10,),
            producer_axis_counts={10: 6},
            block_sizes={10: 16, 20: 48},
        )

        assert partition is not None
        self.assertEqual(partition.fanin, 3)
        self.assertEqual(
            [(segment.begin, segment.length) for segment in partition.segments],
            [(0, 3)],
        )

    def test_uniform_partition_accepts_compact_partial_domains(
        self,
    ) -> None:
        overlapping = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
                _access(
                    2,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    offsets=(16,),
                ),
            ),
            [[10], [20]],
        )
        overlapping_maps = tuple(
            requirement.predecessor_map
            for requirement in overlapping.edges[0].readiness
            if requirement.predecessor_map is not None
        )
        self.assertIsNone(
            prove_uniform_task_partition(
                overlapping_maps,
                consumer_axis_order=(20,),
                consumer_axis_counts={20: 4},
                producer_axis_order=(10,),
                producer_axis_counts={10: 8},
                block_sizes={10: 16, 20: 32},
            )
        )

        identity = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        identity_map = identity.edges[0].readiness[0].predecessor_map
        assert identity_map is not None
        prefix = prove_uniform_task_partition(
            (identity_map,),
            consumer_axis_order=(20,),
            consumer_axis_counts={20: 3},
            producer_axis_order=(10,),
            producer_axis_counts={10: 8},
            block_sizes={10: 16, 20: 32},
        )
        assert prefix is not None
        self.assertEqual(prefix.producer_tasks, 8)
        self.assertEqual(prefix.participating_producer_tasks, 6)
        self.assertFalse(prefix.covers_producer_domain)

        suffix = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    offsets=(32,),
                ),
            ),
            [[10], [20]],
        )
        suffix_map = suffix.edges[0].readiness[0].predecessor_map
        assert suffix_map is not None
        suffix_partition = prove_uniform_task_partition(
            (suffix_map,),
            consumer_axis_order=(20,),
            consumer_axis_counts={20: 3},
            producer_axis_order=(10,),
            producer_axis_counts={10: 8},
            block_sizes={10: 16, 20: 32},
        )
        assert suffix_partition is not None
        self.assertEqual(
            suffix_partition.producer_key_by_task,
            (None, None, 0, 0, 1, 1, 2, 2),
        )

    def test_tile_id_indices_use_scalar_extent(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    block_ids=(10,),
                    scalar=(True,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    scalar=(True,),
                ),
            ),
            [[10], [20]],
        )
        predecessor_map = plan.edges[0].readiness[0].predecessor_map
        assert predecessor_map is not None

        self.assertEqual(
            predecessor_task_ids(
                predecessor_map,
                consumer_coordinates={20: 2},
                block_sizes={10: 128, 20: 128},
                producer_axis_order=(10,),
                producer_axis_counts={10: 4},
            ),
            frozenset((2,)),
        )

    def test_multiple_stores_fall_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=0, kind="store", block_ids=(10,)),
                _access(2, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertFalse(edge.is_task_ready)

    def test_masked_store_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,), masked=True),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertEqual(plan.edges[0].readiness[0].granularity, "root")

    def test_nonzero_or_dynamic_grid_start_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
            noncanonical_task_origin_block_ids=frozenset((10,)),
        )

        self.assertEqual(plan.events[0].granularity, "root")
        self.assertEqual(plan.waits[0].placement, "access")

    def test_tracks_latest_writer_and_intervening_readers(self) -> None:
        task_families = tuple(
            TaskFamily(
                axes=(LogicalTaskAxis(root, 128),),
            )
            for root in range(4)
        )
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store"),
                _access(1, root=1, kind="load", block_ids=(1,)),
                _access(2, root=2, kind="store", block_ids=(2,)),
                _access(3, root=3, kind="load", block_ids=(3,)),
            ),
            task_families=task_families,
        )

        self.assertEqual(
            [
                (edge.producer_root, edge.consumer_root, edge.kinds)
                for edge in plan.edges
            ],
            [
                (0, 1, frozenset((TileDependencyKind.READ_AFTER_WRITE,))),
                (0, 2, frozenset((TileDependencyKind.WRITE_AFTER_WRITE,))),
                (1, 2, frozenset((TileDependencyKind.WRITE_AFTER_READ,))),
                (2, 3, frozenset((TileDependencyKind.READ_AFTER_WRITE,))),
            ],
        )

    def test_partial_write_retains_uncovered_reaching_definition(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(96,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(96,),
                    block_ids=(20,),
                ),
                _access(
                    2,
                    root=1,
                    kind="store",
                    shape=(96,),
                    block_ids=(20,),
                ),
                _access(
                    3,
                    root=2,
                    kind="load",
                    shape=(96,),
                    block_ids=(30,),
                ),
            ),
            task_families=(
                TaskFamily((LogicalTaskAxis(10, 96),)),
                TaskFamily((LogicalTaskAxis(20, 64),)),
                TaskFamily((LogicalTaskAxis(30, 96),)),
            ),
        )

        self.assertEqual(
            [
                (
                    edge.producer_root,
                    edge.consumer_root,
                    edge.kinds,
                    tuple(
                        dependency.region.address_interval
                        for dependency in edge.access_dependencies
                    ),
                )
                for edge in plan.edges
            ],
            [
                (
                    0,
                    1,
                    frozenset(
                        (
                            TileDependencyKind.READ_AFTER_WRITE,
                            TileDependencyKind.WRITE_AFTER_WRITE,
                        )
                    ),
                    ((0, 64), (0, 64)),
                ),
                (
                    0,
                    2,
                    frozenset((TileDependencyKind.READ_AFTER_WRITE,)),
                    ((64, 96),),
                ),
                (
                    1,
                    2,
                    frozenset((TileDependencyKind.READ_AFTER_WRITE,)),
                    ((0, 64),),
                ),
            ],
        )

    def test_fanout_keeps_one_edge_per_consumer(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store"),
                _access(1, root=1, kind="load", block_ids=(1,)),
                _access(2, root=2, kind="load", block_ids=(2,)),
            ),
            [[0], [1], [2]],
        )

        self.assertEqual(
            [(edge.producer_root, edge.consumer_root) for edge in plan.edges],
            [(0, 1), (0, 2)],
        )
        self.assertEqual(len(plan.events), 1)
        self.assertEqual(plan.events[0].granularity, "task")
        self.assertEqual(
            [(wait.consumer_root, wait.event_id) for wait in plan.waits],
            [(1, 0), (2, 0)],
        )

    def test_mixed_accesses_retain_their_strongest_readiness(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store"),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(1,)),
                _access(2, root=0, allocation_id=1, kind="store"),
                _access(
                    3,
                    root=1,
                    allocation_id=1,
                    kind="load",
                    block_ids=(1,),
                    scales=(-1,),
                ),
            ),
            [[0], [1]],
        )

        self.assertEqual(len(plan.edges), 2)
        self.assertEqual(
            [(event.producer_root, event.granularity) for event in plan.events],
            [(0, "task"), (0, "root")],
        )
        self.assertEqual(
            [
                (wait.consumer_root, wait.consumer_access_id, wait.placement)
                for wait in plan.waits
            ],
            [(1, 1, "root_entry"), (1, 3, "access")],
        )

    def test_mixed_exact_and_unknown_accesses_use_root_completion(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(1, 64),
                    strides=(64, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(1, 64),
                    strides=(64, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    kind="load",
                    shape=(1, 64),
                    strides=(64, 1),
                    block_ids=(None, None),
                    scales=(1, 1),
                    offsets=(None, None),
                    masked=True,
                ),
            ),
            [[10, 11], [20]],
        )
        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=(
                InstantiatedTaskFamily(
                    logical_axis_order=(10, 11),
                    physical_axis_order=(10, 11),
                    axis_counts_items=((10, 1), (11, 4)),
                    block_sizes_items=((10, 1), (11, 16)),
                ),
                InstantiatedTaskFamily(
                    logical_axis_order=(20,),
                    physical_axis_order=(20,),
                    axis_counts_items=((20, 1),),
                    block_sizes_items=((20, 1),),
                ),
            ),
            available_access_ids_by_root=(frozenset((0,)), frozenset((1, 2))),
            access_program_points={
                1: AccessProgramPoint(
                    ((20, "outer"), (21, "inner")),
                    loop_id=7,
                    loop_depth=1,
                    root_statement_index=3,
                ),
                2: AccessProgramPoint(
                    None,
                    loop_id=8,
                    loop_depth=1,
                    root_statement_index=4,
                ),
            },
            axis_geometry={10: (1, 1), 11: (4, 16), 20: (1, 1), 21: (4, 16)},
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=2,
        )

        self.assertEqual(schedule.task_ready_edges, frozenset())
        self.assertEqual(schedule.root_completion_edges, frozenset(((0, 1),)))
        self.assertEqual(schedule.access_cohorts, ())

    def test_singleton_producer_uses_root_completion(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=(
                InstantiatedTaskFamily(
                    logical_axis_order=(10,),
                    physical_axis_order=(10,),
                    axis_counts_items=((10, 1),),
                    block_sizes_items=((10, 128),),
                ),
                InstantiatedTaskFamily(
                    logical_axis_order=(20,),
                    physical_axis_order=(20,),
                    axis_counts_items=((20, 4),),
                    block_sizes_items=((20, 32),),
                ),
            ),
            available_access_ids_by_root=(frozenset((0,)), frozenset((1,))),
            access_program_points={},
            axis_geometry={10: (1, 128), 20: (4, 32)},
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=4,
        )

        self.assertEqual(schedule.task_ready_edges, frozenset())
        self.assertEqual(schedule.task_waits_by_root, {})
        self.assertEqual(schedule.root_completion_edges, frozenset(((0, 1),)))

    def test_root_completion_path_elides_redundant_exact_task_wait(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    block_ids=(None,),
                    offsets=(None,),
                ),
                _access(2, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    block_ids=(None,),
                    offsets=(None,),
                ),
                _access(4, root=2, allocation_id=2, kind="store", block_ids=(30,)),
                _access(
                    5,
                    root=3,
                    allocation_id=2,
                    kind="load",
                    block_ids=(None,),
                    offsets=(None,),
                ),
                _access(6, root=3, allocation_id=0, kind="load", block_ids=(40,)),
            ),
            [[10], [20], [30], [40]],
        )
        task_families = tuple(
            InstantiatedTaskFamily(
                logical_axis_order=(10 + root * 10,),
                physical_axis_order=(10 + root * 10,),
                axis_counts_items=((10 + root * 10, 8),),
                block_sizes_items=((10 + root * 10, 16),),
            )
            for root in range(4)
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=task_families,
            available_access_ids_by_root=(
                frozenset((0,)),
                frozenset((1, 2)),
                frozenset((3, 4)),
                frozenset((5, 6)),
            ),
            access_program_points={},
            axis_geometry={
                10: (8, 16),
                20: (8, 16),
                30: (8, 16),
                40: (8, 16),
            },
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=8,
        )

        self.assertEqual(
            schedule.root_completion_edges,
            frozenset(((0, 1), (1, 2), (2, 3))),
        )
        self.assertEqual(schedule.task_ready_edges, frozenset())
        self.assertEqual(schedule.task_waits_by_root, {})

    def test_worker_schedule_derives_access_ready_overlap(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(30, 31),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21], [30]],
        )
        task_families = (
            InstantiatedTaskFamily(
                logical_axis_order=(10, 11),
                physical_axis_order=(11, 10),
                axis_counts_items=((10, 1), (11, 8)),
                block_sizes_items=((10, 1), (11, 16)),
            ),
            InstantiatedTaskFamily(
                logical_axis_order=(20, 21),
                physical_axis_order=(21, 20),
                axis_counts_items=((20, 1), (21, 4)),
                block_sizes_items=((20, 1), (21, 32)),
            ),
            InstantiatedTaskFamily(
                logical_axis_order=(30,),
                physical_axis_order=(30,),
                axis_counts_items=((30, 1),),
                block_sizes_items=((30, 1),),
            ),
        )
        kwargs = {
            "dependency_plan": dependency_plan,
            "task_families": task_families,
            "available_access_ids_by_root": (
                frozenset((0,)),
                frozenset((1, 2)),
                frozenset((3,)),
            ),
            "access_program_points": {
                3: AccessProgramPoint(
                    ((30, "outer"), (31, "inner")),
                    loop_id=7,
                    loop_depth=1,
                    root_statement_index=3,
                )
            },
            "axis_geometry": {
                10: (1, 1),
                11: (8, 16),
                20: (1, 1),
                21: (4, 32),
                30: (1, 1),
                31: (4, 32),
            },
            "excluded_roots": frozenset(),
            "preordered_edges": frozenset(),
            "physical_worker_limit": 8,
        }

        task_ready_graph = canonicalize_task_readiness(
            instantiate_event_graph(dependency_plan, task_families),
            dependency_plan,
        )
        root_two_event = next(
            event
            for event in task_ready_graph.events
            if any(use.consumer_root == 2 for use in event.uses)
        )
        self.assertEqual(root_two_event.expected_arrivals, (4,))
        self.assertEqual(root_two_event.uses[0].placement, "root_entry")

        schedule = build_cross_loop_schedule(**kwargs, requested_worker_count=6)

        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(
            (
                event.producer_root,
                event.uses[0].consumer_root,
                event.local_use.consumer_root if event.local_use is not None else None,
                event.expected_arrivals,
            ),
            (0, 1, 1, 2),
        )
        self.assertEqual(len(schedule.local_triggers), 1)
        local_trigger = schedule.local_triggers[0]
        local_event = schedule.event_graph.event(local_trigger.event_index)
        self.assertEqual(local_trigger.use_index, 0)
        self.assertEqual(
            local_event.uses[local_trigger.use_index].consumer_root,
            1,
        )
        self.assertEqual(
            local_trigger.possible_workers,
            frozenset(range(schedule.worker_limit)),
        )
        self.assertEqual(schedule.worker_limit, 6)
        self.assertEqual(len(schedule.access_cohorts), 1)
        self.assertEqual(schedule.access_cohorts[0].stage_sizes, (3, 1))
        self.assertEqual(schedule.worker_schedule.placement(2, 0), (5, 1))
        self.assertEqual(schedule.worker_schedule.placement(0, 6), (0, 1))

        snapped = build_cross_loop_schedule(**kwargs, requested_worker_count=7)
        self.assertEqual(snapped.worker_limit, 6)
        self.assertEqual(snapped.worker_schedule, schedule.worker_schedule)

        fallback = build_cross_loop_schedule(**kwargs, requested_worker_count=0)
        self.assertEqual(
            sum(
                not fallback.event_graph.event(event.graph_event_index).is_family_done
                for event in fallback.counted_events
                if event.graph_event_index is not None
            ),
            1,
        )
        self.assertEqual(fallback.access_cohorts, ())
        self.assertEqual(fallback.task_ready_edges, frozenset(((0, 1),)))
        self.assertEqual(
            fallback.root_completion_edges,
            frozenset(((1, 2),)),
        )
        (family_done,) = tuple(
            event
            for event in fallback.counted_events
            if event.graph_event_index is not None
            and fallback.event_graph.event(event.graph_event_index).is_family_done
        )
        self.assertEqual(family_done.key_count, 1)
        self.assertEqual(family_done.producer_root, 1)
        self.assertEqual(tuple(use.consumer_root for use in family_done.uses), (2,))
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            "must be nonnegative",
        ):
            build_cross_loop_schedule(**kwargs, requested_worker_count=-1)

        short_families = (
            dataclasses.replace(
                task_families[0],
                axis_counts_items=((10, 1), (11, 4)),
                block_sizes_items=((10, 1), (11, 32)),
            ),
            dataclasses.replace(
                task_families[1],
                axis_counts_items=((20, 1), (21, 2)),
                block_sizes_items=((20, 1), (21, 64)),
            ),
            task_families[2],
        )
        short_schedule = build_cross_loop_schedule(
            **{
                **kwargs,
                "task_families": short_families,
                "axis_geometry": {
                    10: (1, 1),
                    11: (4, 32),
                    20: (1, 1),
                    21: (2, 64),
                    30: (1, 1),
                    31: (2, 64),
                },
            }
        )
        self.assertEqual(
            sum(
                not short_schedule.event_graph.event(
                    event.graph_event_index
                ).is_family_done
                for event in short_schedule.counted_events
                if event.graph_event_index is not None
            ),
            1,
        )
        self.assertEqual(short_schedule.access_cohorts, ())
        self.assertEqual(short_schedule.task_ready_edges, frozenset(((0, 1),)))
        self.assertEqual(
            short_schedule.root_completion_edges,
            frozenset(((1, 2),)),
        )

    def test_multi_producer_join_uses_one_keyed_event(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(2, root=2, allocation_id=0, kind="load", block_ids=(30,)),
                _access(3, root=2, allocation_id=1, kind="load", block_ids=(30,)),
            ),
            [[10], [20], [30]],
        )
        task_families = tuple(
            InstantiatedTaskFamily(
                logical_axis_order=(block_id,),
                physical_axis_order=(block_id,),
                axis_counts_items=((block_id, 8),),
                block_sizes_items=((block_id, 16),),
            )
            for root, block_id in enumerate((10, 20, 30))
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=task_families,
            available_access_ids_by_root=(
                frozenset((0,)),
                frozenset((1,)),
                frozenset((2, 3)),
            ),
            access_program_points={},
            axis_geometry={10: (8, 16), 20: (8, 16), 30: (8, 16)},
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=8,
        )

        self.assertEqual(schedule.task_ready_edges, frozenset(((0, 2), (1, 2))))
        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(schedule.task_waits_by_root, {})
        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(event.uses[0].consumer_root, 2)
        self.assertEqual(event.expected_arrivals, 2)
        self.assertEqual(
            [
                (contributor.producer_root, contributor.expected_arrivals)
                for contributor in event.contributors
            ],
            [(0, 1), (1, 1)],
        )

    def test_repeated_join_predecessors_coalesce_consumer_tasks(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(32,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(8,),
                    block_ids=(30,),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(8, 4),
                    strides=(4, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(8,),
                    block_ids=(20,),
                ),
            ),
            [[10], [30], [22, 20, 21]],
        )
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, 32),), ((10, 1),)),
            InstantiatedTaskFamily((30,), (30,), ((30, 8),), ((30, 1),)),
            InstantiatedTaskFamily(
                (22, 20, 21),
                (22, 20, 21),
                ((22, 4), (20, 8), (21, 1)),
                ((22, 1), (20, 1), (21, 4)),
            ),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=task_families,
            available_access_ids_by_root=(
                frozenset((0,)),
                frozenset((1,)),
                frozenset((2, 3)),
            ),
            access_program_points={},
            axis_geometry={
                10: (32, 1),
                20: (8, 1),
                21: (1, 4),
                22: (4, 1),
                30: (8, 1),
            },
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=32,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        self.assertEqual(schedule.local_triggers, ())
        event = schedule.counted_events[0]
        self.assertIsNone(event.local_use)
        self.assertEqual(event.key_count, 8)
        self.assertEqual(event.expected_arrivals, 5)
        self.assertEqual(event.uses[0].key_by_task, tuple(i // 4 for i in range(32)))
        self.assertEqual(
            [contributor.expected_arrivals for contributor in event.contributors],
            [4, 1],
        )

    def test_large_flattened_ready_groups_have_no_task_product_cutoff(self) -> None:
        heads = 513
        width = 4
        splits = 4
        elements = heads * width
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(elements,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(heads, width),
                    strides=(width, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10], [22, 20, 21]],
        )
        task_families = (
            InstantiatedTaskFamily(
                (10,),
                (10,),
                ((10, elements),),
                ((10, 1),),
            ),
            InstantiatedTaskFamily(
                (22, 20, 21),
                (22, 20, 21),
                ((22, splits), (20, heads), (21, 1)),
                ((22, 1), (20, 1), (21, width)),
            ),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=task_families,
            available_access_ids_by_root=(frozenset((0,)), frozenset((1,))),
            access_program_points={},
            axis_geometry={
                10: (elements, 1),
                20: (heads, 1),
                21: (1, width),
                22: (splits, 1),
            },
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=128,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(event.key_count, heads)
        self.assertEqual(event.expected_arrivals, width)
        self.assertEqual(
            event.uses[0].key_by_task,
            tuple(task // splits for task in range(heads * splits)),
        )
        self.assertEqual(len(event.contributors[0].task_to_key_segments), 1)
        self.assertEqual(len(event.uses[0].task_to_key_segments), 1)

    def test_strided_ready_groups_use_exact_coordinates_with_overlapping_hulls(
        self,
    ) -> None:
        columns = 8
        splits = 4
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(2, columns),
                    strides=(columns, 1),
                    block_ids=(None, 10),
                    scales=(1, 1),
                    offsets=(0, 0),
                    full_slice=(True, False),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(2, columns),
                    strides=(columns, 1),
                    block_ids=(None, 20),
                    scales=(1, 1),
                    offsets=(0, 0),
                    full_slice=(True, False),
                ),
            ),
            [[10], [22, 20]],
        )
        task_families = (
            InstantiatedTaskFamily((10,), (10,), ((10, columns),), ((10, 1),)),
            InstantiatedTaskFamily(
                (22, 20),
                (22, 20),
                ((22, splits), (20, columns)),
                ((22, 1), (20, 1)),
            ),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=task_families,
            available_access_ids_by_root=(frozenset((0,)), frozenset((1,))),
            access_program_points={},
            axis_geometry={10: (columns, 1), 20: (columns, 1), 22: (splits, 1)},
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=32,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(event.key_count, columns)
        self.assertEqual(event.expected_arrivals, 1)
        self.assertEqual(
            event.uses[0].key_by_task,
            tuple(task // splits for task in range(columns * splits)),
        )

    def test_multiple_access_events_in_one_root_fall_back_together(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(30, 31),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(30, 32),
                ),
            ),
            [[10, 11], [20, 21], [30]],
        )
        task_families = (
            InstantiatedTaskFamily(
                (10, 11),
                (11, 10),
                ((10, 1), (11, 8)),
                ((10, 1), (11, 16)),
            ),
            InstantiatedTaskFamily(
                (20, 21),
                (21, 20),
                ((20, 1), (21, 8)),
                ((20, 1), (21, 16)),
            ),
            InstantiatedTaskFamily((30,), (30,), ((30, 1),), ((30, 1),)),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            task_families=task_families,
            available_access_ids_by_root=(
                frozenset((0,)),
                frozenset((1,)),
                frozenset((2, 3)),
            ),
            access_program_points={
                2: AccessProgramPoint(
                    ((30, "outer"), (31, "first_inner")),
                    loop_id=7,
                    loop_depth=1,
                    root_statement_index=0,
                ),
                3: AccessProgramPoint(
                    ((30, "outer"), (32, "second_inner")),
                    loop_id=8,
                    loop_depth=1,
                    root_statement_index=1,
                ),
            },
            axis_geometry={
                10: (1, 1),
                11: (8, 16),
                20: (1, 1),
                21: (8, 16),
                30: (1, 1),
                31: (8, 16),
                32: (8, 16),
            },
            excluded_roots=frozenset(),
            preordered_edges=frozenset(),
            physical_worker_limit=4,
        )

        self.assertEqual(schedule.access_cohorts, ())
        self.assertEqual(
            schedule.root_completion_edges,
            frozenset(((0, 2), (1, 2))),
        )

    def test_worker_schedule_handles_independent_components(self) -> None:
        accesses: list[TileAccess] = []
        task_families: list[InstantiatedTaskFamily] = []
        access_points: dict[int, AccessProgramPoint] = {}
        axis_geometry: dict[int, tuple[int, int]] = {}
        available_access_ids: list[frozenset[int]] = []
        for component in range(2):
            root_base = component * 3
            block_base = 10 + component * 30
            access_base = component * 4
            allocation_base = component * 2
            accesses.extend(
                (
                    _access(
                        access_base,
                        root=root_base,
                        allocation_id=allocation_base,
                        kind="store",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base, block_base + 1),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                    _access(
                        access_base + 1,
                        root=root_base + 1,
                        allocation_id=allocation_base,
                        kind="load",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base + 10, block_base + 11),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                    _access(
                        access_base + 2,
                        root=root_base + 1,
                        allocation_id=allocation_base + 1,
                        kind="store",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base + 10, block_base + 11),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                    _access(
                        access_base + 3,
                        root=root_base + 2,
                        allocation_id=allocation_base + 1,
                        kind="load",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base + 20, block_base + 21),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                )
            )
            task_families.extend(
                (
                    InstantiatedTaskFamily(
                        logical_axis_order=(block_base, block_base + 1),
                        physical_axis_order=(block_base + 1, block_base),
                        axis_counts_items=((block_base, 1), (block_base + 1, 8)),
                        block_sizes_items=((block_base, 1), (block_base + 1, 16)),
                    ),
                    InstantiatedTaskFamily(
                        logical_axis_order=(block_base + 10, block_base + 11),
                        physical_axis_order=(block_base + 11, block_base + 10),
                        axis_counts_items=(
                            (block_base + 10, 1),
                            (block_base + 11, 4),
                        ),
                        block_sizes_items=(
                            (block_base + 10, 1),
                            (block_base + 11, 32),
                        ),
                    ),
                    InstantiatedTaskFamily(
                        logical_axis_order=(block_base + 20,),
                        physical_axis_order=(block_base + 20,),
                        axis_counts_items=((block_base + 20, 1),),
                        block_sizes_items=((block_base + 20, 1),),
                    ),
                )
            )
            available_access_ids.extend(
                (
                    frozenset((access_base,)),
                    frozenset((access_base + 1, access_base + 2)),
                    frozenset((access_base + 3,)),
                )
            )
            access_points[access_base + 3] = AccessProgramPoint(
                (
                    (block_base + 20, "outer"),
                    (block_base + 21, "inner"),
                ),
                loop_id=7 + component,
                loop_depth=1,
                root_statement_index=3,
            )
            axis_geometry.update(
                {
                    block_base: (1, 1),
                    block_base + 1: (8, 16),
                    block_base + 10: (1, 1),
                    block_base + 11: (4, 32),
                    block_base + 20: (1, 1),
                    block_base + 21: (4, 32),
                }
            )

        dependency_plan = build_tile_dependency_graph(
            tuple(accesses),
            [list(family.logical_axis_order) for family in task_families],
        )
        kwargs = {
            "dependency_plan": dependency_plan,
            "task_families": tuple(task_families),
            "available_access_ids_by_root": tuple(available_access_ids),
            "access_program_points": access_points,
            "axis_geometry": axis_geometry,
            "excluded_roots": frozenset(),
            "preordered_edges": frozenset(),
            "physical_worker_limit": 8,
        }

        schedule = build_cross_loop_schedule(**kwargs, requested_worker_count=0)
        self.assertEqual(
            sum(
                not schedule.event_graph.event(event.graph_event_index).is_family_done
                for event in schedule.counted_events
                if event.graph_event_index is not None
            ),
            2,
        )
        self.assertEqual(schedule.access_cohorts, ())
        self.assertEqual(
            schedule.task_ready_edges,
            frozenset(((0, 1), (3, 4))),
        )
        self.assertEqual(
            schedule.root_completion_edges,
            frozenset(((1, 2), (4, 5))),
        )
        overlapped = build_cross_loop_schedule(**kwargs, requested_worker_count=6)
        self.assertEqual(overlapped.worker_limit, 6)
        self.assertEqual(
            [
                (plan.producer_root, plan.consumer_root, plan.stage_sizes)
                for plan in overlapped.access_cohorts
            ],
            [(1, 2, (3, 1)), (4, 5, (3, 1))],
        )
        self.assertEqual(overlapped.root_completion_edges, frozenset())
        self.assertEqual(overlapped.worker_schedule.placement(2, 0), (5, 1))
        self.assertEqual(overlapped.worker_schedule.placement(5, 0), (5, 5))

    def test_alias_names_share_an_allocation_dependency(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    tensor_name="base",
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    tensor_name="view",
                    block_ids=(1,),
                ),
            ),
            [[0], [1]],
        )

        self.assertEqual(len(plan.edges), 1)
        self.assertEqual(plan.edges[0].tensor_names, frozenset(("base", "view")))


@onlyBackends(["triton"])
class TestCrossLoopDependencyIntegration(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_unequal_tiles_choose_proven_readiness(self) -> None:
        for batch, width, producer_width, consumer_width in (
            (2, 64, 16, 32),
            (4, 64, 16, 32),
            (2, 64, 32, 16),
        ):
            with self.subTest(
                batch=batch,
                width=width,
                producer_width=producer_width,
                consumer_width=consumer_width,
            ):
                x = torch.arange(
                    batch * width,
                    device=DEVICE,
                    dtype=torch.float32,
                ).reshape(batch, width)
                for launch in range(2):
                    code, out = code_and_output(
                        cartesian_affine_chain,
                        (x + launch,),
                        block_sizes=[1, producer_width, 1, consumer_width],
                        pid_type="persistent_blocked",
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                    torch.testing.assert_close(out, ((x + launch) + 1) * 2)
                self.assertNotIn("tile_dependency_root_completion", code)
                if producer_width < consumer_width:
                    self.assertIn("tile_dependency_continuation_previous", code)
                    self.assertNotIn("tile_dependency_task_wait", code)
                else:
                    self.assertIn("tile_dependency_keyed_event_wait", code)
                    self.assertNotIn("tile_dependency_task_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_continuation_accepts_non_power_of_two_fanin(self) -> None:
        x = torch.arange(2 * 96, device=DEVICE, dtype=torch.float32).reshape(2, 96)
        for launch in range(2):
            code, out = code_and_output(
                three_way_affine_chain,
                (x + launch,),
                block_sizes=[1, 16, 1, 16],
                pid_type="persistent_blocked",
                num_sm_multiplier=1,
                num_warps=1,
            )
            expected_input = x + launch + 1
            expected = (
                expected_input[:, :32]
                + expected_input[:, 32:64]
                + expected_input[:, 64:]
            )
            torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("* tl.cast(3, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_counted_event_supports_chaining(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        for launch in range(2):
            code, out = code_and_output(
                counted_event_chain,
                (x + launch,),
                pid_type="persistent_blocked",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, torch.sum(x + launch + 1).reshape(1))
        self.assertGreaterEqual(code.count("tile_dependency_continuation_previous"), 2)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertIn("tile_dependency_root_completion_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_tiles_keep_exact_task_readiness(self) -> None:
        x = torch.arange(140, device=DEVICE, dtype=torch.float32).reshape(2, 70)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_prefix_uses_counted_continuation(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, (tmp, out) = code_and_output(
                partial_prefix_continuation,
                (x + launch,),
                block_sizes=[16, 32],
                pid_type="persistent_blocked",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(tmp, x + launch + 1)
            torch.testing.assert_close(out, (x[:64] + launch + 1) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("< 4", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_in_place_preserves_unowned_reaching_definition(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, out = code_and_output(
                partial_prefix_in_place_chain,
                (x + launch,),
                block_sizes=[16, 32, 16],
                pid_type="persistent_blocked",
                num_sm_multiplier=1,
                num_warps=1,
            )
            expected = x + launch + 1
            expected = torch.cat((expected[:64] * 2, expected[64:]))
            torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tile_dependency_keyed_event_wait", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_multi_producer_join_uses_one_counted_event(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32)
        y = torch.arange(128, device=DEVICE, dtype=torch.float32) + 3
        for launch in range(2):
            code, out = code_and_output(
                multi_producer_join,
                (x + launch, y + launch),
                block_sizes=[16, 16, 16],
                pid_type="persistent_blocked",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, x + launch + 1 + (y + launch) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(2, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_repeated_join_waits_once_on_a_coalesced_key(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            coalesced_multi_producer_join,
            (x, y),
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack([x + 1 + (y * 2)[:, None] + split for split in range(4)])
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_keyed_event_wait", code)
        self.assertIn("tl.cast(5, tl.uint32)", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_single_producer_fanout_waits_once_per_ready_group(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        code, out = code_and_output(
            coalesced_single_producer_fanout,
            (x,),
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack([x + 1 + split for split in range(4)])
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_keyed_event_wait", code)
        self.assertIn("tl.cast(4, tl.uint32)", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_fan_in_one_nested_continuation_needs_no_counter(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            direct_nested_continuation,
            (x,),
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1).reshape(4, 2).sum(dim=-1) * 2)
        self.assertIn("tl.cast(2, tl.uint32) - 1", code)
        self.assertNotIn("tl.cast(1, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_zero_task_roots_do_not_allocate_task_events(self) -> None:
        x = torch.empty((0, 64), device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        self.assertEqual(out.shape, x.shape)
        self.assertNotIn("tile_dependency_task_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_l2_remapped_roots_use_logical_task_readiness(self) -> None:
        for batch in (3, 4):
            with self.subTest(batch=batch):
                x = torch.arange(
                    batch * 64, device=DEVICE, dtype=torch.float32
                ).reshape(batch, 64)
                code, out = code_and_output(
                    cartesian_affine_chain,
                    (x,),
                    block_sizes=[1, 16, 1, 32],
                    l2_groupings=[2, 2],
                    pid_type="persistent_blocked",
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertNotIn("tile_dependency_task_wait", code)
                self.assertIn("tile_dependency_continuation_previous", code)
                self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_size_one_view_uses_task_readiness(self) -> None:
        x = torch.arange(32 * 128, device=DEVICE, dtype=torch.float32).reshape(32, 128)
        code, out = code_and_output(
            size_one_view_chain,
            (x,),
            block_sizes=[4, 1, 4, 32],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, ((x + 1) * 2).unsqueeze(0))
        self.assertIn("tile_dependency_keyed_event_wait", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nonzero_grid_start_uses_root_completion(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32)
        bound = offset_affine_chain.bind((x,))
        assert bound.host_function is not None
        dependency_plan = bound.host_function.device_ir.tile_dependency_graph
        assert dependency_plan is not None
        self.assertEqual(dependency_plan.events[0].granularity, "root")

        code, out = code_and_output(
            offset_affine_chain,
            (x,),
            block_sizes=[16, 16],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x[32:] + 1) * 2)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_specialized_quotient_retains_static_task_geometry(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            specialized_quotient_chain,
            (x, 8, 2),
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_continuation_task", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_continuation_follows_each_roots_pid_order(self) -> None:
        x = torch.arange(64 * 1024, device=DEVICE, dtype=torch.float32).reshape(
            64, 1024
        )
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            loop_orders=[[1, 0], [0, 1]],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tile_dependency_scheduled_physical_task", code)
        self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_join_combines_both_producers(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        code, out = code_and_output(
            cartesian_affine_join,
            (x,),
            block_sizes=[1, 16, 1, 16, 1, 32],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, x * 2)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_completion", code)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(4, tl.uint32) - 1", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_root_waits_for_multiple_producers(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32).reshape(1, 64)
        code, out = code_and_output(
            singleton_root_join,
            (x,),
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, torch.sum(x * 2, dim=-1))
        self.assertGreaterEqual(code.count("tile_dependency_root_completion_wait"), 2)
        self.assertIn("if tl.program_id(0) == 0:", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_stream_uses_root_completion(self) -> None:
        for batch in (1, 2):
            with self.subTest(batch=batch):
                x = torch.arange(
                    batch * 4096, device=DEVICE, dtype=torch.float32
                ).reshape(batch, 4096)
                code, out = code_and_output(
                    streamed_singleton_reduction,
                    (x,),
                    block_sizes=[1, 16],
                    pid_type="persistent_blocked",
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, torch.sum(x + 1, dim=-1) + x[:, 0] + 1)
                self.assertNotIn("tile_dependency_ordered_group", code)
                if batch == 1:
                    self.assertIn("tile_dependency_root_completion", code)
                else:
                    self.assertIn("tile_dependency_keyed_event_wait", code)
                    self.assertNotIn("tile_dependency_root_completion", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_task_events_are_capture_safe_and_stream_local(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        bound = cartesian_affine_chain.bind((x,))
        compiled = bound.compile_config(
            helion.Config(
                block_sizes=[1, 16, 1, 32],
                pid_type="persistent_blocked",
                num_sm_multiplier=1,
                num_warps=1,
            )
        )
        compiled(x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(x)
        for value in (3.0, 7.0, -2.0):
            x.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, (x + 1) * 2)

        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        inputs = (
            torch.full((2, 64), 11.0, device=DEVICE),
            torch.full((2, 64), 19.0, device=DEVICE),
        )
        outputs = []
        for stream, input_value in zip(streams, inputs, strict=True):
            with torch.cuda.stream(stream):
                outputs.append(compiled(input_value))
        for stream in streams:
            stream.synchronize()
        for input_value, output in zip(inputs, outputs, strict=True):
            torch.testing.assert_close(output, (input_value + 1) * 2)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_grouped_schedule_requires_the_proven_access_order(self) -> None:
        torch.manual_seed(0)
        block_sizes = [1, 16, 1, 16]

        for batch, intermediate, group_size, reverse_groups in (
            (1, 128, 32, False),
            (1, 128, 32, True),
            (2, 128, 32, False),
            (4, 128, 32, False),
            (2, 96, 32, False),
            (2, 128, 64, False),
        ):
            with self.subTest(
                batch=batch,
                intermediate=intermediate,
                group_size=group_size,
                reverse_groups=reverse_groups,
            ):
                # Positive inputs avoid cancellation-dominated relative error;
                # this test is intended to catch readiness failures.
                x = torch.rand((batch, 64), device=DEVICE, dtype=torch.float16)
                w13 = torch.rand(
                    (64, 2 * intermediate), device=DEVICE, dtype=torch.float16
                )
                w2 = torch.rand((intermediate, 64), device=DEVICE, dtype=torch.float16)
                kernel_args = (
                    x,
                    w13,
                    w2,
                    group_size,
                    hl.constexpr(reverse_groups),
                )
                if batch == 1 and not reverse_groups:
                    bound = grouped_affine_chain.bind(kernel_args)
                    assert bound.host_function is not None
                    dependency_plan = (
                        bound.host_function.device_ir.tile_dependency_graph
                    )
                    assert dependency_plan is not None
                    self.assertTrue(
                        all(
                            access.root in (0, 1, 2)
                            for access in dependency_plan.accesses
                        )
                    )
                    downstream_edges = dependency_plan.edges_between(1, 2)
                    self.assertEqual(len(downstream_edges), 2)
                    self.assertEqual(
                        {wait.placement for wait in dependency_plan.waits_for_root(2)},
                        {"access"},
                    )
                    nested_block_ids = {
                        axis.consumer_block_id
                        for edge in downstream_edges
                        for requirement in edge.readiness
                        if requirement.predecessor_map is not None
                        for axis in requirement.predecessor_map.axes
                        if axis.consumer_block_id
                        not in bound.host_function.device_ir.grid_block_ids[2]
                    }
                    self.assertEqual(len(nested_block_ids), 1)
                worker_config = {}
                if not reverse_groups and group_size == 32:
                    producer_tasks = batch * (2 * intermediate // block_sizes[1])
                    consumer_tasks = batch * (64 // block_sizes[3])
                    arrivals_per_key = 2 * group_size // block_sizes[1]
                    minimum_workers = (producer_tasks + consumer_tasks + 1) // 2
                    worker_config[CROSS_LOOP_NUM_WORKERS_CONFIG] = (
                        (minimum_workers + arrivals_per_key - 1)
                        // arrivals_per_key
                        * arrivals_per_key
                    )
                code, out = code_and_output(
                    grouped_affine_chain,
                    kernel_args,
                    block_sizes=block_sizes,
                    pid_type="persistent_blocked",
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=2,
                    **worker_config,
                )

                gate_up = (x.float() @ w13.float()).half()
                gate, up = gate_up.chunk(2, dim=-1)
                groups = intermediate // group_size
                if reverse_groups:
                    gate = (
                        gate.reshape(batch, groups, group_size)
                        .flip(1)
                        .reshape(batch, intermediate)
                    )
                    up = (
                        up.reshape(batch, groups, group_size)
                        .flip(1)
                        .reshape(batch, intermediate)
                    )
                activated = gate.float() * up.float()
                scale = (
                    activated.abs().reshape(batch, groups, group_size).amax(dim=-1) + 1
                )
                activation = activated.half()
                expected = (
                    activation.float().reshape(batch, groups, group_size)
                    * scale[:, :, None]
                ).reshape(batch, intermediate) @ w2.float()
                torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)

                if reverse_groups:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertIn("tile_dependency_root_completion", code)
                elif group_size != 32:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertNotIn("tile_dependency_cohort_wait", code)
                    self.assertIn("tile_dependency_continuation_previous", code)
                    self.assertIn("tile_dependency_keyed_event_wait", code)
                else:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertNotIn("tile_dependency_root_completion", code)
                    self.assertNotIn("tile_dependency_task_wait", code)
                    self.assertTrue(
                        "tile_dependency_continuation_previous" in code
                        or "tile_dependency_keyed_event_wait" in code
                    )
                    self.assertIn("tile_dependency_cohort_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_default_schedule_coarsens_only_the_access_local_use(self) -> None:
        torch.manual_seed(0)
        x = torch.rand((1, 64), device=DEVICE, dtype=torch.float16)
        w13 = torch.rand((64, 256), device=DEVICE, dtype=torch.float16)
        w2 = torch.rand((128, 64), device=DEVICE, dtype=torch.float16)
        kernel_args = (x, w13, w2, 32, hl.constexpr(False))
        bound = grouped_affine_chain.bind(kernel_args)
        workers = bound.config_spec.user_defined_tunables[CROSS_LOOP_NUM_WORKERS_CONFIG]
        self.assertIsInstance(workers, IntegerFragment)
        assert isinstance(workers, IntegerFragment)
        self.assertEqual((workers.low, workers.high), (0, 256))
        self.assertEqual(
            bound.config_spec.default_config()[CROSS_LOOP_NUM_WORKERS_CONFIG],
            0,
        )

        code, out = code_and_output(
            grouped_affine_chain,
            kernel_args,
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=4,
            num_stages=2,
        )

        gate_up = (x.float() @ w13.float()).half()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = gate.float() * up.float()
        scale = activated.abs().reshape(1, 4, 32).amax(dim=-1) + 1
        expected = (
            activated.half().float().reshape(1, 4, 32) * scale[:, :, None]
        ).reshape(1, 128) @ w2.float()
        torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)
        self.assertIn("tile_dependency_root_completion", code)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertNotIn("tile_dependency_cohort_wait", code)


if __name__ == "__main__":
    import unittest

    unittest.main()
