from __future__ import annotations

from typing import Literal

import torch

import helion
from helion._compiler.cross_loop_dependencies import CrossLoopAccess
from helion._compiler.cross_loop_dependencies import LogicalTaskAxis
from helion._compiler.cross_loop_dependencies import TaskFamily
from helion._compiler.cross_loop_dependencies import TileDependencyKind
from helion._compiler.cross_loop_dependencies import build_cross_loop_dependency_plan
from helion._compiler.cross_loop_dependencies import predecessor_task_ids
from helion._compiler.cross_loop_dependencies import prove_uniform_task_partition
from helion._compiler.tile_dependency_planner import AccessProgramPoint
from helion._compiler.tile_dependency_planner import InstantiatedTaskFamily
from helion._compiler.tile_dependency_planner import build_generic_schedule_plan
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    tile_dependency_schedule=helion.TileDependencySchedule(),
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
    masked: bool = False,
    tensor_name: str = "tmp",
) -> CrossLoopAccess:
    return CrossLoopAccess(
        access_id=access_id,
        memory_op_index=access_id,
        graph_id=root,
        root=root,
        allocation_id=allocation_id,
        kind=kind,
        tensor_name=tensor_name,
        tensor_shape=shape,
        tensor_strides=strides,
        storage_offset=0,
        subscript_dims=tuple(range(len(block_ids))),
        subscript_affine_block_ids=block_ids,
        subscript_index_scales=scales,
        subscript_offsets=offsets,
        subscript_is_scalar=scalar or tuple(False for _ in block_ids),
        has_explicit_mask=masked,
    )


class TestCrossLoopDependencies(TestCase):
    def test_dependency_plan_retains_logical_task_families(self) -> None:
        task_families = (
            TaskFamily(
                root=0,
                graph_id=7,
                axes=(LogicalTaskAxis(10, None),),
                access_ids=(0,),
            ),
            TaskFamily(
                root=1,
                graph_id=9,
                axes=(LogicalTaskAxis(20, None),),
                access_ids=(1,),
            ),
        )
        plan = build_cross_loop_dependency_plan(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            task_families=task_families,
        )

        self.assertEqual(plan.task_families, task_families)
        self.assertEqual(plan.task_families[0].logical_axis_order, (10,))

    def test_source_phase_boundary_satisfies_allocation_dependency(self) -> None:
        plan = build_cross_loop_dependency_plan(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            task_families=(
                TaskFamily(0, 0, (LogicalTaskAxis(10, None),)),
                TaskFamily(1, 1, (LogicalTaskAxis(20, None),)),
            ),
            root_phases=(0, 1),
        )

        self.assertEqual(plan.edges, ())

    def test_edge_retains_every_alias_of_the_allocation(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
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
        self.assertEqual(predecessor_map.producer_access_id, 0)
        self.assertEqual(predecessor_map.consumer_access_id, 1)
        self.assertEqual(
            [
                (axis.producer_block_id, axis.consumer_block_id, axis.tensor_dim)
                for axis in predecessor_map.axes
            ],
            [(10, 20, 0)],
        )

    def test_reversed_mapping_falls_back_to_root(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
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

    def test_unequal_tiles_map_to_every_overlapping_producer(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
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

    def test_uniform_partition_rejects_overlap_or_incomplete_coverage(self) -> None:
        overlapping = build_cross_loop_dependency_plan(
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

        identity = build_cross_loop_dependency_plan(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        identity_map = identity.edges[0].readiness[0].predecessor_map
        assert identity_map is not None
        self.assertIsNone(
            prove_uniform_task_partition(
                (identity_map,),
                consumer_axis_order=(20,),
                consumer_axis_counts={20: 3},
                producer_axis_order=(10,),
                producer_axis_counts={10: 8},
                block_sizes={10: 16, 20: 32},
            )
        )

    def test_tile_id_indices_use_scalar_extent(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=0, kind="store", block_ids=(10,)),
                _access(2, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertFalse(edge.is_task_ready)
        self.assertEqual(edge.readiness[0].producer_access_ids, (0, 1))

    def test_masked_store_falls_back_to_root(self) -> None:
        plan = build_cross_loop_dependency_plan(
            (
                _access(0, root=0, kind="store", block_ids=(10,), masked=True),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertEqual(plan.edges[0].readiness[0].granularity, "root")

    def test_nonzero_or_dynamic_grid_start_falls_back_to_root(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
            (
                _access(0, root=0, kind="store"),
                _access(1, root=1, kind="load", block_ids=(1,)),
                _access(2, root=2, kind="store", block_ids=(2,)),
                _access(3, root=3, kind="load", block_ids=(3,)),
            ),
            [[0], [1], [2], [3]],
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

    def test_fanout_keeps_one_edge_per_consumer(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
        plan = build_cross_loop_dependency_plan(
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

    def test_single_consumer_stream_dominates_later_root_ready_access(self) -> None:
        dependency_plan = build_cross_loop_dependency_plan(
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
        schedule = build_generic_schedule_plan(
            dependency_plan=dependency_plan,
            task_families=(
                InstantiatedTaskFamily(
                    root=0,
                    logical_axis_order=(10, 11),
                    physical_axis_order=(10, 11),
                    axis_counts_items=((10, 1), (11, 4)),
                    block_sizes_items=((10, 1), (11, 16)),
                ),
                InstantiatedTaskFamily(
                    root=1,
                    logical_axis_order=(20,),
                    physical_axis_order=(20,),
                    axis_counts_items=((20, 1),),
                    block_sizes_items=((20, 1),),
                ),
            ),
            available_access_ids_by_root=(frozenset((0,)), frozenset((1, 2))),
            access_program_points={
                1: AccessProgramPoint(
                    1,
                    ((20, "outer"), (21, "inner")),
                    loop_id=7,
                    loop_depth=1,
                    root_statement_index=3,
                ),
                2: AccessProgramPoint(
                    2,
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

        self.assertEqual(schedule.task_ready_edges, frozenset(((0, 1),)))
        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.access_cohorts), 1)
        self.assertTrue(schedule.access_cohorts[0].is_per_coordinate)
        self.assertEqual(schedule.access_cohorts[0].milestone_count, 4)

    def test_alias_names_share_an_allocation_dependency(self) -> None:
        plan = build_cross_loop_dependency_plan(
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
                self.assertNotIn("tile_dependency_whole_value", code)
                if producer_width < consumer_width:
                    self.assertIn("tile_dependency_continuation_arrivals", code)
                    self.assertNotIn("tile_dependency_task_wait", code)
                else:
                    self.assertIn("tile_dependency_task_epochs", code)
                    self.assertIn("tile_dependency_task_wait", code)

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
        self.assertIn("tile_dependency_continuation_arrivals", code)
        self.assertIn("* tl.cast(3, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_whole_value", code)

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
        self.assertIn("tile_dependency_task_epochs", code)
        self.assertNotIn("tile_dependency_whole_value", code)

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
        self.assertNotIn("tile_dependency_task_epochs", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_l2_remapped_roots_use_root_completion(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32).reshape(4, 64)
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
        self.assertNotIn("tile_dependency_task_epochs", code)
        self.assertIn("tile_dependency_whole_value", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nonzero_grid_start_uses_root_completion(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32)
        bound = offset_affine_chain.bind((x,))
        assert bound.host_function is not None
        dependency_plan = bound.host_function.device_ir.cross_loop_dependency_plan
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
        self.assertNotIn("tile_dependency_task_epochs", code)
        self.assertIn("tile_dependency_whole_value", code)

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
        self.assertIn("tile_dependency_continuation_arrivals", code)
        self.assertIn("tile_dependency_continuation_physical_task", code)
        self.assertNotIn("tile_dependency_whole_value", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_join_waits_for_both_producers(self) -> None:
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
        self.assertIn("tile_dependency_task_epochs", code)
        self.assertGreaterEqual(code.count("tile_dependency_task_wait"), 2)
        self.assertNotIn("tile_dependency_whole_value", code)

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
        self.assertGreaterEqual(code.count("tile_dependency_singleton_input_wait"), 2)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_stream_uses_graph_derived_per_coordinate_readiness(self) -> None:
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
                    self.assertIn("tile_dependency_cohort_arrivals", code)
                    self.assertIn("tile_dependency_cohort_wait", code)
                    self.assertNotIn("tile_dependency_whole_value", code)
                else:
                    self.assertIn("tile_dependency_whole_value", code)

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
                        bound.host_function.device_ir.cross_loop_dependency_plan
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
                code, out = code_and_output(
                    grouped_affine_chain,
                    kernel_args,
                    block_sizes=block_sizes,
                    pid_type="persistent_blocked",
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=2,
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
                    self.assertIn("tile_dependency_whole_value", code)
                else:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertNotIn("tile_dependency_whole_value", code)
                    self.assertNotIn("tile_dependency_task_wait", code)
                    self.assertIn("tile_dependency_continuation_arrivals", code)
                    self.assertIn("tile_dependency_cohort_arrivals", code)


if __name__ == "__main__":
    import unittest

    unittest.main()
