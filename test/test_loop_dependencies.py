from __future__ import annotations

import ast
import textwrap
from types import SimpleNamespace
from typing import Any
from typing import cast
import unittest
from unittest import mock

import pytest
import torch

import helion
from helion import exc
from helion._compiler import ast_extension
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import LoopType
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._compiler.loop_dependency_checker import TileAccessKind
from helion._compiler.loop_dependency_checker import TileDependencyAnalysis
from helion._compiler.loop_dependency_checker import TileDependencyKind
from helion._compiler.loop_dependency_checker import TileDependencySynchronization
from helion._compiler.loop_dependency_checker import analyze_top_level_tile_dependencies
from helion._compiler.program_id import ForEachProgramID
from helion._compiler.program_id import _ast_fingerprint
from helion._compiler.program_id import _clone_opaque_loop_segment
from helion._compiler.program_id import _clone_opaque_statements
from helion._compiler.program_id import _consumer_affinity_completion_prefixes
from helion._compiler.program_id import _partitioned_materialization_geometry
from helion._compiler.program_id import _prepend_schedule_to_opaque_loop
from helion._compiler.tile_dependency_schedule import build_tile_dependency_stage_graph
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


def _analyze_source(source: str) -> TileDependencyAnalysis:
    module = ast_extension.convert(ast.parse(textwrap.dedent(source)))
    assert isinstance(module, ast.Module)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    for statement in function.body:
        if isinstance(statement, ast.For):
            assert isinstance(statement, ExtendedAST)
            statement._loop_type = LoopType.GRID
    return analyze_top_level_tile_dependencies(function.body)


@helion.kernel(autotune_effort="none")
def tile_dependency_info_across_barrier(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def implicit_tile_dependency_chain(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp[tile] * 2
    return out


@helion.kernel(
    autotune_effort="none",
    tile_dependency_schedule=helion.TileDependencySchedule(),
)
def implicit_tile_dependency_fanout(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.empty_like(x)
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        out0[tile] = tmp[tile] * 2
    for tile in hl.tile(x.size(0)):
        out1[tile] = tmp[tile] * 3
    return out0, out1


@helion.kernel(
    autotune_effort="none",
    tile_dependency_schedule=helion.TileDependencySchedule(),
)
def implicit_tile_dependency_three_stage(x: torch.Tensor) -> torch.Tensor:
    tmp0 = torch.empty_like(x)
    tmp1 = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp0[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        tmp1[tile] = tmp0[tile] * 2
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp1[tile] - 3
    return out


@helion.kernel(
    autotune_effort="none",
    tile_dependency_schedule=helion.TileDependencySchedule(),
)
def implicit_tile_dependency_matmul_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    _, p = c.size()
    tmp = torch.empty([m, n], device=a.device, dtype=a.dtype)
    out = torch.empty([m, p], device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        tmp[tile_m, tile_n] = acc
    for tile_m, tile_p in hl.tile([m, p]):
        acc = hl.zeros([tile_m, tile_p], dtype=torch.float32)
        for tile_reduce_n in hl.tile(n):
            acc = torch.addmm(
                acc,
                tmp[tile_m, tile_reduce_n],
                c[tile_reduce_n, tile_p],
            )
        out[tile_m, tile_p] = acc
    return out


class TestTileDependencyAnalysis(unittest.TestCase):
    def test_partitioned_materialization_geometry_is_shape_derived(self) -> None:
        self.assertEqual(
            _partitioned_materialization_geometry(
                producer_tasks=768,
                finalized_members=40,
                finalize_partition_block=1,
                materialize_tasks=8,
            ),
            (8, 4, 16, 1),
        )
        self.assertEqual(
            _partitioned_materialization_geometry(
                producer_tasks=768,
                finalized_members=40,
                finalize_partition_block=1,
                materialize_tasks=16,
            ),
            (8, 4, 16, 2),
        )
        self.assertIsNone(
            _partitioned_materialization_geometry(
                producer_tasks=17,
                finalized_members=5,
                finalize_partition_block=1,
                materialize_tasks=4,
            )
        )

    def test_reduction_fanout_consumes_an_upstream_producer_once(self) -> None:
        plan = ForEachProgramID._OneWaveReductionFanout(
            producer_root=11,
            reduction_root=12,
            consumer_root=13,
            task_count=32,
            upstream_root=10,
            upstream_tasks=512,
            upstream_tasks_per_partition=16,
        )
        self.assertEqual(plan.start_root, 10)
        self.assertEqual(plan.continuation_roots, frozenset({11, 12, 13}))

    def test_opaque_tile_body_clone_is_structurally_identical(self) -> None:
        body = ast.parse("value = value * 2\nout[index] = value\n").body
        cloned = _clone_opaque_statements(body)
        self.assertEqual(_ast_fingerprint(cloned), _ast_fingerprint(body))
        self.assertIsNot(cloned[0], body[0])

    def test_tile_dependency_loop_staging_preserves_computation(self) -> None:
        loop = ast.parse(
            "for k in tl.range(0, 128, 16):\n"
            "    partial = tl.load(pointer + k)\n"
            "    accumulator = accumulator + partial\n"
        ).body[0]
        assert isinstance(loop, ast.For)
        computation = _ast_fingerprint(loop.body)

        first = _clone_opaque_loop_segment(loop, end=ast.parse("64", mode="eval").body)
        second = _clone_opaque_loop_segment(
            loop, begin=ast.parse("64", mode="eval").body
        )
        waited = _prepend_schedule_to_opaque_loop(
            loop,
            ast.parse("ready = tl.load(counter)\n").body,
            force_serial_pipeline=True,
        )

        self.assertEqual(_ast_fingerprint(first.body), computation)
        self.assertEqual(_ast_fingerprint(second.body), computation)
        self.assertEqual(_ast_fingerprint(waited.body[1:]), computation)

    def test_opaque_tile_body_can_be_outlined_without_rewriting(self) -> None:
        device_function = object.__new__(DeviceFunction)
        device_function.arguments = []
        device_function.wrapper_only_params = []
        cast("Any", device_function).namespace = SimpleNamespace(
            create_name=lambda name, _value: name
        )
        device_function.triton_noinline_helpers = []
        device_function.triton_noinline_helper_constexprs = {}
        device_function._variable_renames = {}
        device_function.dce_vars = []
        cast("Any", device_function).codegen = SimpleNamespace(module_statements=[])
        cast("Any", device_function).helper_manager = SimpleNamespace(
            codegen_helper_functions=list
        )
        body = ast.parse("value = tl.load(pointer)\ntl.store(output, value)\n").body
        computation = _ast_fingerprint(body)
        environment = SimpleNamespace(backend_name="triton")
        with mock.patch.object(CompileEnvironment, "current", return_value=environment):
            helper_name, arguments = device_function.register_triton_noinline_helper(
                "opaque_tile", body, noinline=True
            )
            helper = device_function.codegen_helper_functions()[0]

        self.assertEqual(helper_name, "opaque_tile")
        self.assertEqual(arguments, ())
        self.assertIsInstance(helper, ast.FunctionDef)
        assert isinstance(helper, ast.FunctionDef)
        self.assertEqual(_ast_fingerprint(helper.body), computation)
        self.assertEqual(
            ast.unparse(helper.decorator_list[0]), "triton.jit(noinline=True)"
        )

    def test_schedule_knobs_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "epoch_replicas"):
            helion.TileDependencySchedule(epoch_replicas=0)
        with self.assertRaisesRegex(ValueError, "producer_order"):
            helion.TileDependencySchedule(producer_order="grouped")  # type: ignore[arg-type]

    def test_consumer_major_order_is_derived_from_completion_prefix(self) -> None:
        self.assertEqual(
            _consumer_affinity_completion_prefixes(
                group_count=10,
                resident_workers=16,
                tasks_per_group=4,
                tiles_per_cohort=2,
                cohort_stride=20,
            ),
            (4, 0),
        )
        self.assertEqual(
            _consumer_affinity_completion_prefixes(
                group_count=10,
                resident_workers=40,
                tasks_per_group=4,
                tiles_per_cohort=2,
                cohort_stride=20,
            ),
            (10, 10),
        )

    def test_normalizes_host_views_and_aliases_to_base_storage(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out, n):
                producer_view = tmp.view(n)
                consumer_view = tmp[:].reshape(n)
                for i in range(n):
                    producer_view[i] = x[i]
                for j in range(n):
                    out[j] = consumer_view[j]
            """
        )
        self.assertEqual(len(analysis.tile_dependencies), 1)
        dependency = analysis.tile_dependencies[0]
        self.assertEqual(dependency.name, "tmp")
        self.assertEqual((dependency.producer_root, dependency.consumer_root), (0, 1))

    def test_explicit_store_is_a_producer_write(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out, n):
                for i in range(n):
                    hl.store(tmp, [i], x[i])
                for j in range(n):
                    out[j] = tmp[j]
            """
        )
        self.assertEqual(len(analysis.tile_dependencies), 1)
        self.assertEqual(analysis.tile_dependencies[0].name, "tmp")

    def test_reused_root_local_name_is_not_a_dependency(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out, n):
                for i in range(n):
                    acc = 0
                    acc = acc + x[i]
                    tmp[i] = acc
                for j in range(n):
                    acc = 0
                    acc = acc + tmp[j]
                    out[j] = acc
            """
        )
        self.assertEqual(
            [dependency.name for dependency in analysis.tile_dependencies],
            ["tmp"],
        )

    def test_unsynchronized_read_after_write(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out, n):
                for i in range(n):
                    tmp[i] = x[i]
                for j in range(n):
                    out[j] = tmp[j]
            """
        )
        self.assertEqual(analysis.root_count, 2)
        self.assertEqual(analysis.source_phase_starts, frozenset())
        self.assertEqual(len(analysis.tile_dependencies), 1)
        dependency = analysis.tile_dependencies[0]
        self.assertEqual((dependency.producer_root, dependency.consumer_root), (0, 1))
        self.assertEqual(dependency.name, "tmp")
        self.assertEqual(
            dependency.kinds, frozenset({TileDependencyKind.READ_AFTER_WRITE})
        )
        self.assertIs(
            dependency.synchronization,
            TileDependencySynchronization.UNSYNCHRONIZED,
        )

    def test_tracks_multiple_consumers_without_creating_a_schedule(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out0, out1, n):
                for i in range(n):
                    tmp[i] = x[i]
                for j in range(n):
                    out0[j] = tmp[j]
                for k in range(n):
                    out1[k] = tmp[k]
            """
        )
        self.assertEqual(
            [
                (dependency.producer_root, dependency.consumer_root)
                for dependency in analysis.tile_dependencies
            ],
            [(0, 1), (0, 2)],
        )
        self.assertEqual(len(analysis.unsynchronized_tile_dependencies), 2)

        schedule = build_tile_dependency_stage_graph(analysis)
        self.assertEqual(schedule.stage_by_root, (0, 1, 1))
        self.assertEqual(schedule.implicit_stage_starts, frozenset({1}))

    def test_combines_read_and_write_kinds(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, n):
                for i in range(n):
                    tmp[i] = x[i]
                for j in range(n):
                    tmp[j] += 1
            """
        )
        self.assertEqual(
            analysis.tile_dependencies[0].kinds,
            frozenset(
                {
                    TileDependencyKind.READ_AFTER_WRITE,
                    TileDependencyKind.WRITE_AFTER_WRITE,
                }
            ),
        )

    def test_write_after_read_tracks_every_live_reader(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, out0, out1, n):
                for i in range(n):
                    out0[i] = x[i]
                for j in range(n):
                    out1[j] = x[j]
                for k in range(n):
                    x[k] = 0
            """
        )
        dependencies = {
            (dependency.producer_root, dependency.consumer_root): dependency.kinds
            for dependency in analysis.tile_dependencies
        }
        self.assertEqual(
            dependencies,
            {
                (0, 2): frozenset({TileDependencyKind.WRITE_AFTER_READ}),
                (1, 2): frozenset({TileDependencyKind.WRITE_AFTER_READ}),
            },
        )

    def test_tuple_unpacked_views_alias_base_storage(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out, n):
                left, right = tmp.unbind(0)
                for i in range(n):
                    left[i] = x[i]
                for j in range(n):
                    out[j] = right[j]
            """
        )
        self.assertEqual(len(analysis.tile_dependencies), 1)
        self.assertEqual(analysis.tile_dependencies[0].name, "tmp")
        self.assertEqual(
            [
                (access.storage, access.kind)
                for accesses in analysis.accesses_by_root
                for access in accesses
                if access.storage == "tmp"
            ],
            [
                ("tmp", TileAccessKind.WRITE),
                ("tmp", TileAccessKind.READ),
            ],
        )

    def test_chained_dependencies_require_distinct_phases(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp0, tmp1, out, n):
                for i in range(n):
                    tmp0[i] = x[i]
                for j in range(n):
                    tmp1[j] = tmp0[j]
                for k in range(n):
                    out[k] = tmp1[k]
            """
        )
        schedule = build_tile_dependency_stage_graph(analysis)
        self.assertEqual(schedule.stage_by_root, (0, 1, 2))
        self.assertEqual(schedule.implicit_stage_starts, frozenset({1, 2}))
        self.assertEqual(
            [stage.roots for stage in schedule.stages],
            [(0,), (1,), (2,)],
        )
        self.assertEqual(
            [
                (edge.producer_stage, edge.consumer_stage, edge.storage)
                for edge in schedule.edges
            ],
            [(0, 1, "tmp0"), (1, 2, "tmp1")],
        )
        self.assertEqual(schedule.unsynchronized_edges, schedule.edges)
        self.assertEqual(
            tuple(edge.storage for edge in schedule.edges_between(1, 2)),
            ("tmp1",),
        )

    def test_schedule_rejects_opaque_nonstorage_dependency(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(out, n):
                carry = 0
                for i in range(n):
                    carry = i
                for j in range(n):
                    out[j] = carry
            """
        )
        with self.assertRaisesRegex(
            exc.TileDependencyScheduleError, "opaque access footprint"
        ):
            build_tile_dependency_stage_graph(analysis, helion.TileDependencySchedule())


@onlyBackends(["triton"])
class TestTileDependencyLowering(RefEagerTestBase, TestCase):
    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_source_barrier_remains_distinct_from_implicit_lowering(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = tile_dependency_info_across_barrier.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        analysis = host_function.compiler_state.tile_dependency_analysis
        schedule = host_function.device_ir.tile_dependency_schedule
        assert analysis is not None
        assert schedule is not None
        self.assertEqual(analysis.source_phase_starts, frozenset({1}))
        self.assertEqual(schedule.stage_by_root, (0, 1))
        self.assertEqual(schedule.implicit_stage_starts, frozenset())
        self.assertIs(
            analysis.tile_dependencies[0].synchronization,
            TileDependencySynchronization.SOURCE_BARRIER,
        )

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager mode")
    def test_implicit_dependency_requires_explicit_schedule(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        with pytest.raises(exc.LoopDependencyError, match="tmp"):
            code_and_output(
                implicit_tile_dependency_chain,
                (x,),
                block_sizes=[8, 8],
                pid_type="persistent_blocked",
            )

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager mode")
    def test_one_completion_counter_synchronizes_multiple_consumers(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, outputs = code_and_output(
            implicit_tile_dependency_fanout,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
        )
        out0, out1 = outputs
        torch.testing.assert_close(out0, (x + 1) * 2)
        torch.testing.assert_close(out1, (x + 1) * 3)
        self.assertEqual(code.count("sem='release'"), 1)
        self.assertIn("ld.acquire.gpu.global.u32", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    def test_three_stage_chain(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_three_stage,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
            num_warps=1,
        )
        torch.testing.assert_close(output, (x + 1) * 2 - 3)
        self.assertEqual(code.count("sem='release'"), 2)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    def test_matmul_chain_allows_reused_accumulator_name(self) -> None:
        a = torch.arange(256, device=DEVICE, dtype=torch.float32).reshape(16, 16)
        b = torch.eye(16, device=DEVICE)
        c = torch.eye(16, device=DEVICE)
        code, output = code_and_output(
            implicit_tile_dependency_matmul_chain,
            (a, b, c),
            block_sizes=[16, 16, 16, 16, 16, 16],
            pid_type="persistent_blocked",
            num_warps=4,
        )
        torch.testing.assert_close(output, a, atol=0, rtol=0)
        self.assertIn("ld.acquire.gpu.global.u32", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)


@onlyBackends(["cute"])
class TestLoops(RefEagerTestBase, TestCase):
    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_tile_dependency_analysis_is_attached_to_device_ir(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = tile_dependency_info_across_barrier.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        analysis = host_function.compiler_state.tile_dependency_analysis
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.source_phase_starts, frozenset({1}))
        self.assertEqual(len(analysis.tile_dependencies), 1)
        dependency = analysis.tile_dependencies[0]
        self.assertIs(
            dependency.synchronization,
            TileDependencySynchronization.SOURCE_BARRIER,
        )
        self.assertEqual(
            host_function.device_ir.tile_dependencies,
            analysis.tile_dependencies,
        )
        self.assertEqual(
            host_function.device_ir.tile_accesses,
            analysis.accesses_by_root,
        )

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_loop_dependency_error1(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.size()):
                out[tile] += x[tile]

            for tile in hl.tile(y.size()):
                out[tile] += y[tile]

            return out

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.LoopDependencyError,
            match="Loop dependency detected: 'out' was written in a previous loop.",
        ):
            code_and_output(kernel, (x, y))

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_loop_dependency_error2(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.size()):
                y[tile] += x[tile]

            for tile in hl.tile(y.size()):
                out[tile] += y[tile]

            return out

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.LoopDependencyError,
            match="Loop dependency detected: 'y' was written in a previous loop.",
        ):
            code_and_output(kernel, (x, y))

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_loop_dependency_error3(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] += x[tile]

            x.sum()

            for tile in hl.tile(y.size()):
                y[tile] += y[tile]

            return x + y

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.TopLevelStatementBetweenLoops,
            match="Statements cannot appear between top level loops.",
        ):
            code_and_output(kernel, (x, y))


if __name__ == "__main__":
    unittest.main()
