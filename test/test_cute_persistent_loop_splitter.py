"""Exercise persistent TCGen05 role and mailbox AST builders without a GPU."""

from __future__ import annotations

import ast
import contextlib
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER,
)
from helion._testing.cute_codegen import mock_cuda_unavailable

pytestmark = pytest.mark.cpu_only


class TestPersistentLoopSplitter(unittest.TestCase):
    """Unit tests for the per-tile / post-loop splitters and the
    role-block partitioner on ``Tcgen05PersistentProgramIDs``.

    Codegen tags every statement that depends on per-tile coordinates via
    ``cute_state.register_tcgen05_per_tile_stmts``. The splitter then hoists the
    rest. Statements that read or write a per-tile name (transitively
    seeded from ``virtual_pid_var``) are also kept in the wrapped body, so
    the PID decomposition emitted by ``_decompose_virtual_pid`` doesn't
    have to plumb tagging through every callsite.

    Statements registered via role-specific tcgen05 registration methods
    are routed through ``_collect_tcgen05_role_blocks`` into role blocks;
    everything else stays in shared blocks. The consumer
    ``_emit_role_block_stmts`` wraps non-shared blocks in
    ``if {role_predicate}: ...``.
    """

    _scheduler_environment_entered = False

    def _make_helper(self) -> tuple[object, object]:
        """Construct a real splitter and a minimal device-function stand-in.

        The splitter only needs ``device_function.cute_state``.
        ``Tcgen05PersistentProgramIDs`` is a concrete subclass of
        ``ProgramIDs`` with the splitter; instantiate it without ``__init__``
        to skip the device-function plumbing."""

        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        class _MinimalDeviceFunction:
            def __init__(self) -> None:
                self.cute_state = CuteDeviceFunctionState()

        splitter = Tcgen05PersistentProgramIDs.__new__(Tcgen05PersistentProgramIDs)
        # The splitter walks ASTs looking for references to the
        # virtual_pid var. Tests use simple statements that don't mention
        # any pid, so an unused sentinel name is fine.
        splitter.virtual_pid_var = "__test_virtual_pid__"  # type: ignore[attr-defined]
        # ``_collect_tcgen05_role_blocks`` calls
        # ``_tcgen05_tma_load_role_predicate`` which would normally hit
        # ``DeviceFunction.current().cute_state.matmul_plan``. Replace
        # with a sentinel string the tests can match against, so this
        # unit suite stays decoupled from the real DeviceFunction stack.
        splitter._tcgen05_tma_load_role_predicate = (  # type: ignore[attr-defined]
            lambda: "__test_tma_load_warp__"
        )
        splitter._tcgen05_mma_exec_role_predicate = (  # type: ignore[attr-defined]
            lambda: "__test_mma_exec_warp__"
        )
        splitter._tcgen05_epi_role_predicate = (  # type: ignore[attr-defined]
            lambda: "__test_epi_warp__"
        )
        return splitter, _MinimalDeviceFunction()

    def _stmt(self, text: str) -> ast.stmt:
        return ast.parse(text).body[0]

    def test_stmt_name_uses_augassign_reads_and_writes_target(self) -> None:
        from helion._compiler.program_id import _stmt_name_uses

        reads, writes = _stmt_name_uses(self._stmt("value += increment"))

        self.assertEqual(reads, {"value", "increment"})
        self.assertEqual(writes, {"value"})

    def test_prune_dead_generated_assignments_tracks_transitive_reads(self) -> None:
        from helion._compiler.program_id import (
            _prune_dead_side_effect_free_generated_assignments,
        )

        statements = [
            self._stmt("tile_start = cta_m * 128"),
            self._stmt("valid_m = min(128, problem_m - tile_start)"),
            self._stmt("store_m = min(128, problem_m - tile_start)"),
            self._stmt("pid_0 = cta_m"),
            self._stmt("pid_1 = cta_n"),
        ]

        kept, live_in = _prune_dead_side_effect_free_generated_assignments(
            statements, {"valid_m", "pid_1"}
        )

        self.assertEqual(
            [ast.unparse(stmt) for stmt in kept],
            [
                "tile_start = cta_m * 128",
                "valid_m = min(128, problem_m - tile_start)",
                "pid_1 = cta_n",
            ],
        )
        self.assertEqual(live_in, {"cta_m", "cta_n", "min", "problem_m"})

    def test_literal_mailbox_access_fields_handles_staged_accesses(self) -> None:
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            _literal_mailbox_access_fields,
        )

        tree = ast.parse(
            """
first = mailbox[cutlass.Int32(3)]
second = mailbox[cutlass.Int32(7), consumer_state.index]
mailbox[cutlass.Int32(3), producer_state.index] = first
"""
        )

        reads, writes = _literal_mailbox_access_fields(tree, "mailbox")

        self.assertEqual(reads, {3, 7})
        self.assertEqual(writes, {3})
        with self.assertRaisesRegex(AssertionError, "literal Int32 field"):
            _literal_mailbox_access_fields(
                ast.parse("value = mailbox[field]"), "mailbox"
            )
        for source in (
            "consume(mailbox)",
            "alias = mailbox",
            "value = mailbox.iterator",
        ):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(AssertionError, "direct literal field access"),
            ):
                _literal_mailbox_access_fields(ast.parse(source), "mailbox")

    def test_omit_shared_loop_rejects_post_loop_dependency(self) -> None:
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        splitter, _ = self._make_helper()
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[],
            shared_body_extracted=[self._stmt("final_state = current_state")],
        )

        self.assertFalse(
            splitter._tcgen05_shared_loop_has_meaningful_work(partition, [])
        )
        splitter._assert_tcgen05_omit_shared_loop_safe(partition, [])
        self.assertTrue(
            splitter._tcgen05_shared_loop_has_meaningful_work(
                partition, [self._stmt("cleanup_state = final_state")]
            )
        )
        with self.assertRaisesRegex(AssertionError, "used by post-loop cleanup"):
            splitter._assert_tcgen05_omit_shared_loop_safe(
                partition, [self._stmt("cleanup_state = final_state")]
            )
        with self.assertRaisesRegex(AssertionError, "used by post-loop cleanup"):
            splitter._assert_tcgen05_omit_shared_loop_safe(
                partition, [self._stmt("final_state += 1")]
            )

        scalar_load_partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[],
            shared_body_extracted=[self._stmt("value = tensor.iterator.load()")],
        )
        self.assertFalse(
            splitter._tcgen05_shared_loop_has_meaningful_work(scalar_load_partition, [])
        )
        self.assertTrue(
            splitter._tcgen05_shared_loop_has_meaningful_work(
                scalar_load_partition, [self._stmt("cleanup(value)")]
            )
        )
        stateful_load_partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[],
            shared_body_extracted=[self._stmt("value = state.load()")],
        )
        self.assertTrue(
            splitter._tcgen05_shared_loop_has_meaningful_work(
                stateful_load_partition, []
            )
        )

    def test_role_local_one_shot_scheduler_omits_tile_loop(self) -> None:
        from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        plan = CuteTcgen05MatmulPlan(
            bm=64,
            bn=64,
            bk=128,
            k_tile_count=16,
            cluster_m=1,
            is_two_cta=False,
            uses_role_local_persistent_body=True,
            cta_thread_count=192,
            physical_m_threads=32,
            acc_stage_count=2,
            ab_stage_count=2,
            c_stage_count=2,
            epi_warp_count=4,
            one_shot_role_scheduler=True,
        )
        splitter._tcgen05_plan = lambda: plan  # type: ignore[method-assign]
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__",
            stmts=[self._stmt("tma_pipeline.producer_acquire(state)")],
        )

        emitted = splitter._build_role_local_while(
            stub_df,
            self._make_minimal_layout(),
            role_block,
            scheduler_var_prefix="one_shot_test",
        )
        emitted_src = ast.unparse(emitted)

        self.assertFalse(any(isinstance(node, ast.While) for node in ast.walk(emitted)))
        self.assertIn("producer_acquire(state)", emitted_src)
        self.assertNotIn("advance_to_next_work", emitted_src)
        self.assertNotIn("get_current_work", emitted_src)

    def _make_role_local_stubs(self, *, num_pid_dims: int = 2) -> tuple[object, object]:
        """Build a richer device-function stub plus per-pid stubs that
        the role-local-while builders need (``new_var`` for variable
        allocation, ``num_pids_expr`` for scheduler tile counts, and
        the per-tile / TMA-load state registrations used by
        ``_partition_tcgen05_role_blocks``).

        Returns ``(stub_device_function, splitter)``. The splitter is
        the same one this class's ``_make_helper`` returns, but with
        ``pid_info`` bound to ``num_pid_dims`` stand-in PIDs."""

        if not self._scheduler_environment_entered:
            # Pure AST splitters need no SDK. The scheduler's width and dtype
            # choices need the real CuTe environment, once per test instance.
            stack = contextlib.ExitStack()
            self.addCleanup(stack.close)
            stack.enter_context(mock_cuda_unavailable())
            stack.enter_context(
                CompileEnvironment(torch.device("cpu"), helion.Settings(backend="cute"))
            )
            self._scheduler_environment_entered = True

        class _StubDeviceFunction:
            def __init__(self) -> None:
                self._counter = 0
                self.cute_state = CuteDeviceFunctionState()
                # The mailbox snapshot reads the scheduler wait-mode knob.
                self.config: dict[str, object] = {}

            def new_var(self, name: str) -> str:
                self._counter += 1
                return f"{name}__{self._counter}"

        class _PidStub:
            def static_num_pids(self, *, include_specializations: bool = False) -> int:
                return 16

            def num_pids_expr(
                self, *, is_device: bool, count_dtype: str | None = None
            ) -> str:
                count = str(self.static_num_pids())
                if count_dtype is None:
                    return count
                return CompileEnvironment.current().backend.cast_expr(
                    count, count_dtype
                )

        splitter, _ = self._make_helper()
        splitter.pid_info = [_PidStub() for _ in range(num_pid_dims)]  # type: ignore[attr-defined]
        return _StubDeviceFunction(), splitter

    def _make_minimal_layout(self, *, cluster_m: int = 1) -> object:
        """Build a fake ``_Tcgen05PersistentLayout`` with the minimum
        fields used by the shared-while tile body builder and the
        role-local while builder. The default ``cluster_m=1`` keeps
        the cluster-only branches inert; tests that exercise cluster
        behavior can override."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        return Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout(
            cluster_m=cluster_m,
            scheduler_owner_warp="owner_warp",
            cluster_scheduler_leader="cluster_leader",
            consumer_leader_var="consumer_leader",
            scheduler_leader_predicate="leader_pred",
            tile_sched_params_var="tile_sched_params",
            tile_sched_var="tile_sched",
            work_tile_var="work_tile",
            work_tile_smem_ptr="work_tile_smem_ptr",
            work_tile_smem="work_tile_smem",
            work_tile_smem_tensor="work_tile_smem_t",
            work_tile_coord_vars=["c_0"],
            work_tile_valid_var="work_tile_valid",
            linear_pid_expr="c_0",
            sched_pipeline_mbars="sm",
            sched_pipeline="sp",
            sched_pipeline_producer_group="pg",
            sched_pipeline_consumer_group="cg",
            sched_producer_state="ps",
            sched_consumer_state="cs",
            sched_barrier_ptr="bp",
            sched_peer_rank="pr",
            sched_peer_m="pm",
            refresh_work_tile_stmts=[self._stmt("c_0 = work_tile_smem[0]")],
            work_tile_publish_stmts=[
                self._stmt("work_tile_smem[0] = work_tile.tile_idx[0]")
            ],
            work_tile_consume_stmts=[],
            work_tile_release_stmts=[],
        )

    def test_clustered_pid_mailbox_rendezvous_precedes_first_acquire(self) -> None:
        """Peer mailbox mbarriers must be initialized cluster-wide first."""
        _stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        splitter._tcgen05_l2_swizzle_size = lambda: 1  # type: ignore[attr-defined]
        layout = self._make_minimal_layout(cluster_m=2)
        layout.work_tile_publish_stmts = [self._stmt("sp.producer_acquire(ps)")]

        emitted = splitter._build_tcgen05_persistent_prelude(layout)
        source = "\n".join(ast.unparse(stmt) for stmt in emitted)
        create = source.index("sp = cutlass.pipeline.PipelineAsync.create(")
        arrive = source.index("cutlass.pipeline.pipeline_init_arrive(")
        wait = source.index("cutlass.pipeline.pipeline_init_wait(")
        acquire = source.index("sp.producer_acquire(ps)")

        self.assertLess(create, arrive)
        self.assertLess(arrive, wait)
        self.assertLess(wait, acquire)
        self.assertIn("cute.make_layout((2, 1, 1))", source)

    def _make_shared_mailbox_bridge(
        self, *, cluster_m: int = 2, cluster_n: int = 1, two_cta: bool = False
    ) -> tuple[Any, Any]:
        """Exercise the real layout and shared-loop builders with CPU metadata."""
        from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan

        df, splitter = self._make_role_local_stubs()
        plan = CuteTcgen05MatmulPlan(
            bm=256 if two_cta else 128,
            bn=128,
            bk=128,
            k_tile_count=1,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            is_two_cta=two_cta,
            uses_role_local_persistent_body=True,
            cta_thread_count=192,
            physical_m_threads=32,
            acc_stage_count=2,
            ab_stage_count=2,
            c_stage_count=2,
            epi_warp_count=4,
        )
        splitter._tcgen05_plan = lambda: plan
        return splitter, splitter._build_tcgen05_persistent_layout(df)

    def _assert_shared_mailbox_reader_order(
        self, statements: list[ast.stmt], layout: Any
    ) -> None:
        """All CTA readers must finish before the elected release, even on exit."""

        def call_indices(name: str) -> list[int]:
            return [
                index
                for index, statement in enumerate(statements)
                for node in ast.walk(statement)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == name
            ]

        waits = call_indices(f"{layout.sched_pipeline}.consumer_wait")
        releases = call_indices(f"{layout.sched_pipeline}.consumer_release")
        barriers = call_indices("cute.arch.sync_threads")
        fences = call_indices("cute.arch.fence_view_async_shared")
        self.assertEqual(len(waits), 1)
        self.assertEqual(len(releases), 1)
        self.assertEqual(len(barriers), 2)
        self.assertEqual(len(fences), 1)
        self.assertEqual(call_indices("cute.arch.sync_warp"), [])
        wait, release = statements[waits[0]], statements[releases[0]]
        self.assertIsInstance(wait, ast.If)
        self.assertIsInstance(release, ast.If)
        self.assertEqual(ast.unparse(wait.test), layout.consumer_leader_var)
        self.assertEqual(ast.unparse(release.test), layout.consumer_leader_var)
        self.assertEqual(len(wait.body), 1)
        self.assertEqual(
            [ast.unparse(stmt) for stmt in release.body],
            [
                f"{layout.sched_pipeline}.consumer_release({layout.sched_consumer_state})",
                f"{layout.sched_consumer_state}.advance()",
            ],
        )
        self.assertEqual(wait.orelse, [])
        self.assertEqual(release.orelse, [])
        # The rendezvous and proxy fence must be top-level, not inside the
        # elected-lane, role or valid-tile branches.
        for index in [*barriers, *fences]:
            self.assertIsInstance(statements[index], ast.Expr)
        reads = [
            index
            for index, statement in enumerate(statements)
            if any(
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == layout.work_tile_smem
                and isinstance(node.ctx, ast.Load)
                for node in ast.walk(statement)
            )
        ]
        self.assertEqual(len(reads), 4)
        for index, target in zip(
            reads,
            [*layout.work_tile_coord_vars, layout.work_tile_valid_var],
            strict=True,
        ):
            statement = statements[index]
            self.assertIsInstance(statement, ast.Assign)
            self.assertEqual([ast.unparse(t) for t in statement.targets], [target])
        self.assertLess(waits[0], barriers[0])
        self.assertLess(barriers[0], fences[0])
        self.assertLess(fences[0], reads[0])
        self.assertLess(reads[-1], barriers[1])
        self.assertLess(barriers[1], releases[0])
        # Reading an invalid first/next tile cannot bypass the balancing
        # release. The only release predicate is the elected CTA leader.
        self.assertNotIn(layout.work_tile_valid_var, ast.unparse(release.test))

    def test_shared_mailbox_bridge_initial_and_next_tile_reader_order(self) -> None:
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        for cluster_n in (1, 2):
            for two_cta in (False, True):
                with self.subTest(cluster_n=cluster_n, two_cta=two_cta):
                    splitter, layout = self._make_shared_mailbox_bridge(
                        cluster_n=cluster_n, two_cta=two_cta
                    )
                    prelude = splitter._build_tcgen05_persistent_prelude(layout)
                    shared = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
                        role_predicate=None,
                        stmts=[self._stmt("consume_residual_tile()")],
                    )
                    body = splitter._build_tcgen05_persistent_tile_body(
                        layout, [shared]
                    )
                    self._assert_shared_mailbox_reader_order(prelude, layout)
                    self._assert_shared_mailbox_reader_order(body, layout)
                    self.assertEqual(ast.unparse(body[1]), "consume_residual_tile()")

    def test_shared_mailbox_bridge_retains_single_arrival_and_stage(self) -> None:
        splitter, layout = self._make_shared_mailbox_bridge()
        prelude = splitter._build_tcgen05_persistent_prelude(layout)
        source = "\n".join(ast.unparse(stmt) for stmt in prelude)
        self.assertIn(
            "cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 2)",
            source,
        )
        pipeline_create = next(
            node
            for statement in prelude
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cutlass.pipeline.PipelineAsync.create"
        )
        options = {
            item.arg: ast.unparse(item.value) for item in pipeline_create.keywords
        }
        self.assertEqual(options["num_stages"], "1")
        self.assertEqual(options["consumer_mask"], "cutlass.Int32(0)")
        self.assertEqual(options["defer_sync"], "True")
        self.assertEqual(source.count(".consumer_release("), 1)
        self.assertEqual(source.count(f"{layout.sched_consumer_state}.advance()"), 1)
        self.assertEqual(source.count("_cute_store_shared_remote_x4("), 1)
        self.assertLess(
            source.index("pipeline_init_wait("), source.index(".producer_acquire(")
        )

    def test_shared_mailbox_bridge_requires_all_cta_readers(self) -> None:
        from helion import exc

        splitter, layout = self._make_shared_mailbox_bridge()
        with self.assertRaisesRegex(
            exc.InvalidConfig, "requires CTA-wide synchronization"
        ):
            splitter._build_tcgen05_persistent_tile_body(
                layout, [], emit_block_wide_sync=False
            )

    def test_unclustered_shared_mailbox_keeps_existing_synchronization(self) -> None:
        splitter, layout = self._make_shared_mailbox_bridge(cluster_m=1)
        self.assertEqual(layout.work_tile_consume_stmts, [])
        self.assertEqual(layout.work_tile_release_stmts, [])
        prelude = splitter._build_tcgen05_persistent_prelude(layout)
        for emit_sync in (False, True):
            with self.subTest(emit_sync=emit_sync):
                body = splitter._build_tcgen05_persistent_tile_body(
                    layout, [], emit_block_wide_sync=emit_sync
                )
                source = "\n".join(ast.unparse(stmt) for stmt in body)
                self.assertEqual(
                    source.count("cute.arch.sync_threads()"), int(emit_sync)
                )
                self.assertNotIn("fence_view_async_shared", source)
                self.assertNotIn("consumer_release", source)
                self.assertNotIn("consumer_wait", source)
        source = "\n".join(ast.unparse(stmt) for stmt in prelude)
        self.assertEqual(source.count("cute.arch.sync_threads()"), 1)
        self.assertNotIn("fence_view_async_shared", source)

    def _make_mailbox_release_stubs(
        self, *, staged: bool = False, two_cta: bool = False
    ) -> tuple[Any, Any, Any, Any]:
        """Use the real AST builders and role predicates without a kernel binding."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        df, splitter = self._make_role_local_stubs()
        plan = SimpleNamespace(
            has_scheduler_warp=True,
            has_c_input_warp=True,
            has_store_warp=True,
            tma_warp_id=5,
            ab_load_warp_count=1,
            exec_warp_id=4,
            epi_warp_count=4,
            store_warp_id=7,
            c_input_warp_id=7,
            is_two_cta=two_cta,
            # (no K split over the cluster: ``tcgen05_cluster_split_k``)
            cluster_split_k=1,
            sched_stage_count=2 if staged else 1,
            grouped=None,
            row_union=None,
            source_tile_m=128,
            source_tile_n=128,
            bn=128,
            bm=256 if two_cta else 128,
            accumulator_view="nm",
            output_offsets=("tile_offset_0", "tile_offset_1"),
            tma_store_full_tiles_only=True,
            is_clc_persistent=False,
        )
        splitter._tcgen05_plan = lambda: plan
        splitter._tcgen05_sched_pipeline_plan = lambda: SimpleNamespace(
            pipeline="sched_pipeline", consumer_state="sched_state"
        )
        splitter._tcgen05_is_two_cta = lambda: two_cta
        splitter._tcgen05_cluster_m = lambda: 2 if two_cta else 1
        # Undo this class's partitioner-only predicate stubs. These tests check
        # the actual full-warp gates that enclose the new full-mask rendezvous.
        for name in (
            "_tcgen05_tma_load_role_predicate",
            "_tcgen05_mma_exec_role_predicate",
            "_tcgen05_epi_role_predicate",
        ):
            setattr(
                splitter,
                name,
                getattr(Tcgen05PersistentProgramIDs, name).__get__(splitter),
            )
        layout = self._make_minimal_layout(cluster_m=2 if two_cta else 1)
        return df, splitter, layout, plan

    def _assert_mailbox_release_sequences(self, root: ast.AST, count: int) -> None:
        """Check state convergence after every emitted per-thread arrival.

        Every consumer thread releases after its own mailbox reads, so no
        warp barrier precedes the release; the arrival count covers all lanes.
        """
        found = 0
        for node in ast.walk(root):
            for _field, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                for index, stmt in enumerate(value):
                    if not isinstance(stmt, ast.Expr):
                        continue
                    if ast.unparse(stmt) != (
                        "sched_pipeline.consumer_release(sched_state)"
                    ):
                        continue
                    found += 1
                    self.assertEqual(
                        [ast.unparse(item) for item in value[index + 1 : index + 3]],
                        ["sched_state.advance()", "cute.arch.sync_warp()"],
                    )
        self.assertEqual(found, count)

    def _assert_mailbox_tile_and_terminal(self, root: ast.If) -> ast.While:
        self.assertNotIn("lane_idx", ast.unparse(root.test))
        self.assertIn(
            "cute.arch.make_warp_uniform(cute.arch.warp_idx())", ast.unparse(root.test)
        )
        loops = [stmt for stmt in root.body if isinstance(stmt, ast.While)]
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self._assert_mailbox_release_sequences(loop, 1)
        terminal = ast.Module(
            body=root.body[root.body.index(loop) + 1 :], type_ignores=[]
        )
        self._assert_mailbox_release_sequences(terminal, 1)
        self._assert_mailbox_release_sequences(root, 2)
        # This terminal sequence executes even when the initial valid flag is
        # false, so an empty worklist still balances its single sentinel stage.
        self.assertIsInstance(loop.test, ast.Name)
        self.assertEqual(loop.orelse, [])
        return loop

    def test_mailbox_release_waits_for_all_readers_before_arrival(self) -> None:
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            _build_sched_pipeline_consumer_release_block,
        )

        blocks = [
            _build_sched_pipeline_consumer_release_block(
                sched_pipeline="sched_pipeline", sched_consumer_state="sched_state"
            )
            for _ in range(2)
        ]
        for block in blocks:
            self.assertEqual(len(block), 3)
            self._assert_mailbox_release_sequences(
                ast.Module(body=block, type_ignores=[]), 1
            )
        self.assertTrue(
            {
                id(node)
                for stmt in blocks[0]
                for node in ast.walk(stmt)
                if isinstance(node, ast.stmt)
            }.isdisjoint(
                {
                    id(node)
                    for stmt in blocks[1]
                    for node in ast.walk(stmt)
                    if isinstance(node, ast.stmt)
                }
            )
        )

    def test_mailbox_release_scheduler_roles_and_terminal(self) -> None:
        from helion._compiler.device_function import DeviceFunction

        for staged in (False, True):
            for two_cta in (False, True):
                for warp_leader in (False, True):
                    df, splitter, layout, _plan = self._make_mailbox_release_stubs(
                        staged=staged, two_cta=two_cta
                    )
                    config = (
                        {
                            TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY: TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER
                        }
                        if warp_leader
                        else {}
                    )
                    # The mailbox snapshot reads the wait mode from the device
                    # function while the wait block reads the current one.
                    df.config = config
                    for predicate in (
                        splitter._tcgen05_tma_load_role_predicate(),
                        splitter._tcgen05_mma_exec_role_predicate(),
                        splitter._tcgen05_epi_role_predicate(),
                    ):
                        with self.subTest(
                            staged=staged,
                            two_cta=two_cta,
                            warp_leader=warp_leader,
                            role=predicate,
                        ):
                            role = splitter._PersistentRoleBlock(
                                role_predicate=predicate,
                                stmts=[self._stmt("role_math(__test_virtual_pid__)")],
                            )
                            with patch.object(
                                DeviceFunction,
                                "current",
                                return_value=SimpleNamespace(config=config),
                            ):
                                emitted = (
                                    splitter._build_role_local_while_with_scheduler(
                                        df,
                                        layout,
                                        role,
                                        scheduler_var_prefix="test",
                                        dependency_stmts=[
                                            self._stmt(
                                                "dependent = __test_virtual_pid__"
                                            )
                                        ],
                                    )
                                )
                            loop = self._assert_mailbox_tile_and_terminal(emitted)
                            lines = [ast.unparse(stmt) for stmt in loop.body]
                            if warp_leader:
                                # Only the acquiring lane reads the mailbox; the
                                # body consumes its broadcast register snapshot.
                                self.assertIn("test_mailbox_", lines[0])
                                self.assertNotIn("work_tile_smem[", lines[0])
                            else:
                                self.assertIn(
                                    "work_tile_smem[cutlass.Int32(0), sched_state.index]"
                                    if staged
                                    else "work_tile_smem[cutlass.Int32(0)]",
                                    lines[0],
                                )
                            self.assertIn("consumer_release", lines[1])
                            self.assertLess(
                                1,
                                next(
                                    i
                                    for i, text in enumerate(lines)
                                    if text.startswith("dependent =")
                                ),
                            )
                            self.assertEqual(
                                ast.unparse(emitted).count(
                                    "sched_pipeline.consumer_wait"
                                ),
                                2,
                            )

    def test_mailbox_release_grouped_modes_and_terminal(self) -> None:
        from helion._compiler.cute.grouped_full_coverage import (
            Tcgen05GroupedFullCoveragePlan,
        )
        from helion._compiler.cute.tcgen05.scheduler_family import SchedulerFamily

        for mode in ("off", "dense", "dense_local", "runtime_clc", "runtime_direct"):
            for staged in (False, True):
                for two_cta in (False, True):
                    if mode == "dense_local" and two_cta:
                        continue
                    df, splitter, layout, plan = self._make_mailbox_release_stubs(
                        staged=staged, two_cta=two_cta
                    )
                    runtime_table = mode in ("runtime_clc", "runtime_direct")
                    uses_pipeline = mode != "runtime_direct"
                    fields = (
                        "cta_tile_idx_m",
                        "cta_tile_idx_n",
                        "metadata_idx",
                        "group_idx",
                        "problem_m",
                        "problem_n",
                        "problem_k",
                        "global_m_start",
                        "valid_m",
                        "store_m",
                    )
                    grouped = SimpleNamespace(
                        **{name: f"grouped_{name}" for name in fields}
                    )
                    grouped.full_coverage = (
                        Tcgen05GroupedFullCoveragePlan(
                            predicate="full_coverage",
                            groups=4,
                            m=1024,
                            n=256,
                            k=256,
                            tile_m=128,
                            tile_n=128,
                            tile_k=128 if two_cta else 64,
                            consumer_local=mode == "dense_local",
                        )
                        if mode in ("dense", "dense_local")
                        else None
                    )
                    grouped.runtime_tile_records = (
                        "tile_records" if runtime_table else None
                    )
                    grouped.runtime_total_clusters = (
                        "total_clusters" if runtime_table else None
                    )
                    grouped.device_split_sizes = True
                    grouped.layout = "problem_sizes"
                    plan.grouped = grouped
                    plan.uses_role_local_persistent_body = True
                    family = {
                        "runtime_clc": SchedulerFamily.GROUPED_RUNTIME_CLC,
                        "runtime_direct": SchedulerFamily.GROUPED_RUNTIME_DIRECT,
                    }.get(mode, SchedulerFamily.GROUPED_MAILBOX)
                    splitter._scheduler_family = lambda family=family: family
                    for predicate in (
                        splitter._tcgen05_tma_load_role_predicate(),
                        splitter._tcgen05_mma_exec_role_predicate(),
                        splitter._tcgen05_epi_role_predicate(),
                    ):
                        with self.subTest(
                            mode=mode, staged=staged, two_cta=two_cta, role=predicate
                        ):
                            role = splitter._PersistentRoleBlock(
                                role_predicate=predicate,
                                stmts=[
                                    self._stmt(
                                        "role_math(pid_0, pid_1, grouped_problem_m, grouped_problem_k, grouped_global_m_start)"
                                    )
                                ],
                            )
                            emitted = (
                                splitter._build_grouped_worklist_nm_role_local_while(
                                    df,
                                    role,
                                    layout=layout if uses_pipeline else None,
                                    scheduler_var_prefix="test",
                                    dependency_stmts=None,
                                    role_prelude_stmts=None,
                                    initialize_tile_counter=True,
                                    emit_pdl_wait=True,
                                )
                            )
                            if not uses_pipeline:
                                self._assert_mailbox_release_sequences(emitted, 0)
                                self.assertNotIn("sched_pipeline", ast.unparse(emitted))
                                continue
                            loop = self._assert_mailbox_tile_and_terminal(emitted)
                            lines = [ast.unparse(stmt) for stmt in loop.body]
                            release_index = next(
                                i
                                for i, text in enumerate(lines)
                                if "consumer_release" in text
                            )
                            self.assertLess(
                                release_index,
                                next(
                                    i
                                    for i, text in enumerate(lines)
                                    if text.startswith("role_math(")
                                ),
                            )
                            # Reads of the reusable mailbox must all precede
                            # the arrival. Immutable runtime-record loads may
                            # follow it after the record index is materialized.
                            next_wait_index = next(
                                i
                                for i, text in enumerate(lines)
                                if "sched_pipeline.consumer_wait" in text
                            )
                            self.assertFalse(
                                any(
                                    "work_tile_smem[" in text
                                    for text in lines[
                                        release_index + 1 : next_wait_index
                                    ]
                                )
                            )
                            if mode == "dense_local":
                                release_gate = loop.body[release_index]
                                self.assertIsInstance(release_gate, ast.If)
                                self.assertEqual(
                                    ast.unparse(release_gate.test), "not full_coverage"
                                )
                                terminal = emitted.body[-1]
                                self.assertIsInstance(terminal, ast.If)
                                self.assertEqual(
                                    ast.unparse(terminal.test), "not full_coverage"
                                )

    def _emit_mailbox_aux_role(
        self,
        *,
        staged: bool,
        two_cta: bool,
        phase: str,
        post_l2: bool,
        tma: bool,
        inline: bool = False,
    ) -> tuple[Any, Any, Any, ast.stmt | list[ast.stmt]]:
        df, splitter, layout, plan = self._make_mailbox_release_stubs(
            staged=staged, two_cta=two_cta
        )
        tensor = SimpleNamespace(shape=(1024, 1024), dtype=torch.bfloat16)
        plan.c_input_aux_tensor_descriptors = [
            SimpleNamespace(host_tensor_val=tensor, broadcast_axis=None)
        ]
        df.tensor_arg = lambda value: SimpleNamespace(name="aux_tensor")
        df.cute_state.tcgen05.aux_pipeline_plan = SimpleNamespace(
            pipeline="aux_pipeline",
            producer_state="aux_state",
            epi_tile_var="epi_tile",
            use_tma_load=tma,
            rings=[
                SimpleNamespace(
                    smem="aux_smem",
                    tma_atom="tma_atom" if tma else None,
                    tma_tensor="tma_tensor" if tma else None,
                )
            ],
        )
        dependencies = (
            [
                self._stmt("tile_offset_0 = __test_virtual_pid__ * 128"),
                self._stmt("tile_offset_1 = __test_virtual_pid__ * 64"),
            ]
            if post_l2
            else None
        )
        emitted = splitter._build_c_input_warp_role_local_while(
            df,
            layout,
            shared_body_extracted=dependencies,
            tile_phase=phase,
            inline_aux_only=inline,
        )
        return df, splitter, layout, emitted

    def test_mailbox_release_aux_edge_and_terminal(self) -> None:
        for staged in (False, True):
            for two_cta in (False, True):
                with self.subTest(staged=staged, two_cta=two_cta):
                    _df, _splitter, _layout, emitted = self._emit_mailbox_aux_role(
                        staged=staged,
                        two_cta=two_cta,
                        phase="edge",
                        post_l2=False,
                        tma=False,
                    )
                    self.assertIsInstance(emitted, ast.If)
                    loop = self._assert_mailbox_tile_and_terminal(emitted)
                    self.assertIn("consumer_release", ast.unparse(loop.body[0]))
                    self.assertNotIn("aux_pipeline", ast.unparse(emitted))

    def test_mailbox_release_aux_early_late_and_terminal(self) -> None:
        for staged in (False, True):
            for two_cta in (False, True):
                for post_l2 in (False, True):
                    for tma in (False, True):
                        for phase in ("all", "full"):
                            with self.subTest(
                                staged=staged,
                                two_cta=two_cta,
                                post_l2=post_l2,
                                tma=tma,
                                phase=phase,
                            ):
                                _df, _splitter, _layout, emitted = (
                                    self._emit_mailbox_aux_role(
                                        staged=staged,
                                        two_cta=two_cta,
                                        phase=phase,
                                        post_l2=post_l2,
                                        tma=tma,
                                    )
                                )
                                self.assertIsInstance(emitted, ast.If)
                                loop = self._assert_mailbox_tile_and_terminal(emitted)
                                source = ast.unparse(loop)
                                release = source.index(
                                    "sched_pipeline.consumer_release"
                                )
                                copy = source.index("cute.copy(")
                                if post_l2:
                                    self.assertLess(
                                        source.index("tile_offset_1 ="), release
                                    )
                                    self.assertLess(release, copy)
                                else:
                                    self.assertLess(copy, release)
                                tail = ast.unparse(emitted).count(
                                    "aux_pipeline.producer_tail(aux_state)"
                                )
                                self.assertEqual(tail, int(tma))

    def test_mailbox_release_inline_aux_has_only_parent_handshake(self) -> None:
        for tma in (False, True):
            with self.subTest(tma=tma):
                df, splitter, layout, inline = self._emit_mailbox_aux_role(
                    staged=True,
                    two_cta=False,
                    phase="all",
                    post_l2=True,
                    tma=tma,
                    inline=True,
                )
                self.assertIsInstance(inline, list)
                inline_module = ast.Module(body=inline, type_ignores=[])
                self._assert_mailbox_release_sequences(inline_module, 0)
                self.assertNotIn("sched_pipeline", ast.unparse(inline_module))
                self.assertNotIn("producer_tail", ast.unparse(inline_module))
                role = splitter._PersistentRoleBlock(
                    role_predicate=splitter._tcgen05_epi_role_predicate(),
                    stmts=[self._stmt("epilogue_and_store()")],
                )
                emitted = splitter._build_role_local_while_with_scheduler(
                    df,
                    layout,
                    role,
                    scheduler_var_prefix="inline",
                    dependency_stmts=None,
                    store_aux_per_tile_stmts=inline,
                    store_aux_predicate="cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(7)",
                )
                loop = self._assert_mailbox_tile_and_terminal(emitted)
                source = ast.unparse(loop)
                self.assertLess(
                    source.index("sched_pipeline.consumer_release"),
                    source.index("cute.copy("),
                )

    def test_tcgen05_persistent_foreach_multi_root_keeps_host_guard(self) -> None:
        """Multi-root tcgen05 role-local codegen is guarded as unvalidated.

        ``ForEachProgramID.codegen_grid()`` delegates to the first case's
        grid today. Until tcgen05 grows a scheduler/grid that spans all root
        cases, the validated guard-lift path must exclude multi-root kernels
        and the host guard must count the combined case space.
        """

        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )
        from helion._compiler.program_id import ForEachProgramID

        class _Case:
            def __init__(self, expr: str) -> None:
                self.expr = expr

            def total_pids_expr(self, *, is_device: bool) -> str:
                return self.expr

        fake_pid = ForEachProgramID("pid_shared")
        fake_pid.cases = [_Case("2"), _Case("3")]  # type: ignore[list-item]
        self_stmt = self._stmt("shared_work = 1")

        class _DeviceFunction:
            def __init__(self) -> None:
                self.body = [self_stmt]
                self.pid = fake_pid
                self.codegen = SimpleNamespace(host_statements=[])
                self.cute_state = SimpleNamespace(
                    tcgen05=SimpleNamespace(tma_role_hoist_anchor=None)
                )

        splitter, _ = self._make_helper()
        splitter.virtual_pid_var = "virtual_pid"  # type: ignore[attr-defined]
        layout = self._make_minimal_layout(cluster_m=1)
        role_stmt = self._stmt("role_work = 1")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="role_warp", stmts=[role_stmt]
        )
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[role_block],
            role_blocks_extracted=[role_block],
            shared_body_extracted=[],
        )
        retargeted: list[object] = []
        device_function = _DeviceFunction()

        splitter._extract_tcgen05_post_loop_stmts = (  # type: ignore[attr-defined]
            lambda device, body: (body, [])
        )
        splitter._split_tcgen05_invariant_setup = (  # type: ignore[attr-defined]
            lambda device, body: ([], [], body)
        )
        splitter._build_tcgen05_persistent_layout = (  # type: ignore[attr-defined]
            lambda device: layout
        )
        splitter._partition_tcgen05_role_blocks = (  # type: ignore[attr-defined]
            lambda device, body: partition
        )
        splitter._retarget_tcgen05_shared_scheduler_to_exec = (  # type: ignore[attr-defined]
            lambda layout_arg: retargeted.append(layout_arg)
        )
        splitter._build_tcgen05_persistent_prelude = (  # type: ignore[attr-defined]
            lambda layout_arg: []
        )
        splitter._build_tcgen05_persistent_tile_body_role_local = (  # type: ignore[attr-defined]
            lambda device, layout_arg, partition_arg, **kwargs: ([], [])
        )

        splitter._setup_tcgen05_persistent_kernel(device_function)

        self.assertEqual(retargeted, [layout])
        total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
        host_src = "\n".join(
            ast.unparse(stmt) for stmt in device_function.codegen.host_statements
        )
        self.assertIn(f"{total_var} = 5\n", host_src)
        self.assertIn(f"if {total_var} > 0", host_src)
        self.assertIn("supports runtime execution only", host_src)

    def test_staged_sched_mailbox_rejects_multi_root_shared_loop(self) -> None:
        """Multi-root staged sched mailboxes reject before prelude allocation."""

        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )
        from helion._compiler.program_id import ForEachProgramID

        class _Case:
            def __init__(self, expr: str) -> None:
                self.expr = expr

            def total_pids_expr(self, *, is_device: bool) -> str:
                return self.expr

        fake_pid = ForEachProgramID("pid_shared")
        fake_pid.cases = [_Case("0"), _Case("1")]  # type: ignore[list-item]
        role_blocks = [
            Tcgen05PersistentProgramIDs._PersistentRoleBlock(
                role_predicate="__test_tma_load_warp__",
                stmts=[self._stmt("tma_value = 1")],
            ),
            Tcgen05PersistentProgramIDs._PersistentRoleBlock(
                role_predicate="__test_mma_exec_warp__",
                stmts=[self._stmt("mma_value = 1")],
            ),
            Tcgen05PersistentProgramIDs._PersistentRoleBlock(
                role_predicate="__test_epi_warp__",
                stmts=[self._stmt("epi_value = 1")],
            ),
        ]
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=role_blocks,
            shared_body_extracted=[],
        )
        shared_stmt = self._stmt("shared_work = 1")

        class _DeviceFunction:
            def __init__(self) -> None:
                self.body = [shared_stmt]
                self.pid = fake_pid
                self.codegen = SimpleNamespace(host_statements=[])
                self.cute_state = SimpleNamespace(
                    tcgen05=SimpleNamespace(tma_role_hoist_anchor=None)
                )

        splitter, _ = self._make_helper()
        splitter.virtual_pid_var = "virtual_pid"  # type: ignore[attr-defined]
        layout = self._make_minimal_layout(cluster_m=2)
        device_function = _DeviceFunction()

        splitter._extract_tcgen05_post_loop_stmts = (  # type: ignore[attr-defined]
            lambda device, body: (body, [])
        )
        splitter._split_tcgen05_invariant_setup = (  # type: ignore[attr-defined]
            lambda device, body: ([], [], body)
        )
        splitter._build_tcgen05_persistent_layout = (  # type: ignore[attr-defined]
            lambda device: layout
        )
        splitter._partition_tcgen05_role_blocks = (  # type: ignore[attr-defined]
            lambda device, body: partition
        )
        splitter._tcgen05_uses_staged_work_tile_mailbox = (  # type: ignore[attr-defined]
            lambda: True
        )

        with self.assertRaisesRegex(exc.InvalidConfig, "omitted shared-loop"):
            splitter._setup_tcgen05_persistent_kernel(device_function)

    def test_unmarked_statements_are_hoisted(self) -> None:
        splitter, df = self._make_helper()
        invariant_a = self._stmt("tma_atom_a = make_atom()")
        invariant_b = self._stmt("ab_pipeline = PipelineTmaUmma.create()")
        per_tile_a = self._stmt("gA = cute.local_tile(tA, (128, 16), (m // 128, None))")
        per_tile_b = self._stmt("acc_pipeline.producer_acquire(state)")
        body = [invariant_a, invariant_b, per_tile_a, per_tile_b]
        df.cute_state.register_tcgen05_per_tile_stmts([per_tile_a, per_tile_b])
        hoisted, epi_prelude, wrapped = splitter._split_tcgen05_invariant_setup(
            df, body
        )
        self.assertEqual(hoisted, [invariant_a, invariant_b])
        self.assertEqual(epi_prelude, [])
        self.assertEqual(wrapped, [per_tile_a, per_tile_b])

    def test_relative_order_is_preserved_in_each_slice(self) -> None:
        """The splitter walks ``body`` in order, so hoisted and wrapped
        slices each preserve the input ordering. This is required:
        pipeline ``.create(...)`` must run before the per-tile
        producer_acquire, and consumer_state must be initialized before
        any tile body uses it."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        per_tile = self._stmt("g = local_tile(t, (128, 16), (m, None))")
        b = self._stmt("b = 2")
        df.cute_state.register_tcgen05_per_tile_stmts([per_tile])
        hoisted, epi_prelude, wrapped = splitter._split_tcgen05_invariant_setup(
            df, [a, per_tile, b]
        )
        self.assertEqual(hoisted, [a, b])
        self.assertEqual(epi_prelude, [])
        self.assertEqual(wrapped, [per_tile])

    def test_no_split_when_no_per_tile_marks(self) -> None:
        """With no per-tile registration, the splitter returns the body
        unchanged (no hoisting). The persistent setup will then wrap the
        whole body each iteration. This is the safe default for kernels
        that do not opt into the per-tile splitter."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        hoisted, epi_prelude, wrapped = splitter._split_tcgen05_invariant_setup(
            df, body
        )
        self.assertEqual(hoisted, [])
        self.assertEqual(epi_prelude, [])
        self.assertEqual(wrapped, body)

    def test_post_loop_extraction_removes_marked_stmts(self) -> None:
        """``_extract_tcgen05_post_loop_stmts`` must move post-loop tagged
        statements out of the body so the splitter never sees them. The
        relative order of the remaining body and the extracted post-loop
        statements is preserved independently."""
        splitter, df = self._make_helper()
        per_tile = self._stmt("g = local_tile(t, (128, 16), (m, None))")
        post_a = self._stmt("acc_pipeline.producer_tail(state)")
        post_b = self._stmt("tmem_alloc.free(ptr)")
        df.cute_state.register_tcgen05_per_tile_stmts([per_tile])
        df.cute_state.register_tcgen05_post_loop_stmts([post_a, post_b])
        body = [per_tile, post_a, post_b]
        remaining, post_loop = splitter._extract_tcgen05_post_loop_stmts(df, body)
        self.assertEqual(remaining, [per_tile])
        self.assertEqual(post_loop, [post_a, post_b])

    def test_post_loop_extraction_passthrough_when_unmarked(self) -> None:
        """With no post-loop marks, extraction is a no-op so the splitter
        just sees the original body. Important: this is the non-persistent
        kernel path, where moving statements is incorrect."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        remaining, post_loop = splitter._extract_tcgen05_post_loop_stmts(df, body)
        self.assertEqual(remaining, body)
        self.assertEqual(post_loop, [])

    def test_post_loop_extraction_preserves_relative_order(self) -> None:
        """Both the remaining body and the extracted post-loop list keep
        the input ordering. The splitter never reorders within either
        slice, which lets codegen rely on emit order."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        post_a = self._stmt("acc_pipeline.producer_tail(state)")
        b = self._stmt("b = 2")
        post_b = self._stmt("tmem_alloc.free(ptr)")
        c = self._stmt("c = 3")
        df.cute_state.register_tcgen05_post_loop_stmts([post_a, post_b])
        body = [a, post_a, b, post_b, c]
        remaining, post_loop = splitter._extract_tcgen05_post_loop_stmts(df, body)
        self.assertEqual(remaining, [a, b, c])
        self.assertEqual(post_loop, [post_a, post_b])

    def test_role_blocks_single_block_when_no_tma_load_marks(self) -> None:
        """``_collect_tcgen05_role_blocks`` returns a single shared block
        carrying the full body when no TMA-load tags are present. This is
        the safe default for non-tcgen05 / universal-MMA paths and any
        kernel that never registers TMA-load role tags. The downstream
        consumer ``_build_tcgen05_persistent_tile_body`` handles the
        single-block case identically to the pre-split implementation."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertEqual(blocks[0].stmts, body)

    def test_role_blocks_partition_tagged_runs(self) -> None:
        """Each maximal run of consecutive TMA-load-tagged statements
        becomes its own role block; surrounding shared statements stay
        in shared blocks. Block ordering preserves the input emit order
        of the per-tile body."""
        splitter, df = self._make_helper()
        shared_a = self._stmt("a = 1")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        tma_load_y = self._stmt("cute.copy(t, s)")
        shared_b = self._stmt("b = 2")
        tma_load_z = self._stmt("tma_pipeline.producer_commit(s)")
        df.cute_state.register_tcgen05_per_tile_stmts(
            [tma_load_x, tma_load_y, tma_load_z]
        )
        df.cute_state.register_tcgen05_tma_load_role_stmts(
            [tma_load_x, tma_load_y, tma_load_z]
        )
        body = [shared_a, tma_load_x, tma_load_y, shared_b, tma_load_z]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        self.assertEqual(len(blocks), 4)
        # First block: shared_a (before any tma_load run)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertEqual(blocks[0].stmts, [shared_a])
        # Second block: tma_load_x + tma_load_y (consecutive)
        self.assertEqual(blocks[1].role_predicate, "__test_tma_load_warp__")
        self.assertEqual(blocks[1].stmts, [tma_load_x, tma_load_y])
        # Third block: shared_b
        self.assertIsNone(blocks[2].role_predicate)
        self.assertEqual(blocks[2].stmts, [shared_b])
        # Fourth block: tma_load_z (singleton run)
        self.assertEqual(blocks[3].role_predicate, "__test_tma_load_warp__")
        self.assertEqual(blocks[3].stmts, [tma_load_z])

    def test_role_blocks_partition_distinguishes_warp_roles(self) -> None:
        splitter, df = self._make_helper()
        shared = self._stmt("tile = pid")
        tma_load = self._stmt("tma_pipeline.producer_acquire(s)")
        mma_exec = self._stmt("acc_pipeline.producer_acquire(s)")
        epi = self._stmt("acc_pipeline.consumer_wait(s)")
        df.cute_state.register_tcgen05_per_tile_stmts([tma_load, mma_exec, epi])
        df.cute_state.register_tcgen05_tma_load_role_stmts([tma_load])
        df.cute_state.register_tcgen05_mma_exec_role_stmts([mma_exec])
        df.cute_state.register_tcgen05_epi_role_stmts([epi])

        blocks = splitter._collect_tcgen05_role_blocks(
            df, [shared, tma_load, mma_exec, epi]
        )

        self.assertEqual(
            [block.role_predicate for block in blocks],
            [
                None,
                "__test_tma_load_warp__",
                "__test_mma_exec_warp__",
                "__test_epi_warp__",
            ],
        )
        self.assertEqual(blocks[1].stmts, [tma_load])
        self.assertEqual(blocks[2].stmts, [mma_exec])
        self.assertEqual(blocks[3].stmts, [epi])

    def test_role_blocks_preserve_relative_order_in_emit(self) -> None:
        """Tagged statements stay in their original positions relative
        to surrounding shared statements. This matters because the
        ``tma_initial_full_tile`` boolean is set in a shared statement
        BEFORE the TMA-load-tagged prefetch IF reads it; reordering
        would dangle a free reference and CuTe DSL would error with
        'cannot access free variable ...'."""
        splitter, df = self._make_helper()
        define_bool = self._stmt("ready = True")
        prefetch = self._stmt("if ready and tma_warp:\n    tma_pipeline.acquire()")
        df.cute_state.register_tcgen05_per_tile_stmts([prefetch])
        df.cute_state.register_tcgen05_tma_load_role_stmts([prefetch])
        body = [define_bool, prefetch]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        # define_bool must appear in the FIRST emitted block; prefetch in
        # the SECOND. Reversing the order would put `prefetch` ahead of
        # `define_bool` and `ready` would be undefined when prefetch reads it.
        self.assertEqual(len(blocks), 2)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertEqual(blocks[0].stmts, [define_bool])
        self.assertEqual(blocks[1].role_predicate, "__test_tma_load_warp__")
        self.assertEqual(blocks[1].stmts, [prefetch])

    def test_role_blocks_emit_consumer_wraps_tma_load_in_role_if(self) -> None:
        """The consumer ``_emit_role_block_stmts`` wraps non-shared
        blocks in ``if {role_predicate}: ...`` and emits shared blocks
        as naked statements. An empty block is a no-op (no degenerate
        ``if {}:`` is emitted)."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        splitter, _df = self._make_helper()
        s = self._stmt("a = 1")

        shared = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate=None, stmts=[s]
        )
        out = splitter._emit_role_block_stmts(shared)
        self.assertEqual(out, [s])

        gated = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="my_warp", stmts=[s]
        )
        out = splitter._emit_role_block_stmts(gated)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], ast.If)
        self.assertEqual(ast.unparse(out[0].test), "my_warp")
        self.assertEqual(out[0].body, [s])

        empty_gated = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="my_warp", stmts=[]
        )
        self.assertEqual(splitter._emit_role_block_stmts(empty_gated), [])

    def test_role_blocks_recurse_into_top_level_for_loop(self) -> None:
        """The role partitioner recurses one level into top-level
        ``for`` / ``while`` loop bodies and wraps tagged children in
        ``if {role_predicate}: <child>`` in place. The containing loop
        stays in the shared block. This is structural prep for the
        TMA-load role's role-local-while lift in ``cute_plan.md`` step
        3b: today the per-K-iter producer ``if`` block lives inside
        the K-loop body and gets wrapped by the recursion; step 3b
        lifts it out as a separate top-level statement.
        """
        splitter, df = self._make_helper()
        producer = self._stmt(
            "if tma_full_tile and tma_warp:\n    tma_pipeline.producer_acquire(state)"
        )
        consumer = self._stmt(
            "if tma_full_tile:\n"
            "    if exec_active:\n"
            "        tma_pipeline.consumer_wait(state)"
        )
        kloop = ast.parse("for k in range(K):\n    pass").body[0]
        # Replace the dummy ``pass`` body with the real two child
        # statements so we can tag the producer.
        kloop.body = [producer, consumer]
        # Tag only the producer; the consumer stays shared inside the
        # loop body. The K-loop itself is not tagged -- it's a shared
        # statement that contains a tagged child.
        df.cute_state.register_tcgen05_tma_load_role_stmts([producer])
        body = [kloop]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        # Single shared block carrying the (mutated) K-loop.
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertIs(blocks[0].stmts[0], kloop)
        # Producer is now wrapped with the role predicate; consumer
        # stays naked.
        self.assertEqual(len(kloop.body), 2)
        wrapped = kloop.body[0]
        self.assertIsInstance(wrapped, ast.If)
        self.assertEqual(ast.unparse(wrapped.test), "__test_tma_load_warp__")
        self.assertEqual(wrapped.body, [producer])
        self.assertIs(kloop.body[1], consumer)

    def test_role_blocks_recurse_into_top_level_while_loop(self) -> None:
        """Same recursion behavior for ``while`` as for ``for``. This
        future-proofs the partitioner in case codegen places tagged
        statements inside a top-level ``while`` (e.g. a producer
        loop driven by pipeline state rather than a static range)."""
        splitter, df = self._make_helper()
        producer = self._stmt("tma_pipeline.producer_acquire(state)")
        wloop = ast.parse("while True:\n    pass").body[0]
        wloop.body = [producer]
        df.cute_state.register_tcgen05_tma_load_role_stmts([producer])
        body = [wloop]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertIs(blocks[0].stmts[0], wloop)
        self.assertEqual(len(wloop.body), 1)
        wrapped = wloop.body[0]
        self.assertIsInstance(wrapped, ast.If)
        self.assertEqual(ast.unparse(wrapped.test), "__test_tma_load_warp__")
        self.assertEqual(wrapped.body, [producer])

    def test_role_blocks_no_recurse_into_non_loop_top_level(self) -> None:
        """Recursion is one level deep AND only into ``for`` / ``while``
        statements -- NOT into ``if`` / function bodies / etc. The
        partitioner asserts that every registered tag is consumed, so
        registering a tag inside a non-loop container fails loudly
        instead of silently dropping the role gate. This keeps the
        contract narrow (only for/while are recursed into) and surfaces
        bad registrations at compile time."""
        splitter, df = self._make_helper()
        inner = self._stmt("tma_pipeline.producer_acquire(state)")
        outer_if = self._stmt("if some_predicate:\n    pass")
        outer_if.body = [inner]
        df.cute_state.register_tcgen05_tma_load_role_stmts([inner])
        body = [outer_if]
        with self.assertRaises(AssertionError):
            splitter._collect_tcgen05_role_blocks(df, body)

    def test_virtual_pid_dependencies_are_traced_transitively(self) -> None:
        """The splitter walks each statement's name uses / defines and
        propagates the per-tile property: anything that reads or writes a
        per-tile name is itself per-tile, and anything it assigns is also
        per-tile. Seeded with ``virtual_pid_var``, this captures the PID
        decomposition + downstream offset chain emitted by
        ``_decompose_virtual_pid`` without requiring those sites to
        register themselves."""
        splitter, df = self._make_helper()
        # Pretend cute_mma marked just one statement explicitly per-tile.
        marked = self._stmt("g = cute.local_tile(t, (128, 16), (m, None))")
        df.cute_state.register_tcgen05_per_tile_stmts([marked])

        # Statements that use virtual_pid_var should fall on the
        # wrapped side; downstream statements that consume their
        # writes (pid_0, pid_1 here) should follow.
        invariant_a = self._stmt("smem_a = cute.arch.alloc_smem(F32, 256)")
        decompose_pid = self._stmt("pid_0 = __test_virtual_pid__ % blocks_0")
        decompose_pid_2 = self._stmt("pid_1 = __test_virtual_pid__ // blocks_0")
        derive_m = self._stmt("m_offset = pid_0 * BLOCK_M")
        derive_n = self._stmt("n_offset = pid_1 * BLOCK_N")
        invariant_b = self._stmt("ab_pipeline = PipelineTmaUmma.create()")

        body = [
            invariant_a,
            decompose_pid,
            decompose_pid_2,
            derive_m,
            derive_n,
            marked,
            invariant_b,
        ]
        hoisted, epi_prelude, wrapped = splitter._split_tcgen05_invariant_setup(
            df, body
        )
        # invariant_a and invariant_b have no per-tile dep — hoisted.
        # decompose_pid uses virtual_pid → wrapped, defines pid_0.
        # decompose_pid_2 uses virtual_pid → wrapped, defines pid_1.
        # derive_m uses pid_0 → wrapped, defines m_offset.
        # derive_n uses pid_1 → wrapped, defines n_offset.
        # marked is explicitly per-tile.
        self.assertEqual(hoisted, [invariant_a, invariant_b])
        self.assertEqual(epi_prelude, [])
        self.assertEqual(
            wrapped,
            [decompose_pid, decompose_pid_2, derive_m, derive_n, marked],
        )

    def test_partition_returns_three_views(self) -> None:
        """``_partition_tcgen05_role_blocks`` returns a structured
        ``_PartitionedRoleBody`` with three independent views of the
        same input body:

        - ``role_blocks_inline`` keeps the role-tagged statements in
          their original positions inside the linear block sequence,
          so the legacy inline-weave consumer
          (``_build_tcgen05_persistent_tile_body``) preserves the
          original emit order.
        - ``role_blocks_extracted`` carries each non-shared role
          block as a standalone unit, ready to be lifted into a
          role-local ``while``.
        - ``shared_body_extracted`` is the input body with every
          top-level tagged statement removed (the extract-and-remove
          view), so the role-local-while consumer can wire it
          directly into the shared ``while``.

        Both extracted views are independent of the inline view, so
        mutating one does not affect the other.
        """
        splitter, df = self._make_helper()
        shared_a = self._stmt("a = 1")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        shared_b = self._stmt("b = 2")
        tma_load_y = self._stmt("cute.copy(t, s)")
        shared_c = self._stmt("c = 3")
        df.cute_state.register_tcgen05_per_tile_stmts([tma_load_x, tma_load_y])
        df.cute_state.register_tcgen05_tma_load_role_stmts([tma_load_x, tma_load_y])
        body = [shared_a, tma_load_x, shared_b, tma_load_y, shared_c]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        # Inline view: shared_a / tma_load_x / shared_b / tma_load_y /
        # shared_c interleaved -- 5 blocks with predicate alternating.
        self.assertEqual(len(partition.role_blocks_inline), 5)
        self.assertIsNone(partition.role_blocks_inline[0].role_predicate)
        self.assertEqual(partition.role_blocks_inline[0].stmts, [shared_a])
        self.assertEqual(
            partition.role_blocks_inline[1].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_inline[1].stmts, [tma_load_x])
        self.assertIsNone(partition.role_blocks_inline[2].role_predicate)
        self.assertEqual(partition.role_blocks_inline[2].stmts, [shared_b])
        self.assertEqual(
            partition.role_blocks_inline[3].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_inline[3].stmts, [tma_load_y])
        self.assertIsNone(partition.role_blocks_inline[4].role_predicate)
        self.assertEqual(partition.role_blocks_inline[4].stmts, [shared_c])
        # Extracted view: only the non-shared blocks, decoupled from
        # the surrounding shared statements -- two role blocks since
        # the runs are non-adjacent.
        self.assertEqual(len(partition.role_blocks_extracted), 2)
        self.assertEqual(
            partition.role_blocks_extracted[0].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_extracted[0].stmts, [tma_load_x])
        self.assertEqual(
            partition.role_blocks_extracted[1].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_extracted[1].stmts, [tma_load_y])
        # Shared body: tagged statements have been removed; only the
        # surrounding shared statements remain in their original order.
        self.assertEqual(
            partition.shared_body_extracted, [shared_a, shared_b, shared_c]
        )

    def test_partition_extracted_views_are_independent_lists(self) -> None:
        """Mutating ``role_blocks_extracted`` or
        ``shared_body_extracted`` must not affect the inline view.
        This isolation is required because the role-local-while
        consumer rewrites the extracted view (e.g. wraps statements in
        an ``ast.If``); the inline view must remain usable for the
        legacy inline-weave consumer in the same kernel."""
        splitter, df = self._make_helper()
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        shared_a = self._stmt("a = 1")
        df.cute_state.register_tcgen05_per_tile_stmts([tma_load_x])
        df.cute_state.register_tcgen05_tma_load_role_stmts([tma_load_x])
        body = [tma_load_x, shared_a]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        # Mutate the extracted view's stmts list and confirm the
        # inline view's stmts list is unchanged.
        partition.role_blocks_extracted[0].stmts.append(shared_a)
        self.assertEqual(partition.role_blocks_inline[0].stmts, [tma_load_x])
        # Mutate shared_body_extracted and confirm the inline view's
        # shared block (which carries the same stmt) is unchanged.
        partition.shared_body_extracted.append(tma_load_x)
        self.assertEqual(partition.role_blocks_inline[1].stmts, [shared_a])

    def test_partition_no_marks_returns_full_body_unchanged(self) -> None:
        """Without TMA-load marks, the partition is a degenerate one
        with the full body in a single shared block in the inline
        view, no extracted role blocks, and the full body in
        ``shared_body_extracted``. The role-local-while consumer can
        still call this safely; it just gets zero role-local whiles
        and the full body in the shared while."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        self.assertEqual(len(partition.role_blocks_inline), 1)
        self.assertIsNone(partition.role_blocks_inline[0].role_predicate)
        self.assertEqual(partition.role_blocks_inline[0].stmts, body)
        self.assertEqual(partition.role_blocks_extracted, [])
        self.assertEqual(partition.shared_body_extracted, body)

    def test_partition_recursion_only_mutates_inline_view(self) -> None:
        """When tagged statements live inside top-level ``for`` /
        ``while`` loop bodies, the partitioner mutates the loop body
        in place (wrapping tagged children in
        ``if {role_predicate}:``). This shape applies ONLY to the
        legacy inline-weave consumer; the extracted view must still
        return zero extracted role blocks for the role-local-while
        consumer because the tagged statements are nested inside a
        per-tile container, not at the top level. The role-local
        producer K-loop lift (``cute_plan.md`` step 3b proper) is
        expected to move the tagged producer block to a top-level
        sibling K-loop so it lands in ``role_blocks_extracted``."""
        splitter, df = self._make_helper()
        producer = self._stmt(
            "if tma_full_tile and tma_warp:\n    tma_pipeline.producer_acquire(state)"
        )
        kloop = ast.parse("for k in range(K):\n    pass").body[0]
        kloop.body = [producer]
        df.cute_state.register_tcgen05_tma_load_role_stmts([producer])
        body = [kloop]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        # No top-level tagged stmts -> no extracted role blocks.
        self.assertEqual(partition.role_blocks_extracted, [])
        # Shared body holds the K-loop unchanged in identity.
        self.assertEqual(partition.shared_body_extracted, [kloop])
        # Inline view's single shared block carries the K-loop, and
        # the K-loop body has been mutated to wrap producer in an If.
        self.assertEqual(len(partition.role_blocks_inline), 1)
        self.assertEqual(partition.role_blocks_inline[0].stmts, [kloop])
        self.assertEqual(len(kloop.body), 1)
        wrapped = kloop.body[0]
        self.assertIsInstance(wrapped, ast.If)
        self.assertEqual(ast.unparse(wrapped.test), "__test_tma_load_warp__")

    def test_role_local_while_emits_scheduler_and_loop(self) -> None:
        """``_build_role_local_while`` emits a single top-level
        statement: an ``if {role_predicate}:`` whose body contains
        the role-local scheduler init + a ``while`` loop iterating
        the role's work tiles.

        The structure mirrors Quack's per-role persistent loop in
        ``gemm_sm100.py``: allocate a
        ``StaticPersistentTileScheduler``, peek at the first work
        tile, run the loop body once per tile, then advance and
        refresh. Cross-warp synchronization in the parent kernel is
        through pipeline barriers, not through ``sync_threads`` --
        intentional, because the role-local while runs on a disjoint
        warp set from the shared while."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()

        # Pre-bound role block: a single statement representing the
        # initial-prefetch IF that today lives in the TMA-load role.
        prefetch = self._stmt(
            "if tma_initial_full_tile and tma_warp:\n"
            "    tma_pipeline.producer_acquire(state)\n"
            "    cute.copy(atom, gA, sA)"
        )
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        emitted = splitter._build_role_local_while(
            stub_df, layout, role_block, scheduler_var_prefix="rl_test"
        )

        # Outer ``if {role_predicate}: ...`` wraps the whole role-
        # local while -- only the predicated warps enter.
        self.assertIsInstance(emitted, ast.If)
        self.assertEqual(ast.unparse(emitted.test), "__test_tma_load_warp__")
        # Inside the ``if``: scheduler params, scheduler create,
        # initial work tile, and the while loop. Anything else is
        # premature scaffolding and would surface as a test failure.
        self.assertGreaterEqual(len(emitted.body), 4)
        params_stmt = emitted.body[0]
        sched_stmt = emitted.body[1]
        init_tile_stmt = emitted.body[2]
        while_stmt = emitted.body[-1]
        self.assertIn("PersistentTileSchedulerParams", ast.unparse(params_stmt))
        self.assertIn("StaticPersistentTileScheduler.create", ast.unparse(sched_stmt))
        self.assertIn("initial_work_tile_info", ast.unparse(init_tile_stmt))
        self.assertIsInstance(while_stmt, ast.While)
        # While guard: ``{work_tile_var}.is_valid_tile``.
        self.assertIn("is_valid_tile", ast.unparse(while_stmt.test))
        # Body must include the role block's statement and a
        # scheduler advance + work-tile refresh.
        body_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertIn("producer_acquire(state)", body_src)
        self.assertIn("advance_to_next_work", body_src)
        self.assertIn("get_current_work", body_src)

    def test_role_local_while_assigns_virtual_pid_var(self) -> None:
        """The role-local ``while`` body MUST bind
        ``self.virtual_pid_var`` from the role-local work-tile
        coordinates before running the role block, because the role
        block's statements may reference ``virtual_pid_var``
        transitively (e.g. through PID decomposition). Without this
        binding, the role-local while would dereference a name that
        only the shared ``while`` defines -- a compile-time error
        in the cute-DSL frontend."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()
        prefetch = self._stmt("tma_pipeline.producer_acquire(state)")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        emitted = splitter._build_role_local_while(
            stub_df, layout, role_block, scheduler_var_prefix="rl_pid_test"
        )
        # Role-local while body's first statement must bind
        # virtual_pid_var so the role block's downstream references
        # see it.
        while_stmt = emitted.body[-1]
        first_in_loop = while_stmt.body[0]
        first_src = ast.unparse(first_in_loop)
        self.assertIn(splitter.virtual_pid_var, first_src)

    def test_role_local_while_rejects_shared_role_block(self) -> None:
        """A shared role block has no predicate, so a role-local
        ``while`` would have nothing to gate on. The helper asserts
        this rather than emitting a ``while`` that runs on every
        warp -- which would be incorrect because the shared body
        runs in the shared ``while``, not in any role-local one."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()
        shared_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate=None, stmts=[self._stmt("a = 1")]
        )
        with self.assertRaises(AssertionError):
            splitter._build_role_local_while(
                stub_df, layout, shared_block, scheduler_var_prefix="rl_shared_test"
            )

    def test_role_local_dependency_stmts_pull_pid_offset_chain(self) -> None:
        """Extracted role blocks still need tile-local names that were
        computed by the shared PID-decomposition prefix. The dependency
        helper pulls the nearest shared definitions transitively so the
        role-local loop can compute ``offset_0`` before issuing TMA
        partitions that read it."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()
        pid_0 = self._stmt("pid_0 = __test_virtual_pid__ % num_blocks_0")
        pid_1 = self._stmt("pid_1 = __test_virtual_pid__ // num_blocks_0")
        offset_0 = self._stmt("offset_0 = pid_0 * 128")
        offset_1 = self._stmt("offset_1 = pid_1 * 256")
        unrelated = self._stmt("acc = 0.0")
        tma_load = self._stmt(
            "gA = cute.local_tile(tensor_a, (128, 16), (offset_0, None))"
        )
        body = [pid_0, pid_1, offset_0, offset_1, unrelated, tma_load]
        stub_df.cute_state.register_tcgen05_per_tile_stmts([tma_load])
        stub_df.cute_state.register_tcgen05_tma_load_role_stmts([tma_load])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)

        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )

        self.assertEqual(len(role_local_whiles), 1)
        while_stmt = role_local_whiles[0].body[-1]
        loop_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertIn("pid_0 = __test_virtual_pid__ % num_blocks_0", loop_src)
        self.assertIn("offset_0 = pid_0 * 128", loop_src)
        self.assertNotIn("offset_1 = pid_1 * 256", loop_src)
        self.assertNotIn("acc = 0.0", loop_src)
        self.assertLess(loop_src.find("offset_0 ="), loop_src.find("cute.local_tile"))

    def test_role_local_dependency_stmts_ignore_role_internal_defs(self) -> None:
        """Role-local dependencies should include only free reads.

        The producer K-loop defines and then reads per-iteration names such as
        ``tcgen05_tma_full_tile``. Those reads are local to the extracted role
        body and must not cause the dependency scanner to pull the shared
        consumer K-loop that happens to write the same names.
        """
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()
        pid_0 = self._stmt("pid_0 = __test_virtual_pid__ % num_blocks_0")
        offset_0 = self._stmt("offset_0 = pid_0 * 128")
        shared_kloop = self._stmt(
            "for offset_2 in range(K):\n"
            "    tcgen05_tma_full_tile = offset_2 + 16 <= K\n"
            "    tcgen05_tma_next_full_tile = offset_2 + 32 <= K\n"
            "    tcgen05_ab_producer_try_token = shared_try_token\n"
            "    tcgen05_ab_pipeline.consumer_wait(tcgen05_ab_consumer_state)"
        )
        producer_kloop = self._stmt(
            "for offset_2 in range(K):\n"
            "    tcgen05_tma_k_tile = offset_2 // 16\n"
            "    tcgen05_tma_full_tile = offset_2 + 16 <= K\n"
            "    tcgen05_tma_next_full_tile = offset_2 + 32 <= K\n"
            "    tcgen05_ab_producer_try_token = "
            "tcgen05_ab_pipeline.producer_try_acquire(tcgen05_ab_producer_state)\n"
            "    if tcgen05_tma_full_tile and tcgen05_tma_next_full_tile "
            "and tcgen05_ab_producer_try_token:\n"
            "        tma_tile = offset_0"
        )
        body = [pid_0, offset_0, shared_kloop, producer_kloop]
        stub_df.cute_state.register_tcgen05_per_tile_stmts([producer_kloop])
        stub_df.cute_state.register_tcgen05_tma_load_role_stmts([producer_kloop])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)

        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )

        self.assertEqual(len(role_local_whiles), 1)
        while_stmt = role_local_whiles[0].body[-1]
        loop_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertEqual(loop_src.count("for offset_2 in range(K):"), 1)
        self.assertIn("offset_0 = pid_0 * 128", loop_src)
        self.assertLess(loop_src.find("offset_0 ="), loop_src.find("for offset_2"))
        self.assertNotIn("consumer_wait", loop_src)

    def test_omit_shared_loop_rejects_observable_shared_stmt(self) -> None:
        """Fully role-local CtaGroup.TWO codegen omits the residual shared
        loop. If the tagged-removed shared view ever contains an observable
        operation, the omission must fail loudly instead of dropping it."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        splitter, _ = self._make_helper()
        unsafe_stmts = [
            "cute.copy(src, dst)",
            "tmp = cute.copy(src, dst)",
            "tmp = some_pipeline_call(src)",
            "tmp = min(num_pid_m, 4, key=some_callable)",
            "if side_effecting_call():\n    tmp = 1",
            "for idx in side_effecting_call():\n    tmp = idx",
        ]
        for stmt_src in unsafe_stmts:
            with self.subTest(stmt_src=stmt_src):
                partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
                    role_blocks_inline=[],
                    role_blocks_extracted=[],
                    shared_body_extracted=[self._stmt(stmt_src)],
                )

                with self.assertRaisesRegex(
                    AssertionError, "discard observable shared statement"
                ):
                    splitter._assert_tcgen05_omit_shared_loop_safe(partition)

    def test_role_local_tile_body_can_skip_shared_body_build(self) -> None:
        """When the caller omits the residual shared loop, the role-local
        builder should not construct and discard that body. Dependency-only
        scalar setup is still allowed because role-local loops clone it."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout(cluster_m=2)
        shared_stmt = self._stmt("pid_0 = __test_virtual_pid__")
        role_stmt = self._stmt("role_value = pid_0")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[role_stmt]
        )
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[role_block],
            shared_body_extracted=[shared_stmt],
        )

        def fail_shared_builder(layout_arg: object, role_blocks_arg: object) -> None:
            self.fail("omitted shared loop should not build shared_tile_body")

        splitter._build_tcgen05_persistent_tile_body = fail_shared_builder  # type: ignore[method-assign]

        role_local_whiles, shared_tile_body = (
            splitter._build_tcgen05_persistent_tile_body_role_local(
                stub_df,
                layout,
                partition,
                build_shared_tile_body=False,
                post_loop_stmts=[],
            )
        )

        self.assertEqual(shared_tile_body, [])
        self.assertEqual(len(role_local_whiles), 1)
        while_stmt = role_local_whiles[0].body[-1]
        loop_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertIn("pid_0 = __test_virtual_pid__", loop_src)
        self.assertIn("role_value = pid_0", loop_src)

    def test_omit_shared_loop_allows_grouped_pid_scalar_setup(self) -> None:
        """Grouped PID decomposition and scalar fallback remnants are pure.

        Explicit guarded CtaGroup.TWO configs can still use grouped PID
        decomposition before the host guard raises. The omit-shared-loop
        safety check must allow these dependency-only expressions while
        still rejecting observable calls.
        """
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        splitter, _ = self._make_helper()
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[],
            shared_body_extracted=[
                self._stmt("group_size_m = min(num_pid_m - first_pid_m, 4)"),
                self._stmt(
                    "for offset_2 in range(cutlass.Int32(0), "
                    "cutlass.Int32(16), cutlass.Int32(_BLOCK_SIZE_2)):\n"
                    "    indices_2 = offset_2\n"
                    "    mask_2 = indices_2 < 16\n"
                    "    load = cutlass.BFloat16(0)"
                ),
            ],
        )

        splitter._assert_tcgen05_omit_shared_loop_safe(partition)

    def test_role_local_tile_body_builder_returns_two_lists(self) -> None:
        """``_build_tcgen05_persistent_tile_body_role_local`` returns
        ``(role_local_whiles, shared_tile_body)``. The role-local
        whiles list has one entry per unique role predicate in the
        partition; the shared tile body is the per-tile body for the
        shared ``while`` (without the extracted role blocks). The
        persistent-kernel setup wires this consumer when top-level role
        blocks are extracted."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()

        shared_a = self._stmt("a = 1")
        shared_sync = self._stmt("if needs_barrier:\n    cute.arch.sync_threads()")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        shared_b = self._stmt("b = 2")
        body = [shared_a, shared_sync, tma_load_x, shared_b]
        stub_df.cute_state.register_tcgen05_per_tile_stmts([tma_load_x])
        stub_df.cute_state.register_tcgen05_tma_load_role_stmts([tma_load_x])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)
        role_local_whiles, shared_tile_body = (
            splitter._build_tcgen05_persistent_tile_body_role_local(
                stub_df, layout, partition
            )
        )
        # One role-local while per unique predicate (just one here).
        self.assertEqual(len(role_local_whiles), 1)
        rl_while = role_local_whiles[0]
        self.assertIsInstance(rl_while, ast.If)
        self.assertEqual(ast.unparse(rl_while.test), "__test_tma_load_warp__")
        # The shared tile body must NOT contain the extracted
        # tagged statement.
        shared_src = "\n".join(ast.unparse(s) for s in shared_tile_body)
        self.assertNotIn("producer_acquire(s)", shared_src)
        # The role-local intermediate keeps every warp in the shared while
        # after role-local mainloop work, so existing CTA-wide barriers must
        # be preserved rather than stripped recursively.
        self.assertIn("if needs_barrier:\n    cute.arch.sync_threads()", shared_src)
        self.assertIn("cute.arch.sync_threads()", shared_src)
        # The role-local while body should hold the tagged stmt.
        rl_src = "\n".join(ast.unparse(s) for s in rl_while.body)
        self.assertIn("producer_acquire(s)", rl_src)

    def test_role_local_tile_body_merges_extracted_blocks_by_predicate(
        self,
    ) -> None:
        """Multiple top-level TMA-load-tagged runs separated by shared
        statements partition into multiple ``role_blocks_extracted``
        entries with the same role predicate. The role-local consumer
        MUST merge them into a single role-local ``while`` so per-tile
        ordering of the role's statements is preserved -- emitting one
        loop per extracted block would run all tiles' first chunk
        before the second chunk and break AB-pipeline ordering."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()
        shared_a = self._stmt("a = 1")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s_x)")
        shared_b = self._stmt("b = 2")
        tma_load_y = self._stmt("tma_pipeline.producer_acquire(s_y)")
        shared_c = self._stmt("c = 3")
        body = [shared_a, tma_load_x, shared_b, tma_load_y, shared_c]
        stub_df.cute_state.register_tcgen05_per_tile_stmts([tma_load_x, tma_load_y])
        stub_df.cute_state.register_tcgen05_tma_load_role_stmts(
            [tma_load_x, tma_load_y]
        )
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)
        # Sanity: partitioner produced two extracted blocks.
        self.assertEqual(len(partition.role_blocks_extracted), 2)
        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )
        # Consumer merges them into one loop per unique predicate.
        self.assertEqual(len(role_local_whiles), 1)
        rl_while = role_local_whiles[0]
        loop_body_src = "\n".join(ast.unparse(s) for s in rl_while.body[-1].body)
        # Both tagged stmts must appear in the same loop body, in
        # source order (x before y).
        x_pos = loop_body_src.find("producer_acquire(s_x)")
        y_pos = loop_body_src.find("producer_acquire(s_y)")
        self.assertNotEqual(x_pos, -1)
        self.assertNotEqual(y_pos, -1)
        self.assertLess(x_pos, y_pos)

    def test_role_local_tile_body_orders_tma_before_exec_before_epi(self) -> None:
        """The exec role may be tagged earlier in source order because
        acc-fragment setup is emitted before TMA partition setup. The
        epi role may be tagged even later from the store path. The role-local
        consumer still emits TMA-load before MMA-exec before epi so producer
        work reaches the AB pipeline first and accumulator stages are published
        before epilogue warps consume them."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()
        mma_exec = self._stmt("acc_pipeline.producer_acquire(s_exec)")
        tma_load = self._stmt("tma_pipeline.producer_acquire(s_tma)")
        epi = self._stmt("acc_pipeline.consumer_wait(s_epi)")
        body = [epi, mma_exec, tma_load]
        stub_df.cute_state.register_tcgen05_per_tile_stmts([epi, mma_exec, tma_load])
        stub_df.cute_state.register_tcgen05_epi_role_stmts([epi])
        stub_df.cute_state.register_tcgen05_mma_exec_role_stmts([mma_exec])
        stub_df.cute_state.register_tcgen05_tma_load_role_stmts([tma_load])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)

        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )

        self.assertEqual(len(role_local_whiles), 3)
        self.assertEqual(
            ast.unparse(role_local_whiles[0].test), "__test_tma_load_warp__"
        )
        self.assertEqual(
            ast.unparse(role_local_whiles[1].test), "__test_mma_exec_warp__"
        )
        self.assertEqual(ast.unparse(role_local_whiles[2].test), "__test_epi_warp__")

    def test_role_local_while_uses_layout_cluster_m(self) -> None:
        """The role-local scheduler MUST use the same cluster shape as
        the shared scheduler so it visits tiles in the same order. The
        shared scheduler in ``_build_tcgen05_persistent_prelude`` uses
        ``(layout.cluster_m, 1, 1)``; if ``_build_role_local_while``
        hardcoded ``(1, 1, 1)`` instead, role-local and shared would
        diverge for ``cluster_m > 1`` and break AB-pipeline ordering
        between TMA-load and consumer warps."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout(cluster_m=2)
        prefetch = self._stmt("tma_pipeline.producer_acquire(state)")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        emitted = splitter._build_role_local_while(
            stub_df, layout, role_block, scheduler_var_prefix="rl_cluster_test"
        )
        # The PersistentTileSchedulerParams call site must reference
        # cluster_m=2 in the cluster shape tuple.
        body_srcs = [ast.unparse(stmt) for stmt in emitted.body]
        params_srcs = [
            src for src in body_srcs if "PersistentTileSchedulerParams" in src
        ]
        self.assertEqual(len(params_srcs), 1)
        self.assertIn("(2, 1, 1)", params_srcs[0])

    def test_role_local_while_skips_pdl_wait_without_two_cta(self) -> None:
        """The Quack PDL wait is paired with the two-CTA launch-dependents."""
        from helion._compiler.cute.tcgen05.scheduler_pids import (
            Tcgen05PersistentProgramIDs,
        )

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout(cluster_m=2)
        prefetch = self._stmt("tma_pipeline.producer_acquire(state)")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        with (
            patch.object(
                Tcgen05PersistentProgramIDs,
                "_tcgen05_is_two_cta",
                return_value=False,
            ),
            patch.object(
                Tcgen05PersistentProgramIDs,
                "_tcgen05_tma_load_role_predicate",
                return_value="__test_tma_load_warp__",
            ),
        ):
            emitted = splitter._build_role_local_while(
                stub_df,
                layout,
                role_block,
                scheduler_var_prefix="rl_non_two_cta_pdl",
            )

        self.assertNotIn("cute.arch.griddepcontrol_wait()", ast.unparse(emitted))
