from __future__ import annotations

import ast
import dataclasses
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import unittest

import torch
from torch.fx import Graph

from helion import exc
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.cute_mma import _collective_load_dependency_nodes
from helion._compiler.cute.cutedsl_compat import emit_pipeline_advance
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
from helion._compiler.cute.device_state import CuteTcgen05StoreValue
from helion._compiler.cute.tcgen05_lifecycle import Tcgen05LifecycleContext
from helion._compiler.cute.tcgen05_pure_matmul import Tcgen05PureMatmulObjectModel
from helion._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStoreBodyCoreParams
from helion._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStorePipelineParams
from helion._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStoreSubtileLoopParams
from helion._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStoreTailParams
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.tile_strategy import DeviceLoopState
from helion._testing import onlyBackends
from helion.language.memory_ops import _codegen_cute_store_tcgen05_tile

if TYPE_CHECKING:
    from torch.fx.node import Node


def _plan(**overrides: object) -> CuteTcgen05MatmulPlan:
    kwargs: dict[str, Any] = {
        "bm": 256,
        "bn": 256,
        "bk": 128,
        "k_tile_count": 16,
        "cluster_m": 2,
        "is_two_cta": True,
        "uses_role_local_persistent_body": True,
        "uses_cluster_m2_one_cta_role_local_bridge": False,
        "cta_thread_count": 256,
        "physical_m_threads": 128,
        "acc_stage_count": 2,
        "ab_stage_count": 2,
        "c_stage_count": 2,
        "epi_warp_count": 4,
    }
    kwargs.update(overrides)
    return CuteTcgen05MatmulPlan(**kwargs)


def _lifecycle(**overrides: object) -> Tcgen05LifecycleContext:
    kwargs: dict[str, Any] = {
        "exec_active": "tcgen05_exec",
        "epi_active": "tcgen05_epi",
        "tma_warp": "tcgen05_tma_warp",
        "tma_pipeline": "tcgen05_ab_pipeline",
        "tma_producer_state": "tcgen05_ab_producer_state",
        "acc_pipeline": "tcgen05_acc_pipeline",
        "acc_producer_state": "tcgen05_acc_producer_state",
        "acc_consumer_state": "tcgen05_acc_consumer_state",
        "tmem_alloc_barrier": "tcgen05_tmem_alloc_barrier",
        "tmem_allocator": "tcgen05_tmem_allocator",
        "tmem_holding_buf": "tcgen05_tmem_holding_buf",
        "tmem_dealloc_mbar_ptr": "tcgen05_tmem_dealloc_mbar_ptr",
        "epi_acc_tmem_ptr": "tcgen05_epi_acc_tmem_ptr",
        "acc_tmem_cols": "tcgen05_acc_tmem_cols",
        "is_two_cta": False,
        "use_tma": True,
        "ab_stage_count": 2,
        "acc_stage_count": 1,
    }
    kwargs.update(overrides)
    return Tcgen05LifecycleContext(**kwargs)


def _kloop_with_inner(*stmts: ast.AST) -> DeviceLoopState:
    return DeviceLoopState(
        strategy=cast("Any", SimpleNamespace(block_ids=[0])),
        block_id_to_info={},
        for_node=ast.For(
            target=ast.Name(id="k", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=1)],
                keywords=[],
            ),
            body=[],
            orelse=[],
            type_comment=None,
        ),
        inner_statements=list(stmts),
        block_thread_axes={0: 0},
    )


def _pure_object(loop: DeviceLoopState) -> Tcgen05PureMatmulObjectModel:
    return Tcgen05PureMatmulObjectModel(
        lifecycle_context=_lifecycle(),
        cleanup_loop=loop,
    )


def _pure_store_value(loop: DeviceLoopState) -> CuteTcgen05StoreValue:
    pure_object = _pure_object(loop)
    return CuteTcgen05StoreValue(
        lifecycle_context=pure_object.lifecycle_context,
        pure_matmul_object=pure_object,
    )


def _node_names(nodes: tuple[Node, ...]) -> set[str]:
    return {node.name for node in nodes}


class _StatementCaptureGenerateAST(GenerateAST):
    def __init__(
        self,
        cute_state: CuteDeviceFunctionState,
        device_loop: DeviceLoopState,
        statement_target: list[ast.AST] | None = None,
    ) -> None:
        self.device_function = SimpleNamespace(cute_state=cute_state)
        target = (
            device_loop.inner_statements
            if statement_target is None
            else statement_target
        )
        self.statements_stack = [target]
        self._statement_owner_fx_node = None
        self._device_loop = device_loop

    def _record_statement_thread_references(
        self,
        statements: list[ast.AST],
        axis_sizes: dict[int, int] | None = None,
    ) -> None:
        return

    def _active_loop_stack(self) -> list[DeviceLoopState]:
        return [self._device_loop]


@onlyBackends(["cute"])
class TestCuteDeviceFunctionState(unittest.TestCase):
    def test_get_tcgen05_store_value_checks_candidate_names(self) -> None:
        state = CuteDeviceFunctionState()
        value = CuteTcgen05StoreValue(lifecycle_context=_lifecycle(), bm=256)
        state.register_tcgen05_store_value("matmul_result", value)

        self.assertIsNone(state.get_tcgen05_store_value(["store_value"]))
        self.assertIs(
            state.get_tcgen05_store_value(["store_value", "matmul_result"]),
            value,
        )

    def test_tcgen05_store_value_carries_lifecycle_context(self) -> None:
        state = CuteDeviceFunctionState()
        lifecycle = _lifecycle()
        value = CuteTcgen05StoreValue(lifecycle_context=lifecycle, bm=256)

        state.register_tcgen05_store_value("matmul_result", value)

        loaded = state.get_tcgen05_store_value(["matmul_result"])
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertIs(loaded.lifecycle_context, lifecycle)
        store_value_fields = {field.name for field in dataclasses.fields(loaded)}
        self.assertNotIn("acc_pipeline", store_value_fields)
        self.assertNotIn("tmem_allocator", store_value_fields)

    def test_pure_matmul_object_model_emits_store_cleanup(self) -> None:
        state = CuteDeviceFunctionState()
        owned = statement_from_string("owned = 1")
        loop = _kloop_with_inner(owned)
        pure_object = _pure_object(loop)
        state.register_tcgen05_kloop_owned_stmts(loop, [owned])
        pure_object.register_pending_store(state)
        state.register_tcgen05_store_value(
            "acc",
            CuteTcgen05StoreValue(
                lifecycle_context=pure_object.lifecycle_context,
                pure_matmul_object=pure_object,
            ),
        )

        post_loop_stmts = pure_object.emit_store_post_loop_stmts(state, ["acc"])

        self.assertTrue(post_loop_stmts)
        self.assertTrue(
            all(state.is_tcgen05_post_loop(stmt) for stmt in post_loop_stmts)
        )
        self.assertIn(
            "tcgen05_acc_pipeline.producer",
            "\n".join(ast.unparse(stmt) for stmt in post_loop_stmts),
        )
        self.assertIsInstance(loop.inner_statements[0], ast.Pass)
        with self.assertRaisesRegex(exc.BackendUnsupported, "multi-store fan-out"):
            pure_object.consume_store(state, ["acc"])

    def test_pure_matmul_object_model_emits_store_role(self) -> None:
        state = CuteDeviceFunctionState()
        pure_object = _pure_object(_kloop_with_inner())
        hoisted_stmt = statement_from_string("tcgen05_store_setup = cutlass.Int32(0)")

        role_stmts = pure_object.emit_store_role_stmts(
            state,
            tma_store_hoisted_stmts=[hoisted_stmt],
            store_body_core=["tcgen05_store_body = cutlass.Int32(1)"],
        )

        self.assertIs(role_stmts[0], hoisted_stmt)
        self.assertEqual(len(role_stmts), 4)
        self.assertFalse(state.is_tcgen05_per_tile(hoisted_stmt))
        self.assertTrue(state.is_tcgen05_per_tile(role_stmts[1]))
        self.assertTrue(state.is_tcgen05_per_tile(role_stmts[2]))
        self.assertTrue(state.is_tcgen05_per_tile(role_stmts[3]))
        self.assertTrue(state.is_tcgen05_epi_role(role_stmts[2]))
        self.assertIn("tcgen05_store_body", ast.unparse(role_stmts[2]))

    def test_pure_matmul_object_model_renders_tma_store_tail(self) -> None:
        pure_object = _pure_object(_kloop_with_inner())

        source = pure_object.render_tma_store_tail_region(
            Tcgen05TmaStoreTailParams(
                late_later_subtile_acquire="        if later:\n            acquire()\n",
                epilog_sync_barrier="tcgen05_barrier",
                c_buffer="tcgen05_c_buffer",
                c_buffer_expr="tcgen05_subtile",
                c_stage_count=2,
                tiled_copy_r2s="tcgen05_tiled_copy_r2s",
                trs_rd="tcgen05_tRS_rD",
                trs_sd="tcgen05_tRS_sD",
                warp_idx="tcgen05_warp_idx",
                tma_store_atom="tcgen05_tma_store_atom",
                bsg_sd="tcgen05_bSG_sD",
                bsg_gd="tcgen05_bSG_gD",
                c_pipeline="tcgen05_c_pipeline",
            )
        )

        self.assertIn("        if later:\n            acquire()\n", source)
        self.assertIn(
            "tcgen05_c_buffer = (tcgen05_subtile) % cutlass.Int32(2)",
            source,
        )
        self.assertIn("tcgen05_barrier.arrive_and_wait()", source)
        self.assertIn(
            "cute.copy(tcgen05_tiled_copy_r2s, tcgen05_tRS_rD, "
            "tcgen05_tRS_sD[(None, None, None, tcgen05_c_buffer)])",
            source,
        )
        self.assertIn("cute.arch.fence_view_async_shared()", source)
        self.assertIn(
            "cute.copy(tcgen05_tma_store_atom, "
            "tcgen05_bSG_sD[(None, tcgen05_c_buffer)], "
            "tcgen05_bSG_gD[(None, cutlass.Int32(_tcgen05_subtile))])",
            source,
        )
        self.assertIn("tcgen05_c_pipeline.producer_commit()", source)

    def test_pure_matmul_object_model_renders_c_store_pipeline_ops(self) -> None:
        pure_object = _pure_object(_kloop_with_inner())
        params = Tcgen05TmaStorePipelineParams(
            c_pipeline="tcgen05_c_pipeline",
            warp_idx="tcgen05_warp_idx",
        )

        pre_loop = pure_object.render_c_store_pre_loop_acquire_lines(
            params,
            first_c_acquire_in_loop=False,
        )
        first_in_loop = pure_object.render_c_store_loop_first_acquire(
            params,
            first_c_acquire_in_loop=True,
        )
        later_in_loop = pure_object.render_c_store_loop_later_acquire(
            params,
            later_c_acquire_before_barrier=False,
        )
        late_later = pure_object.render_c_store_loop_late_later_acquire(
            params,
            later_c_acquire_before_barrier=True,
        )
        tail = pure_object.render_c_store_pipeline_tail(params)

        self.assertEqual(
            pre_loop,
            [
                (
                    "if tcgen05_epi and tcgen05_warp_idx == cutlass.Int32(0):\n"
                    "    tcgen05_c_pipeline.producer_acquire()"
                )
            ],
        )
        self.assertIn("_tcgen05_subtile == 0", first_in_loop)
        self.assertIn("tcgen05_c_pipeline.producer_acquire()", first_in_loop)
        self.assertIn("_tcgen05_subtile != 0", later_in_loop)
        self.assertIn("tcgen05_c_pipeline.producer_acquire()", later_in_loop)
        self.assertIn("_tcgen05_subtile != 0", late_later)
        self.assertIn("tcgen05_c_pipeline.producer_acquire()", late_later)
        self.assertIn("tcgen05_c_pipeline.producer_tail()", tail)
        self.assertEqual(
            pure_object.render_c_store_pre_loop_acquire_lines(
                params,
                first_c_acquire_in_loop=True,
            ),
            [],
        )
        self.assertEqual(
            pure_object.render_c_store_loop_first_acquire(
                params,
                first_c_acquire_in_loop=False,
            ),
            "",
        )
        self.assertEqual(
            pure_object.render_c_store_loop_later_acquire(
                params,
                later_c_acquire_before_barrier=True,
            ),
            "",
        )
        self.assertEqual(
            pure_object.render_c_store_loop_late_later_acquire(
                params,
                later_c_acquire_before_barrier=False,
            ),
            "",
        )

    def test_pure_matmul_object_model_renders_acc_consumer_advance(self) -> None:
        pure_object = _pure_object(_kloop_with_inner())

        source = pure_object.render_acc_consumer_advance()

        self.assertEqual(
            source,
            "if tcgen05_epi:\n"
            + emit_pipeline_advance("tcgen05_acc_consumer_state", indent="    "),
        )

    def test_pure_matmul_object_model_renders_tma_store_subtile_loop(self) -> None:
        pure_object = _pure_object(_kloop_with_inner())

        source = pure_object.render_tma_store_subtile_loop(
            Tcgen05TmaStoreSubtileLoopParams(
                subtile_count="tcgen05_subtile_count",
                epi_active="tcgen05_epi_active",
                first_subtile_acquire="        if first:\n            acquire_first()\n",
                later_subtile_acquire="        if later:\n            acquire_later()\n",
                acc_t2r_region_body="        tcgen05_tRS_rD.store(tcgen05_acc_vec)\n",
                tail=Tcgen05TmaStoreTailParams(
                    late_later_subtile_acquire=(
                        "        if late_later:\n            acquire_late()\n"
                    ),
                    epilog_sync_barrier="tcgen05_barrier",
                    c_buffer="tcgen05_c_buffer",
                    c_buffer_expr="tcgen05_subtile",
                    c_stage_count=2,
                    tiled_copy_r2s="tcgen05_tiled_copy_r2s",
                    trs_rd="tcgen05_tRS_rD",
                    trs_sd="tcgen05_tRS_sD",
                    warp_idx="tcgen05_warp_idx",
                    tma_store_atom="tcgen05_tma_store_atom",
                    bsg_sd="tcgen05_bSG_sD",
                    bsg_gd="tcgen05_bSG_gD",
                    c_pipeline="tcgen05_c_pipeline",
                ),
            )
        )

        self.assertIn(
            "for _tcgen05_subtile in cutlass.range("
            "tcgen05_subtile_count, unroll_full=True):",
            source,
        )
        self.assertIn("    if tcgen05_epi_active:", source)
        self.assertIn("        if first:\n            acquire_first()\n", source)
        self.assertIn("        if later:\n            acquire_later()\n", source)
        self.assertIn("tcgen05_tRS_rD.store(tcgen05_acc_vec)", source)
        self.assertIn("        if late_later:\n            acquire_late()\n", source)
        self.assertIn("tcgen05_c_pipeline.producer_commit()", source)

    def test_pure_matmul_object_model_builds_store_body_core(self) -> None:
        pure_object = _pure_object(_kloop_with_inner())

        body_core = pure_object.build_tma_store_body_core(
            Tcgen05TmaStoreBodyCoreParams(
                setup_lines=["tcgen05_setup = cutlass.Int32(0)"],
                subtile_loop=Tcgen05TmaStoreSubtileLoopParams(
                    subtile_count="tcgen05_subtile_count",
                    epi_active="tcgen05_epi_active",
                    first_subtile_acquire="",
                    later_subtile_acquire="",
                    acc_t2r_region_body=(
                        "        tcgen05_tRS_rD.store(tcgen05_acc_vec)\n"
                    ),
                    tail=Tcgen05TmaStoreTailParams(
                        late_later_subtile_acquire="",
                        epilog_sync_barrier="tcgen05_barrier",
                        c_buffer="tcgen05_c_buffer",
                        c_buffer_expr="tcgen05_subtile",
                        c_stage_count=2,
                        tiled_copy_r2s="tcgen05_tiled_copy_r2s",
                        trs_rd="tcgen05_tRS_rD",
                        trs_sd="tcgen05_tRS_sD",
                        warp_idx="tcgen05_warp_idx",
                        tma_store_atom="tcgen05_tma_store_atom",
                        bsg_sd="tcgen05_bSG_sD",
                        bsg_gd="tcgen05_bSG_gD",
                        c_pipeline="tcgen05_c_pipeline",
                    ),
                ),
                pipeline_tail_lines=["tcgen05_c_pipeline.producer_tail()"],
            )
        )

        self.assertEqual(len(body_core), 3)
        self.assertEqual(body_core[0], "tcgen05_setup = cutlass.Int32(0)")
        self.assertIn("for _tcgen05_subtile in cutlass.range", body_core[1])
        self.assertIn("tcgen05_tRS_rD.store(tcgen05_acc_vec)", body_core[1])
        self.assertIn(pure_object.render_acc_consumer_advance(), body_core[1])
        self.assertEqual(body_core[2], "tcgen05_c_pipeline.producer_tail()")

    def test_pure_lifecycle_store_rejects_extra_mask_before_fallback(self) -> None:
        cute_state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()
        cute_state.register_tcgen05_store_value(
            "acc",
            _pure_store_value(loop),
        )
        state = SimpleNamespace(
            device_function=SimpleNamespace(
                cute_state=cute_state,
                variable_aliases=lambda name: (name,),
            )
        )

        with self.assertRaisesRegex(exc.BackendUnsupported, "extra store mask"):
            _codegen_cute_store_tcgen05_tile(
                cast("Any", state),
                torch.empty(1, 1),
                (),
                (),
                ast.Name(id="mask", ctx=ast.Load()),
                "acc",
            )

    def test_pure_lifecycle_store_rejects_rank_before_fallback(self) -> None:
        cute_state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()
        cute_state.register_tcgen05_store_value(
            "acc",
            _pure_store_value(loop),
        )
        state = SimpleNamespace(
            device_function=SimpleNamespace(
                cute_state=cute_state,
                variable_aliases=lambda name: (name,),
            )
        )

        with self.assertRaisesRegex(exc.BackendUnsupported, "rank-2 tensor target"):
            _codegen_cute_store_tcgen05_tile(
                cast("Any", state),
                torch.empty(1, 1, 1),
                (),
                (),
                None,
                "acc",
            )

    def test_tcgen05_lifecycle_context_renders_store_post_loop(self) -> None:
        lifecycle = _lifecycle()

        lines = lifecycle.render_store_post_loop_lines(
            tma_store_pipeline_tail="if tcgen05_warp0:\n    tcgen05_c_pipeline.producer_tail()"
        )

        joined = "\n".join(lines)
        self.assertEqual(
            lines[0],
            "if tcgen05_warp0:\n    tcgen05_c_pipeline.producer_tail()",
        )
        self.assertIn("tcgen05_ab_pipeline.producer", joined)
        self.assertIn("tcgen05_acc_pipeline.producer", joined)
        self.assertIn("tcgen05_tmem_allocator = cutlass.utils.TmemAllocator", joined)
        self.assertIn("tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)", joined)
        self.assertIn("cute.arch.sync_threads()", lines)

    def test_tcgen05_lifecycle_context_two_cta_skips_cta_sync(self) -> None:
        lifecycle = _lifecycle(is_two_cta=True)

        lines = lifecycle.render_store_post_loop_lines()
        joined = "\n".join(lines)

        self.assertIn("cute.arch.griddepcontrol_launch_dependents()", joined)
        self.assertNotIn("cute.arch.sync_threads()", lines)

    def test_register_tcgen05_matmul_plan_rejects_mixed_plan(self) -> None:
        state = CuteDeviceFunctionState()
        state.register_tcgen05_matmul_plan(_plan())

        with self.assertRaisesRegex(
            exc.BackendUnsupported, "mixed tcgen05 matmul collective plans"
        ):
            state.register_tcgen05_matmul_plan(_plan(bn=128))

    def test_root_lane_loop_suppression_is_one_shot(self) -> None:
        state = CuteDeviceFunctionState()

        self.assertFalse(state.consume_root_lane_loop_suppression())
        state.request_root_lane_loop_suppression()
        self.assertTrue(state.consume_root_lane_loop_suppression())
        self.assertFalse(state.consume_root_lane_loop_suppression())

    def test_collective_handled_load_names_stay_for_load_suppression(self) -> None:
        state = CuteDeviceFunctionState()

        state.register_collective_handled_load("operand_load")

        self.assertTrue(state.is_collective_handled_load("operand_load"))
        self.assertFalse(state.is_collective_handled_load("indices"))

    def test_collective_load_dependency_names_keeps_pair_shared_ancestors(
        self,
    ) -> None:
        graph = Graph()
        source = graph.placeholder("source")
        shared_index = graph.call_function(operator.neg, (source,))
        lhs_load = graph.call_function(operator.pos, (shared_index,))
        rhs_load = graph.call_function(operator.abs, (shared_index,))

        self.assertIn(
            shared_index.name,
            _node_names(
                _collective_load_dependency_nodes(
                    lhs_load,
                    {shared_index, lhs_load, rhs_load},
                    {lhs_load, rhs_load},
                )
            ),
        )

    def test_collective_load_dependency_names_keeps_indirect_pair_shared_ancestors(
        self,
    ) -> None:
        graph = Graph()
        source = graph.placeholder("source")
        shared_base = graph.call_function(operator.neg, (source,))
        lhs_index = graph.call_function(operator.pos, (shared_base,))
        rhs_index = graph.call_function(operator.abs, (shared_base,))
        lhs_load = graph.call_function(operator.invert, (lhs_index,))
        rhs_load = graph.call_function(operator.truth, (rhs_index,))

        self.assertIn(
            shared_base.name,
            _node_names(
                _collective_load_dependency_nodes(
                    lhs_load,
                    {shared_base, lhs_index, rhs_index, lhs_load, rhs_load},
                    {lhs_load, rhs_load},
                )
            ),
        )

    def test_collective_load_dependency_names_excludes_transitive_user_sharing(
        self,
    ) -> None:
        graph = Graph()
        source = graph.placeholder("source")
        shared_base = graph.call_function(operator.neg, (source,))
        lhs_index = graph.call_function(operator.pos, (shared_base,))
        rhs_index = graph.call_function(operator.abs, (shared_base,))
        lhs_load = graph.call_function(operator.invert, (lhs_index,))
        rhs_load = graph.call_function(operator.truth, (rhs_index,))
        graph.call_function(operator.not_, (lhs_index,))

        names = _node_names(
            _collective_load_dependency_nodes(
                lhs_load,
                {shared_base, lhs_index, rhs_index, lhs_load, rhs_load},
                {lhs_load, rhs_load},
            )
        )
        self.assertNotIn(lhs_index.name, names)
        self.assertNotIn(shared_base.name, names)

    def test_collective_load_dependency_names_excludes_user_shared_ancestors(
        self,
    ) -> None:
        graph = Graph()
        source = graph.placeholder("source")
        shared_index = graph.call_function(operator.neg, (source,))
        lhs_load = graph.call_function(operator.pos, (shared_index,))
        rhs_load = graph.call_function(operator.abs, (shared_index,))
        graph.call_function(operator.invert, (shared_index,))

        self.assertNotIn(
            shared_index.name,
            _node_names(
                _collective_load_dependency_nodes(
                    lhs_load,
                    {shared_index, lhs_load, rhs_load},
                    {lhs_load, rhs_load},
                )
            ),
        )

    def test_tcgen05_kloop_ownership_consumes_only_exact_statements(self) -> None:
        state = CuteDeviceFunctionState()
        owned_a = statement_from_string("owned_a = 1")
        owned_b = statement_from_string("owned_b = owned_a + 1")
        loop = _kloop_with_inner(owned_a, owned_b)

        state.register_tcgen05_kloop_owned_stmts(loop, [owned_a, owned_b])
        state.replace_tcgen05_owned_kloop_stmts_with_pass(loop, [owned_a, owned_b])

        self.assertEqual(len(loop.inner_statements), 1)
        self.assertIsInstance(loop.inner_statements[0], ast.Pass)

    def test_tcgen05_kloop_ownership_preserves_preexisting_prelude(self) -> None:
        state = CuteDeviceFunctionState()
        prelude = statement_from_string("preexisting_arange = arange")
        owned_a = statement_from_string("owned_a = 1")
        owned_b = statement_from_string("owned_b = owned_a + 1")
        loop = _kloop_with_inner(prelude, owned_a, owned_b)

        state.register_tcgen05_kloop_owned_stmts(loop, [owned_a, owned_b])
        state.replace_tcgen05_owned_kloop_stmts_with_pass(loop, [owned_a, owned_b])

        self.assertEqual(loop.inner_statements[0], prelude)
        self.assertIsInstance(loop.inner_statements[1], ast.Pass)

    def test_tcgen05_kloop_final_cleanup_preserves_preexisting_prelude(self) -> None:
        state = CuteDeviceFunctionState()
        prelude = statement_from_string("preexisting_arange = arange")
        owned_a = statement_from_string("owned_a = 1")
        owned_b = statement_from_string("owned_b = owned_a + 1")
        loop = _kloop_with_inner(prelude, owned_a, owned_b)

        state.register_tcgen05_pure_lifecycle_pending_store(loop)
        state.request_tcgen05_owned_kloop_cleanup(loop)
        state.register_tcgen05_kloop_owned_stmts(loop, [owned_a, owned_b])
        state.finalize_tcgen05_owned_kloop_cleanup(loop)

        self.assertEqual(loop.inner_statements[0], prelude)
        self.assertIsInstance(loop.inner_statements[1], ast.Pass)

    def test_tcgen05_kloop_ownership_rejects_unowned_statement(self) -> None:
        state = CuteDeviceFunctionState()
        prelude = statement_from_string("preexisting_arange = arange")
        owned = statement_from_string("owned = 1")
        unowned = statement_from_string("user_value = owned + 1")
        loop = _kloop_with_inner(prelude, owned, unowned)

        state.register_tcgen05_kloop_owned_stmts(loop, [owned])

        self.assertTrue(state.is_tcgen05_kloop_owned_stmt(loop, owned))
        self.assertFalse(state.is_tcgen05_kloop_owned_stmt(loop, unowned))
        self.assertEqual(
            state.tcgen05_unowned_kloop_stmts(loop, [owned, unowned]), [unowned]
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"first unowned statement: user_value = owned \+ 1",
        ):
            state.replace_tcgen05_owned_kloop_stmts_with_pass(loop, [owned, unowned])
        self.assertEqual(loop.inner_statements, [prelude, owned, unowned])

    def test_tcgen05_kloop_final_cleanup_rejects_unowned_suffix(self) -> None:
        state = CuteDeviceFunctionState()
        prelude = statement_from_string("preexisting_arange = arange")
        owned = statement_from_string("owned = 1")
        unowned = statement_from_string("user_value = owned + 1")
        loop = _kloop_with_inner(prelude, owned, unowned)

        state.register_tcgen05_pure_lifecycle_pending_store(loop)
        state.request_tcgen05_owned_kloop_cleanup(loop)
        state.register_tcgen05_kloop_owned_stmts(loop, [owned])

        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"first unowned statement: user_value = owned \+ 1",
        ):
            state.finalize_tcgen05_owned_kloop_cleanup(loop)
        self.assertEqual(loop.inner_statements, [prelude, owned, unowned])

    def test_tcgen05_pure_lifecycle_rejects_missing_store(self) -> None:
        state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()

        state.register_tcgen05_pure_lifecycle_pending_store(loop)

        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "requires exactly one store consuming the matmul result",
        ):
            state.finalize_tcgen05_pure_lifecycle_stores()

    def test_tcgen05_pure_lifecycle_rejects_duplicate_pending_store(
        self,
    ) -> None:
        state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()

        state.register_tcgen05_pure_lifecycle_pending_store(loop)

        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "only one pending matmul/store pair per K loop",
        ):
            state.register_tcgen05_pure_lifecycle_pending_store(loop)

    def test_tcgen05_store_value_consume_rejects_fanout(self) -> None:
        state = CuteDeviceFunctionState()
        value = CuteTcgen05StoreValue(lifecycle_context=_lifecycle(), bm=256)
        state.register_tcgen05_store_value("matmul_result", value)

        self.assertIs(
            state.consume_tcgen05_store_value(["matmul_result"]),
            value,
        )
        with self.assertRaisesRegex(exc.BackendUnsupported, "multi-store fan-out"):
            state.consume_tcgen05_store_value(["matmul_result"])

    def test_codegen_state_marks_collective_dependency_statement(self) -> None:
        state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()
        graph = Graph()
        fx_node = graph.placeholder("indices")
        state.register_collective_handled_load(
            "operand_load", dependency_nodes=(fx_node,)
        )
        codegen = _StatementCaptureGenerateAST(state, loop)
        stmt = statement_from_string("indices = tile_offset")

        with codegen.statement_owner_node(fx_node):
            codegen.add_statement(stmt)

        self.assertEqual(loop.inner_statements, [stmt])
        self.assertTrue(state.is_tcgen05_kloop_owned_stmt(loop, stmt))

    def test_codegen_state_does_not_mark_non_inner_statement_target(self) -> None:
        state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()
        graph = Graph()
        fx_node = graph.placeholder("indices")
        state.register_collective_handled_load(
            "operand_load", dependency_nodes=(fx_node,)
        )
        outer_prefix: list[ast.AST] = []
        codegen = _StatementCaptureGenerateAST(state, loop, outer_prefix)
        stmt = statement_from_string("indices = tile_offset")

        with codegen.statement_owner_node(fx_node):
            codegen.add_statement(stmt)

        self.assertEqual(outer_prefix, [stmt])
        self.assertEqual(loop.inner_statements, [])
        self.assertFalse(state.is_tcgen05_kloop_owned_stmt(loop, stmt))

    def test_codegen_state_leaves_unrelated_statement_unowned(self) -> None:
        state = CuteDeviceFunctionState()
        loop = _kloop_with_inner()
        graph = Graph()
        owned_fx_node = graph.placeholder("indices")
        fx_node = graph.placeholder("unrelated")
        state.register_collective_handled_load(
            "operand_load",
            dependency_nodes=(owned_fx_node,),
        )
        codegen = _StatementCaptureGenerateAST(state, loop)
        stmt = statement_from_string("user_value = 1")

        with codegen.statement_owner_node(fx_node):
            codegen.add_statement(stmt)

        self.assertEqual(loop.inner_statements, [stmt])
        self.assertFalse(state.is_tcgen05_kloop_owned_stmt(loop, stmt))
