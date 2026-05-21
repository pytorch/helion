from __future__ import annotations

import ast
import dataclasses
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import unittest

from torch.fx import Graph

from helion import exc
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.cute_mma import _collective_load_dependency_nodes
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
from helion._compiler.cute.device_state import CuteTcgen05StoreValue
from helion._compiler.cute.tcgen05_lifecycle import Tcgen05LifecycleContext
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.tile_strategy import DeviceLoopState
from helion._testing import onlyBackends

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
