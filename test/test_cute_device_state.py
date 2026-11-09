from __future__ import annotations

import ast
import dataclasses
import operator
from types import SimpleNamespace
from typing import Any
from typing import cast
import unittest

import pytest
import torch
from torch.fx import Graph

from helion import exc
from helion._compiler.cute.access_regions import tile_begin_symbol
from helion._compiler.cute.cute_mma import _emit_tcgen05_device_segments_setup
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.device_state import CuteTcgen05GroupedPlan
from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
from helion._compiler.cute.device_state import CuteTcgen05StoreValue
from helion._compiler.cute.device_state import Tcgen05GroupedDMode
from helion._compiler.cute.device_state import Tcgen05GroupedSchedulerMode
from helion._compiler.cute.device_state import Tcgen05Orientation
from helion._compiler.cute.tcgen05_lifecycle import Tcgen05LifecycleContext
from helion._compiler.tile_strategy import DeviceLoopState
from helion._testing import onlyBackends

pytestmark = pytest.mark.cpu_only


def _plan(**overrides: object) -> CuteTcgen05MatmulPlan:
    kwargs: dict[str, Any] = {
        "bm": 256,
        "bn": 256,
        "bk": 128,
        "k_tile_count": 16,
        "cluster_m": 2,
        "is_two_cta": True,
        "uses_role_local_persistent_body": True,
        "cta_thread_count": 256,
        "physical_m_threads": 128,
        "acc_stage_count": 2,
        "ab_stage_count": 2,
        "c_stage_count": 2,
        "epi_warp_count": 4,
    }
    kwargs.update(overrides)
    return CuteTcgen05MatmulPlan(**kwargs)


def _grouped_plan(**overrides: object) -> CuteTcgen05GroupedPlan:
    kwargs: dict[str, Any] = {
        "orientation": Tcgen05Orientation.MN,
        "layout": "layout",
        "count": "2",
        "sched_params": "sched_params",
        "problem_sizes": "problem_sizes",
        "starts": "starts",
        "metadata_idx": "metadata_idx",
        "group_idx": "group_idx",
        "cta_tile_idx_m": "cta_tile_idx_m",
        "cta_tile_idx_n": "cta_tile_idx_n",
        "problem_m": "problem_m",
        "problem_n": "problem_n",
        "problem_k": "problem_k",
        "global_m_start": "global_m_start",
    }
    kwargs.update(overrides)
    return CuteTcgen05GroupedPlan(**kwargs)


def _runtime_direct_grouped_plan() -> CuteTcgen05GroupedPlan:
    return _grouped_plan(
        orientation=Tcgen05Orientation.NM,
        scheduler_mode=Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
        runtime_tile_records="runtime_tile_records",
        runtime_total_clusters="runtime_total_clusters",
        valid_m="valid_m",
        store_m="store_m",
        source_m_tile=32,
        d_mode=Tcgen05GroupedDMode.ALL_TILES,
        d_tensormap="d_tensormap",
    )


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


@onlyBackends(["cute"])
class TestCuteDeviceFunctionState(unittest.TestCase):
    def test_grouped_scheduler_modes_enforce_complete_state(self) -> None:
        device_search = _grouped_plan()
        self.assertFalse(device_search.uses_runtime_tile_table)

        runtime_direct = _runtime_direct_grouped_plan()
        self.assertTrue(runtime_direct.uses_runtime_tile_table)

        invalid_replacements = (
            lambda: dataclasses.replace(
                runtime_direct,
                scheduler_mode=Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH,
            ),
            lambda: _grouped_plan(
                scheduler_mode=Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
            ),
            lambda: _grouped_plan(fixed_tensormaps=True),
            lambda: _grouped_plan(
                orientation=Tcgen05Orientation.NM,
                valid_m="valid_m",
                store_m="store_m",
                source_m_tile=32,
                m_size=128,
                real_groups="real_groups",
                d_mode=Tcgen05GroupedDMode.ALL_TILES,
                d_tensormap="d_tensormap",
            ),
            lambda: _grouped_plan(device_layout_kind="split_sizes"),
            lambda: dataclasses.replace(
                runtime_direct,
                scheduler_mode=Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
            ),
        )
        for make_invalid in invalid_replacements:
            with self.assertRaises(AssertionError):
                make_invalid()

        runtime_clc = dataclasses.replace(
            runtime_direct,
            scheduler_mode=Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
            fixed_tensormaps=True,
            d_mode=Tcgen05GroupedDMode.NONE,
            d_tensormap=None,
        )
        self.assertTrue(runtime_clc.uses_runtime_tile_table)

    def test_grouped_device_metadata_setup_uses_one_physical_thread(self) -> None:
        grouped = _grouped_plan(
            orientation=Tcgen05Orientation.NM,
            valid_m="valid_m",
            store_m="store_m",
            source_m_tile=32,
            m_size=33,
            device_layout_kind="split_sizes",
            d_mode=Tcgen05GroupedDMode.ALL_TILES,
            d_tensormap="d_tensormap",
        )
        prefix: list[ast.AST] = []
        device_function = SimpleNamespace(new_var=lambda name: name)

        _emit_tcgen05_device_segments_setup(
            prefix,
            cast("Any", device_function),
            grouped,
            n_size=64,
            k_size=64,
            layout_dtype=torch.int32,
        )

        source = ast.unparse(ast.Module(body=prefix, type_ignores=[]))
        self.assertIn(
            "if cute.arch.thread_idx()[0] == 0 and cute.arch.thread_idx()[1] == 0 "
            "and (cute.arch.thread_idx()[2] == 0):",
            source,
        )
        self.assertNotIn("if cute.arch.thread_idx()[0] == 0:\n", source)

    def test_grouped_runtime_mode_matches_parent_warp_topology(self) -> None:
        runtime_direct = _runtime_direct_grouped_plan()
        _plan(grouped=runtime_direct)
        with self.assertRaises(AssertionError):
            _plan(grouped=runtime_direct, scheduler_warp_count=1)

        device_search_nm = dataclasses.replace(
            runtime_direct,
            scheduler_mode=Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH,
            runtime_tile_records=None,
            runtime_total_clusters=None,
            real_groups="real_groups",
        )
        _plan(grouped=device_search_nm, scheduler_warp_count=1)
        with self.assertRaises(AssertionError):
            _plan(grouped=device_search_nm)

        runtime_clc = dataclasses.replace(
            runtime_direct,
            scheduler_mode=Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
            fixed_tensormaps=True,
            d_mode=Tcgen05GroupedDMode.NONE,
            d_tensormap=None,
        )
        _plan(
            grouped=runtime_clc,
            scheduler_warp_count=1,
            persistence_model="clc_persistent",
        )
        with self.assertRaises(AssertionError):
            _plan(grouped=runtime_clc, persistence_model="clc_persistent")

    def test_tcgen05_store_value_d_store_layout_uses_output_layout(self) -> None:
        value = CuteTcgen05StoreValue(
            lifecycle_context=_lifecycle(), output_block_ids=(0, 1)
        )

        self.assertEqual(
            value.d_store_layout,
            "cutlass.utils.layout.LayoutEnum.ROW_MAJOR",
        )
        self.assertEqual(
            dataclasses.replace(value, output_column_major=True).d_store_layout,
            "cutlass.utils.layout.LayoutEnum.COL_MAJOR",
        )
        self.assertEqual(
            dataclasses.replace(
                value, orientation=Tcgen05Orientation.NM
            ).d_store_layout,
            "cutlass.utils.layout.LayoutEnum.COL_MAJOR",
        )

    def test_get_tcgen05_store_value_checks_candidate_names(self) -> None:
        state = CuteDeviceFunctionState()
        value = CuteTcgen05StoreValue(
            lifecycle_context=_lifecycle(), output_block_ids=(0, 1), bm=256
        )
        state.register_tcgen05_store_value("matmul_result", value)

        self.assertIsNone(state.get_tcgen05_store_value(["store_value"]))
        self.assertIs(
            state.get_tcgen05_store_value(["store_value", "matmul_result"]),
            value,
        )

    def test_tcgen05_store_value_carries_lifecycle_context(self) -> None:
        state = CuteDeviceFunctionState()
        lifecycle = _lifecycle()
        value = CuteTcgen05StoreValue(
            lifecycle_context=lifecycle, output_block_ids=(0, 1), bm=256
        )

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

    def test_collective_handled_loads_are_keyed_by_node(self) -> None:
        # FX names are unique only within a graph: the root graph's first load
        # is ``load`` like the K loop's operand load, and must not be taken for
        # it (a scalar load beside the matmul then read 0).
        state = CuteDeviceFunctionState()
        k_loop, root = Graph(), Graph()
        operand_load = k_loop.call_function(operator.pos, (k_loop.placeholder("a"),))
        root_load = root.call_function(operator.pos, (root.placeholder("a"),))
        operand_load.name = root_load.name = "load"

        state.register_collective_handled_load(operand_load)

        self.assertTrue(state.is_collective_handled_load(operand_load))
        self.assertFalse(state.is_collective_handled_load(root_load))

    def test_tcgen05_fanout_assigns_first_wait_and_last_release(self) -> None:
        state = CuteDeviceFunctionState()
        graph = Graph()
        stores = tuple(graph.placeholder(f"store_{index}") for index in range(3))
        value = CuteTcgen05StoreValue(
            lifecycle_context=_lifecycle(),
            output_block_ids=(0, 1),
            output_stores=stores,
        )
        self.assertEqual(
            state.claim_tcgen05_store_site(value, stores[0]), (False, False)
        )
        self.assertEqual(
            state.claim_tcgen05_store_site(value, stores[1]), (True, False)
        )
        self.assertEqual(state.claim_tcgen05_store_site(value, stores[2]), (True, True))
        with self.assertRaisesRegex(exc.BackendUnsupported, "distinct stores"):
            state.claim_tcgen05_store_site(value, stores[2])

    def test_tcgen05_fanout_requires_a_complete_store_set(self) -> None:
        state = CuteDeviceFunctionState()
        graph = Graph()
        stores = tuple(graph.placeholder(f"store_{index}") for index in range(2))
        value = CuteTcgen05StoreValue(
            lifecycle_context=_lifecycle(), output_block_ids=(0, 1)
        )
        self.assertEqual(
            state.claim_tcgen05_store_site(value, stores[0]), (False, True)
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "complete output store set"
        ):
            state.claim_tcgen05_store_site(value, stores[1])

    def test_tcgen05_fanout_rejects_different_execution_scopes(self) -> None:
        state = CuteDeviceFunctionState()
        stores = tuple(Graph().placeholder(f"store_{index}") for index in range(2))
        value = CuteTcgen05StoreValue(
            lifecycle_context=_lifecycle(),
            output_block_ids=(0, 1),
            output_stores=stores,
        )
        with self.assertRaisesRegex(exc.BackendUnsupported, "one FX graph"):
            state.claim_tcgen05_store_site(value, stores[0])

    def test_tcgen05_fanout_rejects_an_unexpected_store(self) -> None:
        state = CuteDeviceFunctionState()
        graph = Graph()
        expected, unexpected = (
            graph.placeholder(name) for name in ("expected", "other")
        )
        value = CuteTcgen05StoreValue(
            lifecycle_context=_lifecycle(),
            output_block_ids=(0, 1),
            output_stores=(expected,),
        )
        with self.assertRaisesRegex(exc.BackendUnsupported, "distinct stores"):
            state.claim_tcgen05_store_site(value, unexpected)


def test_loop_states_built_in_turn_name_distinct_tile_begins() -> None:
    # The state of a finished loop is collected before the next loop's begin
    # symbol is built, and CPython may hand the next state the same address:
    # the symbol names the state's serial, never its address.
    symbols = set()
    for _ in range(64):
        state = _kloop_with_inner()
        symbol = tile_begin_symbol(0, state)
        assert f"{id(state):x}" not in symbol.name
        symbols.add(symbol)
    assert len(symbols) == 64
