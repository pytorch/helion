from __future__ import annotations

import unittest

from helion import exc
from helion._compiler.attribute_alias import AttributeAlias
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.emitters import PlannedRootEmitter
from helion._compiler.cute.state import ROOT_PLAN_SLOTS
from helion._compiler.cute.state import CuteCodegenState
from helion._compiler.cute.state import CuteEnvState
from helion._compiler.cute.state import CuteFlashState
from helion._compiler.cute.state import CuteLaneState
from helion._compiler.cute.state import CuteRootPlans
from helion._compiler.cute.state import CuteTcgen05State
from helion._compiler.cute.tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from helion._compiler.triton.backend import TritonBackend
from helion.runtime.config import Config


class _Inner:
    def __init__(self) -> None:
        self.count = 1


class _Outer:
    count: AttributeAlias[int] = AttributeAlias("inner.count")

    def __init__(self) -> None:
        self.inner = _Inner()


class TestAttributeAlias(unittest.TestCase):
    def test_reads_and_writes_forward_to_the_path(self) -> None:
        outer = _Outer()
        self.assertEqual(outer.count, 1)
        outer.count = 5
        self.assertEqual(outer.inner.count, 5)
        self.assertIsInstance(_Outer.count, AttributeAlias)
        self.assertEqual(repr(_Outer.count), "AttributeAlias('inner.count')")


class TestCuteDeviceFunctionStateNamespaces(unittest.TestCase):
    def test_fields_live_on_the_four_namespaces(self) -> None:
        state = CuteDeviceFunctionState()
        self.assertIsInstance(state.tcgen05, CuteTcgen05State)
        self.assertIsInstance(state.flash, CuteFlashState)
        self.assertIsInstance(state.lanes, CuteLaneState)
        self.assertIsInstance(state.root_plans, CuteRootPlans)
        self.assertFalse(hasattr(state, "simt_cluster_n"))
        self.assertFalse(hasattr(state, "attention_flash_threads"))
        self.assertFalse(hasattr(state, "topk_plan"))

    def test_tcgen05_registries_use_the_namespace(self) -> None:
        state = CuteDeviceFunctionState()
        self.assertFalse(state.has_tcgen05_post_loop_marks)
        marker = object()
        state.register_tcgen05_post_loop_stmts([marker])  # pyrefly: ignore
        self.assertEqual(state.tcgen05.post_loop_stmt_ids, {id(marker)})
        self.assertTrue(state.has_tcgen05_post_loop_marks)
        state.request_root_lane_loop_suppression()
        self.assertTrue(state.lanes.suppress_root_lane_loops)
        self.assertTrue(state.consume_root_lane_loop_suppression())
        self.assertFalse(state.consume_root_lane_loop_suppression())


class TestBackendStateHooks(unittest.TestCase):
    def test_cute_backend_builds_its_state_objects(self) -> None:
        backend = CuteBackend()
        self.assertIsInstance(backend.new_env_state(), CuteEnvState)
        self.assertIsInstance(backend.new_codegen_state(), CuteCodegenState)
        state = backend.new_device_state(Config())
        self.assertIsInstance(state, CuteDeviceFunctionState)
        self.assertEqual(state.tcgen05.program_id_divisor, 1)
        self.assertEqual(state.lanes.launch_grid_multiplier, 1)

    def test_split_k_seeds_the_divisor_and_grid_multiplier(self) -> None:
        state = CuteBackend().new_device_state(
            Config(**{TCGEN05_SPLIT_K_CONFIG_KEY: 4})
        )
        self.assertEqual(state.tcgen05.program_id_divisor, 4)
        self.assertEqual(state.lanes.launch_grid_multiplier, 4)
        self.assertFalse(state.tcgen05.split_k_applied)

    def test_default_hook_ignores_the_split_k_key(self) -> None:
        # Another backend carries no CuTe state at all: a Triton config
        # carrying the CuTe key seeds nothing (the seed used to sit behind
        # ``backend_name == "cute"``), so ``codegen_function_call`` never
        # sees a grid multiplier the tcgen05 pipeline did not apply.
        backend = TritonBackend()
        self.assertIsNone(
            backend.new_device_state(Config(**{TCGEN05_SPLIT_K_CONFIG_KEY: 4}))
        )
        self.assertIsNone(backend.new_env_state())
        self.assertIsNone(backend.new_codegen_state())

    def test_planned_root_emitter_slots_are_root_plan_fields(self) -> None:
        self.assertIn("topk_plan", ROOT_PLAN_SLOTS)
        with self.assertRaisesRegex(ValueError, "not a CuteRootPlans slot"):
            PlannedRootEmitter(
                "bogus",
                slot="no_such_plan",
                module="topk",
                entry="codegen_topk_root",
                failure=lambda: exc.BackendUnsupported("cute", "bogus"),
            )


if __name__ == "__main__":
    unittest.main()
