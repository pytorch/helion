from __future__ import annotations

from typing import Any
import unittest

from helion import exc
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
from helion._compiler.cute.device_state import CuteTcgen05StoreValue
from helion._testing import onlyBackends


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


@onlyBackends(["cute"])
class TestCuteDeviceFunctionState(unittest.TestCase):
    def test_get_tcgen05_store_value_checks_candidate_names(self) -> None:
        state = CuteDeviceFunctionState()
        value = CuteTcgen05StoreValue(bm=256)
        state.register_tcgen05_store_value("matmul_result", value)

        self.assertIsNone(state.get_tcgen05_store_value(["store_value"]))
        self.assertIs(
            state.get_tcgen05_store_value(["store_value", "matmul_result"]),
            value,
        )

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
