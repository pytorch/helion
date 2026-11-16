"""Host shape-bake policy on CPU metadata, without binding or device codegen."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest

import pytest

from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.host_metadata import can_bake_tensor_shapes
from helion._compiler.cute.state import CuteCodegenState
from helion._compiler.cute.state import cute_codegen_state
from helion._compiler.generate_ast import GenerateAST
from helion.runtime.config import Config

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from helion._compiler.cute.collective_matmul import CollectiveMmaSite
    from helion._compiler.device_function import DeviceFunction


class TestCuteRegisterChainShapeBake(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.state = CuteDeviceFunctionState()
        self.codegen = object.__new__(GenerateAST)
        self.codegen.backend_state = CuteCodegenState()
        self.codegen.device_function = cast(
            "DeviceFunction",
            SimpleNamespace(cute_state=self.state, config=Config()),
        )

    def chain(self, *, lowered: bool) -> None:
        self.codegen.device_function.config = Config(cute_register_chain=True)
        for identity in range(2):
            cute_codegen_state(self.codegen).declare_matmul()
            self.state.lanes.collective_mma_sites.append(
                cast("CollectiveMmaSite", SimpleNamespace(identity=identity))
            )
        self.state.lanes.collective_register_chain_lowered = lowered

    def test_complete_lowered_chain_can_bake(self) -> None:
        self.chain(lowered=True)
        self.assertTrue(cute_codegen_state(self.codegen).uses_matmul)
        self.assertTrue(can_bake_tensor_shapes(self.codegen))

    def test_requested_but_unlowered_chain_cannot_bake(self) -> None:
        self.chain(lowered=False)
        self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_extra_matmul_declaration_after_chain_cannot_bake(self) -> None:
        self.chain(lowered=True)
        cute_codegen_state(self.codegen).declare_matmul()
        self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_extra_matmul_declaration_before_chain_cannot_bake(self) -> None:
        cute_codegen_state(self.codegen).declare_matmul()
        self.chain(lowered=False)
        self.assertFalse(can_bake_tensor_shapes(self.codegen))
        self.state.lanes.collective_register_chain_lowered = True
        self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_declarations_do_not_count_only_boolean_transitions(self) -> None:
        self.chain(lowered=True)
        cute_codegen_state(self.codegen).uses_matmul = False
        self.assertFalse(cute_codegen_state(self.codegen).uses_matmul)
        cute_codegen_state(self.codegen).declare_matmul()
        self.assertTrue(cute_codegen_state(self.codegen).uses_matmul)
        self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_incomplete_ownership_cannot_bake(self) -> None:
        self.chain(lowered=True)
        original = list(self.state.lanes.collective_mma_sites)
        for sites in ([], original[:1], [*original, original[0]]):
            with self.subTest(sites=len(sites)):
                self.state.lanes.collective_mma_sites = sites
                self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_incompatible_wrapper_cannot_inherit_chain_exemption(self) -> None:
        self.chain(lowered=True)
        cute_codegen_state(self.codegen).wrapper_plans = [
            {"kind": "tcgen05_grouped_static"}
        ]
        self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_default_and_ordinary_matmul_policy_is_unchanged(self) -> None:
        self.assertFalse(self.state.lanes.collective_register_chain_lowered)
        self.assertTrue(can_bake_tensor_shapes(self.codegen))
        cute_codegen_state(self.codegen).declare_matmul()
        self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_plain_tcgen05_plans_bake(self) -> None:
        """The plain tcgen05 GEMM plans (A/B TMA loads, D TMA store over the
        problem tensors) bake; the grouped variants keep runtime layouts."""
        cute_codegen_state(self.codegen).declare_matmul()
        ab = {
            "kind": "tcgen05_ab_tma",
            "orientation": "nm",
            "row_union_schedule": "nm_m256_n80_k64",
        }
        d = {"kind": "tcgen05_d_tma", "rank3_mnl_tensor": True}
        cute_codegen_state(self.codegen).wrapper_plans = [ab, d]
        self.assertTrue(can_bake_tensor_shapes(self.codegen))
        cute_codegen_state(self.codegen).wrapper_plans = [dict(ab, orientation="mn")]
        self.assertTrue(can_bake_tensor_shapes(self.codegen))
        for key in (
            "shared_rhs",
            "fixed_ab_tensormaps",
            "dynamic_ab_tensormaps",
            "full_allocation_a",
            "full_allocation_b",
            "device_split_sizes",
            "fixed_grouped_b_rank3",
            "source_m_tile",
        ):
            with self.subTest(key=key):
                cute_codegen_state(self.codegen).wrapper_plans = [{**ab, key: True}, d]
                self.assertFalse(can_bake_tensor_shapes(self.codegen))
        for key in ("fixed_tensormap", "bounded_output_shift", "dynamic_d_tensormap"):
            with self.subTest(key=key):
                cute_codegen_state(self.codegen).wrapper_plans = [ab, {**d, key: True}]
                self.assertFalse(can_bake_tensor_shapes(self.codegen))
        for extra in ("tcgen05_grouped_static_persistent", "tcgen05_aux_tma"):
            with self.subTest(extra=extra):
                cute_codegen_state(self.codegen).wrapper_plans = [
                    ab,
                    d,
                    {"kind": extra},
                ]
                self.assertFalse(can_bake_tensor_shapes(self.codegen))

    def test_existing_safe_wrapper_policy_is_unchanged(self) -> None:
        cute_codegen_state(self.codegen).declare_matmul()
        for kind in (
            "helion_small_biased_attention",
            "helion_flash",
            "chunk_prepare_tma",
            "chunk_recurrence_sm100",
            "chunk_recurrence_warp_dv4",
            "helion_flash_bwd",
        ):
            with self.subTest(kind=kind):
                cute_codegen_state(self.codegen).wrapper_plans = [{"kind": kind}]
                self.assertTrue(can_bake_tensor_shapes(self.codegen))
                cute_codegen_state(self.codegen).wrapper_plans.append(
                    {"kind": "unknown"}
                )
                self.assertFalse(can_bake_tensor_shapes(self.codegen))


if __name__ == "__main__":
    unittest.main()
