"""Host shape-bake policy on CPU metadata, without binding or device codegen."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest

from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.generate_ast import GenerateAST
from helion.runtime.config import Config

if TYPE_CHECKING:
    from helion._compiler.cute.collective_matmul import CollectiveMmaSite
    from helion._compiler.device_function import DeviceFunction


class TestCuteRegisterChainShapeBake(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.state = CuteDeviceFunctionState()
        self.codegen = object.__new__(GenerateAST)
        self.codegen._cute_uses_matmul = False
        self.codegen._cute_matmul_declaration_count = 0
        self.codegen.cute_wrapper_plans = []
        self.codegen.device_function = cast(
            "DeviceFunction",
            SimpleNamespace(cute_state=self.state, config=Config()),
        )

    def chain(self, *, lowered: bool) -> None:
        self.codegen.device_function.config = Config(cute_register_chain=True)
        for identity in range(2):
            self.codegen.cute_uses_matmul = True
            self.state.collective_mma_sites.append(
                cast("CollectiveMmaSite", SimpleNamespace(identity=identity))
            )
        self.state.collective_register_chain_lowered = lowered

    def test_complete_lowered_chain_can_bake(self) -> None:
        self.chain(lowered=True)
        self.assertTrue(self.codegen.cute_uses_matmul)
        self.assertTrue(self.codegen._cute_can_bake_tensor_shapes())

    def test_requested_but_unlowered_chain_cannot_bake(self) -> None:
        self.chain(lowered=False)
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_extra_matmul_declaration_after_chain_cannot_bake(self) -> None:
        self.chain(lowered=True)
        self.codegen.cute_uses_matmul = True
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_extra_matmul_declaration_before_chain_cannot_bake(self) -> None:
        self.codegen.cute_uses_matmul = True
        self.chain(lowered=False)
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())
        self.state.collective_register_chain_lowered = True
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_declarations_do_not_count_only_boolean_transitions(self) -> None:
        self.chain(lowered=True)
        self.codegen.cute_uses_matmul = False
        self.assertFalse(self.codegen.cute_uses_matmul)
        self.codegen.cute_uses_matmul = True
        self.assertTrue(self.codegen.cute_uses_matmul)
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_incomplete_ownership_cannot_bake(self) -> None:
        self.chain(lowered=True)
        original = list(self.state.collective_mma_sites)
        for sites in ([], original[:1], [*original, original[0]]):
            with self.subTest(sites=len(sites)):
                self.state.collective_mma_sites = sites
                self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_incompatible_wrapper_cannot_inherit_chain_exemption(self) -> None:
        self.chain(lowered=True)
        self.codegen.cute_wrapper_plans = [{"kind": "tcgen05_grouped_static"}]
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_default_and_ordinary_matmul_policy_is_unchanged(self) -> None:
        self.assertFalse(self.state.collective_register_chain_lowered)
        self.assertTrue(self.codegen._cute_can_bake_tensor_shapes())
        self.codegen.cute_uses_matmul = True
        self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())

    def test_existing_safe_wrapper_policy_is_unchanged(self) -> None:
        self.codegen.cute_uses_matmul = True
        for kind in (
            "helion_small_biased_attention",
            "helion_flash",
            "chunk_prepare_tma",
            "chunk_recurrence_sm100",
            "chunk_recurrence_warp_dv4",
            "helion_flash_bwd",
        ):
            with self.subTest(kind=kind):
                self.codegen.cute_wrapper_plans = [{"kind": kind}]
                self.assertTrue(self.codegen._cute_can_bake_tensor_shapes())
                self.codegen.cute_wrapper_plans.append({"kind": "unknown"})
                self.assertFalse(self.codegen._cute_can_bake_tensor_shapes())


if __name__ == "__main__":
    unittest.main()
