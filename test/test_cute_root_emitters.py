"""Pins for the CuTe root-emitter and kernel-generator registries: dispatch
order, plan slots, the ``GenerateAST`` forwarders tests still patch, and the
per-generator transform policies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from typing import cast
import unittest
from unittest.mock import patch

import pytest

from helion import exc
from helion._compiler.cute import emitters
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.emitters import ROOT_EMITTERS
from helion._compiler.cute.emitters import PlannedRootEmitter
from helion._compiler.cute.kernel_generators import KERNEL_GENERATORS
from helion._compiler.emitter_protocols import KernelTransforms
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.triton.backend import TritonBackend
from helion._testing import onlyBackends
from helion.runtime.config import Config

pytestmark = pytest.mark.cpu_only

ROOT_ORDER = (
    "required_computed_fragment",
    "register_subblocks",
    "radix_selection",
    "requested_computed_fragment",
    "block_scaled",
    "topk",
    "register_region",
    "chunk_prepare",
    "chunk_recurrence",
    "gdn_recurrence",
    "mma_dag",
    "single_token_rank1",
    "split_single_token_rank1",
    "fixed_token_rank1",
    "warp_mma_gemm",
    "attention_flash",
    "computed_fragment",
)
GENERATOR_ORDER = (
    "dependent_phases",
    "grouped_prefix_adaptive_n",
    "packed_operand",
    "row_resident",
    "grouped_warp_tf32",
    "flat_grouped_native",
    "bounded_cache",
    "split_k_cluster",
    "split_k_workspace",
    "materialized_fission",
    "ordered_phases",
)


def _generator(name: str) -> Any:
    (generator,) = [g for g in KERNEL_GENERATORS if g.name == name]
    return generator


@onlyBackends(["cute"])
class TestRootEmitterRegistry(unittest.TestCase):
    def test_dispatch_order_is_pinned(self) -> None:
        self.assertEqual(tuple(e.name for e in ROOT_EMITTERS), ROOT_ORDER)

    def test_planned_slots_exist_and_entries_resolve(self) -> None:
        state = CuteDeviceFunctionState()
        planned = [
            value
            for value in vars(emitters).values()
            if isinstance(value, PlannedRootEmitter)
        ]
        self.assertEqual(len(planned), 13)
        for emitter in planned:
            self.assertIsNone(getattr(state.root_plans, emitter.slot), emitter.name)
            self.assertTrue(callable(emitters._entry(emitter.module, emitter.entry)))

    def test_planned_decline_semantics(self) -> None:
        root_plans = SimpleNamespace()
        codegen = cast(
            "GenerateAST",
            SimpleNamespace(
                device_function=SimpleNamespace(
                    cute_state=SimpleNamespace(root_plans=root_plans)
                )
            ),
        )
        for emitter, error, clears in (
            (emitters.RADIX_SELECTION_ROOT, exc.InvalidConfig, False),
            (emitters.TOPK_ROOT, exc.BackendUnsupported, False),
            (emitters.MMA_DAG_ROOT, exc.BackendUnsupported, True),
        ):
            setattr(root_plans, emitter.slot, object())
            with self.assertRaises(error):
                emitter.decline(codegen, None)
            self.assertEqual(
                getattr(root_plans, emitter.slot) is None, clears, emitter.name
            )

    def test_attention_flash_plan_and_decline_semantics(self) -> None:
        emitter = emitters.ATTENTION_FLASH_ROOT
        bwd_fields = ("bwd_block_ids", "bwd_match")
        fwd_fields = ("block_ids", "score_plan", "gated_match", "flex_match")

        def codegen_with(**slots: object) -> Any:
            flash = SimpleNamespace(
                bwd_block_ids=None, block_ids=None, gated_match=None
            )
            vars(flash).update(slots)
            state = SimpleNamespace(flash=flash)
            return cast(
                "GenerateAST",
                SimpleNamespace(device_function=SimpleNamespace(cute_state=state)),
            )

        # The backward detector wins over a forward plan; a forward plan is
        # gated when its gate match is recorded.
        self.assertIsNone(emitter.plan(codegen_with()))
        self.assertEqual(
            emitter.plan(codegen_with(block_ids=[0])),
            emitters._FLASH_FWD,
        )
        self.assertEqual(
            emitter.plan(codegen_with(block_ids=[0], gated_match=object())),
            emitters._FLASH_GATED,
        )
        self.assertEqual(
            emitter.plan(codegen_with(bwd_block_ids=[0], block_ids=[0])),
            emitters._FLASH_BWD,
        )

        for plan, message, cleared in (
            (
                emitters._FLASH_BWD,
                "flash attention backward failed late validation",
                bwd_fields,
            ),
            (
                emitters._FLASH_GATED,
                "gated attention failed late validation",
                fwd_fields,
            ),
            (emitters._FLASH_FWD, "flash attention failed late validation", fwd_fields),
        ):
            codegen = codegen_with(
                **dict.fromkeys(bwd_fields + fwd_fields, object()), threads=256
            )
            flash = codegen.device_function.cute_state.flash
            with self.assertRaisesRegex(exc.BackendUnsupported, message):
                emitter.decline(codegen, plan)
            for field in bwd_fields + fwd_fields:
                self.assertEqual(
                    getattr(flash, field) is None, field in cleared, (plan, field)
                )
            # Only the forward refusals reset the thread count the detector set.
            self.assertEqual(
                flash.threads,
                256 if plan == emitters._FLASH_BWD else 128,
                plan,
            )

    def test_registry_entries_are_the_module_emitters(self) -> None:
        by_name = {entry.name: entry for entry in ROOT_EMITTERS}
        self.assertIs(by_name["block_scaled"], emitters.BLOCK_SCALED_ROOT)
        self.assertIs(by_name["topk"], emitters.TOPK_ROOT)
        self.assertIs(by_name["computed_fragment"], emitters.COMPUTED_FRAGMENT_ROOT)
        gated = by_name["requested_computed_fragment"]
        self.assertIsInstance(gated, emitters.GatedRootEmitter)
        self.assertIs(gated.inner, emitters.COMPUTED_FRAGMENT_ROOT)
        # A patch of the optional owner's ``try_emit`` intercepts its entries;
        # the independent required owner is covered by the full-vote module.
        codegen = cast("GenerateAST", object.__new__(GenerateAST))
        with patch.object(
            emitters.ComputedFragmentRootEmitter, "try_emit", return_value=True
        ) as owner:
            self.assertTrue(by_name["computed_fragment"].try_emit(codegen))
        owner.assert_called_once_with(codegen)

    def test_generator_order_is_pinned(self) -> None:
        self.assertEqual(tuple(g.name for g in KERNEL_GENERATORS), GENERATOR_ORDER)

    def test_transform_policies(self) -> None:
        plain = KernelTransforms()
        fused = KernelTransforms(store_transform=lambda *args: args)
        self.assertEqual(plain.carried, frozenset())
        self.assertEqual(KernelTransforms(extra_params=[]).carried, frozenset())
        self.assertEqual(fused.carried, frozenset({"store_transform"}))

        adaptive = _generator("grouped_prefix_adaptive_n").transform_policy
        config = Config.from_dict({"block_sizes": [1]})
        self.assertTrue(adaptive.honors(config, plain))
        self.assertFalse(adaptive.honors(config, fused))
        with self.assertRaises(exc.InvalidConfig):
            adaptive.decline(config)

        warp_raw = Config.from_dict(
            {"block_sizes": [1], "cute_mma_f32_conversion": "warp_raw"}
        )
        stripped = _generator("grouped_warp_tf32").transform_policy.decline(warp_raw)
        self.assertEqual(stripped.config, {"block_sizes": [1]})

        bounded = Config.from_dict(
            {"block_sizes": [1], "cute_reduction_sequence": "bounded"}
        )
        policy = _generator("bounded_cache").transform_policy
        self.assertTrue(
            policy.honors(bounded, KernelTransforms(bounded_cache_request=1))
        )
        self.assertEqual(
            policy.decline(bounded).config["cute_reduction_sequence"], "scalar"
        )

        workspace = _generator("split_k_workspace")
        self.assertTrue(
            workspace.transform_policy.honors(config, KernelTransforms(host_prefix=[]))
        )
        with self.assertRaises(exc.BackendUnsupported):
            workspace.transform_policy.decline(config)

        packed = _generator("packed_operand").transform_policy
        self.assertFalse(
            packed.honors(
                Config.from_dict({"cute_materialized_schedule": "rows"}), plain
            )
        )

    def test_other_backends_register_nothing(self) -> None:
        backend = TritonBackend()
        self.assertEqual(backend.root_emitters(), ())
        self.assertEqual(backend.kernel_generators(), ())
        self.assertEqual(backend.retry_policy().signals, ())
