from __future__ import annotations

import unittest
from unittest import mock

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl


@onlyBackends(["triton"])
class TestCacheModifier(RefEagerTestBase, TestCase):
    @skipIfRefEager("Config spec inspection not applicable in ref eager mode")
    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_autotune_load_cache_modifier_registered_for_amd(self):
        @helion.kernel
        def kernel_with_loads_and_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val_x = hl.load(x, [tile])
                val_y = hl.load(y, [tile])
                hl.store(out, [tile], val_x + val_y)
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        with mock.patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=True,
        ):
            bound_kernel = kernel_with_loads_and_store.bind((x, y))

        fragment = bound_kernel.config_spec.load_cache_modifiers
        self.assertEqual(fragment.length, 2)
        self.assertEqual(fragment.inner.choices, ("", ".cg"))

    @skipIfRefEager("Config spec inspection not applicable in ref eager mode")
    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_load_cache_modifier_not_registered_without_backend_choices(self):
        @helion.kernel
        def kernel_with_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = hl.load(x, [tile]) + hl.load(y, [tile])
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        with mock.patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=False,
        ):
            bound_kernel = kernel_with_loads.bind((x, y))

        self.assertEqual(bound_kernel.config_spec.load_cache_modifiers.length, 0)

    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_configured_load_cache_modifier_emitted(self):
        @helion.kernel(
            config={
                "block_size": 16,
                "indexing": "pointer",
                "load_cache_modifiers": [".cg", ""],
            }
        )
        def kernel_with_configured_modifier(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val_x = hl.load(x, [tile])
                val_y = hl.load(y, [tile])
                out[tile] = val_x + val_y
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        with mock.patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=True,
        ):
            code, result = code_and_output(kernel_with_configured_modifier, (x, y))

        torch.testing.assert_close(result, x + y)
        self.assertIn("cache_modifier", code)
        self.assertIn(".cg", code)
        self.assertNotIn("cache_modifier=''", code)

    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_default_load_cache_modifier_config_is_elided(self):
        @helion.kernel(config={"block_size": 16, "indexing": "pointer"})
        def kernel_with_default_modifiers(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val_x = hl.load(x, [tile])
                val_y = hl.load(y, [tile])
                out[tile] = val_x + val_y
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        with mock.patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=True,
        ):
            code, result = code_and_output(kernel_with_default_modifiers, (x, y))

        torch.testing.assert_close(result, x + y)
        self.assertNotIn("cache_modifier=", code)


if __name__ == "__main__":
    unittest.main()
