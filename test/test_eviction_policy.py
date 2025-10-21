from __future__ import annotations

import unittest

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


class TestEvictionPolicy(RefEagerTestBase, TestCase):
    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    def test_hl_load_eviction_policy_emitted(self, indexing: str):
        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")
        elif indexing == "block_ptr" and supports_tensor_descriptor():
            self.skipTest("Block pointer is not used if tensor descriptor is available")

        @helion.kernel(config={"indexing": indexing, "block_size": 16})
        def copy_with_eviction(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val = hl.load(x, [tile], eviction_policy="evict_last")
                out[tile] = val
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(copy_with_eviction, (x,))
        torch.testing.assert_close(result, x)
        self.assertExpectedJournal(code)
        self.assertIn("eviction_policy", code)
        self.assertIn("evict_last", code)

    @skipIfRefEager("Config spec inspection not applicable in ref eager mode")
    def test_autotune_eviction_policy_registered(self):
        """Test that eviction policy tunable is automatically registered for loads in device loops."""

        @helion.kernel
        def kernel_with_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val_x = hl.load(x, [tile])
                val_y = hl.load(y, [tile])
                out[tile] = val_x + val_y
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        bound_kernel = kernel_with_loads.bind((x, y))
        config_spec = bound_kernel.config_spec

        from helion.autotuner import EnumFragment
        from helion.autotuner import ListOf

        fragment = config_spec.load_eviction_policies
        self.assertIsInstance(fragment, ListOf)
        self.assertEqual(fragment.length, 2)
        self.assertIsInstance(fragment.inner, EnumFragment)
        self.assertIn("", fragment.inner.choices)
        self.assertIn("first", fragment.inner.choices)
        self.assertIn("last", fragment.inner.choices)

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    def test_eviction_policy_in_generated_code(self, indexing: str):
        """Test that eviction policies appear in generated code when configured."""

        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")
        elif indexing == "block_ptr" and supports_tensor_descriptor():
            self.skipTest("Block pointer is not used if tensor descriptor is available")

        @helion.kernel(
            config={
                "block_size": 16,
                "load_eviction_policies": ["", "last"],
                "indexing": indexing,
            }
        )
        def kernel_with_eviction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val_x = hl.load(x, [tile])  # No eviction policy
                val_y = hl.load(y, [tile])  # Should get evict_last
                out[tile] = val_x + val_y
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(kernel_with_eviction, (x, y))
        torch.testing.assert_close(result, x + y)

        # Check that evict_last appears in the generated code
        self.assertIn("evict_last", code)
        self.assertExpectedJournal(code)

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    def test_explicit_eviction_policy_overrides_tunable(self, indexing: str):
        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")
        elif indexing == "block_ptr" and supports_tensor_descriptor():
            self.skipTest("Block pointer is not used if tensor descriptor is available")

        @helion.kernel(
            config={
                "block_size": 16,
                "load_eviction_policies": ["first", "first"],
                "indexing": indexing,
            }
        )
        def kernel_with_override(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                # Explicit eviction_policy should override tunable
                val_x = hl.load(x, [tile], eviction_policy="evict_last")
                val_y = hl.load(y, [tile])
                out[tile] = val_x + val_y
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(kernel_with_override, (x, y))
        torch.testing.assert_close(result, x + y)

        self.assertIn("evict_last", code)
        self.assertExpectedJournal(code)

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    def test_multiple_loads_different_policies(self, indexing: str):
        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")
        elif indexing == "block_ptr" and supports_tensor_descriptor():
            self.skipTest("Block pointer is not used if tensor descriptor is available")

        @helion.kernel(
            config={
                "block_size": 16,
                "load_eviction_policies": ["first", "last", ""],
                "indexing": indexing,
            }
        )
        def kernel_multiple_loads(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val_x = hl.load(x, [tile])  # evict_first
                val_y = hl.load(y, [tile])  # evict_last
                val_z = hl.load(z, [tile])  # None
                out[tile] = val_x + val_y + val_z
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)
        z = torch.randn([128], device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(kernel_multiple_loads, (x, y, z))
        torch.testing.assert_close(result, x + y + z)

        self.assertIn("evict_first", code)
        self.assertIn("evict_last", code)
        self.assertExpectedJournal(code)


instantiate_parametrized_tests(TestEvictionPolicy)


if __name__ == "__main__":
    unittest.main()
