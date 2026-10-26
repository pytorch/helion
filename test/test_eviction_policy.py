from __future__ import annotations

import contextlib
import re
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
import helion._compat as _compat
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl
from helion.runtime.settings import _get_backend

# The load keyword carrying the L1 eviction priority (Triton's tl.load vs
# CuTe's cute.arch.load); asserting on it keeps the `# src[...]` comments
# that echo the kernel source from satisfying the checks.
_EVICTION_KWARG = (
    "level1_eviction_priority" if _get_backend() == "cute" else "eviction_policy"
)


def _hinted_loads(code: str, policy: str) -> list[str]:
    """The generated lines that load with L1 eviction ``policy``.

    CuTe lowers a hinted scalar load to an exact-policy PTX helper
    (``cute/scalar_policy_loads.py``) whose last argument names the policy.
    """
    lowered = re.compile(
        rf"_cute_scalar_policy_load\(.*'{policy.removeprefix('evict_')}'\)"
    )
    return [
        line
        for line in code.splitlines()
        if f"{_EVICTION_KWARG}='{policy}'" in line or lowered.search(line)
    ]


@onlyBackends(["triton", "cute"])
class TestEvictionPolicy(RefEagerTestBase, TestCase):
    @contextlib.contextmanager
    def _indexing_context(self, indexing: str) -> None:
        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")

        if indexing == "block_ptr" and supports_tensor_descriptor():
            original_cached = _compat._supports_tensor_descriptor
            original_cached.cache_clear()
            patches = [
                mock.patch.object(
                    _compat, "_supports_tensor_descriptor", lambda: False
                ),
                mock.patch.object(_compat, "supports_tensor_descriptor", lambda: False),
                mock.patch(
                    "test.test_eviction_policy.supports_tensor_descriptor",
                    lambda: False,
                ),
            ]
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                try:
                    yield
                finally:
                    original_cached.cache_clear()
            return

        yield

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    def test_hl_load_eviction_policy_emitted(self, indexing: str):
        with self._indexing_context(indexing):

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
            self.assertTrueIfInNormalMode(bool(_hinted_loads(code, "evict_last")))

    @skipIfRefEager("Config spec inspection not applicable in ref eager mode")
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    @skipIfRocm("ROCm does not support eviction policy")
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
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    def test_eviction_policy_in_generated_code(self, indexing: str):
        """Test that eviction policies appear in generated code when configured."""

        with self._indexing_context(indexing):

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
            self.assertTrueIfInNormalMode(bool(_hinted_loads(code, "evict_last")))

    @skipIfRefEager("Generated code inspection not applicable in ref eager mode")
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    @skipIfRocm("ROCm does not support eviction policy")
    def test_scalar_eviction_policy_applies_to_every_load(self) -> None:
        @helion.kernel(
            config={
                "block_size": 16,
                "load_eviction_policies": "last",
                "indexing": "pointer",
            }
        )
        def kernel_with_eviction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128], device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(kernel_with_eviction, (x, y))

        torch.testing.assert_close(result, x + y)
        self.assertEqual(len(_hinted_loads(code, "evict_last")), 2)

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    def test_explicit_eviction_policy_overrides_tunable(self, indexing: str):
        with self._indexing_context(indexing):

            @helion.kernel(
                config={
                    "block_size": 16,
                    # Only the load without an explicit policy owns a slot
                    "load_eviction_policies": ["first"],
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

            self.assertTrueIfInNormalMode(bool(_hinted_loads(code, "evict_last")))
            self.assertTrueIfInNormalMode(bool(_hinted_loads(code, "evict_first")))

    @skipIfRefEager("Generated code inspection not applicable in ref eager mode")
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    @skipIfRocm("ROCm does not support eviction policy")
    def test_tile_index_load_keeps_later_eviction_slots(self) -> None:
        """A tile-index read owns the first slot; the loads after it keep theirs."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size(), block_size=[16, 16]):
                rows = tile_m.index[:, None]
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n] + rows
            return out

        x = torch.randn([64, 64], device=DEVICE, dtype=torch.float32)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.float32)
        self.assertEqual(
            kernel.bind((x, y)).config_spec.load_eviction_policies.length, 3
        )
        code, result = code_and_output(
            kernel,
            (x, y),
            load_eviction_policies=["", "first", "last"],
            indexing="pointer",
        )
        rows = torch.arange(64, device=DEVICE, dtype=torch.float32)[:, None]
        torch.testing.assert_close(result, x + y + rows)
        hinted = {
            policy: _hinted_loads(code, policy)
            for policy in ("evict_first", "evict_last")
        }
        self.assertEqual(len(hinted["evict_first"]), 1)
        self.assertEqual(len(hinted["evict_last"]), 1)
        self.assertRegex(hinted["evict_first"][0], r"\bx\b")
        self.assertRegex(hinted["evict_last"][0], r"\by\b")

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    @skipIfTileIR("tileir backend will ignore `eviction_policy` hint")
    def test_multiple_loads_different_policies(self, indexing: str):
        with self._indexing_context(indexing):

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

            self.assertTrueIfInNormalMode(bool(_hinted_loads(code, "evict_first")))
            self.assertTrueIfInNormalMode(bool(_hinted_loads(code, "evict_last")))


instantiate_parametrized_tests(TestEvictionPolicy)


if __name__ == "__main__":
    unittest.main()
