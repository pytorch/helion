from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
import helion._compat as _compat
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTensorDescriptor
import helion.language as hl


@onlyBackends(["triton"])
class TestCacheModifier(RefEagerTestBase, TestCase):
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
                    "test.test_cache_modifier.supports_tensor_descriptor",
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

    @parametrize("indexing", ("pointer", "block_ptr"))
    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_hl_load_cache_modifier_emitted(self, indexing: str):
        with self._indexing_context(indexing):

            @helion.kernel(config={"indexing": indexing, "block_size": 16})
            def copy_with_cache_modifier(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(x.size(0)):
                    val = hl.load(x, [tile], cache_modifier=".cg")
                    out[tile] = val
                return out

            x = torch.randn([128], device=DEVICE, dtype=torch.float32)
            code, result = code_and_output(copy_with_cache_modifier, (x,))
            torch.testing.assert_close(result, x)
            self.assertIn("cache_modifier", code)
            self.assertIn(".cg", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_hl_load_cache_modifier_ignored_for_tensor_descriptor(self):
        @helion.kernel(config={"indexing": "tensor_descriptor", "block_size": [8, 16]})
        def copy_with_cache_modifier(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                val = hl.load(x, [tile_m, tile_n], cache_modifier=".cg")
                out[tile_m, tile_n] = val
            return out

        x = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(copy_with_cache_modifier, (x,))
        torch.testing.assert_close(result, x)
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertNotIn("cache_modifier", code)
        self.assertNotIn(".cg", code)

    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_hl_load_cache_modifier_preserved_for_tensor_descriptor_fallback(self):
        @helion.kernel(config={"indexing": "tensor_descriptor", "block_size": 16})
        def copy_with_cache_modifier(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                val = hl.load(x, [tile], cache_modifier=".cg")
                out[tile] = val
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(copy_with_cache_modifier, (x,))
        torch.testing.assert_close(result, x)
        self.assertNotIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("cache_modifier", code)
        self.assertIn(".cg", code)

    @parametrize("indexing", ("pointer", "block_ptr"))
    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_hl_store_cache_modifier_emitted(self, indexing: str):
        with self._indexing_context(indexing):

            @helion.kernel(config={"indexing": indexing, "block_size": 16})
            def copy_with_store_cache_modifier(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(x.size(0)):
                    hl.store(out, [tile], x[tile], cache_modifier=".wt")
                return out

            x = torch.randn([128], device=DEVICE, dtype=torch.float32)
            code, result = code_and_output(copy_with_store_cache_modifier, (x,))
            torch.testing.assert_close(result, x)
            self.assertIn("cache_modifier", code)
            self.assertIn(".wt", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_hl_store_cache_modifier_ignored_for_tensor_descriptor(self):
        @helion.kernel(config={"indexing": "tensor_descriptor", "block_size": [8, 16]})
        def copy_with_store_cache_modifier(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                hl.store(out, [tile_m, tile_n], x[tile_m, tile_n], cache_modifier=".wt")
            return out

        x = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(copy_with_store_cache_modifier, (x,))
        torch.testing.assert_close(result, x)
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertNotIn("cache_modifier", code)
        self.assertNotIn(".wt", code)

    @skipIfTileIR("tileir backend will ignore `cache_modifier` hint")
    def test_hl_store_cache_modifier_preserved_for_tensor_descriptor_fallback(self):
        @helion.kernel(config={"indexing": "tensor_descriptor", "block_size": 16})
        def copy_with_store_cache_modifier(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                hl.store(out, [tile], x[tile], cache_modifier=".wt")
            return out

        x = torch.randn([128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(copy_with_store_cache_modifier, (x,))
        torch.testing.assert_close(result, x)
        self.assertNotIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("cache_modifier", code)
        self.assertIn(".wt", code)


instantiate_parametrized_tests(TestCacheModifier)


if __name__ == "__main__":
    unittest.main()
