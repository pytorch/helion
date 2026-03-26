from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import ClassVar
import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._utils import counters
from helion.autotuner import StrictLocalAutotuneCache
from helion.autotuner.base_search import BaseSearch
import helion.language as hl


class BasicSearch(BaseSearch):
    def autotune(self, *, skip_cache: bool = False):
        return self.config_spec.default_config()


class MultiConfigSearch(BaseSearch):
    """Compiles two configs (default + random) and returns the default."""

    def autotune(self, *, skip_cache: bool = False):
        default = self.config_spec.default_config()
        config_gen = self.config_spec.create_config_generation()
        other = config_gen.random_config()
        for cfg in (default, other):
            fn = self.kernel.compile_config(cfg, allow_print=False)
            fn(*self.args)
        return default


class MinimizingBasicSearch(BaseSearch):
    """Like BasicSearch but returns a minimized config, simulating the
    finishing phase in PopulationBasedSearch.run_finishing_phase."""

    def autotune(self, *, skip_cache: bool = False):
        config = self.config_spec.default_config()
        # Compile and run the kernel so device_caches is populated
        fn = self.kernel.compile_config(config, allow_print=False)
        fn(*self.args)
        # Return a minimized config (only block_sizes), like run_finishing_phase does
        return config.minimize(self.config_spec)


class CapturingSearch(BaseSearch):
    """Search that captures os.environ during autotuning.

    Overrides _autotune() (not autotune()) so that BaseSearch.autotune()
    runs its full setup including patch.dict env overrides.
    """

    captured_env: ClassVar[dict[str, str | None]] = {}

    def _autotune(self):
        CapturingSearch.captured_env = {
            "TRITON_STORE_BINARY_ONLY": os.environ.get("TRITON_STORE_BINARY_ONLY"),
        }
        return self.config_spec.default_config()


def get_add_kernel():
    kernel = import_path(EXAMPLES_DIR / "add.py").add
    a = torch.randn(16, device=DEVICE, dtype=torch.bfloat16)
    args_a = (a, a)
    b = torch.randn(16, device=DEVICE, dtype=torch.float16)
    args_b = (b, b)
    return kernel, args_a, a + a, args_b, b + b


def get_matmul_kernel():
    kernel = import_path(EXAMPLES_DIR / "matmul.py").matmul
    a = torch.randn(16, 16, device=DEVICE, dtype=torch.bfloat16)
    args_a = (a, a, lambda acc, tile: torch.relu(acc))
    args_b = (a, a, lambda acc, tile: torch.sigmoid(acc))
    return kernel, args_a, torch.relu(a @ a), args_b, torch.sigmoid(a @ a)


def get_welford_kernel():
    kernel = import_path(EXAMPLES_DIR / "welford.py").welford
    eager = import_path(EXAMPLES_DIR / "welford.py").eager_layer_norm

    s, d = 2**10, 2**4
    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)
    args_a = (weight, bias, x)
    result_a = eager(*args_a)

    s, d = 2**10, 2**6
    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)
    args_b = (weight, bias, x)
    result_b = eager(*args_b)

    return kernel, args_a, result_a, args_b, result_b


def get_list_tensor_kernel():
    """Kernel that takes a list of tensors as input to test list caching.

    This tests that the cache correctly handles list arguments where tensors
    may have different shapes - the cache key should capture all tensor shapes.
    """

    @helion.kernel()
    def list_sum_2(tensors: list[torch.Tensor]) -> torch.Tensor:
        """Sum the first two tensors in the list element-wise."""
        n = tensors[0].size(0)
        out = torch.empty_like(tensors[0])
        for tile in hl.tile(n):
            out[tile] = tensors[0][tile] + tensors[1][tile]
        return out

    # Same shapes - should cache hit
    a1 = torch.randn(16, device=DEVICE, dtype=torch.float32)
    a2 = torch.randn(16, device=DEVICE, dtype=torch.float32)
    args_a = ([a1, a2],)
    result_a = a1 + a2

    # Different shapes - should cache miss
    b1 = torch.randn(32, device=DEVICE, dtype=torch.float32)
    b2 = torch.randn(32, device=DEVICE, dtype=torch.float32)
    args_b = ([b1, b2],)
    result_b = b1 + b2

    return list_sum_2, args_a, result_a, args_b, result_b


def get_list_tensor_different_shapes_kernel():
    """Kernel with list of 2D tensors that have different shapes within the list.

    This tests that the cache key correctly captures all tensor shapes in the list,
    not just the first one.
    """

    @helion.kernel()
    def list_gather_sum_2(
        tensors: list[torch.Tensor], indices: torch.Tensor
    ) -> torch.Tensor:
        """Gather from the first two tensors and sum the results.

        Each tensor in the list must have the same second dimension (D).
        """
        n = indices.size(0)
        d = tensors[0].size(1)
        out = tensors[0].new_zeros([n, d])
        for tile_n in hl.tile(n):
            idx = indices[tile_n]
            out[tile_n, :] = tensors[0][idx, :] + tensors[1][idx, :]
        return out

    # Tensors with same D but different N (number of rows)
    t1 = torch.randn(100, 16, device=DEVICE, dtype=torch.float32)
    t2 = torch.randn(200, 16, device=DEVICE, dtype=torch.float32)
    indices = torch.randint(0, 100, (8,), device=DEVICE)
    args_a = ([t1, t2], indices)
    result_a = t1[indices] + t2[indices]

    # Different table sizes - should cache miss due to different shapes
    t3 = torch.randn(150, 16, device=DEVICE, dtype=torch.float32)
    t4 = torch.randn(250, 16, device=DEVICE, dtype=torch.float32)
    indices_b = torch.randint(0, 150, (16,), device=DEVICE)
    args_b = ([t3, t4], indices_b)
    result_b = t3[indices_b] + t4[indices_b]

    return list_gather_sum_2, args_a, result_a, args_b, result_b


KERNELS = {
    "add": get_add_kernel,
    "matmul": get_matmul_kernel,
    "welford": get_welford_kernel,
    "list_tensor": get_list_tensor_kernel,
    "list_tensor_different_shapes": get_list_tensor_different_shapes_kernel,
}


@onlyBackends(["triton"])
class TestCache(RefEagerTestDisabled, TestCase):
    @parametrize(
        "name",
        ("add", "matmul", "welford", "list_tensor", "list_tensor_different_shapes"),
    )
    def test_kernel(self, name):
        kernel, args_a, result_a, args_b, result_b = KERNELS[name]()

        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel.settings.autotune_effort = "full"

        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 0)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        kernel.reset()

        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        kernel.reset()

        result = kernel(*args_b)
        torch.testing.assert_close(result, result_b, rtol=1e-2, atol=5e-2)

        self.assertEqual(counters["autotune"]["cache_miss"], 2)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 2)

    def test_key_affects_cache_specialization(self):
        counters["autotune"].clear()
        self.addCleanup(counters["autotune"].clear)

        def shape_key(x: torch.Tensor) -> tuple[int, ...]:
            return tuple(x.size())

        @helion.kernel(
            autotuner_fn=StrictLocalAutotuneCache[BasicSearch],
            key=shape_key,
        )
        def add_one(x: torch.Tensor):
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + 1
            return out

        a = torch.randn(16, device=DEVICE, dtype=torch.float32)
        b = torch.randn(32, device=DEVICE, dtype=torch.float32)

        result_a = add_one(a)
        torch.testing.assert_close(result_a, a + 1)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 0)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        add_one.reset()

        result_a_again = add_one(a)
        torch.testing.assert_close(result_a_again, a + 1)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        add_one.reset()

        result_b = add_one(b)
        torch.testing.assert_close(result_b, b + 1)

        self.assertEqual(counters["autotune"]["cache_miss"], 2)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 2)

    def test_assert_cache_hit(self):
        counters["autotune"].clear()
        self.addCleanup(counters["autotune"].clear)

        kernel, args_a, result_a, args_b, result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel.settings.autotune_effort = "full"

        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a)
        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 0)

        kernel.reset()
        with patch.dict(os.environ, {"HELION_ASSERT_CACHE_HIT": "1"}):
            result = kernel(*args_a)
            torch.testing.assert_close(result, result_a)
            self.assertEqual(counters["autotune"]["cache_miss"], 1)
            self.assertEqual(counters["autotune"]["cache_hit"], 1)

        kernel.reset()
        with patch.dict(os.environ, {"HELION_ASSERT_CACHE_HIT": "1"}):
            with self.assertRaises(exc.CacheAssertionError) as cm:
                kernel(*args_b)

            self.assertIn("add", str(cm.exception))
            # cache_miss incremented before error, but cache_put not (autotuning prevented)
            self.assertEqual(counters["autotune"]["cache_miss"], 2)
            self.assertEqual(counters["autotune"]["cache_put"], 1)

    def test_force_autotune_skips_read_but_writes(self):
        """HELION_FORCE_AUTOTUNE=1 skips the cache read but writes back the result."""
        counters["autotune"].clear()
        self.addCleanup(counters["autotune"].clear)

        kernel, args_a, result_a, args_b, result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel.settings.autotune_effort = "full"

        # First call: populates the cache
        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)
        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        kernel.reset()

        # Second call with force_autotune: should skip the cached entry
        # but still write the new result back
        kernel.settings.force_autotune = True
        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

        self.assertEqual(counters["autotune"]["cache_hit"], 0)
        self.assertEqual(counters["autotune"]["cache_miss"], 2)
        self.assertEqual(counters["autotune"]["cache_put"], 2)

        kernel.reset()

        # Third call without force_autotune: should hit the cache
        kernel.settings.force_autotune = False
        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 2)

    def test_skip_cache_skips_read_and_write(self):
        """HELION_SKIP_CACHE=1 skips both reading and writing the cache."""
        counters["autotune"].clear()
        self.addCleanup(counters["autotune"].clear)

        kernel, args_a, result_a, args_b, result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel.settings.autotune_effort = "full"

        # First call with SKIP_CACHE: should not write to the cache
        with patch.dict(os.environ, {"HELION_SKIP_CACHE": "1"}):
            result = kernel(*args_a)
            torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 0)

        kernel.reset()

        # Second call without SKIP_CACHE: should miss (nothing was written)
        result = kernel(*args_a)
        torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)
        self.assertEqual(counters["autotune"]["cache_miss"], 2)
        self.assertEqual(counters["autotune"]["cache_hit"], 0)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

    def test_backend_cache_key_before_compilation(self):
        """backend_cache_key returns None before the kernel is compiled."""
        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        bound = kernel.bind(args_a)
        config = bound.config_spec.default_config()
        self.assertIsNone(bound.backend_cache_key(config))

    def test_backend_cache_key_after_compilation(self):
        """backend_cache_key returns a base32 string after compilation."""
        import re

        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel(*args_a)

        bound = kernel.bind(args_a)
        key = bound.backend_cache_key()
        self.assertIsNotNone(key)
        self.assertIsInstance(key, str)
        self.assertGreater(len(key), 0)
        self.assertRegex(key, re.compile(r"^[A-Z2-7]+$"))

    def test_backend_cache_key_stable(self):
        """backend_cache_key returns the same value on repeated calls."""
        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel(*args_a)

        bound = kernel.bind(args_a)
        key1 = bound.backend_cache_key()
        key2 = bound.backend_cache_key()
        self.assertIsNotNone(key1)
        self.assertEqual(key1, key2)

    def test_backend_cache_key_explicit_config(self):
        """backend_cache_key returns the same key with implicit, Config, and dict configs."""

        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel(*args_a)

        bound = kernel.bind(args_a)
        key_implicit = bound.backend_cache_key()
        config = bound._require_implicit_config()
        key_config = bound.backend_cache_key(config)
        key_dict = bound.backend_cache_key(dict(config))
        self.assertIsNotNone(key_implicit)
        self.assertEqual(key_implicit, key_config)
        self.assertEqual(key_implicit, key_dict)

    def test_backend_cache_key_matches_cache_directory(self):
        """backend_cache_key corresponds to an actual directory in the Triton cache."""
        import pathlib

        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        kernel(*args_a)

        bound = kernel.bind(args_a)
        key = bound.backend_cache_key()
        self.assertIsNotNone(key)

        cache_root = pathlib.Path(os.environ["TRITON_CACHE_DIR"])
        cache_dir = cache_root / key
        self.assertTrue(
            cache_dir.is_dir(), f"Expected cache directory {cache_dir} to exist"
        )

    def test_backend_cache_key_written_to_cache_file(self):
        """backend_cache_key is persisted in the .best_config JSON file.

        Uses MinimizingBasicSearch to simulate the finishing phase that
        returns a minimized config (default values stripped), which is
        what real autotuners do via run_finishing_phase + Config.minimize.
        """
        import json
        import pathlib

        from helion.autotuner.local_cache import get_helion_cache_dir

        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[MinimizingBasicSearch]
        kernel(*args_a)

        # Find the .best_config file written by put()
        cache_root = get_helion_cache_dir()
        best_config_files = list(pathlib.Path(cache_root).glob("*.best_config"))
        self.assertGreater(len(best_config_files), 0, "No .best_config file found")

        data = json.loads(best_config_files[0].read_text())
        self.assertIn("backend_cache_key", data)
        self.assertIsInstance(data["backend_cache_key"], str)
        self.assertGreater(len(data["backend_cache_key"]), 0)

    def test_triton_cache_dir_set_under_helion_cache(self):
        """TRITON_CACHE_DIR is set under the Helion cache root after compilation."""
        import pathlib

        from helion.autotuner.local_cache import get_helion_cache_dir

        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRITON_CACHE_DIR", None)
            kernel(*args_a)

            self.assertIn("TRITON_CACHE_DIR", os.environ)
            triton_dir = pathlib.Path(os.environ["TRITON_CACHE_DIR"])
            helion_root = get_helion_cache_dir()
            self.assertTrue(
                triton_dir.is_relative_to(helion_root / "triton"),
                f"Expected {triton_dir} to be under {helion_root / 'triton'}",
            )

    def test_triton_cache_dir_respects_user_override(self):
        """User-set TRITON_CACHE_DIR is not overwritten by Helion."""
        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]

        user_dir = "/tmp/my_custom_triton_cache"
        with patch.dict(os.environ, {"TRITON_CACHE_DIR": user_dir}):
            kernel(*args_a)
            self.assertEqual(os.environ["TRITON_CACHE_DIR"], user_dir)

    def test_cache_key_includes_backend(self):
        """Different backends produce different cache key hashes."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        base_fields = {
            "specialization_key": ("test",),
            "extra_results": (),
            "kernel_source_hash": "abc123",
            "hardware": "NVIDIA B200",
            "runtime_name": "13.0",
        }
        key_triton = LooseAutotuneCacheKey(**base_fields, backend="triton")
        key_tileir = LooseAutotuneCacheKey(**base_fields, backend="tileir")
        key_triton2 = LooseAutotuneCacheKey(**base_fields, backend="triton")

        self.assertNotEqual(key_triton.stable_hash(), key_tileir.stable_hash())
        self.assertEqual(key_triton.stable_hash(), key_triton2.stable_hash())

    def test_store_binary_only_set_during_autotuning(self):
        """TRITON_STORE_BINARY_ONLY is set to '1' during autotuning by default."""
        CapturingSearch.captured_env.clear()
        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[CapturingSearch]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRITON_STORE_BINARY_ONLY", None)
            kernel(*args_a)

        self.assertEqual(
            CapturingSearch.captured_env.get("TRITON_STORE_BINARY_ONLY"), "1"
        )

    def test_store_binary_only_respects_user_override(self):
        """User-set TRITON_STORE_BINARY_ONLY is not overwritten during autotuning."""
        CapturingSearch.captured_env.clear()
        kernel, args_a, _result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[CapturingSearch]

        with patch.dict(os.environ, {"TRITON_STORE_BINARY_ONLY": "0"}):
            kernel(*args_a)

        self.assertEqual(
            CapturingSearch.captured_env.get("TRITON_STORE_BINARY_ONLY"), "0"
        )

    def test_ephemeral_triton_cache(self):
        """Autotuning with multiple candidates keeps only the winner."""
        kernel, args_a, result_a, _args_b, _result_b = KERNELS["add"]()

        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[BasicSearch]
        with (
            tempfile.TemporaryDirectory() as baseline_tmp,
            patch.dict(os.environ, {"HELION_CACHE_DIR": baseline_tmp}),
        ):
            os.environ.pop("TRITON_CACHE_DIR", None)
            kernel(*args_a)
            baseline_cache = Path(baseline_tmp) / "triton" / "0"
            baseline_count = len(list(baseline_cache.iterdir()))

        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[MultiConfigSearch]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"HELION_CACHE_DIR": tmp}),
        ):
            os.environ.pop("TRITON_CACHE_DIR", None)
            result = kernel(*args_a)
            torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

            triton_cache = Path(tmp) / "triton" / "0"
            self.assertTrue(triton_cache.exists())
            entries = [p for p in triton_cache.iterdir() if not p.name.startswith(".")]
            self.assertEqual(len(entries), baseline_count)

            bound = kernel.bind(args_a)
            cache_key = bound.backend_cache_key()
            self.assertIsNotNone(cache_key)
            self.assertTrue((triton_cache / cache_key).exists())

            kernel.reset()
            result2 = kernel(*args_a)
            torch.testing.assert_close(result2, result_a, rtol=1e-2, atol=5e-2)

    def test_keep_triton_cache_disables_ephemeral(self):
        """HELION_KEEP_TRITON_CACHE=1 writes all candidates to the real cache."""
        kernel, args_a, result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[MultiConfigSearch]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {"HELION_CACHE_DIR": tmp, "HELION_KEEP_TRITON_CACHE": "1"},
            ),
        ):
            os.environ.pop("TRITON_CACHE_DIR", None)
            result = kernel(*args_a)
            torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

            triton_cache = Path(tmp) / "triton" / "0"
            self.assertEqual(os.environ.get("TRITON_CACHE_DIR"), str(triton_cache))
            self.assertTrue(triton_cache.exists())
            entries = [p for p in triton_cache.iterdir() if not p.name.startswith(".")]
            self.assertGreaterEqual(len(entries), 2)

    def test_ephemeral_triton_cache_minimized_config(self):
        """Ephemeral cache works when the autotuner returns a minimized config."""
        kernel, args_a, result_a, _args_b, _result_b = KERNELS["add"]()
        kernel.reset()
        kernel.settings.autotuner_fn = StrictLocalAutotuneCache[MinimizingBasicSearch]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"HELION_CACHE_DIR": tmp}),
        ):
            os.environ.pop("TRITON_CACHE_DIR", None)
            result = kernel(*args_a)
            torch.testing.assert_close(result, result_a, rtol=1e-2, atol=5e-2)

            triton_cache = Path(tmp) / "triton" / "0"
            self.assertTrue(triton_cache.exists())
            entries = [p for p in triton_cache.iterdir() if not p.name.startswith(".")]
            self.assertGreaterEqual(len(entries), 1)

            bound = kernel.bind(args_a)
            cache_key = bound.backend_cache_key()
            self.assertIsNotNone(cache_key)
            self.assertTrue((triton_cache / cache_key).exists())


instantiate_parametrized_tests(TestCache)


if __name__ == "__main__":
    unittest.main()
