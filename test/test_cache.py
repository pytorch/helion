from __future__ import annotations

import os
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
from helion._utils import counters
from helion.autotuner import StrictLocalAutotuneCache
from helion.autotuner.base_search import BaseSearch
import helion.language as hl


class BasicSearch(BaseSearch):
    def autotune(self, *, skip_cache: bool = False):
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


KERNELS = {
    "add": get_add_kernel,
    "matmul": get_matmul_kernel,
    "welford": get_welford_kernel,
}


class TestCache(RefEagerTestDisabled, TestCase):
    @parametrize("name", ("add", "matmul", "welford"))
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


instantiate_parametrized_tests(TestCache)


if __name__ == "__main__":
    unittest.main()
