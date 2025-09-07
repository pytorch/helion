from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._utils import counters
from helion.autotuner import StrictLocalAutotuneCache
from helion.autotuner.base_search import BaseSearch
import helion.language as hl


class BasicSearch(BaseSearch):
    def autotune(self):
        return self.config_spec.default_config()


class TestCache(RefEagerTestDisabled, TestCase):
    def test_basic(self):
        @helion.kernel(
            autotuner_fn=StrictLocalAutotuneCache[BasicSearch], use_default_config=False
        )
        def add(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        a = torch.randn(16, device=DEVICE, dtype=torch.bfloat16)
        args_a = (a, a)
        b = torch.randn(16, device=DEVICE, dtype=torch.float16)
        args_b = (b, b)

        result = add(*args_a)
        torch.testing.assert_close(result, a + a)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 0)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        add.reset()

        result = add(*args_a)
        torch.testing.assert_close(result, a + a)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        add.reset()

        result = add(*args_b)
        torch.testing.assert_close(result, b + b)

        self.assertEqual(counters["autotune"]["cache_miss"], 2)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 2)


if __name__ == "__main__":
    unittest.main()
