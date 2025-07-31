from __future__ import annotations

from pathlib import Path
import unittest

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion._utils import counters
from helion.autotuner import StrictLocalAutotuneCache
from helion.autotuner.base_search import BaseSearch

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


class BasicSearch(BaseSearch):
    def autotune(self):
        return self.config_spec.default_config()


class TestCache(RefEagerTestDisabled, TestCase):
    def test_basic(self):
        a = torch.randn(16, device=DEVICE, dtype=torch.bfloat16)
        args_a = (a, a)
        b = torch.randn(16, device=DEVICE, dtype=torch.float16)
        args_b = (b, b)

        bound_kernel = basic_kernels.add.bind(args_a)
        config = StrictLocalAutotuneCache(BasicSearch(bound_kernel, args_a)).autotune()
        bound_kernel.set_config(config)
        result = bound_kernel(*args_a)
        torch.testing.assert_close(result, a + a)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 0)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        basic_kernels.add.reset()

        bound_kernel = basic_kernels.add.bind(args_a)
        config = StrictLocalAutotuneCache(BasicSearch(bound_kernel, args_a)).autotune()
        bound_kernel.set_config(config)
        result = bound_kernel(*args_a)
        torch.testing.assert_close(result, a + a)

        self.assertEqual(counters["autotune"]["cache_miss"], 1)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 1)

        basic_kernels.add.reset()

        bound_kernel = basic_kernels.add.bind(args_b)
        config = StrictLocalAutotuneCache(BasicSearch(bound_kernel, args_b)).autotune()
        bound_kernel.set_config(config)
        result = bound_kernel(*args_b)
        torch.testing.assert_close(result, b + b)

        self.assertEqual(counters["autotune"]["cache_miss"], 2)
        self.assertEqual(counters["autotune"]["cache_hit"], 1)
        self.assertEqual(counters["autotune"]["cache_put"], 2)


if __name__ == "__main__":
    unittest.main()
