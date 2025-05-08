from __future__ import annotations

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
import helion.language as hl


class TestLogging(TestCase):
    def test_log_set(self):
        import logging

        from helion._logging._internal import init_logs_from_string

        init_logs_from_string("foo.bar,+fuzz.baz")
        self.assertTrue(
            helion._logging._internal._LOG_REGISTERY.log_levels["foo.bar"], logging.INFO
        )
        self.assertTrue(
            helion._logging._internal._LOG_REGISTERY.log_levels["fuzz.baz"],
            logging.DEBUG,
        )

    def test_kernel_log(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[[1]], num_warps=16, num_stages=8, indexing="pointer"
            )
        )
        def add(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(4, device=DEVICE)

        with self.assertLogs("helion.runtime.kernel", level="DEBUG") as cm:
            add(x, x)
        self.assertTrue(
            any("INFO:helion.runtime.kernel:Output code:" in msg for msg in cm.output)
        )
        self.assertTrue(
            any("DEBUG:helion.runtime.kernel:Debug string:" in msg for msg in cm.output)
        )
