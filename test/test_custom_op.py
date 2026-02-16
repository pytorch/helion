from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestCustomOp(RefEagerTestBase, TestCase):
    def test_custom_op(self):
        """Test directly registering a helion kernel as PyTorch custom op"""

        @torch.library.custom_op("testlib::sub_one", mutates_args=())
        @helion.kernel(autotune_effort="none")
        def sub_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] - 1.0
            return out

        @sub_one.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        x = torch.randn([64, 64], device=DEVICE)
        expected = x - 1.0

        # Test calling via torch.ops.testlib.sub_one (registry access)
        result_registry = torch.ops.testlib.sub_one(x)
        torch.testing.assert_close(result_registry, expected)

        # Verify the direct call also works (backwards compatibility)
        result_direct = sub_one(x)
        torch.testing.assert_close(result_direct, expected)
