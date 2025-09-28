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
import helion.language as hl


class TestEvictionPolicy(RefEagerTestBase, TestCase):
    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    def test_hl_load_eviction_policy_emitted(self, indexing: str):
        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")

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
        if indexing != "tensor_descriptor":
            # TODO(oulgen): Update this on a machine that supports tensor_descriptor
            self.assertExpectedJournal(code)
        self.assertIn("eviction_policy", code)
        self.assertIn("evict_last", code)


instantiate_parametrized_tests(TestEvictionPolicy)


if __name__ == "__main__":
    unittest.main()
