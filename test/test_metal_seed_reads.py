"""A value an MPP K loop's accumulator starts from, read after the loop:
MPP refuses it (it read the op's result, the loop having updated the
storage in place), and a config without metal_matmul falls back to a
lowering that keeps the value.  Binds CPU tensors only, so these run on
the CPU lane."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion import exc
from helion._testing import skipUnlessBackends
from helion._testing.metal_seed_reads import SEED_READ_CONFIG
from helion._testing.metal_seed_reads import carry_read_after_its_update
from helion._testing.metal_seed_reads import carry_read_after_its_update_op
from helion._testing.metal_seed_reads import seed_read_after_its_loop

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


class TestMetalSeedReads(unittest.TestCase):
    def test_seed_read_after_its_loop(self) -> None:
        args = (torch.randn(32, 64), torch.randn(64, 32), torch.randn(32, 32))
        bound = seed_read_after_its_loop.bind(args)
        mpp = helion.Config.from_dict(
            {**SEED_READ_CONFIG.config, "metal_matmul": "mpp"}
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "a value an MPP K loop's accumulator starts from, read after the loop",
        ):
            bound.to_code(mpp)
        self.assertNotIn("_helion_mpp_run(", bound.to_code(SEED_READ_CONFIG))

    def test_carry_read_after_its_update(self) -> None:
        """A copy of the carry inside the K loop, read after the op that
        updates its storage in place: MPP refuses it (it would read this K
        tile's result), and the default config falls back to a lowering that
        keeps the value."""
        args = (torch.randn(32, 64), torch.randn(64, 32))
        mpp = helion.Config.from_dict(
            {**SEED_READ_CONFIG.config, "metal_matmul": "mpp"}
        )
        for kernel in (carry_read_after_its_update, carry_read_after_its_update_op):
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind(args)
                with self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    "holds before an op, read after the op",
                ):
                    bound.to_code(mpp)
                self.assertNotIn("_helion_mpp_run(", bound.to_code(SEED_READ_CONFIG))


if __name__ == "__main__":
    unittest.main()
