"""``hl.barrier()`` on Metal: ticketed phases in one flat launch
(``metal/ticketed_phases.py``), run on the GPU.  What codegen plans for them
is checked without one in ``test_metal_phase_codegen.py``."""

from __future__ import annotations

import sys
import unittest

import pytest
import torch

import helion
from helion._testing import code_and_output
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal import run_with_msl
from helion._testing.metal_phases import matvec_then_norm
from helion._testing.metal_phases import pingpong
from helion._testing.metal_phases import row_sums_then_scale
from helion._testing.metal_phases import scale_by_first
from helion._testing.metal_phases import scale_then_matmul
from helion._testing.metal_phases import scale_then_reverse
from helion._testing.metal_phases import three_phases

# Metal runs on macOS only: CI's Linux jobs skip the whole file, and other
# backends' runs on a Mac skip it too.
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]

DEVICE = "mps"


def _pingpong_ref(a: torch.Tensor) -> torch.Tensor:
    for _ in range(4):
        a = a.flip(0) * 0.5 + 1.0
    return a


@unittest.skipUnless(torch.backends.mps.is_available(), "requires a Metal GPU")
class TestMetalPhases(unittest.TestCase):
    def test_pingpong(self) -> None:
        """Bit-exact at grids far beyond what the GPU keeps resident (16384
        threadgroups per phase at 1 << 20): no deadlock and no stale reads."""
        for n, block in ((4096, 64), (1 << 20, 64), (1 << 20, 256)):
            a = torch.randn(n, device=DEVICE)
            expected = _pingpong_ref(a.clone())
            _, out = code_and_output(
                pingpong,
                (a, torch.empty_like(a)),
                block_sizes=[block] * 4,
                pid_type="flat",
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_scalar_tensor_handed_on(self) -> None:
        """Phase 1 reads the 0-d tensor phase 0 wrote, across 64
        threadgroups."""
        x = torch.randn(4096, device=DEVICE)
        scale = torch.zeros((), device=DEVICE)
        _, out = code_and_output(
            scale_by_first, (x, scale), block_sizes=[1, 64], pid_type="flat"
        )
        torch.testing.assert_close(scale, x[0] * 2.0, rtol=0, atol=0)
        torch.testing.assert_close(out, x * (x[0] * 2.0), rtol=0, atol=0)

    def test_roots_and_views(self) -> None:
        x = torch.randn(64, 96, device=DEVICE)
        _, out = code_and_output(
            row_sums_then_scale,
            (x,),
            block_sizes=[1, 128, 4, 32],
            pid_type="flat",
        )
        expected = x * 2.0 + x.sum(-1).flip(0)[:, None]
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_root_layouts(self) -> None:
        """Roots with different thread layouts (32 x 8 and 256) run on one
        flat 256-thread launch, each with its own ``tid``."""
        x = torch.randn(4, 512, device=DEVICE)
        w = torch.randn(1024, 512, device=DEVICE)
        _, out = code_and_output(
            matvec_then_norm,
            (x, w),
            block_sizes=[32, 256, 4, 8, 1],
            num_threads=[32, 256, 1, 8, 1],
            num_warps=8,
            pid_type="flat",
        )
        y = x @ w.T
        expected = y * torch.rsqrt((y * y).mean(-1, keepdim=True))
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_partial_lines(self) -> None:
        """Tiles of 16 fp32 (half a 128-byte line): ``x``, updated in place
        in phase 0, and ``y``, written there, are read back reversed in phase
        1 through their coherent aliases, bit-exact; ``out`` keeps plain
        access."""
        x = torch.randn(1 << 16, device=DEVICE)
        expected_x = x * 2.0 + 1.0
        expected = (expected_x + expected_x * 0.5).flip(0)
        out, _, msl = run_with_msl(
            scale_then_reverse,
            (x, torch.empty_like(x)),
            helion.Config(block_sizes=[16, 16], pid_type="flat"),
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertIn("x__coherent", msl)
        self.assertIn("y__coherent", msl)
        self.assertNotIn("out__coherent", msl)

    def test_surplus_threads(self) -> None:
        """Block 1 beside block 64: phase 0's in-place update lands once,
        though its root runs on all 64 threads of the launch."""
        x = torch.randn(1 << 12, device=DEVICE)
        expected_x = x * 2.0 + 1.0
        expected = (expected_x + expected_x * 0.5).flip(0)
        out, _, _ = run_with_msl(
            scale_then_reverse,
            (x, torch.empty_like(x)),
            helion.Config(block_sizes=[1, 64], pid_type="flat"),
        )
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_coherent_mpp_operand(self) -> None:
        """An MPP matmul stages the operand an earlier phase wrote, as it does
        every operand the kernel writes, through its coherent alias."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        w = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        out, _, msl = run_with_msl(
            scale_then_matmul,
            (x, w),
            helion.Config(block_sizes=[16, 64, 32, 32, 32], pid_type="flat"),
        )
        self.assertIn("matmul2d", msl)
        self.assertIn("xs__coherent[", msl)
        self.assertNotIn("tensor<device coherent", msl)
        expected = (x * 2.0).float() @ w.float()
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-2)

    def test_narrow_matmul_roots(self) -> None:
        """Matmul phases whose roots lay out fewer threads than the launch:
        the split-K example (an MPP root of one thread, then a 128-thread sum
        of the partials) at three splits, and an 8 x 16 matmul root in a
        64 x 16 launch."""
        split_k_matmul = import_example("split_k_barrier").split_k_matmul
        for m, n, split_k, block_sizes in (
            (16, 16, 64, [16, 16, 16, 1, 1]),
            (32, 32, 16, [8, 8, 16, 1, 1]),
            (16, 16, 256, [16, 16, 32, 2, 1]),
        ):
            a = torch.randn(m, 4096, device=DEVICE)
            b = torch.randn(4096, n, device=DEVICE)
            with self.subTest(split_k=split_k, block_sizes=block_sizes):
                out, _, _ = run_with_msl(
                    split_k_matmul,
                    (a, b),
                    helion.Config(
                        block_sizes=block_sizes,
                        reduction_loops=[1024],
                        split_k=split_k,
                        pid_type="flat",
                    ),
                )
                torch.testing.assert_close(out, a @ b, rtol=1e-3, atol=1e-2)
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        w = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        out, _, _ = run_with_msl(
            scale_then_matmul,
            (x, w),
            helion.Config(block_sizes=[1, 64, 32, 32, 32], pid_type="flat"),
        )
        expected = (x * 2.0).float() @ w.float()
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-2)

    def test_empty_phases(self) -> None:
        """A phase with no tasks raises no flag: the next phase waits for the
        last earlier phase that has tasks instead of spinning forever (an
        empty input, an MoE expert without tokens)."""
        for n, m in ((4096, 0), (0, 4096), (4096, 4096)):
            a = torch.randn(n, device=DEVICE)
            b = torch.randn(m, device=DEVICE)
            expected = ((a + 1.0) * 3.0).flip(0)
            expected_b = b * 2.0
            with self.subTest(n=n, m=m):
                _, out = code_and_output(
                    three_phases, (a, b), block_sizes=[64, 64, 64], pid_type="flat"
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                torch.testing.assert_close(b, expected_b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
