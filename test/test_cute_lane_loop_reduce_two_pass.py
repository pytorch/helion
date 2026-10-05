"""A reduction over a lane-looped tile axis combines across threads ONCE.

``col_reduce_sum``-style kernels reduce a 2-D tile along its threaded row
axis while the row block is also traversed by a per-thread lane loop.  The
reduction used to fall through to the per-element strided thread reduction,
so every synthetic lane paid a full cross-thread combine (a warp shuffle
tree, or a CTA-wide two-stage shared-memory reduction when the row threads
sit above the column threads on the linear thread index).  The two-pass
lane-reduction marker now owns these reductions: each thread accumulates
across its lanes and the cross-thread combine runs once per row tile.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _col_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = col_acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _col_reduce_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.full([tile_n], float("-inf"), dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc = torch.maximum(
                col_acc, torch.amax(x[tile_m, tile_n].to(torch.float32), dim=0)
            )
        out[tile_n] = col_acc.to(out.dtype)
    return out


def _col_sum(t: torch.Tensor) -> torch.Tensor:
    return t.float().sum(0).to(t.dtype)


def _col_max(t: torch.Tensor) -> torch.Tensor:
    return t.float().amax(0).to(t.dtype)


def _lane_loop_body(code: str) -> str:
    """Return the source of the row lane loop (``for lane_0 in range(N)``)."""
    lines = code.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("for lane_0 in")
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return "\n".join(body)


@onlyBackends(["cute"])
class TestCuteLaneLoopReduceTwoPass(TestCase):
    def _check(
        self,
        kernel: object,
        ref: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        **config: object,
    ) -> str:
        code, out = code_and_output(kernel, (x,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out, ref(x), atol=2e-2, rtol=2e-2)
        # A second, different input must produce a different (correct) output:
        # kernels allocate their own result, so a lowering that skipped the
        # combine could otherwise pass on a stale buffer.
        y = torch.randn_like(x) * 3
        _, out_y = code_and_output(kernel, (y,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out_y, ref(y), atol=5e-2, rtol=2e-2)
        self.assertFalse(torch.allclose(out.float(), out_y.float()))
        return code

    def test_cross_warp_group_reduces_once_per_row_tile(self) -> None:
        """Row threads above the column threads: pre=8, span=128 (4 warps).

        The lane loop must only accumulate; the two-stage shared reduction
        runs once after it (per row tile), not once per element.
        """
        x = torch.randn(2048, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_sum,
            _col_sum,
            x,
            block_sizes=[2048, 16],
            num_threads=[16, 8],
            cute_vector_widths=[1, 2],
        )
        self.assertEqual(code.count("_cute_grouped_reduce_shared_two_stage("), 1)
        body = _lane_loop_body(code)
        self.assertNotIn("_cute_grouped_reduce", body)
        self.assertNotIn("warp_reduction", body)
        self.assertIn("_lane_acc", body)
        self.assertIn("pre=8, group_span=128, group_count=1", code)

    def test_single_warp_group_reduces_once_per_row_tile(self) -> None:
        """Row threads above the column threads inside one warp: pre=8,
        span=32 -> a single grouped warp shuffle after the lane loop."""
        x = torch.randn(1024, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_sum,
            _col_sum,
            x,
            block_sizes=[1024, 16],
            num_threads=[4, 8],
            cute_vector_widths=[1, 2],
        )
        self.assertEqual(code.count("_cute_grouped_reduce_warp("), 1)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)
        self.assertNotIn("_cute_grouped_reduce", _lane_loop_body(code))
        self.assertIn("pre=8, group_span=32", code)

    def test_bottom_axis_warp_reduces_once_per_row_tile(self) -> None:
        """Row threads at the bottom of the thread index (pre=1): a plain
        32-lane warp reduction once after the lane loop, even without the
        resident reduction sequence."""
        x = torch.randn(2048, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_sum,
            _col_sum,
            x,
            block_sizes=[2048, 4],
            num_threads=[32, 4],
            cute_reduction_sequence="scalar",
            cute_lane_layouts=["strided", "blocked"],
        )
        self.assertEqual(code.count("warp_reduction_sum("), 1)
        self.assertNotIn("_cute_grouped_reduce", code)
        self.assertNotIn("warp_reduction", _lane_loop_body(code))

    def test_max_reduction_cross_warp_group(self) -> None:
        x = torch.randn(2048, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_max,
            _col_max,
            x,
            block_sizes=[2048, 16],
            num_threads=[16, 8],
            cute_vector_widths=[1, 2],
        )
        self.assertEqual(code.count("_cute_grouped_reduce_shared_two_stage("), 1)
        self.assertNotIn("_cute_grouped_reduce", _lane_loop_body(code))


@onlyBackends(["cute"])
def test_owned_marker_restore_when_two_pass_split_is_unsafe() -> None:
    """jagged_layer_norm at 256 threads x 2 lanes: the lane loop carries extra
    cross-lane sums, so the two-pass split is refused and the owned marker is
    restored with its own running accumulator (finalized every lane)."""
    from examples.jagged_layer_norm import jagged_layer_norm_kernel
    from examples.jagged_layer_norm import reference_jagged_layer_norm_pytorch

    torch.manual_seed(0)
    lengths = torch.randint(1, 65, (32,), device=DEVICE)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=DEVICE), torch.cumsum(lengths, 0)]
    )
    x = torch.randn(int(x_offsets[-1]), 512, dtype=torch.float32, device=DEVICE)
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    code, out = code_and_output(
        kernel,
        (x, x_offsets, 1e-6),
        block_sizes=[1, 512, 32, 512, 32, 512, 32],
        cute_lane_layouts=["strided"] * 7,
        cute_vector_widths=[1] * 7,
        num_threads=[1, 256, 1, 256, 1, 256, 1],
    )
    assert "_lane_acc = " in code
    torch.testing.assert_close(
        out,
        reference_jagged_layer_norm_pytorch(x, x_offsets, 1e-6),
        rtol=1e-4,
        atol=1e-4,
    )
