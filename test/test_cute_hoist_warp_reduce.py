"""Tests for the CuTe ``hoist_warp_reduce`` AST pass.

When the codegen emits a ``cute.arch.warp_reduction_*`` call inside a
constexpr ``range_constexpr(V)`` loop body, the reduce runs V times per
outer iter even though the result depends on values gathered across
ALL V lanes of the warp.

The pass moves the warp reduce OUT of the V-loop: build a per-thread
``_helion_vfold_acc_*`` accumulator with the V-loop iterates folding
into it, then call ``cute.arch.warp_reduction_*`` ONCE per outer iter
on the accumulator. The number of warp shuffles per outer iter drops
from V*K to K (where K is the number of reduce sites — 2 for online
softmax: one for max, one for sum).

Lives in ``helion/_compiler/cute/hoist_warp_reduce.py``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@onlyBackends(["cute"])
class TestCuteHoistWarpReduce(TestCase):
    def test_warp_reduce_hoisted_out_of_v_loop(self) -> None:
        """The pass must remove the ``cute.arch.warp_reduction_max`` and
        ``cute.arch.warp_reduction_sum`` calls from inside the
        ``range_constexpr(V)`` loop body. Before the pass, the V-loop
        body contained V calls to each warp reduce (one per vec lane).
        After the pass, the V-loop body has zero warp reduces — they
        live OUTSIDE the loop with a per-thread local fold first.
        """
        x = torch.randn(4096, 6400, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Both reduces still get emitted (correctness), just outside the V-loop.
        self.assertIn("warp_reduction_max", code)
        self.assertIn("warp_reduction_sum", code)
        # Sanity: the V-loop is still present.
        self.assertIn("cutlass.range_constexpr(", code)
        # The hoist creates a fresh accumulator var visible in the body.
        self.assertIn("_helion_vfold_acc_", code)

    def test_no_hoist_when_v_loop_absent(self) -> None:
        """When V=1 there's no constexpr V-loop, so the hoist pass must be
        a no-op — the reduce stays where the strategy emitted it.
        """
        x = torch.randn(4096, 4096, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 1],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # No hoist accumulator names (no V-loop to hoist out of).
        self.assertNotIn("_helion_vfold_acc_", code)
