"""Tests for the CuTe ``merge_sibling_v_loops`` AST pass.

After ``hoist_warp_reduce`` runs, two-pass online softmax still has TWO
``range_constexpr(V)`` loops per outer iter — one for max, one for sum.
Both loops bitcast the SAME ``_tile_unroll_vec_*`` hoist var to extract
the per-lane scalar.

The pass introduces a small ``cute.make_fragment(V, cutlass.Float32)``
cache. V-loop 1 stores ``Float32(values)`` into the cache once per lane;
V-loop 2 reads the cached fp32 value back instead of re-running the
``Uint16 -> Float16`` bitcast chain. Eliminates the redundant per-lane
bitcast+cast pair.

A second peephole removes
``A = Float<N>(warp_reduction_*(...)) ; B = Float32(A)`` round-trips
left over after ``hoist_warp_reduce`` promoted the accumulator to fp32
— the Float<N> wrap is dead in that situation.

Lives in ``helion/_compiler/cute/merge_sibling_v_loops.py``.
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
class TestCuteMergeSiblingVLoops(TestCase):
    def test_merge_fires_on_two_v_loop_pattern(self) -> None:
        """The pass must fire on the canonical two-V-loop shape and emit
        a ``_helion_vmerge_cache_*`` fragment populated in V-loop 1 and
        read in V-loop 2.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Cache fragment allocated.
        self.assertIn("_helion_vmerge_cache_", code)
        # Cache write in V-loop 1 (writes Float32(values) at the
        # vec_lane index).
        self.assertIn("_helion_vmerge_cache_0[vec_lane_1]", code)
        # The cache is allocated as Float32 (promotes from fp16 so V-loop
        # 2 doesn't need the redundant Float32 cast).
        self.assertIn(
            "_helion_vmerge_cache_0 = cute.make_fragment(4, cutlass.Float32)",
            code,
        )

    def test_cast_elision_on_warp_reduction(self) -> None:
        """The double-cast peephole must collapse
        ``A = Float16(warp_reduction(...)); B = Float32(A)`` into
        ``A = warp_reduction(...); B = A``. The inner Float16 wrap on
        the max-reduce becomes dead after ``hoist_warp_reduce`` promoted
        the accumulator to fp32.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The original pattern ``local_amax = Float16(warp_reduction_max(...))``
        # must have been elided to just ``local_amax = warp_reduction_max(...)``.
        self.assertNotIn(
            "local_amax = cutlass.Float16(cute.arch.warp_reduction_max",
            code,
        )
        self.assertIn(
            "local_amax = cute.arch.warp_reduction_max",
            code,
        )

    def test_no_merge_when_v_loop_absent(self) -> None:
        """When V=1 there's no constexpr V-loop, so the merge pass must
        be a no-op — no cache fragment should be emitted.
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
        self.assertNotIn("_helion_vmerge_cache_", code)
