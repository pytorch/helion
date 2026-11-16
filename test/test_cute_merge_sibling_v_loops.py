"""Tests for the CuTe ``merge_sibling_v_loops`` AST pass.

After ``hoist_warp_reduce`` runs, two-pass online softmax still has TWO
``range_constexpr(V)`` loops per outer iter — one for max, one for sum.
Both loops bitcast the SAME ``_tile_unroll_vec_*`` hoist var to extract
the per-lane scalar.

The pass introduces a small ``cute.make_rmem_tensor(V, cutlass.Float32)``
cache. V-loop 1 stores ``Float32(values)`` into the cache once per lane;
V-loop 2 reads the cached fp32 value back instead of re-running the
``Uint16 -> Float16`` bitcast chain. Eliminates the redundant per-lane
bitcast+cast pair.

A second peephole removes repeated casts to the same dtype. Narrowing
the reduction result and then widening it must retain the rounding.

Lives in ``helion/_compiler/cute/merge_sibling_v_loops.py``.
"""

from __future__ import annotations

import pytest
import torch

from test._cute_kernels import _reduction_kernel

from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends

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
            "_helion_vmerge_cache_0 = cute.make_rmem_tensor(4, cutlass.Float32)",
            code,
        )

    def test_narrowing_cast_preserved_on_warp_reduction(self) -> None:
        """An accumulator promotion does not make a result cast redundant."""
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
        self.assertIn(
            "local_amax = cutlass.Float16(cute.arch.warp_reduction_max",
            code,
        )
        self.assertNotIn(
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
