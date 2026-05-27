"""Tests for the CuTe ``fuse_two_pass_loads`` AST pass.

When a kernel reads the same gmem tensor in two sequential inner-tile
loops over the same range, the pass detects the redundant load and
caches the first sweep's values in a small ``cute.make_fragment(...)``.
The second sweep then reads from the fragment instead of issuing a
second LDG, eliminating the duplicate HBM/L1 traffic.

The pass canonicalizes per-sweep variable names (``mask_<N>``,
``lane_base_<N>``, ``vec_lane_<N>``, ``_tile_unroll_vec_<N>_<M>``) so
the two sweeps' load expressions compare equal modulo the rename
suffix. Without canonicalization, ``vec_lane_1`` vs ``vec_lane_2``
would mis-key the fuser's match table and the cache would never fire.

Lives in ``helion/_compiler/cute/fuse_two_pass_loads.py``.
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
class TestCuteFuseTwoPassLoads(TestCase):
    def test_fuser_fires_with_vec_hoist(self) -> None:
        """When the vec hoist runs, the two sweeps' loads use names like
        ``lane_base_1`` vs ``lane_base_2``, ``vec_lane_1`` vs ``vec_lane_2``,
        and ``_tile_unroll_vec_1_0`` vs ``_tile_unroll_vec_1_1``. The
        alias map canonicalizes these so the fuser matches and emits one
        ``_fuse_cache_*`` fragment instead of re-reading from gmem.

        The cache_size cap is 64; for inner-tile trip=8 and lane_trip=1
        the cache_size=8 fits.
        """
        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
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
        self.assertIn("_fuse_cache_", code)
        # When the load-pipeline pass is OFF the consume sweep just
        # reads from cache so the kernel has 1 ``cute.arch.load`` total.
        # With pipelining ON the reduce sweep's load is hoisted into a
        # prologue + per-iter prefetch (2 load sites), so the total is
        # 2. The consume sweep emits NO load (cache hit) either way.
        load_count = code.count("cute.arch.load(")
        self.assertTrue(
            load_count in {1, 2},
            f"expected 1 or 2 cute.arch.load sites, got {load_count}",
        )
        # The consume sweep (second top-level ``for tile_offset`` loop)
        # must not contain a gmem load.
        consume_marker = "for tile_offset_2 in range"
        consume_start = code.rfind(consume_marker)
        self.assertGreater(consume_start, 0)
        self.assertNotIn(
            "cute.arch.load(",
            code[consume_start:],
            "consume sweep must read from _fuse_cache_, not gmem",
        )

    def test_fuser_skips_when_cache_size_too_large(self) -> None:
        """The fuser caps cache_size at 64 to avoid the register-pressure
        regression measured earlier when the per-thread fragment grew
        beyond the register budget. So for trip > 64 the consume sweep
        still loads from gmem.

        For (4096, 12672) with block_size 128, trip = 99, V = 4, so the
        cache_size would be 99 — fuser must bail.
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
        # Both sweeps load from gmem (cache fragment NOT allocated since
        # the fuser bails on cache_size > 64).
        self.assertNotIn("_fuse_cache_", code)
        # When the load-pipeline pass is OFF the kernel has exactly
        # 2 ``cute.arch.load`` calls (one per sweep).  With pipelining
        # ON each load site is hoisted into a prologue and a per-iter
        # prefetch, so the total is 4.  Both forms are correct; assert
        # at least 2.
        self.assertGreaterEqual(code.count("cute.arch.load("), 2)
        # Each sweep must contain at least one gmem load.
        first_sweep_marker = "for tile_offset_2 in range"
        first_sweep_start = code.find(first_sweep_marker)
        second_sweep_start = code.find(first_sweep_marker, first_sweep_start + 1)
        self.assertGreater(second_sweep_start, first_sweep_start)
        first_sweep = code[first_sweep_start:second_sweep_start]
        second_sweep = code[second_sweep_start:]
        self.assertIn("cute.arch.load(", first_sweep)
        self.assertIn("cute.arch.load(", second_sweep)
