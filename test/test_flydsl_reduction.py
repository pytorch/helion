"""Unit tests for the flydsl backend's whole-row reduction (W=1 and W>1).

Covers single-wavefront (W=1) and cross-wavefront (W>1) reductions plus the
fp16 V=8 (128-bit BufferCopy) knob on top of the minimal elementwise backend:
the reduction loop lowers to a runtime ``range()`` (scf.for), warp shuffle folds
(sum/max/min), cross-warp block reduce via shared memory when a row spans W
wavefronts (W = (chunk // V) // 64), whole-row (``:``) and explicit
``hl.tile(n)`` reductions, column-tail masking (W=1 and W>1), fp32, multi-row
blocks (bm>1), the persistent (non-looped) fallback, and autotuning over the
reduction knobs.

Constexpr register-caching lands in a follow-up PR.

flydsl is AMD/ROCm-only and experimental, so the whole module is skipped when
it is not importable.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

pytest.importorskip("flydsl")


@helion.kernel(backend="flydsl")
def rms_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty_like(x)
    inv_rms = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
        inv_rms[tile_m] = inv_rms_tile
    return out, inv_rms.reshape(-1, 1)


@helion.kernel(backend="flydsl")
def rms_norm_fwd_no_invrms(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
    return out


@helion.kernel(backend="flydsl")
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


@helion.kernel(backend="flydsl")
def tiled_sum(x: torch.Tensor) -> torch.Tensor:
    # Explicit inner-tile reduction (hl.tile(n)) rather than whole-row ``:``.
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc += x[tile_m, tile_n].to(torch.float32).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="flydsl")
def row_min(x: torch.Tensor) -> torch.Tensor:
    # Whole-row min reduction (exercises the reduction-strategy ``min`` fold).
    m, _n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amin(x[tile_m, :], dim=-1)
    return out


def _add_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    # A named combine fn is exactly what the hl.reduce primitive takes.
    return a + b


@helion.kernel(backend="flydsl")
def row_hl_reduce(x: torch.Tensor) -> torch.Tensor:
    # Explicit hl.reduce primitive (NOT the torch.* reduction path).
    m, _n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = hl.reduce(_add_combine, x[tile_m, :].to(torch.float32), dim=-1)
    return out


def ref_rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    h = x.float()
    v = h.pow(2).mean(-1, keepdim=True)
    return (w * (h * torch.rsqrt(v + eps))).to(x.dtype)


def ref_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _tol(dt: torch.dtype) -> tuple[float, float]:
    # (rtol, atol) passed as explicit kwargs; a **dict unpack trips type checkers
    # because assert_close also takes bool/str kwargs a dict[str, float] can't type.
    return (1e-2, 1e-2) if dt == torch.float16 else (1e-4, 1e-4)


@onlyBackends(["flydsl"])
class TestFlydslReduction(TestCase):
    def _rms(self, m: int, n: int, dt: torch.dtype, **cfg: object) -> str:
        x = torch.randn(m, n, device=DEVICE, dtype=dt)
        w = torch.randn(n, device=DEVICE, dtype=dt)
        code, (out, _) = code_and_output(rms_norm_fwd, (x, w, 1e-5), **cfg)
        _rtol, _atol = _tol(dt)
        torch.testing.assert_close(
            out.float(), ref_rms(x, w).float(), rtol=_rtol, atol=_atol
        )
        return code

    def _softmax(self, m: int, n: int, dt: torch.dtype, **cfg: object) -> str:
        x = torch.randn(m, n, device=DEVICE, dtype=dt)
        code, out = code_and_output(softmax_decomposed, (x,), **cfg)
        _rtol, _atol = _tol(dt)
        torch.testing.assert_close(
            out.float(), ref_softmax(x).float(), rtol=_rtol, atol=_atol
        )
        return code

    def test_reduction_lowers_to_runtime_scf_for(self) -> None:
        # The reduction loop must be a runtime range() (scf.for), not the
        # compile-time-unrolled range_constexpr.
        code = self._rms(8, 1024, torch.float16, block_sizes=[1], reduction_loops=[64])
        self.assertIn("range(0, 1024", code)
        self.assertNotIn("range_constexpr(0, 1024", code)

    def test_w1_single_warp_thread_count(self) -> None:
        # A 64-lane chunk (chunk // V == 64) keeps one warp per row: W=1.
        code = self._rms(8, 4096, torch.float16, block_sizes=[1], reduction_loops=[256])
        self.assertIn("_num_threads=64", code)

    def test_persistent_reduction(self) -> None:
        self._rms(8, 256, torch.float16, block_sizes=[1], reduction_loops=[None])

    def test_multi_row_block(self) -> None:
        for bm in (2, 4):
            self._rms(16, 512, torch.float16, block_sizes=[bm], reduction_loops=[256])

    def test_multi_row_block_no_smem_reduce(self) -> None:
        # bm>1 uses one warp per row (W=1), so the smem block reduce must NOT be
        # emitted. Verifies the _flydsl_warps_per_row=1 guard holds for bm>1.
        code = self._rms(
            16, 4096, torch.float16, block_sizes=[2], reduction_loops=[256]
        )
        self.assertNotIn("_flydsl_bsum", code)
        self.assertNotIn("_FlyDSLRedBuf", code)
        self.assertIn("_flydsl_wsum", code)  # warp-only path still emitted

    def test_softmax_w1(self) -> None:
        # Whole-row softmax with a 64-lane chunk (W=1).
        self._softmax(8, 512, torch.float16, block_sizes=[1], reduction_loops=[256])

    def test_min_reduction(self) -> None:
        # Whole-row min fold via the reduction-strategy path (sum/max covered too).
        x = torch.randn(8, 4096, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(row_min, (x,), block_sizes=[1], reduction_loops=[256])
        torch.testing.assert_close(out.float(), x.float().amin(-1))

    def test_hl_reduce_unsupported(self) -> None:
        # The explicit hl.reduce primitive is not yet supported on flydsl (its
        # per-thread vector is shuffled directly and stored to a scalar slot,
        # which crashes the AMDGPU backend). It must raise cleanly, not crash.
        # torch.sum/mean/amax/amin are the supported reduction path.
        x = torch.randn(8, 256, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.BackendUnsupported):
            code_and_output(row_hl_reduce, (x,), block_sizes=[1])

    def test_explicit_tile_reduction(self) -> None:
        # Explicit hl.tile(n) reduction (N a multiple of the inner tile).
        x = torch.randn(8, 1024, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(tiled_sum, (x,), block_sizes=[1, 256])
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-2, atol=1e-1)

    def test_explicit_tile_reduction_tail(self) -> None:
        # Explicit hl.tile(n) reduction with a tail (N % inner-tile != 0): the
        # per-element column mask on the load must drop out-of-range columns so
        # the reduction does not sum neighbour-row data.
        for n in (300, 700, 1000):
            x = torch.randn(8, n, device=DEVICE, dtype=torch.float32)
            _, out = code_and_output(tiled_sum, (x,), block_sizes=[1, 256])
            torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-3)

    def test_cross_wavefront_thread_count(self) -> None:
        # W = (chunk // V) // 64 -> num_threads = 64*W. Cover V=4 and V=8 so the
        # (chunk, V) -> W mapping is locked on both vector widths.
        cases = (
            (1, 256, 4),
            (2, 512, 4),
            (4, 1024, 4),
            (8, 2048, 4),
            (16, 4096, 4),  # W=16: max warps/row, _FlyDSLRedBuf _TOTAL=17
            (1, 512, 8),
            (2, 1024, 8),
            (4, 2048, 8),
            (8, 4096, 8),  # W=8 V=8: exercises fp16 128-bit copy at W>1
        )
        for w, chunk, v in cases:
            code = self._rms(
                8,
                16384,
                torch.float16,
                block_sizes=[1],
                reduction_loops=[chunk],
                cute_vector_widths=[v],
            )
            self.assertIn(f"_num_threads={64 * w}", code)

    def test_column_tail_w_gt_1(self) -> None:
        # N not a multiple of the chunk (64*W*V) across multiple chunks.
        for n in (1000, 6244):
            self._rms(
                8,
                n,
                torch.float16,
                block_sizes=[1],
                reduction_loops=[512],
                cute_vector_widths=[4],
            )
            self._softmax(
                8,
                n,
                torch.float16,
                block_sizes=[1],
                reduction_loops=[512],
                cute_vector_widths=[4],
            )

    def test_fp16_v8_uses_128bit_copy(self) -> None:
        code = self._rms(
            8,
            4096,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[512],
            cute_vector_widths=[8],
        )
        self.assertIn("BufferCopy128b", code)

    def test_fp16_v8_w_gt_1(self) -> None:
        # V=8 AND W>1 together: chunk=1024, V=8 -> W = 1024//8//64 = 2, so the
        # 128-bit copy and the cross-wave smem block reduce must BOTH engage
        # (the combination this PR exists for; the V=8 tests above are W=1).
        code = self._rms(
            8,
            16384,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[1024],
            cute_vector_widths=[8],
        )
        self.assertIn("_num_threads=128", code)  # W=2
        self.assertIn("BufferCopy128b", code)  # V=8
        self.assertIn("_flydsl_bsum", code)  # smem block reduce engaged

    def test_fp32_v8_rejected(self) -> None:
        # fp32 V=8 = 256-bit copy: unrepresentable, must raise cleanly.
        x = torch.randn(8, 4096, device=DEVICE, dtype=torch.float32)
        w = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.BackendUnsupported):
            code_and_output(
                rms_norm_fwd,
                (x, w, 1e-5),
                block_sizes=[1],
                reduction_loops=[512],
                cute_vector_widths=[8],
            )

    def test_fp32_reduction(self) -> None:
        self._rms(
            8,
            4096,
            torch.float32,
            block_sizes=[1],
            reduction_loops=[1024],
            cute_vector_widths=[4],
        )

    def test_softmax_cross_wavefront(self) -> None:
        for chunk in (512, 2048):
            self._softmax(
                8,
                16384,
                torch.float16,
                block_sizes=[1],
                reduction_loops=[chunk],
                cute_vector_widths=[4],
            )

    def test_softmax_fp16_v8(self) -> None:
        code = self._softmax(
            8,
            4096,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[512],
            cute_vector_widths=[8],
        )
        self.assertIn("BufferCopy128b", code)

    def test_autotune_selects_valid_config(self) -> None:
        # The flydsl autotune enumerates (block_sizes, reduction_loops) and
        # FiniteSearch-picks a valid, correct config.
        x = torch.randn(8, 4096, device=DEVICE, dtype=torch.float16)
        w = torch.randn(4096, device=DEVICE, dtype=torch.float16)
        bk = rms_norm_fwd.bind((x, w, 1e-5))
        cfg = bk.autotune((x, w, 1e-5), force=True)
        self.assertIn("block_sizes", cfg.config)
        out, _ = bk.compile_config(cfg)(x, w, 1e-5)
        torch.testing.assert_close(
            out.float(), ref_rms(x, w).float(), rtol=1e-2, atol=1e-2
        )

    def test_autotune_explicit_tile_reduction(self) -> None:
        # autotune() on a kernel with an explicit hl.tile(n) inner reduction
        # returns a valid, correct config (per-element tail masking + guarded
        # stores make it safe).
        x = torch.randn(8, 500, device=DEVICE, dtype=torch.float32)
        bk = tiled_sum.bind((x,))
        cfg = bk.autotune((x,), force=True)
        self.assertIn("block_sizes", cfg.config)
        out = bk.compile_config(cfg)(x)
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-3)

    def test_constexpr_range_emits_and_correct(self) -> None:
        # constexpr_range=True with a small tile count (N=1024, chunk=256 -> 4
        # tiles <= 16) unrolls the reduction loop into range_constexpr, enabling
        # in_local[] register caching (2-read HBM). Assert the unrolled form IS
        # emitted AND the result is still correct (checked inside _rms).
        code = self._rms(
            8,
            1024,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[256],
            constexpr_range=True,
        )
        self.assertIn("range_constexpr(0, 1024", code)
        self.assertNotIn("range(0, 1024", code)

    def test_constexpr_softmax_correct(self) -> None:
        # Whole-row softmax under constexpr_range (register-cached passes).
        code = self._softmax(
            8,
            1024,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[256],
            constexpr_range=True,
        )
        self.assertIn("range_constexpr(0, 1024", code)

    def test_constexpr_large_tile_count_falls_back(self) -> None:
        # constexpr_range=True but tile count > 16 (N=16384, chunk=256 -> 64
        # tiles) must fall back to the runtime scf.for (unrolling 64 tiles would
        # blow up VGPRs), so range_constexpr is NOT emitted.
        code = self._rms(
            8,
            16384,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[256],
            constexpr_range=True,
        )
        self.assertNotIn("range_constexpr(0, 16384", code)
        self.assertIn("range(0, 16384", code)

    def test_autotune_offers_constexpr(self) -> None:
        # autotune on a small-N rms_norm reaches the constexpr_range=True
        # candidates without crashing and returns a valid, correct config.
        x = torch.randn(8, 1024, device=DEVICE, dtype=torch.float16)
        w = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        bk = rms_norm_fwd.bind((x, w, 1e-5))
        cfg = bk.autotune((x, w, 1e-5), force=True)
        self.assertIn("block_sizes", cfg.config)
        out, _ = bk.compile_config(cfg)(x, w, 1e-5)
        torch.testing.assert_close(
            out.float(), ref_rms(x, w).float(), rtol=1e-2, atol=1e-2
        )

    def test_persistent_single_pass_emits_and_correct(self) -> None:
        # reduction_loops=[N] with constexpr_range=True: chunk equals the row
        # size, so the reduction loop has one iteration.  The in_local[] path
        # caches the whole row in registers across reduce+normalize — zero second
        # HBM read.  N=4096, V=8 -> thread_count = 4096/8 = 512 (W=8).
        # ReductionLoopSpec._normalize must NOT collapse chunk=N to None
        # (allow_wide_persistent_reduction=True on FlyDSL).
        code = self._rms(
            8,
            4096,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[4096],
            cute_vector_widths=[8],
            constexpr_range=True,
        )
        # Single-iteration range_constexpr — step == chunk == N.
        self.assertIn("range_constexpr(0, 4096, 4096)", code)
        # Must not fall back to runtime scf.for.
        self.assertNotIn("range(0, 4096", code)
        # 512 threads = N / V = 4096 / 8.
        self.assertIn("_num_threads=512", code)

    def test_persistent_single_pass_normalization_preserves_chunk(self) -> None:
        # Verify that ReductionLoopSpec._normalize does NOT collapse chunk=N to
        # None for FlyDSL (allow_wide_persistent_reduction=True).  If it did,
        # reduction_loops=[4096] would silently become persistent (None) and the
        # looped single-iteration path would never be reached.
        # We confirm normalization is correct by checking the generated code
        # uses range_constexpr (looped path) rather than a persistent no-loop.
        code = self._rms(
            8,
            4096,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[4096],
            cute_vector_widths=[8],
            constexpr_range=True,
        )
        # If normalization collapsed chunk=N to None, the kernel would be
        # persistent (no range_constexpr, 64 threads only).  Looped path wins.
        self.assertIn("range_constexpr(0, 4096, 4096)", code)
        self.assertNotIn("range(0, 4096", code)

    def test_nonpot_constexpr_range_correct(self) -> None:
        # N=20000 is non-power-of-two.  chunk=4096, V=8 → 5 tiles, W=8 warps.
        # Last tile (roffset=16384) covers columns 16384..19999 — only 3616 of
        # 4096 elements are valid.  rolled_col_pred masks the tail; full tiles
        # use const_expr(roffset + chunk <= N) to elide the predicate.
        # Correctness proves both the tail masking and the full-tile fast path.
        code = self._rms(
            8,
            20000,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[4096],
            cute_vector_widths=[8],
            constexpr_range=True,
        )
        self.assertIn("range_constexpr(0, 20000, 4096)", code)
        self.assertNotIn("range(0, 20000", code)
        # W=8: thread_count = 4096 / 8 = 512.
        self.assertIn("_num_threads=512", code)
        # Full tiles elide the select; only the tail tile has it.
        self.assertIn("const_expr(", code)

    def test_nonpot_tail_one_element(self) -> None:
        # N=4097: one element past a full chunk (chunk=4096, V=4, W=1).
        # Tile 0 (roffset=0): full, 4096 elements — no select().
        # Tile 1 (roffset=4096): 1 valid element, 1023 OOB — select() masks.
        # Verifies the extreme boundary case of the tail predicate.
        code = self._rms(
            8,
            4097,
            torch.float16,
            block_sizes=[1],
            reduction_loops=[4096],
            cute_vector_widths=[4],
            constexpr_range=True,
        )
        self.assertIn("range_constexpr(0, 4097, 4096)", code)
        self.assertNotIn("range(0, 4097", code)

    def test_oob_guard_admits_safe_overshoot(self) -> None:
        # The allocation-aware OOB guard (backend._add) must ADMIT chunks whose
        # last-tile wide read stays inside the tensor's hipMalloc'd segment.
        # These are the "pow2+1" shapes (N = k*chunk + 1) that the old
        # fixed-byte guard incorrectly rejected, forcing autotune onto a slow
        # fallback.  Explicitly compiling the fast config must succeed and
        # produce a correct result: the last tile's 1-valid-element tail is
        # masked by rolled_col_pred, so the OOB read is harmless.
        #
        # N=24577 = 6*4096+1, f16, M=4096: store_bytes ≈192MB, 2MiB slack.
        #   chunk=2048 V=4 → 13 tiles; last tile: 1 valid elem, 15 OOB.
        #   chunk=2048 V=8 → 13 tiles; last tile: 1 valid elem, 15 OOB (128-bit).
        # N=32769 = 8*4096+1, f16, M=4096: same safe-segment argument.
        # N=16385 = 2*8192+1, bf16, M=4096: different dtype (2B/elem same as f16),
        #   different N to prove the guard isn't special-cased to 24577/32769.
        # N=24577, M=256: small M → store_bytes=12MB, 2MiB segment,
        #   much smaller tensor; guard must use the real segment size.
        from helion.runtime import Config

        for M, N, chunk, v, dt in [
            (4096, 24577, 2048, 4, torch.float16),
            (4096, 24577, 2048, 8, torch.float16),
            (4096, 32769, 4096, 4, torch.float16),
            (4096, 32769, 4096, 8, torch.float16),
            (4096, 16385, 2048, 4, torch.bfloat16),
            (4096, 16385, 2048, 8, torch.bfloat16),
            (256, 24577, 2048, 4, torch.float16),
        ]:
            x = torch.randn(M, N, device=DEVICE, dtype=dt)
            w = torch.randn(N, device=DEVICE, dtype=dt)
            bk = rms_norm_fwd_no_invrms.bind((x, w, 1e-5))
            cfg = Config(
                block_sizes=[1],
                reduction_loops=[chunk],
                cute_vector_widths=[v],
                constexpr_range=True,
            )
            out = bk.compile_config(cfg)(x, w, 1e-5)
            torch.testing.assert_close(
                out.float(),
                ref_rms(x, w).float(),
                rtol=2e-2,
                atol=2e-2,
                msg=f"M={M} N={N} chunk={chunk} V={v} dt={dt}",
            )

    def test_oob_guard_rejects_zero_slack_hang_config(self) -> None:
        # The allocation-aware OOB guard must REJECT a chunk/V pair whose
        # last-tile wide read escapes the tensor's hipMalloc'd segment.
        # "Zero slack" means M*N*itemsize is an exact 2 MiB multiple, so the
        # segment ends exactly at the tensor — any overshoot page-faults.
        #
        # Two shapes, both zero-slack (verified: store_bytes % 2MiB == 0):
        #   4096×5120 f16: 4096*5120*2 = 40 MiB.  chunk=2048 V=8 → 8-elem OOB.
        #   8192×3072 f16: 8192*3072*2 = 48 MiB.  chunk=2048 V=8 → 1024-elem OOB.
        # Also check V=4 on 4096×5120: chunk=2048 V=4 → 4-elem OOB = 8 bytes;
        # still rejected because store_bytes+8 > seg_bytes (zero slack).
        # Proxy: the autotune candidate list must contain no config matching
        # the rejected (chunk, V) pair for each shape.
        import unittest.mock as mock

        from helion.autotuner import FiniteSearch

        # (M, N, reject_chunk, reject_v)
        cases = [
            (4096, 5120, 2048, 8),
            (4096, 5120, 2048, 4),
            (8192, 3072, 2048, 8),
        ]
        orig_init = FiniteSearch.__init__

        def _run_and_capture(bk: object, args: tuple[object, ...]) -> list[object]:
            captured: list[object] = []

            def _capture(self: object, *a: object, **kw: object) -> None:
                orig_init(self, *a, **kw)  # pyrefly: ignore[bad-argument-type]
                captured.extend(getattr(self, "configs", []) or [])

            with mock.patch.object(FiniteSearch, "__init__", _capture):
                bk.autotune(args, force=True)  # pyrefly: ignore[missing-attribute]
            return captured

        for M, N, rej_chunk, rej_v in cases:
            x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
            w = torch.randn(N, device=DEVICE, dtype=torch.float16)
            bk = rms_norm_fwd_no_invrms.bind((x, w, 1e-5))
            candidates = _run_and_capture(bk, (x, w, 1e-5))
            bad = [
                c
                for c in candidates
                if c.config.get("reduction_loops") == [rej_chunk]  # pyrefly: ignore[missing-attribute]
                and c.config.get("cute_vector_widths") == [rej_v]  # pyrefly: ignore[missing-attribute]
            ]
            self.assertEqual(
                bad,
                [],
                msg=f"Guard must exclude chunk={rej_chunk} V={rej_v} for {M}x{N} "
                f"(zero-slack hang shape); "
                f"found: {[dict(c.config) for c in bad]}",  # pyrefly: ignore[missing-attribute,bad-index]
            )

    def test_oob_guard_autotune_pow2plus1_selects_fast_config(self) -> None:
        # End-to-end: autotune for pow2+1 shapes must select a 1-row-per-block
        # config (block_sizes=[1]) with a large chunk (>=1024) and
        # constexpr_range=True — not the slow 256-block sub-warp fallback that
        # the old fixed-byte guard forced.  Correctness is also checked.
        #
        # N=24577 = 6*4096+1: the primary regression shape.
        # N=32769 = 8*4096+1: the second regression shape.
        # N=16385 = 2*8192+1, bf16: different dtype + different pow2+1 family.
        for M, N, dt in [
            (4096, 24577, torch.float16),
            (4096, 32769, torch.float16),
            (4096, 16385, torch.bfloat16),
        ]:
            x = torch.randn(M, N, device=DEVICE, dtype=dt)
            w = torch.randn(N, device=DEVICE, dtype=dt)
            bk = rms_norm_fwd_no_invrms.bind((x, w, 1e-5))
            cfg = bk.autotune((x, w, 1e-5), force=True)
            c = cfg.config
            self.assertEqual(
                c.get("block_sizes"), [1], msg=f"M={M} N={N}: expected bm=1 winner"
            )
            rl = c.get("reduction_loops") or []
            self.assertTrue(
                rl and rl[0] >= 1024,  # pyrefly: ignore[bad-index]
                msg=f"M={M} N={N}: expected chunk>=1024, got reduction_loops={rl}",
            )
            self.assertTrue(
                c.get("constexpr_range"),
                msg=f"M={M} N={N}: expected constexpr_range=True in fast winner",
            )
            out = bk.compile_config(cfg)(x, w, 1e-5)
            torch.testing.assert_close(
                out.float(),
                ref_rms(x, w).float(),
                rtol=2e-2,
                atol=2e-2,
                msg=f"M={M} N={N} dt={dt}",
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
