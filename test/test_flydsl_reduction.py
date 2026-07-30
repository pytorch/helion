"""Unit tests for the flydsl backend's whole-row looped reduction.

Covers the work landed in the runtime-scf.for rewrite and the cross-wavefront
(W>1) + fp16 V=8 autotune knob:

* the reduction loop lowers to a runtime ``range()`` (scf.for), NOT the
  compile-time-unrolled ``range_constexpr``;
* W wavefronts per row: ``thread_count = chunk // V`` so ``W = thread_count //
  64`` and V (the shared ``cute_vector_widths`` knob) are independent;
* fp16/bf16 V=8 (128-bit BufferCopy); fp32 V=8 is cleanly rejected;
* column-tail masking with W>1 (N not a multiple of the chunk);
* multi-row blocks (bm>1) and the persistent (non-looped) fallback still work.

Also covers adjacent flydsl backend paths so the whole staged backend has
coverage: the non-reduction elementwise map, explicit ``hl.tile(n)``
reductions, sum/max/min folds, and the elementwise ``torch.minimum`` override.

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
def elementwise_double(x: torch.Tensor) -> torch.Tensor:
    # Pure elementwise map: no reduction (grid mapping + load/store/mul only).
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] * 2.0
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
    # Whole-row min reduction (exercises the reduce_ops ``min`` fold path).
    m, _n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amin(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="flydsl")
def elementwise_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Elementwise torch.minimum (exercises the minimum op override, which has
    # no native flydsl Vector method and uses -max(-a,-b)).
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = torch.minimum(x[tile_m, :], y[tile_m, :])
    return out


def ref_rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    h = x.float()
    v = h.pow(2).mean(-1, keepdim=True)
    return (w * (h * torch.rsqrt(v + eps))).to(x.dtype)


def ref_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _tol(dt: torch.dtype) -> dict[str, float]:
    return (
        {"rtol": 1e-2, "atol": 1e-2}
        if dt == torch.float16
        else {"rtol": 1e-4, "atol": 1e-4}
    )


class TestFlydslReduction(TestCase):
    def _rms(self, m: int, n: int, dt: torch.dtype, **cfg: object) -> str:
        x = torch.randn(m, n, device=DEVICE, dtype=dt)
        w = torch.randn(n, device=DEVICE, dtype=dt)
        code, (out, _) = code_and_output(rms_norm_fwd, (x, w, 1e-5), **cfg)
        torch.testing.assert_close(out.float(), ref_rms(x, w).float(), **_tol(dt))
        return code

    def _softmax(self, m: int, n: int, dt: torch.dtype, **cfg: object) -> str:
        x = torch.randn(m, n, device=DEVICE, dtype=dt)
        code, out = code_and_output(softmax_decomposed, (x,), **cfg)
        torch.testing.assert_close(out.float(), ref_softmax(x).float(), **_tol(dt))
        return code

    def test_reduction_lowers_to_runtime_scf_for(self) -> None:
        # The reduction loop must be a runtime range() (scf.for), not the
        # compile-time-unrolled range_constexpr.
        code = self._rms(8, 1024, torch.float16, block_sizes=[1], reduction_loops=[64])
        self.assertIn("range(0, 1024", code)
        self.assertNotIn("range_constexpr(0, 1024", code)

    def test_cross_wavefront_thread_count(self) -> None:
        # W = (chunk // V) // 64 -> num_threads = 64*W. V pinned to 4.
        for w, chunk in ((1, 256), (2, 512), (4, 1024), (8, 2048)):
            code = self._rms(
                8,
                16384,
                torch.float16,
                block_sizes=[1],
                reduction_loops=[chunk],
                cute_vector_widths=[4],
            )
            self.assertIn(f"_num_threads={64 * w}", code)

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

    def test_multi_row_block(self) -> None:
        for bm in (2, 4):
            self._rms(16, 512, torch.float16, block_sizes=[bm], reduction_loops=[256])

    def test_persistent_reduction(self) -> None:
        self._rms(8, 256, torch.float16, block_sizes=[1], reduction_loops=[None])

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

    def test_elementwise_map_no_reduction(self) -> None:
        # Non-reduction path: grid mapping + elementwise + load/store, no fold.
        for bm, dt in ((1, torch.float16), (4, torch.float16), (1, torch.float32)):
            x = torch.randn(16, 512, device=DEVICE, dtype=dt)
            _, out = code_and_output(elementwise_double, (x,), block_sizes=[bm])
            torch.testing.assert_close(out, x * 2.0)

    def test_elementwise_map_column_tail(self) -> None:
        # Non-reduction path with N not a multiple of the vector width.
        x = torch.randn(8, 300, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(elementwise_double, (x,), block_sizes=[1])
        torch.testing.assert_close(out, x * 2.0)

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

    def test_min_reduction(self) -> None:
        # Whole-row min fold (reduce_ops "min"; sum/max already covered).
        x = torch.randn(8, 4096, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(row_min, (x,), block_sizes=[1], reduction_loops=[256])
        torch.testing.assert_close(out.float(), x.float().amin(-1))

    def test_elementwise_minimum(self) -> None:
        # torch.minimum override (no native flydsl min method; -max(-a,-b)).
        x = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(elementwise_min, (x, y), block_sizes=[1])
        torch.testing.assert_close(out, torch.minimum(x, y))

    def test_autotune_selects_valid_config(self) -> None:
        # The flydsl autotune enumerates (block_sizes, reduction_loops,
        # cute_vector_widths) and FiniteSearch-picks a valid, correct config.
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
        # autotune() on a kernel with an explicit hl.tile(n) inner reduction now
        # returns a valid, correct config (per-element tail masking + guarded
        # stores make it safe; it previously raised BackendUnsupported).
        x = torch.randn(8, 500, device=DEVICE, dtype=torch.float32)
        bk = tiled_sum.bind((x,))
        cfg = bk.autotune((x,), force=True)
        self.assertIn("block_sizes", cfg.config)
        out = bk.compile_config(cfg)(x)
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    import unittest

    unittest.main()
