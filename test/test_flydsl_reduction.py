"""Unit tests for the flydsl backend — elementwise and explicit-tile codegen.

Covers backend plumbing that does NOT require whole-row W=1 reductions:

* elementwise map (grid mapping + load/store/mul, no reduction fold);
* explicit ``hl.tile(n)`` reduction (user-tiled inner loop, not whole-row ``:``)
* ``torch.minimum`` override (no native flydsl Vector method; -max(-a,-b));
* autotune for explicit-tile reductions;
* OOB-guard rejection for bad chunk configs;
* fp32 V=8 rejection (256-bit copy unsupported).

Whole-row W=1 reductions (rms_norm, softmax, warp shuffles) are in PR1b.

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


class TestFlydslReduction(TestCase):
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

    def test_large_chunk_oob_rejected(self) -> None:
        # chunk=4096 with N=7680 and V=8 would OOB the divided buffer on the
        # last pass (max_div_idx=1023 >= n_div=960). The autotune _add() guard
        # rejects this config so it is never offered as a candidate. Verify
        # that autotune on N=7680 never selects a chunk that exceeds N//V.
        x = torch.randn(8, 7680, device=DEVICE, dtype=torch.float16)
        w = torch.randn(7680, device=DEVICE, dtype=torch.float16)
        bk = rms_norm_fwd.bind((x, w, 1e-5))
        cfg = bk.autotune((x, w, 1e-5), force=True)
        rl = cfg.config.get("reduction_loops") or []
        vw = cfg.config.get("cute_vector_widths") or []
        if rl and rl[0] is not None:
            chunk = int(rl[0])
            v = int(vw[0]) if vw else 4
            n_div = (7680 + v - 1) // v
            tc = chunk // v
            last_offset = ((7680 + chunk - 1) // chunk - 1) * chunk
            max_div_idx = last_offset // v + tc - 1
            self.assertLess(
                max_div_idx,
                n_div,
                f"autotune selected OOB config: chunk={chunk}, V={v}",
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
