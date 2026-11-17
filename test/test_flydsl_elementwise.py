"""Unit tests for the flydsl backend's elementwise (non-reduction) path.

Covers the minimal flydsl backend: grid mapping, per-thread vector load/store,
elementwise math, and column-tail masking -- no whole-row reductions.

flydsl is AMD/ROCm-only and experimental, so the whole module is skipped when
it is not importable.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

pytest.importorskip("flydsl")


@helion.kernel(backend="flydsl")
def elementwise_double(x: torch.Tensor) -> torch.Tensor:
    # Pure elementwise map: no reduction (grid mapping + load/store/mul only).
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] * 2.0
    return out


@helion.kernel(backend="flydsl")
def elementwise_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Elementwise torch.minimum (exercises the minimum op override, which has
    # no native flydsl Vector method and uses -max(-a,-b)).
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = torch.minimum(x[tile_m, :], y[tile_m, :])
    return out


def _add_grid_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Both dims tiled by the top-level loop: FlyDSL maps grid tiles to rows only.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _add_grid_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A 1-D tensor tiled by the top-level loop.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def _add_grid_cols(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # The top-level loop tiles the second dim; the first is a whole ``:``.
    out = torch.empty_like(x)
    for tile_n in hl.tile(x.size(1)):
        out[:, tile_n] = x[:, tile_n] + y[:, tile_n]
    return out


def _shift_cols(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A column index computed from the top-level tile.
    out = torch.full_like(x, float("nan"))
    for tile_n in hl.tile(x.size(1) - 1):
        out[:, tile_n.index + 1] = x[:, tile_n.index + 1] + y[:, tile_n.index + 1]
    return out


def _shift_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A row index computed from the top-level tile.
    out = torch.full_like(x, float("nan"))
    for tile_m in hl.tile(x.size(0) - 1):
        out[tile_m.index + 1, :] = x[tile_m.index + 1, :] + y[tile_m.index + 1, :]
    return out


def _pick_col(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A constant column index.
    out = x.new_empty(x.size(0))
    for tile_m in hl.tile(x.size(0)):
        out[tile_m] = x[tile_m, 3] + y[tile_m, 3]
    return out


def _inner_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # An inner tile, not the top-level one, indexes the rows of y.
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m, x.size(1)], dtype=torch.float32)
        for tile_k in hl.tile(y.size(0)):
            acc = acc + y[tile_k, :].to(torch.float32).sum(0)[None, :]
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) + acc).to(x.dtype)
    return out


def _masked_load(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            keep = (tile_n.index < 100)[None, :]
            out[tile_m, tile_n] = hl.load(x, [tile_m, tile_n], extra_mask=keep)
    return out


def _scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # A per-row 1-D operand indexed by the row tile.
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] * s[tile_m][:, None]
    return out


class TestFlydslImports(TestCase):
    # No GPU or HELION_BACKEND needed: runs wherever flydsl is installed.
    def test_library_imports_resolve(self) -> None:
        # Every import line emitted into generated code must resolve against
        # the installed flydsl, so a flydsl module rename fails here first.
        from helion._compiler.flydsl.backend import FlyDSLBackend

        for name, stmt in FlyDSLBackend().library_imports.items():
            with self.subTest(name=name):
                exec(stmt, {})


@onlyBackends(["flydsl"])
class TestFlydslElementwise(TestCase):
    def test_unsupported_tensor_access_rejected(self) -> None:
        # All but _scale_rows used to run and silently return wrong values.
        x2 = torch.randn(64, 512, device=DEVICE, dtype=torch.float16)
        x1 = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        s = torch.randn(64, device=DEVICE, dtype=torch.float16)
        cases = (
            (_add_grid_2d, (x2, x2), "more than one dimension"),
            (_add_grid_cols, (x2, x2), "column dimension"),
            (_add_grid_1d, (x1, x1), "1-D tensor"),
            (_scale_rows, (x2, s), "1-D tensor"),
            (_shift_cols, (x2, x2), "tensor index"),
            (_shift_rows, (x2, x2), "tensor index"),
            (_pick_col, (x2, x2), "constant index 3"),
            (_inner_row, (x2, x2[:32]), "row dimension"),
            (_masked_load, (x2, x2), "extra_mask"),
        )
        for fn, args, message in cases:
            for effort in ("none", "quick"):
                with self.subTest(kernel=fn.__name__, autotune=effort):
                    kernel = helion.kernel(backend="flydsl", autotune_effort=effort)(fn)
                    with self.assertRaisesRegex(exc.BackendUnsupported, message):
                        kernel(*args)

    def test_elementwise_map_no_reduction(self) -> None:
        # Non-reduction path: grid mapping + elementwise + load/store, no fold.
        for bm, dt in ((1, torch.float16), (4, torch.float16), (1, torch.float32)):
            x = torch.randn(16, 512, device=DEVICE, dtype=dt)
            _, out = code_and_output(elementwise_double, (x,), block_sizes=[bm])
            torch.testing.assert_close(out, x * 2.0)

    def test_generated_code_uses_buffer_ops(self) -> None:
        # Codegen-only assertion (no kernel run): the elementwise buffer path
        # must emit flydsl's vectorized load/store ops, not a scalar fallback.
        x = torch.randn(16, 512, device=DEVICE, dtype=torch.float16)
        bound = elementwise_double.bind((x,))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("make_buffer_tensor", code)
        # No reduction, so finalize_device_body emits no warp-reduce helpers.
        self.assertNotIn("_flydsl_wsum", code)
        self.assertIn("copy_atom_call", code)

    def test_elementwise_map_column_tail(self) -> None:
        # Non-reduction path with N not a multiple of the vector width.
        x = torch.randn(8, 300, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(elementwise_double, (x,), block_sizes=[1])
        torch.testing.assert_close(out, x * 2.0)

    def test_elementwise_single_wavefront_row(self) -> None:
        # Whole-row ``:`` that fits in one wavefront (N <= 64): the column loop is
        # degenerate (single pass), so the buffer setup must emit inline in the
        # live body rather than into the never-emitted loop prefix. Regression for
        # the N<=64 dangling-reference / empty-launcher-arg codegen bug.
        for n in (16, 32, 64):
            for dt in (torch.float16, torch.float32):
                x = torch.randn(8, n, device=DEVICE, dtype=dt)
                _, out = code_and_output(elementwise_double, (x,), block_sizes=[1])
                torch.testing.assert_close(out, x * 2.0)

    def test_elementwise_minimum(self) -> None:
        # torch.minimum override (no native flydsl min method; -max(-a,-b)).
        x = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(elementwise_min, (x, y), block_sizes=[1])
        torch.testing.assert_close(out, torch.minimum(x, y))

    def test_autotune_elementwise(self) -> None:
        # The flydsl autotuner enumerates the bm (rows/block) knob and
        # FiniteSearch-picks a valid, correct config in-process (no subprocess
        # benchmark workers, which the flydsl JIT cannot survive).
        x = torch.randn(64, 512, device=DEVICE, dtype=torch.float16)
        bk = elementwise_double.bind((x,))
        cfg = bk.autotune((x,), force=True)
        self.assertIn("block_sizes", cfg.config)
        out = bk.compile_config(cfg)(x)
        torch.testing.assert_close(out, x * 2.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
