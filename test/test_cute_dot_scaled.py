"""The CuTe ``hl.dot_scaled`` fallback: dequantize each operand element, then dot.

Every element is multiplied by the e8m0 scale of its own 32-wide K group, read
from the scale tensor at the thread's row and K coordinates
(``cute/matmul_ops.py``); the scale tile's own load is dropped
(``cute/dot_scaled_scales.py``) together with its K-group axis.
"""

from __future__ import annotations

import torch

from test.test_dot_scaled import _reference_dot_scaled

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _scaled_e4m3(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = hl.dot_scaled(
            x[tile_m, :],
            x_scale[tile_m, :],
            "e4m3",
            y[:, tile_n],
            y_scale[tile_n, :],
            "e4m3",
            acc=acc,
        )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _scaled_k_tiles(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=64):
            acc = hl.dot_scaled(
                x[tile_m, tile_k],
                x_scale[tile_m, :],
                "e4m3",
                y[tile_k, tile_n],
                y_scale[tile_n, :],
                "e4m3",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _shared_scale_tile(
    x: torch.Tensor, scale: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        tile_scale = scale[tile_m, :]
        out[tile_m, tile_n] = hl.dot_scaled(
            x[tile_m, :], tile_scale, "e4m3", y[:, tile_n], tile_scale, "e4m3"
        )
    return out


def _inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = (torch.randn(m, k, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
    y = (torch.randn(k, n, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
    x_scale = torch.randint(118, 138, (m, k // 32), device=DEVICE, dtype=torch.uint8)
    y_scale = torch.randint(118, 138, (n, k // 32), device=DEVICE, dtype=torch.uint8)
    return x, x_scale, y, y_scale


@onlyBackends(["cute"])
@skipIfCudaCapabilityLessThan(
    (8, 9), reason="the e4m3 operand decode (cvt.rn.f16x2.e4m3x2) needs SM89+"
)
class TestCuteDotScaled(TestCase):
    def test_scale_tile_axis_gets_no_threads(self) -> None:
        # With M split over 4 threads and two lanes, the dropped K-group axis of
        # the scale tiles keeps no thread axis, so the launch shape covers every
        # row.
        args = _inputs(128, 128, 128)
        code, out = code_and_output(
            _scaled_e4m3,
            args,
            block_sizes=[8, 2],
            num_threads=[4, 0, 0, 0],
            cute_vector_widths=[2, 1, 1, 8],
        )
        torch.testing.assert_close(
            out, _reference_dot_scaled(*args), atol=1e-2, rtol=1e-3
        )
        # Only the per-element reads at K // 32 remain.
        self.assertEqual(code.count("cute.arch.load(x_scale.iterator"), 1)
        self.assertIn("// 32) * cutlass.Int32(x_scale.layout.stride[1])", code)

    def test_k_tiled_operands_are_refused(self) -> None:
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "full K slices of the same row tile"
        ):
            code_and_output(_scaled_k_tiles, _inputs(64, 64, 128), block_sizes=[32, 32])

    def test_non_power_of_two_k_groups_by_its_extent(self) -> None:
        # K = 96 runs on a 128-wide reduction dim; the K groups are 96 // groups
        # wide (not 128 // groups), and the padded lanes past K read no scale
        # (a 255 byte past the last group would make their zero operand NaN).
        torch.manual_seed(0)
        m = n = 64
        k = 96
        x = (torch.randn(m, k, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        y = (torch.randn(k, n, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        for groups in (2, 3, 4):
            x_scale = torch.full((m, groups + 1), 255, device=DEVICE, dtype=torch.uint8)
            y_scale = torch.full((n, groups + 1), 255, device=DEVICE, dtype=torch.uint8)
            x_scale[:, :groups] = torch.randint(
                118, 138, (m, groups), device=DEVICE, dtype=torch.uint8
            )
            y_scale[:, :groups] = torch.randint(
                118, 138, (n, groups), device=DEVICE, dtype=torch.uint8
            )
            args = (x, x_scale[:, :groups], y, y_scale[:, :groups])
            with self.subTest(groups=groups):
                code, out = code_and_output(_scaled_e4m3, args, block_sizes=[16, 16])
                torch.testing.assert_close(
                    out, _reference_dot_scaled(*args), atol=1e-2, rtol=1e-3
                )
                self.assertIn(f"// {k // groups})", code)

    def test_one_scale_tile_for_both_operands_is_refused(self) -> None:
        # The scale tile is rewired to its tensor for x; y keeps the tile load
        # (indexed by x's rows), which the fallback refuses instead of the
        # rewrite erasing a node still in use.
        x, x_scale, y, _y_scale = _inputs(64, 64, 128)
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "full K slices of the same row tile"
        ):
            code_and_output(_shared_scale_tile, (x, x_scale, y), block_sizes=[16, 16])
