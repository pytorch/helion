"""CuTe reductions over unsigned integer (uint8, uint32) values.

CuTe DSL 4.7 allocates a shared-memory buffer of an unsigned integer type as
the signed integer of its width and reads it back as such, so the
cross-warp, tree and grouped reductions joined an Int8 partial with their
Uint8 identity and failed to compile (``TYPE_UNSTABLE_JOIN``) for amax/amin;
read back unconverted, a value with the top bit set would also compare as
negative.  ``reduce_helpers._smem_read`` converts every shared-memory read to
the buffer's intended type.  The values here have the top bit set.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _row_amax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].amax(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_amin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].amin(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.int64, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_argmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.int64, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].argmax(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _last_amax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), x.size(1)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :, :].amax(-1)
    return out


_ROW_KERNELS = (
    (_row_amax, lambda x: x.amax(-1)),
    (_row_amin, lambda x: x.amin(-1)),
    (_row_sum, lambda x: x.sum(-1)),
    (_row_argmax, lambda x: x.argmax(-1)),
)


def _unsigned(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    if dtype is torch.uint8:
        values = torch.randint(0, 256, shape, dtype=torch.int64)
    else:
        # Odd values up to 2**32 - 1, half of them with the top bit set.
        values = torch.randint(0, 2**31, shape, dtype=torch.int64) * 2 + 1
    return values.to(dtype).to(DEVICE)


@onlyBackends(["cute"])
class TestCuteUnsignedReductions(TestCase):
    def _check_rows(self, shape: tuple[int, int], config: dict[str, object]) -> None:
        for dtype in (torch.uint8, torch.uint32):
            x = _unsigned(shape, dtype)
            # torch's uint32 reductions are CPU-only; compare in int64.
            wide = x.cpu().to(torch.int64)
            for kernel, reference in _ROW_KERNELS:
                with self.subTest(dtype=dtype, kernel=kernel.name):
                    _code, out = code_and_output(kernel, (x,), **config)
                    torch.testing.assert_close(
                        out.cpu().to(torch.int64), reference(wide), atol=0, rtol=0
                    )

    def test_warp_reduction(self) -> None:
        self._check_rows((64, 32), {"block_sizes": [1]})

    def test_cross_warp_reduction(self) -> None:
        self._check_rows((16, 256), {"block_sizes": [1]})

    def test_rows_sharing_the_launch(self) -> None:
        # Eight 48-wide rows per CTA: grouped two-stage shared reduce.
        self._check_rows((64, 48), {"block_sizes": [8]})

    def test_rolled_and_synthetic_lane_reductions(self) -> None:
        self._check_rows((16, 512), {"block_sizes": [1], "reduction_loops": [64]})
        self._check_rows((4, 4096), {"block_sizes": [1]})

    def test_reduction_above_a_sibling_axis(self) -> None:
        for dtype in (torch.uint8, torch.uint32):
            for shape, block_size in (((32, 8, 5), 4), ((8, 16, 64), 1)):
                x = _unsigned(shape, dtype)
                with self.subTest(dtype=dtype, shape=shape):
                    code, out = code_and_output(
                        _last_amax, (x,), block_sizes=[block_size]
                    )
                    torch.testing.assert_close(
                        out.cpu().to(torch.int64),
                        x.cpu().to(torch.int64).amax(-1),
                        atol=0,
                        rtol=0,
                    )
                    self.assertIn("_cute_grouped_reduce", code)
