"""Statements beside a collective-MMA matmul that read loads or other tensors.

At the default ``num_threads`` the collective path suppresses the root lane
loops, so the statements after the matmul run once per thread.  A load uniform
across the tile (``w[0]``, ``w[tile.id]``) is lowered and reads its element,
even though the root graph's first load shares the FX node name ``load`` with
the K loop's operand load.  A statement that needs the tile's per-element
coordinates (``out2[tile_m, tile_n] = x[tile_m, tile_n]``) is refused with
``BackendUnsupported`` rather than failing inside the CuTe DSL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

_KERNEL = {"backend": "cute", "static_shapes": True, "autotune_effort": "none"}


@helion.kernel(**_KERNEL)
def _scalar_store(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        side[tile_m.id, tile_n.id] = w[1] * w[2] + 1.0
    return out


@helion.kernel(**_KERNEL)
def _scalar_atomic(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        hl.atomic_add(side, [tile_n.id], w[tile_m.id])
    return out


@helion.kernel(**_KERNEL)
def _tile_id_loads(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        side[tile_m.id, tile_n.id] = w[tile_m.id] * 2.0 + hl.load(w, [tile_n.id])
    return out


@helion.kernel(**_KERNEL)
def _tensor_copy(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        side[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(**_KERNEL)
def _tensor_loaded_before(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        early = x[tile_m, tile_n]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        side[tile_m, tile_n] = early + 1.0
    return out


@helion.kernel(**_KERNEL)
def _row_store(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        side[tile_m, 0] = x[tile_m, 0] * 2.0
    return out


def _config(compute: str, block_sizes: list[int]) -> helion.Config:
    # No ``num_threads``, as in the default config: the collective path then
    # suppresses the root lane loops.
    return helion.Config(
        block_sizes=block_sizes,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_stages=2,
    )


_SIDE: dict[str, tuple[object, Callable[..., torch.Tensor]]] = {
    "scalar_store": (
        _scalar_store,
        lambda w, tiles_m, tiles_n: torch.full(
            (tiles_m, tiles_n), float(w[1] * w[2] + 1.0), device=w.device
        ),
    ),
    "scalar_atomic": (
        _scalar_atomic,
        lambda w, tiles_m, tiles_n: w[:tiles_m].sum().expand(tiles_n),
    ),
    "tile_id_loads": (
        _tile_id_loads,
        lambda w, tiles_m, tiles_n: w[:tiles_m, None] * 2.0 + w[None, :tiles_n],
    ),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("kernel_name", sorted(_SIDE))
@skipUnlessBackends(["cute"])
def test_uniform_loads_beside_collective_matmul(compute: str, kernel_name: str) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    kernel, side_reference = _SIDE[kernel_name]
    torch.manual_seed(0)
    w = torch.randn(16, device=DEVICE)
    # A full grid of tiles, and partial edge tiles in M, N and K.
    for (m, n, k), block_sizes in (
        ((256, 256, 512), [64, 64, 32]),
        ((200, 136, 96), [128, 64, 32]),
    ):
        a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
        tiles_m = -(-m // block_sizes[0])
        tiles_n = -(-n // block_sizes[1])
        expected_side = side_reference(w, tiles_m, tiles_n)
        side = torch.zeros(expected_side.shape, device=DEVICE)
        args = (a, b, w, side)
        bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
        config = _config(compute, block_sizes)
        assert "cute.gemm(" in bound.to_code(config)
        bound.set_config(config)
        out = bound(*args)
        torch.testing.assert_close(out, a.float() @ b.float(), atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(side, expected_side)


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("kernel", [_tensor_copy, _tensor_loaded_before, _row_store])
@skipUnlessBackends(["cute"])
def test_per_element_statements_beside_collective_matmul_are_refused(
    compute: str, kernel: object
) -> None:
    args = (
        torch.empty((256, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
        torch.empty((256, 256), dtype=torch.float32),
        torch.empty((256, 256), dtype=torch.float32),
    )
    with (
        _mock_cuda_unavailable(),
        _cpu_target(),
        pytest.raises(exc.BackendUnsupported, match="beside a collective matmul"),
    ):
        bound = kernel._bind_isolated(args)  # pyrefly: ignore [missing-attribute]
        bound.to_code(_config(compute, [64, 64, 32]))
