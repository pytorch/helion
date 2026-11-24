"""16-bit matmuls of whole rows whose product is stored as it is, in the form
helion.inductor writes for a K in one block (the output an argument), and the
same matmuls read by a pointwise rounding first."""

from __future__ import annotations

import torch

import helion.language as hl


def mm_stored(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([a.size(0), b.size(1)]):
        out[tile_y, tile_x] = torch.mm(a[tile_y, :], b[:, tile_x])


def mm_read(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([a.size(0), b.size(1)]):
        product = torch.mm(a[tile_y, :], b[:, tile_x]).to(torch.float32)
        out[tile_y, tile_x] = product.to(out.dtype)


def bmm_stored(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([a.size(0), a.size(1), b.size(2)]):
        out[tile_z, tile_y, tile_x] = torch.bmm(
            a[tile_z, tile_y, :], b[tile_z, :, tile_x]
        )


def bmm_read(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([a.size(0), a.size(1), b.size(2)]):
        product = torch.bmm(a[tile_z, tile_y, :], b[tile_z, :, tile_x])
        out[tile_z, tile_y, tile_x] = product.to(torch.float32).to(out.dtype)
