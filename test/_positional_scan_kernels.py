"""Small positional scan regressions. Every kernel also contains a square value
(two dims on one canonical axis) so ordinary lowering cannot own the root."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def affine(a0, b0, a1, b1):
    # Composition of x -> a*x + b maps; associative, NOT commutative.
    return a0 * a1, b0 * a1 + b1


def _lower(c: int) -> torch.Tensor:
    idx = hl.arange(c)
    return (idx[:, None] >= idx[None, :]).to(torch.float32)


@helion.kernel(backend="cute", static_shapes=True)
def masked_cumsum(x: torch.Tensor) -> torch.Tensor:
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        y = x[tile, :, :].float() * _lower(c)[None, :, :]
        out[tile, :, :] = hl.cumsum(y, dim=2)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def masked_reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        y = x[tile, :, :].float() * _lower(c)[None, :, :]
        out[tile, :, :] = hl.cumsum(y, dim=1, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def affine_scan(a: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Tuple state, non-commutative combine; forward over dim 1, reverse over dim 2.
    b, c = a.size(0), hl.specialize(a.size(1))
    fa = torch.empty([b, c, c], dtype=torch.float32, device=a.device)
    rb = torch.empty([b, c, c], dtype=torch.float32, device=a.device)
    for tile in hl.tile(b, block_size=1):
        av = a[tile, :, :].float()
        xv = x[tile, :, :].float() * _lower(c)[None, :, :]
        p, q = hl.associative_scan(affine, (av, xv), dim=1)
        r, s = hl.associative_scan(affine, (av, xv), dim=2, reverse=True)
        fa[tile, :, :] = p + q
        rb[tile, :, :] = r - s
    return fa, rb


@helion.kernel(backend="cute", static_shapes=True)
def tail_reverse_cumsum(
    x: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Scan dim is a 16-wide tile over n=20: the last tile is partial, and host
    # storage continues past n (x is a slice view), so tail data is not zero.
    # The square mask store routes the root without a dot (no scan-export family).
    b, n = x.size(0), x.size(1)
    c = hl.specialize(k.size(1))
    out = torch.empty([b, n], dtype=torch.float32, device=x.device)
    masked = torch.empty(
        [b, (n + 15) // 16, c, c], dtype=torch.float32, device=k.device
    )
    for tile_b, tile_n in hl.tile([b, n], block_size=[1, 16]):
        out[tile_b, tile_n] = hl.cumsum(x[tile_b, tile_n].float(), dim=1, reverse=True)
        masked[tile_b, tile_n.id, :, :] = (
            k[tile_b, :, :].float() * _lower(c)[None, :, :]
        )
    return out, masked


@helion.kernel(backend="cute", static_shapes=True)
def bf16_cumsum(x: torch.Tensor) -> torch.Tensor:
    # bf16 state: each combine rounds to bf16, in traversal order.
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.bfloat16, device=x.device)
    for tile in hl.tile(b, block_size=1):
        y = x[tile, :, :] * _lower(c)[None, :, :].to(torch.bfloat16)
        out[tile, :, :] = hl.cumsum(y, dim=2, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def padded_reverse_exp_cumsum(x: torch.Tensor) -> torch.Tensor:
    # C=40 scans a 64-wide padded tile; a reverse scan starts at the padded
    # end, where exp(0) is not zero.
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        out[tile, :, :] = hl.cumsum(
            torch.exp(x[tile, :, :].float()), dim=2, reverse=True
        )
    return out
