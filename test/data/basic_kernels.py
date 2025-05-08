import helion
import helion.language as hl
import torch
from torch import empty_like
from helion._testing import DEVICE


@helion.kernel
def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel
def torch_ops_pointwise(x, y):
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(torch.add(torch.sin(x[tile]), torch.cos(y[tile])))
    return out


@helion.kernel
def hl_zeros_usage(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        tmp = hl.zeros(tile, dtype=x.dtype)
        tmp += x[tile]
        tmp += x[tile]
        out[tile] = tmp
    return out


@helion.kernel
def hl_full_usage(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        tmp = hl.full(tile, 1, dtype=x.dtype)
        tmp += x[tile]
        tmp += x[tile]
        out[tile] = tmp
    return out


@helion.kernel
def pointwise_device_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n, m = x.shape
    for tile_n in hl.tile(n):
        for tile_m in hl.tile(m):
            out[tile_n, tile_m] = torch.sigmoid(x[tile_n, tile_m] + 1)
    return out


global_tensor = torch.randn([512], device=DEVICE)
global_float = 0.5


@helion.kernel
def use_globals(a):
    out = empty_like(a)
    for tile0, tile1 in hl.tile(out.size()):
        out[tile0, tile1] = (
            torch.sin(torch.add(a[tile0, tile1], global_tensor[None, tile1]))
            + global_float
        )
    return out


def add_global_float(x, tile) -> torch.Tensor:
    return x + global_float


@helion.kernel
def inplace_mul(x, c):
    (x,) = torch.broadcast_tensors(x)
    for tile in hl.tile(x.size()):
        x[tile] *= c
    return x
