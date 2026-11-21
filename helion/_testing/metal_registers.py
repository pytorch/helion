"""Kernels and configs the tests of MPP's register operands share
(``mpp_registers``): the GPU module (``test/test_metal_mpp_registers_mps.py``)
runs the kernels and the CPU-only one (``test/test_metal_mpp_registers.py``)
reads their code and MSL; the affine weights' kernel is
``metal_fragment.affine_mm``."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

F32 = torch.float32


def register_config(
    block_sizes: list[int], num_threads: list[int] | None = None, **knobs: object
) -> helion.Config:
    """An MPP config (``metal_matmul`` left at its default) whose launch is
    one 32-thread SIMD group, by default 4 x 8 threads over M and N."""
    return helion.Config.from_dict(
        {"block_sizes": block_sizes, "num_threads": num_threads or [4, 8, 1], **knobs}
    )


@helion.kernel(backend="metal", static_shapes=True)
def inner_tiled_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(2 x) @ y with M and N tiled in a loop of a one-program grid, so the
    launch's thread axes codegen sees differ inside the K loop."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=F32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * 2, y[tk, tn])
            out[tm, tn] = acc
    return out
