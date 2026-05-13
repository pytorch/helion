"""Pinned matmul kernel for the byte-identity golden test.

Hosts the kernel definition at a stable file path so the
``src[<file>:<line>]`` comments embedded in the generated kernel
do not drift when the test file changes. Pair file:
``test/golden/tcgen05_role_local_monolithic_4096_bf16.py.expected``
encodes ``bound.to_triton_code(seed_config)`` for the retained
``ROLE_LOCAL_MONOLITHIC`` seed (cute_plan.md §10.1).

Edits here change line numbers in the embedded ``src[...]:``
comments and break byte-identity; the
``test_tcgen05_role_local_monolithic_byte_identical_golden`` test
catches the drift, so re-run with ``EXPECTTEST_ACCEPT=1`` after any
intentional change here.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute")
def cute_matmul_role_local_monolithic_4096_bf16(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out
