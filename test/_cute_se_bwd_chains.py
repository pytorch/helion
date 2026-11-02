"""Two-GEMM ``hl.barrier()`` chains for the CuTe per-phase tcgen05 tests.

These are the backward kernels ``examples/squeeze_and_excitation_net.py`` had
before its backward was restructured to compute ``grad_c`` once
(``squeeze_and_excitation_net_bwd_dc`` followed by one matmul per gradient).
Each chain is a pointwise phase and two GEMM phases separated by barriers,
the shape the phase planner (``materialized_matmul_block_ids``,
``tcgen05_region_pid_types``) is pinned against; the example no longer has a
kernel of that shape, so the bodies live here as plain functions for
``helion.kernel(fn, backend="cute", ...)``.
"""

from __future__ import annotations

import torch

import helion.language as hl


def se_bwd_dx_chain(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """grad_x = grad_out * d + ((grad_out * x * d * (1 - d)) @ b.T * (c > 0)) @ a.T

    A pointwise phase (grad_cb), the relu-backward masked GEMM (grad_c) and
    the grad_x GEMM whose epilogue adds ``grad_out * d``.
    """
    m, n = x.size()
    k = a.size(1)

    grad_cb = torch.empty([m, n], dtype=x.dtype, device=x.device)
    grad_c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for point_m, point_n in hl.tile((m, n)):
        grad_cb[point_m, point_n] = (
            grad_out[point_m, point_n]
            * x[point_m, point_n]
            * d[point_m, point_n]
            * (1.0 - d[point_m, point_n])
        )
    hl.barrier()

    for row_c, col_c in hl.tile((m, k)):
        acc_c = hl.zeros([row_c, col_c], dtype=torch.float32)
        for reduce_n in hl.tile(n):
            acc_c = torch.addmm(acc_c, grad_cb[row_c, reduce_n], b[col_c, reduce_n].T)
        grad_c[row_c, col_c] = acc_c.to(x.dtype) * (c[row_c, col_c] > 0)
    hl.barrier()

    for row_x, col_x in hl.tile((m, n)):
        acc_x = hl.zeros([row_x, col_x], dtype=torch.float32)
        for reduce_k in hl.tile(k):
            acc_x = torch.addmm(acc_x, grad_c[row_x, reduce_k], a[col_x, reduce_k].T)
        grad_x[row_x, col_x] = grad_out[row_x, col_x] * d[row_x, col_x] + acc_x.to(
            x.dtype
        )

    return grad_x


def se_bwd_da_chain(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """grad_a = x.T @ ((grad_out * x * d * (1 - d)) @ b.T * (c > 0))

    A pointwise phase (grad_cb), the relu-backward masked GEMM (grad_c) and
    the grad_a GEMM with a transposed (column-major) A operand and a plain
    store.
    """
    m, n = x.size()
    k = c.size(1)

    grad_cb = torch.empty([m, n], dtype=x.dtype, device=x.device)
    grad_c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    grad_a = torch.empty([n, k], dtype=x.dtype, device=x.device)

    for point_m, point_n in hl.tile((m, n)):
        grad_cb[point_m, point_n] = (
            grad_out[point_m, point_n]
            * x[point_m, point_n]
            * d[point_m, point_n]
            * (1.0 - d[point_m, point_n])
        )
    hl.barrier()

    for row_c, col_c in hl.tile((m, k)):
        acc_c = hl.zeros([row_c, col_c], dtype=torch.float32)
        for reduce_n in hl.tile(n):
            acc_c = torch.addmm(acc_c, grad_cb[row_c, reduce_n], b[col_c, reduce_n].T)
        grad_c[row_c, col_c] = acc_c.to(x.dtype) * (c[row_c, col_c] > 0)
    hl.barrier()

    for row_a, col_a in hl.tile((n, k)):
        acc_a = hl.zeros([row_a, col_a], dtype=torch.float32)
        for reduce_m in hl.tile(m):
            acc_a = torch.addmm(acc_a, x[reduce_m, row_a].T, grad_c[reduce_m, col_a])
        grad_a[row_a, col_a] = acc_a

    return grad_a
