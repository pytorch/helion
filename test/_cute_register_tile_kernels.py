"""Small column-reduction kernels shared by the register-tile codegen (CPU) and
numerics (GPU) tests.  Every kernel reduces a ``[m, n]`` tile along its strided
axis while the stride-1 axis stays a free tile, the shape the CuTe register-tile
lowering (``helion/_compiler/cute/register_tile_reductions.py``) targets."""

from __future__ import annotations

from typing import Any

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[:, tile_n].float().sum(0).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_scale(x: torch.Tensor) -> torch.Tensor:
    """Column normalization: a lane-invariant value derived from the reduction
    (``r``) feeds a lane-varying store."""
    m, n = x.size()
    y = torch.empty_like(x)
    for tile_n in hl.tile(n):
        t = x[:, tile_n].float()
        r = 1.0 / ((t * t).sum(0) + 1.0)
        y[:, tile_n] = (t * r[None, :]).to(x.dtype)
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_scale_three(x: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Three loaded tiles stay live until the consume pass."""
    m, n = x.size()
    y = torch.empty_like(x)
    for tile_n in hl.tile(n):
        t = x[:, tile_n].float()
        r = 1.0 / ((t * t).sum(0) + 1.0)
        y[:, tile_n] = (
            t * r[None, :] + u[:, tile_n].float() * r[None, :] + w[:, tile_n].float()
        ).to(x.dtype)
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_stats(x: torch.Tensor) -> torch.Tensor:
    """Two reductions (sum and sum of squares) over one tile."""
    m, n = x.size()
    out = torch.empty((2, n), dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        t = x[:, tile_n].float()
        out[0, tile_n] = t.sum(0)
        out[1, tile_n] = (t * t).sum(0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_sum_f32_out(x: torch.Tensor) -> torch.Tensor:
    """A bf16 tile summed into fp32 to exercise widening output packets."""
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[:, tile_n].float().sum(0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_softmax(x: torch.Tensor) -> torch.Tensor:
    """Two chained reductions: the sum's input depends on the max."""
    m, n = x.size()
    y = torch.empty_like(x)
    for tile_n in hl.tile(n):
        t = x[:, tile_n].float()
        mx = t.amax(0)
        e = torch.exp(t - mx[None, :])
        s = e.sum(0)
        y[:, tile_n] = (e / s[None, :]).to(x.dtype)
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_var(x: torch.Tensor) -> torch.Tensor:
    """Two-pass variance: the second reduction reads the first one's mean."""
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        t = x[:, tile_n].float()
        mean = t.sum(0) / m
        d = t - mean[None, :]
        out[tile_n] = (d * d).sum(0) / m
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_sum_guarded(x: torch.Tensor, flag: int) -> torch.Tensor:
    """A scalar-argument guard around a per-column value computed after the
    reduction: it lands inside the tile element loops."""
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        s = x[:, tile_n].float().sum(0)
        if flag > 0:
            s = s * 2.0
        out[tile_n] = s.to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_scale_into(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``col_scale`` into a caller-provided output, which may alias ``x``."""
    m, n = x.size()
    for tile_n in hl.tile(n):
        t = x[:, tile_n].float()
        r = 1.0 / ((t * t).sum(0) + 1.0)
        y[:, tile_n] = (t * r[None, :]).to(x.dtype)
    return y


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def col_sum_dynamic(x: torch.Tensor) -> torch.Tensor:
    """``col_sum`` with dynamic shapes: the reduction extent is a runtime value
    the bound kernel must not bake in."""
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[:, tile_n].float().sum(0).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def col_sum_nested(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A rolled device loop next to the reduction inside the tile body."""
    m, n = x.size()
    k = w.size(0)
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        s = x[:, tile_n].float().sum(0)
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for i in hl.grid(k):
            acc = acc + w[i, tile_n].float()
        out[tile_n] = (s + acc).to(x.dtype)
    return out


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_bwd(
    grad_out: torch.Tensor,
    grad_inv_rms: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backward ``helion.experimental.backward`` generates for the
    ``rms_norm_fwd`` of ``test_autodiff.py`` (``test_rms_norm_multiout``):
    two chained feature reductions over the row tile's columns (the mean that
    feeds ``inv_rms``, the ``grad_x`` sum) beside the ``grad_weight`` partial,
    a reduction over the row tile itself."""
    m, n = x.size()
    grad_x = torch.empty_like(x)
    block_m = hl.register_block_size(m)
    num_blocks = (m + block_m - 1) // block_m
    grad_weight_parts = torch.zeros(
        [num_blocks, n], dtype=torch.float32, device=x.device
    )
    for tile_m in hl.tile(m, block_size=block_m):
        go = grad_out[tile_m, :]
        g_inv = grad_inv_rms[tile_m]
        xt = x[tile_m, :]
        w = weight[:]
        mean = torch.mean(xt * xt, [-1])
        inv_rms = torch.rsqrt(mean + 1e-5)
        normalized = xt * inv_rms.reshape(-1, 1)
        go_w = go * w
        grad_weight_parts[tile_m.id, :] = torch.sum(go * normalized, [0], True)
        d_inv = g_inv + torch.sum(go_w * xt, [1], True).reshape(-1)
        d_mean = (d_inv * -0.5 * torch.pow(inv_rms, 3)).reshape(-1, 1) / n
        grad_x[tile_m, :] = go_w * inv_rms.reshape(-1, 1) + 2 * (d_mean * xt)
    return grad_x, grad_weight_parts.sum(0).to(weight.dtype)


def rms_norm_bwd_arguments(m: int = 64, n: int = 32) -> tuple[torch.Tensor, ...]:
    """``(grad_out, grad_inv_rms, x, weight)`` of ``rms_norm_bwd`` on CPU."""
    return (
        torch.empty((m, n), dtype=torch.float32),
        torch.empty((m,), dtype=torch.float32),
        torch.empty((m, n), dtype=torch.float32),
        torch.empty((n,), dtype=torch.float32),
    )


def column_config(
    bound: Any,
    *,
    block: int,
    tile_threads: int,
    reduction_threads: int,
    vec: int = 4,
    reduction_loop: int | None = None,
    **extra: Any,
) -> helion.Config:
    """Config for the one tile block and the one reduction block of a column
    kernel: ``block`` columns per CTA, ``tile_threads`` threads along the
    columns (``vec`` elements each), ``reduction_threads`` along the rows,
    persistent unless ``reduction_loop`` rows are looped per chunk."""
    spec = bound.config_spec
    (reduction,) = [
        block_size.block_id
        for block_size in bound.env.block_sizes
        if block_size.reduction
    ]
    (columns,) = spec.block_sizes.valid_block_ids()
    threads = {columns: tile_threads, reduction: reduction_threads}
    return spec.normalized_config(
        helion.Config(
            block_sizes=[block],
            num_threads=[
                threads.get(block_id, 0)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[reduction_loop],
            cute_vector_widths=[
                vec if block_id == columns else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            **extra,
        )
    )
