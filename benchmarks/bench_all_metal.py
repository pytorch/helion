"""Benchmark suite: Helion Metal vs PyTorch eager vs torch.compile vs MLX.

Operators:
  1. Vector add
  2. Matmul
  3. Matmul → ReLU (fused epilogue)
  4. RMSNorm
  5. Two-layer MLP:  x → Linear → ReLU → Linear  (multi-matmul + activation)
  6. GQA (Grouped-Query Attention)

Usage (on a Mac with MPS):
    python benchmarks/bench_all_metal.py
"""

from __future__ import annotations

import math
import sys
import time
from typing import Callable

import mlx.core as mx
import mlx.nn as mlx_nn
import torch

import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Helion kernels
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", autotune_effort="full")
def helion_vecadd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal", autotune_effort="full")
def helion_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="full")
def helion_matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc)
    return out


@helion.kernel(backend="metal", autotune_effort="full")
def helion_softmax(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


@helion.kernel(backend="metal", autotune_effort="full")
def helion_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        mean_val = torch.sum(values, dim=1, keepdim=True) / m
        centered = values - mean_val
        var_val = torch.sum(centered * centered, dim=1, keepdim=True) / m
        rstd_val = torch.rsqrt(var_val + eps)
        normalized = centered * rstd_val
        out[tile_n, :] = normalized * weight[None, :] + bias[None, :]
    return out


@helion.kernel(
    backend="metal",
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def helion_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_n in hl.tile(n):
        labels_tile = labels[tile_n]
        base_indices_tile = tile_n.index * v
        flat_indices = base_indices_tile + labels_tile
        logits_at_target = hl.load(logits_flat, [flat_indices])
        logits_rows = logits[tile_n, :]
        max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        shifted = logits_rows - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
        losses[tile_n] = log_sum_exp - logits_at_target
    return losses.mean()


@helion.kernel(backend="metal", autotune_effort="full")
def helion_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        sq = values * values
        mean_sq = torch.sum(sq, dim=1, keepdim=True) / m
        rms = torch.rsqrt(mean_sq + eps)
        out[tile_n, :] = values * rms * weight[None, :]
    return out


@helion.kernel(
    backend="metal",
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 32, 32], num_warps=4),
)
def helion_batched_fused_attention(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Batched fused attention: single dispatch for all B*H heads.

    Q[B*H, M, D], Kt[B*H, D, N], V[B*H, N, D] -> out[B*H, M, D].
    """
    num_heads = q.size(0)
    m_dim = q.size(1)
    head_dim = hl.specialize(q.size(2))
    n_dim = v.size(1)
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([num_heads, m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        for tile_n in hl.tile(n_dim):
            qk = torch.bmm(q[tile_b, tile_m, :], kt[tile_b, :, tile_n])
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            p2 = p.to(v.dtype)
            acc = torch.baddbmm(acc, p2, v[tile_b, tile_n, :])
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out


# Two-layer MLP cannot be a single Helion kernel today (would need two
# matmuls composed), so we call helion_matmul_relu + helion_matmul
# sequentially.  This still demonstrates end-to-end throughput.


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def bench_torch(
    fn: Callable[..., object],
    *args: torch.Tensor,
    warmup: int = 10,
    rep: int = 50,
) -> float:
    """Wall-clock benchmark with MPS synchronization.  Returns median ms or NaN on failure."""
    try:
        for _ in range(warmup):
            fn(*args)
        torch.mps.synchronize()
    except Exception:
        return float("nan")

    times: list[float] = []
    for _ in range(rep):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]


def bench_mlx(
    fn: Callable[..., mx.array],
    *args: mx.array,
    warmup: int = 10,
    rep: int = 50,
) -> float:
    """Wall-clock benchmark for MLX with mx.eval synchronization.  Returns median ms or NaN."""
    try:
        for _ in range(warmup):
            out = fn(*args)
            mx.eval(out)
    except Exception:
        return float("nan")

    times: list[float] = []
    for _ in range(rep):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEP = "=" * 96
COL_HDR = (
    f"{'Size':<16} {'Eager':>10} {'Inductor':>10} {'Helion':>10} "
    f"{'MLX':>10}  {'vs Eager':>9} {'vs Ind':>9} {'vs MLX':>9}"
)


def _fmt_time(t: float) -> str:
    return f"{t:>8.4f}ms" if not math.isnan(t) else f"{'N/A':>10}"


def _fmt_speedup(baseline: float, helion: float) -> str:
    if math.isnan(baseline) or math.isnan(helion):
        return f"{'N/A':>9}"
    return f"{baseline / helion:>8.2f}x"


def row(label: str, t_e: float, t_i: float, t_h: float, t_m: float) -> str:
    """Format a benchmark row.  Speedup = baseline / Helion, >1 means Helion faster."""
    return (
        f"{label:<16} "
        f"{_fmt_time(t_e)} {_fmt_time(t_i)} {_fmt_time(t_h)} "
        f"{_fmt_time(t_m)}  "
        f"{_fmt_speedup(t_e, t_h)} {_fmt_speedup(t_i, t_h)} {_fmt_speedup(t_m, t_h)}"
    )


GPU_COOLDOWN_SEC = 2


def cooldown() -> None:
    """Sleep briefly between benchmark sizes to avoid GPU thermal throttling."""
    torch.mps.synchronize()
    time.sleep(GPU_COOLDOWN_SEC)


def section(title: str) -> None:
    cooldown()
    print(f"\n{SEP}")
    print(title)
    print(SEP)
    print(COL_HDR)
    print("-" * len(COL_HDR))


# ---------------------------------------------------------------------------
# 1. Vector add
# ---------------------------------------------------------------------------


def bench_vecadd() -> None:
    section("1. VECTOR ADD (float32)")
    device = torch.device("mps")
    sizes = [1024, 16_384, 262_144, 1_048_576, 4_194_304, 16_777_216]

    eager_fn = torch.add
    compiled_fn = torch.compile(torch.add, backend="inductor")

    for n in sizes:
        x_pt = torch.randn(n, device=device, dtype=torch.float32)
        y_pt = torch.randn(n, device=device, dtype=torch.float32)
        x_mlx = mx.array(x_pt.cpu().numpy())
        y_mlx = mx.array(y_pt.cpu().numpy())
        mx.eval(x_mlx, y_mlx)

        t_e = bench_torch(eager_fn, x_pt, y_pt)
        t_i = bench_torch(compiled_fn, x_pt, y_pt)
        t_h = bench_torch(helion_vecadd, x_pt, y_pt)
        t_m = bench_mlx(mx.add, x_mlx, y_mlx)

        if n >= 1_000_000:
            label = f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            label = f"{n // 1_000}K"
        else:
            label = str(n)
        print(row(label, t_e, t_i, t_h, t_m))
        cooldown()


# ---------------------------------------------------------------------------
# 2. Matmul
# ---------------------------------------------------------------------------


def bench_matmul() -> None:
    section("2. MATMUL (float32)")
    device = torch.device("mps")
    sizes = [128, 256, 512, 1024, 2048]

    eager_fn = torch.mm
    compiled_fn = torch.compile(torch.mm, backend="inductor")

    for sz in sizes:
        x_pt = torch.randn(sz, sz, device=device, dtype=torch.float32)
        y_pt = torch.randn(sz, sz, device=device, dtype=torch.float32)
        x_mlx = mx.array(x_pt.cpu().numpy())
        y_mlx = mx.array(y_pt.cpu().numpy())
        mx.eval(x_mlx, y_mlx)

        t_e = bench_torch(eager_fn, x_pt, y_pt)
        t_i = bench_torch(compiled_fn, x_pt, y_pt)
        t_h = bench_torch(helion_matmul, x_pt, y_pt)
        t_m = bench_mlx(mx.matmul, x_mlx, y_mlx)

        print(row(f"{sz}x{sz}", t_e, t_i, t_h, t_m))
        cooldown()


# ---------------------------------------------------------------------------
# 3. Matmul + ReLU
# ---------------------------------------------------------------------------


def bench_matmul_relu() -> None:
    section("3. MATMUL + RELU (float32, fused epilogue)")
    device = torch.device("mps")
    sizes = [128, 256, 512, 1024, 2048]

    def eager_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.mm(x, y))

    compiled_fn = torch.compile(eager_fn, backend="inductor")

    def mlx_fn(x: mx.array, y: mx.array) -> mx.array:
        return mx.maximum(x @ y, 0)

    for sz in sizes:
        x_pt = torch.randn(sz, sz, device=device, dtype=torch.float32)
        y_pt = torch.randn(sz, sz, device=device, dtype=torch.float32)
        x_mlx = mx.array(x_pt.cpu().numpy())
        y_mlx = mx.array(y_pt.cpu().numpy())
        mx.eval(x_mlx, y_mlx)

        t_e = bench_torch(eager_fn, x_pt, y_pt)
        t_i = bench_torch(compiled_fn, x_pt, y_pt)
        t_h = bench_torch(helion_matmul_relu, x_pt, y_pt)
        t_m = bench_mlx(mlx_fn, x_mlx, y_mlx)

        print(row(f"{sz}x{sz}", t_e, t_i, t_h, t_m))
        cooldown()


# ---------------------------------------------------------------------------
# 4. RMSNorm
# ---------------------------------------------------------------------------


def bench_rmsnorm() -> None:
    section("4. RMSNORM (float32)")
    device = torch.device("mps")
    shapes = [(256, 1024), (1024, 1024), (1024, 4096), (4096, 2560), (4096, 4096)]

    for m, n in shapes:
        x_pt = torch.randn(m, n, device=device, dtype=torch.float32)
        w_pt = torch.randn(n, device=device, dtype=torch.float32)
        x_mlx = mx.array(x_pt.cpu().numpy())
        w_mlx = mx.array(w_pt.cpu().numpy())
        mx.eval(x_mlx, w_mlx)

        eps = 1e-6

        def eager_rmsnorm(
            x: torch.Tensor, _eps: float = eps, _w: torch.Tensor = w_pt
        ) -> torch.Tensor:
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + _eps)
            return x * rms * _w

        compiled_rmsnorm = torch.compile(eager_rmsnorm, backend="inductor")

        mlx_rms_layer = mlx_nn.RMSNorm(n, eps=eps)
        mlx_rms_layer.weight = w_mlx
        mx.eval(mlx_rms_layer.parameters())

        t_e = bench_torch(eager_rmsnorm, x_pt)
        t_i = bench_torch(compiled_rmsnorm, x_pt)
        t_h = bench_torch(helion_rms_norm, x_pt, w_pt)
        t_m = bench_mlx(mlx_rms_layer, x_mlx)

        print(row(f"({m}, {n})", t_e, t_i, t_h, t_m))


# ---------------------------------------------------------------------------
# 5. Softmax
# ---------------------------------------------------------------------------


def bench_softmax() -> None:
    section("5. SOFTMAX (float32)")
    device = torch.device("mps")
    shapes = [(256, 1024), (1024, 1024), (1024, 4096), (4096, 2560)]

    for m, n in shapes:
        x_pt = torch.randn(m, n, device=device, dtype=torch.float32)
        x_mlx = mx.array(x_pt.cpu().numpy())
        mx.eval(x_mlx)

        def eager_softmax(
            x: torch.Tensor = x_pt,
        ) -> torch.Tensor:
            return torch.nn.functional.softmax(x, dim=-1)

        torch._dynamo.reset()
        compiled_softmax = torch.compile(eager_softmax, backend="inductor")

        def mlx_softmax(x: mx.array = x_mlx) -> mx.array:
            return mx.softmax(x, axis=-1)

        t_e = bench_torch(eager_softmax)
        t_i = bench_torch(compiled_softmax)
        t_h = bench_torch(helion_softmax, x_pt)
        t_m = bench_mlx(mlx_softmax)

        print(row(f"({m}, {n})", t_e, t_i, t_h, t_m))


# ---------------------------------------------------------------------------
# 6. Layer Norm
# ---------------------------------------------------------------------------


def bench_layernorm() -> None:
    section("6. LAYERNORM (float32)")
    device = torch.device("mps")
    shapes = [(256, 1024), (1024, 1024), (1024, 4096), (4096, 2560)]

    for m, n in shapes:
        x_pt = torch.randn(m, n, device=device, dtype=torch.float32)
        w_pt = torch.randn(n, device=device, dtype=torch.float32)
        b_pt = torch.randn(n, device=device, dtype=torch.float32)
        x_mlx = mx.array(x_pt.cpu().numpy())
        w_mlx = mx.array(w_pt.cpu().numpy())
        b_mlx = mx.array(b_pt.cpu().numpy())
        mx.eval(x_mlx, w_mlx, b_mlx)

        def eager_ln(
            x: torch.Tensor = x_pt,
            _w: torch.Tensor = w_pt,
            _b: torch.Tensor = b_pt,
            _n: int = n,
        ) -> torch.Tensor:
            return torch.nn.functional.layer_norm(x, [_n], _w, _b)

        torch._dynamo.reset()
        compiled_ln = torch.compile(eager_ln, backend="inductor")

        mlx_ln = mlx_nn.LayerNorm(n)
        mlx_ln.weight = w_mlx
        mlx_ln.bias = b_mlx
        mx.eval(mlx_ln.parameters())

        t_e = bench_torch(eager_ln)
        t_i = bench_torch(compiled_ln)
        t_h = bench_torch(helion_layer_norm, x_pt, w_pt, b_pt)
        t_m = bench_mlx(mlx_ln, x_mlx)

        print(row(f"({m}, {n})", t_e, t_i, t_h, t_m))


# ---------------------------------------------------------------------------
# 7. Cross Entropy
# ---------------------------------------------------------------------------


def bench_cross_entropy() -> None:
    section("7. CROSS ENTROPY (float32)")
    device = torch.device("mps")
    # (batch, vocab_size)
    configs = [(256, 1024), (1024, 1024), (1024, 4096), (4096, 2560)]

    for n, v in configs:
        logits_pt = torch.randn(n, v, device=device, dtype=torch.float32)
        labels_pt = torch.randint(0, v, (n,), device=device)
        logits_mlx = mx.array(logits_pt.cpu().numpy())
        labels_mlx = mx.array(labels_pt.cpu().numpy().astype("int32"))
        mx.eval(logits_mlx, labels_mlx)

        def eager_ce(
            _logits: torch.Tensor = logits_pt,
            _labels: torch.Tensor = labels_pt,
        ) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(_logits, _labels)

        torch._dynamo.reset()
        compiled_ce = torch.compile(eager_ce, backend="inductor")

        def mlx_ce(
            _logits: mx.array = logits_mlx,
            _labels: mx.array = labels_mlx,
        ) -> mx.array:
            return mx.mean(mlx_nn.losses.cross_entropy(_logits, _labels))

        t_e = bench_torch(eager_ce)
        t_i = bench_torch(compiled_ce)
        t_h = bench_torch(helion_cross_entropy, logits_pt, labels_pt)
        t_m = bench_mlx(mlx_ce)

        print(row(f"({n}, {v})", t_e, t_i, t_h, t_m))


# ---------------------------------------------------------------------------
# 8. Two-layer MLP:  x @ W1 → ReLU → @ W2
# ---------------------------------------------------------------------------


def bench_two_layer_mlp() -> None:
    section("8. TWO-LAYER MLP: x@W1 -> ReLU -> @W2 (float32)")
    device = torch.device("mps")
    # (batch, in_features, hidden, out_features)
    configs = [
        (128, 512, 1024, 512),
        (256, 1024, 2048, 1024),
        (512, 1024, 2048, 1024),
    ]

    for batch, in_f, hid, out_f in configs:
        x_pt = torch.randn(batch, in_f, device=device, dtype=torch.float32)
        w1_pt = torch.randn(in_f, hid, device=device, dtype=torch.float32)
        w2_pt = torch.randn(hid, out_f, device=device, dtype=torch.float32)

        x_mlx = mx.array(x_pt.cpu().numpy())
        w1_mlx = mx.array(w1_pt.cpu().numpy())
        w2_mlx = mx.array(w2_pt.cpu().numpy())
        mx.eval(x_mlx, w1_mlx, w2_mlx)

        def eager_mlp(
            x: torch.Tensor,
            _w1: torch.Tensor = w1_pt,
            _w2: torch.Tensor = w2_pt,
        ) -> torch.Tensor:
            return torch.mm(torch.relu(torch.mm(x, _w1)), _w2)

        compiled_mlp = torch.compile(eager_mlp, backend="inductor")

        def helion_mlp(
            x: torch.Tensor,
            _w1: torch.Tensor = w1_pt,
            _w2: torch.Tensor = w2_pt,
        ) -> torch.Tensor:
            h = helion_matmul_relu(x, _w1)
            return helion_matmul(h, _w2)

        def mlx_mlp(
            x: mx.array,
            _w1: mx.array = w1_mlx,
            _w2: mx.array = w2_mlx,
        ) -> mx.array:
            return mx.maximum(x @ _w1, 0) @ _w2

        label = f"{batch}x{in_f}->{hid}->{out_f}"

        t_e = bench_torch(eager_mlp, x_pt)
        t_i = bench_torch(compiled_mlp, x_pt)
        t_h = bench_torch(helion_mlp, x_pt)
        t_m = bench_mlx(mlx_mlp, x_mlx)

        print(row(label, t_e, t_i, t_h, t_m))
        cooldown()


# ---------------------------------------------------------------------------
# 6. GQA (Grouped-Query Attention)
# ---------------------------------------------------------------------------


def bench_gqa() -> None:
    section("9. GQA (Grouped-Query Attention, float32)")
    device = torch.device("mps")
    # (batch, q_heads, kv_heads, seq_len, head_dim)
    configs = [
        (1, 8, 2, 64, 32),
        (1, 8, 2, 128, 32),
        (1, 8, 2, 256, 32),
        (2, 8, 2, 64, 32),
        (2, 8, 2, 128, 32),
        (2, 8, 2, 256, 32),
    ]

    for batch, q_heads, kv_heads, seq_len, head_dim in configs:
        q = torch.randn(batch, q_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, kv_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, kv_heads, seq_len, head_dim, device=device)

        n_rep = q_heads // kv_heads
        k_exp = k.repeat_interleave(n_rep, dim=1)
        v_exp = v.repeat_interleave(n_rep, dim=1)

        # Eager
        def eager_gqa(
            _q: torch.Tensor = q,
            _k: torch.Tensor = k_exp,
            _v: torch.Tensor = v_exp,
        ) -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(_q, _k, _v)

        torch._dynamo.reset()
        compiled_gqa = torch.compile(eager_gqa, backend="inductor")

        # Helion
        num_heads = batch * q_heads
        q3 = q.reshape(num_heads, seq_len, head_dim).contiguous()
        k3 = k_exp.reshape(num_heads, seq_len, head_dim).contiguous()
        v3 = v_exp.reshape(num_heads, seq_len, head_dim).contiguous()
        kt3 = k3.transpose(1, 2).contiguous()

        def helion_gqa(
            _q3: torch.Tensor = q3,
            _kt3: torch.Tensor = kt3,
            _v3: torch.Tensor = v3,
        ) -> torch.Tensor:
            return helion_batched_fused_attention(_q3, _kt3, _v3)

        # MLX
        q_mlx = mx.array(q.cpu().numpy())
        k_exp_mlx = mx.array(k_exp.cpu().numpy())
        v_exp_mlx = mx.array(v_exp.cpu().numpy())
        mx.eval(q_mlx, k_exp_mlx, v_exp_mlx)

        scale = 1.0 / math.sqrt(head_dim)

        def mlx_gqa(
            _q: mx.array = q_mlx,
            _k: mx.array = k_exp_mlx,
            _v: mx.array = v_exp_mlx,
            _scale: float = scale,
        ) -> mx.array:
            scores = (_q @ mx.transpose(_k, (0, 1, 3, 2))) * _scale
            weights = mx.softmax(scores, axis=-1)
            return weights @ _v

        label = f"B{batch}H{q_heads}K{kv_heads}S{seq_len}"

        t_e = bench_torch(eager_gqa)
        t_i = bench_torch(compiled_gqa)
        t_h = bench_torch(helion_gqa)
        t_m = bench_mlx(mlx_gqa)

        print(row(label, t_e, t_i, t_h, t_m))
        cooldown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import os
    import warnings

    # Suppress noisy Helion autotuner / cache logs and warnings
    os.environ["HELION_AUTOTUNE_LOG_LEVEL"] = "ERROR"
    warnings.filterwarnings("ignore")

    # Redirect stderr to /dev/null to suppress direct prints from cache layer
    _real_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    print("Helion Metal vs PyTorch Eager vs torch.compile(inductor) vs MLX")
    print(f"MLX {mx.__version__}  |  PyTorch {torch.__version__}")
    print("Device: Apple Silicon (MPS / Metal GPU)")
    print()
    print("Columns: median wall-clock ms (lower is better)")
    print("vs X = X time / Helion time  (>1 = Helion faster)")

    bench_vecadd()
    bench_matmul()
    bench_matmul_relu()
    bench_rmsnorm()
    bench_softmax()
    bench_layernorm()
    bench_cross_entropy()
    bench_two_layer_mlp()
    bench_gqa()

    # Restore stderr
    sys.stderr.close()
    sys.stderr = _real_stderr

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()
