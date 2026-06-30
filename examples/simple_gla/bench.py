"""
Simple GLA - local benchmark vs FLA
===================================

Runs Helion's chunk_simple_gla against FLA on the 6 production shapes from
flash-linear-attention/benchmarks/ops/registry.py:184. Reports forward and
forward+backward timings.

Not a CI artifact, just a dev tool for tracking perf as we tune.

Usage:
    HELION_AUTOTUNE_EFFORT=none python -m examples.simple_gla.bench
    HELION_AUTOTUNE_EFFORT=quick python -m examples.simple_gla.bench --fwd-only
"""

from __future__ import annotations

import argparse
import math

import torch

from .chunk import chunk_simple_gla
from .chunk import fla_chunk_simple_gla_native
from .chunk import naive_recurrent_simple_gla
from helion._testing import DEVICE
from helion._testing import do_bench

# Helion passes if its drift from the fp32 reference is within ACC_FACTOR of
# FLA's, so the gate tracks FLA's own bf16 accuracy instead of a fixed bound.
# The factor absorbs benign reduction-order differences between two correct
# bf16 kernels; the floor keeps the test from being absurdly strict on shapes
# where FLA is near-exact.
ACC_FACTOR = 2.0
ACC_FLOOR = 1e-3

# (name, B, T, H, D): D is both query/key dim and value dim (D=DV) per FLA registry.
# Shapes from flash-linear-attention/benchmarks/ops/registry.py:184.
SHAPES = [
    ("B1_T8192_H96_D128", 1, 8192, 96, 128),
    ("B2_T16384_H16_D128", 2, 16384, 16, 128),
    ("B4_T2048_H16_D128", 4, 2048, 16, 128),
    ("B4_T4096_H64_D128", 4, 4096, 64, 128),
    ("B8_T2048_H32_D256", 8, 2048, 32, 256),
    ("B8_T1024_H8_D64", 8, 1024, 8, 64),
]

# bf16 matches FLA's prod baseline (flash-linear-attention/benchmarks/ops/registry.py:202).
DTYPE = torch.bfloat16


def _make_inputs(
    b: int, t: int, h: int, d: int, requires_grad: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    k = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    v = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    # scalar log-decay per head per timestep, token-first [B, T, H], negative.
    g = torch.nn.functional.logsigmoid(
        torch.randn(b, t, h, dtype=torch.float32, device=DEVICE)
    ).requires_grad_(requires_grad)
    return q, k, v, g


def _accuracy(b: int, t: int, h: int, d: int, scale: float, fwd_only: bool) -> str:
    """Check Helion and FLA against the fp32 naive recurrence on one shape.

    g is the token-first [B, T, H] scalar log-decay, shared by FLA and the naive
    reference (no transpose). Drift is FLA's get_err_ratio (relative RMS error);
    a tensor passes if Helion's ratio is within ACC_FACTOR of FLA's. Returns
    "<verdict> <h>/<fla>" with the worst ratio on each side, suffixed " bwd:oom"
    when the fp32 naive backward runs out of memory and the gradient check is
    skipped.
    """
    from fla.utils import get_err_ratio  # pyrefly: ignore

    q, k, v, g = _make_inputs(b, t, h, d, requires_grad=not fwd_only)
    q_fla = q.detach().transpose(1, 2).contiguous().requires_grad_(not fwd_only)
    k_fla = k.detach().transpose(1, 2).contiguous().requires_grad_(not fwd_only)
    v_fla = v.detach().transpose(1, 2).contiguous().requires_grad_(not fwd_only)
    g_fla = g.detach().clone().requires_grad_(not fwd_only)

    # name -> (helion ratio, fla ratio) vs the fp32 reference.
    ratios: dict[str, tuple[float, float]] = {}

    with torch.no_grad():
        ref = naive_recurrent_simple_gla(
            q.float(), k.float(), v.float(), g.float(), scale=scale
        )
    h_out = chunk_simple_gla(q.detach(), k.detach(), v.detach(), g.detach(), scale=scale)
    f_out = fla_chunk_simple_gla_native(
        q_fla.detach(), k_fla.detach(), v_fla.detach(), g_fla.detach(), scale=scale
    ).transpose(1, 2)
    ratios["o"] = (get_err_ratio(h_out, ref), get_err_ratio(f_out, ref))

    suffix = ""
    if not fwd_only:
        try:
            qn = q.float().detach().requires_grad_(True)
            kn = k.float().detach().requires_grad_(True)
            vn = v.float().detach().requires_grad_(True)
            gn = g.float().detach().requires_grad_(True)
            go = torch.randn(b, h, t, d, dtype=torch.float32, device=DEVICE)
            naive_recurrent_simple_gla(qn, kn, vn, gn, scale=scale).backward(go)
            chunk_simple_gla(q, k, v, g, scale=scale).backward(go.to(DTYPE))
            fla_chunk_simple_gla_native(q_fla, k_fla, v_fla, g_fla, scale=scale).transpose(
                1, 2
            ).backward(go.to(DTYPE))
            for name, hg, fg, ng in (
                ("dq", q.grad, q_fla.grad.transpose(1, 2), qn.grad),
                ("dk", k.grad, k_fla.grad.transpose(1, 2), kn.grad),
                ("dv", v.grad, v_fla.grad.transpose(1, 2), vn.grad),
                ("dg", g.grad, g_fla.grad, gn.grad),
            ):
                ratios[name] = (get_err_ratio(hg, ng), get_err_ratio(fg, ng))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            suffix = " bwd:oom"

    fails = [
        n for n, (hr, fr) in ratios.items() if hr > max(ACC_FACTOR * fr, ACC_FLOOR)
    ]
    worst_h = max(hr for hr, _ in ratios.values())
    worst_fla = max(fr for _, fr in ratios.values())
    verdict = f"FAIL {','.join(fails)}" if fails else "ok"
    return f"{verdict} {worst_h:.1e}/{worst_fla:.1e}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fwd-only", action="store_true", help="skip forward+backward bench"
    )
    parser.add_argument(
        "--no-accuracy", action="store_true", help="skip the accuracy check"
    )
    args = parser.parse_args()

    rows: list[str] = []

    for name, b, t, h, d in SHAPES:
        scale = 1.0 / math.sqrt(d)

        # Pre-transpose FLA's q/k/v to its token-first layout off the timer, so
        # FLA isn't charged a transpose tax our head-first kernel doesn't pay. g
        # is already token-first [B,T,H], so it needs no transpose.
        q, k, v, g = _make_inputs(b, t, h, d, requires_grad=False)
        q_fla = q.transpose(1, 2).contiguous()  # token-first [B,T,H,D] for FLA
        k_fla = k.transpose(1, 2).contiguous()
        v_fla = v.transpose(1, 2).contiguous()

        def helion_fwd(
            q: torch.Tensor = q,
            k: torch.Tensor = k,
            v: torch.Tensor = v,
            g: torch.Tensor = g,
            scale: float = scale,
        ) -> torch.Tensor:
            return chunk_simple_gla(q, k, v, g, scale=scale)

        def fla_fwd(
            q: torch.Tensor = q_fla,
            k: torch.Tensor = k_fla,
            v: torch.Tensor = v_fla,
            g: torch.Tensor = g,
            scale: float = scale,
        ) -> torch.Tensor:
            return fla_chunk_simple_gla_native(q, k, v, g, scale=scale)

        # Warm up so each kernel autotunes on a clean GPU before the timed region.
        helion_fwd()
        fla_fwd()
        torch.cuda.empty_cache()

        h_fwd_ms = do_bench(helion_fwd)
        f_fwd_ms = do_bench(fla_fwd)
        fwd_pct = 100.0 * f_fwd_ms / h_fwd_ms  # pyrefly: ignore [unsupported-operation]

        fb_cols = ""
        if not args.fwd_only:
            q_g, k_g, v_g, g_g = _make_inputs(b, t, h, d, requires_grad=True)
            grad_out = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE)
            # FLA-layout grad leaves: transpose off the timer, then mark as leaves.
            # g is already token-first [B,T,H], reused as-is; grad_out is transposed
            # for FLA's backward.
            q_fla_g = q_g.detach().transpose(1, 2).contiguous().requires_grad_(True)
            k_fla_g = k_g.detach().transpose(1, 2).contiguous().requires_grad_(True)
            v_fla_g = v_g.detach().transpose(1, 2).contiguous().requires_grad_(True)
            g_fla_g = g_g.detach().clone().requires_grad_(True)
            grad_out_fla = grad_out.transpose(1, 2).contiguous()

            def helion_fb(
                q: torch.Tensor = q_g,
                k: torch.Tensor = k_g,
                v: torch.Tensor = v_g,
                g: torch.Tensor = g_g,
                go: torch.Tensor = grad_out,
                scale: float = scale,
            ) -> None:
                o = chunk_simple_gla(q, k, v, g, scale=scale)
                o.backward(go)
                q.grad = k.grad = v.grad = g.grad = None

            def fla_fb(
                q: torch.Tensor = q_fla_g,
                k: torch.Tensor = k_fla_g,
                v: torch.Tensor = v_fla_g,
                g: torch.Tensor = g_fla_g,
                go: torch.Tensor = grad_out_fla,
                scale: float = scale,
            ) -> None:
                fla_chunk_simple_gla_native(q, k, v, g, scale=scale).backward(go)
                q.grad = k.grad = v.grad = g.grad = None

            helion_fb()
            fla_fb()
            torch.cuda.empty_cache()

            h_fb_ms = do_bench(helion_fb)
            f_fb_ms = do_bench(fla_fb)
            fb_pct = 100.0 * f_fb_ms / h_fb_ms  # pyrefly: ignore [unsupported-operation]
            fb_cols = f" {h_fb_ms:>10.3f}ms {f_fb_ms:>10.3f}ms {fb_pct:>9.1f}%"

        acc = (
            "skip" if args.no_accuracy else _accuracy(b, t, h, d, scale, args.fwd_only)
        )

        row = (
            f"{name:<22} {acc:>20} "
            f"{h_fwd_ms:>10.3f}ms {f_fwd_ms:>10.3f}ms {fwd_pct:>9.1f}%" + fb_cols
        )
        rows.append(row)

    header = (
        f"{'Shape':<22} {'acc h/fla':>20} "
        f"{'helion fwd':>11} {'fla fwd':>11} {'fwd %FLA':>10}"
    )
    if not args.fwd_only:
        header += f" {'helion f+b':>11} {'fla f+b':>11} {'f+b %FLA':>10}"
    print()
    print("acc: Helion & FLA vs fp32 naive (err ratio; ok if helion <= 2x fla)")
    print(header)
    print("-" * (len(header) + 4))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
