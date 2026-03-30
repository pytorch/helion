"""
AOT benchmark for recurrent step kernels.

Usage:
    # Autotune:
    HELION_AUTOTUNE_PRECOMPILE=spawn HELION_AUTOTUNE_IGNORE_ERRORS=1 \
    python -m helion.experimental.aot_runner --phase all \
      -- python -m examples.linear.aot_recurrent_bench

    # Just benchmark (no tuning):
    python -m examples.linear.aot_recurrent_bench
"""

from __future__ import annotations

import os

import torch
from triton.testing import do_bench

from helion._testing import DEVICE

from .linear_attention_engine import (
    recurrent_step_correction_fused,
    recurrent_step_fused,
)

DTYPE = torch.bfloat16

CONFIGS = [
    # (BH, D, DV)
    (4, 32, 16),
    (32, 64, 64),
    (32, 128, 128),
    (64, 128, 128),
    (128, 128, 128),
    (256, 128, 128),
    (256, 64, 64),
]


def main() -> None:
    aot_mode = os.environ.get("HELION_AOT_MODE", "")
    print(f"AOT Mode: {aot_mode or 'disabled'}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    print(f"{'Config':<28} {'fused (ms)':>10} {'corr (ms)':>10}")
    print("-" * 52)

    for BH, D, DV in CONFIGS:
        torch.manual_seed(0)
        q = torch.randn(BH, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(BH, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(BH, DV, device=DEVICE, dtype=DTYPE)
        state = torch.randn(BH, D, DV, device=DEVICE, dtype=DTYPE)
        alpha = torch.sigmoid(torch.randn(BH, device=DEVICE, dtype=DTYPE))
        beta = torch.sigmoid(torch.randn(BH, device=DEVICE, dtype=DTYPE))

        # No-correction
        try:
            recurrent_step_fused(q, k, v, state.clone(), alpha)
            fwd_ms = do_bench(
                lambda q=q, k=k, v=v, alpha=alpha: recurrent_step_fused(
                    q, k, v, state.clone(), alpha
                )
            )
        except Exception as e:
            fwd_ms = float("nan")

        # Correction
        try:
            recurrent_step_correction_fused(q, k, v, state.clone(), alpha, beta)
            corr_ms = do_bench(
                lambda q=q, k=k, v=v, alpha=alpha, beta=beta: recurrent_step_correction_fused(
                    q, k, v, state.clone(), alpha, beta
                )
            )
        except Exception as e:
            corr_ms = float("nan")

        label = f"BH={BH}, D={D}, DV={DV}"
        print(f"  {label:<26} {fwd_ms:>8.3f}ms {corr_ms:>8.3f}ms")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
