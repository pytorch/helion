"""Small-shape correctness smoke test for all attngym variant kernels.

Usage: python benchmarks/cute/attngym_smoke.py [variant ...]
Honors HELION_BACKEND. Pins block_sizes=[1, 128, 128] (the flash envelope)
rather than relying on the default config. Under the cute backend, also
reports whether the flash path fired for each variant.
"""

from __future__ import annotations

import dataclasses
import os
import sys

from attngym_kernels import kernel_call_args  # pyrefly: ignore [missing-import]
from attngym_variants import VARIANTS  # pyrefly: ignore [missing-import]
from attngym_variants import make_qkv  # pyrefly: ignore [missing-import]
from attngym_variants import reference_output  # pyrefly: ignore [missing-import]
import torch

from helion._testing import code_and_output


def smoke(name: str, seq: int = 512) -> bool:
    variant = VARIANTS[name]
    scale = max(1, variant.m // seq)
    small = dataclasses.replace(
        variant,
        b=min(variant.b, 2),
        m=variant.m // scale,
        n=variant.n // scale,
    )
    device = torch.device("cuda")
    q, k, v = make_qkv(small, device, seed=0)
    kernel, args = kernel_call_args(name, q, k, v, device)
    code, out = code_and_output(kernel, args, block_sizes=[1, 128, 128])
    flash = "flash_fa4_shared_storage" in code or "ws_overlap" in code
    score_mod, mask_mod = variant.make_mods(device)
    ref = reference_output(small, q, k, v, score_mod, mask_mod)
    diff = (out.float() - ref.float()).abs().max().item()
    tol = 3e-2 if variant.dtype is torch.bfloat16 else 2e-2
    ok = diff < tol
    marker = ""
    if os.environ.get("HELION_BACKEND") == "cute":
        marker = " flash" if flash else " generic"
    print(
        f"{name:16s} m={small.m:6d} n={small.n:6d} max_abs_diff={diff:.5f} "
        f"{'PASS' if ok else 'FAIL'}{marker}"
    )
    return ok


def main() -> None:
    names = sys.argv[1:] or list(VARIANTS)
    failures = []
    for name in names:
        try:
            ok = smoke(name)
        except Exception as e:
            print(f"{name:16s} ERROR {type(e).__name__}: {str(e)[:120]}")
            ok = False
        if not ok:
            failures.append(name)
    if failures:
        print("FAILURES:", ", ".join(failures))
        sys.exit(1)
    print("all pass")


if __name__ == "__main__":
    main()
