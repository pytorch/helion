"""
Epilogue Subtiling Example
==========================

This example demonstrates matmul kernels with heavy epilogues that benefit from
epilogue subtiling on Blackwell (sm_100+).  Epilogue subtiling splits the store
from ``[BLOCK_M, BLOCK_N]`` into ``SUBTILE_FACTOR x [BLOCK_M, BLOCK_N / SUBTILE_FACTOR]``,
halving the accumulator shared-memory footprint and enabling an extra pipeline stage.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Kernel 1 -- Matmul + Residual + Bias + GELU + Cast
# ---------------------------------------------------
# CUTLASS-style residual + bias + GELU forward epilogue with two
# fp32 reads (residual, bias) fused into the output tile.


# %%
@helion.kernel(static_shapes=True)
def matmul_bias_residual_gelu_cast(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])

        val = acc * 1.25
        val = val + residual[tile_m, tile_n].to(torch.float32) * 0.5
        val = val + bias[tile_n]
        val = torch.nn.functional.gelu(val)
        out[tile_m, tile_n] = val.to(torch.float16)

    return out


# %%
# Kernel 2 -- Matmul + Bias + GELU with Auxiliary Output
# ------------------------------------------------------
# cuBLASLt / CUTLASS-style GELU+AUX forward epilogue that writes both
# the pre-activation (aux) and post-GELU (out) tensors.


# %%
@helion.kernel(static_shapes=True)
def matmul_bias_gelu_aux(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    aux = torch.empty([m, n], dtype=torch.float16, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])

        pre = acc * 1.25
        pre = pre + bias[tile_n]
        aux[tile_m, tile_n] = pre.to(torch.float16)
        out[tile_m, tile_n] = torch.nn.functional.gelu(pre).to(torch.float16)

    return out, aux


# %%
# Verification
# ------------


# %%
def check(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
    w = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
    bias = torch.randn([n], device=DEVICE, dtype=HALF_DTYPE)
    residual = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

    def baseline_residual_gelu(
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        acc = x.float() @ w.float()
        val = acc * 1.25 + residual.float() * 0.5 + bias.float()
        return torch.nn.functional.gelu(val).half()

    run_example(
        matmul_bias_residual_gelu_cast,
        baseline_residual_gelu,
        (x, w, bias, residual),
    )

    def baseline_gelu_aux(
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acc = x.float() @ w.float()
        pre = acc * 1.25 + bias.float()
        return torch.nn.functional.gelu(pre).half(), pre.half()

    run_example(
        matmul_bias_gelu_aux,
        baseline_gelu_aux,  # pyrefly: ignore[bad-argument-type]
        (x, w, bias),
    )


# %%
# Main
# ----


# %%
def main() -> None:
    check(8192, 8192, 8192)


if __name__ == "__main__":
    main()
