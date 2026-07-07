"""Layer normalization (forward only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        mean_val = torch.sum(acc, dim=-1) / n
        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n
        rstd_val = torch.rsqrt(var_val + 1e-5)
        out[tile_m, :] = (
            centered * rstd_val[:, None] * weight[:].to(torch.float32)
            + bias[:].to(torch.float32)
        ).to(x.dtype)
    return out


def _layer_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return F.layer_norm(x, [x.size(1)], weight, bias, eps=1e-5)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    return [
        ("torch", _layer_norm_torch),
        ("torch_compile", torch.compile(_layer_norm_torch)),
    ]


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    triton_tutorial_shapes = [(4096, 512 * i) for i in range(2, 32)]
    realistic_shapes = [
        (2048, 3584),
        (8192, 4096),
        (8192, 5120),
        (8192, 7168),
        (2048, 8192),
        (4096, 16384),
        (1024, 36864),
        (1152, 36864),
    ]
    shapes = list(dict.fromkeys(triton_tutorial_shapes + realistic_shapes))
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        M, N = shape
        x = torch.randn([M, N], device="cuda", dtype=torch.float16)
        w = torch.randn([N], device="cuda", dtype=torch.float16)
        b = torch.randn([N], device="cuda", dtype=torch.float16)

        def helion_call() -> torch.Tensor:
            return layer_norm(x, w, b)

        base_calls = [(name, (lambda fn=fn: fn(x, w, b))) for name, fn in baselines]
        return helion_call, base_calls, f"{M:>5d}  {N:>6d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        warmup=50,
        rep=200,
        verbose=verbose,
        shape_header=f"{'M':>5s}  {'N':>6s}",
    )


if __name__ == "__main__":
    main()
