"""Root mean square normalization (forward only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


def _rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return F.rms_norm(x, [x.size(1)], weight, eps=eps)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    return [
        ("torch", _rms_norm_torch),
        ("torch_compile", torch.compile(_rms_norm_torch)),
    ]


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    tritonbench_shapes = [
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
        (2048, 8192),
        (2048, 16384),
        (2048, 32768),
    ]
    tritonbench_npot_shapes = [
        (2048, 48),
        (2048, 96),
        (2048, 127),
        (2048, 768),
        (2048, 1023),
        (2048, 1536),
        (2048, 2047),
        (2048, 3072),
        (2048, 5120),
        (2048, 6144),
    ]
    realistic_shapes = [
        (4096, 3584),
        (4096, 4096),
        (4096, 5120),
        (4096, 7168),
        (4096, 8192),
        (4096, 12288),
        (8192, 4096),
        (8192, 8192),
        (16384, 4096),
        (16384, 8192),
        (145956, 384),
        (380668, 512),
        (589824, 256),
        (1179648, 256),
    ]
    shapes = list(
        dict.fromkeys(tritonbench_shapes + tritonbench_npot_shapes + realistic_shapes)
    )
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        M, N = shape
        x = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
        weight = torch.randn([N], device="cuda", dtype=torch.bfloat16)

        def helion_call() -> torch.Tensor:
            return rms_norm(x, weight)

        base_calls = [(name, (lambda fn=fn: fn(x, weight))) for name, fn in baselines]
        return helion_call, base_calls, f"{M:>8d}  {N:>6d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>8s}  {'N':>6s}",
    )


if __name__ == "__main__":
    main()
