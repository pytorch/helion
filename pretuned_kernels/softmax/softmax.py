"""Row-wise softmax."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def softmax(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[tile_m, :] = torch.nn.functional.softmax(x[tile_m, :], dim=1)
    return out


def _softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=1)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    return [
        ("torch", _softmax_torch),
        ("torch_compile", torch.compile(_softmax_torch)),
    ]


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    triton_tutorial_shapes = [(4096, 128 * i) for i in range(2, 100)]
    realistic_shapes = [
        (4096, 16384),
        (2048, 32768),
    ]
    shapes = list(dict.fromkeys(triton_tutorial_shapes + realistic_shapes))
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        M, N = shape
        x = torch.randn([M, N], device="cuda", dtype=torch.float16)

        def helion_call() -> torch.Tensor:
            return softmax(x)

        base_calls = [(name, (lambda fn=fn: fn(x))) for name, fn in baselines]
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
