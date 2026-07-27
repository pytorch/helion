"""Element-wise vector addition."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.aot_kernel()
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def _vector_add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    No torch.compile baseline: the reference is a single memory-bandwidth-bound
    elementwise add, which torch.compile cannot fuse or speed up -- a redundant,
    not-faster baseline.
    """
    return [("torch", _vector_add_torch)]


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    shapes = [2**i for i in range(19, 29)]
    baselines = _baselines()

    def make_calls(n: int) -> tuple:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        def helion_call() -> torch.Tensor:
            return vector_add(x, y)

        base_calls = [(name, (lambda fn=fn: fn(x, y))) for name, fn in baselines]
        return helion_call, base_calls, f"{n:>10d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        warmup=50,
        rep=200,
        verbose=verbose,
        shape_header=f"{'N':>10s}",
    )


if __name__ == "__main__":
    main()
