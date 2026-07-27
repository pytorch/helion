"""Cross entropy loss for token classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


@helion.aot_kernel(
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.empty([n], dtype=logits.dtype, device=logits.device)
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


def _cross_entropy_torch(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    return [
        ("torch", _cross_entropy_torch),
        ("torch_compile", torch.compile(_cross_entropy_torch)),
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
        (2048, 32000),
        (4096, 32000),
        (8192, 32000),
        (8192, 128000),
        (16384, 128000),
        (32768, 128000),
    ]
    realistic_shapes = [
        (2048, 128256),
        (4096, 128256),
        (8192, 128256),
        (16384, 128256),
        (2048, 129280),
        (4096, 129280),
        (8192, 129280),
        (2048, 151936),
        (4096, 151936),
        (8192, 151936),
        (2048, 152064),
        (4096, 152064),
        (8192, 152064),
        (1024, 256000),
        (2048, 256000),
    ]
    shapes = list(dict.fromkeys(tritonbench_shapes + realistic_shapes))
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        tokens, vocab = shape
        logits = torch.randn([tokens, vocab], device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, [tokens], device="cuda", dtype=torch.int64)

        def helion_call() -> torch.Tensor:
            return cross_entropy(logits, labels)

        base_calls = [
            (name, (lambda fn=fn: fn(logits, labels))) for name, fn in baselines
        ]
        return helion_call, base_calls, f"{tokens:>8d}  {vocab:>8d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'tokens':>8s}  {'vocab':>8s}",
    )


if __name__ == "__main__":
    main()
