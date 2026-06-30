"""Cross entropy loss for token classification."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton.testing as tt

import helion
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel(
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


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    def _p(*args: object) -> None:
        if verbose:
            print(*args)

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

    _p(f"GPU: {torch.cuda.get_device_name()}")
    _p(
        f"{'tokens':>8s}  {'vocab':>8s}  {'helion (ms)':>12s}  "
        f"{'torch (ms)':>12s}  {'speedup':>8s}"
    )
    _p("-" * 67)

    speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    best_shape = (0, 0)
    for tokens, vocab in shapes:
        logits = torch.randn([tokens, vocab], device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, [tokens], device="cuda", dtype=torch.int64)
        cross_entropy(logits, labels)  # warmup
        ms_helion = tt.do_bench(
            lambda logits=logits, labels=labels: cross_entropy(logits, labels),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        ms_torch = tt.do_bench(
            lambda logits=logits, labels=labels: F.cross_entropy(logits, labels),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        speedup = ms_torch / ms_helion if ms_helion > 0 else float("nan")
        speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_shape = (tokens, vocab)
        _p(
            f"{tokens:>8d}  {vocab:>8d}  {ms_helion:>12.4f}  "
            f"{ms_torch:>12.4f}  {speedup:>7.2f}x"
        )

    geomean = math.exp(
        sum(math.log(s) for s in speedups if s > 0) / max(len(speedups), 1)
    )
    _p(
        f"\nHelion faster on {helion_wins}/{len(shapes)} shapes; "
        f"geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (tokens, vocab)={best_shape}."
    )
    return {
        "helion_wins": helion_wins,
        "total": len(shapes),
        "geomean": round(geomean, 4),
        "best_speedup": round(best_speedup, 4),
    }


if __name__ == "__main__":
    main()
