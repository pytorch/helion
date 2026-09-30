"""Sorted row-wise top-k, optionally fused with softmax over the selected logits.

The GB300 configs cover BF16 inputs with 65536 rows, widths 64 through 1024,
and K in {8, 16, 32}. Values retain the input dtype and indices use int32.
With ``softmax=True``, normalization uses FP32 arithmetic over the selected K
logits, then rounds the probabilities to BF16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# The policy is part of the recorded full-autotune configuration schema.
STRUCTURAL_POLICY = helion.CuteStructuralPolicy(
    cute_region_fission=True,
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_flatten_nested_reductions=True,
    cute_materialize_transformed_operands=True,
)


@helion.aot_kernel(
    backend="cute", static_shapes=True, cute_structural_policy=STRUCTURAL_POLICY
)
def topk(
    x: torch.Tensor,
    k: int,
    softmax: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the largest K values in descending order and their column indices."""
    rows = x.size(0)
    k = hl.specialize(k)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int32, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=True, sorted=True)
        if softmax:
            vals = torch.softmax(vals.to(torch.float32), dim=-1).to(x.dtype)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


SHAPES = [
    (65536, n, k, softmax)
    for softmax in (False, True)
    for n in (64, 128, 256, 512, 1024)
    for k in (8, 16, 32)
]  # (M, N, K, softmax)


def _topk_torch(
    x: torch.Tensor,
    k: int,
    softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    if softmax:
        values = torch.softmax(values.float(), dim=-1).to(x.dtype)
    return values, indices.to(torch.int32)


def check_output(
    x: torch.Tensor,
    k: int,
    softmax: bool,
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Check values and selected columns while allowing different valid tie orders."""
    values, indices = output
    assert values.shape == indices.shape == (x.size(0), k)
    assert values.dtype == x.dtype
    assert indices.dtype == torch.int32
    assert values.device == indices.device == x.device
    assert bool(((indices >= 0) & (indices < x.size(1))).all())
    ordered_indices = indices.sort(dim=-1).values
    assert bool((ordered_indices[:, 1:] != ordered_indices[:, :-1]).all())

    expected_logits = torch.topk(x, k, dim=-1, largest=True, sorted=True).values
    selected_logits = x.gather(1, indices.long())
    torch.testing.assert_close(selected_logits, expected_logits, rtol=0, atol=0)
    if softmax:
        expected = torch.softmax(expected_logits.float(), dim=-1)
        # BF16 has a maximum relative rounding error of 1/256. Compare against
        # FP32 probabilities to avoid counting two rounded references as error.
        torch.testing.assert_close(values.float(), expected, rtol=0.004, atol=1e-7)
        row_sums = values.float().sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), rtol=0, atol=0.004
        )
        assert bool(torch.isfinite(values).all())
        assert bool((values >= 0).all())
        assert bool((values[:, 1:] <= values[:, :-1]).all())
    else:
        torch.testing.assert_close(values, expected_logits, rtol=0, atol=0)


def _check_case(x: torch.Tensor, k: int, softmax: bool) -> None:
    original = x.clone()
    output = topk(x, k, softmax)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    check_output(original, k, softmax, output)


def use_cudagraph() -> bool:
    """Benchmark both implementations under CUDA graphs with cold L2."""
    return True


def correctness_check() -> None:
    """Check all pretuned shapes, including BF16 ties and fused softmax."""
    torch.manual_seed(0)
    for m, n, k, softmax in SHAPES:
        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        _check_case(x, k, softmax)


@dataclass
class _CapturedCall:
    """Keep input and output storage alive for the captured graph's lifetime."""

    graph: torch.cuda.CUDAGraph
    output: tuple[torch.Tensor, torch.Tensor]
    x: torch.Tensor

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.graph.replay()
        return self.output


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import capture_cuda_graph  # pyrefly: ignore[missing-import]
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    def make_calls(shape: tuple[int, int, int, bool]) -> tuple:
        m, n, k, softmax = shape
        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        original = x.clone()

        def helion_call() -> tuple[torch.Tensor, torch.Tensor]:
            return topk(x, k, softmax)

        def torch_call() -> tuple[torch.Tensor, torch.Tensor]:
            return _topk_torch(x, k, softmax)

        def capture_checked(
            call: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
            graph, output = capture_cuda_graph(call)
            # Verify the captured graph actually writes both output tensors.
            output[0].fill_(float("nan"))
            output[1].fill_(-1)
            graph.replay()
            torch.testing.assert_close(x, original, rtol=0, atol=0)
            check_output(original, k, softmax, output)
            return _CapturedCall(graph, output, x)

        return (
            capture_checked(helion_call),
            [("torch", capture_checked(torch_call))],
            f"{m:>5d}  {n:>5d}  {k:>3d}  {softmax!s:>7s}",
        )

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        rep=100,
        thermal_warmup_ms=1000,
        verbose=verbose,
        shape_header=f"{'M':>5s}  {'N':>5s}  {'K':>3s}  {'softmax':>7s}",
    )


if __name__ == "__main__":
    main()
