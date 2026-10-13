"""Top-k routing weights from a full-row softmax, with a measured GB300 config.

The supported FP32 recipe has 32768 tokens, 256 experts and K=8. Its fixed
ten-argument call disables grouping, renormalization, bias and softcap, with
scale=1. Selected weights retain their full-row probability mass rather than
summing to one.

The adjacent AOT module contains the measured configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


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
def moe_softmax_routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    grouped: hl.constexpr,
    groups: hl.constexpr,
    selected_groups: hl.constexpr,
    renormalize: hl.constexpr,
    scale: hl.constexpr,
    softcap: hl.constexpr,
    use_bias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = logits.size(0)
    k = hl.specialize(k)
    weights = torch.empty((rows, k), device=logits.device, dtype=torch.float32)
    indices = torch.empty((rows, k), device=logits.device, dtype=torch.int32)
    for row in hl.tile(rows):
        x = logits[row, :].to(torch.float32)
        if softcap != 0:
            x = torch.tanh(x / softcap) * softcap  # pyrefly: ignore[unsupported-operation]
        if use_bias:
            x = x + bias[:]  # pyrefly: ignore[unsupported-operation]
        probabilities = torch.softmax(x, dim=-1)  # pyrefly: ignore[bad-argument-type]
        w, idx = torch.topk(probabilities, k, dim=-1, sorted=True)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        weights[row, :] = w
        indices[row, :] = idx
    return (weights, indices)


SHAPES = [(32768, 256, 8)]  # tokens, experts, K


def make_inputs(shape: tuple[int, int, int], device: str = "cuda") -> tuple:
    """Fixed full-row softmax: no grouping, renormalization, bias or softcap."""
    if shape not in SHAPES:
        raise ValueError(f"No measured MoE softmax routing recipe for {shape}")
    rows, experts, k = shape
    generator = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(
        rows, experts, device=device, dtype=torch.float32, generator=generator
    )
    bias = torch.randn(experts, device=device, dtype=torch.float32, generator=generator)
    return logits, bias, k, False, 1, 1, False, 1.0, 0.0, False


def _torch_reference(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probabilities, k, dim=-1, sorted=True)
    return weights, indices.to(torch.int32)


def check_output(
    logits: torch.Tensor, k: int, output: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Check tie-valid selection and weights from the complete softmax mass."""
    weights, indices = output
    assert weights.shape == indices.shape == (logits.size(0), k)
    assert weights.dtype == torch.float32 and indices.dtype == torch.int32
    assert weights.device == indices.device == logits.device
    assert bool(((indices >= 0) & (indices < logits.size(1))).all())
    sorted_indices = indices.sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all())
    probabilities = torch.softmax(logits.float(), dim=-1)
    expected = probabilities.topk(k, dim=-1, sorted=True).values
    selected = probabilities.gather(1, indices.long())
    # Compare the entire selected probability multiset, not arbitrary tie IDs.
    torch.testing.assert_close(
        selected.sort(dim=-1, descending=True).values, expected, rtol=0, atol=0
    )
    torch.testing.assert_close(weights, selected, rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(weights, expected, rtol=3e-5, atol=2e-6)
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    actual_mass, expected_mass = weights.sum(dim=-1), expected.sum(dim=-1)
    torch.testing.assert_close(actual_mass, expected_mass, rtol=3e-5, atol=k * 2e-6)
    # This is selected mass from the full softmax, not a renormalized top-k.
    assert bool((actual_mass <= 1 + k * 2e-6).all())


@dataclass
class _CapturedCall:
    graph: torch.cuda.CUDAGraph
    output: tuple[torch.Tensor, torch.Tensor]
    logits: torch.Tensor
    bias: torch.Tensor

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.graph.replay()
        return self.output


def check_case(shape: tuple[int, int, int]) -> tuple:
    """Validate direct and poisoned graph calls outside the measured interval."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import capture_cuda_graph  # pyrefly: ignore[missing-import]

    args = make_inputs(shape)
    logits, bias, k = args[:3]
    original_logits, original_bias = logits.clone(), bias.clone()

    def helion_call() -> tuple[torch.Tensor, torch.Tensor]:
        return moe_softmax_routing(*args)

    def torch_call() -> tuple[torch.Tensor, torch.Tensor]:
        return _torch_reference(logits, k)

    def check(output: tuple[torch.Tensor, torch.Tensor]) -> None:
        torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
        torch.testing.assert_close(bias, original_bias, rtol=0, atol=0)
        check_output(original_logits, k, output)

    def capture_checked(
        call: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ) -> _CapturedCall:
        check(call())
        graph, output = capture_cuda_graph(call)
        output[0].fill_(float("nan"))
        output[1].fill_(-1)
        graph.replay()
        check(output)
        return _CapturedCall(graph, output, logits, bias)

    return (
        capture_checked(helion_call),
        [("torch", capture_checked(torch_call))],
        f"{shape[0]:>6d}  {shape[1]:>7d}  {shape[2]:>3d}",
    )


def use_cudagraph() -> bool:
    return True


def correctness_check() -> None:
    for shape in SHAPES:
        check_case(shape)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    return run_sweep(
        SHAPES,
        check_case,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        rep=100,
        thermal_warmup_ms=1000,
        verbose=verbose,
        shape_header=f"{'tokens':>6s}  {'experts':>7s}  {'K':>3s}",
    )


if __name__ == "__main__":
    main()
