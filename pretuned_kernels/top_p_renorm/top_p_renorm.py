"""Dense top-p renormalization through a seventeen-launch ordinary AIR pipeline.

The recorded GB300 recipe supports contiguous FP32[16, 128512], scalar p=0.1,
and 32 histogram partitions. It is a measured fixed-config diagnostic, not a
full cold-search winner. Floating atomic accumulation can change the cutoff
within the bounded FP32 ambiguity checked below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import torch

# Support both the repository runner's file loader and direct script execution.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretuned_kernels.top_p_renorm import stages

if TYPE_CHECKING:
    from collections.abc import Callable

SHAPES = [(16, 128512, 0.1)]  # (rows, vocabulary, top-p)


def _stage(name: str, *args: object) -> Any:  # noqa: ANN401
    # The recorded stages have distinct tensor and tuple return signatures.
    return getattr(stages, name)(*args)


def top_p_renorm(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Renormalize a whole-threshold-tie top-p subset of nonnegative probabilities.

    Only the recorded shape and scalar cutoff have saved configurations. The
    histogram uses p times the input row mass, preserving the recorded AIR
    arithmetic even when a normalized input sums to one only up to rounding.
    """
    if (
        probs.shape != (16, 128512)
        or probs.dtype != torch.float32
        or not probs.is_contiguous()
        or not isinstance(p, float)
        or p != 0.1
    ):
        raise ValueError(
            "Recorded AIR requires contiguous FP32[16,128512], scalar p=0.1"
        )
    partitions = 32
    prefix = torch.empty((probs.shape[0],), dtype=torch.int32, device=probs.device)
    remaining = torch.empty((probs.shape[0],), dtype=torch.float32, device=probs.device)
    current_count = torch.empty_like(prefix)
    previous_count = torch.empty_like(prefix)
    capacity = ((probs.shape[1] // 32 + 255) // 256) * 256
    compact_in = torch.empty(
        (probs.shape[0], capacity), dtype=probs.dtype, device=probs.device
    )
    for pass_id in range(3):
        hist, counts, compact_out, allocated = _stage(
            "histogram",
            probs,
            prefix,
            current_count,
            compact_in,
            previous_count,
            pass_id,
            partitions,
        )
        groups = _stage("histogram_groups", hist)
        prefix, remaining = _stage(
            "choose", hist, counts, groups, prefix, remaining, p, pass_id
        )
        if pass_id < 2:
            next_count = _stage("selected_count", counts, prefix, pass_id)
            previous_count, current_count = current_count, next_count
            compact_in = compact_out
    partials = _stage("apply_partials", probs, prefix)
    mass_groups = _stage("apply_groups", partials)
    return _stage("apply_output", probs, prefix, mass_groups)


def _top_p_torch(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Independent sort/CDF reference with whole equal-value threshold groups."""
    values, order = probs.sort(descending=True, stable=True)
    positions = torch.arange(probs.shape[1], device=probs.device)[None, :]
    starts = torch.cat(
        (
            torch.ones_like(values[:, :1], dtype=torch.bool),
            values[:, 1:] != values[:, :-1],
        ),
        -1,
    )
    group = torch.where(starts, positions, 0).cummax(-1).values
    greater = (values.double().cumsum(-1) - values.double()).gather(1, group)
    target = p * probs.double().sum(-1, keepdim=True)
    keep_sorted = greater < target
    keep = torch.empty_like(keep_sorted).scatter(1, order, keep_sorted)
    weights = torch.where(keep, probs, 0)
    return weights / weights.sum(-1, keepdim=True)


def check_output(probs: torch.Tensor, p: float, actual: torch.Tensor) -> None:
    """Check cutoff support and normalization without assuming atomic order.

    Allow bounded FP32 cutoff ambiguity while checking row mass, normalization,
    and equal-value threshold groups.
    """
    assert actual.shape == probs.shape and actual.dtype == probs.dtype
    assert actual.device == probs.device
    assert bool(torch.isfinite(actual).all()) and bool((actual >= 0).all())
    selected = actual > 0
    sorted_x, order = probs.sort(descending=True, stable=True)
    positions = torch.arange(probs.shape[1], device=probs.device)[None, :]
    start = torch.cat(
        (
            torch.ones_like(sorted_x[:, :1], dtype=torch.bool),
            sorted_x[:, 1:] != sorted_x[:, :-1],
        ),
        -1,
    )
    group = torch.where(start, positions, 0).cummax(-1).values
    greater = (sorted_x.double().cumsum(-1) - sorted_x.double()).gather(1, group)
    cutoff = torch.full((probs.shape[0],), p, device=probs.device)
    lo = (cutoff.double() * probs.double().sum(-1)).float()
    hi = lo.clone()
    for _ in range(8):
        lo = torch.nextafter(lo, torch.zeros_like(lo))
        hi = torch.nextafter(hi, torch.ones_like(hi))
    kept = selected.gather(1, order)
    assert bool((kept | (greater >= lo[:, None]) | (sorted_x == 0)).all())
    assert bool((~kept | (greater < hi[:, None])).all())
    assert bool(
        ((sorted_x[:, 1:] != sorted_x[:, :-1]) | (kept[:, 1:] == kept[:, :-1])).all()
    )
    weights = torch.where(selected, probs, 0)
    expected = weights / weights.sum(-1, keepdim=True)
    torch.testing.assert_close(
        actual, expected, rtol=8 * torch.finfo(torch.float32).eps, atol=0
    )
    torch.testing.assert_close(
        actual.sum(-1),
        torch.ones(probs.shape[0], device=probs.device),
        rtol=2e-6,
        atol=0,
    )


def _make_input(shape: tuple[int, int, float]) -> torch.Tensor:
    if shape not in SHAPES:
        raise ValueError(f"No recorded AIR recipe for {shape}")
    rows, width, p = shape
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((rows, width), device="cuda", generator=generator)
    return logits.softmax(-1)


@dataclass
class _CapturedCall:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    probs: torch.Tensor

    def __call__(self) -> torch.Tensor:
        self.graph.replay()
        return self.output


def _capture_checked(
    call: Callable[[], torch.Tensor], probs: torch.Tensor, p: float
) -> _CapturedCall:
    from pretuned_kernels._bench import capture_cuda_graph

    original = probs.clone()
    graph, output = capture_cuda_graph(call)
    # Capture only records writes; validate after a replay overwrites poison.
    torch.testing.assert_close(probs, original, rtol=0, atol=0)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(probs, original, rtol=0, atol=0)
    check_output(original, p, output)
    return _CapturedCall(graph, output, probs)


def check_case(shape: tuple[int, int, float]) -> None:
    """Check eager calls and full-public-call graph replay for the recorded shape."""
    probs = _make_input(shape)
    p = shape[2]
    original = probs.clone()
    check_output(original, p, top_p_renorm(probs, p))
    torch.testing.assert_close(probs, original, rtol=0, atol=0)
    check_output(original, p, _top_p_torch(probs, p))
    torch.testing.assert_close(probs, original, rtol=0, atol=0)
    _capture_checked(lambda: top_p_renorm(probs, p), probs, p)
    _capture_checked(lambda: _top_p_torch(probs, p), probs, p)


def correctness_check() -> None:
    for shape in SHAPES:
        check_case(shape)


def use_cudagraph() -> bool:
    return True


def main(verbose: bool = True) -> dict:
    from pretuned_kernels._bench import run_sweep

    correctness_check()

    def make_calls(shape: tuple[int, int, float]) -> tuple:
        rows, width, p = shape
        probs = _make_input(shape)
        return (
            _capture_checked(lambda: top_p_renorm(probs, p), probs, p),
            [("torch", _capture_checked(lambda: _top_p_torch(probs, p), probs, p))],
            f"{rows:>5d}  {width:>6d}  {p:>4.1f}",
        )

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        rep=100,
        thermal_warmup_ms=1000,
        verbose=verbose,
        shape_header=f"{'rows':>5s}  {'vocab':>6s}  {'p':>4s}",
    )


if __name__ == "__main__":
    main()
