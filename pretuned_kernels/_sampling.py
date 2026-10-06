"""Bounded CDF accuracy, separate from distribution validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar

import torch

from helion._compiler.rng_utils import UINT32_TO_UNIFORM_SCALE
from helion._compiler.rng_utils import philox_int32_ref
from helion._compiler.rng_utils import philox_rand_ref

if TYPE_CHECKING:
    from collections.abc import Callable

CDF_FP32_NEIGHBORS = 8


def philox_uniform(seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Exact word0 mapping with device seed reads and graph-safe scalar scaling.

    Keep the existing Philox recurrence; a Python scalar multiplier avoids the
    reference helper's pageable host-to-device upload of a CUDA scale tensor.
    """
    signed = philox_int32_ref(seed, offset).to(torch.int64)
    magnitude = torch.where(signed < 0, -signed - 1, signed)
    return magnitude.to(torch.float32) * UINT32_TO_UNIFORM_SCALE


@dataclass(frozen=True)
class CDFReference:
    first_allowed: torch.Tensor
    last_allowed: torch.Tensor
    positive_support: torch.Tensor
    exact_indices: torch.Tensor
    uniform_lower: torch.Tensor
    uniform_upper: torch.Tensor
    index_dtype: torch.dtype


def probability_window(uniform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight representable FP32 neighbors in each direction, clipped to [0, 1]."""
    assert uniform.dtype == torch.float32
    assert bool(((uniform >= 0) & (uniform < 1)).all())
    lower = uniform.clone()
    upper = uniform.clone()
    zero = torch.zeros_like(uniform)
    one = torch.ones_like(uniform)
    for _ in range(CDF_FP32_NEIGHBORS):
        lower = torch.nextafter(lower, zero)
        upper = torch.nextafter(upper, one)
    return lower.double(), upper.double()


def reference_from_weights(
    weights: torch.Tensor,
    uniform: torch.Tensor,
    index_dtype: torch.dtype,
) -> CDFReference:
    assert weights.dtype == torch.float64 and weights.ndim == 2
    assert uniform.shape == (weights.size(0),)
    assert index_dtype in (torch.int32, torch.int64)
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    cumulative_mass = weights.cumsum(-1)
    assert bool((cumulative_mass[:, -1] > 0).all())
    # Normalize using the same high-precision terminal sum to make the final
    # endpoint exactly one; no low-precision candidate intermediates are used.
    cdf = cumulative_mass / cumulative_mass[:, -1:]
    lower, upper = probability_window(uniform)
    first = torch.searchsorted(cdf, lower[:, None], right=True).squeeze(-1)
    last = (
        torch.searchsorted(cdf, upper[:, None], right=True)
        .squeeze(-1)
        .clamp_max(weights.size(1) - 1)
    )
    exact = torch.searchsorted(cdf, uniform.double()[:, None], right=True).squeeze(-1)
    return CDFReference(
        first,
        last,
        weights > 0,
        exact.to(index_dtype),
        lower,
        upper,
        index_dtype,
    )


def torch_reference(
    x: torch.Tensor,
    seed: torch.Tensor,
    vocab: int,
    logits: bool,
    index_dtype: torch.dtype,
) -> CDFReference:
    values = x.reshape(-1, vocab).double()
    if logits:
        # Unnormalized FP64 masses suffice; the CDF performs normalization.
        weights = (values - values.amax(-1, keepdim=True)).exp()
    else:
        weights = values
    uniform = philox_rand_ref(
        seed[0], torch.arange(values.size(0), device=x.device, dtype=torch.int64)
    )
    return reference_from_weights(weights, uniform, index_dtype)


def assert_cdf_accuracy(actual: object, expected: object) -> None:
    """Existing Settings callback ABI: raise AssertionError on any invalid row."""
    assert isinstance(actual, torch.Tensor), "CDF result must be a tensor"
    assert isinstance(expected, CDFReference), "CDF reference object required"
    assert actual.dtype == expected.index_dtype, "CDF index dtype mismatch"
    assert actual.device == expected.first_allowed.device, "CDF device mismatch"
    assert actual.shape == expected.first_allowed.shape, "CDF result shape mismatch"
    vocab = expected.positive_support.size(1)
    assert bool(((actual >= 0) & (actual < vocab)).all()), "CDF index out of range"
    supported = expected.positive_support.gather(1, actual.long()[:, None]).squeeze(-1)
    assert bool(supported.all()), "CDF result has zero probability"
    valid = (actual >= expected.first_allowed) & (actual <= expected.last_allowed)
    assert bool(valid.all()), (
        "CDF interval misses the eight-neighbor probability window"
    )


OutputT = TypeVar("OutputT", bound=torch.Tensor | tuple[torch.Tensor, ...])


@dataclass
class CapturedCall(Generic[OutputT]):
    """Own all inputs and outputs for the lifetime of a validated graph."""

    graph: torch.cuda.CUDAGraph
    output: OutputT
    inputs: tuple[torch.Tensor, ...]

    def __call__(self) -> OutputT:
        self.graph.replay()
        return self.output


def _tensors(
    value: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return (value,) if isinstance(value, torch.Tensor) else value


def capture_checked(
    call: Callable[[], OutputT],
    inputs: tuple[torch.Tensor, ...],
    seed: torch.Tensor,
    validate_output: Callable[[OutputT], None],
) -> CapturedCall[OutputT]:
    """Check graph writes, immutable inputs and a seed changed after capture."""
    from pretuned_kernels._bench import capture_cuda_graph

    originals = tuple(value.clone() for value in inputs)
    original_seed = seed.clone()
    graph, output = capture_cuda_graph(call)
    for value, original in zip(inputs, originals, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    first = None
    for index in range(3):
        if index == 1:
            seed.fill_(8147)
        else:
            seed.copy_(original_seed)
        expected_seed = seed.clone()
        for value in _tensors(output):
            value.fill_(-1)
        graph.replay()
        for value, original in zip(inputs, originals, strict=True):
            torch.testing.assert_close(
                value, expected_seed if value is seed else original, rtol=0, atol=0
            )
        validate_output(output)
        if index == 0:
            first = tuple(value.clone() for value in _tensors(output))
        if index == 2:
            assert first is not None
            for value, original in zip(_tensors(output), first, strict=True):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
    return CapturedCall(graph, output, inputs)


def cdf_sample(weights: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Torch CDF selection; the caller includes Philox in its timed boundary."""
    cumulative = weights.double().cumsum(-1)
    cumulative = cumulative / cumulative[:, -1:]
    return (
        torch.searchsorted(cumulative, uniform.double()[:, None], right=True)
        .squeeze(-1)
        .to(torch.int32)
    )
