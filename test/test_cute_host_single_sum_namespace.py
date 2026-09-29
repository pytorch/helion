"""Generated callers must preserve local names that shadow the new helper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_host_paired_sum import _config
from test.test_cute_host_paired_sum_seeds import _population

import helion
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _argument(
    _cute_try_single_sum_cast: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(_cute_try_single_sum_cast)
    for i, j in hl.tile(partial.shape):
        partial[i, j] = _cute_try_single_sum_cast[i, j]
    return _cute_try_single_sum_cast, partial.sum(0).to(torch.float16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _earlier_local(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _cute_try_single_sum_cast = a
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    return _cute_try_single_sum_cast, partial.sum(0).to(torch.float16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _later_local(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    _cute_try_single_sum_cast = partial
    return a, _cute_try_single_sum_cast.sum(0).to(torch.float16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cute_try_single_sum_cast(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    return a, partial.sum(0).to(torch.float16)


@pytest.mark.parametrize(
    "kernel", (_argument, _earlier_local, _later_local, _cute_try_single_sum_cast)
)
def test_collision_retains_real_caller_fallback_and_never_seeds(
    kernel: helion.Kernel,
) -> None:
    initial_cuda_state = torch.cuda.is_initialized()
    original = torch.arange(17 * 20, dtype=torch.float32).view(17, 20)
    bound = _cpu_bind(kernel, (original,))
    assert not bound.config_spec.cute_host_paired_sum_available
    assert all(
        "cute_host_paired_sum" not in config
        for config in _population(bound, (original,))
    )
    for layout in ("mapped", "narrow"):
        with pytest.raises(InvalidConfig, match="terminal sum/cast"):
            bound.to_code(_config(bound, layout))
    bound.set_config(_config(bound, "off"))
    host = bound._run
    assert host is not None
    calls = []

    def producer(kernel, grid, *values, **kwargs):
        values[1].copy_(values[0])
        calls.append(id(kernel))

    inputs = [original, original.clone()]
    with patch.dict(host.__kwdefaults__, {"_launcher": producer}):
        outputs = [bound(value) for value in inputs]
    for output, value in zip(outputs, inputs, strict=True):
        assert output[0] is value
        assert torch.equal(output[1], value.sum(0).to(torch.float16))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert outputs[0][1].data_ptr() != outputs[1][1].data_ptr()
    assert bound._run is host
    assert torch.cuda.is_initialized() == initial_cuda_state
