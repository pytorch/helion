"""Read-only host metadata keeps output allocation proofs without hiding effects.

Covers ``hl.constexpr`` metadata and read-only expanded views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import vector_scaled_mm

import helion
from helion._compiler.cute.promote_output_axis import _fresh_tensors
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language.constexpr import ConstExpr as literal_metadata

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _boolean_metadata(x: torch.Tensor) -> torch.Tensor:
    scalar_mode = hl.constexpr(x.ndim == 1)
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        if scalar_mode:
            out[rows] = x[rows]
        else:
            out[rows] = x[rows] + 1
    return out


def _shape_metadata(x: torch.Tensor) -> torch.Tensor:
    shape = hl.constexpr(x.shape)
    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    for rows in hl.tile(x.size(0)):
        out[rows] = x[rows]
    return out


def _aliased_metadata(x: torch.Tensor) -> torch.Tensor:
    scalar_mode = literal_metadata(x.ndim == 1)
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        if scalar_mode:
            out[rows] = x[rows]
    return out


def _tensor_argument(x: torch.Tensor) -> torch.Tensor:
    metadata = hl.constexpr(x)
    out = torch.empty_like(metadata)
    for rows in hl.tile(x.size(0)):
        out[rows] = x[rows]
    return out


def _opaque_container(x: torch.Tensor) -> torch.Tensor:
    metadata = hl.constexpr({"tensor": x})
    out = torch.empty_like(metadata["tensor"])
    for rows in hl.tile(x.size(0)):
        out[rows] = x[rows]
    return out


def _rebind_and_flag(out: torch.Tensor, x: torch.Tensor) -> bool:
    out.set_(x)
    return x.ndim == 1


def _nested_mutation(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    scalar_mode = hl.constexpr(_rebind_and_flag(out, x))
    for rows in hl.tile(x.size(0)):
        if scalar_mode:
            out[rows] = x[rows]
    return out


def _later_rebind(x: torch.Tensor) -> torch.Tensor:
    scalar_mode = hl.constexpr(x.ndim == 1)
    out = torch.empty_like(x)
    out.set_(x)
    for rows in hl.tile(x.size(0)):
        if scalar_mode:
            out[rows] = x[rows]
    return out


def _later_write(x: torch.Tensor) -> torch.Tensor:
    scalar_mode = hl.constexpr(x.ndim == 1)
    out = torch.empty_like(x)
    out[0] = x[0]
    for rows in hl.tile(x.size(0)):
        if scalar_mode:
            out[rows] = x[rows]
    return out


def _expanded_then_rebound(a: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    m, n = replacement.shape
    expanded = a.reshape(-1, 1).expand(m, n)
    out = torch.empty_like(replacement)
    out.set_(replacement)
    for rows, cols in hl.tile((m, n)):
        out[rows, cols] = expanded[rows, cols]
    return out


def _expanded_then_host_write(
    a: torch.Tensor, replacement: torch.Tensor
) -> torch.Tensor:
    m, n = replacement.shape
    expanded = a.reshape(-1, 1).expand(m, n)
    out = torch.empty_like(replacement)
    out[0, 0] = replacement[0, 0]
    for rows, cols in hl.tile((m, n)):
        out[rows, cols] = expanded[rows, cols]
    return out


def _fresh(function: Callable[..., torch.Tensor], *args: torch.Tensor) -> set:
    kernel = helion.kernel(
        function, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated(args)
    with bound.env, bound.host_function:
        return _fresh_tensors(bound.host_function)


@pytest.mark.parametrize(
    "function", (_boolean_metadata, _shape_metadata, _aliased_metadata)
)
def test_constexpr_metadata_preserves_fresh_output(
    function: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    fresh = _fresh(function, torch.empty(128))
    assert len(fresh) == 1
    assert next(iter(fresh)).shape == (128,)


@pytest.mark.parametrize(
    "function",
    (
        _tensor_argument,
        _opaque_container,
        _nested_mutation,
        _later_rebind,
        _later_write,
    ),
)
def test_constexpr_does_not_admit_opaque_values_or_host_mutation(
    function: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    assert not _fresh(function, torch.empty(128))


@pytest.mark.parametrize(
    ("shape", "tile"),
    (
        ((64, 4096, 4096), (64, 32, 128)),
        ((128, 5120, 2880), (64, 128, 64)),
        ((256, 2048, 4096), (64, 32, 128)),
        ((4096, 2880, 4096), (128, 64, 128)),
    ),
)
def test_expand_preserves_tma_output_proof(
    shape: tuple[int, int, int], tile: tuple[int, int, int]
) -> None:
    bound = vector_scaled_mm._bind_isolated(scaled_args(*shape))
    assert bound.config_spec.cute_tcgen05_matmul_operands_tma_provable
    source = bound.to_code(
        helion.Config(
            block_sizes=list(tile),
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_ab_stages=2,
        )
    )
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source


@pytest.mark.parametrize(
    "function", (_expanded_then_rebound, _expanded_then_host_write)
)
def test_expand_does_not_hide_later_host_mutation(
    function: Callable[..., torch.Tensor],
) -> None:
    assert not _fresh(function, torch.empty(128), torch.empty((128, 64)))
