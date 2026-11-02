"""Persistent-row packets through host-side views of an input.

A view such as ``x.view(rows, width)`` has no input source of its own, but it
keeps the input's runtime base pointer.  Packet loads and stores through it
are only aligned when that base is, so a caller's ``buf[1:]`` must fall back
to scalar accesses instead of faulting with a misaligned address.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _center_into_view(x: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    rows = destination.view(x.size(0), x.size(1))
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(1))
        data = x[row, columns].float()
        rows[row, columns] = (data - data.amax(-1)[:, None]).to(destination.dtype)
    return destination


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _center_view(x: torch.Tensor, width: hl.constexpr) -> torch.Tensor:
    source = x.view(-1, width)
    out = torch.empty(source.shape, dtype=x.dtype, device=x.device)
    for row in hl.tile(source.size(0)):
        columns = hl.arange(width)
        data = source[row, columns].float()
        out[row, columns] = (data - data.amax(-1)[:, None]).to(x.dtype)
    return out


def _config(bound: BoundKernel) -> helion.Config:
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    spec = bound.config_spec
    return helion.Config(
        block_sizes=[16],
        num_threads=[
            4 if block == reduction else 8
            for block in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            8 if block == reduction else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_lane_layouts=[
            "strided" if block == reduction else "blocked"
            for block in spec.cute_lane_layouts.valid_block_ids()
        ],
    )


def _flat(offset: int, device: torch.device | str | None = None) -> torch.Tensor:
    # ``offset`` elements into a fresh allocation: 2 bytes for offset 1.
    return torch.randn(37 * 32 + offset, dtype=torch.bfloat16, device=device)[offset:]


@pytest.mark.parametrize("offset", [0, 1])
def test_store_through_input_view_requires_an_aligned_input_base(offset: int) -> None:
    x = torch.empty((37, 32), dtype=torch.bfloat16)
    destination = _flat(offset)
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_center_into_view, (x, destination))
        code = bound.to_code(_config(bound))
    assert ("_cute_store_u16_vec(" in code) is (offset == 0), code


@pytest.mark.parametrize("offset", [0, 1])
def test_load_through_input_view_requires_an_aligned_input_base(offset: int) -> None:
    flat = _flat(offset)
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_center_view, (flat, 32))
        code = bound.to_code(_config(bound))
    packet_loads = [
        line
        for line in code.splitlines()
        if "source.iterator" in line and "ir.VectorType.get(" in line
    ]
    assert bool(packet_loads) is (offset == 0), code


@skipIfNotCUDA()
def test_misaligned_input_views_match_eager() -> None:
    x = torch.randn((37, 32), dtype=torch.bfloat16, device=DEVICE)
    destination = _flat(1, DEVICE)
    bound = _center_into_view._bind_isolated((x, destination))
    bound.compile_config(_config(bound))(x, destination)
    expected = (x.float() - x.float().amax(-1, keepdim=True)).to(x.dtype)
    torch.testing.assert_close(destination.view(37, 32), expected, rtol=0, atol=0)

    flat = _flat(1, DEVICE)
    bound = _center_view._bind_isolated((flat, 32))
    result = bound.compile_config(_config(bound))(flat, 32)
    source = flat.view(-1, 32).float()
    expected = (source - source.amax(-1, keepdim=True)).to(flat.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
