"""CuTe binds kernels whose slice bounds are traced under dynamic shapes.

A slice bound on a dynamic dim is clamped at runtime (``normalize_slice``):
``y[:4]`` loads ``y[:torch.sym_min(y.size(0), 4)]``, so the load or store
keeps the bound as an FX node inside its slice.  The fragment analyses that
the autotuner facts run at bind for every root (register snapshots,
lane-private loads, local atomic registers, warp results) must read the
node's value before computing the subscript's shape.
"""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render]


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _literal_head(y: torch.Tensor) -> torch.Tensor:
    out = y.new_empty(4)
    for _ in hl.tile(1):
        out[:] = y[:4]
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _specialized_head(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(0))
    out = torch.empty_like(x)
    for _ in hl.tile(1):
        out[:] = x[:] + y[:n]
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _scale_by_weight(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    width = hl.specialize(width)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        out[tile_r, :] = x[tile_r, :] * weight[:width]
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _scale_by_dynamic_weight(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        out[tile_r, :] = x[tile_r, :] * weight[:width]
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _store_head(y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for _ in hl.tile(1):
        out[:4] = y[:4]
    return out


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        pytest.param(_literal_head, (torch.randn(8),), id="y[:4]"),
        pytest.param(
            _specialized_head, (torch.randn(64), torch.randn(128)), id="y[:n]"
        ),
        pytest.param(
            _scale_by_weight,
            (torch.randn(32, 64), torch.randn(80)),
            id="weight[:width]",
        ),
        pytest.param(
            _scale_by_dynamic_weight,
            (torch.randn(32, 64), torch.randn(80)),
            id="weight[:dynamic_width]",
        ),
        # The snapshot proof also walks to the store through a bounded slice.
        pytest.param(_store_head, (torch.randn(8), torch.randn(6)), id="out[:4]"),
    ],
)
def test_traced_slice_bound_binds_and_renders(kernel, args) -> None:
    bound = cpu_bind(kernel, args)
    code = bound.to_code(bound.config_spec.default_config())
    ast.parse(code)
    if kernel is _literal_head:
        # The literal stop stays clamped to the dynamic length at runtime.
        assert "min(4, y.size(0))" in code


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _local_atomic_tickets(tickets: torch.Tensor, n: hl.constexpr) -> torch.Tensor:
    for row in hl.grid(tickets.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        columns = hl.arange(n)
        previous = hl.atomic_add(local, [columns % 17], torch.ones_like(columns))
        tickets[row, :n] = previous
    return tickets


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _warp_counts(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    groups = hl.specialize(x.size(1)) // 32
    for row in hl.grid(x.size(0)):
        sums = x[row, :].reshape(groups, 32).sum(-1, dtype=torch.int32)
        counts[row, :groups] = sums
    return counts


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _lane_private_copy(
    x: torch.Tensor, copy: torch.Tensor, n: hl.constexpr
) -> torch.Tensor:
    histogram = torch.empty((x.size(0), 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.float32)
        columns = hl.arange(n)
        values = hl.load(x, [row, columns], extra_mask=columns < x.size(1))
        hl.atomic_add(local, [columns % 17], values)
        copy[row, :n] = values + 1
        histogram[row, :] = local
    return histogram


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        pytest.param(
            _local_atomic_tickets,
            (torch.zeros(2, 40, dtype=torch.int32), 32),
            id="local_atomic_registers",
        ),
        pytest.param(
            _warp_counts,
            (
                torch.ones(2, 64, dtype=torch.int32),
                torch.zeros(2, 4, dtype=torch.int32),
            ),
            id="warp_results",
        ),
        pytest.param(
            _lane_private_copy,
            (torch.randn(2, 30), torch.randn(2, 40), 32),
            id="register_loads",
        ),
    ],
)
def test_traced_slice_bound_store_binds(kernel, args) -> None:
    # Each chain ends in a store through a bounded slice of a dynamic dim; the
    # fragment knob facts walk it when the kernel binds.  These roots lower
    # through the computed fragment, whose host windows are full slices only:
    # rendering refuses them cleanly instead of failing inside codegen.
    bound = cpu_bind(kernel, args)
    with pytest.raises(helion.exc.BackendUnsupported, match="bounded slice"):
        bound.to_code(bound.config_spec.default_config())


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestTracedSliceBoundsOnTheGpu(TestCase):
    def test_weight_prefix_matches_eager(self) -> None:
        """``weight[:width]`` keeps its clamped bound as a traced node; the
        kernel binds, runs and matches eager."""
        x = torch.randn(33, 64, device=DEVICE)
        weight = torch.randn(80, device=DEVICE)
        torch.testing.assert_close(
            _scale_by_weight(x, weight), x * weight[:64], atol=0, rtol=0
        )
