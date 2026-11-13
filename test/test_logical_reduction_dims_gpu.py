from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["triton", "cute"])


@helion.kernel(static_shapes=True)
def _logical_axes(x: torch.Tensor, operation: hl.constexpr) -> torch.Tensor:
    output_dtype = torch.uint8 if x.dtype == torch.uint8 else torch.bool
    out = torch.empty((x.size(0),), device=x.device, dtype=output_dtype)
    for row in hl.tile(x.size(0)):
        values = x[row, :, :]
        if operation == "any":
            out[row] = values.any(dim=(1, 2))
        else:
            out[row] = values.all(dim=(-1, -2), keepdim=True).flatten()
    return out


@helion.kernel(static_shapes=True)
def _logical_full_head(x: torch.Tensor, operation: hl.constexpr) -> torch.Tensor:
    out = torch.empty((x.size(0),), device=x.device, dtype=torch.bool)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        if operation == "any":
            out[row.id] = torch.any(values > 2)
        else:
            out[row.id] = torch.all(values >= 0)
    return out


@helion.kernel(static_shapes=False)
def _logical_unmerged(x: torch.Tensor, operation: hl.constexpr) -> torch.Tensor:
    out = torch.empty((x.size(0),), device=x.device, dtype=torch.bool)
    for row in hl.tile(x.size(0)):
        if operation == "any":
            out[row] = x[row, :, :].any((1, 2))
        else:
            out[row] = x[row, :, :].all((1, 2))
    return out


@pytest.mark.parametrize("dtype", (torch.float32, torch.bool, torch.uint8))
def test_logical_tuple_axes_preserve_tail_masks_and_dtype(dtype):
    values = torch.arange(5 * 7 * 9, device=DEVICE).reshape(5, 7, 9) % 3
    values[0] = 0
    values[1] = 1
    values = values.to(dtype)
    if dtype == torch.float32:
        values[2] = 0
        values[2, 0, 0] = float("nan")
        values[3] = 0
        values[3, -1, -1] = float("inf")
        values[1, 0, 0] = -float("inf")
        values[4] = 1
        values[4, 0, 0] = float("nan")
        values[4, -1, -1] = float("inf")
    # Each reduction owns its loaded producer; shared producer chains have
    # the same layout restrictions as multi-axis numeric reductions.
    for operation, reference in (
        ("any", values.any((1, 2))),
        ("all", values.all((1, 2))),
    ):
        _, actual = code_and_output(_logical_axes, (values, operation), block_sizes=[2])
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("shape", ((5, 1, 9), (5, 9, 1), (5, 1, 1), (5, 0, 9)))
def test_logical_unit_and_empty_axes(shape):
    values = torch.ones(shape, device=DEVICE, dtype=torch.uint8)
    values[0] = 0
    for operation, reference in (
        ("any", values.any((1, 2))),
        ("all", values.all((1, 2))),
    ):
        _, actual = code_and_output(_logical_axes, (values, operation), block_sizes=[2])
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@onlyBackends(["triton"])
def test_dynamic_logical_reductions_execute_unmerged_chains():
    # The CPU test_dynamic_vote_keeps_existing_layout_admission covers CuTe's
    # layout admission. Triton executes both single-axis stages here, detecting
    # incorrect original-overload axes.
    values = torch.ones((5, 7, 9), device=DEVICE)
    values[0] = 0
    values[1, 3, 4] = 0
    values[2] = 0
    values[2, -1, -1] = float("nan")
    for operation, reference in (
        ("any", values.any((1, 2))),
        ("all", values.all((1, 2))),
    ):
        _, actual = code_and_output(
            _logical_unmerged, (values, operation), block_sizes=[2]
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_full_head_vote_reduces_every_lane():
    values = torch.zeros((3, 32, 128), device=DEVICE)
    values[1, -1, -1] = 3
    values[2, 0, 0] = -1
    for operation, reference in (
        ("any", (values > 2).any((1, 2))),
        ("all", (values >= 0).all((1, 2))),
    ):
        _, actual = code_and_output(_logical_full_head, (values, operation))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
