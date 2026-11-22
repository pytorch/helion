"""Scalar operands of half-precision ops in kernels bound for an MPS device.

The inputs are fake MPS tensors, so the code renders on CPU without a Metal
device: the kernel's device, not a real tensor, is what decides how Inductor
lowers the scalar.
"""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

_CONFIG = helion.Config(block_sizes=[16, 64])


@helion.kernel(backend="metal", static_shapes=True)
def _add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size()):
        out[rows, cols] = x[rows, cols] + 1
    return out


@helion.kernel(backend="metal", static_shapes=True)
def _keep_above(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size()):
        out[rows, cols] = torch.where(x[rows, cols] > 0.1, x[rows, cols], 0.0)
    return out


def _fake_mps(dtype: torch.dtype) -> torch.Tensor:
    with FakeTensorMode():
        return torch.empty(16, 64, dtype=dtype, device="mps")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scalar_operands_are_literals_on_mps(dtype: torch.dtype) -> None:
    # Inductor rounds the scalar of a bf16/fp16 add (on MPS) and compare (on
    # every device) to the tensor's dtype on the host; an unbacked symbol
    # (zuf0) in its place left the shader with an undeclared identifier.
    add = _add_one.bind((_fake_mps(dtype),)).to_code(_CONFIG)
    assert "zuf" not in add, add
    assert "v_0 = 1.0" in add, add

    compare = _keep_above.bind((_fake_mps(dtype),)).to_code(_CONFIG)
    assert "zuf" not in compare, compare
    rounded = torch.tensor(0.1, dtype=dtype).item()
    assert f"v_0 = {rounded!r}" in compare, compare
