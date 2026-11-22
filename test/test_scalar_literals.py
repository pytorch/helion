"""Scalar operands of half-precision ops render as literals on a fake CUDA
device: the kernel's device, not a real tensor, decides how Inductor lowers
the scalar, so the code renders on CPU."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["triton"])]


@helion.kernel(backend="triton", static_shapes=True)
def _keep_above(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size()):
        out[rows, cols] = torch.where(x[rows, cols] > 0.1, x[rows, cols], 0.0)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compare_scalar_is_a_literal_on_cuda(dtype: torch.dtype) -> None:
    # Inductor rounds the scalar of a bf16/fp16 comparison to the tensor's
    # dtype on the host; under fake-tensor mode on a non-CPU device that
    # gave an unbacked symbol (a GuardOnDataDependentSymNode error).
    with FakeTensorMode():
        x = torch.empty(16, 64, dtype=dtype, device="cuda")
    code = _keep_above.bind((x,)).to_code(helion.Config(block_sizes=[16, 64]))
    assert "zuf" not in code, code
    rounded = torch.tensor(0.1, dtype=dtype).item()
    assert f"tl.full([], {rounded!r}, tl.float32)" in code, code
