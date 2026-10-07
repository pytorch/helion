from __future__ import annotations

import logging

from examples.broadcast_matmul import broadcast_matmul
from examples.matmul import matmul
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose_a", [False, True])
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_transposed_matmul_reference(dtype: torch.dtype, transpose_a: bool) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(913)
    a = torch.randn((256, 256), device=DEVICE, dtype=dtype, generator=generator)
    b = torch.randn((256, 256), device=DEVICE, dtype=dtype, generator=generator)
    if transpose_a:
        a = a.T
    else:
        b = b.T
    kernel = helion.kernel(matmul.fn, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b))
    compiled = bound.compile_config(bound.config_spec.autotune_reference_config())
    for _ in range(3):
        a.normal_(generator=generator)
        b.normal_(generator=generator)
        expected = (a.float() @ b.float()).to(dtype)
        torch.testing.assert_close(compiled(a, b), expected, rtol=1e-2, atol=1e-1)


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_transposed_matmul_staging_needs_no_thread_barrier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The transposed operand is staged into shared memory by a copy loop on
    # each side of the full-tile branch, every thread writing its own
    # elements.  The cross-thread barrier pass accepts the branch and adds
    # nothing to the emitter's own barriers (the codegen is what it was
    # before the pass existed).
    a = torch.empty((256, 256), device=DEVICE, dtype=torch.float16).T
    b = torch.empty((256, 256), device=DEVICE, dtype=torch.float16)
    bound = helion.kernel(matmul.fn, backend="cute", static_shapes=True).bind((a, b))
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.lane_loop_distribution"
    ):
        code = bound.to_code(bound.config_spec.autotune_reference_config())
    assert code.count("sA_mma[") == 2, code
    assert "thread barrier placed" not in caplog.text


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_broadcast_matmul_reference_input_view(dtype: torch.dtype) -> None:
    x = torch.randn((2, 256, 768), device=DEVICE, dtype=dtype)
    w = torch.randn((768, 256), device=DEVICE, dtype=dtype)
    bound = helion.kernel(broadcast_matmul.fn, backend="cute", static_shapes=True).bind(
        (x, w)
    )
    compiled = bound.compile_config(bound.config_spec.autotune_reference_config())
    for _ in range(3):
        x.normal_()
        w.normal_()
        torch.testing.assert_close(compiled(x, w), x @ w, rtol=1e-2, atol=1e-1)
