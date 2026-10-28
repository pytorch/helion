"""CUDA numerics of the materialized se_block backward GEMMs against the Triton
backend and the fused CuTe form."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._se_block_derived_operand_kernels import se_block_bwd_dw
from test._se_block_derived_operand_kernels import se_block_bwd_dx

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import Kernel

_SHAPES = ((256, 128), (1024, 512), (4096, 1024))


def _inputs(m: int, n: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(1234 + m)
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16, generator=generator)
    w = torch.randn([n, n], device=DEVICE, dtype=torch.float16, generator=generator)
    grad = torch.randn([m, n], device=DEVICE, dtype=torch.float16, generator=generator)
    return {"grad_out": grad, "x": x, "w": w, "s": torch.sigmoid(x @ w)}


def _args(kernel: Kernel[Any], values: dict[str, torch.Tensor]) -> tuple[Any, ...]:
    if kernel is se_block_bwd_dx:
        return (values["grad_out"], values["x"], values["w"], values["s"])
    return (values["grad_out"], values["x"], values["s"])


def _reference(kernel: Kernel[Any], values: dict[str, torch.Tensor]) -> torch.Tensor:
    """fp32 math over the once-rounded fp16 derived operand (the example's
    contract)."""
    grad, x, s = values["grad_out"].float(), values["x"].float(), values["s"].float()
    derived = (2.0 * grad * x * s * (1.0 - s)).to(torch.float16).float()
    if kernel is se_block_bwd_dx:
        return (2.0 * grad * s + derived @ values["w"].float().T).to(torch.float16)
    return (x.T @ derived).to(torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", _SHAPES, ids=["x".join(map(str, s)) for s in _SHAPES])
@pytest.mark.parametrize("kernel", [se_block_bwd_dx, se_block_bwd_dw], ids=["dx", "dw"])
def test_materialized_backward_matches_triton_and_the_fused_form(
    kernel: Kernel[Any], shape: tuple[int, int]
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    m, n = shape
    values = _inputs(m, n)
    args = _args(kernel, values)
    materialized = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )._bind_isolated(args)
    # dx reuses its [M, K] recipe across N (256-column tiles), dw its [K, N]
    # recipe across N1 (128-row tiles): the gate wants two tiles' worth.
    reused_tile = 256 if kernel is se_block_bwd_dx else 128
    assert (materialized.env.cute_fission_plan is not None) is (n >= 2 * reused_tile)
    fused = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_materialize_transformed_operands=False,
    )._bind_isolated(args)
    assert fused.env.cute_fission_plan is None
    triton = helion.kernel(
        kernel.fn, backend="triton", static_shapes=True, autotune_effort="none"
    )._bind_isolated(args)
    reference = _reference(kernel, values)
    # The example's tolerances; the materialized dx adds its seed after the
    # contraction (one fp32 reassociation), the fused and Triton forms before.
    tolerance = {"rtol": 1e-2, "atol": 1e-1}
    actual = materialized(*args)
    torch.testing.assert_close(actual, reference, **tolerance)
    torch.testing.assert_close(actual, triton(*args), **tolerance)
    torch.testing.assert_close(actual, fused(*args), **tolerance)
