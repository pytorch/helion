from __future__ import annotations

import pytest
import sympy
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute.epilogue_fanout import _chain_key
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language._tracing_ops import _get_symnode

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True)
def _scaled_matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
    beta: float,
    promote_bias: hl.constexpr,
) -> torch.Tensor:
    m, k = x.size()
    n = y.size(1)
    bias_dtype = torch.float32 if promote_bias else bias.dtype
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, col])
        residual = bias[row, col].to(bias_dtype)
        out[row, col] = alpha * acc + beta * residual
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _device_scaled_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = y.size(1)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, col])
        out[row, col] = row.begin * acc
    return out


def test_scalar_provenance_distinguishes_runtime_arguments_from_device_indices() -> (
    None
):
    x = torch.empty((128, 128), dtype=torch.bfloat16)
    bound = _cpu_bind(_scaled_matmul, (x, x.clone(), x.clone(), 1.25, -0.5, False))
    assert bound.host_function is not None
    scalars = [
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is _get_symnode
        and isinstance(node.meta.get("val"), torch.SymFloat)
    ]
    assert len(scalars) == 2
    assert all(cute_epilogue._runtime_scalar(node) is not None for node in scalars)

    bound = _cpu_bind(_device_scaled_matmul, (x, x.clone()))
    assert bound.host_function is not None
    indices = [
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is _get_symnode and node.meta.get("helion_host_scalar") is False
    ]
    assert indices
    assert all(cute_epilogue._runtime_scalar(node) is None for node in indices)


def test_fanout_chain_key_distinguishes_runtime_scalars() -> None:
    def chain(name: str) -> cute_epilogue.Tcgen05UnaryEpilogueChain:
        return cute_epilogue.Tcgen05UnaryEpilogueChain(
            steps=(
                cute_epilogue._AuxiliaryTensorExprStep(
                    expr=cute_epilogue._BinaryTensorExpr(
                        op_name="mul",
                        op_template="{lhs} * {rhs}",
                        lhs=cute_epilogue._CurrentTensorExpr(),
                        rhs=cute_epilogue._RuntimeScalarExpr(sympy.Symbol(name)),
                    )
                ),
            )
        )

    assert _chain_key(chain("alpha")) == _chain_key(chain("alpha"))
    assert _chain_key(chain("alpha")) != _chain_key(chain("beta"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("promote_bias", [False, True])
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_native_epilogue_reuses_compiled_kernel_with_new_scalars(
    dtype: torch.dtype, promote_bias: bool
) -> None:
    x, y, bias = (torch.randn((128, 128), device=DEVICE, dtype=dtype) for _ in range(3))
    bound = _scaled_matmul.bind((x, y, bias, 1.25, -0.5, promote_bias))
    compiled = bound.compile_config(bound.config_spec.autotune_reference_config())
    for alpha, beta in [(1.25, -0.5), (-0.75, 2.0), (0.0, 0.5)]:
        x.normal_()
        y.normal_()
        bias.normal_()
        residual = bias.float() if promote_bias else bias
        expected = (alpha * (x.float() @ y.float()) + (beta * residual).float()).to(
            dtype
        )
        actual = compiled(x, y, bias, alpha, beta, promote_bias)
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.01)
