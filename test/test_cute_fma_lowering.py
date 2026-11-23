"""``ops.fma`` is one ``cute.math.fma`` in the generated CuTe code.

Inductor lowers ``torch.addcmul``, ``add`` with ``alpha``, ``addcdiv`` with a
value and ``prims.fma`` to ``ops.fma`` on a CUDA device.  Its Triton form is
``tl.fma``; the base ``add(mul(x, y), z)`` left the fusion to NVVM and ptxas,
which fuse the addend's product instead when that is a product too (the
softmax backward's ``fma(-p, total, grad * p)``).  An fp32 or fp64 result is
one fma; 16-bit floats, integers and an fp32 product of two Python scalars
keep the product and sum.  The device results are checked in
``test_cute_pointwise_vec.py``.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch._inductor.inductor_prims  # noqa: F401  (registers prims.fma)

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _addcmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        out[tile] = torch.addcmul(a[tile], b[tile], c[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _add_alpha(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        out[tile] = torch.add(a[tile], b[tile], alpha=0.3)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _add_scalar_alpha(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        out[tile] = torch.add(a[tile], 2.5, alpha=0.3)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _negated_fma(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.ops.prims.fma(-x[tile], y[tile], a[tile] * b[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _plain_fma(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.ops.prims.fma(x[tile], y[tile], z[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _addcmul_of_negated_self(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        out[tile] = torch.addcmul(-a[tile], b[tile], c[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _add_in_place(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(a.size(0)):
        a[tile] = a[tile].add_(b[tile], alpha=2.0)
    return a


def _body_of(kernel: helion.Kernel, *dtypes: torch.dtype) -> str:
    # CPU tensors traced as cuda:0: the lowerings ask for ops.fma on a CUDA
    # device only.
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            kernel, tuple(torch.zeros(64, dtype=dtype) for dtype in dtypes)
        )
        code = bound.to_code(helion.Config(block_sizes=[32]))
    return code.split("def _helion_")[1].split("\ndef ")[0]


def _body(kernel: helion.Kernel, dtype: torch.dtype, count: int) -> str:
    return _body_of(kernel, *[dtype] * count)


FMA = r"cute\.math\.fma\((cutlass\.\w+)\((\w+)\), \1\((\w+)\), \1\((\w+)\)\)"


@pytest.mark.parametrize(
    "dtype,cute_type",
    [(torch.float32, "cutlass.Float32"), (torch.float64, "cutlass.Float64")],
)
def test_addcmul_is_one_fma_in_the_result_dtype(
    dtype: torch.dtype, cute_type: str
) -> None:
    body = _body(_addcmul, dtype, 3)
    (call,) = re.finditer(FMA, body)
    assert call.group(1) == cute_type
    # fma(tensor1, tensor2, self): no separate product is left to contract.
    assert call.group(2, 3, 4) == ("load_1", "load_2", "load")
    assert not re.search(r"= load_1 \* load_2\b", body)
    line = body[body.rfind("\n", 0, call.start()) + 1 : body.find("\n", call.end())]
    assert line.strip().endswith(f"= {call.group(0)}"), line


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sixteen_bit_results_keep_the_product_and_sum(dtype: torch.dtype) -> None:
    # cute.math.fma of 16-bit operands is the fp32 fma, whose result the
    # cast to the dtype would round a second time; the product and sum in
    # the dtype are contracted to one 16-bit fma.
    bodies = {
        kernel: _body(kernel, dtype, count)
        for kernel, count in ((_addcmul, 3), (_add_alpha, 2))
    }
    for body in bodies.values():
        assert "cute.math.fma(" not in body
        assert "cutlass.Float32(" not in body
    assert re.search(
        r"(\w+) = load_1 \* load_2\n\s+\w+ = \1 \+ load\n", bodies[_addcmul]
    )


def test_an_fp32_product_of_python_scalars_is_folded_not_fused() -> None:
    # torch.add(x, 2.5, alpha=0.3): the two literals multiply in double
    # precision when the kernel is traced (0.75) and one add remains.  An
    # fma would multiply by alpha rounded to fp32.
    body = _body(_add_scalar_alpha, torch.float32, 1)
    assert "cute.math.fma(" not in body
    product = re.search(
        r"(\w+) = 2\.5\n\s+(\w+) = 0\.3\n\s+(\w+) = \1 \* \2\n\s+\w+ = \3 \+ load\n",
        body,
    )
    assert product, body


def test_an_fp64_product_of_python_scalars_is_fused() -> None:
    # In fp64 neither literal is rounded, so the fma is the graph's value
    # rounded once; the folded sum would add the rounded product 0.75.
    body = _body(_add_scalar_alpha, torch.float64, 1)
    (call,) = re.finditer(FMA, body)
    assert call.group(1) == "cutlass.Float64"
    other, alpha, addend = call.group(2, 3, 4)
    assert re.search(rf"\b{other} = 2\.5\n\s+{alpha} = 0\.3\n", body), body
    assert addend == "load"


def test_add_with_alpha_is_one_fma() -> None:
    body = _body(_add_alpha, torch.float32, 2)
    (call,) = re.finditer(FMA, body)
    assert call.group(2) == "load_1" and call.group(4) == "load"


@pytest.mark.parametrize(
    "dtypes,computes_in,result",
    [
        ((torch.float32, torch.float64), "cutlass.Float64", "cutlass.Float32"),
        ((torch.bfloat16, torch.float32), "cutlass.Float32", "cutlass.BFloat16"),
    ],
)
def test_an_in_place_add_is_an_fma_in_the_dtype_its_operands_promote_to(
    dtypes: tuple[torch.dtype, torch.dtype], computes_in: str, result: str
) -> None:
    # ``a.add_(b, alpha=2.0)`` with a wider ``b`` computes in the dtype of
    # ``b`` and rounds the result to the dtype of ``a``: the node's result
    # dtype is the narrower one, and an fma in it would round ``b`` first.
    body = _body_of(_add_in_place, *dtypes)
    (call,) = re.finditer(FMA, body)
    assert call.group(1) == computes_in
    line = body[body.rfind("\n", 0, call.start()) + 1 : call.start()]
    fused = line.split(" = ")[0].strip()
    assert re.search(rf"\b\w+ = {re.escape(result)}\({fused}\)\n", body), body
    # The kernel stores the rounded result back.
    assert f".store({result}(" in body


def test_integer_results_keep_the_product_and_sum() -> None:
    body = _body(_addcmul, torch.int32, 3)
    assert "cute.math.fma(" not in body
    assert re.search(r"= load_1 \* load_2\n", body)
    # prims.fma of integers reaches ops.fma: no fma of integer operands.
    body = _body(_plain_fma, torch.int32, 3)
    assert not re.search(r"cute\.math\.fma\(cutlass\.Int", body)


def test_fma_fuses_its_own_product_beside_a_product_addend() -> None:
    body = _body(_negated_fma, torch.float32, 4)
    assert not re.search(FMA, body)
    # fma(-x, y, c) is written fma(-(-x), -y, c), which folds to
    # fma(x, -y, c): a quotient x keeps its division, where NVVM would move
    # the negation into the dividend and compute it a second time.
    (call,) = re.finditer(
        r"cute\.math\.fma\(-cutlass\.Float32\((\w+)\), -cutlass\.Float32\((\w+)\), "
        r"cutlass\.Float32\((\w+)\)\)",
        body,
    )
    negated, factor, addend = call.groups()
    assert re.search(rf"\b{negated} = -load\n", body)
    assert factor == "load_1"
    # The addend is the separately rounded product a * b.
    assert re.search(rf"\b{addend} = load_2 \* load_3\n", body)


def test_only_a_negated_multiplicand_moves_its_negation() -> None:
    (call,) = re.finditer(FMA, _body(_plain_fma, torch.float32, 3))
    assert call.group(2, 3, 4) == ("load", "load_1", "load_2")
    # addcmul's first argument is the addend, not a multiplicand.
    body = _body(_addcmul_of_negated_self, torch.float32, 3)
    (call,) = re.finditer(FMA, body)
    assert call.group(2, 3) == ("load_1", "load_2")
    assert re.search(rf"\b{call.group(4)} = -load\n", body)
