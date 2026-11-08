"""Inductor's pointwise GEMM epilogues, written the way its Helion lowering emits them.

Each ``*_mm_into`` kernel stores ``epilogue(acc)`` of ``acc = a @ b``, most
with Inductor's spelling: constants are 0-d ``hl.full`` tensors, masks are
``.to(torch.bool)`` conversions, and a repeated operand (``y * y``) reaches the
device IR as ``mul(y, y)``. Some literal ``where`` forms instead
write Python numbers, which trace as ``aten.scalar_tensor``, and raw boolean
masks. ``INDUCTOR_EPILOGUES`` pairs each kernel with its
auxiliary inputs, its torch reference, and the chain step its tcgen05 epilogue
renders.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def half_bias_mm_into(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
) -> None:
    """``(x @ w + bias) * 0.5``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        half = hl.full([], 0.5, dtype=torch.float32)
        out[rows, cols] = ((bias[cols].to(torch.float32) + acc) * half).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def half_first_bias_mm_into(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
) -> None:
    """``0.5 * (x @ w + bias)``: the constant first."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        half = hl.full([], 0.5, dtype=torch.float32)
        out[rows, cols] = (half * (bias[cols].to(torch.float32) + acc)).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def one_minus_bias_mm_into(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
) -> None:
    """``(1 - x @ w) * bias``: the constant minus the carrier."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        one = hl.full([], 1.0, dtype=torch.float32)
        out[rows, cols] = ((one - acc) * bias[cols].to(torch.float32)).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def tanh_grad_mm_into(
    a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> None:
    """Tanh backward ``grad * (1 - y * y)``: one load used twice."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        yv = y[rows, cols].to(torch.float32)
        one = hl.full([], 1.0, dtype=torch.float32)
        out[rows, cols] = (acc * (one - yv * yv)).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def dropout_mm_into(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> None:
    """Dropout backward ``grad * mask.to(bool).to(f32) * 2``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        keep = mask[rows, cols].to(torch.bool).to(torch.float32)
        scale = hl.full([], 2.0, dtype=torch.float32)
        out[rows, cols] = (acc * (keep * scale)).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def keep_bias_mm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``mask * (x @ w) + bias``: the mask first, then a bias row."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        keep = mask[rows, cols].to(torch.bool).to(torch.float32)
        out[rows, cols] = (keep * acc + bias[cols].to(torch.float32)).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def relu_grad_mm_into(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> None:
    """ReLU backward ``where(x <= 0, 0, grad)``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        zero = hl.full([], 0.0, dtype=torch.float32)
        dead = x[rows, cols].to(torch.float32) <= zero
        out[rows, cols] = torch.where(dead, zero, acc).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def threshold_grad_mm_into(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> None:
    """Threshold backward ``where(mask.to(bool), 0, grad)``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        zero = hl.full([], 0.0, dtype=torch.float32)
        dead = mask[rows, cols].to(torch.bool)
        out[rows, cols] = torch.where(dead, zero, acc).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def relu_keep_literal_mm_into(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> None:
    """``torch.where(x > 0, grad, 0.0)`` with a Python literal: it traces to
    ``aten.scalar_tensor``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        zero = hl.full([], 0.0, dtype=torch.float32)
        alive = x[rows, cols].to(torch.float32) > zero
        out[rows, cols] = torch.where(alive, acc, 0.0).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_fill_fp32_mm_into(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> None:
    """An attention-mask fill into an FP32 output: ``where(mask, -inf, acc)``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = torch.where(mask[rows, cols], float("-inf"), acc)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def bf16_literal_where_mm_into(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> None:
    """``torch.where(mask, acc.to(bf16), 0.5)``: the literal traces as a BF16
    ``aten.scalar_tensor``."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = torch.where(mask[rows, cols], acc.to(torch.bfloat16), 0.5)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def gelu_tanh_grad_mm_into(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> None:
    """GELU (tanh) backward of ``x``: ``acc`` is read twice, in two sums that
    join at the store."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        v = x[rows, cols].to(torch.float32)
        half = hl.full([], 0.5, dtype=torch.float32)
        cube = hl.full([], 0.044715, dtype=torch.float32)
        root = hl.full([], 0.7978845608028654, dtype=torch.float32)
        one = hl.full([], 1.0, dtype=torch.float32)
        three = hl.full([], 3.0, dtype=torch.float32)
        square = v * v
        t = torch.tanh((v + square * v * cube) * root)
        inner_grad = acc * (v * half) * (one - t * t) * root
        left = inner_grad + inner_grad * cube * (square * three)
        out[rows, cols] = (left + acc * (t + one) * half).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def selu_grad_mm_into(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> None:
    """SELU backward ``where(x <= 0, acc * k1 * exp(x), acc * k2)``: both
    branches read the carrier."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        v = x[rows, cols].to(torch.float32)
        zero = hl.full([], 0.0, dtype=torch.float32)
        negative = hl.full([], 1.7580993408473766, dtype=torch.float32)
        positive = hl.full([], 1.0507009873554805, dtype=torch.float32)
        out[rows, cols] = torch.where(
            v <= zero, acc * negative * torch.exp(v), acc * positive
        ).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def hardswish_grad_mm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Hardswish backward of a dropout gradient ``g = acc * keep``:
    ``where(x <= -3, 0, where(x < 3, g * (x / 3 + 0.5), g))``. The inner
    select reads ``g`` in both branches; the chain continues below it."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        v = x[rows, cols].to(torch.float32)
        keep = mask[rows, cols].to(torch.bool).to(torch.float32)
        grad = acc * (keep * hl.full([], 1.25, dtype=torch.float32))
        third = hl.full([], 0.3333333333333333, dtype=torch.float32)
        half = hl.full([], 0.5, dtype=torch.float32)
        live = torch.where(
            v < hl.full([], 3.0, dtype=torch.float32), grad * (v * third + half), grad
        )
        dead = v <= hl.full([], -3.0, dtype=torch.float32)
        out[rows, cols] = torch.where(
            dead, hl.full([], 0.0, dtype=torch.float32), live
        ).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_scalar_fill_mm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    fill: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``where(mask, fill[()], acc * 2.0)``: a device scalar selected
    against the carrier."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = torch.where(
            mask[rows, cols].to(torch.bool),
            fill[()],
            acc * hl.full([], 2.0, dtype=torch.float32),
        ).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def quick_gelu_mm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """QuickGELU ``acc * sigmoid(acc * 1.702)``: the step reads the carrier
    on both sides."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        scale = hl.full([], 1.702, dtype=torch.float32)
        out[rows, cols] = (acc * torch.sigmoid(acc * scale)).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_residual_mm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``acc + where(mask, residual, 0)`` with a BF16 residual: the select's
    branches render as one type."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        kept = torch.where(mask[rows, cols], residual[rows, cols], 0.0)
        out[rows, cols] = (acc + kept).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_bias_first_mm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``where(mask, bias[col], 0) + acc`` with an FP32 row: the select is the
    first operand."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        kept = torch.where(mask[rows, cols], bias[cols], 0.0)
        out[rows, cols] = (kept + acc).to(out.dtype)


@dataclasses.dataclass(frozen=True)
class InductorEpilogue:
    """One epilogue form: its kernel, inputs, reference and rendered chain step."""

    name: str
    kernel: helion.Kernel
    # Auxiliary inputs after ``a`` and ``b``, in kernel order: ``"row"`` an
    # ``(n,)`` BF16 bias, ``"tile"`` an ``(m, n)`` BF16 tensor, ``"mask"`` an
    # ``(m, n)`` boolean mask, ``"scalar"`` a rank-0 FP32 tensor, ``"row32"``
    # an ``(n,)`` FP32 row.
    aux: tuple[str, ...]
    # ``reference(acc, *aux)``: the FP32 result from the FP32 accumulator.
    reference: Callable[..., torch.Tensor]
    # A regex for the epilogue line the tcgen05 chain renders for the form.
    step: str
    # The output's dtype (an FP32 store under 16-bit operands traces the
    # epilogue to its store).
    out_dtype: torch.dtype = torch.bfloat16


INDUCTOR_EPILOGUES: tuple[InductorEpilogue, ...] = (
    InductorEpilogue(
        "half_bias",
        half_bias_mm_into,
        ("row",),
        lambda acc, bias: (bias.float() + acc) * 0.5,
        r"= tcgen05_chain_step(?:_\d+)* \* 0\.5$",
    ),
    InductorEpilogue(
        "half_first_bias",
        half_first_bias_mm_into,
        ("row",),
        lambda acc, bias: 0.5 * (bias.float() + acc),
        r"= tcgen05_chain_step(?:_\d+)* \* 0\.5$",
    ),
    InductorEpilogue(
        "one_minus_bias",
        one_minus_bias_mm_into,
        ("row",),
        lambda acc, bias: (1.0 - acc) * bias.float(),
        r"= 1\.0 - tcgen05_acc_loaded(?:_\d+)*$",
    ),
    InductorEpilogue(
        "tanh_grad",
        tanh_grad_mm_into,
        ("tile",),
        lambda acc, y: acc * (1.0 - y.float() * y.float()),
        r"= 1\.0 - tcgen05_aux_product(?:_\d+)*$",
    ),
    InductorEpilogue(
        "dropout",
        dropout_mm_into,
        ("mask",),
        lambda acc, mask: acc * (mask.float() * 2.0),
        r"= tcgen05_aux_expr(?:_\d+)* \* 2\.0$",
    ),
    InductorEpilogue(
        "keep_bias",
        keep_bias_mm_into,
        ("mask", "row"),
        lambda acc, mask, bias: mask.float() * acc + bias.float(),
        r"= tcgen05_aux_expr(?:_\d+)* \* tcgen05_acc_loaded(?:_\d+)*$",
    ),
    InductorEpilogue(
        "relu_grad",
        relu_grad_mm_into,
        ("tile",),
        lambda acc, x: torch.where(x.float() <= 0.0, 0.0, acc),
        r"= cute\.where\(tcgen05_aux_expr(?:_\d+)*, 0\.0, "
        r"tcgen05_acc_loaded(?:_\d+)*\)$",
    ),
    InductorEpilogue(
        "relu_keep_literal",
        relu_keep_literal_mm_into,
        ("tile",),
        lambda acc, x: torch.where(x.float() > 0.0, acc, 0.0),
        r"= cute\.where\(tcgen05_aux_expr(?:_\d+)*, "
        r"tcgen05_acc_loaded(?:_\d+)*, 0\.0\)$",
    ),
    InductorEpilogue(
        "masked_fill_fp32",
        masked_fill_fp32_mm_into,
        ("mask",),
        lambda acc, mask: torch.where(mask, float("-inf"), acc),
        r"= cute\.where\(tcgen05_aux_loaded(?:_\d+)*, float\('-inf'\), "
        r"tcgen05_acc_loaded(?:_\d+)*\)$",
        out_dtype=torch.float32,
    ),
    InductorEpilogue(
        "bf16_literal_where",
        bf16_literal_where_mm_into,
        ("mask",),
        lambda acc, mask: torch.where(mask, acc.to(torch.bfloat16).float(), 0.5),
        r"= cute\.where\(tcgen05_aux_loaded(?:_\d+)*, tcgen05_chain_step(?:_\d+)*, "
        r"0\.5\)$",
    ),
    InductorEpilogue(
        "gelu_tanh_grad",
        gelu_tanh_grad_mm_into,
        ("tile",),
        lambda acc, x: _gelu_tanh_grad(acc, x.float()),
        r"= tcgen05_chain_step(?:_\d+)* \+ tcgen05_chain_step(?:_\d+)*$",
    ),
    InductorEpilogue(
        "selu_grad",
        selu_grad_mm_into,
        ("tile",),
        lambda acc, x: torch.where(
            x.float() <= 0.0,
            acc * 1.7580993408473766 * torch.exp(x.float()),
            acc * 1.0507009873554805,
        ),
        r"= cute\.where\(tcgen05_aux_expr(?:_\d+)*, tcgen05_chain_step(?:_\d+)*, "
        r"tcgen05_chain_step(?:_\d+)*\)$",
    ),
    InductorEpilogue(
        "hardswish_grad",
        hardswish_grad_mm_into,
        ("tile", "mask"),
        lambda acc, x, mask: _hardswish_grad(acc * (mask.float() * 1.25), x.float()),
        r"= cute\.where\(tcgen05_aux_expr(?:_\d+)*, tcgen05_chain_step(?:_\d+)*, "
        r"tcgen05_chain_step(?:_\d+)*\)$",
    ),
    InductorEpilogue(
        "masked_scalar_fill",
        masked_scalar_fill_mm_into,
        ("mask", "scalar"),
        lambda acc, mask, fill: torch.where(mask, fill, acc * 2.0),
        r"= cute\.where\(tcgen05_aux_loaded(?:_\d+)*, tcgen05_scalar(?:_\d+)*, "
        r"tcgen05_chain_step(?:_\d+)*\)$",
    ),
    InductorEpilogue(
        "quick_gelu",
        quick_gelu_mm_into,
        (),
        lambda acc: acc * torch.sigmoid(acc * 1.702),
        r"= tcgen05_acc_loaded(?:_\d+)* \* tcgen05_chain_step(?:_\d+)*$",
    ),
    InductorEpilogue(
        "masked_residual",
        masked_residual_mm_into,
        ("mask", "tile"),
        lambda acc, mask, residual: acc + torch.where(mask, residual.float(), 0.0),
        r"= cute\.where\(tcgen05_aux_loaded(?:_\d+)*, tcgen05_aux_expr(?:_\d+)*, "
        r"0\.0\)$",
    ),
    InductorEpilogue(
        "masked_bias_first",
        masked_bias_first_mm_into,
        ("mask", "row32"),
        lambda acc, mask, bias: torch.where(mask, bias, 0.0) + acc,
        r"= cute\.where\(tcgen05_aux_loaded(?:_\d+)*, tcgen05_aux_loaded(?:_\d+)*, "
        r"0\.0\)$",
    ),
    InductorEpilogue(
        "threshold_grad",
        threshold_grad_mm_into,
        ("mask",),
        lambda acc, mask: torch.where(mask, 0.0, acc),
        r"= cute\.where\(tcgen05_aux_loaded(?:_\d+)*, 0\.0, "
        r"tcgen05_acc_loaded(?:_\d+)*\)$",
    ),
)


def _gelu_tanh_grad(acc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = torch.tanh((x + 0.044715 * x * x * x) * 0.7978845608028654)
    inner = acc * (x * 0.5) * (1.0 - t * t) * 0.7978845608028654
    return inner + inner * 0.044715 * (x * x * 3.0) + acc * (t + 1.0) * 0.5


def _hardswish_grad(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    live = torch.where(x < 3.0, grad * (x * 0.3333333333333333 + 0.5), grad)
    return torch.where(x <= -3.0, 0.0, live)


def inductor_epilogue_args(
    epilogue: InductorEpilogue,
    m: int,
    n: int,
    k: int,
    *,
    device: torch.device | str,
    random: bool,
) -> tuple[torch.Tensor, ...]:
    """``(a, b, *aux, out)`` for ``epilogue``: seeded random values, or empty."""
    generator = torch.Generator(device=device).manual_seed(0) if random else None

    def make(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        if not random:
            return torch.empty(shape, dtype=dtype, device=device)
        if dtype is torch.bool:
            return torch.rand(shape, generator=generator, device=device) > 0.3
        return torch.randn(shape, generator=generator, device=device).to(dtype)

    aux_shapes = {
        "row": ((n,), torch.bfloat16),
        "tile": ((m, n), torch.bfloat16),
        "mask": ((m, n), torch.bool),
        "scalar": ((), torch.float32),
        "row32": ((n,), torch.float32),
    }
    return (
        make((m, k), torch.bfloat16),
        make((k, n), torch.bfloat16),
        *(make(*aux_shapes[kind]) for kind in epilogue.aux),
        torch.empty((m, n), dtype=epilogue.out_dtype, device=device),
    )
