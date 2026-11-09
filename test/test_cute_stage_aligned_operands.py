"""CPU-only tests of the CuTe unaligned-operand staging pass.

The pass (``helion/_compiler/cute/stage_aligned_operands.py``) pads matmul
operands whose strides fail the TensorMap proof through a host copy so the
tcgen05 pipeline keeps its TMA producers. These tests bind GEMMs on CPU
tensors under the fake sm_100 target and inspect the host function and the
rendered host prologue; no GPU is used.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_kernels import _gemm
from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_tcgen05_ordinary_exploration import _ordinary_gemm

import helion
from helion._compiler.cute.stage_aligned_operands import _needs_staging
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

# The B200 model with tcgen05 MMA support, CUDA hidden, the inputs traced
# as ``cuda:0`` tensors of a 148-SM device, and the reduction SMEM budget.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=tcgen05_mma_support(),
        forbid_native=False,
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
    ),
]

CPU_DEVICE = "cpu"


def _inplace_gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _dot_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            x_tile = x[tile_m, tile_k]
            y_tile = y[tile_k, tile_n]
            acc = hl.dot(x_tile, y_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def _inplace_through_alias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``a`` is a matmul operand and is written through the host alias
    # ``scratch``: it must stay the caller's tensor, not a padded copy.
    m, k = a.size()
    _, n = b.size()
    scratch = a.detach()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        scratch[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _merged_view_operands(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Operands reached through ``view`` / ``contiguous`` are not staged: a
    # view that merges the padded axis fails on the slice and ``contiguous``
    # already copies.
    m, k = a.size()
    _, n = b.size()
    a2 = a.reshape(-1).view(m, k)
    b2 = b.contiguous()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a2[tile_m, tile_k], b2[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _annotated_alias_store(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    scratch: torch.Tensor = a.detach()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        scratch[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _tuple_alias_store(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    scratch, other = a, b
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], other[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        scratch[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _hl_store_into_operand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        hl.store(a, [tile_m, tile_n], acc.to(a.dtype))
    return out


def _dict_return(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return {"out": out, "a": a}


def _layout_query(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    assert a.is_contiguous()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _bias_rows(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # No matmul: an unaligned bias operand is read, never staged.
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_m, tile_n]
    return out


def _bind(fn: object, *args: torch.Tensor, **settings: object) -> BoundKernel:
    return helion.kernel(fn, backend="cute", static_shapes=True, **settings).bind(args)  # pyrefly: ignore[bad-argument-type]


def _staging_statements(bound: BoundKernel) -> list[str]:
    return [
        ast.unparse(statement)
        for statement in bound.host_function.body
        if isinstance(statement, ast.Assign)
        and "constant_pad_nd" in ast.unparse(statement)
    ]


def test_needs_staging_rule() -> None:
    aligned = torch.empty((64, 4096), dtype=torch.bfloat16)
    assert _needs_staging(aligned) is None
    assert _needs_staging(aligned.T) is None
    # Odd bf16 rows: 8194-byte stride; padded to 4160 elements (128-byte rows).
    odd = torch.empty((4097, 4097), dtype=torch.bfloat16)
    assert _needs_staging(odd) == 63
    # An aligned extent that still fails the proof pads a whole block so the
    # copy is never a zero-pad alias.
    assert _needs_staging(torch.empty((4096, 4097), dtype=torch.bfloat16).T) == 64
    # fp8: 128-byte rows need 128-element multiples.
    assert _needs_staging(torch.empty((128, 4100), dtype=torch.float8_e4m3fn)) == 124
    # Under-aligned base offset.
    assert _needs_staging(torch.empty((64, 4096), dtype=torch.bfloat16)[1:]) is None
    assert _needs_staging(torch.empty((128, 4097), dtype=torch.bfloat16)[1:]) == 63
    assert (
        _needs_staging(torch.empty((8, 64, 4096), dtype=torch.bfloat16)[:, :, 1:]) == 1
    )
    # Rank-1 and rank-4 operands and axes of extent one are ignored.
    assert _needs_staging(torch.empty(4097, dtype=torch.bfloat16)) is None
    assert _needs_staging(torch.empty((2, 4, 1024, 68), dtype=torch.bfloat16)) is None
    # Small operands keep the scalar producers (STAGING_MIN_ELEMENTS).
    assert _needs_staging(torch.empty((16, 4100), dtype=torch.float8_e4m3fn)) is None
    assert _needs_staging(torch.empty((63, 4097), dtype=torch.bfloat16)) is None
    assert _needs_staging(torch.empty((1, 4096), dtype=torch.bfloat16)) is None


def test_odd_gemm_operands_are_staged_and_tma_proven() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_ordinary_gemm, a, b)
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 63])[:, :1025]",
        "b = torch.constant_pad_nd(b, [0, 63])[:, :1025]",
    ]
    assert bound.host_function.backend_rewrote_ast
    # The TMA producers the staged layout unlocks are checked on a GPU by
    # ``TestCuteStageAlignedOperandsGpu`` below; the CPU target renders the
    # host prologue only.
    code = bound.to_code(bound.config_spec.default_config())
    assert "constant_pad_nd(a, [0, 63])" in code


def test_only_the_unaligned_operand_is_staged() -> None:
    a = torch.empty((1024, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1024), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_dot_gemm, a, b)
    # ``x`` has 2050-byte rows; ``y`` has 2048-byte rows and stays in place.
    assert _staging_statements(bound) == [
        "x = torch.constant_pad_nd(x, [0, 63])[:, :1025]"
    ]


def test_aligned_gemm_is_left_alone() -> None:
    a = torch.empty((1024, 1024), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_ordinary_gemm, a, a)
    assert _staging_statements(bound) == []
    assert not bound.host_function.backend_rewrote_ast


def test_written_operand_and_non_matmul_reads_are_not_staged() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    out = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_inplace_gemm, a, a, out)
    # ``out`` is written and returned: never staged even though unaligned.
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 63])[:, :1025]",
        "b = torch.constant_pad_nd(b, [0, 63])[:, :1025]",
    ]
    bias = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_bias_rows, bias, bias)
    assert _staging_statements(bound) == []


def test_operand_written_through_an_alias_is_not_staged() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_inplace_through_alias, a, b)
    assert _staging_statements(bound) == [
        "b = torch.constant_pad_nd(b, [0, 63])[:, :1025]"
    ]


@pytest.mark.parametrize(
    "kernel,staged",
    (
        (_annotated_alias_store, ["b"]),
        # The tuple unpacking aliases both operands.
        (_tuple_alias_store, []),
        (_hl_store_into_operand, ["b"]),
        (_dict_return, ["b"]),
        (_layout_query, ["b"]),
    ),
    ids=lambda value: value.__name__ if callable(value) else "-".join(value) or "none",
)
def test_other_mentions_of_an_operand_keep_it_unstaged(kernel, staged) -> None:
    # Any mention of ``a`` beyond tile reads, metadata and the matmul operand
    # itself (an annotated or tuple alias that is stored into, an ``hl.store``
    # into it, a ``return`` of it, a layout query the copy would answer
    # differently) keeps the argument on its original tensor; an operand
    # mentioned only in those forms is still staged.
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(kernel, a, b)
    assert _staging_statements(bound) == [
        f"{name} = torch.constant_pad_nd({name}, [0, 63])[:, :1025]" for name in staged
    ]


def test_operands_behind_merging_views_are_not_staged() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_merged_view_operands, a, b)
    assert _staging_statements(bound) == []


def test_setting_disables_the_pass() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_ordinary_gemm, a, a, cute_stage_unaligned_operands=False)
    assert _staging_statements(bound) == []
    assert "cute_stage_unaligned_operands=False" in bound.format_kernel_decorator(
        bound.config_spec.default_config(), bound.settings
    )


# GPU tests of the CuTe unaligned-operand staging pass.
#
# A GEMM over odd-sized rows (``K = 1025`` bf16: 2050-byte strides) fails the
# TensorMap proof; with the staging pass the host pads the operands and the
# tcgen05 default config keeps its TMA producers, and the result matches torch.


@pytest.mark.cuda
@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteStageAlignedOperandsGpu(RefEagerTestDisabled, TestCase):
    def _check(self, a: torch.Tensor, b: torch.Tensor, staged: tuple[str, ...]) -> None:
        bound = _gemm.bind((a, b))
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
        for name in ("a", "b"):
            self.assertEqual(f"{name} = torch.constant_pad_nd(" in code, name in staged)
        if staged:
            self.assertIn("tma_tensor_a", code)
            self.assertIn("tma_tensor_b", code)
        bound.set_config(config)
        torch.testing.assert_close(bound(a, b), a @ b, rtol=1e-2, atol=1e-1)

    def test_odd_rows_both_operands(self) -> None:
        a = torch.randn(1025, 1025, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1025, 1025, device=DEVICE, dtype=torch.bfloat16)
        self._check(a, b, ("a", "b"))

    def test_odd_k_stages_a_only(self) -> None:
        a = torch.randn(512, 1025, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1025, 512, device=DEVICE, dtype=torch.bfloat16)
        self._check(a, b, ("a",))

    def test_transposed_odd_operand(self) -> None:
        a = torch.randn(1025, 512, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(512, 1025, device=DEVICE, dtype=torch.bfloat16)
        # ``a.T`` ([512, 1025], strides (1, 512)) has 1024-byte outer strides
        # and stays in place; ``b.T`` ([1025, 512], strides (1, 1025)) has
        # 2050-byte outer strides and is relaid out row-major.
        self._check(a.T, b.T, ("b",))

    def test_aligned_operands_untouched(self) -> None:
        a = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        self._check(a, a, ())
