"""CPU source/cache regressions for native MMA TensorMap alignment admission."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from examples.aot_example import matmul_custom_key
import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_scalar_staging_sync import _native_matmul

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@pytest.fixture(autouse=True)
def _target(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HELION_CUTE_MMA_IMPL", "auto")
    with _cpu_target():
        yield


def _kernel() -> Kernel[torch.Tensor]:
    return helion.kernel(
        matmul_custom_key.fn,
        key=matmul_custom_key._key_fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        dot_precision="ieee",
    )


def _inputs(
    dtype: torch.dtype,
    *,
    row_padding_a: int = 0,
    row_padding_b: int = 0,
    offset_a: int = 0,
    offset_b: int = 0,
    transpose_b: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_storage = torch.empty(64 * (64 + row_padding_a) + offset_a, dtype=dtype)
    b_storage = torch.empty(64 * (64 + row_padding_b) + offset_b, dtype=dtype)
    a = a_storage.as_strided((64, 64), (64 + row_padding_a, 1), offset_a)
    b = b_storage.as_strided((64, 64), (64 + row_padding_b, 1), offset_b)
    return a, b.T if transpose_b else b


def _tma_operands(bound: BoundKernel[torch.Tensor]) -> set[str]:
    source = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
    assert "make_trivial_tiled_mma(" in source and "cute.gemm(" in source
    result = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.copy"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            if node.args[0].id == "tma_atom_a":
                result.add("a")
            if node.args[0].id == "tma_atom_b":
                result.add("b")
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize(
    "padding_a,padding_b,expected",
    [
        (0, 0, {"a", "b"}),
        (1, 0, set()),
        (0, 1, set()),
        (1, 1, set()),
        (8, 8, {"a", "b"}),
    ],
)
def test_native_mma_checks_each_outer_stride(
    dtype: torch.dtype,
    transpose_b: bool,
    padding_a: int,
    padding_b: int,
    expected: set[str],
) -> None:
    bound = _kernel()._bind_isolated(
        _inputs(
            dtype,
            row_padding_a=padding_a,
            row_padding_b=padding_b,
            transpose_b=transpose_b,
        )
    )
    assert _tma_operands(bound) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "offset_a,offset_b,expected", [(1, 0, set()), (0, 1, set()), (8, 8, {"a", "b"})]
)
def test_native_mma_checks_each_base_pointer(
    dtype: torch.dtype, offset_a: int, offset_b: int, expected: set[str]
) -> None:
    bound = _kernel()._bind_isolated(
        _inputs(dtype, offset_a=offset_a, offset_b=offset_b)
    )
    assert _tma_operands(bound) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_contiguous_double_tail_keeps_native_mma_without_invalid_tma(
    dtype: torch.dtype,
) -> None:
    bound = _kernel()._bind_isolated(
        (torch.empty((73, 78), dtype=dtype), torch.empty((78, 41), dtype=dtype))
    )
    assert _tma_operands(bound) == set()


@pytest.mark.parametrize("unaligned_first", [False, True])
def test_custom_key_separates_layout_alignment_in_both_discovery_orders(
    unaligned_first: bool,
) -> None:
    kernel = _kernel()
    aligned = _inputs(torch.float16, row_padding_a=8, row_padding_b=8)
    unaligned = _inputs(torch.float16, row_padding_a=1, row_padding_b=1)
    pointer_unaligned = _inputs(
        torch.float16, row_padding_a=8, row_padding_b=8, offset_a=1
    )
    assert kernel._key_fn is not None
    assert (
        kernel._key_fn(*aligned)
        == kernel._key_fn(*unaligned)
        == kernel._key_fn(*pointer_unaligned)
    )
    args = [unaligned, aligned] if unaligned_first else [aligned, unaligned]
    bindings = [kernel.bind(values) for values in args]
    assert bindings[0] is not bindings[1]
    assert kernel.specialization_key(aligned) != kernel.specialization_key(unaligned)
    for values, bound in zip(args, bindings, strict=True):
        assert _tma_operands(bound) == ({"a", "b"} if values is aligned else set())
    pointer_bound = kernel.bind(pointer_unaligned)
    assert pointer_bound not in bindings
    assert _tma_operands(pointer_bound) == set()
    for values, bound in zip(args, bindings, strict=True):
        assert kernel.bind(values) is bound


def test_fp8_keeps_guarded_tma_admission() -> None:
    args = _inputs(torch.float8_e4m3fn)
    bound = _native_matmul._bind_isolated(args)
    assert _tma_operands(bound) == {"a", "b"}
