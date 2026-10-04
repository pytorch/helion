from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from helion._compiler.cute.chained_vector_leaf import VectorLeafPlan
from helion._compiler.cute.chained_vector_leaf import emit_vector_leaf
from helion._compiler.cute.chained_vector_leaf import prove_vector_leaf
from helion.runtime import default_cute_launcher

if TYPE_CHECKING:
    from types import ModuleType


def _plan(
    column: str = "base + element",
    *,
    row: str = "row",
    mask: str | None = "row < limit",
    definitions: dict[str, str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> VectorLeafPlan | None:
    return prove_vector_leaf(
        (row, column),
        definitions or {},
        element="element",
        uniform_names={"row", "base", "limit", "temporary", "shift", "adjust"},
        shape=(4, 24),
        strides=(24, 1),
        dtype=dtype,
        mask=mask,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_vector_leaf_keeps_exact_uniform_fixed_width_base(dtype: torch.dtype) -> None:
    row = "(cutlass.Int32(row) * 1073741824) // 1073741824"
    plan = _plan(row="temporary", definitions={"temporary": row}, dtype=dtype)
    assert plan is not None
    assert plan.indices[0] == ast.unparse(ast.parse(row, mode="eval").body)
    assert plan.indices[0] != "row"
    assert plan.width == 8 and plan.axis == 1
    assert plan.element_bytes == (4 if dtype == torch.float32 else 2)


@pytest.mark.parametrize(
    "column",
    [
        "element * 2",
        "element // 8",
        "element % 8",
        "base - element",
        "element - element + base",
        "cutlass.Int32(base + element)",
        "(cutlass.Int32(element) * 1073741824) // 1073741824",
        "element + (element - element)",
        "row if element < 4 else element",
    ],
)
def test_vector_leaf_rejects_unproven_varying_operations(column: str) -> None:
    assert _plan(column) is None


@pytest.mark.parametrize(
    "definitions",
    [
        {"temporary": "element < limit"},
        {"temporary": "hidden", "hidden": "element < limit"},
        {"temporary": "unknown < limit"},
        {"temporary": "temporary"},
        {"temporary": "hidden", "hidden": "temporary"},
        {"temporary": "tensor[element]"},
        {"temporary": "tensor[0]"},
    ],
)
def test_uniform_mask_annotation_cannot_hide_its_dependencies(
    definitions: dict[str, str],
) -> None:
    assert _plan(mask="temporary", definitions=definitions) is None


def test_mask_expansion_and_nonstride_one_rejection() -> None:
    plan = _plan(
        mask="temporary",
        definitions={"temporary": "operator.lt(row, limit)"},
    )
    assert plan is not None and plan.mask == "operator.lt(row, limit)"
    assert (
        prove_vector_leaf(
            ("row", "base + element"),
            {},
            element="element",
            uniform_names={"row", "base"},
            shape=(4, 24),
            strides=(48, 2),
            dtype=torch.float16,
        )
        is None
    )
    assert _plan(dtype=torch.int32) is None
    assert _plan("base", mask=None) is None
    assert _plan(row="row + element") is None


def _source(
    dtype: torch.dtype, wrapped_row: bool = False, mixed_column: bool = False
) -> str:
    row = "(cutlass.Int32(row) * 1073741824) // 1073741824" if wrapped_row else "row"
    column = (
        "cutlass.Int32(base) + element + cutlass.Int64(shift)"
        if mixed_column
        else "base + element"
    )
    plan = _plan(column, row=row, dtype=dtype)
    assert plan is not None

    def pointer(indices: tuple[str, ...]) -> str:
        r, c = indices
        return (
            "tensor.iterator + "
            f"(cutlass.Int32({r}) * cutlass.Int32(tensor.layout.stride[0]) + "
            f"cutlass.Int32({c}) * cutlass.Int32(tensor.layout.stride[1]))"
        )

    def scalar(element: str) -> tuple[list[str], str]:
        return (
            [
                f"scalar_row = {row}",
                f"scalar_col = {column.replace('element', element)}",
            ],
            (
                f"({pointer(('scalar_row', 'scalar_col'))}).load() "
                "if (0 <= scalar_row < 4) & (0 <= scalar_col < 24) & (row < limit) "
                f"else {plan.dtype}(0)"
            ),
        )

    emission = emit_vector_leaf(
        plan,
        tensor="tensor",
        prefix="leaf",
        pointer_for_indices=pointer,
        scalar_for_element=scalar,
    )
    lines = [
        "from __future__ import annotations",
        "import cutlass",
        "import cutlass.cute as cute",
        "@cute.kernel",
        "def vector_leaf_test(tensor, output, flags, row, base, limit"
        + (", shift):" if mixed_column else "):"),
        "    if cute.arch.thread_idx()[0] == 0:",
        *("        " + line for line in emission.lines),
        "        for element in cutlass.range_constexpr(8):",
        f"            output[element] = {emission.values}[element]",
        f"        flags[0] = cutlass.Int32({emission.vectorized})",
    ]
    return "\n".join(lines) + "\n"


def test_emission_preserves_scalar_address_and_widens_only_guards() -> None:
    source = _source(torch.float32, True)
    ast.parse(source)
    assert "leaf_index_0 = cutlass.Int32(row) * 1073741824 // 1073741824" in source
    assert "cutlass.Int64(leaf_last) == cutlass.Int64(leaf_index_1) + 7" in source
    assert "leaf_last_address == leaf_address + 28" in source
    assert (
        "cutlass.Int32(leaf_index_0) * cutlass.Int32(tensor.layout.stride[0])" in source
    )
    assert (
        "cutlass.Int32(scalar_row) * cutlass.Int32(tensor.layout.stride[0])" in source
    )
    assert "if not leaf_vectorized" in source
    assert "else cutlass.Float32(0)" in source
    assert "num_bits_per_copy=128" in source
    assert "tensor.shape[0] == 4" in source
    assert "tensor.layout.stride[1] == 1" in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_vector_leaf_gpu_fast_and_scalar_fallbacks(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module: ModuleType = PyCodeCache.load(_source(dtype))
    storage = (torch.arange(4 * 48, dtype=torch.float32, device="cuda") + 0.3125).to(
        dtype
    )
    contiguous = storage[: 4 * 24].view(4, 24)
    strided = storage.view(4, 48)[:, ::2]
    before = storage.clone()
    output = torch.empty(8, device="cuda", dtype=dtype)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    for tensor, row, base, limit, expected_vector in (
        (contiguous, 1, 0, 4, True),
        (contiguous, 2, 8, 4, True),
        (contiguous, 1, 1, 4, False),
        (contiguous, 1, 20, 4, False),
        (contiguous, 1, -4, 4, False),
        (contiguous, 3, 0, 3, False),
        (strided, 1, 0, 4, False),
    ):
        default_cute_launcher(
            module.vector_leaf_test,
            (1,),
            tensor,
            output,
            flags,
            row,
            base,
            limit,
            block=(32, 1, 1),
        )
        expected = torch.zeros_like(output)
        if row < limit:
            for element in range(8):
                if 0 <= base + element < 24:
                    expected[element] = tensor[row, base + element]
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert bool(flags.item()) == expected_vector
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


def test_vector_leaf_gpu_uniform_fixed_width_wrap_is_not_cancelled() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = PyCodeCache.load(_source(torch.float32, True))
    tensor = torch.arange(96, device="cuda", dtype=torch.float32).view(4, 24)
    output = torch.empty(8, device="cuda", dtype=torch.float32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    for row in (1, 2, 4):
        default_cute_launcher(
            module.vector_leaf_test,
            (1,),
            tensor,
            output,
            flags,
            row,
            0,
            5,
            block=(32, 1, 1),
        )
        # Int32(row * 2**30) // 2**30 is 1, -2, 0, not row.
        wrapped = ((row * 2**30 + 2**31) % 2**32 - 2**31) // 2**30
        expected = tensor[wrapped, :8] if wrapped >= 0 else torch.zeros_like(output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert bool(flags.item()) == (wrapped >= 0)


def _int32(value: int) -> int:
    return (value + 2**31) % 2**32 - 2**31


@pytest.mark.parametrize("widen_before_subtract", [False, True])
def test_mixed_width_endpoint_guard_rejects_noncontiguous_interior(
    widen_before_subtract: bool,
) -> None:
    expression = (
        "cutlass.Int32(base) + element + cutlass.Int64(shift) - cutlass.Int32(adjust)"
        if widen_before_subtract
        else "cutlass.Int32(base) + element - cutlass.Int32(adjust) + cutlass.Int64(shift)"
    )
    assert _plan(expression) is not None
    accepted = rejected = 0
    for edge in (-(2**31), 2**31):
        for delta in range(-16, 17):
            base = edge + delta
            for adjust in (-17, -1, 0, 1, 17, 2**31 - 4):
                narrowed = [_int32(_int32(base) + element) for element in range(8)]
                values = (
                    [value - _int32(adjust) for value in narrowed]
                    if widen_before_subtract
                    else [_int32(value - _int32(adjust)) for value in narrowed]
                )
                # The uniform Int64 offset makes the first index zero but must
                # not hide an intermediate Int32 discontinuity inside a vector.
                shift = -values[0]
                values = [value + shift for value in values]
                guard = 0 <= values[0] <= 24 - 8 and values[-1] == values[0] + 7
                if guard:
                    accepted += 1
                    assert values == list(range(8))
                else:
                    rejected += 1
    assert accepted and rejected


def test_vector_leaf_gpu_mixed_width_intermediate_overflow() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = PyCodeCache.load(_source(torch.float32, mixed_column=True))
    tensor = torch.arange(96, device="cuda", dtype=torch.float32).view(4, 24) + 0.3125
    output = torch.empty(8, device="cuda", dtype=torch.float32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    for base in (2**31 - 8, 2**31 - 4, 2**31 - 1, -(2**31), -(2**31) + 4):
        shift = -base
        default_cute_launcher(
            module.vector_leaf_test,
            (1,),
            tensor,
            output,
            flags,
            0,
            base,
            4,
            shift,
            block=(32, 1, 1),
        )
        indices = [_int32(base + element) + shift for element in range(8)]
        expected = torch.zeros_like(output)
        for element, index in enumerate(indices):
            if 0 <= index < 24:
                expected[element] = tensor[0, index]
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert bool(flags.item()) == (indices == list(range(8)))
