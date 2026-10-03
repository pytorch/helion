"""CPU proof tests for optional direct vector copies of scalar recipes."""

from __future__ import annotations

import ast
import dataclasses
import itertools
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.contiguous_copy import CopyTensorFacts
from helion._compiler.cute.contiguous_copy import plan_contiguous_copy

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.cute.contiguous_copy import ContiguousCopy


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _fresh() -> Callable[[str], str]:
    counter = itertools.count()
    return lambda hint: f"copy_{next(counter)}_{hint}"


def _body(source: str) -> list[ast.Assign]:
    return cast("list[ast.Assign]", ast.parse(source).body)


def _plan(
    source: str,
    *,
    strides: tuple[int, ...] = (24, 1),
    alignment: int = 16,
    aligned_names: dict[str, int] | None = None,
    dtype: str = "cutlass.Float16",
) -> ContiguousCopy | None:
    return plan_contiguous_copy(
        _body(source),
        _expr("value"),
        coordinate="k",
        tensors={"A": CopyTensorFacts(dtype, strides, alignment)},
        aligned_names={"k": 8, "origin": 8} if aligned_names is None else aligned_names,
    )


_GATHER = """
valid_m = m < count
row = (indices.iterator + m).load() if valid_m else cutlass.Int32(0)
global_k = cutlass.Int32(origin + k)
valid_k = global_k < limit
value = (A.iterator + cutlass.Int32(row) * cutlass.Int32(A.layout.stride[0]) + global_k * cutlass.Int32(A.layout.stride[1])).load() if valid_m and valid_k else cutlass.Float16(0)
"""


@dataclasses.dataclass
class _Pointer:
    tensor: _Tensor
    offset: int = 0
    swizzle: bool = False

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.tensor, self.offset + offset, self.swizzle)

    @property
    def physical_offset(self) -> int:
        return self.offset ^ ((self.offset >> 1) & 8) if self.swizzle else self.offset

    def toint(self) -> _Pointer:
        assert not self.swizzle
        return self

    def load(self) -> int | float:
        assert 0 <= self.offset < len(self.tensor.values)
        self.tensor.loads.append(self.offset)
        return self.tensor.values[self.offset]


class _Tensor:
    def __init__(
        self,
        values: list[int | float],
        shape: tuple[int, ...],
        *,
        pointer_swizzle: bool = False,
    ) -> None:
        self.values = list(values)
        self.loads: list[int] = []
        self.iterator = _Pointer(self, swizzle=pointer_swizzle)
        self.element_type = float
        self.layout = SimpleNamespace(
            shape=shape, stride=(shape[1], 1) if len(shape) == 2 else (1,)
        )

    def __setitem__(self, index: tuple[int, ...], value: float) -> None:
        pointer = self.iterator + _index(index, self.layout)
        self.values[pointer.physical_offset] = value


def _index(coordinate: tuple[int, ...], layout: SimpleNamespace) -> int:
    assert all(0 <= i < dim for i, dim in zip(coordinate, layout.shape, strict=True))
    return sum(i * stride for i, stride in zip(coordinate, layout.stride, strict=True))


def _i32(value: int) -> int:
    return (int(value) + 2**31) % 2**32 - 2**31


def _evaluate_copy(
    plan: ContiguousCopy,
    *,
    origin: int,
    k: int,
    limit: int,
    count: int = 2,
    pointer_swizzle: bool = False,
) -> tuple[list[int | float], list[int], list[int], int]:
    source = _Tensor(list(range(4 * 24)), (4, 24))
    indices = _Tensor([3, 1], (2,))
    shared = _Tensor([-999] * 32, (2, 16), pointer_swizzle=pointer_swizzle)
    copies: list[tuple[int, int]] = []
    width = plan.width

    def cp_async(
        dst: _Pointer, src: _Pointer, size: int, cache: str, *, cp_size: int = 16
    ) -> None:
        assert size == 16 and cache == "cg"
        assert src.offset % width == 0 and dst.offset % width == 0
        assert cp_size in (0, 16)
        copies.append((dst.offset, src.offset))
        if cp_size:
            assert 0 <= src.offset <= len(src.tensor.values) - width
            dst.tensor.values[dst.offset : dst.offset + width] = src.tensor.values[
                src.offset : src.offset + width
            ]
        else:
            dst.tensor.values[dst.offset : dst.offset + width] = [0] * width

    def typed_copy(atom: str, src: SimpleNamespace, dst: SimpleNamespace) -> None:
        assert atom == "copy128"
        # A typed copy dereferences the shared pointer including its swizzle;
        # converting that pointer directly to LLVM loses this transformation.
        destination = dst.iterator
        physical = destination.physical_offset
        assert [(destination + lane).physical_offset for lane in range(width)] == list(
            range(physical, physical + width)
        )
        cp_async(_Pointer(destination.tensor, physical), src.iterator, 16, "cg")

    def make_ptr(
        dtype: type, value: _Pointer, space: str, *, assumed_align: int
    ) -> _Pointer:
        assert dtype is float and space == "gmem" and assumed_align == 16
        assert value.offset % width == 0
        return value

    def assume(value: int, *, divby: int) -> int:
        assert value % divby == 0
        return value

    def make_copy_atom(op: str, dtype: type, *, num_bits_per_copy: int) -> str:
        assert op == "cg" and dtype is float and num_bits_per_copy == 128
        return "copy128"

    body = plan.emit_to_aligned_smem(
        "shared",
        (_expr("m"), _expr("k")),
        _fresh(),
        preserve_pointer_swizzle=pointer_swizzle,
    )
    namespace = {
        "A": source,
        "indices": indices,
        "shared": shared,
        "m": 1,
        "k": k,
        "origin": origin,
        "limit": limit,
        "count": count,
        "operator": operator,
        "cutlass": SimpleNamespace(
            Int32=_i32,
            Int64=int,
            Float16=float,
            Float32=float,
            range=lambda n, unroll: range(n),
        ),
        "cute": SimpleNamespace(
            crd2idx=_index,
            arch=SimpleNamespace(cp_async_shared_global=cp_async),
            assume=assume,
            AddressSpace=SimpleNamespace(gmem="gmem"),
            nvgpu=SimpleNamespace(
                cpasync=SimpleNamespace(CopyG2SOp=lambda cache: cache),
                LoadCacheMode=SimpleNamespace(GLOBAL="cg"),
            ),
            make_ptr=make_ptr,
            make_copy_atom=make_copy_atom,
            make_layout=lambda shape, stride: SimpleNamespace(
                shape=shape, stride=stride
            ),
            make_tensor=lambda iterator, layout: SimpleNamespace(
                iterator=iterator, layout=layout
            ),
            copy=typed_copy,
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<copy-test>",
            "exec",
        ),
        namespace,
    )
    output = [
        shared.values[(shared.iterator + 16 + k + lane).physical_offset]
        for lane in range(width)
    ]
    return output, source.loads, indices.loads, len(copies)


@pytest.mark.parametrize("origin,k", [(0, 0), (0, 8), (8, 0), (16, 0), (16, 8)])
@pytest.mark.parametrize("limit", [0, 1, 3, 7, 8, 9, 15, 16, 17, 23, 24])
@pytest.mark.parametrize("pointer_swizzle", [False, True])
def test_gather_full_and_partial_vectors_match_scalar_loads(
    origin: int, k: int, limit: int, pointer_swizzle: bool
) -> None:
    plan = _plan(_GATHER)
    assert plan is not None
    output, loads, index_loads, copies = _evaluate_copy(
        plan, origin=origin, k=k, limit=limit, pointer_swizzle=pointer_swizzle
    )
    expected = [24 + origin + k + i if origin + k + i < limit else 0 for i in range(8)]
    assert output == expected
    full = origin + k + 7 < limit
    empty = origin + k >= limit
    assert copies == int(full or (empty and not pointer_swizzle))
    assert loads == (
        []
        if full
        else [24 + origin + k + i for i in range(8) if origin + k + i < limit]
    )
    # Invariant gathered row IDs are loaded once for either copy path.
    assert index_loads == [1]


def test_invalid_row_preserves_zero_and_skips_gather() -> None:
    plan = _plan(_GATHER)
    assert plan is not None
    output, loads, index_loads, copies = _evaluate_copy(
        plan, origin=0, k=0, limit=24, count=1
    )
    assert output == [0] * 8
    assert loads == [] and index_loads == [] and copies == 1


@pytest.mark.parametrize(
    "offset",
    [
        "cutlass.Int32(index_pointer.load()) * 24",
        "cutlass.Int32(24 // (count - m)) * 8",
        "cutlass.Int32(A.layout.stride[count + m]) * 8",
    ],
)
@pytest.mark.parametrize("pointer_swizzle", [False, True])
def test_inactive_pointer_expression_remains_guarded(
    offset: str, pointer_swizzle: bool
) -> None:
    plan = _plan(
        f"index_pointer = indices.iterator + m + 8\n"
        f"value = (A.iterator + {offset} + origin + k).load() "
        "if m < count else cutlass.Float16(0)"
    )
    assert plan is not None
    output, loads, index_loads, copies = _evaluate_copy(
        plan, origin=0, k=0, limit=24, count=1, pointer_swizzle=pointer_swizzle
    )
    assert output == [0] * 8
    assert loads == [] and index_loads == [] and copies == 0


@pytest.mark.parametrize("k", [0, 4, 8, 12])
@pytest.mark.parametrize("limit", [0, 1, 3, 4, 7, 8, 13, 16])
@pytest.mark.parametrize("pointer_swizzle", [False, True])
def test_fp32_four_element_copy_matches_exact_scalar_tails(
    k: int, limit: int, pointer_swizzle: bool
) -> None:
    plan = _plan(
        _GATHER.replace("Float16", "Float32"),
        dtype="cutlass.Float32",
        aligned_names={"k": 4, "origin": 4},
    )
    assert plan is not None and plan.width == 4
    output, loads, index_loads, copies = _evaluate_copy(
        plan, origin=0, k=k, limit=limit, pointer_swizzle=pointer_swizzle
    )
    assert output == [24 + k + lane if k + lane < limit else 0 for lane in range(4)]
    assert copies == int(k + 3 < limit or (k >= limit and not pointer_swizzle))
    assert loads == (
        [] if copies else [24 + k + lane for lane in range(4) if k + lane < limit]
    )
    assert index_loads == [1]


def test_fp32_alignment_is_in_bytes_and_casts_cannot_narrow() -> None:
    source = _GATHER.replace("Float16", "Float32")
    assert _plan(source, dtype="cutlass.Float32", alignment=8) is None
    assert _plan(source, dtype="cutlass.Float32", strides=(26, 1)) is None
    assert _plan(source.replace("Float32", "Float16"), dtype="cutlass.Float32") is None


def test_equal_inactive_endpoints_preserve_valid_middle_lanes() -> None:
    plan = _plan(
        "value = (A.iterator + 24 + k).load() if 2 < k < 5 else cutlass.Float16(0)"
    )
    assert plan is not None
    output, loads, _index_loads, copies = _evaluate_copy(plan, origin=0, k=0, limit=24)
    assert output == [0, 0, 0, 27, 28, 0, 0, 0]
    assert loads == [27, 28] and copies == 0


@pytest.mark.parametrize("predicate", [0, 1, 3, -5])
def test_invariant_numeric_predicate_keeps_truthiness(predicate: int) -> None:
    plan = _plan(
        f"value = (A.iterator + 24 + k).load() if {predicate} else cutlass.Float16(0)"
    )
    assert plan is not None
    output, loads, _index_loads, copies = _evaluate_copy(plan, origin=0, k=0, limit=24)
    assert output == (list(range(24, 32)) if predicate else [0] * 8)
    assert loads == [] and copies == 1


@pytest.mark.parametrize(
    "source",
    [
        "value = (A.iterator + k).load() * 2",
        "value = cutlass.Float32((A.iterator + k).load())",
        "value = (A.iterator + 2 * k).load()",
        "value = (A.iterator + k // 2).load()",
        "value = (A.iterator + (k % 8)).load()",
        "row = (indices.iterator + k).load()\nvalue = (A.iterator + row * 24 + k).load()",
        "value = (A.iterator + k).load() if k != 3 else cutlass.Float16(0)",
        "value = (A.iterator + k).load() if k < 2 or k >= 5 else cutlass.Float16(0)",
        "value = (A.iterator + k).load() if k + 3 < limit else cutlass.Float16(0)",
        "value = (A.iterator + k).load(volatile=True)",
        "value = unknown((A.iterator + k).load())",
        "(A.iterator + k).store(1)\nvalue = (A.iterator + k).load()",
        "value = (A.iterator + k).load()\nvalue = value + 1",
    ],
)
def test_rejects_unproven_recipe(source: str) -> None:
    assert _plan(source) is None


@pytest.mark.parametrize(
    "strides,alignment,aligned_names",
    [
        ((25, 1), 16, {"k": 8, "origin": 8}),
        ((24, 2), 16, {"k": 8, "origin": 8}),
        ((24, 1), 8, {"k": 8, "origin": 8}),
        ((24, 1), 16, {"k": 8}),
        ((24, 1), 16, {"k": 1, "origin": 8}),
    ],
)
def test_rejects_missing_or_mismatched_layout_facts(
    strides: tuple[int, ...], alignment: int, aligned_names: dict[str, int]
) -> None:
    assert (
        _plan(
            _GATHER, strides=strides, alignment=alignment, aligned_names=aligned_names
        )
        is None
    )


def test_same_dtype_cast_and_interval_guards_are_supported() -> None:
    plan = _plan(
        "value = cutlass.Float16((A.iterator + k).load()) if 0 <= k < limit else cutlass.Float16(0)"
    )
    assert plan is not None
    assert plan.source_tensor == "A"


def test_extended_ast_stride_literals_are_recognized() -> None:
    statement = statement_from_string(
        "value = (A.iterator + cutlass.Int32(row) * cutlass.Int32(A.layout.stride[0]) "
        "+ cutlass.Int32(k) * cutlass.Int32(A.layout.stride[1])).load() "
        "if k < limit else cutlass.Float16(0)"
    )
    assert isinstance(statement, ast.Assign)
    value = expr_from_string("value")
    assert isinstance(value, ast.expr)
    plan = plan_contiguous_copy(
        [statement],
        value,
        coordinate="k",
        tensors={"A": CopyTensorFacts("cutlass.Float16", (24, 1), 16)},
        aligned_names={"k": 8},
    )
    assert plan is not None
    assert plan.emit_to_aligned_smem("shared", (_expr("m"), _expr("k")), _fresh())


def test_bfloat16_is_supported_without_changing_scalar_dtype() -> None:
    plan = plan_contiguous_copy(
        _body("value = (A.iterator + k).load() if k < limit else cutlass.BFloat16(0)"),
        _expr("value"),
        coordinate="k",
        tensors={"A": CopyTensorFacts("cutlass.BFloat16", (1,), 16)},
        aligned_names={"k": 8},
    )
    assert plan is not None
    code = ast.unparse(
        ast.Module(
            body=plan.emit_to_aligned_smem(
                "shared", (_expr("m"), _expr("k")), _fresh()
            ),
            type_ignores=[],
        )
    )
    assert "BFloat16(0)" in code and "Float16(0)" not in code.replace("BFloat16(0)", "")


def test_emit_twice_uses_independent_temporaries_and_leaves_input_unchanged() -> None:
    statements = _body(_GATHER)
    original = ast.dump(ast.Module(body=[*statements], type_ignores=[]))
    plan = plan_contiguous_copy(
        statements,
        _expr("value"),
        coordinate="k",
        tensors={"A": CopyTensorFacts("cutlass.Float16", (24, 1), 16)},
        aligned_names={"k": 8, "origin": 8},
    )
    assert plan is not None
    fresh = _fresh()
    first = plan.emit_to_aligned_smem("sA", (_expr("m"), _expr("k")), fresh)
    second = plan.emit_to_aligned_smem("sA", (_expr("m"), _expr("k")), fresh)

    def writes(body: list[ast.stmt]) -> set[str]:
        return {
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    assert not writes(first) & writes(second)
    assert ast.dump(ast.Module(body=[*statements], type_ignores=[])) == original
