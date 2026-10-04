from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import importlib
import operator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.chained_rectangular_leaf import RectangularLeafPlan
from helion._compiler.cute.chained_rectangular_leaf import prove_rectangular_leaf


class _Integer(int):
    bits = 32

    def __new__(cls, value):
        return super().__new__(
            cls, (int(value) + 2 ** (cls.bits - 1)) % 2**cls.bits - 2 ** (cls.bits - 1)
        )

    def _binary(self, other, function, reverse=False):
        result_type = (
            _Int64 if isinstance(self, _Int64) or isinstance(other, _Int64) else _Int32
        )
        left, right = (int(other), int(self)) if reverse else (int(self), int(other))
        result = function(left, right)
        if result_type is _Int64:
            assert -(1 << 63) <= result < 1 << 63, "guard Int64 arithmetic overflow"
        return result_type(result)

    def __add__(self, other):
        return self._binary(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        return self._binary(other, operator.sub, True)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return self._binary(other, operator.floordiv)

    def __mod__(self, other):
        return self._binary(other, operator.mod)

    def __and__(self, other):
        return self._binary(other, operator.and_)

    def __or__(self, other):
        return self._binary(other, operator.or_)

    def __xor__(self, other):
        return self._binary(other, operator.xor)

    def __lshift__(self, other):
        return self._binary(other, operator.lshift)

    def __rshift__(self, other):
        return self._binary(other, operator.rshift)

    def __neg__(self):
        return type(self)(-int(self))

    def __invert__(self):
        return type(self)(~int(self))


class _Int32(_Integer):
    bits = 32


class _Int64(_Integer):
    bits = 64


def _environment(**values):
    return {
        "cutlass": SimpleNamespace(Int32=_Int32, Int64=_Int64),
        "operator": operator,
        **{
            name: value if isinstance(value, _Integer) else _Int32(value)
            for name, value in values.items()
        },
    }


def _plan(indices=("start + r", "head + c"), *, definitions=None, **kwargs):
    options = {
        "row": "r",
        "column": "c",
        "uniform_names": {"start", "head", "end", "capture", "temporary", "other"},
        "tile_shape": (4, 8),
        "shape": (64, 64),
        "strides": (64, 1),
        "dtype": torch.bfloat16,
    }
    options.update(kwargs)
    return prove_rectangular_leaf(indices, definitions or {}, **options)


def _rectangle(plan: RectangularLeafPlan, **values) -> bool:
    namespace = _environment(**values)
    enabled = bool(eval(plan.guard, namespace))
    assert not {plan.row, plan.column} & {
        node.id
        for node in ast.walk(ast.parse(plan.guard))
        if isinstance(node, ast.Name)
    }
    if not enabled:
        return False
    origin = tuple(int(eval(text, namespace)) for text in plan.origin)
    indices = tuple(int(eval(text, namespace)) for text in plan.origin_indices)
    assert int(eval(plan.base, namespace)) == sum(
        index * stride for index, stride in zip(indices, plan.strides, strict=True)
    )
    for row in range(plan.tile_shape[0]):
        for column in range(plan.tile_shape[1]):
            cell = {**namespace, plan.row: _Int32(row), plan.column: _Int32(column)}
            actual = tuple(int(eval(text, cell)) for text in plan.indices)
            assert all(
                0 <= index < extent
                for index, extent in zip(actual, plan.shape, strict=True)
            )
            assert plan.mask is None or bool(eval(plan.mask, cell))
            flat = sum(
                index * stride
                for index, stride in zip(actual, plan.strides, strict=True)
            )
            assert flat == (origin[0] + row) * plan.pitch + origin[1] + column
    return True


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("permuted", [False, True])
def test_exact_rectangle_origins_permuted_compact_axes_and_full_mask(dtype, permuted):
    indices = ("head + c", "start + r") if permuted else ("start + r", "head + c")
    plan = _plan(
        indices,
        strides=(1, 64) if permuted else (64, 1),
        dtype=dtype,
        mask="(start + r < end) & (head + c < 64)",
    )
    assert plan is not None and plan.view_shape == (64, 64) and plan.pitch == 64
    for start in (0, 3, 31, 60, 61):
        for head in (0, 4, 8, 24, 56, 60):
            expected = start <= 60 and head <= 56 and head % (16 // dtype.itemsize) == 0
            assert _rectangle(plan, start=start, head=head, end=64) == expected
    namespace = _environment(start=3, head=24, end=64)
    assert tuple(int(eval(text, namespace)) for text in plan.origin) == (3, 24)
    with pytest.raises(FrozenInstanceError):
        plan.pitch = 0  # pyrefly: ignore[read-only]


def test_uniform_wrapping_quotient_modulo_and_narrowing_are_not_reassociated():
    expression = "((cutlass.Int32(capture) * 1073741824) // 32) % 64"
    plan = _plan(("temporary + r", "head + c"), definitions={"temporary": expression})
    assert plan is not None
    assert plan.indices[0] == ast.unparse(
        ast.parse(f"({expression}) + r", mode="eval").body
    )
    assert _rectangle(plan, capture=4, head=8)
    assert _rectangle(plan, capture=8, head=8)
    uniform_cast = _plan(("cutlass.Int32(start) + r", "head + c"))
    varying_cast = _plan(("cutlass.Int32(start + r)", "head + c"))
    assert uniform_cast is not None and varying_cast is not None
    assert _rectangle(uniform_cast, start=_Int64(1 << 32), head=0)
    assert not _rectangle(varying_cast, start=_Int64(1 << 32), head=0)


@pytest.mark.parametrize("cast", ["cutlass.Int32", "cutlass.Int64"])
def test_every_varying_intermediate_is_bounded_even_when_endpoints_cancel(cast):
    expression = f"{cast}(r * 1073741824) - {cast}(r * 1073741824) + start + r"
    plan = _plan((expression, "head + c"), tile_shape=(5, 8))
    assert plan is not None
    namespace = _environment(start=0, head=0, r=0)
    assert int(eval(plan.indices[0], namespace)) == 0
    namespace["r"] = _Int32(4)
    assert int(eval(plan.indices[0], namespace)) == 4
    assert not _rectangle(plan, start=0, head=0)


def test_guard_rejects_int32_overflow_before_later_widening_or_cancellation():
    plan = _plan(("(cutlass.Int32(start) + r) - cutlass.Int64(start) + r", "head + c"))
    assert plan is not None and plan.pitch == 128
    assert _rectangle(plan, start=0, head=0)
    assert not _rectangle(plan, start=(1 << 31) - 2, head=0)


def test_original_uniform_subtree_and_varying_source_order_remain_distinct():
    uniform = _plan(("r + (start + other)", "head + c"))
    varying = _plan(("(r + start) + other", "head + c"))
    assert uniform is not None and varying is not None
    values = {"start": (1 << 31) - 1, "other": -((1 << 31) - 1), "head": 0}
    assert _rectangle(uniform, **values)
    assert not _rectangle(varying, **values)


@pytest.mark.parametrize(
    "value",
    [
        -(1 << 63),
        -(1 << 31) - 1,
        -(1 << 31),
        -1,
        0,
        (1 << 31) - 1,
        1 << 31,
        (1 << 63) - 1,
    ],
)
def test_eager_failed_guard_algebra_does_not_overflow_int64(value):
    plan = _plan(("start + r", "head + c"), mask="start + r < end")
    assert plan is not None
    assert (
        _rectangle(plan, start=_Int64(value), head=_Int64(value), end=_Int64(value))
        is False
    )


@pytest.mark.parametrize(
    "mask,enabled",
    [
        ("True", True),
        ("False", False),
        ("r + start < end", False),
        ("operator.le(r + start, end)", True),
        ("not (r + start > end)", True),
        ("(r + start < end) | (head + c < 64)", True),
        ("(r + start < end) & (head + c < 64)", False),
        ("0 <= r + start <= end", True),
        ("r + start != end", False),
        ("r - r == 0", True),
        ("head - c >= 0", True),
        ("head - c > 0", True),
        ("c - head <= 0", True),
        ("not (head - c < 0)", True),
        ("(r < 2) | (r >= 2)", False),
        ("not ((r < 2) and (r >= 2))", False),
    ],
)
def test_full_mask_not_just_tensor_oob_and_conservative_boolean_proof(mask, enabled):
    plan = _plan(mask=mask)
    assert plan is not None
    assert _rectangle(plan, start=3, head=8, end=6) == enabled


@pytest.mark.parametrize(
    "expression",
    [
        "start + r // 2",
        "start + r % 4",
        "start + (r & 1)",
        "r * capture",
        "start + (r << 1)",
        "start + (r if c < 4 else c)",
        "tensor[r]",
        "unknown + r",
        "arbitrary(r)",
        "r + (capture // other)",
        "r + (capture % 0)",
        "r + (capture // -1)",
        "r + (capture << 32)",
        "r + (capture >> -1)",
        "r * 9223372036854775807",
        "r + (",
    ],
)
def test_unsupported_or_undefined_forms_fail_closed(expression):
    assert _plan((expression, "head + c")) is None


@pytest.mark.parametrize(
    "definitions",
    [
        {"temporary": "temporary"},
        {"temporary": "other", "other": "temporary"},
        {"temporary": "unknown"},
        {"temporary": "tensor[0]"},
        {"temporary": "capture + r % 4"},
    ],
)
def test_definitions_expand_before_uniform_annotations(definitions):
    assert _plan(("temporary + r", "head + c"), definitions=definitions) is None


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": torch.int32},
        {"dtype": torch.bool},
        {"dtype": torch.float64},
        {"shape": (0, 64)},
        {"tile_shape": (0, 8)},
        {"tile_shape": (65, 8)},
        {"tile_shape": (4, 65)},
        {"strides": (65, 1)},
        {"strides": (64, 0)},
        {"strides": (-64, 1)},
        {"shape": (True, 64)},
        {"shape": (65536, 65536), "strides": (65536, 1)},
        {"shape": (64, 10), "strides": (10, 1)},
    ],
)
def test_metadata_descriptor_geometry_and_dtype_fail_closed(options):
    assert _plan(**options) is None


def test_rank_three_flattening_head_origins_and_axis_bounds():
    plan = _plan(
        ("capture", "start + r", "head + c"), shape=(3, 64, 64), strides=(4096, 64, 1)
    )
    assert plan is not None and plan.view_shape == (192, 64)
    assert _rectangle(plan, capture=2, start=3, head=24)
    assert tuple(
        int(eval(text, _environment(capture=2, start=3, head=24)))
        for text in plan.origin
    ) == (131, 24)
    # A descriptor rectangle could cross a host-axis boundary, but the
    # original row index must not cross from one head into the next.
    assert not _rectangle(plan, capture=0, start=62, head=0)
    assert not _rectangle(plan, capture=3, start=0, head=0)


def test_large_coefficient_guard_envelope_rejected_before_emission():
    expression = "((start + r) * 2147483647) * 2147483647 - ((start + r) * 2147483647) * 2147483647 + r"
    assert _plan((expression, "head + c")) is None


def test_singleton_and_unaligned_origin_fallback():
    plan = _plan(tile_shape=(1, 1))
    assert plan is not None
    assert _rectangle(plan, start=63, head=56)
    assert not _rectangle(plan, start=63, head=57)


@pytest.mark.parametrize(
    "row",
    [
        "start + r",
        "cutlass.Int32(start + r)",
        "cutlass.Int64(start) + r",
        "start + (2 * r - r)",
        "start - (-r)",
        "start + (r + c - c)",
    ],
)
def test_all_accepted_cells_preserve_typed_affine_expression_order(row):
    plan = _plan((row, "head + c"), mask="start + r < end")
    assert plan is not None
    accepted = 0
    for start in (-1, 0, 3, 60, 61, (1 << 31) - 2, 1 << 32):
        for head in (0, 8, 56, 60):
            for end in (3, 63, 64):
                accepted += _rectangle(plan, start=_Int64(start), head=head, end=end)
    assert accepted > 0


@pytest.mark.parametrize(
    "shape,strides",
    [
        ((1, 64, 64), (4096, 64, 1)),
        ((64, 1, 64), (64, 1, 1)),
    ],
)
def test_unit_extent_compact_axes_do_not_imply_a_padded_host(shape, strides):
    indices = (
        ("0", "start + r", "head + c")
        if shape[0] == 1
        else ("start + r", "0", "head + c")
    )
    plan = _plan(indices, shape=shape, strides=strides)
    assert plan is not None
    assert _rectangle(plan, start=3, head=24)


def test_large_compact_pitch_keeps_rejected_runtime_guard_products_in_int64():
    plan = _plan(shape=(2, 1 << 28), strides=(1 << 28, 1), tile_shape=(1, 8))
    assert plan is not None
    for value in (-(1 << 63), -(1 << 31), (1 << 31) - 1, (1 << 63) - 1):
        assert not _rectangle(plan, start=_Int64(value), head=_Int64(value))
    assert _rectangle(plan, start=1, head=24)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_actual_cute_dynamic_guard_and_descriptor_origins_verify_without_cuda(dtype):
    import cutlass
    import cutlass.cute as cute

    plan = _plan(
        ("temporary + r", "cutlass.Int32(head) + c"),
        definitions={"temporary": "(cutlass.Int32(capture) // 32) % 64"},
        dtype=dtype,
        mask="not (temporary + r >= end)",
    )
    assert plan is not None
    ir = importlib.import_module("cutlass._mlir.ir")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            namespace = {
                "cutlass": cutlass,
                "operator": operator,
                "capture": cutlass.Int64(cute.arch.block_idx()[0]),
                "head": cutlass.Int64(cute.arch.block_idx()[1]),
                "end": cutlass.Int64(cute.arch.block_idx()[2]),
            }
            enabled = eval(plan.guard, namespace)
            assert isinstance(enabled, cutlass.Boolean)
            pointer = cute.make_ptr(
                cutlass.Int64, 128, cute.AddressSpace.gmem, assumed_align=16
            )
            pointer.store(cutlass.Int64(enabled))
            for index, expression in enumerate(plan.origin, 1):
                (pointer + index).store(eval(expression, namespace))
        assert module.operation.verify()
        source = str(module)
        assert "arith.trunci" in source and "arith.extsi" in source
        assert "arith.andi" in source and "llvm.store" in source
    assert torch.cuda.is_initialized() is before
