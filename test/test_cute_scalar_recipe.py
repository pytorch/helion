from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import operator
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _body(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _fresh() -> Callable[[str], str]:
    counter = itertools.count()
    return lambda hint: f"_test_{next(counter)}_{hint}"


def _f16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _i32(value: int) -> int:
    return (int(value) + 2**31) % 2**32 - 2**31


_CUTLASS = SimpleNamespace(
    Float16=_f16,
    Float32=_f32,
    Int32=_i32,
    Int64=int,
    Uint8=lambda value: int(value) % 256,
    Boolean=bool,
    min=min,
    max=max,
)


@dataclasses.dataclass
class _Pointer:
    values: list[float | int]
    reads: list[int]
    offset: int = 0

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.values, self.reads, self.offset + offset)

    def load(self) -> float | int:
        assert 0 <= self.offset < len(self.values), "unmasked out-of-bounds read"
        self.reads.append(self.offset)
        return self.values[self.offset]


def _tensor(values: list[float | int], shape: tuple[int, ...]) -> SimpleNamespace:
    stride = (shape[1], 1) if len(shape) == 2 else (1,)
    return SimpleNamespace(
        iterator=_Pointer(values, []),
        layout=SimpleNamespace(shape=shape, stride=stride),
        shape=shape,
        stride=stride,
    )


def _evaluate(
    statements: Sequence[ast.stmt], value: ast.expr, arguments: Mapping[str, object]
) -> object:
    module = ast.Module(
        body=[
            *(ast.parse(ast.unparse(statement)).body[0] for statement in statements),
            ast.Assign(
                targets=[ast.Name(id="_result", ctx=ast.Store())],
                value=_expr(ast.unparse(value)),
            ),
        ],
        type_ignores=[],
    )
    namespace = {
        "cutlass": _CUTLASS,
        "math": math,
        "operator": operator,
        **arguments,
    }
    exec(
        compile(ast.fix_missing_locations(module), "<scalar-recipe-test>", "exec"),
        namespace,
    )
    return namespace["_result"]


@pytest.mark.parametrize("m,k", [(-1, 0), (0, 0), (1, 3), (2, 2), (3, 1), (0, 4)])
def test_indirect_gather_replays_masks_and_casts(m: int, k: int) -> None:
    statements = _body(
        "valid = 0 <= m < indices.shape[0] and 0 <= k < A.layout.shape[1]\n"
        "row = (indices.iterator + m).load() if valid else cutlass.Int32(0)\n"
        "offset = cutlass.Int64(row) * A.layout.stride[0] + k * A.layout.stride[1]\n"
        "loaded = (A.iterator + offset).load() if valid else cutlass.Float16(0)\n"
        "rounded = cutlass.Float16(loaded * scale)\n"
        "result = cutlass.Float32(rounded) + cutlass.Float32(bias)\n"
    )
    inputs = {
        "A": _tensor([1.0006, -0.0, 3.25, -7.125, 2.999, 9.5, -4.4, 0.002], (2, 4)),
        "indices": _tensor([1, 0, 1], (3,)),
        "m": m,
        "k": k,
        "scale": 0.3333,
        "bias": -0.17,
    }
    expected = _evaluate(statements, _expr("result"), inputs)
    memory = [inputs["A"].iterator, inputs["indices"].iterator]  # type: ignore[union-attr]
    expected_reads = [list(item.reads) for item in memory]
    for item in memory:
        item.reads.clear()
    recipe = build_recipe(_expr("result"), statements, inputs.keys())
    assert recipe is not None
    replay, value = recipe.emit({"m": _expr("copy_m"), "k": _expr("copy_k")}, _fresh())
    actual = _evaluate(
        replay, value, {**inputs, "m": -100, "k": -100, "copy_m": m, "copy_k": k}
    )
    assert struct.pack("d", actual) == struct.pack("d", expected)  # type: ignore[arg-type]
    assert [item.reads for item in memory] == expected_reads


def test_shared_load_dependency_is_evaluated_once() -> None:
    statements = _body("x = (A.iterator + k).load()\ny = x * x\n")
    tensor = _tensor([2.0, 3.0], (2,))
    recipe = build_recipe(_expr("y + x"), statements, {"A", "k"})
    assert recipe is not None
    replay, value = recipe.emit({}, _fresh())
    assert _evaluate(replay, value, {"A": tensor, "k": 1}) == 12.0
    assert tensor.iterator.reads == [1]


@pytest.mark.parametrize("valid", [False, True])
def test_quantized_byte_load_and_decode_preserve_masked_reads(valid: bool) -> None:
    statements = _body(
        "raw = cute.arch.load(A.iterator + index, cutlass.Uint8) "
        "if valid else cutlass.Uint8(0)\n"
        "pair = _cute_float4_e2m1fn_x2_to_float32(raw)\n"
        "scale = _cute_fp8e4m3fn_to_float32(raw)\n"
    )
    recipe = build_recipe(
        _expr("pair[0] + pair[1] * scale"),
        statements,
        {"A", "index", "valid"},
    )
    assert recipe is not None
    replay, value = recipe.emit({"index": _expr("copy_index")}, _fresh())
    tensor = _tensor([0x21], (1,))
    actual = _evaluate(
        replay,
        value,
        {
            "A": tensor,
            "copy_index": 0 if valid else 999,
            "valid": valid,
            "cute": SimpleNamespace(
                arch=SimpleNamespace(load=lambda pointer, dtype: pointer.load())
            ),
            "_cute_float4_e2m1fn_x2_to_float32": lambda raw: (raw & 15, raw >> 4),
            "_cute_fp8e4m3fn_to_float32": float,
        },
    )
    assert actual == (67.0 if valid else 0.0)
    assert tensor.iterator.reads == ([0] if valid else [])


@pytest.mark.parametrize(
    "helper", ["_cute_float4_e2m1fn_x2_to_float32", "_cute_fp8e4m3fn_to_float32"]
)
def test_rebound_numeric_helper_cannot_hide_an_effect(helper: str) -> None:
    assert (
        build_recipe(
            _expr(f"{helper}(value)"),
            _body(f"{helper} = untrusted"),
            {"value", "untrusted"},
        )
        is None
    )


def test_shadowed_assignments_keep_original_dependencies() -> None:
    statements = _body("x = a + 1\ny = x * 2\nx = a + 10\nz = y + x\n")
    recipe = build_recipe(_expr("z"), statements, {"a"})
    assert recipe is not None
    replay, value = recipe.emit({}, _fresh())
    assert _evaluate(replay, value, {"a": 3}) == 21
    names = [statement.targets[0].id for statement in replay]  # type: ignore[attr-defined]
    assert len(names) == len(set(names)) == 4


def test_shadowed_pointer_offset_keeps_original_gather() -> None:
    statements = _body(
        "row = (indices.iterator + m).load()\n"
        "offset = cutlass.Int64(row) * A.stride[0]\n"
        "row = 999\n"
        "value = (A.iterator + offset + k).load()\n"
    )
    recipe = build_recipe(_expr("value"), statements, {"indices", "A", "m", "k"})
    assert recipe is not None
    replay, value = recipe.emit({}, _fresh())
    assert (
        _evaluate(
            replay,
            value,
            {
                "indices": _tensor([1], (1,)),
                "A": _tensor([1, 2, 3, 4], (2, 2)),
                "m": 0,
                "k": 1,
            },
        )
        == 4
    )


def test_boundary_cut_applies_only_to_current_version() -> None:
    statements = _body("index = 2\nold = index + 3\nindex = 99\n")
    recipe = build_recipe(_expr("old + index"), statements, {"index"})
    assert recipe is not None
    replay, value = recipe.emit({"index": _expr("replacement")}, _fresh())
    assert _evaluate(replay, value, {"replacement": 7}) == 12


def test_shadowed_external_snapshot_is_unavailable() -> None:
    statements = _body("old = index + 3\nindex = 99\n")
    assert build_recipe(_expr("old"), statements, {"index"}) is None


def test_architecture_indices_are_pure_and_substitutable() -> None:
    statements = _body(
        "lane = cutlass.Int32(cute.arch.thread_idx()[0])\n"
        "origin = cutlass.Int32(cute.arch.block_idx()[1]) * BLOCK\n"
        "index = origin + lane\n"
        "mask = index < width\n"
    )
    recipe = build_recipe(
        _expr("index if mask else -1"), statements, {"BLOCK", "width", "lane"}
    )
    assert recipe is not None
    replay, value = recipe.emit({"lane": _expr("other_lane")}, _fresh())
    fake_cute = SimpleNamespace(arch=SimpleNamespace(block_idx=lambda: (0, 2, 0)))
    assert (
        _evaluate(
            replay,
            value,
            {"cute": fake_cute, "BLOCK": 32, "width": 80, "other_lane": 4},
        )
        == 68
    )


def test_ternary_does_not_speculate_an_inline_load() -> None:
    recipe = build_recipe(
        _expr("(A.iterator + index).load() if valid else cutlass.Float16(0)"),
        [],
        {"A", "index", "valid"},
    )
    assert recipe is not None
    statements, value = recipe.emit({}, _fresh())
    tensor = _tensor([1.0], (1,))
    assert (
        _evaluate(statements, value, {"A": tensor, "index": 999, "valid": False}) == 0
    )
    assert tensor.iterator.reads == []


def test_cute_math_keywords_and_bitcast_are_preserved() -> None:
    expression = _expr(
        "cutlass.Uint32(bits).bitcast(cutlass.Float32) + cute.math.exp2(x, fastmath=True)"
    )
    recipe = build_recipe(expression, [], {"bits", "x"})
    assert recipe is not None
    statements, value = recipe.emit({}, _fresh())
    assert not statements
    assert ast.dump(value) == ast.dump(expression)


def test_substitution_is_simultaneous_and_copies_inputs() -> None:
    expression = _expr("m + n")
    replacements = {"m": _expr("n + 1"), "n": _expr("k + 2")}
    before = (
        ast.dump(expression),
        {key: ast.dump(value) for key, value in replacements.items()},
    )
    recipe = build_recipe(expression, [], {"m", "n"})
    assert recipe is not None
    statements, value = recipe.emit(replacements, _fresh())
    assert _evaluate(statements, value, {"n": 10, "k": 20}) == 33
    assert before == (
        ast.dump(expression),
        {key: ast.dump(value) for key, value in replacements.items()},
    )
    value.left = ast.Constant(value=100)  # type: ignore[attr-defined]
    assert ast.unparse(recipe.emit(replacements, _fresh())[1]) == "n + 1 + (k + 2)"


@pytest.mark.parametrize(
    "source,value,boundaries",
    [
        ("x = absent + 1", "x", set()),
        ("y = x + 1\nx = 3", "y", set()),
        ("acc = 0\nacc = acc + x", "acc", {"x"}),
        ("acc = 0\nnext_value = acc + x\nacc = next_value", "acc", {"x"}),
        ("acc = 0\nacc += x", "acc", {"x"}),
        ("x = helper(a)", "x", {"a", "helper"}),
        ("x = 1\nfor i in range(2):\n    x = i", "x", set()),
        ("x = 1\nif condition:\n    x = 2", "x", {"condition"}),
        ("x = 1\ndel x", "x", set()),
        ("x: int = 1", "x", set()),
        ("x = y = 1", "x", set()),
        ("cutlass = other", "cutlass.Int32(x)", {"other", "x"}),
        ("int = other", "int(x)", {"other", "x"}),
    ],
)
def test_unavailable_or_mutable_definitions_are_rejected(
    source: str, value: str, boundaries: set[str]
) -> None:
    assert build_recipe(_expr(value), _body(source), boundaries) is None


@pytest.mark.parametrize(
    "expression",
    [
        "A.iterator.store(x)",
        "cute.arch.atomic_add(A.iterator, x)",
        "cute.arch.sync_threads()",
        "cute.arch.warp_reduction_sum(x)",
        "cute.arch.shuffle_sync(x, 0)",
        "cute.arch.clock64()",
        "cute.arch.load(A.iterator + x, cutlass.Uint8, volatile=True)",
        "cute.arch.load(A.iterator + x, unknown)",
        "unknown(x)",
        "x.unknown()",
        "(A.iterator + x).load(volatile=True)",
        "cute.arch.thread_idx(x)",
        "x @ x",
        "(lambda: x)()",
        "[i for i in x]",
        "(y := x)",
        "cutlass.Int32(**x)",
        "x.side_effectful_property",
    ],
)
def test_side_effects_and_unsupported_expressions_are_rejected(expression: str) -> None:
    assert build_recipe(_expr(expression), [], {"A", "x", "unknown"}) is None


@pytest.mark.parametrize(
    "load", ["A.iterator.load()", "cute.arch.load(A.iterator, cutlass.Uint8)"]
)
@pytest.mark.parametrize(
    "effect",
    [
        "B.iterator.store(1)",
        "cute.arch.sync_threads()",
        "result = unknown()",
        "for i in range(2):\n    B.iterator.store(i)",
    ],
)
def test_memory_snapshot_cannot_cross_effects(effect: str, load: str) -> None:
    statements = _body(f"x = {load}\n{effect}\n")
    assert build_recipe(_expr("x"), statements, {"A", "B"}) is None


def test_unrelated_effect_does_not_supply_a_recipe_or_hide_a_fresh_load() -> None:
    recipe = build_recipe(
        _expr("x"), _body("unknown()\nx = A.iterator.load()\n"), {"A"}
    )
    assert recipe is not None
    statements, value = recipe.emit({}, _fresh())
    assert _evaluate(statements, value, {"A": _tensor([3], (1,))}) == 3


def test_explicit_mutable_names_cannot_be_boundary_inputs() -> None:
    assert (
        build_recipe(_expr("acc + x"), [], {"acc", "x"}, mutable_names={"acc"}) is None
    )


def test_fresh_names_cannot_capture_replacement_inputs() -> None:
    recipe = build_recipe(_expr("x"), _body("x = m + 1"), {"m"})
    assert recipe is not None
    with pytest.raises(ValueError, match="unused Python identifier"):
        recipe.emit({"m": _expr("copy_m")}, lambda hint: "copy_m")


def test_rebound_math_namespace_cannot_hide_a_memory_effect() -> None:
    statements = _body(
        "x = A.iterator.load()\nmath = untrusted\nunused = math.exp(1)\n"
    )
    assert build_recipe(_expr("x"), statements, {"A", "untrusted"}) is None


def test_compiler_extended_ast_is_preserved_without_mutation() -> None:
    statement = statement_from_string("value = cutlass.Int32(index) + 1")
    expression = expr_from_string("value")
    assert isinstance(expression, ast.expr)
    before = ast.dump(statement), ast.dump(expression)
    recipe = build_recipe(expression, [statement], {"index"})
    assert recipe is not None
    replay, value = recipe.emit({"index": _expr("new_index")}, _fresh())
    assert _evaluate(replay, value, {"new_index": 2**32 + 7}) == 8
    assert before == (ast.dump(statement), ast.dump(expression))
