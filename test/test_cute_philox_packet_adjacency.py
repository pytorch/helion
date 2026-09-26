from __future__ import annotations

import ast
import functools
import itertools

import pytest

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import func

from helion._compiler.cute.philox_packets import _aligned_packet_adjacency
from helion._compiler.cute.philox_packets import lower_philox_packets
from helion._compiler.cute.philox_stream import SCALAR_ASM
from helion._compiler.cute.philox_stream import SCALAR_CONSTRAINTS

CORE = "cutlass.Int64(base + cutlass.Int32(lane))"
PROVED = (
    CORE,
    "base + cutlass.Int32(lane)",
    "cutlass.Int32(lane) + base",
    "cutlass.Int64(cutlass.Int32(base) + cutlass.Int32(lane))",
    "cutlass.Uint64(cutlass.Uint32(base) + cutlass.Uint32(lane))",
    "cutlass.Int64(cutlass.Uint32(base) + cutlass.Int32(lane))",
    "cutlass.Int64(cutlass.Int32(cutlass.Uint64(base) + cutlass.Uint64(lane)))",
    "cutlass.Uint32(cutlass.Int64(base) + cutlass.Int32(lane))",
    "cutlass.Uint64(cutlass.Int64(cutlass.Int32(base) + cutlass.Int32(lane)))",
    f"({CORE}) * 0 + ({CORE}) * 1",
    f"1 * ({CORE}) + 0 * ({CORE})",
    f"({CORE}) * 1",
)


def _proof(expression):
    return _aligned_packet_adjacency(
        ast.parse(expression, mode="eval").body, "lane", {"base"}
    )


@functools.lru_cache
def _integer_ir(expression, dtype, width):
    # Actual dynamic CuTe lowering, rather than host-static Uint64 arithmetic
    # which can raise Python OverflowError instead of exhibiting device wrap.
    # This builds only scalar integer MLIR: no native compilation, kernel,
    # CUDA context or tensor allocation.
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("packet_index", ([dtype.mlir_type], []))
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                base = dtype(block.arguments[0])
                code = compile(expression, "<typed-packet-index>", "eval")
                offsets = [
                    eval(code, {"cutlass": cutlass}, {"base": base, "lane": lane})
                    for lane in range(width)
                ]
                alignments = [offsets[start] & 3 == 0 for start in range(0, width, 4)]
                equalities = [
                    offsets[start + delta] == offsets[start] + delta
                    for start in range(0, width, 4)
                    for delta in range(1, 4)
                ]
                values = [v.ir_value() for v in (*alignments, *equalities)]
                func.ReturnOp(values)
                fn.attributes["function_type"] = ir.TypeAttr.get(
                    ir.FunctionType.get([dtype.mlir_type], [v.type for v in values])
                )
        assert module.operation.verify()
        ids = {block.arguments[0]: 0}
        instructions = []
        for operation in block.operations:
            operands = [ids[value] for value in operation.operands]
            if operation.name == "func.return":
                return instructions, operands, dtype.width
            assert len(operation.results) == 1
            value = operation.results[0]
            assert isinstance(value.type, ir.IntegerType)
            assert all(isinstance(v.type, ir.IntegerType) for v in operation.operands)
            ids[value] = len(ids)
            extra = None
            if operation.name == "arith.constant":
                extra = ir.IntegerAttr(operation.attributes["value"]).value
            elif operation.name == "arith.cmpi":
                extra = ir.IntegerAttr(operation.attributes["predicate"]).value
                assert extra == 0  # Equality only; no signedness is discarded.
            else:
                assert operation.name in {
                    "arith.addi",
                    "arith.muli",
                    "arith.andi",
                    "arith.bitcast",
                    "arith.extsi",
                    "arith.extui",
                    "arith.trunci",
                }, operation.name
            instructions.append((operation.name, operands, value.type.width, extra))
    raise AssertionError("missing scalar return")


def _execute_integer_ir(plan, bits):
    instructions, returns, width = plan
    values = [(bits % (1 << width), width)]
    for name, operands, width, extra in instructions:
        args = [values[index] for index in operands]
        if name == "arith.constant":
            value = extra
        elif name == "arith.addi":
            value = args[0][0] + args[1][0]
        elif name == "arith.muli":
            value = args[0][0] * args[1][0]
        elif name == "arith.andi":
            value = args[0][0] & args[1][0]
        elif name == "arith.cmpi":
            value = int(args[0] == args[1])
        elif name == "arith.extsi":
            value, source_width = args[0]
            if value >= 1 << (source_width - 1):
                value -= 1 << source_width
        elif name == "arith.bitcast":
            value, source_width = args[0]
            assert source_width == width
        else:
            assert name in {"arith.extui", "arith.trunci"}
            value = args[0][0]
        values.append((value % (1 << width), width))
    return [values[index][0] for index in returns]


@pytest.mark.parametrize("expression", PROVED)
@pytest.mark.parametrize("width", (4, 8))
def test_aligned_proof_with_actual_sdk_integer_promotion(expression, width):
    assert _proof(expression)
    boundaries = {0, 1 << 31, 1 << 32, 1 << 63, 1 << 64}
    values = sorted(
        {
            (boundary + delta) % (1 << 64)
            for boundary in boundaries
            for delta in range(-7, 8)
        }
    )
    for dtype in (cutlass.Int32, cutlass.Uint32, cutlass.Int64, cutlass.Uint64):
        plan = _integer_ir(expression, dtype, width)
        for bits in values:
            checks = _execute_integer_ir(plan, bits)
            groups = width // 4
            for group in range(groups):
                if checks[group]:
                    assert all(checks[groups + 3 * group : groups + 3 * group + 3]), (
                        expression,
                        dtype,
                        bits,
                        group,
                    )


SHIFTED = (
    ("cutlass.Int64(cutlass.Int32(base + cutlass.Int32(lane))) + 2", 1 << 31),
    (
        "cutlass.Int64(cutlass.Int32(base + cutlass.Int32(lane))) + cutlass.Int64(2)",
        1 << 31,
    ),
    (
        "cutlass.Int64(cutlass.Int32(base + cutlass.Int32(lane)) + cutlass.Int64(2))",
        1 << 31,
    ),
    ("cutlass.Uint64(cutlass.Uint32(base + cutlass.Uint32(lane))) + 2", 1 << 32),
    (
        "cutlass.Int64(cutlass.Int32(cutlass.Int64(base) + cutlass.Int64(lane))) + 2",
        1 << 31,
    ),
)


def _lower(expression, width):
    tree = ast.parse(
        f"""for lane in cutlass.range_constexpr({width}):
    offset = {expression}
    random = _cute_inline_asm_elementwise((seed[0], offset), asm={SCALAR_ASM!r}, constraints={SCALAR_CONSTRAINTS!r}, dtype=cutlass.Float32, is_pure=True)
    result = random
"""
    )
    names = itertools.count()
    lowered = lower_philox_packets(
        tree.body,
        seed_names={"seed"},
        integer_names={"base"},
        new_name=lambda prefix: f"{prefix}_{next(names)}",
    )
    return tree, ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[]))


@pytest.mark.parametrize("expression,boundary", SHIFTED)
@pytest.mark.parametrize("start", (0, 4))
def test_shifted_narrow_wrap_retains_runtime_equalities(expression, boundary, start):
    assert not _proof(expression)
    checks = _execute_integer_ir(
        _integer_ir(expression, cutlass.Int64, 8), boundary - 2 - start
    )
    group = start // 4
    assert checks[group]
    assert not all(checks[2 + 3 * group : 2 + 3 * group + 3])
    original, lowered = _lower(expression, 8)
    branch = lowered.body[0]
    assert isinstance(branch, ast.If)
    assert (
        len([node for node in ast.walk(branch.test) if isinstance(node, ast.Compare)])
        == 8
    )
    assert ast.dump(branch.orelse[0]) == ast.dump(original.body[0])


@pytest.mark.parametrize("width", (4, 8))
def test_only_equality_guards_change_and_scalar_fallback_is_exact(width):
    original, lowered = _lower(PROVED[9], width)
    branch = lowered.body[0]
    assert isinstance(branch, ast.If)
    comparisons = [
        node for node in ast.walk(branch.test) if isinstance(node, ast.Compare)
    ]
    assert len(comparisons) == width // 4
    assert all(
        isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.BitAnd)
        for node in comparisons
    )
    assert ast.dump(branch.orelse[0]) == ast.dump(original.body[0])
    compile(lowered, "<packet-adjacency-ast>", "exec")
    # No memory vectorization callback is involved in this proof. The same
    # register tuple feeds the same scalar body and all memory/tail predicates
    # are left to the unchanged existing affine vectorizer.
    assert isinstance(branch.body[-1], ast.For)


@pytest.mark.parametrize(
    "expression",
    (
        "cutlass.Int64(base + 2 * cutlass.Int32(lane))",
        "cutlass.Int64(base - cutlass.Int32(lane))",
        "cutlass.Int64(base + cutlass.Int32(lane)) + 1",
        "cutlass.Int64(cutlass.Int32(base + cutlass.Int32(lane)) + 2)",
        "cutlass.Int64(base + cutlass.Float32(lane))",
        "cutlass.Int64(unknown + cutlass.Int32(lane))",
        f"({CORE}) * 0 + cutlass.Int64(base + 2 * cutlass.Int32(lane)) * 1",
        f"({CORE}) * 0.0 + ({CORE}) * 1.0",
    ),
)
def test_unproved_or_differently_typed_expressions_decline(expression):
    assert not _proof(expression)


def test_small_width_exhaustive_cast_boundary_lemma():
    def cast(value, width, signed):
        value %= 1 << width
        return value - (1 << width) if signed and value >= 1 << (width - 1) else value

    for width in range(3, 9):
        for sign1, sign2 in itertools.product((False, True), repeat=2):
            for base in range(-(1 << width), 1 << width):
                offsets = [
                    cast(cast(base + lane, width, sign1), width + 1, sign2)
                    for lane in range(8)
                ]
                for start in (0, 4):
                    if offsets[start] % 4 == 0:
                        assert offsets[start : start + 4] == list(
                            range(offsets[start], offsets[start] + 4)
                        )
