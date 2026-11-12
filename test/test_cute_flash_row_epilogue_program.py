"""The flash row epilogue's program matcher on hand-built FX graphs.

The program computes in fp32; a 16-bit value takes the roundings torch gives
it: a cast after an op with a 16-bit result (unless the op only selects values
the type holds), a 16-bit common dtype for a binary op's operands, and in a
compare a rounded scalar.
"""

from __future__ import annotations

import pytest
import torch

from helion._compiler.cute import flash_row_epilogue
from helion._testing.flash_row_programs import casts_of
from helion._testing.flash_row_programs import match_row_program
from helion._testing.flash_row_programs import program_consts
from helion._testing.flash_row_programs import row_accumulator
from helion._testing.flash_row_programs import row_value

pytestmark = pytest.mark.cpu_only

aten = torch.ops.aten


def test_a_bf16_compare_rounds_its_scalar_operand() -> None:
    # Torch compares ``bf16 > 0.3`` against ``bf16(0.3)`` = 0.30078125.  A
    # compare of two bf16 values needs no rounding.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    narrow = row_value(
        graph,
        torch.ops.prims.convert_element_type.default,
        (o, torch.bfloat16),
        torch.bfloat16,
    )
    mask = row_value(graph, aten.gt.Scalar, (narrow, 0.3), torch.bool)
    value = row_value(graph, aten.where.self, (mask, o, 0.0), torch.float32)
    program = match_row_program(graph, o, value)
    assert program is not None
    assert 0.30078125 in program_consts(program) and 0.3 not in program_consts(program)

    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    narrow = row_value(
        graph,
        torch.ops.prims.convert_element_type.default,
        (o, torch.bfloat16),
        torch.bfloat16,
    )
    halved = row_value(graph, aten.div.Tensor, (o, 2.0), torch.float32)
    other = row_value(
        graph,
        torch.ops.prims.convert_element_type.default,
        (halved, torch.bfloat16),
        torch.bfloat16,
    )
    mask = row_value(graph, aten.gt.Tensor, (narrow, other), torch.bool)
    value = row_value(graph, aten.where.self, (mask, o, 0.0), torch.float32)
    program = match_row_program(graph, o, value)
    assert program is not None
    assert casts_of(program, "o_norm") == [torch.bfloat16]
    assert casts_of(program, "gt") == []


def test_a_bf16_op_rounds_its_result_unless_it_selects_held_values() -> None:
    # An op whose result is one of its value operands, their negation or
    # magnitude (relu, abs, neg, maximum, minimum, clamp, where, amax, amin)
    # gives a value bf16 holds when bf16 holds each operand, a literal bound
    # or branch included, and a 0-d fp32 tensor beside bf16, which the op
    # converts to bf16 first.  exp, sum, a clamp to 0.3 and a where with a
    # 0.3 branch round their result.
    graph = torch.fx.Graph()
    x = row_value(graph, "x", (), torch.bfloat16)
    y = row_value(graph, "y", (), torch.bfloat16)
    wide = row_value(graph, "wide", (), torch.float32, ())
    keep = row_value(graph, "keep", (), torch.bool)

    def rounding(target: object, *args: object, rows: bool = False) -> object:
        shape = (2, 4) if rows else (2, 4, 8)
        node = row_value(graph, target, args, torch.bfloat16, shape)
        return flash_row_epilogue._rounded_result(node)

    for target in (aten.relu.default, aten.abs.default, aten.neg.default):
        assert rounding(target, x) is None
    assert rounding(aten.maximum.default, x, y) is None
    assert rounding(aten.minimum.default, x, y) is None
    assert rounding(aten.clamp.default, x, 0.0, 6.0) is None
    assert rounding(aten.clamp_min.default, x, 0.5) is None
    assert rounding(aten.where.self, keep, x, 0.0) is None
    assert rounding(aten.amax.default, x, [-1], rows=True) is None
    assert rounding(aten.amin.default, x, [-1], rows=True) is None
    assert rounding(aten.exp.default, x) is torch.bfloat16
    assert rounding(aten.sum.dim_IntList, x, [-1], rows=True) is torch.bfloat16
    assert rounding(aten.clamp.default, x, 0.3) is torch.bfloat16
    assert rounding(aten.maximum.default, x, wide) is None
    assert rounding(aten.where.self, keep, x, 0.3) is torch.bfloat16


def test_a_wider_operand_is_rounded_to_the_common_dtype() -> None:
    # ``o.to(bf16) * scale`` with a 0-d fp32 scale is a bf16 product: torch
    # rounds the scale to bf16 first, and the product after.  A product the
    # store reads takes the store's rounding instead of a cast of its own.
    for feeds_store in (True, False):
        graph = torch.fx.Graph()
        o = row_accumulator(graph)
        narrow = row_value(
            graph,
            torch.ops.prims.convert_element_type.default,
            (o, torch.bfloat16),
            torch.bfloat16,
        )
        scale = row_value(graph, aten.scalar_tensor.default, (0.1,), torch.float32, ())
        product = row_value(graph, aten.mul.Tensor, (narrow, scale), torch.bfloat16)
        if feeds_store:
            program = match_row_program(graph, o, product, torch.bfloat16)
        else:
            value = row_value(graph, aten.add.Tensor, (product, o), torch.float32)
            program = match_row_program(graph, o, value)
        assert program is not None
        assert casts_of(program, "const") == [torch.bfloat16]
        if feeds_store:
            assert program.op(program.output).op == "mul"
        else:
            output = program.op(program.output)
            assert output.op == "add"
            rounded = program.op(output.inputs[0])
            assert rounded.op == "cast" and rounded.attrs == (torch.bfloat16,)
            assert program.op(rounded.inputs[0]).op == "mul"


def test_a_wider_operand_read_twice_is_converted_once() -> None:
    # Two bf16 ops reading one 0-d fp32 scale share its conversion to bf16.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    narrow = row_value(
        graph,
        torch.ops.prims.convert_element_type.default,
        (o, torch.bfloat16),
        torch.bfloat16,
    )
    scale = row_value(graph, aten.scalar_tensor.default, (0.1,), torch.float32, ())
    product = row_value(graph, aten.mul.Tensor, (narrow, scale), torch.bfloat16)
    total = row_value(graph, aten.add.Tensor, (product, scale), torch.bfloat16)
    program = match_row_program(graph, o, total, torch.bfloat16)
    assert program is not None
    assert casts_of(program, "const") == [torch.bfloat16]


def test_a_boolean_operand_of_float_arithmetic_is_converted_to_fp32() -> None:
    # o * (o > 0.3) multiplies by 0.0 or 1.0 in fp32, as torch promotes the
    # compare; a where's condition stays Boolean.  Boolean arithmetic
    # (a mask times a mask) is no row program.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    mask = row_value(graph, aten.gt.Scalar, (o, 0.3), torch.bool)
    product = row_value(graph, aten.mul.Tensor, (o, mask), torch.float32)
    value = row_value(graph, aten.where.self, (mask, product, 0.0), torch.float32)
    program = match_row_program(graph, o, value)
    assert program is not None
    assert casts_of(program, "gt") == [torch.float32]
    where = program.op(program.output)
    assert where.op == "where" and program.op(where.inputs[0]).op == "gt"
    product_op = program.op(where.inputs[1])
    assert product_op.op == "mul" and program.op(product_op.inputs[1]).op == "cast"

    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    mask = row_value(graph, aten.gt.Scalar, (o, 0.3), torch.bool)
    both = row_value(graph, aten.mul.Tensor, (mask, mask), torch.bool)
    value = row_value(graph, aten.where.self, (both, o, 0.0), torch.float32)
    assert match_row_program(graph, o, value) is None


def test_a_bool_op_with_a_list_argument_falls_back() -> None:
    # A bool op of two arguments that is no compare (a permute of a mask) is
    # no row program node: the matcher returns None, and the kernel takes the
    # scalar path, as for any op the program lacks.  amax of bools and an any
    # over listed dims round nothing either.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    mask = row_value(graph, aten.gt.Scalar, (o, 0.0), torch.bool)
    moved = row_value(graph, aten.permute.default, (mask, [0, 1, 2]), torch.bool)
    value = row_value(graph, aten.where.self, (moved, o, 0.0), torch.float32)
    assert match_row_program(graph, o, value) is None
    assert [moved] == flash_row_epilogue.LAST_REJECTED
    for target, dims in ((aten.amax.default, [-1]), (aten.any.dims, [2])):
        reduced = row_value(graph, target, (mask, dims), torch.bool, (2, 4))
        assert flash_row_epilogue._rounded_result(reduced) is None


def test_a_bf16_tensor_constant_holds_its_rounded_value() -> None:
    # Torch multiplies by the bf16 constant bf16(0.3) = 0.30078125, so the
    # fp32 program does too; an fp32 constant keeps its literal.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    weight = row_value(graph, aten.full.default, ([2, 4, 8], 0.3), torch.bfloat16)
    eps = row_value(graph, aten.full.default, ([2, 4, 8], 1e-6), torch.float32)
    scaled = row_value(graph, aten.mul.Tensor, (o, weight), torch.float32)
    value = row_value(graph, aten.add.Tensor, (scaled, eps), torch.float32)
    program = match_row_program(graph, o, value)
    assert program is not None
    assert program_consts(program) == [0.30078125, 1e-6]


def test_an_overflowing_half_constant_renders_as_inf() -> None:
    # ``scalar_tensor(1e5, dtype=float16)`` is inf, which repr spells as a
    # bare name: the program renders it as a float('inf') literal.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    big = row_value(graph, aten.scalar_tensor.default, (1e5,), torch.float16, ())
    value = row_value(graph, aten.minimum.default, (o, big), torch.float32)
    program = match_row_program(graph, o, value)
    assert program is not None
    assert program_consts(program) == [float("inf")]
    _prologue, epilogue = flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=1,
        chunk_width=8,
        elem_var="j",
        o_alloc="{o} = alloc()",
        o_load=["load_o({i}, {o})"],
        o_elem="{o}[{j}]",
        aux_allocs=[],
        aux_loads=[],
        aux_elems=[],
        out_alloc="{out} = alloc()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=[],
        indent="",
        prefix="e",
    )
    assert "cutlass.Float32(float('inf'))" in epilogue
    assert "Float32(inf)" not in epilogue
