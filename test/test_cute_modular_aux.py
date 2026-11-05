from __future__ import annotations

from dataclasses import replace
import itertools

import pytest
import sympy
import torch

from helion._compiler.cute.modular_aux import modular_aux_load_map
from helion._compiler.cute.modular_aux import prove_modular_aux_load_map
from helion.language import memory_ops
from helion.language import tile_ops


def _mod(expression, period):
    result = sympy.Mod(expression, period)
    assert isinstance(result, sympy.Expr)
    return result


def strides(shape, first):
    return (1, shape[0]) if first == 0 else (shape[1], 1)


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("permutation", ((0, 1), (1, 0)))
@pytest.mark.parametrize("contiguous", (0, 1))
def test_modular_dense_layout_directions_and_repeated_offsets_exhaustive(
    axis, permutation, contiguous
):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    extents = (24, 10) if axis == 0 else (10, 24)
    split, other = (row, col) if axis == 0 else (col, row)
    canonical = (6, 10)
    shape = tuple(canonical[i] for i in permutation)
    indices = tuple((_mod(split, 6), other)[i] for i in permutation)
    physical = strides(shape, contiguous)
    proof = prove_modular_aux_load_map(indices, (row, col), extents, shape, physical)
    assert proof is not None
    tiles = (3, 2) if axis == 0 else (2, 3)
    assert proof.tile_reason(*tiles) is None
    offsets = set()
    for m in range(extents[0]):
        for n in range(extents[1]):
            digit = (m, n)[axis] % 6
            remaining = (m, n)[1 - axis]
            expected = sum(
                (digit, remaining)[i] * st
                for i, st in zip(permutation, physical, strict=True)
            )
            assert proof.offset(m, n) == expected
            offsets.add(expected)
    assert offsets == set(range(60))  # four logical repeats, not fake bijection
    source, coords = proof.view_source("view", "original", "tm", "tn", *tiles)
    assert "original.iterator" in source and "original.layout.stride" in source
    assert "crd2idx" not in source and "recast" not in source
    assert coords[axis] == f"cutlass.Int32({('tm', 'tn')[axis]}) % 2"
    assert coords[1 - axis] == ("tm", "tn")[1 - axis]
    assert proof.tile_reason(*((4, 2) if axis == 0 else (2, 4))) is not None


@pytest.mark.parametrize(
    "fault",
    (
        "zero",
        "negative",
        "one",
        "wrong_modulus",
        "offset",
        "other_axis",
        "unknown_symbol",
        "two_mods",
        "overlap",
        "gapped",
        "negative_stride",
        "zero_stride",
        "rank3",
        "wrong_extent",
        "bool_size",
        "overflow",
        "noninteger_root",
    ),
)
def test_pure_modular_map_negatives(fault):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    symbols = (row, col)
    extents = (256, 96)
    shape = (128, 96)
    physical = (96, 1)
    indices = (_mod(row, 128), col)
    if fault in ("zero", "negative", "one"):
        shape = ({"zero": 0, "negative": -128, "one": 1}[fault], 96)
    elif fault == "wrong_modulus":
        indices = (_mod(row, 64), col)
    elif fault == "offset":
        indices = (_mod(row + 1, 128), col)
    elif fault == "other_axis":
        indices = (_mod(col, 128), row)
    elif fault == "unknown_symbol":
        indices = (_mod(sympy.Symbol("foreign", integer=True), 128), col)
    elif fault == "two_mods":
        indices = (_mod(row, 128), _mod(col, 96))
    elif fault == "overlap":
        physical = (1, 1)
    elif fault == "gapped":
        physical = (192, 1)
    elif fault == "negative_stride":
        physical = (-96, 1)
    elif fault == "zero_stride":
        physical = (96, 0)
    elif fault == "rank3":
        shape = (1, 128, 96)
        physical = (12288, 96, 1)
    elif fault == "wrong_extent":
        extents = (256, 128)
    elif fault == "bool_size":
        shape = (True, 96)
    elif fault == "overflow":
        extents = (2**28, 96)
    elif fault == "noninteger_root":
        symbols = (sympy.Symbol("row"), col)
    assert (
        prove_modular_aux_load_map(indices, symbols, extents, shape, physical) is None
    )


@pytest.mark.parametrize("invalid_size", (128.0, True, "128"))
def test_untyped_modular_map_dimension_inputs_decline(invalid_size):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    assert (
        prove_modular_aux_load_map(
            (_mod(row, 128), col),
            (row, col),
            (256, 96),
            (invalid_size, 96),
            (96, 1),
        )
        is None
    )


def graph_case(
    axis=0, transpose=False, contiguous=1, square=False, modulo_after_view=True
):
    graph = torch.fx.Graph()
    source = graph.placeholder("aux")
    extents = (256, 96) if axis == 0 else (128, 256)
    shape = (128, extents[1 - axis])
    shape = shape[::-1] if transpose else shape
    source.meta["val"] = torch.empty_strided(
        shape, strides(shape, contiguous), dtype=torch.float32
    )
    bm, bn = (64, 32) if not square else (64, 64)
    m, n = graph.placeholder("m"), graph.placeholder("n")
    tm = graph.call_function(tile_ops.tile_index, (m,))
    tm.meta["val"] = torch.empty((bm,), dtype=torch.int32)
    tn = graph.call_function(tile_ops.tile_index, (n,))
    tn.meta["val"] = torch.empty((bn,), dtype=torch.int32)
    if not modulo_after_view:
        base = (tm, tn)[axis]
        mod = graph.call_function(torch.ops.aten.remainder.Scalar, (base, 128))
        mod.meta["val"] = base.meta["val"]
        row = graph.call_function(
            torch.ops.aten.view.default, (mod if axis == 0 else tm, [m, 1])
        )
        row.meta["val"] = torch.empty((bm, 1), dtype=torch.int32)
        col = graph.call_function(
            torch.ops.aten.view.default, (mod if axis == 1 else tn, [1, n])
        )
        col.meta["val"] = torch.empty((1, bn), dtype=torch.int32)
        digits = (row, col)
    else:
        row = graph.call_function(torch.ops.aten.view.default, (tm, [m, 1]))
        row.meta["val"] = torch.empty((bm, 1), dtype=torch.int32)
        col = graph.call_function(torch.ops.aten.view.default, (tn, [1, n]))
        col.meta["val"] = torch.empty((1, bn), dtype=torch.int32)
        base = (row, col)[axis]
        mod = graph.call_function(torch.ops.aten.remainder.Scalar, (base, 128))
        mod.meta["val"] = base.meta["val"]
        digits = (mod, col) if axis == 0 else (row, mod)
    if transpose:
        digits = digits[::-1]
    load = graph.call_function(memory_ops.load, (source, list(digits), None, None))
    load.meta["val"] = torch.empty((bm, bn), dtype=torch.float32)
    context = {
        "carrier_tile_shape": (bm, bn),
        "carrier_tile_index_nodes": (m, n),
        "carrier_global_shape": extents,
    }
    return (
        graph,
        {
            "source": source,
            "m": m,
            "n": n,
            "tm": tm,
            "tn": tn,
            "row": row,
            "col": col,
            "mod": mod,
            "load": load,
        },
        context,
    )


@pytest.mark.parametrize(
    "axis,transpose,contiguous,after_view",
    itertools.product((0, 1), (False, True), (0, 1), (False, True)),
)
def test_typed_fx_modular_axis_storage_and_index_order(
    axis, transpose, contiguous, after_view
):
    _graph, nodes, context = graph_case(
        axis, transpose, contiguous, modulo_after_view=after_view
    )
    mapping = modular_aux_load_map(nodes["load"], **context)
    assert mapping is not None and mapping.split_axis == axis
    assert mapping.period == 128 and len(mapping.owned_index_nodes) == 5
    assert nodes["load"] not in mapping.owned_index_nodes
    assert nodes["source"] not in mapping.owned_index_nodes
    assert mapping.tile_reason(64, 32) is None
    assert replace(mapping, owned_index_nodes=()).offset(200, 64) == mapping.offset(
        200, 64
    )


@pytest.mark.parametrize(
    "fault",
    (
        "mask",
        "eviction",
        "kwargs",
        "bool",
        "int",
        "dtype_result",
        "rank",
        "stride",
        "period0",
        "periodnegative",
        "periodbool",
        "periodfloat",
        "nonliteral",
        "wrong_period",
        "integer_wrap",
        "unknown",
        "double_mod",
        "foreign_graph",
        "foreign_source",
        "foreign_root",
        "wrong_axis_square",
        "bare_vector_square",
        "wrong_metadata",
        "missing_source",
        "missing_index_dtype",
        "wrong_device",
        "overflow",
        "period_no_grid",
        "partial_extent",
    ),
)
def test_typed_fx_modular_fail_closed(fault):
    graph, nodes, context = graph_case(
        square=fault in ("wrong_axis_square", "bare_vector_square")
    )
    load = nodes["load"]
    if fault == "mask":
        load.args = (*load.args[:2], True, None)
    elif fault == "eviction":
        load.args = (*load.args[:3], "first")
    elif fault == "kwargs":
        load.kwargs = {"extra_mask": None}
    elif fault in ("bool", "int"):
        nodes["source"].meta["val"] = torch.empty(
            (128, 96), dtype=torch.bool if fault == "bool" else torch.int32
        )
    elif fault == "dtype_result":
        load.meta["val"] = torch.empty(
            context["carrier_tile_shape"], dtype=torch.float16
        )
    elif fault == "rank":
        nodes["source"].meta["val"] = torch.empty((1, 128, 96))
    elif fault == "stride":
        nodes["source"].meta["val"] = torch.empty_strided((128, 96), (192, 1))
    elif fault.startswith("period") and fault != "period_no_grid":
        nodes["mod"].args = (
            nodes["row"],
            {
                "period0": 0,
                "periodnegative": -128,
                "periodbool": True,
                "periodfloat": 128.0,
            }[fault],
        )
    elif fault == "nonliteral":
        nodes["mod"].args = (nodes["row"], nodes["n"])
    elif fault == "wrong_period":
        nodes["mod"].args = (nodes["row"], 64)
    elif fault == "integer_wrap":
        wrap = graph.call_function(torch.ops.aten.mul.Tensor, (nodes["row"], 2**30))
        wrap.meta["val"] = nodes["row"].meta["val"]
        nodes["mod"].args = (wrap, 128)
    elif fault == "unknown":
        nodes["mod"].target = torch.ops.aten.sin.default
    elif fault == "double_mod":
        extra = graph.call_function(
            torch.ops.aten.remainder.Scalar, (nodes["mod"], 128)
        )
        extra.meta["val"] = nodes["mod"].meta["val"]
        load.args = (nodes["source"], [extra, nodes["col"]], None, None)
    elif fault in ("foreign_graph", "foreign_source", "foreign_root"):
        foreign = torch.fx.Graph()
        other = foreign.placeholder("other")
        if fault == "foreign_graph":
            other.meta["val"] = nodes["row"].meta["val"]
            nodes["mod"].args = (other, 128)
        elif fault == "foreign_source":
            other.meta["val"] = nodes["source"].meta["val"]
            load.args = (other, *load.args[1:])
        else:
            context["carrier_tile_index_nodes"] = (other, nodes["n"])
    elif fault == "wrong_axis_square":
        nodes["row"].args = (nodes["tm"], [1, nodes["m"]])
        nodes["row"].meta["val"] = torch.empty((1, 64), dtype=torch.int32)
    elif fault == "bare_vector_square":
        nodes["mod"].args = (nodes["tm"], 128)
        nodes["mod"].meta["val"] = nodes["tm"].meta["val"]
    elif fault == "wrong_metadata":
        nodes["mod"].meta["val"] = torch.empty((1, 64), dtype=torch.int32)
    elif fault == "missing_source":
        nodes["source"].meta.clear()
    elif fault == "missing_index_dtype":
        nodes["row"].meta["val"] = torch.empty((64, 1), dtype=torch.int64)
    elif fault == "wrong_device":
        load.meta["val"] = torch.empty(context["carrier_tile_shape"], device="meta")
    elif fault == "overflow":
        context["carrier_global_shape"] = (2**28, 96)
    elif fault == "period_no_grid":
        nodes["source"].meta["val"] = torch.empty((96, 96))
        nodes["mod"].args = (nodes["row"], 96)
    elif fault == "partial_extent":
        context["carrier_global_shape"] = (257, 96)
    assert modular_aux_load_map(load, **context) is None
