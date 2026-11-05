from __future__ import annotations

import itertools

import pytest
import sympy
import torch

from helion._compiler.cute.mapped_aux import mapped_aux_load_map
from helion._compiler.cute.mapped_aux import prove_mapped_aux_load_map
from helion.language import memory_ops
from helion.language import tile_ops


def _mod(value: sympy.Expr, modulus: int) -> sympy.Expr:
    result = sympy.Mod(value, modulus)
    assert isinstance(result, sympy.Expr)
    return result


def _dense_strides(shape: tuple[int, ...], order: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    span = 1
    for axis in order:
        strides[axis] = span
        span *= shape[axis]
    return tuple(strides)


@pytest.mark.parametrize("split_axis", (0, 1))
@pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(3))))
@pytest.mark.parametrize("storage_order", tuple(itertools.permutations(range(3))))
def test_complete_mapped_aux_bijection(split_axis, permutation, storage_order):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    extents = (256, 96) if split_axis == 0 else (96, 256)
    split, other = (row, col) if split_axis == 0 else (col, row)
    canonical_shape = (2, 128, 96)
    digits = (sympy.floor(split / 128), _mod(split, 128), other)
    shape = tuple(canonical_shape[axis] for axis in permutation)
    indices = tuple(digits[axis] for axis in permutation)
    strides = _dense_strides(shape, storage_order)
    proof = prove_mapped_aux_load_map(indices, (row, col), extents, shape, strides)
    assert proof is not None
    assert proof.split_axis == split_axis
    assert proof.owned_index_nodes == ()
    tiles = (128, 32) if split_axis == 0 else (32, 128)
    assert proof.tile_reason(*tiles) is None
    offsets = set()
    for m in range(extents[0]):
        for n in range(extents[1]):
            source = (m, n)[split_axis]
            remaining = (m, n)[1 - split_axis]
            expected_digits = (source // 128, source % 128, remaining)
            expected = sum(
                expected_digits[axis] * stride
                for axis, stride in zip(permutation, strides, strict=True)
            )
            assert proof.offset(m, n) == expected
            offsets.add(expected)
    assert offsets == set(range(256 * 96))
    emitted, coords = proof.view_source(
        "view", "original_aux", "tile_m", "tile_n", *tiles
    )
    assert "original_aux.iterator + cute.crd2idx" in emitted
    assert "original_aux.layout.stride" in emitted
    assert (
        coords[split_axis] == f"cutlass.Int32({('tile_m', 'tile_n')[split_axis]}) % 1"
    )


@pytest.mark.parametrize(
    "invalid",
    (
        "offset",
        "symbol",
        "noninteger",
        "nonfinite",
        "overlap",
        "negative_stride",
        "shape",
        "boolean",
        "overflow",
    ),
)
def test_mapped_aux_map_fails_closed(invalid):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    symbols = (row, col)
    indices = [sympy.floor(row / 128), _mod(row, 128), col]
    shape = (2, 128, 96)
    strides = (12288, 1, 128)
    extents = (256, 96)
    if invalid == "offset":
        indices[1] = sympy.Add(indices[1], sympy.Integer(1))
    elif invalid == "symbol":
        indices[0] = sympy.Symbol("unknown", integer=True)
    elif invalid == "noninteger":
        symbols = (sympy.Symbol("not_integer"), col)
    elif invalid == "nonfinite":
        indices[0] = sympy.oo
    elif invalid == "overlap":
        strides = (12288, 1, 1)
    elif invalid == "negative_stride":
        strides = (12288, -1, 128)
    elif invalid == "shape":
        shape = (3, 128, 96)
    elif invalid == "boolean":
        shape = (True, 128, 96)
    else:
        shape = (2**24, 128, 96)
        extents = (2**24 * 128, 96)
        strides = (12288, 1, 128)
    assert prove_mapped_aux_load_map(indices, symbols, extents, shape, strides) is None


def _index_graph(*, square: bool = False):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty_strided(
        (2, 128, 96), (12288, 1, 128), dtype=torch.bfloat16
    )
    bm, bn = (32, 32) if square else (64, 32)
    m = graph.placeholder("m_size")
    n = graph.placeholder("n_size")
    tile_m = graph.call_function(tile_ops.tile_index, (m,))
    tile_m.meta["val"] = torch.empty((bm,), dtype=torch.int32)
    tile_n = graph.call_function(tile_ops.tile_index, (n,))
    tile_n.meta["val"] = torch.empty((bn,), dtype=torch.int32)
    row = graph.call_function(torch.ops.aten.view.default, (tile_m, [m, 1]))
    row.meta["val"] = torch.empty((bm, 1), dtype=torch.int32)
    col = graph.call_function(torch.ops.aten.view.default, (tile_n, [1, n]))
    col.meta["val"] = torch.empty((1, bn), dtype=torch.int32)
    quotient = graph.call_function(
        torch.ops.aten.div.Tensor_mode, (row, 128), {"rounding_mode": "floor"}
    )
    quotient.meta["val"] = torch.empty((bm, 1), dtype=torch.int32)
    remainder = graph.call_function(torch.ops.aten.remainder.Scalar, (row, 128))
    remainder.meta["val"] = torch.empty((bm, 1), dtype=torch.int32)
    load = graph.call_function(
        memory_ops.load, (source, [quotient, remainder, col], None, None)
    )
    load.meta["val"] = torch.empty((bm, bn), dtype=torch.bfloat16)
    context = {
        "carrier_tile_shape": (bm, bn),
        "carrier_tile_index_nodes": (m, n),
        "carrier_global_shape": (256, 96),
    }
    nodes = {
        "source": source,
        "row": row,
        "col": col,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "quotient": quotient,
        "remainder": remainder,
        "load": load,
    }
    return graph, nodes, context


def test_mapped_aux_fx_complete_provenance_and_pure_owners():
    _graph, nodes, context = _index_graph()
    proof = mapped_aux_load_map(nodes["load"], **context)
    assert proof is not None
    assert set(proof.owned_index_nodes) == {
        nodes["tile_m"],
        nodes["tile_n"],
        nodes["row"],
        nodes["col"],
        nodes["quotient"],
        nodes["remainder"],
    }
    assert nodes["source"] not in proof.owned_index_nodes
    assert nodes["load"] not in proof.owned_index_nodes
    assert proof.tile_reason(128, 32) is None
    assert proof.tile_reason(256, 32) is not None
    assert proof.tile_reason(128, 64) is not None


@pytest.mark.parametrize(
    "fault",
    (
        "mask",
        "eviction",
        "dtype",
        "missing_source",
        "unrelated_index",
        "square_wrong_axis",
        "integer_wrap",
        "bad_intermediate_shape",
        "other_graph",
        "unknown",
        "wrong_extent",
        "kwargs",
    ),
)
def test_mapped_aux_fx_provenance_faults_are_rejected(fault):
    graph, nodes, context = _index_graph(square=fault == "square_wrong_axis")
    load = nodes["load"]
    if fault == "mask":
        load.args = (nodes["source"], load.args[1], True, None)
    elif fault == "eviction":
        load.args = (nodes["source"], load.args[1], None, "first")
    elif fault == "dtype":
        nodes["source"].meta["val"] = torch.empty_strided(
            (2, 128, 96), (12288, 1, 128), dtype=torch.bool
        )
    elif fault == "missing_source":
        nodes["source"].meta.clear()
    elif fault == "unrelated_index":
        unrelated = graph.placeholder("unrelated")
        nodes["tile_m"].args = (unrelated,)
    elif fault == "square_wrong_axis":
        nodes["row"].args = (
            nodes["tile_m"],
            [1, context["carrier_tile_index_nodes"][0]],
        )
        nodes["row"].meta["val"] = torch.empty((1, 32), dtype=torch.int32)
    elif fault == "integer_wrap":
        overflow = graph.call_function(torch.ops.aten.mul.Tensor, (nodes["row"], 2**30))
        overflow.meta["val"] = nodes["row"].meta["val"]
        nodes["quotient"].args = (overflow, 128)
    elif fault == "bad_intermediate_shape":
        nodes["quotient"].meta["val"] = torch.empty((1, 64), dtype=torch.int32)
    elif fault == "other_graph":
        other = torch.fx.Graph()
        external = other.placeholder("external")
        external.meta["val"] = nodes["row"].meta["val"]
        nodes["quotient"].args = (external, 128)
    elif fault == "unknown":
        nodes["quotient"].target = torch.ops.aten.sin.default
    elif fault == "wrong_extent":
        context["carrier_global_shape"] = (128, 96)
    else:
        load.kwargs = {"unknown": True}
    assert mapped_aux_load_map(load, **context) is None
