from __future__ import annotations

import ast
from contextlib import contextmanager
import dataclasses
from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_batched_aux_tma import _cpu_codegen

import helion
from helion._compiler.cute.mapped_aux_index import _Index
from helion._compiler.cute.mapped_aux_index import _prove_rendered_range
from helion._compiler.cute.mapped_aux_index import _UnsupportedMappedIndex
from helion._compiler.cute.mapped_aux_index import analyze_mapped_aux_load
from helion._compiler.cute.mapped_aux_index import render_mapped_aux_indices
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import memory_ops

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True)
def _rank_three(
    gc: torch.Tensor,
    d: torch.Tensor,
    out: torch.Tensor,
    chunks: int,
    head_dim: int,
) -> torch.Tensor:
    chunks = hl.specialize(chunks)
    head_dim = hl.specialize(head_dim)
    work, rows, columns = out.shape
    heads = columns // head_dim
    for bi, mi, ni in hl.tile([work, rows, columns], block_size=[1, None, None]):
        w = bi.index[:, None, None]
        m = mi.index[None, :, None]
        n = ni.index[None, None, :]
        head = n // head_dim
        base = ((w // chunks * heads + head) * chunks + w % chunks) * rows
        decay = hl.load(gc, [base + m])
        last = hl.load(gc, [base + rows - 1])
        diagonal = hl.load(d, [head])
        out[bi, mi, ni] = decay + last + diagonal
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _rank_two(source: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rows, columns = out.shape
    for mi, ni in hl.tile([rows, columns]):
        index = mi.index[:, None] * columns + ni.index[None, :]
        out[mi, ni] = hl.load(source, [index])
    return out


@contextmanager
def _bound(rank: int = 3, *, rows: int = 8, columns: int = 32):
    with _cpu_codegen(), FakeTensorMode():
        if rank == 3:
            values = (
                torch.empty((2 * 4 * 8 * rows,)),
                torch.empty((4,)),
                torch.empty((16, rows, columns)),
                8,
                columns // 4,
            )
            bound = _rank_three._bind_isolated(values)
        else:
            bound = _rank_two._bind_isolated(
                (torch.empty((rows * columns,)), torch.empty((rows, columns)))
            )
        host_function = bound.host_function
        assert host_function is not None
        with bound.env, host_function:
            graph = host_function.device_ir.graphs[-1].graph
            loads = {
                node.name: node
                for node in graph.nodes
                if node.target is memory_ops.load
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].name in ("gc", "d", "source")
            }
            yield bound, graph, loads


def _analyze(load, *, rank=3, rows=8, columns=32, config=None):
    return analyze_mapped_aux_load(
        load,
        output_block_ids=tuple(range(rank)),
        output_shape=(16, rows, columns) if rank == 3 else (rows, columns),
        config=config,
    )


@pytest.mark.parametrize("rows,columns", [(8, 32), (32, 32), (64, 256)])
def test_rank_three_batch_head_crossings(rows, columns):
    with _bound(rows=rows, columns=columns) as (_, _, loads):
        assert set(loads) == {"decay", "last", "diagonal"}
        descriptors = {}
        for name, load in loads.items():
            descriptor = _analyze(load, rows=rows, columns=columns)
            assert descriptor is not None
            descriptors[name] = descriptor
        assert descriptors["decay"].block_dependencies == (0, 1, 2)
        assert descriptors["last"].block_dependencies == (0, 2)
        assert descriptors["diagonal"].block_dependencies == (2,)
        evaluators = {
            name: descriptor.indices[0].compile()
            for name, descriptor in descriptors.items()
        }
        for work in (0, 7, 8, 15):
            for row in (0, rows - 1):
                for column in (0, columns // 4 - 1, columns // 4, columns - 1):
                    coords = {"output_0": work, "output_1": row, "output_2": column}
                    head = column // (columns // 4)
                    base = ((work // 8 * 4 + head) * 8 + work % 8) * rows
                    assert evaluators["decay"](coords) == base + row
                    assert evaluators["last"](coords) == base + rows - 1
                    assert evaluators["diagonal"](coords) == head


@pytest.mark.parametrize("config", [None, [8, 32], [16, 16], [32, 64]])
def test_config_only_resolves_flat_zero_not_selected_geometry(config):
    with _bound(rows=64, columns=256) as (_, _, loads):
        default = _analyze(loads["decay"], rows=64, columns=256)
        selected = _analyze(
            loads["decay"],
            rows=64,
            columns=256,
            config=None if config is None else helion.Config(block_sizes=config),
        )
        assert default is not None and selected == default


def test_rank_two_render_and_immutable_descriptor():
    with _bound(2) as (_, _, loads):
        descriptor = _analyze(next(iter(loads.values())), rank=2)
        assert descriptor is not None
        code = render_mapped_aux_indices(descriptor, (ast.Name("row"), ast.Name("col")))
        for row, col in ((0, 0), (2, 5), (7, 31)):
            result = eval(
                ast.unparse(code[0]),
                {"cutlass": SimpleNamespace(Int32=int), "row": row, "col": col},
            )
            assert result == row * 32 + col
        for field in dataclasses.fields(descriptor):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(descriptor, field.name, None)
        with pytest.raises(ValueError, match="output rank"):
            render_mapped_aux_indices(descriptor, (ast.Name("row"),))


@pytest.mark.parametrize(
    "change",
    [
        "mask",
        "eviction",
        "host_data",
        "wrong_origin",
        "wrong_extent",
        "wrong_axis_equal_size",
        "reshape",
        "transpose",
        "cast",
        "negative_divisor",
        "zero_divisor",
        "tensor_divisor",
        "float_literal",
        "bool_literal",
        "float_index_dtype",
        "negative_numerator",
        "out_of_bounds",
        "register_mask",
        "nonfull_slice",
        "overflow_cancelled",
    ],
)
def test_uncertain_or_changed_semantics_fail_closed(change):
    with _bound(rows=32, columns=32) as (_, graph, loads):
        load = loads["decay"]
        original_index = load.args[1][0]
        options = {"rows": 32, "columns": 32}
        if change == "mask":
            load.args = (*load.args[:2], True, None)
        elif change == "eviction":
            load.args = (*load.args[:2], None, "evict_first")
        elif change == "host_data":
            load.args = (load.args[0], [loads["diagonal"]], None, None)
        elif change == "wrong_origin":
            assert (
                analyze_mapped_aux_load(
                    load, output_block_ids=(0, 1, 99), output_shape=(16, 32, 32)
                )
                is None
            )
            return
        elif change == "wrong_extent":
            options["rows"] = 31
        elif change == "wrong_axis_equal_size":
            assert (
                analyze_mapped_aux_load(
                    load, output_block_ids=(0, 2, 1), output_shape=(16, 32, 32)
                )
                is None
            )
            return
        elif change in ("register_mask", "nonfull_slice"):
            register = next(node for node in graph.nodes if node.name == "m")
            if change == "register_mask":
                register.args = (*register.args[:2], True, None)
            else:
                register.args = (
                    register.args[0],
                    [None, slice(0, 1), None],
                    None,
                    None,
                )
        else:
            with graph.inserting_before(load):
                if change in ("reshape", "transpose", "cast"):
                    target, arguments = {
                        "reshape": (
                            torch.ops.aten.reshape.default,
                            (original_index, [-1]),
                        ),
                        "transpose": (
                            torch.ops.aten.transpose.int,
                            (original_index, 1, 2),
                        ),
                        "cast": (
                            torch.ops.prims.convert_element_type.default,
                            (original_index, torch.int64),
                        ),
                    }[change]
                    new = graph.call_function(target, arguments)
                elif change == "overflow_cancelled":
                    overflow = graph.call_function(
                        torch.ops.aten.mul.Tensor, (original_index, 1 << 30)
                    )
                    overflow.meta["val"] = original_index.meta["val"]
                    new = graph.call_function(
                        torch.ops.aten.sub.Tensor, (overflow, overflow)
                    )
                elif change in ("float_literal", "bool_literal", "out_of_bounds"):
                    literal = {
                        "float_literal": 0.5,
                        "bool_literal": True,
                        "out_of_bounds": 2048,
                    }[change]
                    new = graph.call_function(
                        torch.ops.aten.add.Tensor, (original_index, literal)
                    )
                elif change == "float_index_dtype":
                    new = graph.call_function(
                        torch.ops.aten.add.Tensor, (original_index, 0)
                    )
                elif change == "negative_numerator":
                    negative = graph.call_function(
                        torch.ops.aten.sub.Tensor, (original_index, 2048)
                    )
                    negative.meta["val"] = original_index.meta["val"]
                    new = graph.call_function(
                        torch.ops.aten.div.Tensor_mode,
                        (negative, 2),
                        {"rounding_mode": "floor"},
                    )
                else:
                    rhs = {
                        "negative_divisor": -2,
                        "zero_divisor": 0,
                        "tensor_divisor": original_index,
                    }[change]
                    new = graph.call_function(
                        torch.ops.aten.div.Tensor_mode,
                        (original_index, rhs),
                        {"rounding_mode": "floor"},
                    )
                new.meta["val"] = original_index.meta["val"]
                if change == "float_index_dtype":
                    new.meta["val"] = original_index.meta["val"].to(torch.float32)
                load.args = (load.args[0], [new], None, None)
        assert _analyze(load, **options) is None


def test_original_intermediate_proofs_and_no_gpu_initialization():
    initialized_before = torch.cuda.is_initialized()
    with _bound() as (_, _, loads):
        descriptor = _analyze(loads["decay"])
        assert descriptor is not None
        assert {proof.operation for proof in descriptor.integer_proofs} >= {
            "tile_index",
            "register_fullslice_none",
            "floor_div",
            "mod",
            "mul",
            "add",
        }
        assert all(
            proof.dtype == torch.int32
            and -(1 << 31) <= proof.minimum <= proof.maximum < (1 << 31)
            for proof in descriptor.integer_proofs
        )
        assert torch.cuda.is_initialized() == initialized_before


def test_descriptor_retains_strided_input_layout_without_pointer_claims():
    with _bound() as (_, _, loads):
        load = loads["diagonal"]
        host = load.args[0]
        original = host.meta["val"]
        host.meta["val"] = torch.empty_strided(
            original.shape, (3,), dtype=original.dtype
        )
        descriptor = _analyze(load)
        assert descriptor is not None
        assert descriptor.input_shape == (4,) and descriptor.input_strides == (3,)
        assert descriptor.host_name == "d"
        # Returned subscript remains a logical head index, NOT a multiplied byte offset.
        assert descriptor.indices[0].compile()({"output_2": 31}) == 3


def test_int64_indices_only_when_every_intermediate_remains_int32_safe():
    with _bound() as (_, _, loads):
        load = loads["diagonal"]
        index = load.args[1][0]
        index.meta["val"] = index.meta["val"].to(torch.int64)
        descriptor = _analyze(load)
        assert descriptor is not None
        assert descriptor.integer_proofs[-1].dtype == torch.int64
        assert descriptor.indices[0].compile()({"output_2": 31}) == 3


def test_rendered_arithmetic_checked_before_cancelling_large_terms():
    variable = sympy.Symbol("output_0", integer=True, nonnegative=True)
    first = sympy.Mul(1 << 30, variable, evaluate=False)
    second = sympy.Mul(-(1 << 30), variable, evaluate=False)
    expression = sympy.Add(first, second, evaluate=False)
    assert sympy.simplify(expression) == 0
    index = _Index(expression, ((variable, 0, 3),))
    with pytest.raises(_UnsupportedMappedIndex):
        _prove_rendered_range(index)
