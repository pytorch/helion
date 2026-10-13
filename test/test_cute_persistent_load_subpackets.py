from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import memory_ops
from helion._compiler.cute.persistent_branch_vec import (
    vectorize_branch_local_persistent_fragments,
)


def fixture(width=8, dtype="Float32", suffix="", mask=None, eviction=""):
    pointer = "key.iterator + lane"
    scalar = f"({pointer}).load()"
    if mask:
        scalar = f"({scalar} if {mask} else cutlass.{dtype}(0))"
    return ast.parse(f"""
for synthetic_lane_3 in cutlass.range_constexpr({width}):
    lane = base + cutlass.Int32(synthetic_lane_3)
    value = _helion_persistent_branch_vec_load(3, {width}, 'cutlass.{dtype}', {eviction!r}, {pointer}, {scalar})
    (out.iterator + synthetic_lane_3).store(value)
    {suffix or "pass"}
""").body


def transform(body, pairs=(("key", "out"),)):
    return vectorize_branch_local_persistent_fragments(
        body, proven_disjoint_tensor_pairs=set(map(frozenset, pairs))
    )


class Scalar(int):
    def bitcast(self, dtype):
        return dtype(self)


class Pointer:
    def __init__(self, values, log, index=0):
        self.values, self.log, self.index = values, log, index

    def __add__(self, amount):
        return Pointer(self.values, self.log, self.index + int(amount))

    def load(self):
        self.log.append(("load", self.index, 1))
        return self.values[self.index]

    def store(self, value):
        self.values[self.index] = value


def execute(body, *, base=8, live=True):
    reads = []
    values = [Scalar(i * 37 + 11) for i in range(64)]
    output = [None] * 16

    def vector_load(pointer, typ, **kwargs):
        width = typ[0]
        reads.append(("load", pointer.index, width))
        assert 0 <= pointer.index <= len(pointer.values) - width
        return pointer.values[pointer.index : pointer.index + width]

    cutlass = SimpleNamespace(
        range_constexpr=range,
        Int32=int,
        Uint32=Scalar,
        Uint16=Scalar,
        Float32=Scalar,
        BFloat16=Scalar,
    )
    cutlass.Uint32.mlir_type = "u32"
    cutlass.Uint16.mlir_type = "u16"
    scope = {
        "cutlass": cutlass,
        "cute": SimpleNamespace(arch=SimpleNamespace(load=vector_load)),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda dims, typ: (dims[0], typ))
        ),
        "key": SimpleNamespace(iterator=Pointer(values, reads)),
        "out": SimpleNamespace(iterator=Pointer(output, [])),
        "base": base,
        "live": live,
        "_helion_persistent_branch_vec_load": lambda *args: args[-1],
        "_cute_load_l2_evict_last": vector_load,
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, "<test>", "exec"), scope)
    return output, reads


@pytest.mark.parametrize(
    "width,dtype,packets",
    [(4, "Float32", 1), (8, "Float32", 2), (16, "Float32", 4), (8, "BFloat16", 1)],
)
def test_same_scalar_values_and_ascending_physical_packets(width, dtype, packets):
    expected, _ = execute(fixture(width, dtype))
    actual, reads = execute(transform(fixture(width, dtype)))
    assert actual == expected
    assert len(reads) == packets
    assert [r[1] for r in reads] == list(range(8, 8 + width, width // packets))
    assert all(r[2] * (4 if dtype == "Float32" else 2) <= 16 for r in reads)


@pytest.mark.parametrize("live", [True, False])
def test_original_mask_and_safe_whole_fragment_anchor(live):
    expected, _ = execute(fixture(mask="live"), base=8 if live else 100000, live=live)
    actual, reads = execute(
        transform(fixture(mask="live")), base=8 if live else 100000, live=live
    )
    assert actual == expected
    assert [r[1] for r in reads] == ([8, 12] if live else [0, 4])


@pytest.mark.parametrize(
    "suffix,pairs",
    [
        ("(key.iterator + 0).store(value)", (("key", "out"),)),
        ("unknown(value)", (("key", "out"),)),
        ("cute.arch.barrier()", (("key", "out"),)),
        ("pass", ()),
    ],
)
def test_alias_and_unknown_effect_keep_original_scalar_load(suffix, pairs):
    code = ast.unparse(
        ast.Module(body=transform(fixture(suffix=suffix), pairs), type_ignores=[])
    )
    assert "ir.VectorType" not in code
    assert ".load()" in code
    assert "_helion_persistent_branch_vec_load" not in code


def test_nonuniform_mask_declines():
    code = ast.unparse(
        ast.Module(
            body=transform(fixture(mask="synthetic_lane_3 % 2")), type_ignores=[]
        )
    )
    assert "ir.VectorType" not in code
    assert "synthetic_lane_3 % 2" in code


def test_original_cache_hint_on_each_packet():
    code = ast.unparse(
        ast.Module(body=transform(fixture(eviction=", cop='cs'")), type_ignores=[])
    )
    assert code.count("cop='cs'") == 2
    code = ast.unparse(
        ast.Module(body=transform(fixture(eviction="__l2_last__")), type_ignores=[])
    )
    assert code.count("_cute_load_l2_evict_last(") == 2


@pytest.mark.parametrize(
    "shape,stride,aligned,packet_width,expected",
    [
        ((2, 128), (128, 1), True, 4, True),
        ((2, 132), (132, 1), True, 4, False),
        ((2, 128), (129, 1), True, 4, False),
        ((2, 128), (132, 1), True, 4, True),
        ((2, 128), (128, 1), False, 4, False),
        ((2, 128), (128, 1), True, 3, False),
    ],
)
def test_logical_extent_and_physical_alignment_remain_separate(
    shape, stride, aligned, packet_width, expected
):
    tensor = torch.empty_strided(shape, stride, dtype=torch.float32)
    # Literal shapes/strides need no substitution; symbolic expressions stay
    # unresolved in the real alignment guard.
    env = SimpleNamespace(
        runtime_value_for_tensor=lambda t: t,
        shape_env=SimpleNamespace(replace=lambda expression: expression),
        specialize_expr=lambda expression: expression,
    )
    strategy = SimpleNamespace(_cute_lane_base_index_var="base")
    requested = []

    def alignment(env, tensor, n):
        requested.append(n)
        return aligned

    with (
        patch.object(memory_ops.CompileEnvironment, "current", return_value=env),
        patch.object(
            memory_ops, "_persistent_vec_base_index_exprs", return_value=["row", "base"]
        ),
        patch.object(memory_ops, "_persistent_assignment_definitions", return_value={}),
        patch.object(
            memory_ops,
            "runtime_tensor_has_specialized_alignment",
            side_effect=alignment,
        ),
    ):
        actual = memory_ops._persistent_vec_is_exact_aligned(
            None, strategy, [], tensor, 8, packet_width=packet_width
        )
    assert actual is expected
    if requested:
        assert requested == [packet_width * 4]


def test_store_alignment_default_unchanged():
    tensor = torch.empty((2, 128))
    # Literal shapes/strides need no substitution; symbolic expressions stay
    # unresolved in the real alignment guard.
    env = SimpleNamespace(
        runtime_value_for_tensor=lambda t: t,
        shape_env=SimpleNamespace(replace=lambda expression: expression),
        specialize_expr=lambda expression: expression,
    )
    strategy = SimpleNamespace(_cute_lane_base_index_var="base")
    with (
        patch.object(memory_ops.CompileEnvironment, "current", return_value=env),
        patch.object(
            memory_ops, "_persistent_vec_base_index_exprs", return_value=["row", "base"]
        ),
        patch.object(memory_ops, "_persistent_assignment_definitions", return_value={}),
        patch.object(
            memory_ops, "runtime_tensor_has_specialized_alignment", return_value=False
        ) as alignment,
    ):
        assert not memory_ops._persistent_vec_is_exact_aligned(
            None, strategy, [], tensor, 8
        )
        assert alignment.call_args.args[-1] == 32
