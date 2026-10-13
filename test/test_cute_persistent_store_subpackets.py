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


def fixture(width=8, mask=None, suffix="", store_offset=0, load_offset=0):
    pointer = f"state.iterator + lane + {load_offset}"
    scalar = f"({pointer}).load()"
    if mask:
        scalar = f"({scalar} if {mask} else cutlass.Float32(0))"
    return ast.parse(f"""
for synthetic_lane_3 in cutlass.range_constexpr({width}):
    lane = base + cutlass.Int32(synthetic_lane_3)
    value = _helion_persistent_branch_vec_load(
        3, {width}, 'cutlass.Float32', '', {pointer}, {scalar})
    _helion_persistent_branch_vec_store(
        3, {width}, 'cutlass.Float32', state.iterator + lane + {store_offset},
        value, {mask or "None"})
    {suffix or "pass"}
""").body


def transform(body):
    return vectorize_branch_local_persistent_fragments(body)


class Word(int):
    def bitcast(self, dtype):
        return dtype(self)


class Pointer:
    def __init__(self, words, events, offset=0):
        self.words, self.events, self.offset = words, events, offset

    def __add__(self, amount):
        return Pointer(self.words, self.events, self.offset + amount)

    def load(self):
        self.events.append(("load", self.offset, 1))
        return self.words[self.offset]

    def store(self, value):
        self.events.append(("store", self.offset, 1))
        self.words[self.offset] = value


def execute(body, live=True):
    # Opaque FP32 bits include signed zero, subnormals, infinities and NaN.
    pattern = [
        0,
        0x80000000,
        1,
        0x007FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0x3F800000,
    ]
    words = list(map(Word, pattern * 8))
    events = []

    def vector_load(pointer, typ):
        width = typ[0]
        events.append(("load", pointer.offset, width))
        return pointer.words[pointer.offset : pointer.offset + width]

    def vector_store(pointer, values):
        events.append(("store", pointer.offset, len(values)))
        pointer.words[pointer.offset : pointer.offset + len(values)] = values

    def marker_store(block, width, dtype, pointer, value, mask):
        if mask is None or mask:
            pointer.store(value)

    Word.mlir_type = "u32"
    scope = {
        "cutlass": SimpleNamespace(
            range_constexpr=range, Int32=int, Uint32=Word, Float32=Word
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(load=vector_load)),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, ty: (shape[0], ty))
        ),
        "state": SimpleNamespace(iterator=Pointer(words, events)),
        "base": 8 if live else 10000,
        "live": live,
        "_helion_persistent_branch_vec_load": lambda *args: args[-1],
        "_helion_persistent_branch_vec_store": marker_store,
        "_cute_store_u32_vec": vector_store,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<test>",
            "exec",
        ),
        scope,
    )
    return words, events


@pytest.mark.parametrize("width", [4, 8, 16])
@pytest.mark.parametrize("live", [True, False])
def test_bit_exact_complete_fragment_and_original_predicate(width, live):
    expected, _ = execute(fixture(width, "live"), live)
    body = transform(fixture(width, "live"))
    actual, events = execute(body, live)
    assert actual == expected
    stores = [e for e in events if e[0] == "store"]
    assert stores == (
        [("store", 8 + offset, 4) for offset in range(0, width, 4)] if live else []
    )
    assert all(e[2] <= 4 for e in events)


@pytest.mark.parametrize(
    "suffix",
    [
        "unknown(value)",
        "cute.arch.barrier()",
        "value.value = 1",
        "state.iterator = replacement",
        "base += 1",
        "(state.iterator + lane + 1).load()",
        "(state.iterator + lane + 1).store(value)",
    ],
)
def test_effect_alias_and_mutation_decline(suffix):
    text = ast.unparse(
        ast.Module(body=transform(fixture(suffix=suffix)), type_ignores=[])
    )
    assert "_cute_store_u32_vec" not in text
    assert ".store(" in text


@pytest.mark.parametrize(
    "shape,stride,aligned,expected",
    [
        ((2, 128), (128, 1), True, True),
        ((2, 132), (132, 1), True, False),
        ((2, 128), (129, 1), True, False),
        ((2, 128), (132, 1), True, True),
        ((2, 128), (128, 1), False, False),
    ],
)
def test_original_full_fragment_and_physical_alignment(
    shape, stride, aligned, expected
):
    tensor = torch.empty_strided(shape, stride, dtype=torch.float32)
    # Literal shapes/strides need no substitution; symbolic expressions stay
    # unresolved in the real alignment guard.
    env = SimpleNamespace(
        runtime_value_for_tensor=lambda value: value,
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
            memory_ops, "runtime_tensor_has_specialized_alignment", return_value=aligned
        ) as alignment,
    ):
        assert (
            memory_ops._persistent_vec_is_exact_aligned(
                None, strategy, [], tensor, 8, packet_width=4
            )
            is expected
        )
        if alignment.called:
            assert alignment.call_args.args[-1] == 16
