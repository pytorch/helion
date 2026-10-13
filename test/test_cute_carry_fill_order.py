from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import chained_fill_ownership as module
from helion._compiler.cute import chained_loop
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts


def fixture(rows=32, columns=32, threads=128):
    indices = (f"i % {columns}", f"i // {columns}")
    return {
        "lines": [
            f"x = (raw.iterator + cutlass.Int32({indices[0]}) * cutlass.Int32(raw.layout.stride[0]) + cutlass.Int32({indices[1]}) * cutlass.Int32(raw.layout.stride[1])).load()",
            "y = cutlass.Float32(cutlass.BFloat16(x)) + 0.25",
        ],
        "value": "y",
        "reads": [module.Read("raw", indices, (rows, 1), 4, False)],
        "index": "i",
        "origins": (),
        "shape": (rows, columns),
        "threads": threads,
        "scratch_mode": "xor",
    }


@pytest.mark.parametrize(
    "rows,columns,threads", [(32, 32, 128), (64, 32, 256), (128, 64, 256)]
)
def test_transposed_external_reads_select_contiguous_first_axis(rows, columns, threads):
    assert module.choose_carry_first_axis(**fixture(rows, columns, threads))


def test_row_major_scratch_conflict_declines_despite_global_gain():
    case = fixture()
    case["scratch_mode"] = "row_major"
    assert not module.choose_carry_first_axis(**case)


def test_already_coalesced_or_conflicting_or_nonaffine_read_declines():
    case = fixture()
    case["reads"] = [replace(case["reads"][0], strides=(1, 32))]
    assert not module.choose_carry_first_axis(**case)
    case = fixture()
    case["reads"].append(replace(case["reads"][0], strides=(1, 32)))
    assert not module.choose_carry_first_axis(**case)
    case = fixture()
    case["reads"] = [
        replace(case["reads"][0], indices=("i % 32", "(i // 32) * (i // 32)"))
    ]
    assert not module.choose_carry_first_axis(**case)


@pytest.mark.parametrize(
    "kind",
    [
        "loaded_index",
        "thread_value",
        "unknown_call",
        "rebind",
        "store",
        "mixed_origin",
        "unknown_stride",
        "shared_read",
    ],
)
def test_unknown_effect_or_physical_read_keeps_original(kind):
    case = fixture()
    if kind == "loaded_index":
        case["reads"][0] = replace(case["reads"][0], indices=("i % 32", "x"))
    elif kind == "thread_value":
        case["value"] = "y + chain_thread"
    elif kind == "unknown_call":
        case["value"] = "unknown(y)"
    elif kind == "rebind":
        case["lines"].append("y = y + 1")
    elif kind == "store":
        case["lines"].append("raw[0, 0] = y")
    elif kind == "mixed_origin":
        case["origins"] = ("origin",)
        case["reads"][0] = replace(
            case["reads"][0], indices=("(origin + i) % 32", "i // 32")
        )
    elif kind == "unknown_stride":
        case["reads"][0] = replace(case["reads"][0], strides=(32, -1))
    elif kind == "shared_read":
        case["reads"][0] = replace(case["reads"][0], shared=True)
    assert not module.choose_carry_first_axis(**case)


def test_uniform_head_modulo_and_signed_adjustment_preserved():
    case = fixture()
    case["origins"] = ("origin",)
    case["lines"][:0] = [
        "head = origin % 7",
        "batch = (origin - head) // 7",
        "adjusted = batch - 1 if (head != 0) & ((head < 0) != (7 < 0)) else batch",
    ]
    case["reads"][0] = replace(
        case["reads"][0],
        indices=("adjusted", "head", "i % 32", "i // 32"),
        strides=(7168, 1024, 32, 1),
    )
    # This is the complete original pointer expression, not a second graph walk.
    case["lines"][3] = (
        "x = (raw.iterator + cutlass.Int32(adjusted) * cutlass.Int32(raw.layout.stride[0]) + cutlass.Int32(head) * cutlass.Int32(raw.layout.stride[1]) + cutlass.Int32(i % 32) * cutlass.Int32(raw.layout.stride[2]) + cutlass.Int32(i // 32) * cutlass.Int32(raw.layout.stride[3])).load()"
    )
    assert module.choose_carry_first_axis(**case)


def bridge(case, policy="auto"):
    source = SimpleNamespace(meta={"val": torch.empty_strided((32, 32), (32, 1))})
    expression = SimpleNamespace(
        lines=list(case["lines"]),
        global_accesses=[(source, list(case["reads"][0].indices))],
        accesses=[object()],
        origins={},
        scan_inputs=[],
        staged_inputs=[],
        fragment_images={},
        coordinate_normalization=None,
        tensor_name=lambda source: "raw",
    )
    cg = SimpleNamespace(
        device_function=SimpleNamespace(
            config=SimpleNamespace(config={"cute_chained_fill_ownership": policy})
        )
    )
    plan = SimpleNamespace(loop=object(), threads=128)
    return cg, plan, expression


def test_default_decline_is_exact_and_accepted_rewrite_preserves_value_association():
    case = fixture()
    cg, plan, expression = bridge(case)
    args = (cg, plan, expression, (32, 32), torch.float32, "xor", "i", case["value"])
    before = list(expression.lines)
    with patch.object(chained_loop, "loop_storage_is_proven", return_value=True):
        selected = module.select_carry_order(*args)
        assert selected is not None
        cg.device_function.config.config["cute_chained_fill_ownership"] = "blocked"
        assert module.select_carry_order(*args) is None
    assert expression.lines == before
    lines, value, coords = selected
    assert coords == ("i % 32", "i // 32")

    # Replay exact rewritten read/cast/pointwise expressions and destinations,
    # with nontrivial values and both coordinate orders. All cells must agree.
    class Pointer:
        def __init__(self, offset=0):
            self.offset = offset

        def __add__(self, x):
            return Pointer(self.offset + x)

        def load(self):
            return float(self.offset * 3 - 1000)

    cutlass = SimpleNamespace(
        Int32=int,
        Float32=float,
        BFloat16=lambda x: float(torch.tensor(x).to(torch.bfloat16)),
    )
    raw = SimpleNamespace(iterator=Pointer(), layout=SimpleNamespace(stride=(32, 1)))
    results = []
    for program, coordinates in [(before, ("i // 32", "i % 32")), (lines, coords)]:
        result = {}
        for i in range(1024):
            ns = {"i": i, "raw": raw, "cutlass": cutlass}
            exec("\n".join(program), ns)
            point = tuple(eval(c, ns) for c in coordinates)
            assert point not in result
            result[point] = eval(value, ns)
        results.append(result)
    assert results[0] == results[1]


def test_bare_flat_index_value_cannot_escape_coordinate_rebinding():
    case = fixture()
    cg, plan, expression = bridge(case)
    with patch.object(chained_loop, "loop_storage_is_proven", return_value=True):
        assert (
            module.select_carry_order(
                cg, plan, expression, (32, 32), torch.float32, "xor", "i", "y + i"
            )
            is None
        )


@pytest.mark.parametrize("mode", ["auto", "blocked"])
def test_initialize_loop_single_expression_visit_zero_trip_and_owned_pointer(mode):
    case = fixture()
    cg, plan, expression = bridge(case, mode)
    incoming = object()
    carry_node = SimpleNamespace(meta={"val": torch.empty((32, 32))})
    carry = SimpleNamespace(input=carry_node, input_index=0)
    loop = SimpleNamespace(
        inputs=(incoming,),
        region=SimpleNamespace(carries=(carry,)),
        begin=0,
        end=0,
        carry_name=lambda i: "chain_loop_carry_0",
    )
    plan.loop = loop
    visits = []

    class Expression:
        def __init__(self, *args):
            self.__dict__.update(expression.__dict__)
            self.coordinate_names = set()
            self.lines = []

        def scalar(self, x):
            return str(x)

        def value(self, node, coords):
            visits.append((node, coords))
            self.lines = [
                line.replace("i // 32", coords[0]).replace("i % 32", coords[1])
                for line in case["lines"]
            ]
            self.global_accesses = [
                (expression.global_accesses[0][0], [coords[1], coords[0]])
            ]
            return case["value"]

    env = SimpleNamespace(
        backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32")
    )
    with (
        patch.object(chained_loop.CompileEnvironment, "current", return_value=env),
        patch.object(chained_matmul, "_shape", return_value=(32, 32)),
        patch.object(chained_matmul, "_Expression", Expression),
        patch.object(chained_loop, "loop_storage_is_proven", return_value=True),
    ):
        lines = chained_loop.initialize_loop(
            cg,
            plan,
            ScratchLayouts(mode="xor"),
            carry_pointers={"chain_loop_carry_0": "existing_owner + 128"},
        )
    assert len(visits) == 1 and visits[0][0] is incoming
    assert lines[0:2] == ["chain_loop_begin = 0", "chain_loop_end = 0"]
    assert "existing_owner + 128" in lines[2]
    assert "cute.make_swizzle(5, 0, 5)" in lines[2]
    assert lines[-1] == "cute.arch.sync_threads()"
    expected = (
        "chain_loop_carry_0_offset % 32, chain_loop_carry_0_offset // 32"
        if mode == "auto"
        else "chain_loop_carry_0_offset // 32, chain_loop_carry_0_offset % 32"
    )
    assert f"chain_loop_carry_0[{expected}]" in "\n".join(lines)
