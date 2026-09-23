from __future__ import annotations

import ast
import itertools
import math
import operator
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

from examples.softmax import softmax_bwd
import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_resident_sequence import _indent
from test.test_cute_resident_sequence import _Tensor
from test.test_cute_resident_sequence import _YieldCollectives

import helion
from helion import exc
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.autotuner_heuristics.cute_resident_sequence import (
    CuteResidentSequenceHeuristic,
)
from helion._compiler.cute.merge_sibling_v_loops import merge_sibling_v_loops
from helion._compiler.cute.resident_sequence import _MARKER
from helion._compiler.cute.resident_sequence import SequenceRegion
from helion._compiler.cute.resident_sequence import materialize_resident_sequences
from helion._compiler.tile_strategy import _lane_reduce_marker_expr
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration


def _source(threads: int, slots: int, vector: int) -> str:
    marker = _lane_reduce_marker_expr(
        "product", "sum", "cutlass.Float32(0)", 32
    ).replace("_helion_lane_reduce", _MARKER)
    scalar = f"""column = base + cutlass.Int32(vec)
old_carry = carry
left = cutlass.Float16((x.iterator + row * width + column).load()) if column < width and row < rows else cutlass.Float16(0)
right = cutlass.Float16((y.iterator + row * width + column).load()) if column < width and row < rows else cutlass.Float16(0)
product = left * right
tile_sum = cutlass.Float16({marker})
widened = cutlass.Float32(tile_sum)
new_carry = old_carry + widened
"""
    return (
        "row = cutlass.Int32(cute.arch.thread_idx()[1])\n"
        "carry = cutlass.Float32(initial)\n"
        f"for offset in range(cutlass.Int32(0), cutlass.Int32(width), cutlass.Int32({threads * slots * vector})):\n"
        f"    for lane in range({slots}):\n"
        f"        base = offset + (cutlass.Int32(lane) * {threads} + cutlass.Int32(cute.arch.thread_idx()[0])) * {vector}\n"
        f"        for vec in cutlass.range_constexpr({vector}):\n"
        + _indent(scalar, 12)
        + "\nif row < rows and cute.arch.thread_idx()[0] == 0:\n"
        "    (out.iterator + row).store(carry)\n"
    )


def _rewrite(
    source: str,
    *,
    threads: int,
    slots: int,
    vector: int,
    groups: int,
    resident: bool = False,
    disjoint: bool = True,
) -> str:
    body = ast.parse(source).body
    loop = next(statement for statement in body if isinstance(statement, ast.For))
    lane = loop.body[0]
    assert isinstance(lane, ast.For)
    setattr(lane, HELION_LANE_LOOP_VAR_ATTR, "lane")
    region = SequenceRegion(
        loop,
        "lane",
        slots,
        "vec",
        vector,
        0,
        threads,
        threads * groups,
        f"cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * {threads}",
    )
    counter = itertools.count()
    result = materialize_resident_sequences(
        list(body),
        regions={id(loop): region},
        tensor_dtypes={
            "x": "cutlass.Float16",
            "y": "cutlass.Float16",
            "out": "cutlass.Float32",
        },
        rename_groups={"new_carry": "carry"},
        disjoint_pairs={frozenset(("x", "out")), frozenset(("y", "out"))}
        if disjoint
        else set(),
        boundary_names={"x", "y", "out", "width", "rows", "initial"},
        resident=resident,
        new_var=lambda hint: f"{hint}_{next(counter)}",
    )

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == "new_carry":
                node.id = "carry"
            return node

    module = ast.Module(body=cast("list[ast.stmt]", result), type_ignores=[])
    Rename().visit(module)
    return ast.unparse(module)


def _execute(
    source: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    threads: int,
    groups: int,
    initial: float,
) -> tuple[np.ndarray, dict[str, int]]:
    tree = ast.parse("def kernel():\n" + _indent(source))
    _YieldCollectives().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<single sequence CPU SIMT>", "exec")
    left, right = _Tensor(x.copy()), _Tensor(y.copy())
    output = _Tensor(np.full(x.shape[0], np.float32(np.nan)))
    generators = []
    for thread in range(threads * groups):
        namespace: dict[str, Any] = {
            "operator": operator,
            "cutlass": SimpleNamespace(
                Boolean=np.bool_,
                Int32=np.int32,
                Int64=np.int64,
                Float16=np.float16,
                Float32=np.float32,
                range_constexpr=range,
            ),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(
                    thread_idx=lambda thread=thread: (
                        thread % threads,
                        thread // threads,
                        0,
                    )
                ),
                make_tensor=lambda pointer, layout: pointer.tensor,
            ),
            "x": left,
            "y": right,
            "out": output,
            "width": x.shape[1],
            "rows": x.shape[0],
            "initial": initial,
        }
        exec(code, namespace)
        generators.append(namespace["kernel"]())
    pending = [None] * len(generators)
    counts: dict[str, int] = {}
    with np.errstate(all="ignore"):
        for _step in range(1000):
            events = []
            for thread, generator in enumerate(generators):
                try:
                    events.append(generator.send(pending[thread]))
                except StopIteration:
                    events.append(None)
            if all(event is None for event in events):
                break
            assert all(event is not None for event in events)
            assert len({event[:2] for event in events}) == 1
            kind = events[0][1]
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "cute.arch.alloc_smem":
                dtype, size = events[0][2:]
                pointer = _Tensor(np.empty(size, dtype=dtype)).iterator
                pending = [pointer] * len(generators)
            elif kind == "cute.arch.sync_threads":
                pending = [None] * len(generators)
            else:
                assert kind == "cute.arch.warp_reduction_sum"
                pending = []
                for warp in range(0, len(generators), 32):
                    values = [event[2] for event in events[warp : warp + 32]]
                    value = values[0]
                    for other in values[1:]:
                        value = np.float32(value + other)
                    pending.extend([value] * 32)
        else:
            raise AssertionError("Too many collective steps")
    assert np.all(left.reads == 1) and np.all(right.reads == 1)
    assert np.all(output.writes == 1)
    return output.values, counts


def _reference(x: np.ndarray, y: np.ndarray, block: int, initial: float) -> np.ndarray:
    carry = np.full(x.shape[0], np.float32(initial))
    with np.errstate(all="ignore"):
        products = np.multiply(x, y, dtype=np.float16)
        for offset in range(0, x.shape[1], block):
            reduced = (
                products[:, offset : offset + block]
                .astype(np.float32)
                .sum(axis=1, dtype=np.float32)
                .astype(np.float16)
            )
            carry = np.add(carry, reduced.astype(np.float32), dtype=np.float32)
    return carry


@pytest.mark.parametrize(
    "threads,slots,vector,groups",
    [(32, 2, 1, 2), (64, 3, 4, 2), (128, 2, 8, 1), (1024, 4, 8, 1)],
)
@pytest.mark.parametrize("extent", ["empty", "short", "full", "tail", "multi"])
@pytest.mark.parametrize("resident", [False, True])
def test_single_sequence_typed_product_and_per_tile_carry(
    threads: int, slots: int, vector: int, groups: int, extent: str, resident: bool
) -> None:
    block = threads * slots * vector
    width = {
        "empty": 0,
        "short": 3,
        "full": block,
        "tail": block + 17,
        "multi": block * 2 + 1,
    }[extent]
    values = np.array([-1, -0.5, -0.125, 0.0625, 0.25, 1], dtype=np.float16)
    random = np.random.default_rng(42)
    x = random.choice(values, (groups, width))
    y = random.choice(values, (groups, width))
    lowered = _rewrite(
        _source(threads, slots, vector),
        threads=threads,
        slots=slots,
        vector=vector,
        groups=groups,
        resident=resident,
    )
    assert "sequence_values" not in lowered
    actual, counts = _execute(
        lowered, x, y, threads=threads, groups=groups, initial=0.125
    )
    np.testing.assert_array_equal(actual, _reference(x, y, block, 0.125))
    assert counts.get("cute.arch.warp_reduction_sum", 0) == math.ceil(width / block)
    assert counts.get("cute.arch.sync_threads", 0) == (
        2 if threads > 32 else 0
    ) * math.ceil(width / block)


@pytest.mark.parametrize(
    "witness",
    ["product_rounding", "tile_sum_rounding", "carry_order", "overflow", "nan"],
)
def test_single_sequence_rounding_witnesses(witness: str) -> None:
    x = np.zeros((1, 128), dtype=np.float16)
    y = np.ones_like(x)
    initial = 0.0
    if witness == "product_rounding":
        x[0, :2] = [1 + 2**-10, -1]
        y[0, 0] = 1 - 2**-10
        assert np.float16(x[0, 0] * y[0, 0]) == 1
        assert np.float32(x[0, 0]) * np.float32(y[0, 0]) != 1
    elif witness == "tile_sum_rounding":
        x[0, [0, 1, 64]] = [1, 2**-11, 2**-10]
        expected = _reference(x, y, 64, 0)
        wrong = (
            np.multiply(x, y, dtype=np.float16)
            .astype(np.float32)
            .sum()
            .astype(np.float16)
        )
        assert expected[0] != np.float32(wrong)
    elif witness == "carry_order":
        initial = 2**24
        x[0, [0, 64]] = [1, -1]
        assert _reference(x, y, 64, initial)[0] != np.float32(initial)
    elif witness == "overflow":
        x[0, [0, 1, 64, 65]] = [65504, 65504, -65504, -65504]
    else:
        x[0, 0] = np.nan
    lowered = _rewrite(_source(32, 2, 1), threads=32, slots=2, vector=1, groups=1)
    lowered = ast.unparse(
        ast.Module(body=merge_sibling_v_loops(ast.parse(lowered).body), type_ignores=[])
    )
    actual, counts = _execute(lowered, x, y, threads=32, groups=1, initial=initial)
    np.testing.assert_array_equal(actual, _reference(x, y, 64, initial))
    assert counts["cute.arch.warp_reduction_sum"] == 2


@pytest.mark.parametrize(
    "change",
    ["zero_markers", "unknown_effect", "aliased_write", "liveout", "lane_bound"],
)
def test_single_sequence_keeps_negative_proofs(change: str) -> None:
    source = _source(32, 2, 1)
    if change == "zero_markers":
        module = ast.parse(source)
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == _MARKER
            ):
                node.func = ast.Attribute(
                    value=ast.Name(id="cutlass", ctx=ast.Load()),
                    attr="Float32",
                    ctx=ast.Load(),
                )
                node.args = node.args[:1]
                node.keywords = []
        source = ast.unparse(module)
    elif change == "unknown_effect":
        source = source.replace(
            "product = left * right",
            "effect = unknown(left)\n            product = left * right",
        )
    elif change == "aliased_write":
        source += "(x.iterator).store(cutlass.Float16(0))\n"
    elif change == "liveout":
        source = source.replace(
            ".store(carry)", ".store(carry + cutlass.Float32(left))"
        )
    else:
        source = source.replace(
            "cutlass.Int32(width),",
            "cutlass.Int32(width) + cutlass.Int32(cute.arch.thread_idx()[0]),",
        )
    with pytest.raises(exc.BackendUnsupported):
        _rewrite(source, threads=32, slots=2, vector=1, groups=1)


def test_single_sequence_requires_disjoint_outputs() -> None:
    with pytest.raises(exc.BackendUnsupported):
        _rewrite(
            _source(32, 2, 1), threads=32, slots=2, vector=1, groups=1, disjoint=False
        )


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("columns", [128, 513, 2048])
@skipUnlessBackends(["cute"])
def test_original_backward_single_reduction_seeds_roundtrip(
    static: bool, columns: int
) -> None:
    kernel = helion.kernel(
        softmax_bwd.fn, backend="cute", static_shapes=static, autotune_effort="none"
    )
    inputs = tuple(torch.empty((4, columns), dtype=torch.float16) for _ in range(2))
    with (
        # Binding specializes hardware facts even for CPU tensors. The imported
        # helper's target mocks must not fall through to the visible CUDA device.
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, inputs)
        host = bound.host_function
        assert host is not None
        with bound.env:
            seeds = CuteResidentSequenceHeuristic.get_seed_configs(
                bound.env, host.device_ir
            )
            assert seeds
            generation = ConfigGeneration(bound.config_spec)
        for seed in seeds:
            if seed.config["cute_reduction_sequence"] != "reload":
                continue
            flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
            assert generation.unflatten(flat) == roundtrip
            assert roundtrip.config["cute_reduction_sequence"] == "reload"
            source = bound.to_code(roundtrip)
            assert "sequence_acc" in source and "sequence_values" not in source
            assert _MARKER not in source
            assert "sum_1 = cutlass.Float16(" in source
            assert "v_1 = cutlass.Float32(sum_1)" in source
            module = ast.parse(source)
            for node in ast.walk(module):
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id.startswith(("lane_", "vec_lane_"))
                ):
                    assert not any(
                        isinstance(child, ast.Call)
                        and "warp_reduction" in ast.unparse(child.func)
                        for child in ast.walk(node)
                    )
