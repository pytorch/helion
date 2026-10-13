from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from test.test_cute_chained_rectangular_leaf import _environment

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_store_expression import StorePoint
from helion._compiler.cute.chained_warp_global_store import render_warp_global_store


def fixture(*, mask="r < 4", extra_lines=(), indices=None, dtype=torch.bfloat16):
    graph = Graph()
    output = graph.placeholder("output")
    output.meta["val"] = torch.empty((3, 8, 16), dtype=dtype)
    value = graph.placeholder("value")
    store = graph.call_function(chain.memory_ops.store, (output, (), value))
    point = StorePoint(
        (4, 8),
        ("r", "c"),
        (
            "rr = cutlass.Int32(chain_origin_1) + r",
            *extra_lines,
            "rounded = cutlass.BFloat16(acc[element])",
        ),
        "output",
        indices or ("chain_origin_0", "rr", "c"),
        ("0 <= chain_origin_0 < 3", "0 <= rr < 8", "0 <= c < 16", mask),
        "rounded",
        (),
        (),
    )
    plan = SimpleNamespace(axes=((0, 3, 1), (1, 8, 4)), loop=None)
    cg = SimpleNamespace(device_function=SimpleNamespace(_expr_args={}))
    return cg, plan, store, point


def render(data, **kwargs):
    cg, plan, store, point = data
    backend = SimpleNamespace(
        dtype_str=lambda dtype: {
            torch.bfloat16: "cutlass.BFloat16",
            torch.float16: "cutlass.Float16",
            torch.float32: "cutlass.Float32",
        }[dtype]
    )
    with patch.object(
        CompileEnvironment,
        "current",
        return_value=SimpleNamespace(
            index_type=lambda: "cutlass.Int32", backend=backend
        ),
    ):
        return render_warp_global_store(
            cg,
            plan,
            store,
            point,
            accumulator="acc",
            thread_slice="actual_slice",
            shape=(4, 8),
            prefix="export",
            index="element",
            fallback=("fallback_calls.append('original')",),
            **kwargs,
        )


def execute(lines, *, row=0, group=1, address=4096, stride=16):
    namespace = _environment(chain_origin_0=group, chain_origin_1=row)
    cutlass = namespace["cutlass"]
    cutlass.BFloat16 = lambda x: float(torch.tensor(x, dtype=torch.bfloat16))
    cutlass.Boolean = bool
    cutlass.range_constexpr = range
    storage = [None] * (3 * 8 * 16)

    class Pointer:
        def __init__(self, offset=0):
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.offset + int(offset))

        def toint(self):
            return address + 2 * self.offset

        def align(self, alignment):
            assert self.toint() % alignment == 0
            return self

    output = SimpleNamespace(
        shape=(3, 8, 16),
        layout=SimpleNamespace(stride=(128, stride, 1)),
        iterator=Pointer(),
    )
    # A deliberately non-row-major fragment. Both copies use the identical
    # supplied owner map; the helper must never infer warp factors/permutations.
    coordinates = [(r, c) for c in range(8) for r in (2, 0, 3, 1)]
    calls = []

    class Slice:
        def partition_C(self, tensor):
            calls.append(tensor)
            return coordinates if tensor == "identity" else (tensor, coordinates)

    def autovec(values, destination):
        (pointer, layout), coords = destination
        for value, (r, c) in zip(values, coords, strict=True):
            offset = pointer.offset + r * layout[1][0] + c * layout[1][1]
            assert storage[offset] is None
            storage[offset] = value

    cute = SimpleNamespace(
        make_tensor=lambda pointer, layout: (pointer, layout),
        make_layout=lambda shape, *, stride: (shape, stride),
        make_identity_tensor=lambda shape: "identity",
        make_rmem_tensor_like=lambda acc, dtype: [None] * len(acc),
        size=len,
        autovec_copy=autovec,
    )
    accumulator = [float(r * 8 + c) + 0.125 for r, c in coordinates]
    namespace.update(
        cute=cute,
        output=output,
        actual_slice=Slice(),
        acc=accumulator,
        fallback_calls=[],
    )
    exec(compile("\n".join(lines), "<actual-affine-store-render>", "exec"), namespace)
    return namespace, storage, calls, coordinates


def test_affine_view_keeps_actual_fragment_owner_and_original_narrowing():
    lines = render(fixture())
    assert lines is not None
    namespace, storage, calls, coordinates = execute(lines)
    assert namespace["fallback_calls"] == [] and len(calls) == 2
    for r, c in coordinates:
        assert storage[128 + r * 16 + c] == float(
            torch.tensor(r * 8 + c + 0.125, dtype=torch.bfloat16)
        )
    assert sum(v is not None for v in storage) == 32
    source = "\n".join(lines)
    assert "rounded = cutlass.BFloat16(acc[element])" in source
    assert "cute.autovec_copy" in source
    assert not any(
        word in source for word in ("sync_threads", "alloc_smem", "make_tiled_mma")
    )


@pytest.mark.parametrize(
    "options",
    (
        {"row": 6},
        {"group": 3},
        {"address": 4098},
        {"address": -512},
        {"address": 2**63 - 16},
        {"stride": 15},
    ),
)
def test_runtime_bounds_layout_and_pointer_failures_preserve_scalar_fallback(options):
    lines = render(fixture())
    assert lines is not None
    namespace, storage, calls, _ = execute(lines, **options)
    assert namespace["fallback_calls"] == ["original"]
    assert not calls and all(v is None for v in storage)


def test_partial_store_mask_cannot_enable_whole_rectangle():
    lines = render(fixture(mask="r < 3"))
    if lines is not None:
        namespace, storage, calls, _ = execute(lines)
        assert namespace["fallback_calls"] == ["original"]
        assert not calls and all(v is None for v in storage)


@pytest.mark.parametrize(
    "change",
    (
        "data_index",
        "reassigned_index",
        "complex_statement",
        "transposed",
        "wrong_shape",
        "unsupported_dtype",
    ),
)
def test_unproven_coordinates_decline_without_rewriting_scalar_semantics(change):
    data = list(fixture())
    if change == "data_index":
        data[3] = replace(data[3], indices=("chain_origin_0", "rr", "acc[element]"))
    elif change == "reassigned_index":
        data[3] = replace(data[3], lines=("rr = r", "rr = c"))
    elif change == "complex_statement":
        data[3] = replace(data[3], lines=("if r: rr = c",))
    elif change == "transposed":
        data[3] = replace(data[3], indices=("chain_origin_0", "c", "r"))
    elif change == "wrong_shape":
        data[3] = replace(data[3], shape=(8, 4))
    else:
        data[2].args[0].meta["val"] = torch.empty((3, 8, 16), dtype=torch.int64)
    assert render(data) is None


def test_exact_fallback_ast_is_retained_as_the_only_false_path():
    lines = render(fixture())
    assert lines is not None
    tree = ast.parse("\n".join(lines))
    last = cast("ast.If", tree.body[-1])
    assert ast.dump(ast.Module(body=last.body, type_ignores=[])) == ast.dump(
        ast.parse("fallback_calls.append('original')")
    )


@pytest.mark.parametrize("captured_value, admitted", ((1, True), (1.5, False)))
def test_only_typed_uniform_loop_capture_may_supply_global_origin(
    captured_value, admitted
):
    data = list(fixture())
    graph = Graph()
    captured = graph.placeholder("capture")
    captured.meta["val"] = captured_value
    data[1].loop = SimpleNamespace(captures=lambda: {captured: "chain_loop_capture_0"})
    data[3] = replace(data[3], indices=("chain_loop_capture_0", "rr", "c"))
    result = render(data)
    assert (result is not None) == admitted
