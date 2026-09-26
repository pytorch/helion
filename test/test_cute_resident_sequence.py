from __future__ import annotations

import ast
import itertools
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.resident_sequence import _MARKER
from helion._compiler.cute.resident_sequence import SequenceRegion
from helion._compiler.cute.resident_sequence import materialize_resident_sequences
from helion._compiler.tile_strategy import _lane_reduce_marker_expr
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Generator


def _reduce(value: str, kind: str = "sum", dtype: str = "Float32") -> str:
    identity = "float('-inf')" if kind == "max" else "0"
    return _lane_reduce_marker_expr(
        value, kind, f"cutlass.{dtype}({identity})", 32
    ).replace("_helion_lane_reduce", _MARKER)


def _source(
    *, threads: int = 32, slots: int = 4, vector: int = 1, softmax: bool = False
) -> str:
    if vector == 1:
        position = f"column = offset + cutlass.Int32(lane) * {threads} + cutlass.Int32(cute.arch.thread_idx()[0])"
        load = "value = cutlass.Float32((x.iterator + row * width + column).load()) if column < width and row < rows else cutlass.Float32(0)"
        inner_prefix = f"{position}\n{load}"
    else:
        position = f"base = offset + (cutlass.Int32(lane) * {threads} + cutlass.Int32(cute.arch.thread_idx()[0])) * {vector}"
        inner_prefix = "column = base + cutlass.Int32(vec)\nvalue = cutlass.Uint32(packet[vec]).bitcast(cutlass.Float32)"
    if softmax:
        initial = "state_max = cutlass.Float32(initial_max)\nstate_sum = cutlass.Float32(initial_sum)"
        calculation = f"""old_max = state_max
old_sum = state_sum
masked = value if column < width else cutlass.Float32(float('-inf'))
maximum = {_reduce("masked", "max")}
new_max = cute.math.max(old_max, maximum, propagate_nan=True)
rescaled = old_sum * cute.math.exp(old_max - new_max)
exponential = cute.math.exp(value - new_max) if column < width else cutlass.Float32(0)
summed = {_reduce("exponential")}
new_sum = rescaled + summed
"""
        finish = "(out.iterator + row * 2).store(state_max)\n(out.iterator + row * 2 + 1).store(state_sum)"
    else:
        initial = "count = cutlass.Float32(initial_count)\nmean = cutlass.Float32(initial_mean)\nmoment = cutlass.Float32(initial_moment)"
        calculation = f"""old_count = count
old_mean = mean
old_moment = moment
valid = cutlass.Int64(column < width)
counted = {_reduce("valid", dtype="Int64")}
summed = {_reduce("value")}
average = summed / cutlass.Float32(counted)
centered = value - average if column < width else cutlass.Float32(0)
squared = centered * centered
variance = {_reduce("squared")}
delta = average - old_mean
new_count = old_count + cutlass.Float32(counted)
weight = cutlass.Float32(counted) / new_count
new_mean = old_mean + delta * weight
cross = old_count * cutlass.Float32(counted) / new_count
new_moment = old_moment + variance + (delta * delta) * cross
"""
        finish = "(out.iterator + row * 3).store(count)\n(out.iterator + row * 3 + 1).store(mean)\n(out.iterator + row * 3 + 2).store(moment)"
    scalar = inner_prefix + "\n" + calculation
    if vector > 1:
        scalar = (
            position
            + f"\npacket = cute.arch.load(x.iterator + row * width + base, ir.VectorType.get([{vector}], cutlass.Uint32.mlir_type))\nfor vec in cutlass.range_constexpr({vector}):\n"
            + _indent(scalar)
        )
    return (
        "row = cutlass.Int32(cute.arch.thread_idx()[1])\n"
        + initial
        + f"\nfor offset in range(cutlass.Int32(0), cutlass.Int32(width), cutlass.Int32({threads * slots * vector})):\n    for lane in range({slots}):\n"
        + _indent(scalar, 8)
        + "\nif row < rows and cute.arch.thread_idx()[0] == 0:\n"
        + _indent(finish)
        + "\n"
    )


def _indent(source: str, width: int = 4) -> str:
    return "\n".join(" " * width + line for line in source.splitlines())


def _rewrite(
    source: str,
    *,
    threads: int = 32,
    groups: int = 2,
    slots: int = 4,
    vector: int = 1,
    resident: bool = True,
    disjoint: bool = True,
    extra_renames: dict[str, str] | None = None,
) -> tuple[list[ast.stmt], str]:
    body = ast.parse(source).body
    (loop,) = [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "offset"
    ]
    lane = loop.body[0]
    assert isinstance(lane, ast.For)
    setattr(lane, HELION_LANE_LOOP_VAR_ATTR, "lane")
    for statement in body:
        for candidate in ast.walk(statement):
            if (
                isinstance(candidate, ast.For)
                and isinstance(candidate.target, ast.Name)
                and candidate.target.id == "consume_lane"
            ):
                setattr(candidate, HELION_LANE_LOOP_VAR_ATTR, "consume_lane")
    region = SequenceRegion(
        loop,
        "lane",
        slots,
        "vec" if vector > 1 else None,
        vector,
        0,
        threads,
        threads * groups,
        f"cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * {threads}",
    )
    counter = itertools.count()
    rewritten = materialize_resident_sequences(
        list(body),
        regions={id(loop): region},
        tensor_dtypes={
            "x": "cutlass.Float32",
            "out": "cutlass.Float32",
            "copied": "cutlass.Float32",
            "metadata": "cutlass.Int32",
        },
        rename_groups={
            "new_count": "count",
            "new_mean": "mean",
            "new_moment": "moment",
            "new_sum": "state_sum",
            "new_max": "state_max",
            **(extra_renames or {}),
        },
        disjoint_pairs={
            frozenset((left, right))
            for left, right in itertools.combinations(
                ("x", "out", "copied", "metadata"), 2
            )
        }
        if disjoint
        else set(),
        boundary_names={
            "x",
            "out",
            "copied",
            "width",
            "rows",
            "initial_count",
            "initial_mean",
            "initial_moment",
            "initial_max",
            "initial_sum",
            "metadata",
        },
        resident=resident,
        new_var=lambda hint: f"{hint}_{next(counter)}",
    )

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            aliases = {
                "new_count": "count",
                "new_mean": "mean",
                "new_moment": "moment",
                "new_sum": "state_sum",
                "new_max": "state_max",
            }
            node.id = aliases.get(node.id, node.id)
            return node

    module = ast.Module(body=cast("list[ast.stmt]", rewritten), type_ignores=[])
    Rename().visit(module)
    return body, ast.unparse(module)


class _Pointer:
    def __init__(self, tensor: _Tensor, offset: int = 0) -> None:
        self.tensor = tensor
        self.offset = int(offset)

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.tensor, self.offset + int(offset))

    def load(self) -> Any:
        assert 0 <= self.offset < self.tensor.values.size
        self.tensor.reads[self.offset] += 1
        return self.tensor.values.flat[self.offset]

    def store(self, value: Any) -> None:
        assert 0 <= self.offset < self.tensor.values.size
        self.tensor.writes[self.offset] += 1
        self.tensor.values.flat[self.offset] = value


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.reads = np.zeros(values.size, dtype=np.int64)
        self.writes = np.zeros(values.size, dtype=np.int64)
        self.iterator = _Pointer(self)

    def __getitem__(self, index: Any) -> Any:
        return self.values[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        self.values[index] = value


class _YieldCollectives(ast.NodeTransformer):
    """Execute the emitted CTA as lockstep Python coroutines on the CPU."""

    def __init__(self) -> None:
        self.counter = itertools.count()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        path = ast.unparse(node.func)
        if path in {
            "cute.arch.warp_reduction_sum",
            "cute.arch.warp_reduction",
            "cute.arch.sync_threads",
            "cute.arch.alloc_smem",
        }:
            return ast.Yield(
                value=ast.Tuple(
                    elts=[
                        ast.Constant(value=next(self.counter)),
                        ast.Constant(value=path),
                        *node.args,
                    ],
                    ctx=ast.Load(),
                )
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
            return ast.Call(
                func=ast.Name(id="bitcast", ctx=ast.Load()),
                args=[node.func.value, *node.args],
                keywords=[],
            )
        return node


def _execute(
    source: str,
    x: np.ndarray,
    *,
    rows: int,
    threads: int,
    groups: int,
    initial: tuple[float, ...],
    softmax: bool = False,
    copied: _Tensor | None = None,
) -> tuple[np.ndarray, _Tensor]:
    tree = ast.parse("def kernel():\n" + _indent(source))
    _YieldCollectives().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<resident sequence CPU SIMT>", "exec")
    tensor = _Tensor(x.copy())
    output = _Tensor(np.full((rows, 2 if softmax else 3), np.float32(np.nan)))
    generators: list[Generator[Any, Any, Any]] = []

    class Uint32:
        mlir_type = np.uint32

        def __new__(cls, value: Any) -> np.uint32:
            return np.uint32(value)

    def vector_load(
        pointer: _Pointer, layout: tuple[int, Any], **hints: str
    ) -> np.ndarray:
        count, dtype = layout
        return np.asarray([(pointer + index).load() for index in range(count)]).view(
            dtype
        )

    for thread in range(threads * groups):
        namespace: dict[str, Any] = {
            "operator": operator,
            "_cute_load_l2_evict_last": vector_load,
            "cutlass": SimpleNamespace(
                Boolean=np.bool_,
                Int32=np.int32,
                Int64=np.int64,
                Uint32=Uint32,
                Float32=np.float32,
                range_constexpr=range,
            ),
            "ir": SimpleNamespace(
                VectorType=SimpleNamespace(get=lambda shape, dtype: (shape[0], dtype))
            ),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(
                    thread_idx=lambda thread=thread: (
                        thread % threads,
                        thread // threads,
                        0,
                    ),
                    load=vector_load,
                ),
                math=SimpleNamespace(
                    max=lambda a, b, **kwargs: np.maximum(a, b),
                    min=lambda a, b, **kwargs: np.minimum(a, b),
                    exp=lambda value: np.float32(np.exp(value)),
                ),
                make_rmem_tensor=lambda shape, dtype: np.empty(shape, dtype=dtype),
                make_tensor=lambda pointer, layout: pointer.tensor,
            ),
            "bitcast": lambda value, dtype: np.asarray(value).view(dtype)[()],
            "x": tensor,
            "out": output,
            "copied": copied,
            "width": x.shape[1],
            "rows": rows,
            "initial_max": initial[0] if softmax else 0,
            "initial_sum": initial[1] if softmax else 0,
            "initial_count": initial[0] if not softmax else 0,
            "initial_mean": initial[1] if not softmax else 0,
            "initial_moment": initial[2] if not softmax else 0,
        }
        exec(code, namespace)
        generators.append(namespace["kernel"]())
    pending: list[Any] = [None] * len(generators)
    finished = [False] * len(generators)
    with np.errstate(all="ignore"):
        for _step in range(10000):
            events = []
            for thread, generator in enumerate(generators):
                try:
                    events.append(generator.send(pending[thread]))
                except StopIteration:
                    events.append(None)
                    finished[thread] = True
            if all(finished):
                break
            assert not any(finished), "divergent collective completion"
            assert len({event[:2] for event in events}) == 1, (
                "divergent collective order"
            )
            kind = events[0][1]
            if kind == "cute.arch.alloc_smem":
                dtype, size = events[0][2:]
                pointer = _Tensor(np.empty(size, dtype=dtype)).iterator
                pending = [pointer] * len(generators)
            elif kind == "cute.arch.sync_threads":
                pending = [None] * len(generators)
            else:
                pending = []
                for warp in range(0, len(generators), 32):
                    values = [event[2] for event in events[warp : warp + 32]]
                    value = values[0]
                    for other in values[1:]:
                        value = (
                            value + other
                            if kind.endswith("_sum")
                            else events[warp][3](value, other)
                        )
                    pending.extend([value] * 32)
        else:
            raise AssertionError("too many collective steps")
    assert np.all(output.writes == 1)
    return output.values, tensor


def _reference(
    x: np.ndarray, block: int, initial: tuple[float, ...], *, softmax: bool = False
) -> np.ndarray:
    result = np.tile(np.array(initial, dtype=np.float32), (x.shape[0], 1))
    with np.errstate(all="ignore"):
        for offset in range(0, x.shape[1], block):
            value = x[:, offset : offset + block].astype(np.float32)
            if softmax:
                maximum = np.maximum(result[:, 0], value.max(axis=1))
                summed = result[:, 1] * np.exp(result[:, 0] - maximum) + np.exp(
                    value - maximum[:, None]
                ).sum(axis=1)
                result = np.stack((maximum, summed), axis=1)
            else:
                count = np.float32(value.shape[1])
                mean = value.sum(axis=1) / count
                variance = np.square(value - mean[:, None]).sum(axis=1)
                delta = mean - result[:, 1]
                total = result[:, 0] + count
                updated_mean = result[:, 1] + delta * (count / total)
                moment = (
                    result[:, 2]
                    + variance
                    + delta * delta * (result[:, 0] * count / total)
                )
                result = np.stack((total, updated_mean, moment), axis=1)
    return result


@pytest.mark.parametrize(
    "threads,groups,slots", [(32, 1, 2), (32, 4, 4), (64, 2, 2), (128, 1, 4)]
)
@pytest.mark.parametrize("tail", [0, 1, 17])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("softmax", [False, True])
def test_ordered_sequence_cpu_semantics(
    threads: int, groups: int, slots: int, tail: int, resident: bool, softmax: bool
) -> None:
    source = _source(threads=threads, slots=slots, softmax=softmax)
    original, lowered = _rewrite(
        source, threads=threads, groups=groups, slots=slots, resident=resident
    )
    assert _MARKER in ast.unparse(ast.Module(body=original, type_ignores=[]))
    assert _MARKER not in lowered
    rows = max(groups - 1, 1)
    columns = threads * slots * 2 + tail
    rng = np.random.default_rng(771)
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    initial = (-0.25, 0.75) if softmax else (3.0, -0.75, 2.0)
    actual, tensor = _execute(
        lowered,
        values,
        rows=rows,
        threads=threads,
        groups=groups,
        initial=initial,
        softmax=softmax,
    )
    expected = _reference(values, threads * slots, initial, softmax=softmax)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-5)
    assert np.all(tensor.reads == (1 if resident else 2))


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize("columns", [0, 1, 127, 128, 129])
@pytest.mark.parametrize("initial_zero", [False, True])
def test_empty_tail_and_nonfinite_sequences(
    softmax: bool, columns: int, initial_zero: bool
) -> None:
    _, lowered = _rewrite(_source(softmax=softmax))
    values = np.linspace(-3, 3, 2 * columns, dtype=np.float32).reshape(2, columns)
    if columns > 2:
        values[0, 1] = np.nan
        values[1, 2] = np.inf
    initial = (
        ((-np.inf, 0.0) if initial_zero else (-1.0, 2.0))
        if softmax
        else ((0.0, 0.0, 0.0) if initial_zero else (3.0, -1.0, 2.0))
    )
    actual, _ = _execute(
        lowered, values, rows=2, threads=32, groups=2, initial=initial, softmax=softmax
    )
    expected = _reference(values, 128, initial, softmax=softmax)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-5, equal_nan=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "store",
        "atomic",
        "unknown",
        "shadow",
        "carry",
        "undefined",
        "outer_if",
        "outer_loop",
        "asm",
        "bound_store",
        "prior_loop",
    ],
)
def test_sequence_rejects_unproved_effects_and_dependencies(mutation: str) -> None:
    source = _source()
    renames = {}
    if mutation == "store":
        source = source.replace(
            "        valid =", "        (x.iterator).store(value)\n        valid ="
        )
    elif mutation == "atomic":
        source = "cute.arch.atomic_add(x.iterator, cutlass.Float32(1))\n" + source
    elif mutation == "unknown":
        source = source.replace("centered * centered", "unmodelled(centered)")
    elif mutation == "shadow":
        source = source.replace(
            "        valid =", "        value = cutlass.Float32(4)\n        valid ="
        )
    elif mutation == "carry":
        source = source.replace(
            "        valid =", "        extra_new = mean + value\n        valid ="
        )
        renames = {"extra_new": "mean"}
    elif mutation == "undefined":
        source = source.replace("centered * centered", "centered * unavailable")
    elif mutation in {"outer_if", "outer_loop", "asm"}:
        prefix = {
            "outer_if": "if cute.arch.thread_idx()[0] < 16:\n",
            "outer_loop": "for trip in range(cute.arch.thread_idx()[0] + 1):\n",
            "asm": "if cute.arch.inline_asm('mov.u32 $0, %tid.x;', has_side_effects=False):\n",
        }[mutation]
        source = prefix + _indent(source)
    elif mutation == "bound_store":
        source = (
            "width = cutlass.Int32(metadata.iterator.load())\n"
            + source
            + "metadata.iterator.store(cutlass.Int32(0))\n"
        )
    elif mutation == "prior_loop":
        source = source.replace(
            "for offset in",
            "for warmup in range(1):\n    previous = cutlass.Float32(cute.arch.thread_idx()[0])\nfor offset in",
        )
        renames = {"previous": "mean"}
    with pytest.raises(exc.BackendUnsupported, match="sequence proof declined"):
        _rewrite(source, extra_renames=renames)


def test_sequence_requires_disjoint_written_storage() -> None:
    with pytest.raises(exc.BackendUnsupported, match="sequence proof declined"):
        _rewrite(_source(), disjoint=False)


@pytest.mark.parametrize(
    "extra",
    [
        "out.append(cutlass.Float32(1))",
        "unmodelled = x[0, 0]",
        "out[0, 0] = cutlass.Float32(1)",
    ],
)
def test_sequence_rejects_unmodelled_memory_effects(extra: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="sequence proof declined"):
        _rewrite(_source() + extra + "\n")


def test_sequence_rejects_excessive_reduction_graph() -> None:
    source = _source().replace(
        "        valid =",
        "".join(f"        extra{i} = {_reduce('value')}\n" for i in range(17))
        + "        valid =",
    )
    with pytest.raises(exc.BackendUnsupported, match="sequence proof declined"):
        _rewrite(source)


def test_sequence_keeps_vector_load_and_loop_metadata() -> None:
    _, source = _rewrite(_source(vector=4), vector=4)
    assert source.count("cute.arch.load(") == 1
    assert "sequence_values" in source
    assert "cutlass.range_constexpr(4)" in source


@pytest.mark.parametrize("kind", ["welford", "softmax"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("online_rewrite", [False, True])
@skipUnlessBackends(["cute"])
def test_unchanged_examples_emit_logical_reduction_phases(
    kind: str,
    static: bool,
    dtype: torch.dtype,
    online_rewrite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import patch

    from examples.aot_example import row_softmax
    from examples.welford import welford

    from test._cute_binding import _cpu_bind

    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "0" if online_rewrite else "1")
    function = welford.fn if kind == "welford" else row_softmax.fn
    kernel = helion.kernel(
        function, backend="cute", static_shapes=static, autotune_effort="none"
    )
    matrix = torch.empty((17, 77), dtype=dtype)
    arguments = (
        (torch.empty(77, dtype=dtype), torch.empty(77, dtype=dtype), matrix)
        if kind == "welford"
        else (matrix,)
    )
    count = 3 if kind == "welford" else 2
    config = helion.Config(
        block_sizes=[4, *([128] * (count - 1))],
        num_threads=[4, *([32] * (count - 1))],
        cute_vector_widths=[1, *([1] * (count - 1))],
        cute_lane_layouts=["blocked", *(["strided"] * (count - 1))],
        cute_reduction_sequence="resident",
        cute_cluster_n=1,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, arguments)
        code = bound.to_code(config)
    assert "_helion_sequence_reduce" not in code
    assert "sequence_acc" in code
    # Preserve the original dependent-phase cache coverage with the rewrite
    # disabled. Independent dynamic max/sum sweeps have no intra-tile replay;
    # static one-trip sweeps can still use the existing sibling-loop cache.
    assert ("cute.make_rmem_tensor" in code) == (
        kind != "softmax" or not online_rewrite or static
    )
    if kind == "softmax":
        tree = ast.parse(code)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        reductions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func)
            in {"cute.arch.warp_reduction", "cute.arch.warp_reduction_sum"}
        ]
        assert sorted(ast.unparse(node.func) for node in reductions) == [
            *(["cute.arch.warp_reduction"] * (2 if online_rewrite else 1)),
            "cute.arch.warp_reduction_sum",
        ]
        for node in reductions:
            current: ast.AST = node
            while current in parents:
                current = parents[current]
                if isinstance(current, ast.For):
                    assert isinstance(current.target, ast.Name)
                    assert current.target.id.startswith("tile_offset_")
