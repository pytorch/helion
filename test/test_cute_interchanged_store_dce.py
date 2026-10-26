from __future__ import annotations

import ast
from itertools import combinations
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_persistent_reduction_coverage import _assert_fragment_norm_coverage

import helion
from helion import exc
from helion._compiler import tile_strategy
from helion._compiler.cute.interchanged_store_dce import eliminate_interchanged_stores
from helion._testing import skipUnlessBackends
import helion.language as hl

_PAIRS = {
    frozenset(pair)
    for pair in combinations(("source", "weight", "output", "column"), 2)
}


def _program() -> ast.For:
    loop = ast.parse(
        """
for lane in range(4):
    col = lane
    grad_w = 0.0
    for row in range(rows):
        index = row * 4 + col
        mask = row < valid_rows and col < valid_cols
        value = (source.iterator + index).load() if mask else 0.0
        grad_w = grad_w + value
        w = (weight.iterator + col).load()
        product = value * w
        reduced = _helion_lane_reduce(product, 'sum', 0.0, 1, 1, 0, '', 1)
        scaled = reduced * 2.0
        result = product - scaled
        if mask:
            (output.iterator + index).store(result)
    (column.iterator + col).store(grad_w)
"""
    ).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, "lane")
    return loop


def _interchanged(loop: ast.For) -> tuple[ast.For, ast.For, list[ast.AST], set[int]]:
    with patch(
        "helion._compiler.cute.interchanged_store_dce.eliminate_interchanged_stores",
        return_value=0,
    ):
        body = tile_strategy.interchange_lane_outside_serial_reductions([loop])
    first = body[0]
    assert isinstance(first, ast.For)
    serial = next(
        statement for statement in first.body if isinstance(statement, ast.For)
    )
    candidates = {
        index
        for index, statement in enumerate(serial.body)
        if "output.iterator" in ast.unparse(statement)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            for node in ast.walk(statement)
        )
    }
    return first, serial, body[1:], candidates


def _prune(
    loop: ast.For,
    pairs: set[frozenset[str]] | None = None,
    protected: set[str] | None = None,
) -> tuple[int, list[ast.AST]]:
    first, serial, final, candidates = _interchanged(loop)
    removed = eliminate_interchanged_stores(
        first,
        serial,
        final,
        candidates,
        _PAIRS if pairs is None else pairs,
        protected or set(),
    )
    return removed, [first, *final]


class _Buffer:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values.copy()
        self.loads = 0
        self.stores = 0

    @property
    def iterator(self) -> _Pointer:
        return _Pointer(self, 0)


class _Pointer:
    def __init__(self, buffer: _Buffer, offset: int) -> None:
        self.buffer = buffer
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.buffer, self.offset + offset)

    def load(self) -> float:
        self.buffer.loads += 1
        return float(self.buffer.values[self.offset])

    def store(self, value: float) -> None:
        self.buffer.stores += 1
        self.buffer.values[self.offset] = value


def _execute(
    body: list[ast.AST], rows: int, valid_rows: int, valid_cols: int
) -> dict[str, _Buffer]:
    body = tile_strategy.split_lane_loop_reductions(
        body,
        uniform_names={
            "source",
            "weight",
            "output",
            "column",
            "rows",
            "valid_rows",
            "valid_cols",
        },
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    body = tile_strategy.restore_unprocessed_lane_reduce_markers(body)
    buffers = {
        "source": _Buffer(np.arange(rows * 4, dtype=np.float64) + 0.5),
        "weight": _Buffer(np.array([1.0, 2.0, 4.0, 8.0])),
        "output": _Buffer(np.full(rows * 4, -99.0)),
        "column": _Buffer(np.full(4, -99.0)),
    }
    namespace = {
        **buffers,
        "rows": rows,
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
        "cutlass": SimpleNamespace(Float32=float, Int32=int),
    }
    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    exec(compile(ast.unparse(module), "<interchanged>", "exec"), namespace)
    return buffers


@pytest.mark.parametrize(
    ("rows", "valid_rows", "valid_cols"), [(3, 3, 4), (3, 2, 3), (0, 0, 4)]
)
def test_overwritten_stores_and_producers_preserve_cpu_results(
    rows: int, valid_rows: int, valid_cols: int
) -> None:
    original_first, _, original_final, _ = _interchanged(_program())
    removed, optimized = _prune(_program())
    assert removed == 1
    before = _execute([original_first, *original_final], rows, valid_rows, valid_cols)
    after = _execute(optimized, rows, valid_rows, valid_cols)
    for name in before:
        np.testing.assert_array_equal(after[name].values, before[name].values)
    assert before["output"].stores == 2 * after["output"].stores
    assert after["output"].stores == valid_rows * valid_cols
    assert before["weight"].loads - after["weight"].loads == rows * 4
    assert before["column"].stores == after["column"].stores == 4
    values = after["source"].values.reshape(rows, 4)
    mask = (np.arange(rows)[:, None] < valid_rows) & (
        np.arange(4)[None, :] < valid_cols
    )
    masked = np.where(mask, values, 0.0)
    product = masked * after["weight"].values
    expected = np.where(mask, product - 2 * product.sum(-1, keepdims=True), -99.0)
    np.testing.assert_array_equal(after["output"].values.reshape(rows, 4), expected)
    np.testing.assert_array_equal(after["column"].values, masked.sum(0))


def test_disjoint_tensor_names_require_compiler_alias_facts() -> None:
    assert _prune(_program(), pairs=set())[0] == 0


def test_rebound_tensor_name_cannot_reuse_argument_alias_fact() -> None:
    loop = _program()
    loop.body.insert(0, ast.parse("source = output").body[0])
    assert _prune(loop)[0] == 0


@pytest.mark.parametrize("source", ["output", "unknown_pointer"])
def test_aliasing_or_unresolved_load_preserves_store(source: str) -> None:
    loop = _program()
    loop.body = (
        ast.parse(ast.unparse(loop).replace("source.iterator", f"{source}.iterator"))
        .body[0]
        .body
    )
    assert _prune(loop)[0] == 0


@pytest.mark.parametrize("replacement", ["index + reduced", "row + col", "index + 1"])
def test_changed_final_address_rejects_overwrite(replacement: str) -> None:
    first, serial, final, candidates = _interchanged(_program())
    last = cast("ast.For", final[-1])
    lane = cast("ast.For", last.body[-1])
    lane.body = (
        ast.parse(
            ast.unparse(lane).replace(
                "output.iterator + index", f"output.iterator + ({replacement})"
            )
        )
        .body[0]
        .body
    )
    assert (
        eliminate_interchanged_stores(first, serial, final, candidates, _PAIRS, set())
        == 0
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("index = row * 4 + col", "index = row * 4 + col + 1"),
        (
            "mask = row < valid_rows and col < valid_cols",
            "mask = row < valid_rows and col < valid_cols - 1",
        ),
        ("col = lane", "col = lane + 1"),
    ],
)
def test_matching_store_text_requires_matching_address_and_mask_definitions(
    before: str, after: str
) -> None:
    first, serial, final, candidates = _interchanged(_program())
    last = cast("ast.For", final[-1])
    lane = cast("ast.For", last.body[-1])
    lane.body = ast.parse(ast.unparse(lane).replace(before, after)).body[0].body
    assert (
        eliminate_interchanged_stores(first, serial, final, candidates, _PAIRS, set())
        == 0
    )


def test_final_pass_load_that_observes_first_store_rejects_elimination() -> None:
    first, serial, final, candidates = _interchanged(_program())
    last = cast("ast.For", final[-1])
    last.body.insert(
        0, ast.parse("observed = (output.iterator + row * 4).load()").body[0]
    )
    assert (
        eliminate_interchanged_stores(first, serial, final, candidates, _PAIRS, set())
        == 0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "mask = reduced > 0",
        "index = int(reduced)",
        "index = cursor\n        cursor = cursor + 1",
        "opaque(output)",
        "if row == 1:\n            break",
        "result = output[index] + result",
    ],
)
def test_unproved_coverage_or_effect_preserves_store(mutation: str) -> None:
    loop = _program()
    source = ast.unparse(loop).replace("if mask:\n", f"{mutation}\n        if mask:\n")
    loop.body = ast.parse(source).body[0].body
    assert _prune(loop)[0] == 0


def test_lane_dependent_serial_range_preserves_store() -> None:
    loop = _program()
    serial = cast("ast.For", loop.body[2])
    serial.iter = ast.parse("range(lane + 1)", mode="eval").body
    assert _prune(loop)[0] == 0


def test_pending_scalar_alias_keeps_producer_binding() -> None:
    removed, body = _prune(_program(), protected={"scaled", "live_alias"})
    assert removed == 1
    first = cast("ast.For", body[0])
    assert "scaled = reduced * 2.0" in ast.unparse(first)


def test_pending_scalar_alias_in_address_preserves_store() -> None:
    assert _prune(_program(), protected={"index", "other_index"})[0] == 0


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
@skipUnlessBackends(["cute"])
def test_norm_backward_emits_one_dx_store(name: str) -> None:
    from examples.layer_norm import layer_norm_bwd
    from examples.rms_norm import rms_norm_bwd

    x = torch.empty(256, 1024, dtype=torch.float16)
    weight = torch.empty(1024, dtype=torch.float16)
    if name == "rms_norm":
        example = rms_norm_bwd
        inputs = (torch.empty_like(x), x, weight, torch.empty(256, 1))
    else:
        example = layer_norm_bwd
        inputs = (
            torch.empty_like(x),
            x,
            torch.empty(256),
            torch.empty(256),
            weight,
            True,
        )
    bound = helion.kernel(
        example.fn,
        backend="cute",
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ).bind(inputs)
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        code = bound.to_code(bound.env.config_spec.default_config())
        with patch(
            "helion._compiler.cute.interchanged_store_dce.eliminate_interchanged_stores",
            return_value=0,
        ):
            original = bound.to_code(bound.env.config_spec.default_config())

    def stores(source: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            and "grad_x.iterator" in ast.unparse(node.func.value)
            for node in ast.walk(ast.parse(source))
        )

    if "fragment_buffer" in code:
        assert name == "layer_norm"
        # The complete fragment owner already emits one write per element;
        # it does not pass through the scalar interchange/DCE rewrite.
        assert stores(original) == stores(code) == 1
        assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(code))
        _assert_fragment_norm_coverage(code, 256, 1024)
        return

    assert stores(original) == 2
    assert stores(code) == 1
    first = next(
        node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.For)
    )
    if name == "rms_norm":
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
            and node.value.id == "weight"
            for node in ast.walk(first)
        )
    assert "grad_weight" in ast.unparse(first)


def _atomic_program(*, row_total: bool = True, extra: str = "") -> ast.For:
    """``_program`` with the reduction added into ``output[row]`` by an atomic
    instead of stored per element; ``row_total=False`` drops the per-lane
    ``column`` accumulator so the reduction is the loop's only effect."""
    init = (
        """
    grad_w = 0.0"""
        if row_total
        else ""
    )
    total = (
        """
        grad_w = grad_w + value"""
        if row_total
        else ""
    )
    suffix = (
        """
    (column.iterator + col).store(grad_w)"""
        if row_total
        else ""
    )
    loop = ast.parse(
        f"""
for lane in range(4):
    col = lane{init}
    for row in range(rows):
        row_mask = row < valid_rows
        mask = row_mask and col < valid_cols
        value = (source.iterator + row * 4 + col).load() if mask else 0.0{total}
        reduced = _helion_lane_reduce(value, 'sum', 0.0, 1, 1, 0, '', 1)
        if row_mask:
            cute.arch.atomic_add((output.iterator + row).llvm_ptr, val=reduced, sem='relaxed'){extra}{suffix}
"""
    ).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, "lane")
    return loop


def _atomics(body: list[ast.AST]) -> list[ast.Call]:
    return [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_add"
    ]


class _AtomicBuffer(_Buffer):
    @property
    def iterator(self) -> _AtomicPointer:
        return _AtomicPointer(self, 0)


class _AtomicPointer(_Pointer):
    def __add__(self, offset: int) -> _AtomicPointer:
        return _AtomicPointer(self.buffer, self.offset + offset)

    @property
    def llvm_ptr(self) -> _AtomicPointer:
        return self

    def atomic_add(self, value: float) -> None:
        self.buffer.stores += 1
        self.buffer.values[self.offset] += value


def _execute_atomics(
    body: list[ast.AST], rows: int, valid_rows: int, valid_cols: int
) -> dict[str, _AtomicBuffer]:
    body = tile_strategy.split_lane_loop_reductions(
        body,
        uniform_names={
            "source",
            "output",
            "column",
            "rows",
            "valid_rows",
            "valid_cols",
        },
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    body = tile_strategy.restore_unprocessed_lane_reduce_markers(body)
    buffers = {
        "source": _AtomicBuffer(np.arange(rows * 4, dtype=np.float64) + 0.5),
        "output": _AtomicBuffer(np.zeros(rows)),
        "column": _AtomicBuffer(np.full(4, -99.0)),
    }
    namespace = {
        **buffers,
        "rows": rows,
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
        "cutlass": SimpleNamespace(Float32=float, Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                atomic_add=lambda pointer, val, sem: pointer.atomic_add(val)
            )
        ),
    }
    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    exec(compile(ast.unparse(module), "<interchanged>", "exec"), namespace)
    return buffers


@pytest.mark.parametrize(
    ("rows", "valid_rows", "valid_cols"), [(3, 3, 4), (3, 2, 3), (0, 0, 4)]
)
def test_an_atomic_consumer_of_the_reduction_is_applied_by_the_final_pass_only(
    rows: int, valid_rows: int, valid_cols: int
) -> None:
    """The first pass keeps its per-lane ``column`` totals; its atomic of the
    reduction's per-lane partial is dropped (a store there is overwritten by
    the final pass, an add would double the total) and the final pass adds
    each row's full reduction once."""
    body = tile_strategy.interchange_lane_outside_serial_reductions([_atomic_program()])
    first, *final = body
    assert isinstance(first, ast.For) and first.target.id == "lane"  # pyrefly: ignore [missing-attribute]
    assert not _atomics([first])
    assert len(_atomics(final)) == 1
    assert "reduced" not in ast.unparse(first)
    assert "grad_w = grad_w + value" in ast.unparse(first)
    buffers = _execute_atomics(body, rows, valid_rows, valid_cols)
    values = buffers["source"].values.reshape(rows, 4)
    mask = (np.arange(rows)[:, None] < valid_rows) & (
        np.arange(4)[None, :] < valid_cols
    )
    masked = np.where(mask, values, 0.0)
    np.testing.assert_array_equal(buffers["output"].values, masked.sum(1))
    assert buffers["output"].stores == valid_rows
    np.testing.assert_array_equal(buffers["column"].values, masked.sum(0))


def test_a_first_pass_whose_only_effect_was_the_atomic_is_dropped() -> None:
    body = tile_strategy.interchange_lane_outside_serial_reductions(
        [_atomic_program(row_total=False)]
    )
    assert len(body) == 1 and isinstance(body[0], ast.For)
    assert body[0].target.id == "row"  # pyrefly: ignore [missing-attribute]
    assert len(_atomics(body)) == 1
    buffers = _execute_atomics(body, 3, 2, 3)
    values = buffers["source"].values.reshape(3, 4)
    expected = np.where(np.arange(4)[None, :] < 3, values, 0.0).sum(1)
    expected[2:] = 0.0
    np.testing.assert_array_equal(buffers["output"].values, expected)


def test_an_atomic_statement_with_other_work_declines_the_interchange() -> None:
    """Dropping the statement from the first pass would drop its other
    effect too, and keeping it would double the total: decline the nest."""
    with pytest.raises(exc.BackendUnsupported, match="atomic consumer"):
        tile_strategy.interchange_lane_outside_serial_reductions(
            [_atomic_program(extra="\n            seen = reduced")]
        )


def _column_sums_added(x: torch.Tensor) -> torch.Tensor:
    b, m = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].sum(dim=0))
    return out


@skipUnlessBackends(["cute"])
def test_column_sums_over_a_lane_looped_row_tile_are_added_once() -> None:
    """Eight rows of a tile on one thread walk a lane loop around the column
    loop; the interchange puts the column loop outside and the kernel holds
    one atomic of the full column sum, with no first pass left."""
    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            helion.kernel(_column_sums_added, backend="cute", autotune_effort="none"),
            (torch.randn(37, 100),),
        )
        config = dict(bound.config_spec.default_config().config)
        config.update(
            block_sizes=[8, 32],
            num_threads=[1, 32],
            cute_vector_widths=[1, 1],
            cute_lane_layouts=["strided", "strided"],
        )
        code = bound.to_code(helion.Config(**config))
    kernel = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert len(_atomics(list(kernel.body))) == 1, code
    assert ast.unparse(kernel).count("for lane_0 in range(8):") == 1, code
    (column_loop,) = [node for node in kernel.body if isinstance(node, ast.For)]
    assert column_loop.target.id == "tile_offset_1"  # pyrefly: ignore [missing-attribute]
