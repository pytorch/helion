"""GPU-free codegen coverage for lane-invariant atomics in CuTe lane loops.

An atomic in a tile body that reads none of a lane loop's coordinates is the
same read-modify-write in every iteration of that loop, so the full nest of
``DeviceGridState.wrap_body`` used to issue it once per lane iteration.  The
lane-loop distribution (``cute/lane_loop_distribution.py``) now places an
atomic like a store: inside exactly the lane loops it depends on, ordered
against the other accesses of its tensor, and a body it cannot place that
way is rejected instead of repeating the atomic.  Per-lane atomics keep their
loops unchanged.

A tile-uniform atomic (a constant, tile-attribute or ``hl.grid`` index and a
scalar value) is issued by the leader thread of every thread axis of the tile
and carries no tile mask (``cute/atomic_ops.py``), so with the placement above
it runs exactly once per tile.

An atomic that has to stay inside the lane loop of a tile axis it is uniform
along (the loop structure nests its own loop inside that one; the loop belongs
to a device loop, whose nest is kept when it is exact) is pinned to the loop's
first lane instead of repeating, inside a user branch when it sits in one,
unless the pin would reorder it against other accesses of its tensor, skip
other work in its statement or run under a condition that varies with the
lane, which rejects the body.  A statement holding several atomics is pinned
when any of them needs it.  A 0-d tensor index (a scalar loaded in the body)
is tile-uniform like a tile attribute, and a ``tile.begin`` access carries no
lane mask, so a partial tile ties neither to a lane loop.
"""

from __future__ import annotations

import ast
import textwrap
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.ast_read_writes import HELION_ATOMIC_UNIFORM_LANES_ATTR
from helion._compiler.cute.lane_loop_distribution import LaneScope
from helion._compiler.cute.lane_loop_distribution import check_full_nest
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _generate(kernel: object, args: tuple[torch.Tensor, ...], **config: object) -> str:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        return bound.to_code(helion.Config.from_dict(config))


def _kernel_function(code: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_"):
            return node
    raise AssertionError(code)


def _is_atomic_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute):
        return node.func.attr.startswith("atomic_")
    return isinstance(node.func, ast.Name) and node.func.id.startswith("_cute_atomic_")


def _parents(function: ast.FunctionDef) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }


def _the_atomic(function: ast.FunctionDef) -> ast.Call:
    (call,) = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _is_atomic_call(node)
    ]
    return call


def _enclosing(
    function: ast.FunctionDef, node: ast.AST, kind: type[ast.AST]
) -> list[ast.AST]:
    """The ``kind`` statements around ``node``, innermost first."""
    parents = _parents(function)
    found: list[ast.AST] = []
    while node in parents:
        node = parents[node]
        if isinstance(node, kind):
            found.append(node)
    return found


def _lane_loops_around(function: ast.FunctionDef, node: ast.AST) -> list[str]:
    return [
        loop.target.id  # pyrefly: ignore [missing-attribute]
        for loop in _enclosing(function, node, ast.For)
    ]


def _guard(function: ast.FunctionDef, call: ast.Call) -> str | None:
    branches = _enclosing(function, call, ast.If)
    return ast.unparse(branches[0].test) if branches else None  # pyrefly: ignore [missing-attribute]


def _first_lanes(guard: str | None) -> set[str]:
    """The lane variables the guard pins to their first iteration."""
    if guard is None:
        return set()
    return {
        node.left.id  # pyrefly: ignore [missing-attribute]
        for node in ast.walk(ast.parse(guard, mode="eval"))
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and ast.unparse(node.comparators[0]) == "0"
    }


def _leader_axes(guard: str | None) -> set[int]:
    if guard is None:
        return set()
    return {
        node.left.slice.value  # pyrefly: ignore [missing-attribute]
        for node in ast.walk(ast.parse(guard, mode="eval"))
        if isinstance(node, ast.Compare)
        and ast.unparse(node.left).startswith("cute.arch.thread_idx()[")
        and ast.unparse(node.comparators[0]) == "0"
    }


def _thread_axes(function: ast.FunctionDef) -> set[int]:
    """The thread axes the kernel reads at all."""
    return {
        node.slice.value  # pyrefly: ignore [missing-attribute]
        for node in ast.walk(function)
        if isinstance(node, ast.Subscript)
        and ast.unparse(node.value) == "cute.arch.thread_idx()"
    }


def _top_level_statement(function: ast.FunctionDef, node: ast.AST) -> ast.stmt:
    parents = _parents(function)
    while parents[node] is not function:
        node = parents[node]
    return node  # pyrefly: ignore [bad-return]


def _first_lane_loop_index(function: ast.FunctionDef) -> int:
    return min(
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.For)
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_then_copy(x: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(counter, [0], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_then_copy_3d(x: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1, tile2 in hl.tile(x.shape):
        hl.atomic_add(counter, [0], 1)
        out[tile0, tile1, tile2] = x[tile0, tile1, tile2]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_count_then_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(counts, [tile0], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_count_then_copy_3d(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1, tile2 in hl.tile(x.shape):
        hl.atomic_add(counts, [tile0, tile1], 1)
        out[tile0, tile1, tile2] = x[tile0, tile1, tile2]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _begin_count_then_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(counts, [tile0.begin], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _col_count_then_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(counts, [tile1], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flag_col_count_copy(
    x: torch.Tensor, counts: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        flag[0] = 1.0
        hl.atomic_add(counts, [tile1], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_element_then_copy(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(total, [0], x[tile0.begin, tile1.begin])
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_then_copy(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(total, [0], x[tile0, tile1.begin])
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_row_then_copy(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(total, [0], x[tile0.begin, tile1][None, :])
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_increment_then_read(out: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(out.shape):
        hl.atomic_add(out, [1, tile1], 1.0)
        out2[tile0, tile1] = out[tile0, tile1]
    return out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_row_increment(out: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(out.shape):
        v = out[tile0, tile1]
        hl.atomic_add(out, [1, tile1], 1.0)
        out2[tile0, tile1] = v
    return out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_row_increment(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(out, [1, tile1], 1.0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_acq_rel_copy(
    x: torch.Tensor, y: torch.Tensor, counter: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out2 = torch.empty_like(y)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(counter, [0], 1, sem="acq_rel")
        out2[tile0, tile1] = y[tile0, tile1]
    return out, out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _max_then_copy(x: torch.Tensor, best: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_max(best, [0], 3.0)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_release_count(x: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(counter, [0], 1, sem="release")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_release_rows(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(counts, [tile0], 1, sem="release")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _acquire_rows_then_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(counts, [tile0], 1, sem="acquire")
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _acquire_copy_release_rows(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(counts, [tile0], 1, sem="acquire")
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(counts, [tile0], 1, sem="release")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_release_elements(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(flags, [tile0, tile1], 1, sem="release")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _acquire_elements_then_copy(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(flags, [tile0, tile1], 1, sem="acquire")
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_flagged_release_elements(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        if flags[0, 0] >= 0:
            hl.atomic_add(flags, [tile0, tile1], 1, sem="release")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reduced_update_by_aranges(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b, m, n = x.shape
    n = hl.specialize(n)
    k = hl.specialize(w.size(0))
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            update = w[:, :] * x[tile_b, tile_m, :].sum(0).sum(0)[None, :]
            hl.atomic_add(out, [hl.arange(0, k), hl.arange(0, n)], update)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reduced_update_by_slices(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b, m, n = x.shape
    n = hl.specialize(n)
    k = hl.specialize(w.size(0))
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            update = w[:, :] * x[tile_b, tile_m, :].sum(0).sum(0)[None, :]
            hl.atomic_add(out, [slice(0, k), slice(0, n)], update)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reduced_update_rereading_its_target(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b, m, n = x.shape
    n = hl.specialize(n)
    k = hl.specialize(w.size(0))
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            total = x[tile_b, tile_m, :].sum(0).sum(0)
            update = w[:, :] * total[None, :] + out[:, :] * 0
            hl.atomic_add(out, [hl.arange(0, k), hl.arange(0, n)], update)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_into_scalar(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(total, [0], x[tile0, tile1])
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _per_lane_atomic(x: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        hl.atomic_add(acc, [tile0, tile1], x[tile0, tile1])
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_count_copy_back(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(x)
    out2 = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        hl.atomic_add(out, [0, 0], 1.0)
        out2[tile0, tile1] = out[tile0, tile1]
    return out, out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_then_offset_the_copy(x: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        old = hl.atomic_add(counter, [0], 1)
        out[tile0, tile1] = x[tile0, tile1] + old
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flagged_count_then_offset_the_copy(
    x: torch.Tensor, counter: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        if flag[0] > 0:
            old = hl.atomic_add(counter, [0], 1)
            out[tile0, tile1] = x[tile0, tile1] + old
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loaded_index_count_then_copy(
    x: torch.Tensor, idx: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        i = idx[tile0.begin]
        hl.atomic_add(counts, [i], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_in_inner_tile(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            hl.atomic_add(counts, [tile0], 1)
            out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _guarded_col_count_then_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        if tile0.begin == 0:
            hl.atomic_add(counts, [tile1], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _guarded_col_count_and_flag_then_copy(
    x: torch.Tensor, counts: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        if tile0.begin == 0:
            hl.atomic_add(counts, [tile1], 1)
            flag[0] = 1.0
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flagged_col_count_then_copy(
    x: torch.Tensor, flags: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        if flags[tile0.begin] > 0:
            hl.atomic_add(counts, [tile1], 1)
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_and_per_lane_add_in_inner_tile(
    x: torch.Tensor, y: torch.Tensor, counter: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        for tile2 in hl.tile(y.size(0)):
            hl.atomic_add(counter, [0], 1)
            hl.atomic_add(out, [tile0, tile1], y[tile2.begin] + 0 * x[tile0, tile1])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_and_copy_first_column_in_inner_tile(
    x: torch.Tensor, counts: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            hl.atomic_add(counts, [tile0], 1)
            out[tile0, tile1.begin] = x[tile0, tile1.begin]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _convert_bytes_and_count(
    packed: torch.Tensor, counter: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        hl.atomic_add(counter, [0], 1)
        out[tile0, tile1] = y
    return out


# One row per program, 32 threads x 8 lanes: a plain lane loop.
_LANES = {"block_sizes": [1, 256], "num_threads": [0, 32], "cute_vector_widths": [1, 1]}
# One thread per element of a 4 x 32 tile: no lane loops.
_ONE_PER_THREAD = {
    "block_sizes": [4, 32],
    "num_threads": [4, 32],
    "cute_vector_widths": [1, 1],
}
# 256 threads x one vector of four: the vectorized lane loop.
_ROW = {"block_sizes": [1, 1024], "num_threads": [0, 256], "cute_vector_widths": [1, 4]}
# Four rows per thread around the vectorized lane loop: two lane loops.
_NESTED = {
    "block_sizes": [4, 256],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 4],
}
# Four rows per thread around a plain column lane loop: two scalar lane loops.
_NESTED_SCALAR = {
    "block_sizes": [4, 256],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 1],
}
# A thread-owned leading axis, a plain lane loop and the vectorized lane loop.
_NESTED_3D = {
    "block_sizes": [2, 4, 256],
    "num_threads": [2, 1, 64],
    "cute_vector_widths": [1, 1, 4],
}
# ``_NESTED`` around a serial inner device loop of eight.
_NESTED_INNER_SERIAL = {
    "block_sizes": [4, 256, 8],
    "num_threads": [1, 64, 0],
    "cute_vector_widths": [1, 4, 1],
}
# The flattened per-thread strategy: one lane loop over both tile axes.
_FLAT = {
    "block_sizes": [1, 2048],
    "num_threads": [1, 256],
    "cute_vector_widths": [1, 1],
    "flatten_loops": [True],
}
_UNIFORM_CASES = [
    pytest.param(_count_then_copy, _LANES, (8, 256), id="lanes"),
    pytest.param(_count_then_copy, _LANES, (8, 250), id="lanes-masked"),
    pytest.param(_count_then_copy, _ROW, (8, 1024), id="vector_lane"),
    pytest.param(_count_then_copy, _NESTED, (8, 256), id="nested"),
    pytest.param(_count_then_copy_3d, _NESTED_3D, (4, 8, 256), id="nested_3d"),
    pytest.param(_count_then_copy, _FLAT, (8, 2048), id="flattened"),
]


def _counter_args(kernel: object, shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    return (torch.empty(shape), torch.zeros((1,), dtype=torch.int32))


@pytest.mark.parametrize(("kernel", "config", "shape"), _UNIFORM_CASES)
def test_tile_uniform_atomic_runs_once_per_tile(
    kernel: object, config: dict[str, object], shape: tuple[int, ...]
) -> None:
    code = _generate(kernel, _counter_args(kernel, shape), **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    # Outside every lane loop, ahead of the first one ...
    assert not _lane_loops_around(function, atomic), code
    statement = _top_level_statement(function, atomic)
    assert function.body.index(statement) < _first_lane_loop_index(function), code
    # ... issued by the leader of every thread axis, without a tile mask.
    guard = _guard(function, atomic)
    assert _leader_axes(guard) == _thread_axes(function), code
    assert "mask" not in (guard or ""), code
    if shape[-1] % 256:
        assert "mask_1" in ast.unparse(function.body[_first_lane_loop_index(function)])


_PARTIAL_CASES = [
    # ``counts[tile0]`` follows the row lane loop, not the column loop.
    pytest.param(_row_count_then_copy, _NESTED, (8, 256), ["lane_0"], {0}, id="rows"),
    # Rows on a thread axis, the middle axis on a lane loop: the atomic sits
    # in that loop and is issued by the column axis's leader thread.
    pytest.param(
        _row_count_then_copy_3d,
        _NESTED_3D,
        (4, 8, 256),
        ["lane_1"],
        {1},
        id="rows_3d",
    ),
    # A tile attribute reads no lane coordinate at all.
    pytest.param(_begin_count_then_copy, _NESTED, (8, 256), [], {0}, id="tile_begin"),
]


def _counts_args(kernel: object, shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    dims = 2 if kernel is _row_count_then_copy_3d else 1
    return (torch.empty(shape), torch.zeros(shape[:dims]))


@pytest.mark.parametrize(
    ("kernel", "config", "shape", "loops", "leaders"), _PARTIAL_CASES
)
def test_partially_indexed_atomic_leaves_the_lanes_it_does_not_read(
    kernel: object,
    config: dict[str, object],
    shape: tuple[int, ...],
    loops: list[str],
    leaders: set[int],
) -> None:
    code = _generate(kernel, _counts_args(kernel, shape), **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == loops, code
    assert _leader_axes(_guard(function, atomic)) == leaders, code


def test_partially_indexed_atomic_keeps_only_the_masks_of_its_axes() -> None:
    # Six rows in tiles of four and 250 columns in a tile of 256: the row
    # count is masked by its own axis but not by the column mask, which
    # would tie it to the column lane loop.
    code = _generate(
        _row_count_then_copy, _counts_args(_row_count_then_copy, (6, 250)), **_NESTED
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    guard = _guard(function, atomic)
    assert guard is not None and "mask_0" in guard and "mask_1" not in guard, code
    assert _lane_loops_around(function, atomic) == ["lane_0"], code
    assert "mask_1" in ast.unparse(function), code


@pytest.mark.parametrize("kernel", [_sum_into_scalar, _per_lane_atomic])
def test_per_element_atomics_keep_their_lane_loop(kernel: object) -> None:
    # The update value is one element per lane: the atomic varies with the
    # lane loop and no thread is collapsed to a leader.
    args = (
        torch.empty((8, 256)),
        torch.zeros((1,) if kernel is _sum_into_scalar else (8, 256)),
    )
    code = _generate(kernel, args, **_LANES)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["lane_1"], code
    assert _guard(function, atomic) is None, code


def test_release_atomic_follows_the_stores_it_orders() -> None:
    # A release atomic keeps every memory access on its side: it leaves the
    # loop after the copy, once per tile, with its ordering intact.
    code = _generate(_copy_then_release_count, _counter_args(None, (8, 256)), **_LANES)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert not _lane_loops_around(function, atomic), code
    statement = _top_level_statement(function, atomic)
    assert function.body.index(statement) > _first_lane_loop_index(function), code
    assert _leader_axes(_guard(function, atomic)) == {0}, code
    assert "sem='release'" in ast.unparse(atomic)


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        pytest.param(
            _copy_then_release_rows, ["lane_1", "barrier", "release"], id="release"
        ),
        pytest.param(
            _acquire_rows_then_copy, ["acquire", "barrier", "lane_1"], id="acquire"
        ),
        pytest.param(
            _acquire_copy_release_rows,
            ["acquire", "barrier", "lane_1", "barrier", "release"],
            id="acquire_release",
        ),
    ],
)
def test_row_fence_barrier_joins_its_atomic_in_the_row_loop(
    kernel: object, expected: list[str]
) -> None:
    # A release / acquire count per row runs in the row loop, issued by the
    # column axis's leader thread.  Its barrier runs there too, once per row,
    # between the column loop and the atomic: after the row's per-lane stores
    # for a release, before its per-lane loads for an acquire.
    args = (torch.empty((8, 256)), torch.zeros((8,), dtype=torch.int32))
    code = _generate(kernel, args, **_NESTED_SCALAR)
    function = _kernel_function(code)
    row_loop = next(node for node in function.body if isinstance(node, ast.For))
    assert ast.unparse(row_loop.target) == "lane_0", code

    def role(statement: ast.stmt) -> str | None:
        if isinstance(statement, ast.For):
            return ast.unparse(statement.target)
        source = ast.unparse(statement)
        if source == "cute.arch.sync_threads()":
            return "barrier"
        return next((sem for sem in ("acquire", "release") if sem in source), None)

    roles = [role(statement) for statement in row_loop.body]
    assert [r for r in roles if r is not None] == expected, code
    barriers = ast.unparse(function).count("cute.arch.sync_threads()")
    assert barriers == expected.count("barrier"), code


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        pytest.param(
            _copy_then_release_elements,
            ["load", "store", "barrier", "release"],
            id="release",
        ),
        pytest.param(
            _acquire_elements_then_copy,
            ["acquire", "barrier", "load", "store"],
            id="acquire",
        ),
    ],
)
def test_per_element_fence_gets_the_block_barrier(
    kernel: object, expected: list[str]
) -> None:
    # Each thread issues the release / acquire of its own elements, which
    # orders only its own accesses, while Helion's memory orders act for the
    # whole program (the Triton backend's bar.sync before every release and
    # after every acquire).  A block barrier sits between the copy and the
    # atomic in the column loop, every thread reaching it once per lane.
    args = (torch.empty((8, 256)), torch.zeros((8, 256), dtype=torch.int32))
    code = _generate(kernel, args, **_NESTED_SCALAR)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["lane_1", "lane_0"], code
    assert _guard(function, atomic) is None, code
    column_loop = _enclosing(function, atomic, ast.For)[0]
    assert isinstance(column_loop, ast.For)

    def role(statement: ast.stmt) -> str | None:
        source = ast.unparse(statement)
        if source == "cute.arch.sync_threads()":
            return "barrier"
        for name in ("acquire", "release", ".load()", ".store("):
            if name in source:
                return name.strip(".()")
        return None

    roles = [role(statement) for statement in column_loop.body]
    assert [r for r in roles if r is not None] == expected, code
    assert ast.unparse(function).count("cute.arch.sync_threads()") == 1, code


def test_per_element_fence_in_a_divergent_branch_rejects_the_config() -> None:
    # The branch reads the tensor the atomic writes, so threads may disagree
    # on it and some would skip the barrier its release needs.
    args = (torch.empty((8, 256)), torch.zeros((8, 256), dtype=torch.int32))
    with pytest.raises(exc.BackendUnsupported, match="inside a branch or while loop"):
        _generate(_copy_then_flagged_release_elements, args, **_NESTED_SCALAR)


def _reduced_update_args() -> tuple[torch.Tensor, ...]:
    return (torch.empty((4, 40, 6)), torch.empty((3, 6)), torch.zeros((3, 6)))


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(_reduced_update_by_aranges, id="aranges"),
        pytest.param(_reduced_update_by_slices, id="slices"),
    ],
)
def test_atomic_after_a_lane_looped_reduction_runs_at_the_first_lane(
    kernel: object,
) -> None:
    # Two rows per program on a thread axis and 16 rows of m per device-loop
    # tile on one lane loop: the [k, n] update reads the sum that lane loop's
    # reduction finalizes, and the reduction split re-runs the loop around
    # the atomic.  The atomic covers neither axis, so it is pinned to the
    # loop's first lane (once per lane would add the sum 16 times).
    code = _generate(kernel, _reduced_update_args(), block_sizes=[2, 16])
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic)[0] == "lane_1", code
    guard = _guard(function, atomic)
    assert _first_lanes(guard) == {"lane_1"}, code
    assert _leader_axes(guard) == {2}, code


def test_atomic_after_a_lane_looped_reduction_beside_its_target_rejects_the_config() -> (
    None
):
    # The update also reads ``out``: issued at the first lane only, the atomic
    # would run before the other lanes' loads of it, and the other threads'
    # loads race with it in a body the reduction split rewrites, where no
    # barrier can order them.
    with pytest.raises(
        exc.BackendUnsupported,
        match="race on a tensor in a body a later pass rewrites",
    ):
        _generate(
            _reduced_update_rereading_its_target,
            _reduced_update_args(),
            block_sizes=[2, 16],
        )


def test_float_max_helper_is_a_tile_uniform_atomic() -> None:
    code = _generate(
        _max_then_copy, (torch.empty((8, 256)), torch.zeros((1,))), **_LANES
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert ast.unparse(atomic.func) == "_cute_atomic_max_float32", code
    assert not _lane_loops_around(function, atomic), code
    assert _leader_axes(_guard(function, atomic)) == {0}, code


@pytest.mark.parametrize("packet_flush", [False, True], ids=["values", "packet"])
def test_tile_uniform_atomic_precedes_the_byte_conversion_loops(
    packet_flush: bool,
) -> None:
    args = (
        torch.empty((8, 256), dtype=torch.int8),
        torch.zeros((1,), dtype=torch.int32),
    )
    code = _generate(
        _convert_bytes_and_count,
        args,
        block_sizes=[1, 256],
        num_threads=[0, 32],
        cute_vector_widths=[1, 4],
        cute_signed_bitfield_bf16=packet_flush,
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert not _lane_loops_around(function, atomic), code
    statement = _top_level_statement(function, atomic)
    assert function.body.index(statement) < _first_lane_loop_index(function), code
    assert _leader_axes(_guard(function, atomic)) == {0}, code


def test_lane_invariant_atomic_between_accesses_of_its_tensor_rejects_the_config() -> (
    None
):
    # The atomic must stay after the per-lane store to ``out`` and before the
    # per-lane load of it; the nest would repeat it per lane instead.
    with pytest.raises(exc.BackendUnsupported, match="lane-invariant atomic on out"):
        _generate(_copy_count_copy_back, (torch.empty((8, 256)),), **_LANES)


def test_atomic_result_needed_by_other_threads_rejects_the_config() -> None:
    # Only the leader thread performs the atomic and holds the previous value.
    with pytest.raises(exc.BackendUnsupported, match="leader thread"):
        _generate(_count_then_offset_the_copy, _counter_args(None, (8, 256)), **_LANES)


def test_atomic_result_is_broadcast_from_the_leader_thread() -> None:
    # Without lane loops the leader shares its old value through shared
    # memory: write, barrier, every thread reads, barrier before a reuse.
    code = _generate(
        _count_then_offset_the_copy, _counter_args(None, (8, 64)), **_ONE_PER_THREAD
    )
    function = _kernel_function(code)
    text = ast.unparse(function)
    after = text[text.index("cute.arch.atomic_add(") :]
    assert "_atomic_prev_smem" in after, code
    assert after.count("cute.arch.sync_threads()") == 2, code
    assert _leader_axes(_guard(function, _the_atomic(function))) == {0, 1}, code


def test_atomic_result_in_a_branch_rejects_the_config() -> None:
    # The broadcast's barriers cannot sit where a thread may skip them: the
    # flag aliases the counter the atomic writes, so the threads may read it
    # before and after another CTA's increment.
    counter = torch.zeros(1, dtype=torch.int32)
    args = (torch.empty((8, 64)), counter, counter.view(torch.float32))
    with pytest.raises(exc.BackendUnsupported, match="inside a branch"):
        _generate(_flagged_count_then_offset_the_copy, args, **_ONE_PER_THREAD)


def test_atomic_result_in_an_unwritten_flag_branch_is_broadcast() -> None:
    # Nothing writes the flag in this launch (the bound kernel's storage
    # disjointness fact keeps it apart from the counter), so every thread
    # takes the branch alike and the broadcast's barriers sit inside it.
    args = (torch.empty((8, 64)), torch.zeros(1, dtype=torch.int32), torch.ones(1))
    code = _generate(_flagged_count_then_offset_the_copy, args, **_ONE_PER_THREAD)
    function = _kernel_function(code)
    (branch,) = (
        node
        for node in function.body
        if isinstance(node, ast.If) and "atomic_add" in ast.unparse(node)
    )
    body = ast.unparse(branch.body)
    assert "_atomic_prev_smem" in body, code
    assert body.count("cute.arch.sync_threads()") == 2, code


# The lane loops around the atomic (innermost first) and the ones it varies
# along; every other enclosing loop must pin it to its first lane.
_PINNED_CASES = [
    # The value's packet belongs to the column loop, which the copy's packet
    # nests inside the row loop.
    pytest.param(
        _first_row_then_copy,
        _NESTED,
        (8, 256),
        ["vec_lane_1", "lane_1", "lane_0"],
        {"vec_lane_1", "lane_1"},
        id="row_packet",
    ),
    # A literal-valued per-column count nested the same way, on a full tile
    # (every statement is placed in every loop) ...
    pytest.param(
        _col_count_then_copy,
        _NESTED,
        (8, 256),
        ["vec_lane_1", "lane_1", "lane_0"],
        {"vec_lane_1", "lane_1"},
        id="column_count",
    ),
    # ... on a partial tile (the column loop would need two instances) ...
    pytest.param(
        _col_count_then_copy,
        _NESTED,
        (6, 250),
        ["vec_lane_1", "lane_1", "lane_0"],
        {"vec_lane_1", "lane_1"},
        id="column_count-masked",
    ),
    # ... and next to a statement that leaves the loops (the placement is
    # emitted, not the original nest).
    pytest.param(
        _flag_col_count_copy,
        _NESTED,
        (8, 256),
        ["vec_lane_1", "lane_1", "lane_0"],
        {"vec_lane_1", "lane_1"},
        id="column_count-placed",
    ),
]


def _pinned_args(kernel: object, shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    x = torch.empty(shape)
    if kernel is _flag_col_count_copy:
        return (x, torch.zeros(shape[1]), torch.zeros(1))
    if kernel is _col_count_then_copy:
        return (x, torch.zeros(shape[1]))
    return (x, torch.zeros(1))


@pytest.mark.parametrize(
    ("kernel", "config", "shape", "loops", "varies"), _PINNED_CASES
)
def test_uniform_atomic_kept_in_a_lane_loop_is_pinned_to_its_first_lane(
    kernel: object,
    config: dict[str, object],
    shape: tuple[int, ...],
    loops: list[str],
    varies: set[str],
) -> None:
    code = _generate(kernel, _pinned_args(kernel, shape), **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == loops, code
    assert _first_lanes(_guard(function, atomic)) == set(loops) - varies, code


@pytest.mark.parametrize("shape", [(8, 256), (8, 250)], ids=["full", "partial"])
def test_uniform_atomic_that_leaves_the_lane_loop_is_not_pinned(
    shape: tuple[int, int],
) -> None:
    # The value's load reads no lane coordinate, and as a ``tile.begin``
    # access no lane mask on the partial tile either: the atomic leaves the
    # loop, once per tile, and no lane pin is needed.
    code = _generate(
        _first_element_then_copy, (torch.empty(shape), torch.zeros(1)), **_LANES
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert not _lane_loops_around(function, atomic), code
    guard = _guard(function, atomic)
    assert _first_lanes(guard) == set() and _leader_axes(guard) == {0}, code
    assert "mask" not in ast.unparse(_top_level_statement(function, atomic)), code


def test_per_row_value_keeps_the_row_loop_only_on_a_partial_tile() -> None:
    # The value ``x[tile0, tile1.begin]`` reads the row lane and its mask but
    # nothing of the column loop: the atomic varies with the row loop and
    # sits in it, outside the column loop, issued by that axis's leader.
    code = _generate(
        _first_column_then_copy, (torch.empty((6, 250)), torch.zeros(1)), **_NESTED
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["lane_0"], code
    guard = _guard(function, atomic)
    assert _first_lanes(guard) == set() and _leader_axes(guard) == {0}, code
    assert guard is not None and "mask_0" in guard and "mask_1" not in guard, code


def test_per_element_atomic_keeps_the_lanes_its_packet_reads() -> None:
    # The value's packet is addressed by the row lane: the atomic varies with
    # both loops and stays unpinned inside them.
    code = _generate(
        _sum_into_scalar, (torch.empty((8, 256)), torch.zeros(1)), **_NESTED
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["vec_lane_1", "lane_1", "lane_0"]
    assert _guard(function, atomic) is None, code


def test_pinned_atomic_precedes_the_per_lane_loads_of_its_tensor() -> None:
    # Issued at the first row lane, the increment precedes every row's load
    # of the same tensor, as it does in the program.
    args = (torch.empty((4, 256)), torch.empty((4, 256)))
    code = _generate(_row_increment_then_read, args, **_NESTED_SCALAR)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["lane_1", "lane_0"], code
    assert _first_lanes(_guard(function, atomic)) == {"lane_0"}, code
    (inner,) = [
        loop
        for loop in ast.walk(function)
        if isinstance(loop, ast.For) and ast.unparse(loop.target) == "lane_1"
    ]
    source = [ast.unparse(statement) for statement in inner.body]
    assert [i for i, s in enumerate(source) if "atomic_add" in s] < [
        i for i, s in enumerate(source) if "out.iterator" in s and ".load()" in s
    ], code


def test_pinned_atomic_after_a_per_lane_access_of_its_tensor_rejects_the_config() -> (
    None
):
    # The later rows' loads (stores) would follow an increment the program
    # issues after (before) all of them.
    args = (torch.empty((4, 256)), torch.empty((4, 256)))
    with pytest.raises(
        exc.BackendUnsupported,
        match="lane-invariant atomic on out issued at the first lane_0 would "
        "follow a per-lane load of it",
    ):
        _generate(_read_then_row_increment, args, **_NESTED_SCALAR)
    with pytest.raises(
        exc.BackendUnsupported,
        match="lane-invariant atomic on out issued at the first lane_0 would "
        "follow a per-lane store to it",
    ):
        _generate(_copy_then_row_increment, args, **_NESTED_SCALAR)


def test_pinned_atomic_beside_a_hoisted_access_of_its_tensor_rejects_the_config() -> (
    None
):
    # The vectorized column loop hoists its loads of ``out`` above the V-loop
    # and flushes its stores after it, beside the pinned increment in the
    # first row iteration.
    args = (torch.empty((4, 256)), torch.empty((4, 256)))
    with pytest.raises(exc.BackendUnsupported, match="hoisted load of it"):
        _generate(_read_then_row_increment, args, **_NESTED)
    with pytest.raises(exc.BackendUnsupported, match="hoisted store to it"):
        _generate(_copy_then_row_increment, args, **_NESTED)


def test_pinned_atomic_keeps_the_later_vectorized_read_of_its_tensor_scalar() -> None:
    # The increment precedes the read of ``out`` in the program, so the
    # memory-effect gate of the tile-vector hoist keeps the read a scalar load
    # inside the V-loop instead of a packet hoisted above it: issued at the
    # first row lane, the pinned increment precedes every row's read.
    args = (torch.empty((4, 256)), torch.empty((4, 256)))
    code = _generate(_row_increment_then_read, args, **_NESTED)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["vec_lane_1", "lane_1", "lane_0"]
    assert _first_lanes(_guard(function, atomic)) == {"lane_0"}, code
    assert "cute.arch.load(out.iterator" not in code, code
    (vloop,) = [
        loop
        for loop in ast.walk(function)
        if isinstance(loop, ast.For) and ast.unparse(loop.target) == "vec_lane_1"
    ]
    source = [ast.unparse(statement) for statement in vloop.body]
    assert [i for i, s in enumerate(source) if "atomic_add" in s] < [
        i for i, s in enumerate(source) if "out.iterator" in s and ".load()" in s
    ], code


def test_fence_between_per_lane_stores_is_not_pinned() -> None:
    # An acquire-release counter between the two copies keeps every store on
    # its side; issued at the first lane it would leave the other lanes'
    # stores unordered, so the body is rejected rather than pinned.
    args = (torch.empty((8, 256)), torch.empty((8, 256)), torch.zeros(1))
    with pytest.raises(
        exc.BackendUnsupported,
        match="lane-invariant atomic on counter issued at the first lane_1 would "
        "not order the other lanes' accesses",
    ):
        _generate(_copy_acq_rel_copy, args, **_LANES)


def _loaded_index_args(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(shape),
        torch.zeros(shape[0], dtype=torch.int32),
        torch.zeros(shape[0]),
    )


@pytest.mark.parametrize(
    ("config", "shape", "pinned"),
    [
        pytest.param(_LANES, (8, 256), [], id="lanes"),
        pytest.param(_LANES, (8, 250), [], id="lanes-masked"),
        pytest.param(_NESTED, (8, 256), [], id="nested"),
        # The index's load is a ``tile.begin`` access: no mask ties it to the
        # row loop on the partial tile either.
        pytest.param(_NESTED, (6, 250), [], id="nested-masked"),
    ],
)
def test_loaded_scalar_index_is_tile_uniform(
    config: dict[str, object], shape: tuple[int, int], pinned: list[str]
) -> None:
    # A 0-d tensor index (a scalar loaded in the body) addresses one element
    # for the whole tile, like a tile attribute: the leader thread issues the
    # atomic once, outside the lane loops or pinned to the first lane of the
    # loop it cannot leave, and no tile mask guards it.
    code = _generate(_loaded_index_count_then_copy, _loaded_index_args(shape), **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == pinned, code
    guard = _guard(function, atomic)
    assert _leader_axes(guard) == _thread_axes(function), code
    assert _first_lanes(guard) == set(pinned), code
    assert "mask" not in (guard or ""), code


@pytest.mark.parametrize(
    ("config", "shape", "loops", "pinned"),
    [
        pytest.param(
            _LANES, (8, 512), ["lane_1", "tile_offset_1"], {"lane_1"}, id="lanes"
        ),
        pytest.param(
            _LANES, (8, 500), ["lane_1", "tile_offset_1"], {"lane_1"}, id="lanes-masked"
        ),
        pytest.param(
            _NESTED,
            (8, 512),
            ["vec_lane_1", "lane_1", "tile_offset_1", "lane_0"],
            {"vec_lane_1", "lane_1"},
            id="nested",
        ),
        pytest.param(
            _NESTED,
            (6, 500),
            ["vec_lane_1", "lane_1", "tile_offset_1", "lane_0"],
            {"vec_lane_1", "lane_1"},
            id="nested-masked",
        ),
    ],
)
def test_uniform_atomic_in_an_inner_loops_lane_nest_is_pinned(
    config: dict[str, object],
    shape: tuple[int, int],
    loops: list[str],
    pinned: set[str],
) -> None:
    # A device loop's lane loops are built around its body before the body
    # exists and are kept when exact; the full-nest check pins the row
    # count, uniform along the column lane, to that loop's first lane (and
    # first vector lane) instead of repeating it once per element.
    args = (torch.empty(shape), torch.zeros(shape[0]))
    code = _generate(_count_in_inner_tile, args, **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == loops, code
    assert _first_lanes(_guard(function, atomic)) == pinned, code


def test_atomic_in_a_user_branch_is_pinned_inside_the_branch() -> None:
    # The codegen's mask guard nests inside the user's branch; the pin joins
    # that innermost guard and the branch itself is untouched.
    args = (torch.empty((6, 250)), torch.zeros(250))
    code = _generate(_guarded_col_count_then_copy, args, **_NESTED)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    inner, outer = _enclosing(function, atomic, ast.If)
    assert _first_lanes(ast.unparse(inner.test)) == {"lane_0"}, code  # pyrefly: ignore [missing-attribute]
    assert ast.unparse(outer.test) == "eq", code  # pyrefly: ignore [missing-attribute]


def test_atomic_sharing_its_branch_with_other_work_rejects_the_config() -> None:
    # Pinned to the first row lane, the branch would skip the flag store in
    # the other lanes along with the atomic.
    args = (torch.empty((6, 250)), torch.zeros(250), torch.zeros(1))
    with pytest.raises(
        exc.BackendUnsupported,
        match="lane-invariant atomic on counts issued at the first lane_0 would "
        "skip the rest of its statement",
    ):
        _generate(_guarded_col_count_and_flag_then_copy, args, **_NESTED)


def test_uniform_atomic_beside_a_per_lane_atomic_in_one_statement_rejects_the_config() -> (
    None
):
    # The inner device loop is one statement to the grid's lane loops.  Its
    # per-lane atomic makes the statement vary with both loops, so nothing is
    # repeated; but the counter is uniform along both, and issuing it at the
    # first lanes only would skip the per-lane atomic in the other lanes.
    args = (
        torch.empty((8, 256)),
        torch.empty(8),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros((8, 256)),
    )
    with pytest.raises(
        exc.BackendUnsupported,
        match="lane-invariant atomic on counter issued at the first lane_0, lane_1 "
        "would skip the rest of its statement",
    ):
        _generate(_count_and_per_lane_add_in_inner_tile, args, **_NESTED_INNER_SERIAL)


@pytest.mark.parametrize("shape", [(8, 512), (8, 500)], ids=["full", "partial"])
def test_dead_inner_lane_loop_is_left_alone(shape: tuple[int, int]) -> None:
    # The inner body indexes its tile only by ``begin``: nothing reads the
    # column lane, the dead-code elimination splices its loop away and the
    # loop repeats nothing.  A pin reading the lane variable would keep the
    # loop alive and run the rest of the body once per lane.
    args = (torch.empty(shape), torch.zeros(shape[0]), torch.empty(shape))
    code = _generate(_count_and_copy_first_column_in_inner_tile, args, **_LANES)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    assert _lane_loops_around(function, atomic) == ["tile_offset_1"], code
    guard = _guard(function, atomic)
    assert _first_lanes(guard) == set() and _leader_axes(guard) == {0}, code
    assert "lane_1" not in code, code


_COLUMN_COUNT = (
    "cute.arch.atomic_add((counts.iterator + cute.crd2idx((indices_1,), "
    "counts.layout)).llvm_ptr, val=1, sem='relaxed')"
)
_PER_LANE_COPY = (
    "(out.iterator + indices_0 * s0 + indices_1 * s1).store("
    "(x.iterator + indices_0 * s0 + indices_1 * s1).load())"
)


def _two_scalar_lane_scopes() -> list[LaneScope]:
    """Two plain lane loops over a partial tile of 6 x 250."""
    return [
        LaneScope(
            f"lane_{axis}",
            frozenset({f"lane_{axis}", f"indices_{axis}", f"mask_{axis}"}),
            frozenset(),
            (),
            frozenset({f"lane_{axis}"}),
            tuple(
                ast.parse(
                    f"indices_{axis} = tile_offset_{axis} + cutlass.Int32(lane_{axis})\n"
                    f"mask_{axis} = indices_{axis} < {size}"
                ).body
            ),
            f"lane_{axis} == 0",
            frozenset({f"mask_{axis}"}),
        )
        for axis, size in enumerate((6, 250))
    ]


def _body_with_a_row_uniform_atomic(source: str) -> list[ast.AST]:
    body: list[ast.AST] = list(ast.parse(textwrap.dedent(source)).body)
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_atomic_call(node):
                setattr(node, HELION_ATOMIC_UNIFORM_LANES_ATTR, frozenset({"lane_0"}))
    return body


@pytest.mark.parametrize(
    ("condition", "accepted"),
    [("flag", True), ("indices_0 > 3", False)],
    ids=["uniform", "per_lane"],
)
def test_pin_inside_a_branch_needs_a_lane_invariant_condition(
    condition: str, accepted: bool
) -> None:
    # Rule level: a column count uniform along the row lane, in a user branch
    # and the codegen's mask guard, beside a per-lane copy.  Pinned to the
    # first row lane it runs under that lane's condition, which is the
    # program only when the condition does not vary with the lane.
    body = _body_with_a_row_uniform_atomic(
        f"if {condition}:\n    if mask_1:\n        {_COLUMN_COUNT}\n{_PER_LANE_COPY}"
    )
    if not accepted:
        with pytest.raises(
            exc.BackendUnsupported,
            match="sits in a statement whose values vary with lane_0",
        ):
            check_full_nest(body, _two_scalar_lane_scopes(), rename_groups={})
        return
    check_full_nest(body, _two_scalar_lane_scopes(), rename_groups={})
    pinned = ast.unparse(body[0])
    assert pinned.startswith(f"if {condition}:\n    if mask_1 and lane_0 == 0:"), pinned


@pytest.mark.parametrize("config", [_NESTED, _NESTED_SCALAR], ids=["vector", "scalar"])
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_statement_needing_only_the_inner_loop_runs_in_the_outer_loop_too(
    config: dict[str, object], shape: tuple[int, int]
) -> None:
    # The count reads the column lane only, the copy both lanes.  The column
    # loop is materialized once, inside the row loop the copy needs, so the
    # count runs there as well, pinned to the first row lane; the flag's
    # load and the branch condition need no lane and are bound before the
    # loops, instead of the whole body running inside every loop.
    args = (torch.empty(shape), torch.zeros(shape[0]), torch.zeros(shape[1]))
    code = _generate(_flagged_col_count_then_copy, args, **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    loops = _lane_loops_around(function, atomic)
    assert loops[-1] == "lane_0" and "lane_1" in loops, code
    assert _first_lanes(_guard(function, atomic)) == {"lane_0"}, code
    flag_loads = [
        index
        for index, statement in enumerate(function.body)
        if not isinstance(statement, ast.For)
        and "flags.iterator" in ast.unparse(statement)
    ]
    assert flag_loads and flag_loads[0] < _first_lane_loop_index(function), code


def test_grid_atomic_leaves_the_unindexed_block_without_an_index() -> None:
    # ``examples/matmul_split_k.py`` under the config of the inactive-block
    # test in ``test_dot_requirements.py``: the outer K tile reaches the
    # kernel only as ``tile_offset_2`` (the inner loop's bounds and the
    # ``begin == 0`` test), so the grid gives it neither a lane loop nor a
    # thread axis, and the atomic on ``[tile_m, tile_n]`` carries no mask of
    # its axis (``tile_offset_2 < k`` holds for every program): it is issued
    # by the leader of the inner loop's thread axis alone.  Nothing reads
    # the block's index, so it is not emitted; were it, it could not claim a
    # thread axis.
    module = import_path(EXAMPLES_DIR / "matmul_split_k.py")
    args = (torch.empty(64, 1024), torch.empty(1024, 64))
    code = _generate(
        module.matmul_split_k,
        args,
        block_sizes=[16, 2, 16],
        num_threads=[0, 2, 8],
        split_k=32,
    )
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    guard = _guard(function, atomic)
    assert _leader_axes(guard) == {2} and "mask" not in (guard or ""), code
    assert _thread_axes(function) == {0, 1, 2}, code
    assert "tile_offset_2" in code
    for statement in ast.walk(function):
        if (
            isinstance(statement, ast.Assign)
            and ast.unparse(statement.targets[0]) == "indices_2"
        ):
            assert "thread_idx" not in ast.unparse(statement), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_count_then_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """A grid index addresses one element for the whole program, like ``tile.begin``."""
    out = torch.empty_like(x)
    for i in hl.grid(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            hl.atomic_add(counts, [i], 1)
            out[i, tile1] = x[i, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _segment_sums(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Each segment's total, every tile's sum added under the segment's grid index."""
    out = torch.zeros([offsets.size(0) - 1], dtype=torch.float32, device=x.device)
    for i in hl.grid(offsets.size(0) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        for tile_k in hl.tile(start, end):
            hl.atomic_add(out, [i], x[tile_k].sum())
    return out


_GRID_ROW = {"block_sizes": [32], "num_threads": [32], "cute_vector_widths": [1]}
_GRID_ROW_WIDE = {"block_sizes": [128], "num_threads": [128], "cute_vector_widths": [1]}


@pytest.mark.parametrize("config", [_GRID_ROW, _GRID_ROW_WIDE], ids=["warp", "cta"])
@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        pytest.param(
            _grid_count_then_copy,
            (torch.empty((6, 100)), torch.zeros((6,))),
            id="count",
        ),
        pytest.param(
            _segment_sums,
            (torch.empty((192,)), torch.tensor([0, 50, 50, 57, 89, 189, 192])),
            id="segment_sums",
        ),
    ],
)
def test_a_grid_indexed_atomic_is_tile_uniform(
    kernel: object, args: tuple[torch.Tensor, ...], config: dict[str, object]
) -> None:
    """``out[i]`` under ``for i in hl.grid(n)`` is one address for every thread of
    the program: the leader thread issues the atomic without the column tile's
    mask instead of every active thread adding the (reduced or constant) value."""
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=232448,
    ):
        code = _generate(kernel, args, **config)
    function = _kernel_function(code)
    atomic = _the_atomic(function)
    guard = _guard(function, atomic)
    assert _leader_axes(guard) == {0}, code
    assert "mask" not in (guard or ""), code
    # Inside the column tile loop only: no lane loop pins it.
    assert _lane_loops_around(function, atomic) == ["tile_offset_1"], code
