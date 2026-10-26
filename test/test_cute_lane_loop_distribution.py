"""GPU-free codegen coverage for CuTe lane-loop distribution.

``DeviceGridState.wrap_body`` used to nest every synthetic lane loop of a
root body around the whole body, so ``concat2d_dim1_simple``'s copy of ``x``
ran once per lane of ``y``'s slice (and vice versa).  Statements now live
inside only the lane loops whose coordinates they depend on.  When the
transform cannot place a statement, the full nest stays only if it is exact:
a nest that would repeat a lane-invariant memory access around per-lane
accesses of its tensor is rejected instead.  Whether an access repeats is
decided by the values it reads, not by the loops the structure places it in: a
partial tile's masks tie every access to every loop, and a device loop's lane
loops are built around its body before the body exists (that nest is checked
the same way, and redistributed like a root body when it is inexact).  A
placement is checked like the nest it
replaces: an inner loop nested inside an outer one (by its hoisted packets, or
because a statement needs both and each loop is materialized once) runs its
statements inside the outer loop whether or not their values change with it.
A ``tile.begin`` access carries no lane mask, so on a partial tile it is
placed as on a full one.  The nest is then checked across threads: two
accesses of one tensor that threads sharing a tile axis could reorder (a
``tile.begin`` read on every thread beside the owner's per-lane store) get a
block-wide barrier between them, or reject the config when that barrier
would sit in a branch.
"""

from __future__ import annotations

import ast
import dataclasses
import gc
import itertools
import re
import textwrap
from typing import TYPE_CHECKING

from examples.concatenate import concat2d_dim1_simple
import pytest
import sympy
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.ast_read_writes import HELION_ACCESS_REGIONS_ATTR
from helion._compiler.cute import lane_loop_distribution
from helion._compiler.cute.block_uniform import block_uniform
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _loops(node: ast.AST, prefix: str) -> list[ast.For]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.For)
        and isinstance(child.target, ast.Name)
        and child.target.id.startswith(prefix)
    ]


def _pointer(tensor: str, *terms: str) -> str:
    """The codegen's pointer arithmetic for an element of ``tensor`` at ``terms``."""
    return (
        f"({tensor}.iterator + "
        + " + ".join(
            f"cutlass.Int32({term}) * cutlass.Int32({tensor}.layout.stride[{dim}])"
            for dim, term in enumerate(terms)
        )
        + ")"
    )


def _accesses(node: ast.AST, tensor: str) -> bool:
    return any(
        isinstance(child, ast.Attribute)
        and child.attr == "iterator"
        and isinstance(child.value, ast.Name)
        and child.value.id == tensor
        for child in ast.walk(node)
    )


def _simple_kernel() -> object:
    return helion.kernel(
        concat2d_dim1_simple.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


def test_rolled_slice_copy_leaves_the_other_slices_lane_loop() -> None:
    # The study-winning config: ``x``'s 512-wide slice is a rolled reduction
    # (4 threads, chunk 256) and ``y``'s 768-wide slice a persistent one
    # (32 threads x 32 synthetic lanes).
    args = (torch.empty((2048, 512)), torch.empty((2048, 768)))
    code = _generate(
        _simple_kernel(),
        args,
        block_sizes=[1],
        num_threads=[0, 4, 32],
        reduction_loops=[256],
        cute_vector_widths=[4, 2, 8],
        cute_lane_layouts=["blocked", "blocked", "blocked"],
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "synthetic_lane_2")
    (rolled_loop,) = _loops(function, "roffset_1")
    # The x copy is a top-level statement that precedes y's lane loop ...
    assert rolled_loop in function.body and lane_loop in function.body, code
    assert function.body.index(rolled_loop) < function.body.index(lane_loop)
    # ... and y's lane loop only touches y and out.
    assert not _accesses(lane_loop, "x"), ast.unparse(lane_loop)
    assert _accesses(lane_loop, "y") and _accesses(lane_loop, "out")


def test_two_persistent_slices_become_sibling_lane_loops() -> None:
    args = (torch.empty((2048, 512)), torch.empty((2048, 768)))
    code = _generate(
        _simple_kernel(),
        args,
        block_sizes=[1],
        num_threads=[0, 32, 32],
        reduction_loops=[None],
        cute_vector_widths=[1, 1, 1],
    )
    function = _kernel_function(code)
    (x_loop,) = _loops(function, "synthetic_lane_1")
    (y_loop,) = _loops(function, "synthetic_lane_2")
    assert x_loop in function.body and y_loop in function.body, code
    assert function.body.index(x_loop) < function.body.index(y_loop)
    assert _accesses(x_loop, "x") and not _accesses(x_loop, "y")
    assert _accesses(y_loop, "y") and not _accesses(y_loop, "x")
    assert not _loops(x_loop, "synthetic_lane_2") and not _loops(
        y_loop, "synthetic_lane_1"
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_then_copy(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    acc = torch.zeros_like(x)
    out = torch.empty_like(y)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(acc, [tile_m, slice(None)], x[tile_m, :])
        out[tile_m, :] = y[tile_m, :]
    return acc, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_into_the_copied_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
    )
    n1 = x.size(1)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m, slice(None, n1)], x[tile_m, :])
        out[tile_m, n1:] = y[tile_m, :]
    return out


_TWO_SLICE_CONFIG = {
    "block_sizes": [1],
    "num_threads": [0, 32, 32],
    "reduction_loops": [None],
    "cute_vector_widths": [1, 1, 1],
}


def test_per_lane_atomic_and_copy_become_sibling_lane_loops() -> None:
    # The atomic accumulates x's slice per lane of x's loop and reads nothing
    # of y's loop; placed like a store, it leaves the loop it does not depend
    # on instead of pinning the full nest, so neither statement repeats per
    # lane of the other's slice.
    args = (torch.empty((64, 512)), torch.empty((64, 768)))
    code = _generate(_atomic_then_copy, args, **_TWO_SLICE_CONFIG)
    function = _kernel_function(code)
    (x_loop,) = _loops(function, "synthetic_lane_1")
    (y_loop,) = _loops(function, "synthetic_lane_2")
    assert x_loop in function.body and y_loop in function.body, code
    assert function.body.index(x_loop) < function.body.index(y_loop)
    assert _accesses(x_loop, "acc") and not _accesses(y_loop, "acc")
    assert _accesses(y_loop, "out") and not _accesses(x_loop, "out")
    assert "atomic_add" in ast.unparse(x_loop)
    assert "atomic_add" not in ast.unparse(y_loop)


def test_per_lane_atomic_into_the_copied_tensor_keeps_the_copy_after_its_loop() -> None:
    # The atomic and the copy write disjoint slices of one tensor.  The
    # transform cannot tell the slices apart, so the copy keeps its place
    # after the atomic's loop; both still run once per lane of their own
    # slice.
    args = (torch.empty((64, 512)), torch.empty((64, 768)))
    code = _generate(_atomic_into_the_copied_tensor, args, **_TWO_SLICE_CONFIG)
    function = _kernel_function(code)
    (x_loop,) = _loops(function, "synthetic_lane_1")
    (y_loop,) = _loops(function, "synthetic_lane_2")
    assert x_loop in function.body and y_loop in function.body, code
    assert function.body.index(x_loop) < function.body.index(y_loop)
    assert _accesses(x_loop, "out") and _accesses(y_loop, "out")
    assert "atomic_add" in ast.unparse(x_loop)
    assert "atomic_add" not in ast.unparse(y_loop)


@pytest.mark.parametrize("vec_width", [1, 4])
def test_single_slice_body_is_unchanged(
    vec_width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One lane loop that every statement depends on: nothing to distribute,
    # and the emitted code is exactly the code of the plain full nest.
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def multiply_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile_m in hl.tile(x.size(0)):
            out[tile_m, :] = x[tile_m, :] * y[tile_m, :]
        return out

    args = (torch.empty((64, 1024)), torch.empty((64, 1024)))
    config = {
        "block_sizes": [1],
        "num_threads": [0, 256],
        "reduction_loops": [None],
        "cute_vector_widths": [1, vec_width],
    }
    code = _generate(multiply_rows, args, **config)
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "synthetic_lane_1")
    assert _accesses(lane_loop, "x") and _accesses(lane_loop, "out"), code
    monkeypatch.setattr(
        lane_loop_distribution,
        "distribute_lane_loops",
        lambda body, scopes, **kwargs: None,
    )
    assert _generate(multiply_rows, args, **config) == code


def test_lane_invariant_constant_leaves_a_single_lane_loop() -> None:
    # A scalar the body binds without reading a lane coordinate is bound once,
    # before the loop, instead of once per lane.
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def scale_rows(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile_m in hl.tile(x.size(0)):
            out[tile_m, :] = x[tile_m, :] * 2.0
        return out

    code = _generate(
        scale_rows,
        (torch.empty((64, 1024)),),
        block_sizes=[1],
        num_threads=[0, 256],
        reduction_loops=[None],
        cute_vector_widths=[1, 4],
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "synthetic_lane_1")
    constants = [
        stmt
        for stmt in function.body[: function.body.index(lane_loop)]
        if isinstance(stmt, ast.Assign) and ast.unparse(stmt.value) == "2.0"
    ]
    assert len(constants) == 1, code
    assert "2.0" not in ast.unparse(lane_loop)
    assert _accesses(lane_loop, "x") and _accesses(lane_loop, "out"), code


def _defined_before_use(function: ast.FunctionDef) -> None:
    """Every locally assigned name is bound before each read, in an enclosing scope.

    The DSL lowers builtin ``range`` loops to region functions that only carry
    out names bound before the loop, so a name assigned inside such a loop is
    not visible after it.  A ``cutlass.range_constexpr`` loop stays a Python
    loop, whose bindings persist.
    """
    assigned = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    def reads(node: ast.AST) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    def writes(node: ast.AST) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }

    def check(body: list[ast.stmt], bound: set[str]) -> set[str]:
        for stmt in body:
            if isinstance(stmt, ast.For):
                unbound = (reads(stmt.iter) & assigned) - bound
                assert not unbound, (unbound, ast.unparse(stmt))
                inner = check(stmt.body, bound | writes(stmt.target))
                if isinstance(stmt.iter, ast.Call) and (
                    ast.unparse(stmt.iter.func) == "cutlass.range_constexpr"
                ):
                    bound = inner
            elif isinstance(stmt, ast.If):
                unbound = (reads(stmt.test) & assigned) - bound
                assert not unbound, (unbound, ast.unparse(stmt))
                bound = check(stmt.body, set(bound)) & check(stmt.orelse, set(bound))
            else:
                unbound = (reads(stmt) & assigned) - bound
                assert not unbound, (unbound, ast.unparse(stmt))
                bound = bound | writes(stmt)
        return bound

    check(function.body, {arg.arg for arg in function.args.args})


_ROW_CONFIG = {
    "block_sizes": [1, 1024],
    "num_threads": [0, 256],
    "cute_vector_widths": [1, 4],
}
# Four rows per thread around the vector lane loop: two live lane loops.
_NESTED_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 4],
}
# A thread-owned leading axis, a plain lane loop and the vector lane loop.
_NESTED_3D_CONFIG = {
    "block_sizes": [2, 4, 256],
    "num_threads": [2, 1, 64],
    "cute_vector_widths": [1, 1, 4],
}
_CONFIG_IDS = ["one_loop", "two_loops", "three_dims"]


def _vector_lane(function: ast.FunctionDef) -> tuple[ast.For, str]:
    """The lane loop holding the constexpr V-loop, and its axis suffix."""
    (vloop,) = _loops(function, "vec_lane_")
    (loop,) = [
        loop
        for loop in _loops(function, "lane_")
        if any(stmt is vloop for stmt in loop.body)
    ]
    assert isinstance(vloop.target, ast.Name)
    return loop, vloop.target.id.removeprefix("vec_lane_")


def _around(
    function: ast.FunctionDef, loop: ast.For
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Statements emitted before and after ``loop`` at every enclosing level."""
    before: list[ast.stmt] = []
    after: list[ast.stmt] = []
    body: list[ast.stmt] = function.body
    while True:
        (position,) = [
            index
            for index, stmt in enumerate(body)
            if stmt is loop or any(node is loop for node in ast.walk(stmt))
        ]
        before.extend(body[:position])
        after.extend(body[position + 1 :])
        if body[position] is loop:
            return before, after
        parent = body[position]
        assert isinstance(parent, ast.For), ast.unparse(parent)
        body = parent.body


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_and_echo(
    idx: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([idx.size(0), w.size(1)], dtype=w.dtype, device=w.device)
    echo = torch.empty([idx.size(0)], dtype=idx.dtype, device=idx.device)
    for tile0, tile1 in hl.tile(out.size()):
        rows = idx[tile0]
        echo[tile0] = rows
        out[tile0, tile1] = w[rows, tile1]
    return out, echo


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_and_echo_3d(
    idx: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        [idx.size(0), w.size(1), w.size(2)], dtype=w.dtype, device=w.device
    )
    echo = torch.empty([idx.size(0)], dtype=idx.dtype, device=idx.device)
    for tile0, tile1, tile2 in hl.tile(out.size()):
        rows = idx[tile0]
        echo[tile0] = rows
        out[tile0, tile1, tile2] = w[rows, tile1, tile2]
    return out, echo


_GATHER_CASES = [
    (
        _gather_and_echo,
        (torch.zeros((8,), dtype=torch.int64), torch.empty((16, 1024))),
        _ROW_CONFIG,
    ),
    (
        _gather_and_echo,
        (torch.zeros((8,), dtype=torch.int64), torch.empty((16, 1024))),
        _NESTED_CONFIG,
    ),
    (
        _gather_and_echo_3d,
        (torch.zeros((4,), dtype=torch.int64), torch.empty((16, 4, 256))),
        _NESTED_3D_CONFIG,
    ),
]


@pytest.mark.parametrize(("kernel", "args", "config"), _GATHER_CASES, ids=_CONFIG_IDS)
def test_gathered_row_read_twice_is_bound_before_the_packet_loop(
    kernel: object, args: tuple[torch.Tensor, ...], config: dict[str, object]
) -> None:
    # The gathered row index feeds the packet load hoisted into the vector
    # lane loop and an echo store that does not depend on that lane.  Both
    # the index load and the echo run once, before the loop (inside the row
    # lane loop when there is one); the packet inside reads the index.
    code = _generate(kernel, args, **config)
    function = _kernel_function(code)
    _defined_before_use(function)
    lane_loop, axis = _vector_lane(function)
    prefix, _ = _around(function, lane_loop)
    assert any(_accesses(stmt, "idx") for stmt in prefix), code
    assert any(_accesses(stmt, "echo") for stmt in prefix), code
    assert not _accesses(lane_loop, "idx") and not _accesses(lane_loop, "echo")
    packet = f"_tile_unroll_vec_{axis}_0 = cute.arch.load(w.iterator"
    assert packet in ast.unparse(lane_loop), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_flag_and_count(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    cnt = torch.empty_like(flags)
    for tile0, tile1 in hl.tile(out.size()):
        f = flags[tile0]
        out[tile0, tile1] = hl.load(x, [tile0, tile1], extra_mask=(f > 0)[:, None])
        cnt[tile0] = f + 1
    return out, cnt


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_flag_and_count_3d(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    cnt = torch.empty_like(flags)
    for tile0, tile1, tile2 in hl.tile(out.size()):
        f = flags[tile0]
        out[tile0, tile1, tile2] = hl.load(
            x, [tile0, tile1, tile2], extra_mask=(f > 0)[:, None, None]
        )
        cnt[tile0] = f + 1
    return out, cnt


_FLAG_CASES = [
    (
        _row_flag_and_count,
        (torch.empty((8, 1024)), torch.zeros((8,), dtype=torch.int32)),
        _ROW_CONFIG,
    ),
    (
        _row_flag_and_count,
        (torch.empty((8, 1024)), torch.zeros((8,), dtype=torch.int32)),
        _NESTED_CONFIG,
    ),
    (
        _row_flag_and_count_3d,
        (torch.empty((4, 4, 256)), torch.zeros((4,), dtype=torch.int32)),
        _NESTED_3D_CONFIG,
    ),
]


@pytest.mark.parametrize(("kernel", "args", "config"), _FLAG_CASES, ids=_CONFIG_IDS)
def test_row_flag_guards_the_packet_and_is_reused_after_the_loop(
    kernel: object, args: tuple[torch.Tensor, ...], config: dict[str, object]
) -> None:
    # The flag guards the hoisted packet (so it is bound before the vector
    # lane loop) and feeds a lane-invariant store after it; both read the
    # same binding.
    code = _generate(kernel, args, **config)
    function = _kernel_function(code)
    _defined_before_use(function)
    lane_loop, axis = _vector_lane(function)
    prefix, suffix = _around(function, lane_loop)
    assert any(_accesses(stmt, "flags") for stmt in prefix), code
    assert any(_accesses(stmt, "cnt") for stmt in suffix), code
    packet = f"_tile_unroll_vec_{axis}_0 = cute.arch.load(x.iterator"
    assert packet in ast.unparse(lane_loop), code
    assert not _accesses(lane_loop, "flags") and not _accesses(lane_loop, "cnt")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_zero(x: torch.Tensor) -> torch.Tensor:
    # ``x`` holds the rows to copy followed by as many spare rows.  Each tile
    # zeroes the spare row of its first row, which no tile reads, so the GPU
    # result does not depend on cross-thread timing; only the tensor name
    # relates the store to the packet loads.
    rows = x.size(0) // 2
    out = torch.empty([rows, x.size(1)], dtype=x.dtype, device=x.device)
    for tile0, tile1 in hl.tile(out.size()):
        v = x[tile0, tile1]
        x[tile0.begin + rows, 0] = 0.0
        out[tile0, tile1] = v
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_zero_3d(x: torch.Tensor) -> torch.Tensor:
    rows = x.size(1) // 2
    out = torch.empty([x.size(0), rows, x.size(2)], dtype=x.dtype, device=x.device)
    for tile0, tile1, tile2 in hl.tile(out.size()):
        v = x[tile0, tile1, tile2]
        x[tile0, tile1.begin + rows, 0] = 0.0
        out[tile0, tile1, tile2] = v
    return out


_ZERO_CASES = [
    (_read_then_zero, (torch.empty((16, 1024)),), _ROW_CONFIG),
    (_read_then_zero, (torch.empty((16, 256)),), _NESTED_CONFIG),
    (_read_then_zero_3d, (torch.empty((2, 8, 256)),), _NESTED_3D_CONFIG),
]


def _scalar_stores(function: ast.FunctionDef, tensor: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
        and f"{tensor}.iterator" in ast.unparse(node)
    ]


def _top_level_lane_loop(function: ast.FunctionDef) -> ast.For:
    (outermost,) = [
        stmt
        for stmt in function.body
        if isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id.startswith("lane_")
    ]
    return outermost


@pytest.mark.parametrize(("kernel", "args", "config"), _ZERO_CASES, ids=_CONFIG_IDS)
def test_store_between_a_hoisted_load_and_its_use_follows_the_loop_nest(
    kernel: object, args: tuple[torch.Tensor, ...], config: dict[str, object]
) -> None:
    # The lane-invariant store writes the tensor the hoisted packet reads and
    # sits between the load site and the store site.  The packet load stands
    # for the load site, which precedes the store, so the store goes after the
    # loop, once, where program order holds for any number of lane iterations
    # (the nest would zero the element again after the next iteration's
    # packet load).  That holds when the packet's loop is itself nested in
    # lane loops the store does not depend on: the store is placed relative
    # to the outer loop and checked against the inner loop's packets.
    code = _generate(kernel, args, **config)
    function = _kernel_function(code)
    _defined_before_use(function)
    (store,) = _scalar_stores(function, "x")
    outermost = _top_level_lane_loop(function)
    assert not any(node is store for node in ast.walk(outermost)), code
    trailing = function.body[function.body.index(outermost) + 1 :]
    assert any(node is store for stmt in trailing for node in ast.walk(stmt)), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _convert_bytes_then_zero(packed: torch.Tensor) -> torch.Tensor:
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        out[tile0.begin, 0] = 0.0
        out[tile0, tile1] = y
    return out


# One row per program; 32 threads own four bytes each per lane iteration.
_BYTE_CONFIG = {
    "block_sizes": [1, 128],
    "num_threads": [0, 32],
    "cute_vector_widths": [1, 4],
}
# The same threads run two lane iterations per row.
_BYTE_CONFIG_TWO_LANES = {**_BYTE_CONFIG, "block_sizes": [1, 256]}
_BYTE_CASES = [
    pytest.param(config, packet_flush, id=f"{lanes}-{protocol}")
    for lanes, config in (
        ("one_lane", _BYTE_CONFIG),
        ("two_lanes", _BYTE_CONFIG_TWO_LANES),
    )
    for protocol, packet_flush in (("values", False), ("packet", True))
]


@pytest.mark.parametrize(("config", "packet_flush"), _BYTE_CASES)
def test_store_between_a_byte_conversion_and_its_flush_precedes_the_loop(
    config: dict[str, object], packet_flush: bool
) -> None:
    # The lane-invariant store sits between the conversion and the store site
    # whose flush overwrites its element.  The flush stands for that site,
    # which follows the store, so the store goes before the loop, once: kept
    # in the nest it would zero the element again in the lane iteration after
    # the one whose flush wrote it.  With ``cute_signed_bitfield_bf16`` the
    # flush converts the whole byte packet after the V-loop and the site only
    # binds the packet under the flush operand's name; that binding is what
    # relates the flush to its place in the body.
    code = _generate(
        _convert_bytes_then_zero,
        (torch.empty((8, 256), dtype=torch.int8),),
        **config,
        cute_signed_bitfield_bf16=packet_flush,
    )
    function = _kernel_function(code)
    _defined_before_use(function)
    assert ("_cute_signed_bitfield_to_bf16_packed(" in code) is packet_flush, code
    assert not any(isinstance(node, ast.Pass) for node in ast.walk(function)), code
    loop, _axis = _vector_lane(function)
    (zeroing,) = _scalar_stores(function, "out")
    assert not any(node is zeroing for node in ast.walk(loop)), code
    before, after = _around(function, loop)
    assert any(node is zeroing for stmt in before for node in ast.walk(stmt)), code
    assert not any(_accesses(stmt, "out") for stmt in after), code
    (flush,) = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_store_u16_vec"
    ]
    if packet_flush:
        # The flush reads the packet under the name the site binds inside the
        # V-loop.
        operand = flush.args[1]
        assert isinstance(operand, ast.Call), code
        (packet, *_bits) = operand.args
        assert isinstance(packet, ast.Name), code
        (vloop,) = _loops(function, "vec_lane_")
        assert any(
            isinstance(stmt, ast.Assign)
            and ast.unparse(stmt).startswith(f"{packet.id} = _tile_unroll_vec_")
            for stmt in vloop.body
        ), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_zero_copy(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    out2 = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        out[tile0, tile1] = y
        out[tile0.begin, 255] = 0.0
        out2[tile0, tile1] = y
    return out, out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_zero_copy_again(packed: torch.Tensor) -> torch.Tensor:
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        out[tile0, tile1] = y
        out[tile0.begin, 0] = 0.0
        out[tile0, tile1] = y * 2
    return out


@pytest.mark.parametrize("packet_flush", [False, True], ids=["values", "packet"])
def test_store_after_a_per_lane_store_of_its_tensor_follows_the_loop(
    packet_flush: bool,
) -> None:
    # The zeroing store follows the flushed copy into ``out`` and precedes a
    # copy into another tensor, whose buffer site (an append, or the packet
    # binding) is register work it does not depend on: it goes after the
    # loop, where it overwrites the copy's element for any number of lane
    # iterations.
    code = _generate(
        _copy_zero_copy,
        (torch.empty((8, 256), dtype=torch.int8),),
        **_BYTE_CONFIG_TWO_LANES,
        cute_signed_bitfield_bf16=packet_flush,
    )
    function = _kernel_function(code)
    _defined_before_use(function)
    loop, _axis = _vector_lane(function)
    (zeroing,) = _scalar_stores(function, "out")
    assert not any(node is zeroing for node in ast.walk(loop)), code
    before, after = _around(function, loop)
    assert any(node is zeroing for stmt in after for node in ast.walk(stmt)), code
    assert not any(_accesses(stmt, "out") for stmt in before), code


@pytest.mark.parametrize(
    "config",
    [
        {**_BYTE_CONFIG_TWO_LANES, "cute_vector_widths": [1, 1]},
        _BYTE_CONFIG_TWO_LANES,
        {**_BYTE_CONFIG_TWO_LANES, "cute_signed_bitfield_bf16": True},
    ],
    ids=["scalar", "values", "packet"],
)
def test_store_between_two_per_lane_stores_of_its_tensor_rejects_the_config(
    config: dict[str, object],
) -> None:
    # The zeroing store must follow the first copy and precede the second;
    # neither side of the loop does, and the nest would zero the element
    # again after the second copy's next lane iteration wrote it.  The
    # config is rejected in the scalar lowering and in both store protocols.
    with pytest.raises(exc.BackendUnsupported, match="lane loop nest"):
        _generate(
            _copy_zero_copy_again, (torch.empty((8, 256), dtype=torch.int8),), **config
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunked_recurrence(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    rows, chunks, columns = x.shape
    out = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        acc = hl.zeros([col], dtype=torch.float32)
        for chunk in hl.grid(chunks):
            out[row.begin, chunk, col] = acc.to(x.dtype)
            acc = acc * decay[row.begin, chunk] + x[row.begin, chunk, col].float()
    return out


def test_loop_carried_accumulator_keeps_its_per_lane_initialization() -> None:
    # The chunk loop rewrites ``acc`` per lane, but at distribution time its
    # update is still written under the loop-output name that a later pass
    # renames to ``acc``.  Read through the rename groups, the initialization
    # depends on the lane like the updates and stays inside the vector lane
    # loop; hoisting it would carry one lane's final value into the next.
    x = torch.zeros((2, 3, 1024), dtype=torch.bfloat16)
    decay = torch.ones((2, 3), dtype=torch.float32)
    code = _generate(
        _chunked_recurrence,
        (x, decay),
        block_sizes=[1024],
        num_threads=[128],
        cute_vector_widths=[1, 8],
    )
    function = _kernel_function(code)
    _defined_before_use(function)
    (vloop,) = _loops(function, "vec_lane_")
    inits = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "acc"
        and ast.unparse(node.value) == "cutlass.Float32(0.0)"
    ]
    assert len(inits) == 1, code
    assert any(node is inits[0] for node in ast.walk(vloop)), code
    assert any(_loops(stmt, "tile_offset_") for stmt in vloop.body), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_copy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(x)
    first = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile0, tile1 in hl.tile(x.size()):
        first[tile0] = out[tile0, 0]
        out[tile0, tile1] = x[tile0, tile1]
    return out, first


def test_read_before_a_flushed_store_precedes_the_loop() -> None:
    # The lane-invariant read of ``out`` precedes the copy whose packets are
    # flushed at the end of the lane loop; it conflicts with that flush and
    # is emitted before the loop, never after it.
    code = _generate(_read_then_copy, (torch.empty((8, 1024)),), **_ROW_CONFIG)
    function = _kernel_function(code)
    _defined_before_use(function)
    (lane_loop,) = _loops(function, "lane_1")
    position = function.body.index(lane_loop)
    assert "_cute_store_u32_vec(out.iterator" in ast.unparse(lane_loop), code
    assert any(_accesses(stmt, "first") for stmt in function.body[:position]), code
    assert not any(_accesses(stmt, "first") for stmt in function.body[position:])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_row_then_update_all(x: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first = x[tile0.begin, tile1]
        x[tile0, tile1] = x[tile0, tile1] + first[None, :]
    return x


# Four rows per thread around the column lane loop, vectorized or scalar.
_NESTED_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 4],
}
_NESTED_SCALAR_CONFIG = {**_NESTED_CONFIG, "cute_vector_widths": [1, 1]}


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_tile_uniform_load_repeated_around_per_lane_stores_rejects_the_config(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    # The original nest is emitted as is: on the full tile the first row's
    # packet load belongs to the column loop, which the copy's packet nests
    # in the row loop (or the column loop would need two instances); on the
    # partial tile every access reads the row mask.  The first row's load
    # changes with neither the row lane nor its mask, so every row iteration
    # would re-issue it after the earlier rows' stores to the same tensor.
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant load of x would repeat"
    ):
        _generate(_first_row_then_update_all, (torch.empty(shape),), **config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_first_row_then_copy(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0.begin, tile1] = 0.0
        out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_zero_first_row(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        out[tile0.begin, tile1] = 0.0
    return out


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_masked_tile_uniform_store_before_per_lane_stores_rejects_the_config(
    config: dict[str, object],
) -> None:
    # On a partial tile the zeroing store reads the row mask, which keeps it
    # inside the row loop; its address does not change with the row lane, so
    # the second row iteration would zero the first row again after the first
    # iteration's copy wrote it.
    args = (torch.empty((6, 250)), torch.empty((6, 250)))
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant store to out would repeat"
    ):
        _generate(_zero_first_row_then_copy, args, **config)


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_masked_tile_uniform_store_after_per_lane_stores_keeps_the_nest(
    config: dict[str, object],
) -> None:
    # Repeated after every row's copy, the zeroing store leaves the first row
    # zero in every order: the nest is exact and kept, the store inside the
    # row loop.
    args = (torch.empty((6, 250)), torch.empty((6, 250)))
    code = _generate(_copy_then_zero_first_row, args, **config)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    zero_stores = [
        call
        for call in ast.walk(row_loop)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "store"
        and "tile_offset_0" in ast.unparse(call.func.value)
    ]
    assert zero_stores, code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_first_column_then_copy_in_inner_tile(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            out[tile0, tile1.begin] = 0.0
            out[tile0, tile1] = x[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_zero_first_column_in_inner_tile(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            out[tile0, tile1] = x[tile0, tile1]
            out[tile0, tile1.begin] = 0.0
    return out


# One row per program; the inner column loop walks 32 threads x 8 lanes.
_INNER_LANES_CONFIG = {
    "block_sizes": [1, 256],
    "num_threads": [0, 32],
    "cute_vector_widths": [1, 1],
}


def test_tile_uniform_store_in_an_inner_loops_lane_nest_is_placed() -> None:
    # A device loop's lane loops are built around its body before the body
    # exists.  The zeroing store ignores the column lane: before the copy it
    # would be re-applied after the first lane's copy of the same element, so
    # the inexact nest is redistributed: the store runs once, before the lane
    # loop, and a barrier orders it before the other threads' copies ...
    args = (torch.empty((8, 512)), torch.empty((8, 512)))
    code = _generate(
        _zero_first_column_then_copy_in_inner_tile, args, **_INNER_LANES_CONFIG
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "lane_1")
    (tile_loop,) = _loops(function, "tile_offset_1")
    position = tile_loop.body.index(lane_loop)
    before = ast.unparse(tile_loop.body[:position])
    assert _accesses(tile_loop.body[0], "out"), code
    assert "cute.arch.sync_threads()" in before, code
    assert _accesses(lane_loop, "out") and _accesses(lane_loop, "x"), code
    # ... and after every lane's copy it is exact, inside the inner loop's
    # lane loop.
    code = _generate(
        _copy_then_zero_first_column_in_inner_tile, args, **_INNER_LANES_CONFIG
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "lane_1")
    (tile_loop,) = _loops(function, "tile_offset_1")
    assert any(node is lane_loop for node in ast.walk(tile_loop)), code
    assert _accesses(lane_loop, "out") and _accesses(lane_loop, "x"), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_row_then_update_all_plus_one(x: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first = x[tile0.begin, tile1]
        x[tile0, tile1] = x[tile0, tile1] + first[None, :] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_first_row_then_copy_plus_one(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0.begin, tile1] = 0.0
        out[tile0, tile1] = x[tile0, tile1] + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_row_increment_beside_copy(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile0, tile1 in hl.tile(x.shape):
        x[tile0.begin, tile1] = x[tile0.begin, tile1] + 1.0
        out[tile0, tile1] = y[tile0, tile1]
    return x, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_row_scaled_beside_copy(
    x: torch.Tensor, out: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0.begin, tile1] = x[tile0.begin, tile1] * 2.0
        y[tile0, tile1] = x[tile0, tile1]
    return out, y


# A loop-invariant constant in the body leaves the lane loops, so these bodies
# are emitted as placements rather than as the original nest.
_PLACED_REPETITION_CASES = [
    pytest.param(_first_row_then_update_all_plus_one, 1, "load of x", id="update"),
    pytest.param(_zero_first_row_then_copy_plus_one, 2, "store to out", id="zero"),
    pytest.param(_first_row_increment_beside_copy, 3, "load of x", id="increment"),
]


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
@pytest.mark.parametrize(("kernel", "arity", "access"), _PLACED_REPETITION_CASES)
def test_placed_nest_repeating_a_tile_uniform_access_rejects_the_config(
    kernel: object,
    arity: int,
    access: str,
    shape: tuple[int, int],
    config: dict[str, object],
) -> None:
    # The first row's access belongs to the column loop, which the copy's
    # packet (its per-lane index) nests inside the row loop: it runs once per
    # row iteration, after the earlier rows' stores to the same tensor, in
    # the placement exactly as in the original nest.
    args = tuple(torch.empty(shape) for _ in range(arity))
    with pytest.raises(
        exc.BackendUnsupported, match=f"lane-invariant {access} would repeat"
    ):
        _generate(kernel, args, **config)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_placed_nest_repeating_an_idempotent_store_keeps_the_placement(
    shape: tuple[int, int],
) -> None:
    # The scaled first row is stored again by every row iteration, over
    # itself, and its load only reads ``x``: the repetition is exact and the
    # placement stands, the constant bound before the row loop.  On the
    # partial tile the first row's accesses need the column loop only; the
    # copy needs both, so the column loop's one instance sits in the row
    # loop and the first row's accesses run there too.
    args = tuple(torch.empty(shape) for _ in range(3))
    code = _generate(_first_row_scaled_beside_copy, args, **_NESTED_CONFIG)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    position = function.body.index(row_loop)
    assert any("2.0" in ast.unparse(s) for s in function.body[:position]), code
    assert _accesses(row_loop, "out") and _accesses(row_loop, "y"), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_row_to_vector(
    x: torch.Tensor, first: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first[tile1] = x[tile0.begin, tile1]
        out[tile0, tile1] = x[tile0, tile1]
    return out


def _guards(function: ast.FunctionDef, address: str) -> list[str]:
    """The conditions guarding the accesses whose address mentions ``address``."""
    guards = []
    for node in ast.walk(function):
        if isinstance(node, ast.IfExp):
            call, test = node.body, node.test
        elif isinstance(node, ast.If) and len(node.body) == 1:
            call, test = node.body[0], node.test
        else:
            continue
        if isinstance(call, ast.Expr):
            call = call.value
        if isinstance(call, ast.Call) and address in ast.unparse(call.func):
            guards.append(ast.unparse(test))
    return guards


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_tile_begin_access_carries_no_lane_mask(config: dict[str, object]) -> None:
    # A ``tile.begin`` component is one address for the whole tile, in range
    # whenever the tile is.  Guarded by the row mask as well, the first row's
    # load would be gated to zero in the lanes past the tile's last row, and
    # the store of ``first`` (a tensor without a row axis, guarded by the
    # column mask only) would write that zero.
    args = (torch.empty((6, 250)), torch.empty(250), torch.empty((6, 250)))
    function = _kernel_function(_generate(_first_row_to_vector, args, **config))
    first_row = _guards(function, "tile_offset_0")
    assert first_row and all("mask_0" not in guard for guard in first_row), first_row
    (store,) = _guards(function, "first.iterator")
    assert "mask_0" not in store and "mask_1" in store, store
    # The per-lane copy keeps both masks.
    copy = _guards(function, "indices_0")
    assert copy and all("mask_0" in guard for guard in copy), copy


# ---------------------------------------------------------------------------
# Block-wide barriers between accesses that threads sharing a tile axis could
# reorder (``add_thread_barriers``).


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_then_update_all(x: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first = x[tile0, tile1.begin]
        x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_then_overwrite_beside_row_sums(
    x: torch.Tensor, out: torch.Tensor, sums: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first = out[tile0, tile1.begin]
        sums[tile0] = x[tile0, tile1].sum(-1)
        out[tile0, tile1] = x[tile0, tile1] + first[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sum_then_normalized_overwrite(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        v = x[tile0, tile1]
        s = v.sum(-1)
        out[tile0, tile1.begin] = s
        out[tile0, tile1] = v / s[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sum_stored_twice(
    x: torch.Tensor, out: torch.Tensor, sums: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        v = x[tile0, tile1]
        out[tile0, tile1] = v * 2.0
        sums[tile0] = v.sum(-1)
        sums[tile0] = v.sum(-1) * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_then_update_all_3d(x: torch.Tensor) -> torch.Tensor:
    for tile0, tile1, tile2 in hl.tile(x.shape):
        first = x[tile0, tile1, tile2.begin]
        x[tile0, tile1, tile2] = x[tile0, tile1, tile2] + first[:, :, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _double_first_column_beside_copy(
    x: torch.Tensor, out: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first = out[tile0, tile1.begin]
        out[tile0, tile1.begin] = first * 2.0
        y[tile0, tile1] = x[tile0, tile1]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_first_column_then_copy_plus_one(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1.begin] = 0.0
        out[tile0, tile1] = x[tile0, tile1] + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_zero_first_column(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        out[tile0, tile1.begin] = 0.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_first_column_then_read(
    x: torch.Tensor, out: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1.begin] = 0.0
        y[tile0, tile1] = out[tile0, tile1] + x[tile0, tile1]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _inner_first_column_then_update_all(x: torch.Tensor) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            first = x[tile0, tile1.begin]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flagged_inner_first_column_then_update_all(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        if flags[tile0.begin] > 0:
            for tile1 in hl.tile(x.size(1)):
                first = x[tile0, tile1.begin]
                x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


# Four threads own the four rows: the row axis has no lane loop.
_THREADED_ROWS_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [4, 64],
    "cute_vector_widths": [1, 4],
}
_THREADED_ROWS_SCALAR_CONFIG = {**_THREADED_ROWS_CONFIG, "cute_vector_widths": [1, 1]}
# One thread walks the whole tile: no axis is shared.
_ONE_THREAD_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [1, 1],
    "cute_vector_widths": [1, 1],
}
# Three lane loops; the last axis spans 32 threads (review 8's N3S / N3C).
_THREE_LOOPS_SCALAR_CONFIG = {
    "block_sizes": [2, 4, 128],
    "num_threads": [1, 2, 32],
    "cute_vector_widths": [1, 1, 1],
}
_THREE_LOOPS_CONFIG = {
    "block_sizes": [2, 4, 128],
    "num_threads": [1, 4, 32],
    "cute_vector_widths": [1, 1, 4],
}
# The last axis has one thread per element on the y thread axis (other warps).
_THREADED_PLANES_CONFIG = {
    "block_sizes": [2, 128, 8],
    "num_threads": [1, 32, 8],
    "cute_vector_widths": [1, 1, 1],
}
# Two rows per thread around an inner tile whose columns are one per y thread.
_INNER_THREADED_COLUMNS_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [2, 64],
    "cute_vector_widths": [1, 1],
}


def _is_barrier(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.arch.sync_threads"
    )


def _barriers(node: ast.AST) -> list[ast.Expr]:
    return [child for child in ast.walk(node) if _is_barrier(child)]  # pyrefly: ignore [bad-return]


def _index_of(body: Sequence[ast.AST], text: str, *, after: int = -1) -> int:
    """The first statement of ``body`` past ``after`` whose source mentions ``text``."""
    return next(
        index
        for index, statement in enumerate(body)
        if index > after and text in ast.unparse(statement)
    )


def _one_barrier_between(body: Sequence[ast.AST], first: str, second: str) -> None:
    """``body`` holds one barrier, between the statements mentioning ``first`` and ``second``."""
    start = _index_of(body, first)
    end = _index_of(body, second, after=start)
    positions = [
        index for index, statement in enumerate(body) if _is_barrier(statement)
    ]
    assert positions and all(start < p < end for p in positions), (
        [ast.unparse(s)[:60] for s in body],
    )


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_uniform_read_before_per_lane_stores_on_a_shared_axis_gets_a_barrier(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    # The first column's load runs on every thread of the column axis and
    # reads the element thread 0 stores inside the column loop; the barrier
    # between the load and the loop keeps every thread's read ahead of any
    # thread's store.  Per thread the placement was already exact.
    code = _generate(_first_column_then_update_all, (torch.empty(shape),), **config)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    (column_loop,) = _loops(row_loop, "lane_1")
    assert len(_barriers(function)) == 1, code
    load = _index_of(row_loop.body, "tile_offset_1")
    barrier = next(i for i, s in enumerate(row_loop.body) if _is_barrier(s))
    assert load < barrier < row_loop.body.index(column_loop), code


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_racing_accesses_beside_a_lane_reduction_reject_the_config(
    config: dict[str, object],
) -> None:
    # As in ``_first_column_then_update_all``, every thread of the column axis
    # reads the first column that thread 0 overwrites in the column loop.  The
    # row sum's ``_helion_lane_reduce`` leaves the body to the reduction split,
    # which would carry no barrier between the read and the stores: the race
    # is refused instead of left open.
    args = (torch.empty((8, 256)), torch.empty((8, 256)), torch.empty(8))
    with pytest.raises(
        exc.BackendUnsupported,
        match=r"race on a tensor in a body a later pass rewrites "
        r"\(_helion_lane_reduce\)",
    ):
        _generate(_first_column_then_overwrite_beside_row_sums, args, **config)


def test_lane_reduction_without_a_shared_thread_axis_keeps_the_config() -> None:
    # One thread per row: no other thread reads the first column, nothing to
    # order.
    args = (torch.empty((8, 256)), torch.empty((8, 256)), torch.empty(8))
    code = _generate(
        _first_column_then_overwrite_beside_row_sums, args, **_ONE_THREAD_CONFIG
    )
    assert not _barriers(_kernel_function(code)), code


def test_reduced_stores_beside_a_lane_reduction_need_no_barrier() -> None:
    # Every thread of the column axis stores the row sum, which the split
    # combines over those threads before either store: each thread writes the
    # same two values in program order, so the last one stands whatever the
    # threads' order.
    args = (torch.empty((8, 256)), torch.empty((8, 256)), torch.empty(8))
    code = _generate(_row_sum_stored_twice, args, **_NESTED_SCALAR_CONFIG)
    assert not _barriers(_kernel_function(code)), code


def test_reduced_store_then_per_element_overwrite_rejects_the_config() -> None:
    # The first column's per-element value must replace the row sum every
    # thread of the column axis stores there first; a thread still storing
    # the sum after thread 0's overwrite leaves the wrong value, and the
    # barrier between the stores cannot sit in the body the split rewrites.
    args = (torch.empty((8, 256)), torch.empty((8, 256)))
    with pytest.raises(exc.BackendUnsupported, match="a body a later pass rewrites"):
        _generate(_row_sum_then_normalized_overwrite, args, **_NESTED_SCALAR_CONFIG)


def test_no_barrier_when_every_axis_has_one_thread() -> None:
    code = _generate(
        _first_column_then_update_all, (torch.empty((8, 256)),), **_ONE_THREAD_CONFIG
    )
    assert not _barriers(_kernel_function(code)), code


def test_uniform_read_on_a_thread_axis_without_a_lane_loop_gets_a_barrier() -> None:
    # Each thread owns one row: nothing distinguishes the first row's packet
    # from a per-thread one in the lane structure, only its address, which
    # ignores the row thread coordinate.  The packet is hoisted above the
    # V-loop and the update flushed after it: the barrier sits between the
    # V-loop and the flush.
    code = _generate(
        _first_row_then_update_all_plus_one,
        (torch.empty((8, 256)),),
        **_THREADED_ROWS_CONFIG,
    )
    function = _kernel_function(code)
    (column_loop,) = _loops(function, "lane_1")
    (vloop,) = _loops(column_loop, "vec_lane_1")
    assert len(_barriers(function)) == 1, code
    barrier = next(i for i, s in enumerate(column_loop.body) if _is_barrier(s))
    flush = _index_of(column_loop.body, "_cute_store_u32_vec")
    assert column_loop.body.index(vloop) < barrier < flush, code
    # Scalar lanes: load and store are statements of the column loop's body.
    code = _generate(
        _first_row_then_update_all_plus_one,
        (torch.empty((8, 256)),),
        **_THREADED_ROWS_SCALAR_CONFIG,
    )
    (column_loop,) = _loops(_kernel_function(code), "lane_1")
    _one_barrier_between(column_loop.body, "tile_offset_0", ".store(")


def test_uniform_load_then_uniform_store_of_one_element_gets_a_barrier() -> None:
    # Every thread of the column axis doubles the same element: without the
    # barrier a thread reads a value another thread already doubled.
    args = tuple(torch.empty((8, 256)) for _ in range(3))
    code = _generate(_double_first_column_beside_copy, args, **_NESTED_CONFIG)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    assert len(_barriers(function)) == 1, code
    _one_barrier_between(row_loop.body, ").load()", ".store(")


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_uniform_store_before_per_lane_stores_gets_a_barrier(
    config: dict[str, object],
) -> None:
    # Another thread's zero store may land after the owner's copy.
    args = (torch.empty((8, 256)), torch.empty((8, 256)))
    code = _generate(_zero_first_column_then_copy_plus_one, args, **config)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    (column_loop,) = _loops(row_loop, "lane_1")
    assert len(_barriers(function)) == 1, code
    store = _index_of(row_loop.body, ".store(")
    barrier = next(i for i, s in enumerate(row_loop.body) if _is_barrier(s))
    assert store < barrier < row_loop.body.index(column_loop), code


def test_idempotent_uniform_stores_need_no_barrier() -> None:
    # A uniform store after the per-lane copies: every thread's last write of
    # the element is the same zero.  Before a per-lane load: each thread's
    # own zero precedes its read (the load keeps the codegen's read-after-
    # write barrier inside the column loop, from ``mark_intra_loop_raw_barriers``).
    args = (torch.empty((8, 256)), torch.empty((8, 256)))
    code = _generate(_copy_then_zero_first_column, args, **_NESTED_CONFIG)
    assert not _barriers(_kernel_function(code)), code
    args = (torch.empty((8, 256)), torch.empty((8, 256)), torch.empty((8, 256)))
    code = _generate(_zero_first_column_then_read, args, **_NESTED_CONFIG)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    (column_loop,) = _loops(row_loop, "lane_1")
    assert len(_barriers(function)) == 1, code
    assert not any(_is_barrier(s) for s in row_loop.body), code
    assert _barriers(column_loop), code


@pytest.mark.parametrize(
    ("config", "loop"),
    [(_THREE_LOOPS_SCALAR_CONFIG, "lane_1"), (_THREE_LOOPS_CONFIG, "lane_0")],
    ids=["two_threads_per_plane", "threaded_planes"],
)
def test_uniform_read_hoisted_out_of_the_innermost_shared_loop_gets_a_barrier(
    config: dict[str, object], loop: str
) -> None:
    # Review 8's kernel: the first column's load is hoisted before the column
    # loop, whose axis spans 32 threads in two warps.  With four threads on
    # the plane axis that axis has no lane loop and the load sits in the row
    # loop's body instead.
    code = _generate(
        _first_column_then_update_all_3d, (torch.empty((4, 8, 256)),), **config
    )
    function = _kernel_function(code)
    (outer,) = _loops(function, loop)
    (column_loop,) = _loops(outer, "lane_2")
    assert len(_barriers(function)) == 1, code
    load = _index_of(outer.body, "tile_offset_2")
    barrier = next(i for i, s in enumerate(outer.body) if _is_barrier(s))
    assert load < barrier < outer.body.index(column_loop), code


def test_uniform_read_on_a_second_thread_axis_gets_a_barrier() -> None:
    # The last axis has one thread per element on the y thread axis; the
    # first column's load and the update are statements of the middle lane
    # loop's body, the barrier between them.
    code = _generate(
        _first_column_then_update_all_3d,
        (torch.empty((4, 256, 16)),),
        **_THREADED_PLANES_CONFIG,
    )
    function = _kernel_function(code)
    (middle,) = _loops(function, "lane_1")
    assert len(_barriers(function)) == 1, code
    _one_barrier_between(middle.body, "tile_offset_2", ".store(")


def test_device_loop_body_gets_a_barrier_for_its_own_thread_axis() -> None:
    # The inner tile's columns are one per y thread without a lane loop; the
    # first column's load and the update are the device loop body's own
    # statements and the barrier separates them there.
    code = _generate(
        _inner_first_column_then_update_all,
        (torch.empty((8, 256)),),
        **_INNER_THREADED_COLUMNS_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    assert len(_barriers(function)) == 1, code
    _one_barrier_between(device_loop.body, "tile_offset_1) *", ".store(")


def _flags_aliasing(x: torch.Tensor) -> torch.Tensor:
    """Flags that are a view of ``x``, which the kernels write in place."""
    return x.view(-1)[: x.size(0)]


def test_racing_accesses_inside_a_branch_reject_the_config() -> None:
    # A branch condition may vary per thread (here a flag the kernel itself
    # overwrites, passed aliased to ``x``), and a block-wide barrier inside a
    # branch some threads skip would deadlock: fail closed.
    x = torch.empty((8, 256))
    args = (x, _flags_aliasing(x))
    with pytest.raises(exc.BackendUnsupported, match="inside a branch"):
        _generate(
            _flagged_inner_first_column_then_update_all,
            args,
            **_INNER_THREADED_COLUMNS_CONFIG,
        )


def _threaded_column_scopes() -> list[lane_loop_distribution.LaneScope]:
    """A row lane loop on one thread and a column lane loop over 64 threads.

    Each column thread owns four consecutive columns, one per lane; the
    trip counts let the barrier pass prove that layout one thread's.
    """
    return [
        lane_loop_distribution.LaneScope(
            "lane_0",
            frozenset({"lane_0", "indices_0"}),
            frozenset(),
            (),
            frozenset({"lane_0"}),
            tuple(ast.parse("indices_0 = tile_offset_0 + cutlass.Int32(lane_0)").body),
            "lane_0 == 0",
            counts={"lane_0": 8},
        ),
        lane_loop_distribution.LaneScope(
            "lane_1",
            frozenset({"lane_1", "indices_1"}),
            frozenset(),
            (),
            frozenset({"lane_1"}),
            tuple(
                ast.parse(
                    "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])"
                    " * 4 + cutlass.Int32(lane_1)"
                ).body
            ),
            "lane_1 == 0",
            counts={"lane_1": 4},
        ),
    ]


_UNIFORM_LOAD = f"v = {_pointer('x', 'indices_0', 'tile_offset_1')}.load()"
_UNIFORM_STORE = f"{_pointer('x', 'indices_0', 'tile_offset_1')}.store(c)"
_PER_LANE_LOAD = f"w = {_pointer('x', 'indices_0', 'indices_1')}.load()"
_PER_LANE_STORE = f"{_pointer('x', 'indices_0', 'indices_1')}.store(c)"


@pytest.mark.parametrize(
    ("outside", "inside", "barrier"),
    [
        pytest.param(_UNIFORM_LOAD, _PER_LANE_STORE, True, id="load_then_stores"),
        pytest.param(_UNIFORM_STORE, _PER_LANE_STORE, True, id="store_then_stores"),
        pytest.param(_UNIFORM_STORE, _PER_LANE_LOAD, False, id="store_then_loads"),
        pytest.param(_UNIFORM_LOAD, _PER_LANE_LOAD, False, id="loads_only"),
    ],
)
def test_thread_barrier_rules(outside: str, inside: str, barrier: bool) -> None:
    # Rule level: a statement outside the column loop (uniform along the 64
    # column threads) before a per-lane access of the same tensor inside it.
    # Only a uniform store before per-lane loads is safe as emitted: each
    # thread's own store precedes its read.
    scopes = _threaded_column_scopes()
    first, second = ast.parse(outside).body[0], ast.parse(inside).body[0]
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [
        lane_loop_distribution.LanePlacement(
            "lane_0", [first, lane_loop_distribution.LanePlacement("lane_1", [second])]
        )
    ]
    lane_loop_distribution.add_thread_barriers(
        items,
        scopes,
        rename_groups={},
        axis_sizes={0: 64},
        definitions=[statement for scope in scopes for statement in scope.setup],
        masks=frozenset(),
    )
    (row,) = items
    assert isinstance(row, lane_loop_distribution.LanePlacement)
    emitted = [
        "loop"
        if isinstance(item, lane_loop_distribution.LanePlacement)
        else ast.unparse(item)
        for item in row.items
    ]
    expected = (
        [outside, "cute.arch.sync_threads()", "loop"] if barrier else [outside, "loop"]
    )
    assert emitted == expected, emitted


def test_thread_barrier_rules_after_the_loop() -> None:
    # A uniform store after the per-lane stores needs none (every thread's
    # last write is the same store); a uniform load after them does.
    for statement, barrier in ((_UNIFORM_STORE, False), (_UNIFORM_LOAD, True)):
        scopes = _threaded_column_scopes()
        second, first = ast.parse(statement).body[0], ast.parse(_PER_LANE_STORE).body[0]
        items: list[ast.AST | lane_loop_distribution.LanePlacement] = [
            lane_loop_distribution.LanePlacement(
                "lane_0",
                [lane_loop_distribution.LanePlacement("lane_1", [first]), second],
            )
        ]
        lane_loop_distribution.add_thread_barriers(
            items,
            scopes,
            rename_groups={},
            axis_sizes={0: 64},
            definitions=[s for scope in scopes for s in scope.setup],
            masks=frozenset(),
        )
        (row,) = items
        assert isinstance(row, lane_loop_distribution.LanePlacement)
        assert [
            _is_barrier(item) for item in row.items if isinstance(item, ast.AST)
        ] == ([True, False] if barrier else [False]), statement


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flagged_first_column_then_update_all(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if flags[tile0.begin] > 0:
            first = x[tile0, tile1.begin]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_then_flagged_update_all(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        first = x[tile0, tile1.begin]
        if flags[tile0.begin] > 0:
            x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _update_all_or_first_column(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if flags[tile0.begin] > 0:
            x[tile0, tile1] = x[tile0, tile1] + 1.0
        else:
            first = x[tile0, tile1.begin]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None]
    return x


@pytest.mark.parametrize(
    "kernel",
    [_flagged_first_column_then_update_all, _update_all_or_first_column],
    ids=["inside_a_branch", "between_the_branches"],
)
def test_racing_accesses_in_a_grid_body_branch_reject_the_config(
    kernel: object,
) -> None:
    # A ``hl.if`` body is emitted inside the lane nest without passing
    # through it, so each lane's pass re-reads the first column the column
    # loop's lane 0 stored: the exactness check rejects the nest before the
    # barrier pass would find the branch divergent (the flag aliases ``x``,
    # which the kernel writes) and the threads' race unorderable.
    x = torch.empty((8, 256))
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant load of x .* inside a branch"
    ):
        _generate(kernel, (x, _flags_aliasing(x)), **_NESTED_CONFIG)


def test_racing_access_before_a_branch_gets_a_barrier_before_the_branch() -> None:
    # The first column's load precedes the branch holding the per-lane
    # update: the barrier goes before the branch, which every thread reaches.
    args = (torch.empty((8, 256)), torch.empty(8))
    code = _generate(_first_column_then_flagged_update_all, args, **_NESTED_CONFIG)
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    (column_loop,) = _loops(row_loop, "lane_1")
    assert len(_barriers(function)) == 1, code
    load = _index_of(row_loop.body, "tile_offset_1")
    barrier = next(i for i, s in enumerate(row_loop.body) if _is_barrier(s))
    assert load < barrier < row_loop.body.index(column_loop), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_first_column_then_update_all(x: torch.Tensor, rows: int) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if tile0.begin < rows:
            first = x[tile0, tile1.begin]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_update_all_or_first_column(x: torch.Tensor, rows: int) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if tile0.begin < rows:
            x[tile0, tile1] = x[tile0, tile1] + 1.0
        else:
            first = x[tile0, tile1.begin]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None]
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_inner_first_column_then_update_all(
    x: torch.Tensor, rows: int
) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        if tile0.begin < rows:
            for tile1 in hl.tile(x.size(1)):
                first = x[tile0, tile1.begin]
                x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_then_flagged_first_column_then_update_all(
    x: torch.Tensor, flags: torch.Tensor, rows: int
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if tile0.begin < rows:
            if flags[tile0.begin] > 0:
                first = x[tile0, tile1.begin]
                x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


# One thread per element: the grid body has no lane loop.
_ELEMENTWISE_THREADS_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [4, 64],
    "cute_vector_widths": [1, 1],
}


def _branches(node: ast.AST) -> list[ast.If]:
    return [child for child in ast.walk(node) if isinstance(child, ast.If)]


def test_racing_accesses_inside_a_uniform_branch_get_a_barrier_inside_it() -> None:
    # ``tile0.begin < rows`` is one value per CTA (a grid tile's begin and a
    # kernel argument), so every thread takes the branch or none does and the
    # barrier between the first column's load and the update goes inside it.
    code = _generate(
        _bounded_first_column_then_update_all,
        (torch.empty((8, 256)), 4),
        **_ELEMENTWISE_THREADS_CONFIG,
    )
    function = _kernel_function(code)
    (branch,) = _branches(function)
    assert len(_barriers(function)) == len(_barriers(branch)) == 1, code
    _one_barrier_between(branch.body, "tile_offset_1) *", ".store(")


def test_two_sides_of_a_uniform_branch_are_not_ordered_against_each_other() -> None:
    # The CTA runs one side of the branch, so only the ``else`` side's own
    # race needs a barrier; the plain update on the other side races with
    # nothing.
    code = _generate(
        _bounded_update_all_or_first_column,
        (torch.empty((8, 256)), 4),
        **_ELEMENTWISE_THREADS_CONFIG,
    )
    function = _kernel_function(code)
    (branch,) = _branches(function)
    assert len(_barriers(function)) == 1 and not _barriers(branch.body[0]), code
    _one_barrier_between(branch.orelse, "tile_offset_1) *", ".store(")


def test_device_loop_inside_a_uniform_branch_gets_its_barrier() -> None:
    # The device loop's own race is ordered inside its body, which every
    # thread runs the same number of times under a uniform branch.
    code = _generate(
        _bounded_inner_first_column_then_update_all,
        (torch.empty((8, 256)), 4),
        **_INNER_THREADED_COLUMNS_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    assert len(_barriers(function)) == 1, code
    _one_barrier_between(device_loop.body, "tile_offset_1) *", ".store(")


# Four rows per thread around one column per thread: the row axis is a lane
# loop, the column axis is shared by the threads.
_ROWS_PER_THREAD_THREADED_COLUMNS_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 1],
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flagged_first_column_update_of_a_fresh_output(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(x.shape):
        out[tile0, tile1] = x[tile0, tile1]
        if flags[tile0.begin] > 0:
            first = out[tile0, tile1.begin]
            out[tile0, tile1] = out[tile0, tile1] + first[:, None] + 1.0
    return out


def test_racing_accesses_in_an_unwritten_flag_branch_get_a_barrier_inside_it() -> None:
    # Every thread loads ``flags`` at the tile's begin and nothing in the
    # kernel can write it (only the fresh ``out`` is written), so the flag is
    # one value per CTA and the first column's race gets its barrier in the
    # branch.
    code = _generate(
        _flagged_first_column_update_of_a_fresh_output,
        (torch.empty((8, 256)), torch.empty(8)),
        **_ELEMENTWISE_THREADS_CONFIG,
    )
    function = _kernel_function(code)
    (branch,) = _branches(function)
    assert len(_barriers(branch)) == 1, code
    _one_barrier_between(branch.body, "tile_offset_1) *", ".store(")


def test_argument_flag_proof_survives_the_bind_tensors_being_freed() -> None:
    # ``flags`` and the in-place ``x`` are both arguments: only the bound
    # kernel's cache-keyed storage-disjointness fact proves the flag
    # unwritten.  The bound kernel holds the bind's tensors weakly; a second
    # config compiled after they are freed must reach the decision the first
    # did, from the recorded fact.
    x = torch.empty((8, 256))
    flags = torch.empty(8)
    configs = (
        _ELEMENTWISE_THREADS_CONFIG,
        {**_ELEMENTWISE_THREADS_CONFIG, "block_sizes": [2, 64], "num_threads": [2, 64]},
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(_flagged_first_column_then_update_all, (x, flags))
        first = bound.to_code(helion.Config.from_dict(configs[0]))
        del x, flags
        gc.collect()
        assert all(ref() is None for ref in bound._runtime_tensor_refs_by_name.values())
        second = bound.to_code(helion.Config.from_dict(configs[1]))
    for code in (first, second):
        (branch,) = _branches(_kernel_function(code))
        assert _barriers(branch), code


def test_uniform_branch_inside_a_lane_loop_gets_its_barrier() -> None:
    # Each row lane's pass runs the branch whole; the first column's load
    # reads the row lane, so it repeats nothing, and the column threads'
    # race in each pass is ordered by a barrier inside the branch.
    code = _generate(
        _bounded_first_column_then_update_all,
        (torch.empty((8, 256)), 4),
        **_ROWS_PER_THREAD_THREADED_COLUMNS_CONFIG,
    )
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    (branch,) = _branches(row_loop)
    assert len(_barriers(function)) == len(_barriers(branch)) == 1, code
    _one_barrier_between(branch.body, "tile_offset_1) *", ".store(")


def test_loaded_branch_inside_a_uniform_one_still_rejects_its_race() -> None:
    # A flag the kernel may overwrite (it aliases ``x``) is a per-thread value,
    # so its branch stays divergent inside a uniform one.
    x = torch.empty((8, 256))
    with pytest.raises(exc.BackendUnsupported, match="race on x inside a branch"):
        _generate(
            _bounded_then_flagged_first_column_then_update_all,
            (x, _flags_aliasing(x), 4),
            **_ELEMENTWISE_THREADS_CONFIG,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flagged_update_all(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if flags[tile0.begin] > 0:
            x[tile0, tile1] = x[tile0, tile1] * 2.0 + 1.0
    return x


# Each thread walks the 16 columns of its row in a lane loop.
_ROW_PER_THREAD_CONFIG = {
    "block_sizes": [4, 16],
    "num_threads": [4, 1],
    "cute_vector_widths": [1, 1],
}


def test_first_column_load_leaves_the_lane_loop_of_its_row() -> None:
    # The first column's load reads no lane: the placement runs it once,
    # before the loop and so before every lane's store.
    code = _generate(
        _first_column_then_update_all,
        (torch.empty((8, 64)),),
        **_ROW_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "lane_1")
    first = _index_of(function.body, "Int32(tile_offset_1) *")
    assert first < function.body.index(lane_loop), code
    assert "Int32(tile_offset_1) *" not in ast.unparse(lane_loop), code


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        pytest.param(
            _flagged_first_column_then_update_all,
            (torch.empty((8, 64)), torch.empty(8)),
            id="loaded_condition",
        ),
        pytest.param(
            _bounded_first_column_then_update_all,
            (torch.empty((8, 64)), 4),
            id="uniform_condition",
        ),
    ],
)
def test_first_column_load_in_a_branch_inside_a_lane_loop_rejects_the_config(
    kernel: object, args: tuple[object, ...]
) -> None:
    # A branch is emitted whole inside the lane loop, so each lane would
    # re-read the first column after lane 0 stored to it: no nest of the
    # branch is the tile program.
    with pytest.raises(
        exc.BackendUnsupported,
        match="lane-invariant load of x would repeat .* inside a branch",
    ):
        _generate(kernel, args, **_ROW_PER_THREAD_CONFIG)  # pyrefly: ignore [bad-argument-type]


def test_per_lane_branch_inside_a_lane_loop_keeps_the_config() -> None:
    # Every statement of the branch reads the lane: repeating the branch per
    # lane runs each lane's own update once.
    code = _generate(
        _flagged_update_all,
        (torch.empty((8, 64)), torch.empty(8)),
        **_ROW_PER_THREAD_CONFIG,
    )
    (lane_loop,) = _loops(_kernel_function(code), "lane_1")
    assert any(isinstance(node, ast.If) for node in ast.walk(lane_loop)), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_update_in_a_while_loop(x: torch.Tensor, iters: int) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        steps = torch.zeros([], device=x.device, dtype=torch.int32)
        while steps < iters:
            first = x[tile0, tile1.begin]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
            steps = steps + 1
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_reads_in_a_device_loop(
    x: torch.Tensor, y: torch.Tensor, reps: int
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        for rep in hl.grid(reps):
            first = x[tile0, tile1.begin]
            y[tile0, tile1] = first[:, None] + rep
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _first_column_reads_in_a_device_loop_then_update_all(
    x: torch.Tensor, y: torch.Tensor, reps: int
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        for rep in hl.grid(reps):
            first = x[tile0, tile1.begin]
            y[tile0, tile1] = first[:, None] + rep
        x[tile0, tile1] = x[tile0, tile1] + 1.0
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_first_column_reads_in_a_device_loop_then_update_all(
    x: torch.Tensor, y: torch.Tensor, rows: int, reps: int
) -> torch.Tensor:
    for tile0, tile1 in hl.tile(x.shape):
        if tile0.begin < rows:
            for rep in hl.grid(reps):
                first = x[tile0, tile1.begin]
                y[tile0, tile1] = first[:, None] + rep
            x[tile0, tile1] = x[tile0, tile1] + 1.0
    return y


@pytest.mark.parametrize(
    ("kernel", "args", "reason"),
    [
        pytest.param(
            _first_column_update_in_a_while_loop,
            (torch.empty((8, 64)), 1),
            "lane-invariant load of x in a loop repeated whole once per lane_1 "
            "meets a per-lane store to it",
            id="while_loop",
        ),
        pytest.param(
            _first_column_reads_in_a_device_loop_then_update_all,
            (torch.empty((8, 64)), torch.empty((8, 64)), 2),
            "lane-invariant load of x would repeat .* from a loop repeated whole",
            id="device_loop_then_store",
        ),
        pytest.param(
            _bounded_first_column_reads_in_a_device_loop_then_update_all,
            (torch.empty((8, 64)), torch.empty((8, 64)), 4, 2),
            "lane-invariant load of x would repeat .* from a loop repeated whole",
            id="device_loop_then_store_in_a_branch",
        ),
    ],
)
def test_first_column_load_in_a_loop_inside_a_lane_loop_rejects_the_config(
    kernel: object, args: tuple[object, ...], reason: str
) -> None:
    # The loop runs whole once per lane of the row, so each lane's pass
    # would re-read the first column: after the store in the loop (a
    # ``while`` loop is a loop too) or after the update following the loop,
    # which lane 0 has already applied.
    with pytest.raises(exc.BackendUnsupported, match=reason):
        _generate(kernel, args, **_ROW_PER_THREAD_CONFIG)  # pyrefly: ignore [bad-argument-type]


def test_first_column_load_in_a_device_loop_alone_keeps_the_config() -> None:
    # Nothing stores to x: each lane's pass of the device loop re-reads the
    # same first column.
    code = _generate(
        _first_column_reads_in_a_device_loop,
        (torch.empty((8, 64)), torch.empty((8, 64)), 2),
        **_ROW_PER_THREAD_CONFIG,
    )
    (lane_loop,) = _loops(_kernel_function(code), "lane_1")
    (device_loop,) = _loops(lane_loop, "tile_offset_2")
    assert "Int32(tile_offset_1) *" in ast.unparse(device_loop), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _if_tile_id_and_end(x: torch.Tensor, skip: int) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        if tile0.id != skip and tile0.end <= x.size(0):
            x[tile0, :] = x[tile0, :] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _if_loaded_flag(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        if flags[tile0.begin] > 0:
            x[tile0, :] = x[tile0, :] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _if_device_loop_tile(x: torch.Tensor) -> torch.Tensor:
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            if tile1.begin == 0:
                x[tile0, tile1] = x[tile0, tile1] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _if_loaded_flag_into_fresh_output(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0 in hl.tile(x.size(0)):
        if flags[tile0.begin] > 0:
            out[tile0, :] = x[tile0, :] + 1.0
    return out


@pytest.mark.parametrize(
    ("kernel", "args", "uniform"),
    [
        pytest.param(
            _if_tile_id_and_end, (torch.empty(8, 256), 1), True, id="grid_tile"
        ),
        pytest.param(
            _if_loaded_flag_into_fresh_output,
            (torch.empty(8, 256), torch.empty(8)),
            True,
            id="unwritten_loaded_value",
        ),
        pytest.param(
            _if_loaded_flag,
            (torch.empty(8, 256), torch.empty(8)),
            True,
            id="argument_flag_proven_apart_at_runtime",
        ),
        pytest.param(
            _if_loaded_flag,
            (_aliased := torch.empty(8, 256), _flags_aliasing(_aliased)),
            False,
            id="flag_aliasing_a_written_argument",
        ),
        pytest.param(
            _if_device_loop_tile, (torch.empty(8, 256),), False, id="device_loop_tile"
        ),
    ],
)
def test_block_uniform_proves_only_cta_uniform_conditions(
    kernel: object, args: tuple[object, ...], uniform: bool
) -> None:
    # A grid tile's id and end against a kernel argument and a tensor size
    # are one value per CTA, and so is a load of ``flags`` at the tile's
    # begin while the kernel writes only a fresh allocation, or writes the
    # argument ``x`` that this launch's storages (the bound kernel's
    # cache-keyed disjointness fact) keep apart from ``flags``.  Passed as a
    # view of ``x`` the flag may change under the threads; a device loop's
    # tile is outside the proof (it fails closed).
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
    host_function = bound.host_function
    assert host_function is not None
    (condition,) = {
        node.args[0]
        for graph in host_function.device_ir.graphs
        # The rolled-reduction copy of a branch takes its condition as a
        # placeholder, which the proof never accepts.
        if not isinstance(graph, ReductionLoopGraphInfo)
        for node in graph.graph.nodes
        if node.target is _tracing_ops._if
    }
    with bound.env, host_function:
        assert block_uniform(condition) is uniform


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_with_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc + bias[tile_n]
    return out


def _statement_lists(function: ast.FunctionDef) -> list[list[ast.stmt]]:
    lists: list[list[ast.stmt]] = []
    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.For, ast.If, ast.While)):
            lists.append(node.body)
        if isinstance(node, (ast.For, ast.If, ast.While)) and node.orelse:
            lists.append(node.orelse)
    return lists


@pytest.mark.parametrize("subtile", [2, 4])
def test_epilogue_subtile_staging_through_shared_memory_keeps_the_nest(
    subtile: int,
) -> None:
    # The epilogue subtile stages each value in a shared-memory view built in
    # the body (``split_smem[...] = value``, a barrier, two loads).  The
    # bias's staging reads the column lane only, so the row loop repeats it:
    # a plain store of the view, re-applied unchanged ahead of the loads
    # reading it, not a statement depending on its own result.
    args = (torch.empty(128, 128), torch.empty(128, 128), torch.empty(128))
    code = _generate(
        _matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=subtile
    )
    function = _kernel_function(code)
    _defined_before_use(function)
    assert code.count(".store(") == subtile, code
    views = sorted(
        {
            node.value.id
            for node in ast.walk(function)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id.startswith("split_smem")
        }
    )
    assert views, code
    for view in views:
        (body,) = [
            statements
            for statements in _statement_lists(function)
            if any(ast.unparse(s).startswith(f"{view}[") for s in statements)
        ]
        store = _index_of(body, f"{view}[")
        barrier = _index_of(body, "cute.arch.sync_threads()", after=store)
        load = _index_of(body, f"= {view}[", after=store)
        assert store < barrier < load, (view, [ast.unparse(s)[:60] for s in body])


@pytest.mark.parametrize(
    ("statement", "read", "written", "defines"),
    [
        pytest.param("smem[indices_1] = v", set(), {"smem"}, False, id="store"),
        pytest.param("smem[indices_1] += v", {"smem"}, {"smem"}, False, id="update"),
        pytest.param("w = smem[indices_1]", {"smem"}, set(), False, id="load"),
        pytest.param(
            "if mask_1:\n    smem[indices_1] = v",
            set(),
            {"smem"},
            False,
            id="guarded_store",
        ),
        pytest.param(
            "smem = cute.make_tensor(smem_ptr, (64,))",
            set(),
            set(),
            True,
            id="definition",
        ),
    ],
)
def test_subscript_assignment_is_an_access_of_the_local_tensor(
    statement: str, read: set[str], written: set[str], defines: bool
) -> None:
    # The subscripted name stays a write (its stored values reach the later
    # loads of the tensor through it) but is set apart from a definition.
    analyzed = lane_loop_distribution._analyze(0, ast.parse(statement).body[0], {})
    assert set(analyzed.tensors_read) == read
    assert set(analyzed.tensors_written) == written
    assert ("smem" in analyzed.writes) == (defines or bool(written))
    assert ("smem" in analyzed.subscripted) == (not defines and bool(written))
    assert ("smem" in analyzed.reads) == (not defines)


_STAGING = """\
smem_ptr = cute.arch.alloc_smem(cutlass.Float32, 64)
smem = cute.make_tensor(smem_ptr, (64,))
smem[indices_1] {op} c
cute.arch.sync_threads()
w = smem[indices_1 * 2]
(out.iterator + indices_0 * s0 + indices_1 * s1).store(w)
"""


@pytest.mark.parametrize(
    ("op", "accepted"), [("=", True), ("+=", False)], ids=["store", "update"]
)
def test_staging_repeated_by_the_row_loop_is_exact_only_as_a_plain_store(
    op: str, accepted: bool
) -> None:
    # Rule level: the staging of a column value in a shared-memory view built
    # in the body reads the column lane only, and the full nest repeats it
    # once per row lane.  A plain store re-applies the same value ahead of
    # the loads reading it; an augmented one accumulates once per row.
    body: list[ast.AST] = list(ast.parse(_STAGING.format(op=op)).body)
    scopes = _threaded_column_scopes()
    if accepted:
        lane_loop_distribution.check_full_nest(body, scopes, rename_groups={})
        return
    with pytest.raises(
        exc.BackendUnsupported,
        match="depends on its own result and would repeat once per lane_0",
    ):
        lane_loop_distribution.check_full_nest(body, scopes, rename_groups={})


@pytest.mark.parametrize(
    ("branch", "accepted"),
    [
        pytest.param(
            f"    loaded = {_pointer('scale', 'indices_1')}.load()\n"
            "    column_scale = loaded * 2.0\n",
            True,
            id="branch_temporary",
        ),
        pytest.param("    column_scale = column_scale * 2.0\n", False, id="carried"),
    ],
)
def test_repeated_if_depends_only_on_values_from_before_it(
    branch: str, accepted: bool
) -> None:
    # The full nest repeats the column-only if once per row lane.  A name a
    # branch assigns before reading it is a temporary of that run; reading
    # the value an earlier run assigned compounds once per row.
    body: list[ast.AST] = list(
        ast.parse(
            f"if flag:\n{branch}"
            f"{_pointer('out', 'indices_0', 'indices_1')}.store(column_scale)\n"
        ).body
    )
    scopes = _threaded_column_scopes()
    if accepted:
        lane_loop_distribution.check_full_nest(body, scopes, rename_groups={})
        return
    with pytest.raises(
        exc.BackendUnsupported,
        match="depends on its own result and would repeat once per lane_0",
    ):
        lane_loop_distribution.check_full_nest(body, scopes, rename_groups={})


_REPEATED_LOOP = """\
for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(256), cutlass.Int32(_BLOCK_SIZE_2)):
    indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[0])
{body}
"""
_REPEATED = "in a loop repeated whole once per lane_0 meets"


@pytest.mark.parametrize(
    ("body", "rejected"),
    [
        pytest.param(
            f"v = {_pointer('x', 'tile_offset_0', 'tile_offset_2')}.load()\n"
            f"w = {_pointer('out', 'indices_0', 'indices_2')}.load()\n"
            f"{_pointer('out', 'indices_0', 'indices_2')}.store(w + v)\n"
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2')}.store(v + 1.0)\n",
            f"a lane-invariant load of x {_REPEATED} another store to it",
            id="invariant_read_modify_write",
        ),
        pytest.param(
            f"v = {_pointer('x', 'tile_offset_0', 'tile_offset_2')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)\n",
            f"a lane-invariant load of x {_REPEATED} a per-lane store to it",
            id="invariant_read_then_per_lane_store",
        ),
        pytest.param(
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2')}.store(c)\n"
            f"w = {_pointer('x', 'indices_0', 'indices_2')}.load()\n"
            f"{_pointer('out', 'indices_0', 'indices_2')}.store(w)\n",
            f"a lane-invariant store to x {_REPEATED} a per-lane load of it",
            id="invariant_store_then_per_lane_load",
        ),
        pytest.param(
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2')}.store(c)\n"
            f"{_pointer('out', 'indices_0', 'indices_2')}.store(c)\n"
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2 + 1')}.store(c)\n",
            None,
            id="two_invariant_stores",
        ),
        pytest.param(
            f"v = {_pointer('x', 'tile_offset_0', 'tile_offset_2')}.load()\n"
            f"{_pointer('out', 'indices_0', 'indices_2')}.store(v)\n",
            None,
            id="invariant_read_of_a_tensor_never_stored",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'tile_offset_2')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)\n",
            None,
            id="per_lane_read_then_per_lane_store",
        ),
    ],
)
def test_device_loop_repeated_by_a_lane_loop_is_checked_inside(
    body: str, rejected: str | None
) -> None:
    # A device loop in the body of a row lane loop runs whole once per lane
    # (its per-lane accesses read the lane's row).  Its lane-invariant
    # accesses are repeated with every iteration of it between two
    # repetitions: the second pass's load of x reads what the first pass
    # stored, in this iteration or a later one, and its store lands on what
    # the first pass loaded per lane.  Stores re-applied unchanged, loads of a
    # tensor the loop never stores and per-lane accesses survive.
    (loop,) = ast.parse(_REPEATED_LOOP.format(body=textwrap.indent(body, "    "))).body
    scopes = [_threaded_column_scopes()[0]]
    if rejected is None:
        lane_loop_distribution.check_full_nest([loop], scopes, rename_groups={})
        return
    with pytest.raises(exc.BackendUnsupported, match=re.escape(rejected)):
        lane_loop_distribution.check_full_nest([loop], scopes, rename_groups={})


# ---------------------------------------------------------------------------
# Accesses of consecutive loop iterations, gathered addresses, slices apart.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _inner_column_zero_then_update_all(x: torch.Tensor) -> torch.Tensor:
    # Column 0 is read in every iteration of the inner loop and updated by its
    # first: each iteration's read has to see the previous iteration's stores.
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(x.size(1)):
            first = x[tile0, 0]
            x[tile0, tile1] = x[tile0, tile1] + first[:, None] + 1.0
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_then_zero(
    x: torch.Tensor, perm: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The gathered read of another thread's element precedes every thread's
    # store to its own.
    for tile0, tile1 in hl.tile(x.shape):
        y[tile0, tile1] = x[tile0, perm[tile1]]
        x[tile0, tile1] = 0.0
    return y


def test_loop_invariant_read_in_a_device_loop_gets_a_barrier_closing_the_body() -> None:
    # Within an iteration the barrier before the store keeps every thread's
    # read of column 0 ahead of thread 0's store to it; the one closing the
    # body keeps that store ahead of the next iteration's reads.  A read of
    # the tile's own first column (``_inner_first_column_then_update_all``)
    # changes with the iteration and needs only the first.
    code = _generate(
        _inner_column_zero_then_update_all,
        (torch.empty((8, 256)),),
        **_INNER_THREADED_COLUMNS_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    assert len(_barriers(function)) == 2, code
    positions = [i for i, s in enumerate(device_loop.body) if _is_barrier(s)]
    load = _index_of(device_loop.body, "cutlass.Int32(0) *")
    store = _index_of(device_loop.body, ".store(")
    assert load < store, code
    assert positions == [store - 1, len(device_loop.body) - 1], code


_GATHER_ARGS = (
    torch.empty((8, 256)),
    torch.empty(256, dtype=torch.int64),
    torch.empty((8, 256)),
)
# One thread per element on both axes: no lane loop.
_THREADED_TILE_CONFIG = {
    "block_sizes": [2, 128],
    "num_threads": [2, 128],
    "cute_vector_widths": [1, 1],
}


def test_gathered_read_before_a_per_lane_store_gets_a_barrier() -> None:
    # A gathered address is any thread's element: the barrier between the
    # gather and the store keeps every thread's gather ahead of any thread's
    # store.  Without lane loops the body is emitted as it is, and gets the
    # barrier all the same.
    code = _generate(_gather_then_zero, _GATHER_ARGS, **_THREADED_TILE_CONFIG)
    function = _kernel_function(code)
    assert not _loops(function, "lane_"), code
    assert len(_barriers(function)) == 1, code
    _one_barrier_between(
        function.body, "cutlass.Int32(load)", ".store(cutlass.Float32(0.0))"
    )


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_gathered_read_in_a_lane_loop_with_a_per_lane_store_rejects_the_config(
    config: dict[str, object],
) -> None:
    # The program gathers every lane before it stores any; in one lane loop
    # the gather of lane i + 1 follows the store of lane i, which may have
    # zeroed the element it reads (the reversed column of another thread).
    # No barrier orders that, so the config fails closed.
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _generate(_gather_then_zero, _GATHER_ARGS, **config)


# The study's two ``concat2d_dim1_simple`` configs (worst20, 2026-09-28).
_STUDY_SHAPE1_CONFIG = {
    "block_sizes": [1],
    "num_threads": [0, 4, 32],
    "reduction_loops": [256],
    "load_eviction_policies": ["last", "last", "l1_l2_first", "l2_last"],
    "cute_proven_bounds": True,
    "cute_independent_reduction": False,
    "cute_replicated_reduction": False,
    "cute_vector_packet_unroll": False,
    "cute_vector_widths": [4, 2, 8],
    "cute_lane_layouts": ["blocked", "blocked", "blocked"],
    "cute_reduction_reloads": ["gmem", "register"],
    "cute_cluster_n": 2,
    "cute_min_blocks_per_mp": 0,
}
_STUDY_SHAPE2_CONFIG = {
    "block_sizes": [1],
    "num_threads": [1, 4, 0],
    "reduction_loops": [512],
    "load_eviction_policies": ["streaming", "l1_l2_last", "l1_l2_last", "l1_l2_last"],
    "cute_proven_bounds": True,
    "cute_independent_reduction": False,
    "cute_replicated_reduction": False,
    "cute_vector_packet_unroll": False,
    "cute_vector_widths": [4, 8, 8],
    "cute_lane_layouts": ["strided", "blocked", "blocked"],
    "cute_reduction_reloads": ["register", "auto"],
    "cute_cluster_n": 2,
    "cute_min_blocks_per_mp": 1,
}


@pytest.mark.parametrize(
    ("shape", "config"),
    [
        ((2048, 512, 768), _STUDY_SHAPE1_CONFIG),
        ((8192, 1024, 2048), _STUDY_SHAPE2_CONFIG),
    ],
    ids=["shape1", "shape2"],
)
def test_concat_simple_study_configs_need_no_barrier(
    shape: tuple[int, int, int], config: dict[str, object]
) -> None:
    # The two copies store ``out[t, :n1]`` and ``out[t, n1:]``, each per-thread
    # along an axis the other ignores; the recorded regions of the two
    # slices are apart, so no order between them is observable.
    m, n1, n2 = shape
    code = _generate(
        _simple_kernel(), (torch.empty((m, n1)), torch.empty((m, n2))), **config
    )
    assert not _barriers(_kernel_function(code)), code


def _kinds(items: list[ast.AST | lane_loop_distribution.LanePlacement]) -> list[str]:
    return [
        "loop"
        if isinstance(item, lane_loop_distribution.LanePlacement)
        else "sync"
        if _is_barrier(item)
        else "stmt"
        for item in items
    ]


def _lane_nest(
    body: str,
) -> tuple[list[ast.AST | lane_loop_distribution.LanePlacement], list[ast.AST]]:
    scopes = _threaded_column_scopes()
    column = lane_loop_distribution.LanePlacement("lane_1", list(ast.parse(body).body))
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [
        lane_loop_distribution.LanePlacement("lane_0", [column])
    ]
    lane_loop_distribution.add_thread_barriers(
        items,
        scopes,
        rename_groups={},
        axis_sizes={0: 64},
        definitions=[statement for scope in scopes for statement in scope.setup],
        masks=frozenset(),
    )
    return items, column.items  # pyrefly: ignore [bad-return]


def _gather_body(column: str) -> str:
    body = (
        f"load = {_pointer('perm', 'indices_1')}.load()\n"
        "column = indices_1 * 1\n"
        f"w = {_pointer('x', 'indices_0', column)}.load()\n"
        f"{_pointer('x', 'indices_0', 'indices_1')}.store(w)\n"
    )
    if column == "column":
        return body.replace(f"load = {_pointer('perm', 'indices_1')}.load()\n", "")
    return body.replace("column = indices_1 * 1\n", "")


def test_gathered_address_rules() -> None:
    # A column index loaded from memory is any thread's; one computed from
    # the thread's own index is its own.  In a lane loop the gather of one
    # lane conflicts with the per-lane store of another and no barrier
    # orders them as the program does; without lane loops the barrier
    # between the gather and the store does.
    _items, emitted = _lane_nest(_gather_body("column"))
    assert _kinds(emitted) == ["stmt", "stmt", "stmt"], emitted  # pyrefly: ignore [bad-argument-type]
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _lane_nest(_gather_body("cutlass.Int32(load)"))
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(_gather_body("cutlass.Int32(load)")).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64},
        definitions=ast.parse(
            "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])"
        ).body,
        masks=frozenset(),
    )
    assert _kinds(items) == ["stmt", "stmt", "sync", "stmt"], items  # pyrefly: ignore [bad-argument-type]


_ROW_DEFINITIONS = (
    "indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
    "indices_1 = cutlass.Int32(cute.arch.thread_idx()[0])\n"
)


@pytest.mark.parametrize(
    ("body", "loop_variable", "expected"),
    [
        pytest.param(
            f"v = {_pointer('x', 'indices_0', '0')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)\n",
            "tile_offset_1",
            ["stmt", "sync", "stmt", "sync"],
            id="invariant_read_then_store",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'tile_offset_1')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)\n",
            "tile_offset_1",
            ["stmt", "sync", "stmt", "sync"],
            id="varying_read_then_store",
        ),
        pytest.param(
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(c)\n"
            f"v = {_pointer('x', 'indices_0', '0')}.load()\n",
            "tile_offset_1",
            ["stmt", "sync", "stmt", "sync"],
            id="store_then_invariant_read",
        ),
        pytest.param(
            f"c = {_pointer('cols', '0')}.load()\n"
            f"{_pointer('x', 'indices_0', 'c')}.store(v)\n"
            f"w = {_pointer('x', 'indices_0', 'indices_1')}.load()\n",
            "tile_offset_0",
            ["stmt", "stmt", "stmt", "sync"],
            id="gathered_column_in_a_varying_row",
        ),
    ],
)
def test_device_loop_iteration_rules(
    body: str, loop_variable: str, expected: list[str]
) -> None:
    # The body of a device loop whose columns are one per x thread and rows
    # one per y thread.  A read whose address ignores the loop variable meets
    # the previous iteration's per-thread stores unless a barrier closes the
    # body.  Without recorded regions, so does a read whose address moves
    # with the iteration: ``tile_offset_1`` and ``tile_offset_1 - 1`` (the
    # previous tile's last column, which the previous iteration stored) look
    # alike to the pass; the regions decide (the next test).  The uniform
    # store of a gathered column beside per-thread loads is benign within an
    # iteration (each thread's own store precedes its load) but, without
    # regions, not across iterations.
    definitions = ast.parse(
        _ROW_DEFINITIONS
        + "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
    ).body
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(body).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=definitions,
        masks=frozenset(),
        loop_variables=frozenset({loop_variable}),
        loop_body=items,
    )
    assert _kinds(items) == expected, [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]


def _tagged(text: str, regions: tuple[object, ...]) -> ast.AST:
    """The statement ``text`` with ``regions`` recorded on its load or store call."""
    statement = ast.parse(text).body[0]
    call = next(
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "store")
    )
    setattr(call, HELION_ACCESS_REGIONS_ATTR, regions)
    return statement


_ROWS = sympy.Symbol("_helion_tile_begin_0", integer=True)
_ROW_BLOCK = sympy.Symbol("_BLOCK_SIZE_0", integer=True, positive=True)
_BEGIN = sympy.Symbol("_helion_tile_begin_1", integer=True)
_BLOCK = sympy.Symbol("_BLOCK_SIZE_1", integer=True, positive=True)


def _columns(begin: sympy.Expr, end: sympy.Expr) -> tuple[object, ...]:
    """The region of one tile of rows and the columns ``[begin, end)``."""
    return ((_ROWS, sympy.Add(_ROWS, _ROW_BLOCK)), (begin, end))


@pytest.mark.parametrize(
    ("load_columns", "shifts", "expected"),
    [
        pytest.param(
            (_BEGIN, sympy.Add(_BEGIN, 1)),
            {_BEGIN: _BLOCK},
            ["stmt", "sync", "stmt"],
            id="first_column",
        ),
        pytest.param(
            (sympy.Add(_BEGIN, -1), _BEGIN),
            {_BEGIN: _BLOCK},
            ["stmt", "stmt", "sync"],
            id="previous_column",
        ),
        pytest.param(
            (sympy.Add(_BEGIN, _BLOCK), sympy.Add(_BEGIN, _BLOCK, 1)),
            {_BEGIN: _BLOCK},
            ["stmt", "stmt", "sync"],
            id="next_tile_first_column",
        ),
        pytest.param(
            (_BEGIN, sympy.Add(_BEGIN, 1)),
            {_BEGIN: None},
            ["stmt", "sync", "stmt", "sync"],
            id="first_column_of_a_two_block_nest",
        ),
        pytest.param(
            (_BEGIN, sympy.Add(_BEGIN, 1)),
            None,
            ["stmt", "sync", "stmt", "sync"],
            id="first_column_of_an_unknown_loop",
        ),
    ],
)
def test_device_loop_iteration_regions(
    load_columns: tuple[sympy.Expr, sympy.Expr],
    shifts: dict[sympy.Symbol, sympy.Expr | None] | None,
    expected: list[str],
) -> None:
    # A uniform read then a per-thread store of one tile, with the regions
    # the codegen records.  The loop advances its begin by the block size:
    # the read of the tile's first column never meets a later iteration's
    # store (no closing barrier), the read of the previous tile's last column
    # is the previous iteration's store (a closing barrier, the within-
    # iteration one being unnecessary), and the read of the next tile's first
    # column is overwritten by the next iteration's store.  A nest over
    # several blocks, or a loop the pass is not told about, proves nothing.
    load = _tagged(
        f"v = {_pointer('x', 'indices_0', 'tile_offset_1')}.load()",
        _columns(*load_columns),
    )
    store = _tagged(
        f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)",
        _columns(_BEGIN, sympy.Add(_BEGIN, _BLOCK)),
    )
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [load, store]
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(
            _ROW_DEFINITIONS
            + "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_shifts=shifts,
    )
    assert _kinds(items) == expected, [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]


def _first_row_columns(begin: sympy.Expr, end: sympy.Expr) -> tuple[object, ...]:
    """The region of the tile's first row and the columns ``[begin, end)``."""
    return ((_ROWS, sympy.Add(_ROWS, 1)), (begin, end))


# Accesses whose column moves back by a block per iteration: the uniform read
# of column 2100 - begin, the uniform store to column block + 2164 - begin
# (iteration j + 1 + 64 / block stores the column iteration j read), the
# per-thread store to column 2100 - begin and the uniform read two blocks
# ahead of it (iteration j + 2 reads the column iteration j stored).
_BACKWARDS_READ = (
    f"v = {_pointer('x', 'tile_offset_0', '2100 - tile_offset_1')}.load()",
    _first_row_columns(sympy.Add(2100, -_BEGIN), sympy.Add(2101, -_BEGIN)),
)
_BACKWARDS_STORE = (
    (
        f"{_pointer('x', 'tile_offset_0', '_BLOCK_SIZE_1 + 2164 - tile_offset_1')}"
        ".store(c)"
    ),
    _first_row_columns(
        sympy.Add(_BLOCK, 2164, -_BEGIN), sympy.Add(_BLOCK, 2165, -_BEGIN)
    ),
)
_BACKWARDS_PER_THREAD_STORE = (
    f"{_pointer('x', 'indices_0', '2100 - tile_offset_1')}.store(c)",
    _columns(sympy.Add(2100, -_BEGIN), sympy.Add(2101, -_BEGIN)),
)
_BACKWARDS_READ_AHEAD = (
    (
        f"v = {_pointer('x', 'tile_offset_0', '2 * _BLOCK_SIZE_1 + 2100 - tile_offset_1')}"
        ".load()"
    ),
    _first_row_columns(
        sympy.Add(sympy.Mul(2, _BLOCK), 2100, -_BEGIN),
        sympy.Add(sympy.Mul(2, _BLOCK), 2101, -_BEGIN),
    ),
)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        pytest.param(
            _BACKWARDS_READ,
            _BACKWARDS_STORE,
            ["stmt", "stmt", "sync"],
            id="read_then_store",
        ),
        pytest.param(
            _BACKWARDS_STORE,
            _BACKWARDS_READ,
            ["stmt", "stmt", "sync"],
            id="store_then_read",
        ),
        pytest.param(
            _BACKWARDS_PER_THREAD_STORE,
            _BACKWARDS_READ_AHEAD,
            ["stmt", "stmt", "sync"],
            id="per_thread_store_then_read",
        ),
    ],
)
def test_device_loop_accesses_moving_backwards_close_the_body(
    first: tuple[str, tuple[object, ...]],
    second: tuple[str, tuple[object, ...]],
    expected: list[str],
) -> None:
    # The two regions are apart within an iteration (no barrier between the
    # statements) and the second access, advanced by one step, lies past the
    # first -- but its column comes back towards the first's by a block per
    # iteration, so one step proves nothing about the next: the read of
    # iteration j meets the store of a later one unless a barrier closes the
    # body.  With the store first the pair (store, later read) is benign
    # (each thread's own store precedes its read) and the pair (read, later
    # store) still closes the body; the per-thread store beside the uniform
    # read ahead of it likewise.
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [
        _tagged(*first),
        _tagged(*second),
    ]
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(
            _ROW_DEFINITIONS
            + "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_shifts={_BEGIN: _BLOCK},
    )
    assert _kinds(items) == expected, [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]


def _inner_loop(column: str) -> str:
    return textwrap.dedent(
        f"""
        for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(_BLOCK_SIZE_2)):
            indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[0])
            v = {_pointer("x", "indices_0", column)}.load()
            w = {_pointer("x", "indices_0", "indices_2")}.load()
            cute.arch.sync_threads()
            {_pointer("x", "indices_0", "indices_2")}.store(v + w)
        """
    )


@pytest.mark.parametrize(
    ("column", "block_size", "expected"),
    [
        pytest.param("tile_offset_2", 64, ["stmt", "sync"], id="uniform_read"),
        pytest.param("indices_2", 64, ["stmt"], id="per_thread_read"),
        pytest.param(
            "indices_2", 32, ["stmt", "sync"], id="per_thread_read_narrow_block"
        ),
    ],
)
def test_nested_loop_meets_itself_across_the_outer_iterations(
    column: str, block_size: int, expected: list[str]
) -> None:
    # An inner device loop is one statement of the outer loop's body.  Its
    # uniform read of a tile's first column in outer iteration j + 1 is the
    # column its per-thread store wrote in iteration j (the inner loop starts
    # over), so the outer body is closed by a barrier; the inner body's own
    # barrier orders only the accesses of one inner iteration.  Reads of the
    # thread's own elements need none: the inner loop steps its columns by
    # the block size, 64 like the thread count, so ``tile_offset_2 + tid``
    # names one thread's column in every inner iteration.  Stepping by 32,
    # iteration 1's thread 0 and iteration 0's thread 32 share a column,
    # and the outer body is closed too.
    (compound,) = ast.parse(_inner_loop(column)).body
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [compound]
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(
            "indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
            "indices_1 = tile_offset_1\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_shifts={
            sympy.Symbol("_helion_tile_begin_1", integer=True): sympy.Integer(1)
        },
        constants={"_BLOCK_SIZE_2": block_size},
    )
    assert _kinds(items) == expected, [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]
    assert len(_barriers(compound)) == 1


def _inner_loop_with_helper(read: str, store: str) -> str:
    return textwrap.dedent(
        f"""
        for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(_BLOCK_SIZE_2)):
            indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[0])
            {read}
            cute.arch.sync_threads()
            {store}
        """
    )


@pytest.mark.parametrize(
    ("read", "store", "expected"),
    [
        pytest.param(
            f"v = _cute_mystery_load({_pointer('x', 'indices_0', 'tile_offset_2')}, 4)",
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)",
            ["stmt", "sync"],
            id="helper_read",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'tile_offset_2')}.load()",
            f"_cute_mystery_store({_pointer('x', 'indices_0', 'indices_2')}, v)",
            ["stmt", "sync"],
            id="helper_store",
        ),
        pytest.param(
            f"v = _cute_mystery_load({_pointer('x', 'indices_0', 'tile_offset_2')}, 4)",
            f"_cute_mystery_store({_pointer('x', 'indices_0', 'indices_2')}, v)",
            ["stmt"],
            id="helper_only",
        ),
    ],
)
def test_nested_loop_meets_itself_through_a_helper_call(
    read: str, store: str, expected: list[str]
) -> None:
    # A call the pass cannot see through reads and writes the tensors it
    # names however it likes.  Beside a visible load or store of x in the
    # same inner loop it is one more access of x, and the loop meets itself
    # across the outer iterations like a loop of plain accesses (the closing
    # barrier); a tensor the loop touches through such calls only is the
    # call's own business against its next instance (an emitter's TMA copy
    # or MMA issue loop synchronizes its own pipeline).
    (compound,) = ast.parse(_inner_loop_with_helper(read, store)).body
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [compound]
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(
            "indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
            "indices_1 = tile_offset_1\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_shifts={
            sympy.Symbol("_helion_tile_begin_1", integer=True): sympy.Integer(1)
        },
    )
    assert _kinds(items) == expected, [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]


def _single_thread_column_scopes() -> list[lane_loop_distribution.LaneScope]:
    """A row lane loop and a column lane loop, both on one thread."""
    scopes = _threaded_column_scopes()
    return [
        scopes[0],
        dataclasses.replace(
            scopes[1],
            setup=tuple(
                ast.parse("indices_1 = tile_offset_1 + cutlass.Int32(lane_1)").body
            ),
        ),
    ]


def test_gathered_lane_conflict_on_a_single_thread_is_rejected() -> None:
    # The lanes of one lane loop run on one thread in loop order while the
    # program runs every lane of the gather before any lane's store: lane 1
    # gathers what lane 0 stored.  No thread count is involved, so the
    # rejection holds when the axis has a single thread (and no barrier is
    # placed anywhere: none is needed between threads).  A column computed
    # from the lane's own index conflicts with nothing.
    scopes = _single_thread_column_scopes()

    def nest(body: str) -> list[ast.AST]:
        column = lane_loop_distribution.LanePlacement(
            "lane_1", list(ast.parse(body).body)
        )
        items: list[ast.AST | lane_loop_distribution.LanePlacement] = [
            lane_loop_distribution.LanePlacement("lane_0", [column])
        ]
        lane_loop_distribution.add_thread_barriers(
            items,
            scopes,
            rename_groups={},
            axis_sizes={0: 1},
            definitions=[statement for scope in scopes for statement in scope.setup],
            masks=frozenset(),
        )
        return column.items  # pyrefly: ignore [bad-return]

    assert _kinds(nest(_gather_body("column"))) == ["stmt", "stmt", "stmt"]  # pyrefly: ignore [bad-argument-type]
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        nest(_gather_body("cutlass.Int32(load)"))


def _vectorized_column_scope(
    *statements: str, vloop_index: int
) -> lane_loop_distribution.LaneScope:
    """A vectorized column lane loop over 64 threads, four elements per thread.

    ``statements`` are the wrapper's own (the per-thread lane base first;
    the packet loads before the V-loop, the flushes after it).
    """
    return lane_loop_distribution.LaneScope(
        "lane_1",
        frozenset({"lane_1", "vec_lane_1", "lane_base_1", "indices_1"}),
        frozenset(),
        tuple(ast.parse("\n".join(statements)).body),
        frozenset({"lane_1", "vec_lane_1", "lane_base_1"}),
        tuple(ast.parse("indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)").body),
        "lane_1 == 0 and vec_lane_1 == 0",
        vloop_index=vloop_index,
        counts={"lane_1": 1, "vec_lane_1": 4},
    )


def _vectorized_nest(
    scope: lane_loop_distribution.LaneScope, body: str, axis_sizes: dict[int, int]
) -> tuple[list[int], list[str]]:
    """The wrapper's barrier positions and the V-loop body's kinds after the pass."""
    placement = lane_loop_distribution.LanePlacement(
        scope.lane_var, list(ast.parse(body).body)
    )
    lane_loop_distribution.add_thread_barriers(
        [placement],
        [scope],
        rename_groups={},
        axis_sizes=axis_sizes,
        definitions=[*scope.attached, *scope.setup],
        masks=frozenset(),
    )
    return sorted(placement.barriers), _kinds(placement.items)


# One lane of four elements per thread (the lane variable ranges over one
# value, as the codegen spells it).
_LANE_BASE = (
    "lane_base_1 = cutlass.Int32(cute.arch.thread_idx()[0]) * 4"
    " + cutlass.Int32(lane_1) * 4"
)


def test_masked_packet_load_fallback_pointer_is_no_access() -> None:
    # The codegen's masked packet load reads through the lane's pointer or,
    # past the tile's end, a fallback pointer at the row's start whose value
    # the select discards.  The fallback names no thread coordinate; taken
    # as an access of its own it would make the load uniform along the
    # threads and put a barrier before the thread's own stores.
    lane = _pointer("x", "tile_offset_0", "lane_base_1")
    fallback = _pointer("x", "tile_offset_0", "0")
    scope = _vectorized_column_scope(
        _LANE_BASE,
        f"packet = cute.arch.load({lane} if lane_base_1 < 1000 else {fallback},"
        " ir.VectorType.get([4], cutlass.Uint32.mlir_type))",
        vloop_index=2,
    )
    barriers, kinds = _vectorized_nest(
        scope, f"{_pointer('x', 'tile_offset_0', 'indices_1')}.store(v)\n", {0: 64}
    )
    assert (barriers, kinds) == ([], ["stmt"])


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        # Element vl of the packet is the thread's own element vl.
        pytest.param("indices_1", ([], ["stmt", "stmt"]), id="own"),
        # Element 3 of the packet is the next thread's element -1 (the
        # flush's first element stands for none of the others): the barrier
        # goes before the flush, among the wrapper's statements.
        pytest.param("indices_1 - 1", ([2], ["stmt", "stmt"]), id="previous"),
        pytest.param("indices_1 + 4", ([2], ["stmt", "stmt"]), id="next"),
    ],
)
def test_packet_flush_is_every_element_of_the_packet(
    read: str, expected: tuple[list[int], list[str]]
) -> None:
    # A store flush after the V-loop writes four consecutive elements of the
    # thread; a read inside the V-loop of another thread's element among
    # them needs the barrier between the two, which the flush's first
    # element alone would not show.
    flush = f"_cute_store_u32_vec({_pointer('x', 'tile_offset_0', 'lane_base_1')}, a)"
    scope = _vectorized_column_scope(_LANE_BASE, flush, vloop_index=1)
    body = (
        f"w = {_pointer('x', 'tile_offset_0', read)}.load()\n"
        f"{_pointer('y', 'tile_offset_0', 'indices_1')}.store(w)\n"
    )
    barriers, kinds = _vectorized_nest(scope, body, {0: 64})
    assert (barriers, kinds) == expected


def test_packet_outside_a_wrapper_forces_nothing() -> None:
    # A vector load the pass cannot place in a wrapper covers elements it
    # cannot name: beside a store of the tensor it gets the barrier.
    lane = _pointer("x", "tile_offset_0", "lane_base_1")
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(
            f"packet = cute.arch.load({lane},"
            " ir.VectorType.get([4], cutlass.Uint32.mlir_type))\n"
            f"{lane}.store(v)\n"
        ).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64},
        definitions=ast.parse(_LANE_BASE).body,
        masks=frozenset(),
    )
    assert _kinds(items) == ["stmt", "sync", "stmt"]


def _root_items(body: str, definitions: str, axis_sizes: dict[int, int]) -> list[str]:
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(body).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes=axis_sizes,
        definitions=ast.parse(definitions).body,
        masks=frozenset(),
    )
    return _kinds(items)


def test_unknown_call_rules() -> None:
    # A device loop whose body calls a helper of unknown purity is pinned,
    # but its loads and stores are visible: two loops only reading ``x``
    # (``softmax_two_pass``'s passes) need no barrier, and a loop storing
    # each thread's own elements (its index defined from the thread's
    # coordinate inside the loop, its value from the helper) needs none
    # against a later per-thread load.  A tensor handed to the helper itself
    # may be read or written by it.
    pass_over_x = (
        "for tile_offset_3 in range(cutlass.Int32(0), cutlass.Int32(256), cutlass.Int32(64)):\n"
        "    indices_3 = tile_offset_3 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
        f"    values = {_pointer('x', 'indices_0', 'indices_3')}.load()\n"
        "    total = _helper(values)\n"
    )
    store_out = (
        "for tile_offset_3 in range(cutlass.Int32(0), cutlass.Int32(256), cutlass.Int32(64)):\n"
        "    indices_3 = tile_offset_3 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
        "    scaled = _helper(total)\n"
        f"    {_pointer('out', 'indices_0', 'indices_3')}.store(scaled)\n"
    )
    sizes = {0: 64, 1: 2}
    assert _root_items(pass_over_x * 2, _ROW_DEFINITIONS, sizes) == ["stmt", "stmt"]
    assert _root_items(
        store_out + f"w = {_pointer('out', 'indices_0', 'indices_1')}.load()\n",
        _ROW_DEFINITIONS,
        sizes,
    ) == ["stmt", "stmt"]
    assert _root_items(
        f"_helper({_pointer('x', 'indices_0', '0')})\n"
        f"w = {_pointer('x', 'indices_0', 'indices_1')}.load()\n",
        _ROW_DEFINITIONS,
        sizes,
    ) == ["stmt", "sync", "stmt"]


def _tagged_store(text: str, regions: tuple[object, ...]) -> ast.AST:
    statement = ast.parse(text).body[0]
    assert isinstance(statement, ast.Expr)
    setattr(statement.value, HELION_ACCESS_REGIONS_ATTR, regions)
    return statement


@pytest.mark.parametrize(
    ("second_slice", "expected"),
    [
        pytest.param((512, 1280), ["stmt", "stmt"], id="apart"),
        pytest.param((511, 1279), ["stmt", "sync", "stmt"], id="meeting"),
    ],
)
def test_stores_to_recorded_regions_apart_need_no_barrier(
    second_slice: tuple[int, int], expected: list[str]
) -> None:
    # ``concat2d_dim1_simple``'s two copies: a packet store per-thread along
    # x (the persistent slice's flush, whose packet covers more than its
    # first element) and a store per-thread along y, of one row's slices.
    # The regions the codegen records decide whether the slices meet.
    row = sympy.Symbol("_helion_tile_begin_0", integer=True)
    first = _tagged_store(
        f"_cute_store_u32_vec({_pointer('out', 'indices_0', 'rindex_1')}, a)",
        ((row, sympy.Add(row, 1)), (sympy.Integer(0), sympy.Integer(512))),
    )
    second = _tagged_store(
        f"{_pointer('out', 'indices_0', '512 + indices_2')}.store(b)",
        (
            (row, sympy.Add(row, 1)),
            tuple(sympy.Integer(bound) for bound in second_slice),
        ),
    )
    definitions = ast.parse(
        "rindex_1 = cutlass.Int32(cute.arch.thread_idx()[0]) * 4\n"
        "indices_2 = cutlass.Int32(cute.arch.thread_idx()[1])\n"
    ).body
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [first, second]
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 4, 1: 32},
        definitions=definitions,
        masks=frozenset(),
    )
    assert _kinds(items) == expected, [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]


_STAGING_DEFINITIONS = (
    "mma_tidx = cutlass.Int32(cute.arch.thread_idx()[0])"
    " + cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(32)\n"
    "mma_copy_tidx = mma_tidx\n"
    "mma_active = cutlass.Int32(cute.arch.thread_idx()[1]) < cutlass.Int32(2)\n"
)


def _staging_loop(
    tensor: str, source: str, flat: str, elements: int, offset: str = ""
) -> str:
    """The tcgen05 emitter's copy loop staging a ``[rows, 16]`` tile of ``source``.

    ``flat`` is the copying thread's flattened coordinate; ``offset`` is
    added to the row of the element it loads.
    """
    swizzled = (
        f"(_row % cute.size({tensor}.shape[0][0]), "
        f"_col % cute.size({tensor}.shape[0][1])), "
        f"_row // cute.size({tensor}.shape[0][0]), "
        f"_col // cute.size({tensor}.shape[0][1])"
    )
    return (
        f"for _load_i in range(({elements} + 64 - 1) // 64):\n"
        f"    _flat = {flat} + cutlass.Int32(_load_i) * cutlass.Int32(64)\n"
        f"    if _flat < cutlass.Int32({elements}):\n"
        "        _row = _flat // cutlass.Int32(16)\n"
        "        _col = _flat % cutlass.Int32(16)\n"
        f"        _gm = tile_offset_0 + _row{offset}\n"
        "        _gk = tile_offset_2 + _col\n"
        f"        {tensor}[{swizzled}] = {source}[_gm, _gk]"
        " if _gm < cutlass.Int32(256) and _gk < cutlass.Int32(256)"
        " else cutlass.Float16(0.0)\n"
    )


def _staging_body(flat: str, offset: str = "") -> str:
    """One K iteration of the partial-TMA matmul (``test_cute_partial_tma``).

    ``x`` is staged into shared memory by the copy threads on both sides of
    the full-tile branch, ``y`` on the partial side only, each side closed
    by the emitter's own barrier.
    """
    guarded = "    if mma_active:\n"
    return (
        "if tcgen05_tma_full_tile:\n"
        + guarded
        + textwrap.indent(_staging_loop("sA_mma", "x", flat, 2048, offset), 8 * " ")
        + "    cute.arch.sync_threads()\n"
        "else:\n"
        + guarded
        + textwrap.indent(_staging_loop("sA_mma", "x", flat, 2048, offset), 8 * " ")
        + guarded
        + textwrap.indent(_staging_loop("sB_mma", "y", flat, 256, offset), 8 * " ")
        + "    cute.arch.sync_threads()\n"
    )


_LANE_COORDINATE = "cutlass.Int32(cute.arch.thread_idx()[0])"


@pytest.mark.parametrize(
    ("flat", "offset", "accepted"),
    [
        pytest.param("mma_copy_tidx", "", True, id="owned_by_both_axes"),
        pytest.param(_LANE_COORDINATE, "", True, id="duplicate_stores"),
        pytest.param(
            _LANE_COORDINATE,
            " + cutlass.Int32(cute.arch.thread_idx()[1])",
            False,
            id="value_varies_along_the_other_axis",
        ),
    ],
)
def test_staging_loops_owning_their_elements_need_no_barrier(
    flat: str, offset: str, accepted: bool
) -> None:
    # The copy loops are pinned statements on the two sides of a branch, so
    # a barrier ordering them could not be placed; none is needed when every
    # address term of their stores reads the thread's flattened coordinate,
    # defined inside the loop from both thread axes: no other thread of the
    # CTA touches those elements, and the emitter's barriers stay as they
    # are.  With a coordinate along one axis alone, the threads of the other
    # axis store the same values to the same elements, which is benign; a
    # value that varies along that axis is a race, and the branch is
    # rejected.
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(_staging_body(flat, offset)).body
    )
    (branch,) = items
    assert isinstance(branch, ast.If)
    emitted = len(_barriers(branch))

    def run() -> None:
        lane_loop_distribution.add_thread_barriers(
            items,
            [],
            rename_groups={},
            axis_sizes={0: 32, 1: 6},
            definitions=ast.parse(_STAGING_DEFINITIONS).body,
            masks=frozenset(),
            loop_variables=frozenset({"tile_offset_2"}),
            loop_body=items,
        )

    if not accepted:
        with pytest.raises(exc.BackendUnsupported, match="inside a branch"):
            run()
        return
    run()
    assert _kinds(items) == ["stmt"], [ast.unparse(s)[:60] for s in items]  # pyrefly: ignore [bad-argument-type]
    assert len(_barriers(branch)) == emitted == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carry_previous_column(x: torch.Tensor) -> torch.Tensor:
    # Iteration j reads the last column of tile j - 1 (uniform along the
    # column threads), which iteration j - 1 stored per thread.
    m, padded = x.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(1, padded):
            carry = x[tile0, tile1.begin - 1]
            x[tile0, tile1] = x[tile0, tile1] + carry[:, None]
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carry_to_the_next_tile(x: torch.Tensor) -> torch.Tensor:
    # Iteration j reads the first column of tile j (uniform) and stores the
    # columns after it, the first column of tile j + 1 included.
    m, padded = x.shape
    n = padded - 1
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            carry = x[tile0, tile1.begin]
            x[tile0, tile1.index + 1] = x[tile0, tile1.index + 1] + carry[:, None]
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_store_moving_backwards(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # Iteration j reads column 2100 - begin_j and stores -1 at column
    # block + 2164 - begin_j, both uniform: the columns move back by a block
    # per iteration, and iteration j + 1 + 64 / block stores the column
    # iteration j read.
    m = x.size(0)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(2048):
            v = x[tile0.begin, 2100 - tile1.begin]
            out[tile0, tile1] = out[tile0, tile1] * 0.0 + v
            x[tile0.begin, tile1.block_size + 2164 - tile1.begin] = -1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_then_read_moving_backwards(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # The same accesses with the store first: iteration j's read still meets
    # the store of a later iteration.
    m = x.size(0)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(2048):
            x[tile0.begin, tile1.block_size + 2164 - tile1.begin] = -1.0
            v = x[tile0.begin, 2100 - tile1.begin]
            out[tile0, tile1] = out[tile0, tile1] * 0.0 + v
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _uniform_read_modify_write_in_a_device_loop(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # Each iteration reads one element of x (uniform), spreads it over the
    # tile of out and increments the element.
    m, n = x.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            v = x[tile0.begin, tile1.begin]
            out[tile0, tile1] = out[tile0, tile1] * 0.0 + v
            x[tile0.begin, tile1.begin] = v + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _repeat_first_column_update(x: torch.Tensor) -> torch.Tensor:
    # The outer loop repeats an inner loop whose iteration reads a tile's
    # first column (uniform) and stores the tile per thread: the next
    # repetition reads what the previous one stored.
    m, n = x.shape
    for tile0 in hl.tile(m):
        for _repeat in hl.tile(8):
            for tile1 in hl.tile(n):
                first = x[tile0, tile1.begin]
                x[tile0, tile1] = x[tile0, tile1] + first[:, None]
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_loops_over_one_block_size(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Two device loops over one registered block size: the second reads the
    # first column of the next tile (uniform), which the first loop updated
    # per thread.
    m, padded = x.shape
    n = padded - 64
    block = hl.register_block_size(n)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n, block_size=block):
            x[tile0, tile1] = x[tile0, tile1] + 1.0
        for tile2 in hl.tile(n, block_size=block):
            first = x[tile0, tile2.begin + tile2.block_size]
            y[tile0, tile2] = first[:, None] + 0.0 * y[tile0, tile2]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_loops_over_one_block_size_war(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The mirror: the first loop reads the next tile's first column
    # (uniform), the second updates every column per thread.
    m, padded = x.shape
    n = padded - 64
    block = hl.register_block_size(n)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n, block_size=block):
            first = x[tile0, tile1.begin + tile1.block_size]
            y[tile0, tile1] = first[:, None] + 0.0 * y[tile0, tile1]
        for tile2 in hl.tile(n, block_size=block):
            x[tile0, tile2] = x[tile0, tile2] + 1.0
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _six_loops_over_one_block_size(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Six device loops over one registered block size, each reading the
    # next tile's first column (uniform), which the previous loop updated
    # per thread.
    m, padded = x.shape
    n = padded - 64
    block = hl.register_block_size(n)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n, block_size=block):
            x[tile0, tile1] = x[tile0, tile1] + 1.0
        for tile2 in hl.tile(n, block_size=block):
            first = x[tile0, tile2.begin + tile2.block_size]
            x[tile0, tile2] = x[tile0, tile2] + first[:, None]
        for tile3 in hl.tile(n, block_size=block):
            first = x[tile0, tile3.begin + tile3.block_size]
            x[tile0, tile3] = x[tile0, tile3] + first[:, None]
        for tile4 in hl.tile(n, block_size=block):
            first = x[tile0, tile4.begin + tile4.block_size]
            x[tile0, tile4] = x[tile0, tile4] + first[:, None]
        for tile5 in hl.tile(n, block_size=block):
            first = x[tile0, tile5.begin + tile5.block_size]
            x[tile0, tile5] = x[tile0, tile5] + first[:, None]
        for tile6 in hl.tile(n, block_size=block):
            first = x[tile0, tile6.begin + tile6.block_size]
            y[tile0, tile6] = first[:, None] + 0.0 * y[tile0, tile6]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_rows_then_zero(
    x: torch.Tensor, rows: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The gather is along the rows (swapped pairwise within a tile).
    for tile0, tile1 in hl.tile(x.shape):
        y[tile0, tile1] = x[rows[tile0], tile1]
        x[tile0, tile1] = 0.0
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _update_to_a_data_dependent_end(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    # The loop's end is loaded; ``tile1.end - 1`` is the tile's last column.
    for tile0 in hl.tile(x.size(0)):
        for tile1 in hl.tile(0, end[0]):
            last = x[tile0, tile1.end - 1]
            x[tile0, tile1] = x[tile0, tile1] + last[:, None]
    return x


# Rows over two threads and a lane loop of two, columns one per thread.
_DEVICE_LOOP_COLUMNS_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [2, 64],
    "cute_vector_widths": [1, 1],
}
# Rows and columns one per thread: no lane loop repeats the device loop.
_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG = {
    "block_sizes": [8, 32],
    "num_threads": [8, 32],
    "cute_vector_widths": [1, 1],
}
# Rows, the repetition loop (no threads), columns one per thread.
_REPEATED_INNER_LOOP_CONFIG = {
    "block_sizes": [4, 1, 64],
    "num_threads": [2, 0, 64],
    "cute_vector_widths": [1, 1, 1],
}
_REPEATED_INNER_LOOP_WIDE_CONFIG = {
    "block_sizes": [4, 1, 128],
    "num_threads": [1, 0, 128],
    "cute_vector_widths": [1, 1, 1],
}
# The registered block size is block 0 (one column per thread), the grid
# tile block 1 (rows over two threads and a lane loop of two).
_SHARED_BLOCK_SIZE_CONFIG = {
    "block_sizes": [64, 4],
    "num_threads": [64, 2],
    "cute_vector_widths": [1, 1],
}
# Rows: four lanes on one thread; columns: one per thread.
_ROW_LANES_ON_ONE_THREAD_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 1],
}
# Columns: 256 lanes on one thread; rows: one per thread.
_COLUMN_LANES_ON_ONE_THREAD_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [4, 1],
    "cute_vector_widths": [1, 1],
}


def _barrier_positions(body: Sequence[ast.AST]) -> list[int]:
    return [index for index, statement in enumerate(body) if _is_barrier(statement)]


def test_carry_from_the_previous_tile_closes_the_device_loop_body() -> None:
    # The read of column begin - 1 and the store of [begin, begin + block)
    # are apart within an iteration, so no barrier separates them; the next
    # iteration's read is this iteration's store, so one closes the body.
    code = _generate(
        _carry_previous_column, (torch.empty((8, 257)),), **_DEVICE_LOOP_COLUMNS_CONFIG
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    assert len(_barriers(function)) == 1, code
    assert _barrier_positions(device_loop.body) == [len(device_loop.body) - 1], code


def test_carry_to_the_next_tile_closes_the_device_loop_body() -> None:
    # ``tile.index + 1`` carries no region, but the address mappings decide
    # the pair within an iteration: the read of column ``begin`` and the
    # store of ``begin + 1 + thread`` never meet, so no barrier separates
    # them.  Across iterations the store of the next tile's first column
    # meets the next iteration's read: one barrier closes the body.
    code = _generate(
        _carry_to_the_next_tile, (torch.empty((8, 257)),), **_DEVICE_LOOP_COLUMNS_CONFIG
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    assert len(_barriers(function)) == 1, code
    assert _barrier_positions(device_loop.body) == [len(device_loop.body) - 1], code


_BACKWARDS_KERNELS = [
    pytest.param(_read_then_store_moving_backwards, id="read_then_store"),
    pytest.param(_store_then_read_moving_backwards, id="store_then_read"),
]


@pytest.mark.parametrize(
    ("kernel", "barriers"),
    [
        pytest.param(_read_then_store_moving_backwards, 1, id="read_then_store"),
        pytest.param(_store_then_read_moving_backwards, 2, id="store_then_read"),
    ],
)
def test_accesses_moving_backwards_close_the_device_loop_body(
    kernel: object, barriers: int
) -> None:
    # The read and the store are apart within an iteration: the pass puts no
    # barrier between them (with the store first, the one between them is the
    # memory-op codegen's own read-after-write barrier).  Advanced by one
    # step the store lies past the read, but its column comes back towards
    # the read's by a block per iteration, so the one-step proof does not
    # extend to the later iterations, one of which stores the column read:
    # a barrier closes the body.
    code = _generate(
        kernel,
        (torch.empty((8, 4096)), torch.empty((8, 4096))),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    assert len(_barriers(function)) == barriers, code
    assert _barrier_positions(device_loop.body)[-1] == len(device_loop.body) - 1, code


def test_uniform_read_modify_write_in_a_repeated_device_loop_rejects_the_config() -> (
    None
):
    # Rows over two threads and a lane loop of two: the grid's lane loop runs
    # the device loop whole once per lane, and the second pass reads the
    # elements the first pass incremented.
    with pytest.raises(
        exc.BackendUnsupported,
        match="a lane-invariant load of x in a loop repeated whole once per lane_0",
    ):
        _generate(
            _uniform_read_modify_write_in_a_device_loop,
            (torch.empty((8, 256)), torch.empty((8, 256))),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


def test_uniform_read_modify_write_in_a_device_loop_keeps_one_barrier() -> None:
    # One row per thread: no lane loop repeats the device loop, and the
    # barrier between the read and the store of the element orders the
    # threads.
    code = _generate(
        _uniform_read_modify_write_in_a_device_loop,
        (torch.empty((8, 256)), torch.empty((8, 256))),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    assert len(_barriers(_kernel_function(code))) == 1, code


def test_repeated_inner_loop_closes_the_outer_body() -> None:
    # The inner loop's body keeps its one barrier (read, barrier, store); the
    # outer body, whose one statement is the inner loop, is closed by another.
    code = _generate(
        _repeat_first_column_update,
        (torch.empty((8, 64)),),
        **_REPEATED_INNER_LOOP_CONFIG,
    )
    function = _kernel_function(code)
    (outer,) = _loops(function, "tile_offset_1")
    (inner,) = _loops(outer, "tile_offset_2")
    assert len(_barriers(inner)) == 1, code
    assert _barrier_positions(outer.body) == [len(outer.body) - 1], code


@pytest.mark.parametrize(
    "kernel",
    [_two_loops_over_one_block_size, _two_loops_over_one_block_size_war],
    ids=["read_after_write", "write_after_read"],
)
def test_two_loops_over_one_block_size_get_a_barrier_between_them(
    kernel: object,
) -> None:
    # The two loops' tiles begin at unrelated columns at any point of the
    # program, so their recorded regions cannot prove the accesses apart
    # (the begin symbol names the loop instance): a barrier separates them.
    code = _generate(
        kernel,
        (torch.empty((8, 320)), torch.empty((8, 320))),
        **_SHARED_BLOCK_SIZE_CONFIG,
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "lane_1")
    first, second = _loops(lane_loop, "tile_offset_2")
    start, end = lane_loop.body.index(first), lane_loop.body.index(second)
    assert any(start < p < end for p in _barrier_positions(lane_loop.body)), code


def test_six_loops_over_one_block_size_get_a_barrier_between_each_pair() -> None:
    # Each loop's state names its own begin symbol, by a serial no later
    # state reuses (the previous state is collected before the next loop is
    # emitted, and its address may come back): every pair of consecutive
    # loops is separated by a barrier.
    code = _generate(
        _six_loops_over_one_block_size,
        (torch.empty((8, 320)), torch.empty((8, 320))),
        **_SHARED_BLOCK_SIZE_CONFIG,
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "lane_1")
    loops = _loops(lane_loop, "tile_offset_")
    assert len(loops) == 6, code
    positions = _barrier_positions(lane_loop.body)
    for first, second in itertools.pairwise(loops):
        start, end = lane_loop.body.index(first), lane_loop.body.index(second)
        assert any(start < p < end for p in positions), code


@pytest.mark.parametrize(
    ("kernel", "config"),
    [
        pytest.param(
            _gather_rows_then_zero, _ROW_LANES_ON_ONE_THREAD_CONFIG, id="rows"
        ),
        pytest.param(
            _gather_then_zero, _COLUMN_LANES_ON_ONE_THREAD_CONFIG, id="columns"
        ),
    ],
)
def test_gathered_store_conflict_on_a_single_thread_axis_rejects_the_config(
    kernel: object, config: dict[str, object]
) -> None:
    # The gathered axis is a lane loop on one thread: lane i + 1 gathers
    # after lane i stored.  No barrier can order lanes of one thread.
    x = torch.empty((8, 256))
    index = torch.zeros(
        8 if kernel is _gather_rows_then_zero else 256, dtype=torch.int64
    )
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _generate(kernel, (x, index, torch.empty((8, 256))), **config)


def test_data_dependent_tile_end_keeps_one_barrier() -> None:
    # ``tile.end - 1`` lies in the tile: the read and the per-thread store
    # meet within an iteration (one barrier) but a later iteration's tile
    # lies past this one's (no closing barrier), as for a static end.
    code = _generate(
        _update_to_a_data_dependent_end,
        (torch.empty((8, 256)), torch.tensor([200])),
        **_DEVICE_LOOP_COLUMNS_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_1")
    positions = _barrier_positions(device_loop.body)
    assert len(_barriers(function)) == 1, code
    assert positions and positions[0] < len(device_loop.body) - 1, code


# -- the address mapping decides ownership (review of fixes10, finding 8) ----


_MAPPED_ROW_DEFINITIONS = (
    "indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
    "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
    "n = 64\n"
)


@pytest.mark.parametrize(
    ("column", "stored", "barrier"),
    [
        pytest.param("indices_1", "indices_1", False, id="identity"),
        pytest.param("indices_1 + 1", "indices_1", True, id="next_column"),
        pytest.param("indices_1 - 1", "indices_1", True, id="previous_column"),
        pytest.param("n - 1 - indices_1", "indices_1", True, id="reversed"),
        pytest.param("(indices_1 + 1) % n", "indices_1", True, id="modular"),
        pytest.param("indices_1 * 2", "indices_1", True, id="scaled"),
        pytest.param("indices_1 // 2", "indices_1", True, id="halved"),
        pytest.param("indices_1 + 1", "indices_1 + 1", False, id="same_offset"),
        pytest.param(
            "cutlass.Int32(indices_1) + 1",
            "operator.add(indices_1, 1)",
            False,
            id="same_offset_spelled_differently",
        ),
    ],
)
def test_thread_mapped_address_rules(column: str, stored: str, barrier: bool) -> None:
    # Rule level: a per-thread load of ``x`` through another mapping of the
    # thread's column beside the thread's store.  Reading the coordinate is
    # not owning the element: the neighbour's column, the reversed one, the
    # modular wrap, a scaled or halved index are other threads' elements and
    # get the barrier.  Only the identity mapping, up to an offset the two
    # accesses share, is the thread's own.
    body = (
        f"w = {_pointer('x', 'indices_0', column)}.load()\n"
        f"{_pointer('x', 'indices_0', stored)}.store(w)\n"
    )
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(body).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(_MAPPED_ROW_DEFINITIONS).body,
        masks=frozenset(),
    )
    expected = ["stmt", "sync", "stmt"] if barrier else ["stmt", "stmt"]
    assert _kinds(items) == expected, _kinds(items)


def test_transposed_address_is_another_threads_element() -> None:
    # Both terms read a coordinate, yet ``x[indices_1, indices_0]`` beside
    # ``x[indices_0, indices_1]`` names the element of the thread with the
    # swapped coordinates: a barrier separates the pair.
    body = (
        f"w = {_pointer('x', 'indices_1', 'indices_0')}.load()\n"
        f"{_pointer('x', 'indices_0', 'indices_1')}.store(w)\n"
    )
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(body).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 8, 1: 8},
        definitions=ast.parse(
            "indices_0 = cutlass.Int32(cute.arch.thread_idx()[1])\n"
            "indices_1 = cutlass.Int32(cute.arch.thread_idx()[0])\n"
        ).body,
        masks=frozenset(),
    )
    assert _kinds(items) == ["stmt", "sync", "stmt"], _kinds(items)


@pytest.mark.parametrize(
    ("column", "rejected"),
    [
        pytest.param("indices_1", False, id="identity"),
        pytest.param("indices_1 + 1", True, id="next_column"),
        pytest.param(
            "tile_offset_1 + 255 - (indices_1 - tile_offset_1)", True, id="reversed"
        ),
        pytest.param("(indices_1 + 1) % 256", True, id="modular"),
    ],
)
def test_mapped_lane_conflict_is_rejected(column: str, rejected: bool) -> None:
    # The lanes of one thread run in loop order while the program runs every
    # lane of the load before any lane's store: a load through another
    # mapping of the lane's column meets the store of another lane (the
    # next column is the next lane's, or the next thread's first), which no
    # barrier orders.  The thread's own column conflicts with nothing.
    scopes = _threaded_column_scopes()
    body = (
        f"w = {_pointer('x', 'indices_0', column)}.load()\n"
        f"{_pointer('x', 'indices_0', 'indices_1')}.store(w)\n"
    )
    column_loop = lane_loop_distribution.LanePlacement(
        "lane_1", list(ast.parse(body).body)
    )
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = [
        lane_loop_distribution.LanePlacement("lane_0", [column_loop])
    ]

    def run() -> None:
        lane_loop_distribution.add_thread_barriers(
            items,
            scopes,
            rename_groups={},
            axis_sizes={0: 64},
            definitions=[statement for scope in scopes for statement in scope.setup],
            masks=frozenset(),
        )

    if rejected:
        with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
            run()
        return
    run()
    assert _kinds(column_loop.items) == ["stmt", "stmt"]  # pyrefly: ignore [bad-argument-type]


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(
            "sA_mma[(((mma_copy_tidx + 1) % cute.size(sA_mma.shape[0][0]), 0), 0, 0)]"
            " = cutlass.Float16(1.0)",
            id="store",
        ),
        pytest.param(
            "probe = sA_mma[(((mma_copy_tidx + 1) % cute.size(sA_mma.shape[0][0]), 0), 0, 0)]",
            id="load",
        ),
    ],
)
def test_staging_access_of_another_threads_element_is_rejected(probe: str) -> None:
    # After the copy loop, a per-thread store to (or load of) the element
    # of the thread flattened one further, through the modular wrap: the
    # coordinate is read, the element is another thread's, and the pair
    # sits in a branch where no barrier can order it.
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(_staging_body("mma_copy_tidx")).body
    )
    (branch,) = items
    assert isinstance(branch, ast.If)
    fence = next(
        index for index, statement in enumerate(branch.orelse) if _is_barrier(statement)
    )
    branch.orelse.insert(fence, ast.parse(probe).body[0])
    with pytest.raises(exc.BackendUnsupported, match="inside a branch"):
        lane_loop_distribution.add_thread_barriers(
            items,
            [],
            rename_groups={},
            axis_sizes={0: 32, 1: 8},
            definitions=ast.parse(_STAGING_DEFINITIONS).body,
            masks=frozenset(),
            loop_variables=frozenset({"tile_offset_2"}),
            loop_body=items,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_next_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The neighbour's element beside the thread's own store.
    n = x.size(0)
    for tile in hl.tile(n):
        y[tile] = hl.load(x, [tile.index + 1], extra_mask=tile.index < n - 1)
        x[tile] = v[tile]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_reversed_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # A permutation of the elements: the tile read back to front.
    n = x.size(0)
    for tile in hl.tile(n):
        y[tile] = x[n - 1 - tile.index]
        x[tile] = v[tile]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_wrapped_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The modular wrap: the next element, the first after the last.
    n = x.size(0)
    for tile in hl.tile(n):
        y[tile] = x[(tile.index + 1) % n]
        x[tile] = v[tile]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_next_column_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The stencil neighbour ``x[t0, t1.index + 1]`` beside ``x[t0, t1]``.
    m, n = x.shape
    for tile0, tile1 in hl.tile([m, n]):
        y[tile0, tile1] = hl.load(
            x, [tile0, tile1.index + 1], extra_mask=(tile1.index < n - 1)[None, :]
        )
        x[tile0, tile1] = v[tile0, tile1]
    return y


_MAPPED_KERNELS = [
    pytest.param(_read_next_then_store, id="next"),
    pytest.param(_read_reversed_then_store, id="reversed"),
    pytest.param(_read_wrapped_then_store, id="wrapped"),
]
# One element per thread: the mapped read is another thread's element.
_ONE_ELEMENT_PER_THREAD_CONFIG = {
    "block_sizes": [256],
    "num_threads": [256],
    "cute_vector_widths": [1],
}
# Eight consecutive elements per thread, a rolled lane loop.
_EIGHT_LANES_PER_THREAD_CONFIG = {
    "block_sizes": [256],
    "num_threads": [32],
    "cute_vector_widths": [1],
}
# Rows over two threads and a lane loop of two, one column per thread.
_ONE_COLUMN_PER_THREAD_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [2, 64],
    "cute_vector_widths": [1, 1],
}
# Two columns per thread.
_TWO_COLUMNS_PER_THREAD_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [2, 32],
    "cute_vector_widths": [1, 1],
}


def _mapped_args() -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty(256) for _ in range(3))


@pytest.mark.parametrize("kernel", _MAPPED_KERNELS)
def test_mapped_read_beside_the_store_gets_a_barrier(kernel: object) -> None:
    # One element per thread: the read of another thread's element and the
    # store of the thread's own are ordered by one barrier between them.
    code = _generate(kernel, _mapped_args(), **_ONE_ELEMENT_PER_THREAD_CONFIG)
    function = _kernel_function(code)
    positions = _barrier_positions(function.body)
    load = _index_of(function.body, "x.iterator")
    store = _index_of(function.body, "x.iterator", after=load)
    assert len(positions) == 1 and load < positions[0] < store, code


@pytest.mark.parametrize("kernel", _MAPPED_KERNELS)
def test_mapped_read_in_a_lane_loop_rejects_the_config(kernel: object) -> None:
    # Eight lanes per thread: lane 7's read of the next element is the next
    # thread's first, which that thread stored in its lane 0 (the reversed
    # and wrapped reads meet other lanes likewise).  No barrier orders the
    # lanes of one thread as the program does.
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _generate(kernel, _mapped_args(), **_EIGHT_LANES_PER_THREAD_CONFIG)


def test_next_column_read_beside_the_store_gets_a_barrier() -> None:
    # The stencil neighbour along the column threads: one barrier between
    # the read and the store, none for the row lanes (the rows are each
    # thread's own in both accesses).
    args = tuple(torch.empty((8, 64)) for _ in range(3))
    code = _generate(
        _read_next_column_then_store, args, **_ONE_COLUMN_PER_THREAD_CONFIG
    )
    function = _kernel_function(code)
    (row_loop,) = _loops(function, "lane_0")
    positions = _barrier_positions(row_loop.body)
    load = _index_of(row_loop.body, "x.iterator")
    store = _index_of(row_loop.body, "x.iterator", after=load)
    assert len(_barriers(function)) == 1, code
    assert len(positions) == 1 and load < positions[0] < store, code
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _generate(_read_next_column_then_store, args, **_TWO_COLUMNS_PER_THREAD_CONFIG)


@pytest.mark.parametrize(
    ("body", "step", "expected"),
    [
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'indices_1')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)\n",
            64,
            ["stmt", "stmt"],
            id="own_columns",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'indices_1')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)\n",
            32,
            ["stmt", "stmt", "sync"],
            id="narrow_step",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'tile_offset_1')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)\n",
            64,
            ["stmt", "sync", "stmt"],
            id="first_column_read",
        ),
        # A store alone, its rows one per thread: no fold.
        pytest.param(
            f"{_pointer('x', 'indices_0', 'indices_1')}.store(v)\n",
            64,
            ["stmt"],
            id="store_alone",
        ),
        # A store alone into a flattened index: row r's iteration j + 1
        # and row r + 1's iteration j store one element, from two threads.
        pytest.param(
            f"{_pointer('x', 'indices_0 * 64 + indices_1')}.store(v)\n",
            64,
            ["stmt", "sync"],
            id="folding_store",
        ),
        # A flattened store whose column a tile mask bounds below the row
        # stride: the padded columns are never stored, so no fold.
        pytest.param(
            "mask_1 = cutlass.Int32(cute.arch.thread_idx()[0]) < 64 and indices_1 < 250\n"
            "if mask_1:\n"
            f"    {_pointer('x', 'indices_0 * 250 + indices_1')}.store(v)\n",
            64,
            ["stmt", "stmt"],
            id="masked_flattened_store",
        ),
        # The mask bounding the column past the row stride: row r's columns
        # 250 to 255 are row r + 1's first six.
        pytest.param(
            "mask_1 = cutlass.Int32(cute.arch.thread_idx()[0]) < 64 and indices_1 < 256\n"
            "if mask_1:\n"
            f"    {_pointer('x', 'indices_0 * 250 + indices_1')}.store(v)\n",
            64,
            ["stmt", "stmt", "sync"],
            id="masked_flattened_store_past_the_stride",
        ),
        # Without the mask the padded columns count.
        pytest.param(
            f"{_pointer('x', 'indices_0 * 250 + indices_1')}.store(v)\n",
            64,
            ["stmt", "sync"],
            id="unmasked_flattened_store",
        ),
    ],
)
def test_device_loop_header_models_the_iterations(
    body: str, step: int, expected: list[str]
) -> None:
    # Without recorded regions the loop's header decides the cross-iteration
    # pairs: stepping the columns by 64, one per thread, iteration j + 1's
    # thread t touches column 64 (j + 1) + t, never iteration j's
    # thread t' column 64 j + t', so per-thread accesses need no closing
    # barrier and neither does the uniform read of the tile's first column
    # (iteration j never stores column 64 (j + 1)).  Stepping by 32 with 64
    # threads, iteration j + 1's thread 0 and iteration j's thread 32 share
    # a column, and the body is closed.  A statement's stores alone meet
    # across the iterations when the thread-to-element map folds (the
    # row and the iteration summed into one index); the tile mask guarding
    # the store bounds the column it names, so the fold is judged over the
    # columns the program stores, not the tile's padding.
    definitions = ast.parse(
        _ROW_DEFINITIONS
        + "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
    ).body
    (header,) = ast.parse(
        "for tile_offset_1 in range(cutlass.Int32(0), cutlass.Int32(1024),"
        f" cutlass.Int32({step})):\n    pass\n"
    ).body
    assert isinstance(header, ast.For)
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(body).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=definitions,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_headers=[header],
    )
    assert _kinds(items) == expected, _kinds(items)


def _column_loop_storing_rows(masked: bool) -> str:
    """An inner device loop over the columns storing ``x[K * row + column]``, under the column tile's mask when ``masked``."""
    store = f"{_pointer('x', 'indices_0 * K + indices_2')}.store(v)\n"
    if masked:
        store = (
            "mask_2 = cutlass.Int32(cute.arch.thread_idx()[0]) < 64 and indices_2 < K\n"
            f"if mask_2:\n    {store}"
        )
    return (
        "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(cutlass.Int32(K)),"
        " cutlass.Int32(64)):\n"
        "    indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
        + textwrap.indent(store, "    ")
    )


@pytest.mark.parametrize(
    ("masked", "expected"),
    [(True, ["stmt"]), (False, ["stmt", "sync"])],
    ids=["masked", "unmasked"],
)
def test_inner_loops_mask_bounds_its_store_across_the_outer_iterations(
    masked: bool, expected: list[str]
) -> None:
    # Two threads step the rows through the outer loop; an inner device loop
    # (one statement of the outer body) stores each row through the
    # flattened ``K * row + column`` with ``K`` a kernel argument.  The
    # column tile's mask inside the inner loop, around its only store,
    # bounds the column below ``K`` for the loop as a whole, so thread
    # t's iteration j + 1 and thread t''s iteration j store different
    # rows' elements and the outer body needs no closing barrier.
    # Unmasked, the padded columns of one row may be the next row's first.
    (header,) = ast.parse(
        "for tile_offset_1 in range(cutlass.Int32(0), cutlass.Int32(1024),"
        " cutlass.Int32(2)):\n    pass\n"
    ).body
    assert isinstance(header, ast.For)
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(_column_loop_storing_rows(masked)).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(
            "indices_0 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_headers=[header],
    )
    assert _kinds(items) == expected, _kinds(items)


def _loop_header(text: str) -> ast.For:
    """``text``, a ``for`` statement's header, as a loop over ``pass``."""
    (header,) = ast.parse(f"{text}:\n    pass\n").body
    assert isinstance(header, ast.For)
    return header


_ENCLOSING_LOOP_HEADERS = (
    (
        "for tile_offset_5 in range(cutlass.Int32(0), cutlass.Int32(cutlass.Int32(M)),"
        " cutlass.Int32(512))"
    ),
    "for lane_5 in range(2)",
)


@pytest.mark.parametrize(
    ("enclosing", "expected"),
    [(True, ["stmt"]), (False, ["stmt", "sync"])],
    ids=["headers", "names"],
)
def test_mask_over_the_enclosing_loops_variables_bounds_the_column(
    enclosing: bool, expected: list[str]
) -> None:
    # An inner device loop steps the rows, two per iteration on two threads,
    # and stores each through the flattened ``M * row + column``.  The
    # column ``tile_offset_5 + tid + 256 lane_5`` reads the enclosing column
    # loop's offset and the enclosing lane loop's variable, one value each
    # throughout the inner body; the mask ``indices_5 < M`` bounds it below
    # the row stride once the headers of those loops say both names are
    # nonnegative, so the stores are one thread's and one iteration's and
    # the inner body needs no closing barrier.  Known as names alone
    # (uniform values of unknown sign), the column is left to its parts,
    # whose range the proof does not know.
    header = _loop_header(
        "for tile_offset_6 in range(cutlass.Int32(0), cutlass.Int32(cutlass.Int32(amax)),"
        " cutlass.Int32(2))"
    )
    outer = (
        [_loop_header(text) for text in _ENCLOSING_LOOP_HEADERS] if enclosing else []
    )
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(
            f"if mask_5:\n    {_pointer('x', 'indices_6 * M + indices_5')}.store(v)\n"
        ).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 256, 1: 2},
        definitions=ast.parse(
            "indices_5 = tile_offset_5 + cutlass.Int32(cute.arch.thread_idx()[0])"
            " + cutlass.Int32(lane_5) * 256\n"
            "mask_5 = indices_5 < M\n"
            "indices_6 = tile_offset_6 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_6"}),
        loop_body=items,
        loop_headers=[*outer, header],
    )
    assert _kinds(items) == expected, _kinds(items)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _short_loop_then_read_ahead_and_store(
    x: torch.Tensor, z: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # A two-iteration loop over the registered block size precedes the loop
    # under test, whose iteration j + 2 stores (uniformly) the column
    # iteration j read (uniformly): the closing barrier keeps every thread's
    # read ahead of the store.  The first loop's variable is the second's.
    m, padded = x.shape
    n = padded - 256
    block = hl.register_block_size(n)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(128, block_size=block):
            z[tile0, tile1] = hl.zeros([tile0, tile1])
        for tile2 in hl.tile(n, block_size=block):
            value = x[tile0.begin, tile2.begin + 128]
            out[tile0, tile2] = out[tile0, tile2] * 0.0 + value
            x[tile0.begin, tile2.begin] = -1.0
    return out


# The registered (column) block is block 0: columns one per thread, rows one
# per thread.
_SHORT_SIBLING_LOOP_CONFIG = {
    "block_sizes": [64, 8],
    "num_threads": [64, 8],
    "cute_vector_widths": [1, 1],
}


def test_short_sibling_loop_over_one_block_size_keeps_the_closing_barrier() -> None:
    # The first loop runs two iterations, the second thirty-two: the loop
    # under test is modelled from its own header, not from the finished
    # sibling's (which proved the read and the store two blocks apart over a
    # range of two iterations and dropped the barrier).
    code = _generate(
        _short_loop_then_read_ahead_and_store,
        (torch.empty((8, 2304)), torch.empty((8, 2304)), torch.empty((8, 2304))),
        **_SHORT_SIBLING_LOOP_CONFIG,
    )
    function = _kernel_function(code)
    first, second = _loops(function, "tile_offset_")
    assert not _barriers(first), code
    assert _barrier_positions(second.body) == [len(second.body) - 1], code
    assert len(_barriers(function)) == 1, code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_previous_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # The previous element beside the thread's own store: the first element
    # of a thread's packet reads the last element of the previous thread's.
    n = x.size(0)
    for tile in hl.tile(n):
        y[tile] = hl.load(x, [tile.index - 1], extra_mask=tile.index >= 1)
        x[tile] = v[tile]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_fourth_next_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # Four elements ahead: the same vector lane of the next lane (or of the
    # next thread's first lane).
    n = x.size(0)
    for tile in hl.tile(n):
        y[tile] = hl.load(x, [tile.index + 4], extra_mask=tile.index + 4 < n)
        x[tile] = v[tile]
    return y


# Four consecutive elements per thread, one vector packet.
_ONE_PACKET_PER_THREAD_CONFIG = {
    "block_sizes": [256],
    "num_threads": [64],
    "cute_vector_widths": [4],
}
# Two lanes of four consecutive elements per thread.
_TWO_PACKETS_PER_THREAD_CONFIG = {
    "block_sizes": [256],
    "num_threads": [32],
    "cute_vector_widths": [4],
}


@pytest.mark.parametrize("shape", [(256,), (1024,)])
def test_packet_store_beside_the_previous_elements_read_gets_a_barrier(
    shape: tuple[int],
) -> None:
    # The flush stores the thread's four elements; the read of element
    # t - 1 in vector lane 0 is the previous thread's element 3, which the
    # packet's first element never shows.  One packet per thread: the
    # barrier between the read and the flush orders the threads.
    code = _generate(
        _read_previous_then_store,
        tuple(torch.empty(shape) for _ in range(3)),
        **_ONE_PACKET_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    assert len(_barriers(function)) == 1, code
    (lane_loop,) = _loops(function, "lane_0")
    flush = _index_of(lane_loop.body, "_cute_store_u32_vec(x.iterator")
    assert all(p < flush for p in _barrier_positions(lane_loop.body)), code


@pytest.mark.parametrize(
    ("kernel", "shape"),
    [
        # Lane 1's element 0 is lane 0's element 3 of the same thread, which
        # the rolled loop stored already.
        pytest.param(_read_previous_then_store, (256,), id="previous_full"),
        pytest.param(_read_previous_then_store, (1024,), id="previous_long"),
        # Lane 1's vector lane vl reads thread t + 1's lane 0 element vl:
        # the vector lane forced equal confines the pair to a vector lane,
        # not to a lane of the loop around it (masked tile: scalar stores).
        pytest.param(_read_fourth_next_then_store, (250,), id="fourth_masked"),
        pytest.param(_read_fourth_next_then_store, (256,), id="fourth_full"),
    ],
)
def test_mapped_read_across_the_packets_of_a_thread_rejects_the_config(
    kernel: object, shape: tuple[int]
) -> None:
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _generate(
            kernel,
            tuple(torch.empty(shape) for _ in range(3)),
            **_TWO_PACKETS_PER_THREAD_CONFIG,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_previous_row_then_store_in_a_device_loop(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # A per-lane read of the previous row (another lane's) beside the
    # per-lane store of the row, inside a device loop.
    m, n = x.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            y[tile0, tile1] = hl.load(
                x,
                [tile0.index - 1, tile1],
                extra_mask=(tile0.index > tile0.begin)[:, None],
            )
            x[tile0, tile1] = v[tile0, tile1]
    return y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _dressed_uniform_read_then_update_in_a_device_loop(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # The uniform read of the tile's first row dressed as per-lane
    # (``tile0.index * 0``), beside the per-lane update of the tile.
    m, n = x.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            value = x[tile0.begin + tile0.index * 0, tile1.begin]
            out[tile0, tile1] = out[tile0, tile1] * 0.0 + value[:, None]
            x[tile0, tile1] = x[tile0, tile1] + 1.0
    return out


_REPEATED_DEVICE_LOOP_KERNELS = [
    pytest.param(
        _read_previous_row_then_store_in_a_device_loop,
        (torch.empty((8, 256)), torch.empty((8, 256)), torch.empty((8, 256))),
        id="previous_row",
    ),
    pytest.param(
        _dressed_uniform_read_then_update_in_a_device_loop,
        (torch.empty((8, 256)), torch.empty((8, 256))),
        id="dressed_uniform",
    ),
]


@pytest.mark.parametrize(("kernel", "args"), _REPEATED_DEVICE_LOOP_KERNELS)
def test_per_lane_accesses_mapped_differently_in_a_repeated_device_loop_reject_the_config(
    kernel: object, args: tuple[torch.Tensor, ...]
) -> None:
    # Rows over two threads and a lane loop of two: the grid's lane loop runs
    # the device loop whole once per lane, and lane 1's pass reads the rows
    # lane 0's pass stored (the previous row's, or the first row's under an
    # address that names the lane without depending on it).  Neither access
    # is lane-invariant by its names, so the address mappings decide.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(kernel, args, **_DEVICE_LOOP_COLUMNS_CONFIG)


@pytest.mark.parametrize(("kernel", "args"), _REPEATED_DEVICE_LOOP_KERNELS)
def test_per_lane_accesses_mapped_differently_in_a_device_loop_per_thread_get_a_barrier(
    kernel: object, args: tuple[torch.Tensor, ...]
) -> None:
    # One row per thread: no lane loop repeats the device loop, and the
    # other row's element is another thread's, ordered by a barrier.
    code = _generate(kernel, args, **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG)
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_")
    assert len(_barriers(device_loop)) == 1, code


def _repeated_loop_nest(
    body: str,
    regions: Sequence[tuple[int, tuple[object, ...]]] = (),
    constants: dict[str, int] | None = None,
    definitions: str = "",
) -> list[str]:
    """The kinds of the row lane loop's items after the pass, the device loop ``body`` among them.

    The row lane loop runs eight lanes on one thread
    (``_threaded_column_scopes``); the columns are one per thread.
    ``regions`` are recorded on the first load or store call of the body
    statements at the given positions, ``constants`` are the block-size
    constants' values and ``definitions`` statements around the nest,
    before the lanes' own.
    """
    (loop,) = ast.parse(_REPEATED_LOOP.format(body=textwrap.indent(body, "    "))).body
    assert isinstance(loop, ast.For)
    for position, recorded in regions:
        call = next(
            node
            for node in ast.walk(loop.body[position + 1])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("load", "store")
        )
        setattr(call, HELION_ACCESS_REGIONS_ATTR, recorded)
    scope = _threaded_column_scopes()[0]
    placement = lane_loop_distribution.LanePlacement("lane_0", [loop])
    lane_loop_distribution.add_thread_barriers(
        [placement],
        [scope],
        rename_groups={},
        axis_sizes={0: 64},
        definitions=[*ast.parse(definitions).body, *scope.setup],
        masks=frozenset(),
        constants=constants,
    )
    return _kinds(placement.items)


@pytest.mark.parametrize(
    ("body", "accepted"),
    [
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'indices_2')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)\n",
            True,
            id="same_mapping",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0', 'indices_2 + 1')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)\n",
            True,
            id="next_column_of_the_lane",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0 - 1', 'indices_2')}.load()\n"
            f"{_pointer('y', 'indices_0', 'indices_2')}.store(v)\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(c)\n",
            False,
            id="previous_lane",
        ),
        pytest.param(
            f"v = {_pointer('x', 'indices_0 - 1', 'indices_2')}.load()\n"
            f"_cute_mystery_store({_pointer('x', 'indices_0', 'indices_2')}, v)\n",
            False,
            id="previous_lane_helper_store",
        ),
        pytest.param(
            f"v = {_pointer('x', 'tile_offset_0 + indices_0 * 0', 'indices_2')}.load()\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)\n",
            False,
            id="dressed_invariant",
        ),
        pytest.param(
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2')}.store(c)\n"
            f"{_pointer('y', 'indices_0', 'indices_2')}.store(c)\n"
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2 + 1')}.store(c)\n",
            True,
            id="two_invariant_stores",
        ),
        pytest.param(
            f"{_pointer('x', 'tile_offset_0', 'tile_offset_2')}.store(c)\n"
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(c)\n",
            False,
            id="invariant_store_beside_a_per_lane_one",
        ),
        pytest.param(
            f"v = {_pointer('x', 'tile_offset_0', '5')}.load()\n"
            f"{_pointer('x', 'indices_0', '7')}.store(v)\n",
            True,
            id="constant_columns_apart",
        ),
        # A store alone, confined to its lane by the row.
        pytest.param(
            f"{_pointer('x', 'indices_0', 'indices_2')}.store(c)\n",
            True,
            id="per_lane_store_alone",
        ),
        # A store alone whose map folds between the iterations: lane l's
        # iteration j + 1 and lane l + 1's iteration j store one element,
        # and lane l + 1's pass lands last.
        pytest.param(
            f"{_pointer('x', 'indices_0 + tile_offset_2')}.store(c)\n",
            False,
            id="store_folding_with_the_iteration",
        ),
        # A store alone every lane shares: re-applied in program order.
        pytest.param(
            f"{_pointer('x', 'tile_offset_0 + tile_offset_2')}.store(c)\n",
            True,
            id="invariant_store_alone",
        ),
    ],
)
def test_device_loop_repeated_per_lane_is_checked_through_its_address_mappings(
    body: str, accepted: bool
) -> None:
    # The device loop under the row lane loop runs whole once per lane; its
    # body's accesses of one tensor are paired through their address
    # mappings with the loop's iteration free on both sides: confined to a
    # lane (the same mapping; a column shift, which is the thread's
    # business), apart (constant columns), or two stores every lane shares
    # (re-applied in program order) are accepted; another lane's row, a
    # helper's store, an address dressed with the lane or shared by every
    # lane beside a per-lane store are not.  A statement's stores alone
    # are paired with each other too: confined to the lane or shared by
    # every lane they are accepted, folding between the iterations they
    # are not.
    if accepted:
        assert _repeated_loop_nest(body) == ["stmt"]
    else:
        with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
            _repeated_loop_nest(body)


def test_helper_call_in_a_device_loop_repeated_per_lane_rejects_the_config() -> None:
    # A call of unknown effects whose operands ignore the lane reads or
    # writes ``x`` again in every pass, between the per-lane stores of the
    # first pass's later iterations and the second's earlier ones.
    body = (
        f"v = _cute_mystery_load({_pointer('x', 'tile_offset_0', 'tile_offset_2')}, 4)\n"
        f"{_pointer('x', 'indices_0', 'indices_2')}.store(v)\n"
    )
    (loop,) = ast.parse(_REPEATED_LOOP.format(body=textwrap.indent(body, "    "))).body
    with pytest.raises(
        exc.BackendUnsupported,
        match="a call of unknown effects on x in a loop repeated whole once per lane_0",
    ):
        lane_loop_distribution.check_full_nest(
            [loop], [_threaded_column_scopes()[0]], rename_groups={}
        )


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        # Vector lane vl of lane l + 1 (or of the next thread's lane 0): the
        # vector lane is forced equal, the lane is not.
        pytest.param("indices_1 + 4", None, id="next_lane"),
        # The next thread's element in the same lane and vector lane.
        pytest.param("indices_1 + 8", ([], ["stmt", "sync", "stmt"]), id="next_thread"),
    ],
)
def test_vector_lane_forced_equal_is_no_lane_confinement(
    read: str, expected: tuple[list[int], list[str]] | None
) -> None:
    # Two lanes of four elements per thread (``lane_base_1 = thread * 8 +
    # lane_1 * 4``): a pair confined to one vector lane may still cross the
    # lanes of the loop around it, which no barrier orders.
    scope = dataclasses.replace(
        _vectorized_column_scope(
            "lane_base_1 = cutlass.Int32(cute.arch.thread_idx()[0]) * 8"
            " + cutlass.Int32(lane_1) * 4",
            vloop_index=1,
        ),
        counts={"lane_1": 2, "vec_lane_1": 4},
    )
    body = (
        f"w = {_pointer('x', 'tile_offset_0', read)}.load()\n"
        f"{_pointer('x', 'tile_offset_0', 'indices_1')}.store(w)\n"
    )
    if expected is None:
        with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
            _vectorized_nest(scope, body, {0: 64})
    else:
        assert _vectorized_nest(scope, body, {0: 64}) == expected


def test_leader_predicates_with_different_values_are_different_threads() -> None:
    # ``tid == 3`` is not the range ``[0, 1)``: thread 3's store of element
    # 0 and thread 0's load of it are two threads' accesses, ordered by a
    # barrier; the same branches with ``tid == 0`` on both sides are one
    # thread's.
    def nest(first: int, second: int) -> list[str]:
        thread = "cutlass.Int32(cute.arch.thread_idx()[0])"
        items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
            ast.parse(
                f"if {thread} == {first}:\n"
                f"    {_pointer('x', f'{thread} - {first}')}.store(v)\n"
                f"if {thread} == {second}:\n"
                f"    w = {_pointer('x', f'{thread} - {second}')}.load()\n"
            ).body
        )
        lane_loop_distribution.add_thread_barriers(
            items,
            [],
            rename_groups={},
            axis_sizes={0: 64},
            definitions=[],
            masks=frozenset(),
        )
        return _kinds(items)

    assert nest(0, 0) == ["stmt", "stmt"]
    assert nest(3, 0) == ["stmt", "sync", "stmt"]


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        # The thread's own element: the loaded base is some integer, the
        # same or another between the two accesses, and the thread's offset
        # rides on it times the block size.
        pytest.param("indices_1", ["stmt", "stmt"], id="own"),
        # The next thread's element beside the store.
        pytest.param("indices_1 + 1", ["stmt", "sync", "stmt"], id="next"),
    ],
)
def test_unknown_values_leave_the_rest_of_the_term_to_the_proof(
    read: str, expected: list[str]
) -> None:
    # ``base`` is loaded, so its value is unknown; a term ``base * 32 +
    # thread`` still confines the pair to the thread, as ``32 (b - b') +
    # (t - t') == 0`` forces ``t == t'`` whatever integers the bases are.
    definitions = (
        f"base = {_pointer('starts', 'tile_offset_0')}.load()\n"
        "indices_1 = base * 32 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
    )
    body = f"{_pointer('x', 'indices_1')}.store(v)\nw = {_pointer('x', read)}.load()\n"
    assert _root_items(body, definitions, {0: 32}) == expected


@pytest.mark.parametrize(
    ("column", "update", "accepted"),
    [
        # Eight lanes: ``8 * base + lane`` confines the update to its lane.
        pytest.param("base * 8 + indices_0", True, True, id="scaled_update"),
        # ``base + lane`` may meet another lane's element (the base is any
        # integer): the load of one lane's pass beside the store of the
        # other's.
        pytest.param("base + indices_0", True, False, id="unscaled_update"),
        # A store alone through a loaded base: whether the lanes' map
        # folds is the program's own knowledge (a jagged tile's rows begin
        # at a loaded offset), which the pass does not model.
        pytest.param("base + indices_0", False, True, id="unscaled_store"),
    ],
)
def test_loaded_base_in_a_repeated_device_loop_keeps_the_lane_term(
    column: str, update: bool, accepted: bool
) -> None:
    # The persistent grouped GEMM's worker loop: the tile's column base is
    # computed from loaded group offsets, and the lanes' columns ride on it
    # times the tile width.
    body = (
        f"base = {_pointer('starts', 'tile_offset_2')}.load()\n"
        f"column = {column}\n"
        + (
            f"w = {_pointer('out', 'tile_offset_0', 'column')}.load()\n"
            if update
            else ""
        )
        + f"{_pointer('out', 'tile_offset_0', 'column')}.store(v)\n"
    )
    if accepted:
        assert _repeated_loop_nest(body) == ["stmt"]
    else:
        with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
            _repeated_loop_nest(body)


def test_packets_include_the_hoisted_vector_load() -> None:
    # The packet load a wrapper hoists above its V-loop is
    # ``cute.arch.load(pointer, VectorType)``: its address is the call's
    # first argument, not the receiver of a ``.load()`` method, and it
    # covers the thread's four elements like the flush after the V-loop.
    lane = _pointer("x", "tile_offset_0", "lane_base_1")
    (wrapper,) = ast.parse(
        "for lane_1 in range(1):\n"
        f"    {_LANE_BASE}\n"
        f"    packet = cute.arch.load({lane},"
        " ir.VectorType.get([4], cutlass.Float32.mlir_type))\n"
        "    for vec_lane_1 in cutlass.range_constexpr(4):\n"
        "        pass\n"
        f"    _cute_store_u32_vec({lane}, packed)\n"
    ).body
    calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call) and lane_loop_distribution._is_packet_call(node)
    ]
    assert len(calls) == 2
    packets = lane_loop_distribution._packets(wrapper)
    assert set(packets) == {id(call) for call in calls}
    assert set(packets.values()) == {("vec_lane_1", "lane_base_1")}


_REPEATED_ROWS = sympy.Symbol("_helion_tile_begin_0_0", integer=True)
_REPEATED_BEGIN = sympy.Symbol("_helion_tile_begin_2_1", integer=True)
_REPEATED_BLOCK = sympy.Symbol("_BLOCK_SIZE_2", integer=True, positive=True)


def test_regions_apart_within_an_iteration_do_not_exempt_a_repeated_loop() -> None:
    # The previous row's element at the next tile's first column beside the
    # tile's store: the recorded regions share the device loop's begin and
    # are apart within one iteration, but lane 1's pass reads in iteration
    # j what lane 0's pass stored in iteration j + 1.  Only literal bounds
    # count inside a loop the lanes repeat whole.
    body = (
        f"v = {_pointer('x', 'indices_0 - 1', 'tile_offset_2 + _BLOCK_SIZE_2')}.load()\n"
        f"{_pointer('y', 'indices_0', 'indices_2')}.store(v)\n"
        f"{_pointer('x', 'indices_0', 'indices_2')}.store(c)\n"
    )
    next_column = sympy.Add(_REPEATED_BEGIN, _REPEATED_BLOCK)
    regions = [
        (
            0,
            (
                (sympy.Add(_REPEATED_ROWS, -1), sympy.Add(_REPEATED_ROWS, 7)),
                (next_column, sympy.Add(next_column, 1)),
            ),
        ),
        (
            2,
            (
                (_REPEATED_ROWS, sympy.Add(_REPEATED_ROWS, 8)),
                (_REPEATED_BEGIN, next_column),
            ),
        ),
    ]
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _repeated_loop_nest(body, regions)


def test_literal_regions_apart_exempt_a_pair_in_a_repeated_loop() -> None:
    # A gathered row read at a column the proof does not know beside the
    # per-lane store of the row's upper half: the terms prove nothing, the
    # recorded literal bounds show the two apart in every iteration of
    # either pass.
    body = (
        f"idx = {_pointer('rows', 'indices_0')}.load()\n"
        f"v = {_pointer('x', 'idx', 'column')}.load()\n"
        f"{_pointer('x', 'indices_0', '64 + indices_2')}.store(v)\n"
    )
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _repeated_loop_nest(body)
    regions = [
        (1, (None, (sympy.Integer(0), sympy.Integer(64)))),
        (
            2,
            (
                (_REPEATED_ROWS, sympy.Add(_REPEATED_ROWS, 8)),
                (sympy.Integer(64), sympy.Integer(128)),
            ),
        ),
    ]
    assert _repeated_loop_nest(body, regions) == ["stmt"]


def test_store_confined_to_an_iteration_of_the_repeated_loop_is_accepted() -> None:
    # Two lanes store one row within an iteration (the tile statement's
    # own duplicate, whichever wins), never across iterations once the
    # loop's step is known: the columns force the iteration equal.
    body = f"{_pointer('x', 'indices_0 // 2', 'indices_2')}.store(c)\n"
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _repeated_loop_nest(body)
    assert _repeated_loop_nest(body, constants={"_BLOCK_SIZE_2": 64}) == ["stmt"]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_previous_row_at_the_next_tiles_column_then_store(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # A per-lane read of the previous row (another lane's) at the next
    # tile's first column beside the per-lane store of the tile: apart
    # within one iteration, while lane 1's pass reads in iteration j the
    # column lane 0's pass stored in iteration j + 1.
    m, padded = x.shape
    n = padded - 64
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            column = tile1.begin + tile1.block_size
            value = hl.load(x, [tile0.index - 1, column], extra_mask=tile0.index >= 1)
            y[tile0, tile1] = hl.zeros([tile0, tile1]) + value[:, None]
            x[tile0, tile1] = v[tile0, tile1]
    return y


def test_previous_row_read_at_the_next_tiles_column_rejects_the_config() -> None:
    code_args = tuple(torch.empty((8, 320)) for _ in range(3))
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(
            _read_previous_row_at_the_next_tiles_column_then_store,
            code_args,
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


def test_previous_row_read_at_the_next_tiles_column_per_thread_gets_a_barrier() -> None:
    # One row per thread: the previous row is another thread's, and the
    # barrier closing the body keeps iteration j's read ahead of iteration
    # j + 1's stores.
    code = _generate(
        _read_previous_row_at_the_next_tiles_column_then_store,
        tuple(torch.empty((8, 320)) for _ in range(3)),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_")
    assert _barrier_positions(device_loop.body) == [len(device_loop.body) - 1], code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_modify_write_in_a_device_loop(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # Every access the lane's own row and the thread's own columns.
    m, n = x.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            y[tile0, tile1] = x[tile0, tile1] + 1.0
            x[tile0, tile1] = v[tile0, tile1]
    return y


# Rows over two threads and a lane loop of two; columns four per thread, one
# vector packet each.
_VECTORIZED_DEVICE_LOOP_COLUMNS_CONFIG = {
    "block_sizes": [4, 64],
    "num_threads": [2, 16],
    "cute_vector_widths": [1, 4],
}


def test_packets_inside_a_repeated_device_loop_are_recognized() -> None:
    # The packet load hoisted above the V-loop and the flush after it, both
    # inside the device loop the row lanes repeat, cover the thread's own
    # four columns of the lane's row: confined to the lane, no barrier and
    # no rejection.
    code = _generate(
        _read_modify_write_in_a_device_loop,
        tuple(torch.empty((8, 256)) for _ in range(3)),
        **_VECTORIZED_DEVICE_LOOP_COLUMNS_CONFIG,
    )
    assert (
        "cute.arch.load(x.iterator" in code and "_cute_store_u32_vec(x.iterator" in code
    ), code
    assert not _barriers(_kernel_function(code)), code


def test_mapped_read_in_a_vectorized_repeated_device_loop_rejects_the_config() -> None:
    # The previous row's packet beside the row's flush is another lane's.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(
            _read_previous_row_then_store_in_a_device_loop,
            tuple(torch.empty((8, 256)) for _ in range(3)),
            **_VECTORIZED_DEVICE_LOOP_COLUMNS_CONFIG,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _update_under_a_loop_indexed_by_its_begin(
    x: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    # An earlier loop over the short block indexes its tile, which gives the
    # block a lane loop; the later loop over it reads ``tile2.begin`` only,
    # and the update nested in it runs once per iteration of that loop.
    m, n = x.shape
    block = hl.register_block_size(n)
    short = hl.register_block_size(128)
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(64, block_size=short):
            z[tile0, tile1] = hl.zeros([tile0, tile1])
        for tile2 in hl.tile(128, block_size=short):
            for tile3 in hl.tile(n, block_size=block):
                x[tile0, tile3] = x[tile0, tile3] + 1.0 + tile2.begin * 0.0
    return x


# The columns (block 0) one per thread, the short block (block 1) 64 elements
# on one thread, the rows (block 2) one per thread.
_SHORT_BLOCK_ON_ONE_THREAD_CONFIG = {
    "block_sizes": [64, 64, 8],
    "num_threads": [64, 1, 8],
    "cute_vector_widths": [1, 1, 1],
}


def test_dead_lane_loop_of_a_later_loop_over_the_block_is_spliced() -> None:
    # The nest around a device loop's body is built before the body: the
    # later loop carries the block's lane loop of 64 although its body reads
    # none of the lane's names.  The loop is spliced away with its index
    # setup, as the checks described the nest (without it the nested
    # update ran 64 times per iteration).
    code = _generate(
        _update_under_a_loop_indexed_by_its_begin,
        (torch.empty((8, 2048)), torch.empty((8, 64))),
        **_SHORT_BLOCK_ON_ONE_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (lane_loop,) = _loops(function, "lane_")
    assert "z.iterator" in ast.unparse(lane_loop), code
    assert "x.iterator" not in ast.unparse(lane_loop), code
    assert not _barriers(function), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_rows_shifted_by_the_iteration(
    v: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # Iteration j stores row r to element r + j: rows r and r + 1 store one
    # element in iterations j + 1 and j, and the program's value is
    # iteration j + 1's.
    m, n = v.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n, block_size=1):
            x[tile0.index + tile1.begin] = v[tile0, tile1.begin]
    return x


_ROWS_ON_TWO_LANES_CONFIG = {
    "block_sizes": [4],
    "num_threads": [2],
    "cute_vector_widths": [1],
}
_ROWS_ONE_PER_THREAD_CONFIG = {
    "block_sizes": [8],
    "num_threads": [8],
    "cute_vector_widths": [1],
}


def test_folding_store_in_a_repeated_device_loop_rejects_the_config() -> None:
    # Rows over two threads and a lane loop of two: lane 1's pass stores
    # iteration j's value over lane 0's iteration j + 1's, which the program
    # ordered the other way and no barrier restores.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(
            _store_rows_shifted_by_the_iteration,
            (torch.empty((8, 256)), torch.empty((264,))),
            **_ROWS_ON_TWO_LANES_CONFIG,
        )


def test_folding_store_per_thread_closes_the_device_loop_body() -> None:
    # One row per thread: thread r + 1's iteration j and thread r's
    # iteration j + 1 store one element, ordered by the barrier closing the
    # body.
    code = _generate(
        _store_rows_shifted_by_the_iteration,
        (torch.empty((8, 256)), torch.empty((264,))),
        **_ROWS_ONE_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_")
    assert _barrier_positions(device_loop.body) == [len(device_loop.body) - 1], code


def _flattened_store(stride: int | str, end: int | str | None) -> str:
    """A store of ``x[indices_0 * stride + indices_2]``, under the column tile's mask ``indices_2 < end`` unless ``end`` is None.

    ``stride`` and ``end`` are literals or expressions over kernel
    arguments (``K``, a dynamic shape's size).
    """
    store = f"{_pointer('x', f'indices_0 * {stride} + indices_2')}.store(c)\n"
    if end is None:
        return store
    return (
        f"mask_2 = cutlass.Int32(cute.arch.thread_idx()[0]) < 64 and indices_2 < {end}\n"
        f"if mask_2:\n    {store}"
    )


@pytest.mark.parametrize(
    ("stride", "end", "constants", "accepted"),
    [
        pytest.param(250, 250, None, True, id="masked_below_the_stride"),
        pytest.param(
            250,
            250,
            {"_BLOCK_SIZE_2": 64},
            True,
            id="masked_below_the_stride_known_step",
        ),
        pytest.param(
            250, 256, {"_BLOCK_SIZE_2": 64}, False, id="masked_past_the_stride"
        ),
        pytest.param(250, None, {"_BLOCK_SIZE_2": 64}, False, id="unmasked"),
        pytest.param(
            256, None, {"_BLOCK_SIZE_2": 64}, True, id="unmasked_exact_stride"
        ),
    ],
)
def test_flattened_store_in_a_repeated_device_loop_is_bounded_by_its_mask(
    stride: int, end: int | None, constants: dict[str, int] | None, accepted: bool
) -> None:
    # A flattened store ``x[stride * row + column]`` alone in the device loop
    # under the row lanes folds between the iterations when the column
    # term's range reaches the row stride.  The column tile's mask guarding
    # the store bounds the column it names (whatever the loop's step, known
    # or not): bounded below the stride the store is one lane's, bounded
    # past it or not at all the padded columns of one row are the next
    # row's first, unless the stride is exactly the columns' padded range.
    if accepted:
        assert _repeated_loop_nest(_flattened_store(stride, end), (), constants) == [
            "stmt"
        ]
    else:
        with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
            _repeated_loop_nest(_flattened_store(stride, end), (), constants)


_NEXT_COLUMN_READ_THEN_FLATTENED_STORE = (
    "mask_2 = cutlass.Int32(cute.arch.thread_idx()[0]) < 64 and indices_2 < K\n"
    "if mask_2:\n"
    f"    v = {_pointer('x', 'indices_0 * K + indices_2 + 1')}.load()\n"
    f"    {_pointer('x', 'indices_0 * K + indices_2')}.store(v)\n"
)


@pytest.mark.parametrize(
    ("body", "accepted"),
    [
        pytest.param(_flattened_store("K", "K"), True, id="masked_by_the_stride"),
        pytest.param(
            _flattened_store("2 * K", "K"), True, id="masked_below_twice_the_stride"
        ),
        pytest.param(_flattened_store("K", None), False, id="unmasked"),
        pytest.param(_flattened_store("K", "D"), False, id="masked_by_another_size"),
        pytest.param(_flattened_store(250, "K"), False, id="literal_stride"),
        pytest.param(
            _flattened_store("(K - 8)", "K"), False, id="stride_short_of_the_end"
        ),
        pytest.param(
            _NEXT_COLUMN_READ_THEN_FLATTENED_STORE,
            False,
            id="next_column_read_beside_the_store",
        ),
    ],
)
def test_flattened_store_with_a_dynamic_stride_is_bounded_by_its_mask(
    body: str, accepted: bool
) -> None:
    # The stride and the mask's end are a kernel argument ``K`` (a dynamic
    # shape's size): the column tile's mask bounds the column below ``K``,
    # which the proof takes as a radix, so ``K * row + column`` (``2 K * row
    # + column`` too) is one lane's row whatever value ``K`` has.  Without
    # the mask, or bounded by another size, the padded columns of one
    # lane's row may be the next lane's row's first.  A literal stride
    # under the symbolic bound has no bound the argument can use, and the
    # column tile's 256 columns reach past 250; a stride short of the end
    # (``K - 8``) and a read of the next column (row r's column ``K`` is
    # row r + 1's first) leave a part of the difference the bound does not
    # cover, and the argument knows no range for the column.
    if accepted:
        assert _repeated_loop_nest(body, (), {"_BLOCK_SIZE_2": 64}) == ["stmt"]
    else:
        with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
            _repeated_loop_nest(body, (), {"_BLOCK_SIZE_2": 64})


def _guarded_loop_nest(test: str, body: str, definitions: str) -> list[str]:
    """The kinds of the row lane loop's items after the pass, a device loop under ``if test:`` among them.

    The loop runs ``body`` once per column tile; ``definitions`` are
    statements around the nest, before the lanes' own.
    """
    (branch,) = ast.parse(
        f"if {test}:\n"
        "    for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(256),"
        " cutlass.Int32(_BLOCK_SIZE_2)):\n" + textwrap.indent(body, "        ")
    ).body
    scope = _threaded_column_scopes()[0]
    placement = lane_loop_distribution.LanePlacement("lane_0", [branch])
    lane_loop_distribution.add_thread_barriers(
        [placement],
        [scope],
        rename_groups={},
        axis_sizes={0: 64},
        definitions=[*ast.parse(definitions).body, *scope.setup],
        masks=frozenset(),
        constants={"_BLOCK_SIZE_2": 64},
    )
    return _kinds(placement.items)


_FLATTENED_STORE = f"{_pointer('x', 'indices_0 * 250 + indices_2')}.store(c)\n"


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(f"tile_offset_2 + {_LANE_COORDINATE} + 6", id="shifted"),
        pytest.param(f"tile_offset_2 + {_LANE_COORDINATE}", id="exact"),
    ],
)
def test_branch_around_a_device_loop_bounds_none_of_the_loops_own_indices(
    index: str,
) -> None:
    # ``if indices_2 < 250:`` around the device loop tests the ``indices_2``
    # defined before it (the thread's coordinate: the branch always runs).
    # The loop defines ``indices_2`` again, one column tile's per iteration
    # (256 or 262 columns per row), so the flattened stores of two lanes'
    # rows fold across the row stride 250 in an order the lanes' passes
    # interleave: the branch's bound is not the loop's index's.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _guarded_loop_nest(
            "indices_2 < 250",
            f"indices_2 = {index}\n{_FLATTENED_STORE}",
            f"indices_2 = {_LANE_COORDINATE}\n",
        )


def test_branch_around_a_device_loop_bounds_the_index_it_tests() -> None:
    # The same branch around a loop storing through the ``indices_2`` the
    # branch tested (five columns per thread, 0 to 315, defined before
    # both) bounds it below the row stride: each store is one lane's row,
    # however often the loop repeats it.
    assert _guarded_loop_nest(
        "indices_2 < 250", _FLATTENED_STORE, f"indices_2 = 5 * {_LANE_COORDINATE}\n"
    ) == ["stmt"]


def test_unbounded_columns_past_the_stride_reject_the_config() -> None:
    # Without the bound, the columns 250 to 315 of one lane's row are the
    # next lane's row's first.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _guarded_loop_nest(
            "indices_0 < 8", _FLATTENED_STORE, f"indices_2 = 5 * {_LANE_COORDINATE}\n"
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_flattened_rows(v: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # Row r's iteration j stores its columns 64 j onwards to out[n r + column]:
    # one row's elements, the column tile's padding past n masked.
    m, n = v.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            out[tile0.index[:, None] * n + tile1.index[None, :]] = v[tile0, tile1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_flattened_rows_overlapping(
    v: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # A row stride eight short of the columns: row r's last eight columns
    # (a late iteration) are row r + 1's first eight (iteration 0), and the
    # program's value is the late iteration's.
    m, n = v.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            out[tile0.index[:, None] * (n - 8) + tile1.index[None, :]] = v[tile0, tile1]
    return out


@pytest.mark.parametrize(
    "config",
    [_DEVICE_LOOP_COLUMNS_CONFIG, _DEVICE_LOOP_ROWS_PER_THREAD_CONFIG],
    ids=["lanes", "threads"],
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_flattened_store_with_a_masked_tail_needs_no_barrier(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    # The column tile's mask bounds the flattened address's column term
    # below the row stride: the store's instances are one row's each, on
    # the lanes as across the iterations, and the padded columns of a
    # partial tile (250 of 256) never alias the next row.
    m, n = shape
    code = _generate(
        _store_flattened_rows, (torch.empty(shape), torch.empty((m * n,))), **config
    )
    assert not _barriers(_kernel_function(code)), code


def test_overlapping_flattened_store_rejects_the_config() -> None:
    # Rows over two threads and a lane loop of two: lane 1's pass stores
    # its row's first eight columns over lane 0's row's last eight, stored
    # in lane 0's last iteration, which the program ordered the other way.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(
            _store_flattened_rows_overlapping,
            (torch.empty((6, 250)), torch.empty((6 * 250,))),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


def test_overlapping_flattened_store_per_thread_closes_the_device_loop_body() -> None:
    # One row per thread: thread r + 1's iteration 0 and thread r's last
    # iteration store one element, ordered by the barrier closing the body.
    code = _generate(
        _store_flattened_rows_overlapping,
        (torch.empty((6, 250)), torch.empty((6 * 250,))),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_")
    assert _barrier_positions(device_loop.body) == [len(device_loop.body) - 1], code


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _store_flattened_rows_overlapping_dynamic(
    v: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    # The overlapping store with dynamic shapes: the stride is a size the
    # host unpacked and passed to the kernel.
    m, n = v.shape
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n):
            out[tile0.index[:, None] * (n - 8) + tile1.index[None, :]] = v[tile0, tile1]
    return out


def test_dynamic_overlapping_flattened_store_rejects_the_config() -> None:
    # The stride is a kernel argument, one uniform value (the host's
    # unpacking of the shape defines nothing the body reads), so the store
    # is paired with itself and the fold found, rather than left unmodelled
    # behind an unknown value and accepted.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(
            _store_flattened_rows_overlapping_dynamic,
            (torch.empty((6, 250)), torch.empty((6 * 250,))),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


def test_dynamic_overlapping_flattened_store_per_thread_closes_the_device_loop_body() -> (
    None
):
    code = _generate(
        _store_flattened_rows_overlapping_dynamic,
        (torch.empty((6, 250)), torch.empty((6 * 250,))),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    function = _kernel_function(code)
    (device_loop,) = _loops(function, "tile_offset_")
    assert _barrier_positions(device_loop.body) == [len(device_loop.body) - 1], code


def _root_body_kinds(body: str) -> list[str]:
    """The kinds of a root body's statements after the pass: no lane loops, 128 threads."""
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(body).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 128},
        definitions=[],
        masks=frozenset(),
    )
    return _kinds(items)


_EPILOGUE_FRAGMENT_SIZING = """\
if True:
    view = cute.group_modes(partition, 3, cute.rank(partition))
    fragment = cute.make_rmem_tensor(view[None, None, None, 0].shape, cutlass.Float32)
    common = cute.max_common_layout(fragment.layout, view[None, None, None, 0].layout)
    bits = min(view.iterator.alignment * 8, cute.size(common) * 16, 256)
    for subtile in cutlass.range(4, unroll_full=True):
        if active:
            piece = view[None, None, None, cutlass.Int32(subtile)]
            cute.copy(atom, fragment, piece)
"""


def test_metadata_queries_of_a_tensor_view_are_no_accesses() -> None:
    # A tcgen05 epilogue sizes its register fragment and its copy atom from
    # a slice of the per-thread output partition: ``view[...].shape``,
    # ``view[...].layout`` and ``view.iterator.alignment`` read no element,
    # so the calls around them, unknown to the pass, touch no tensor and the
    # branch holds no pair to order.
    assert _root_body_kinds(_EPILOGUE_FRAGMENT_SIZING) == ["stmt"]


def test_pointer_inside_an_unknown_call_in_a_branch_rejects_the_config() -> None:
    # The tensor's pointer handed to such a call is an access of unknown
    # effects, racing the per-thread load beside it where no barrier can be
    # placed.
    with pytest.raises(exc.BackendUnsupported, match="inside a branch"):
        _root_body_kinds(
            "if True:\n"
            "    v = view[cutlass.Int32(cute.arch.thread_idx()[0])]\n"
            "    bits = min(view.iterator.toint(), 8)\n"
        )


_GATHERED_ROWS = (
    "row = start + cutlass.Int64(indices_0)\n"
    f"v = {_pointer('x', 'row * 256 + indices_2')}.load()\n"
    f"{_pointer('x', 'row * 256 + indices_2')}.store(v + 1.0)\n"
)
_BLOCK_INDEX = "pid = cutlass.Int32(cute.arch.block_idx()[0])\n"
_UNIFORM_START = _BLOCK_INDEX + f"start = {_pointer('offsets', 'pid')}.load()\n"


@pytest.mark.parametrize(
    ("definitions", "accepted"),
    [
        pytest.param(_UNIFORM_START, True, id="block_index"),
        pytest.param(
            _BLOCK_INDEX + f"start = {_pointer('offsets', 'pid')}.load() "
            "if cutlass.Int32(pid) < 4 else cutlass.Int64(0)\n",
            True,
            id="masked_by_the_block_index",
        ),
        pytest.param(
            _BLOCK_INDEX
            + f"start = {_pointer('offsets', 'pid + cutlass.Int32(cute.arch.thread_idx()[0])')}.load()\n",
            False,
            id="per_thread_address",
        ),
        pytest.param(
            _BLOCK_INDEX + f"start = {_pointer('offsets', 'pid')}.load() "
            "if cutlass.Int32(cute.arch.thread_idx()[0]) < 4 else cutlass.Int64(0)\n",
            False,
            id="masked_per_thread",
        ),
        pytest.param(
            _UNIFORM_START + f"{_pointer('offsets', 'pid')}.store(c)\n",
            False,
            id="written_tensor",
        ),
        pytest.param(
            _BLOCK_INDEX
            + "carry = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Int64)\n"
            + f"start = {_pointer('carry', '0')}.load()\n",
            False,
            id="register_tensor",
        ),
        pytest.param(
            _BLOCK_INDEX + "shared = cute.make_tensor("
            "cute.arch.alloc_smem(cutlass.Int64, 4), cute.make_layout((4,)))\n"
            + f"start = {_pointer('shared', '0')}.load()\n",
            False,
            id="tensor_made_in_view",
        ),
    ],
)
def test_uniform_load_around_the_nest_is_one_value_for_the_lanes(
    definitions: str, accepted: bool
) -> None:
    # Rows gathered through a segment's start (a jagged batch): the device
    # loop's load and store of ``x[256 (start + row) + column]`` under the
    # row lanes are one lane's, the column staying below the row stride,
    # when the start is one value for every lane and thread, which it is
    # when the block's index loads it from a kernel argument nothing in
    # view writes (under a uniform mask or none).  Loaded per thread, masked
    # per thread, from a tensor the kernel also stores to, from a thread's
    # own register tensor or from a tensor made in view (some thread's
    # fill), the start may be any row's and the lanes' passes interleave.
    if accepted:
        assert _repeated_loop_nest(
            _GATHERED_ROWS, (), {"_BLOCK_SIZE_2": 64}, definitions
        ) == ["stmt"]
    else:
        with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
            _repeated_loop_nest(_GATHERED_ROWS, (), {"_BLOCK_SIZE_2": 64}, definitions)


_GATHERED_ROWS_DYNAMIC = (
    "row = start + cutlass.Int64(indices_0)\n"
    "mask_2 = cutlass.Int32(cute.arch.thread_idx()[0]) < 64 and indices_2 < K\n"
    "if mask_2:\n"
    f"    v = {_pointer('x', 'row * K + indices_2')}.load()\n"
    f"    {_pointer('x', 'row * K + indices_2')}.store(v + 1.0)\n"
)


def test_uniform_start_rows_with_a_dynamic_stride_are_one_lanes() -> None:
    # The jagged batch with dynamic shapes: the row stride ``K`` is a kernel
    # argument, the column tile's mask bounds the column below it and the
    # segment's start is one value for every lane and thread, so ``K *
    # (start + row) + column`` is one lane's row and the loop needs no
    # split.  Loaded per thread, the start may be any row's.
    assert _repeated_loop_nest(
        _GATHERED_ROWS_DYNAMIC, (), {"_BLOCK_SIZE_2": 64}, _UNIFORM_START
    ) == ["stmt"]
    per_thread = _BLOCK_INDEX + (
        f"start = {_pointer('offsets', 'pid + cutlass.Int32(cute.arch.thread_idx()[0])')}"
        ".load()\n"
    )
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _repeated_loop_nest(
            _GATHERED_ROWS_DYNAMIC, (), {"_BLOCK_SIZE_2": 64}, per_thread
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _increment_segment_rows(
    offsets: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # Block b's rows begin at offsets[b], one value every thread and lane of
    # the block loads alike; its rows are updated in place through the
    # flattened address, one lane's row each.
    rows, n = v.shape
    for b in hl.grid(offsets.size(0)):
        start = offsets[b]
        for tile_r in hl.tile(rows):
            for tile_c in hl.tile(n):
                index = (start + tile_r.index)[:, None] * n + tile_c.index[None, :]
                x[index] = x[index] + v[tile_r, tile_c]
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _increment_gathered_rows(
    offsets: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # Each row's place is gathered per row: two lanes' rows may be one.
    rows, n = v.shape
    for tile_r in hl.tile(rows):
        places = offsets[tile_r]
        for tile_c in hl.tile(n):
            index = places[:, None] * n + tile_c.index[None, :]
            x[index] = x[index] + v[tile_r, tile_c]
    return x


_SEGMENT_ARGS = (
    torch.zeros(3, dtype=torch.int32),
    torch.empty((8, 128)),
    torch.empty((24 * 128,)),
)


def test_segment_rows_through_a_uniform_start_need_no_barrier() -> None:
    # The segment's start is loaded by the block's index from a tensor the
    # kernel never writes: one value for every lane and thread, so the rows
    # ``start + r`` are one lane's each and the in-place update through the
    # flattened address needs neither a barrier nor a split of the loop.
    code = _generate(
        _increment_segment_rows, _SEGMENT_ARGS, **_DEVICE_LOOP_COLUMNS_CONFIG
    )
    assert not _barriers(_kernel_function(code)), code


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _store_segment_rows_dynamic(
    offsets: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # The jagged kernels' form with dynamic shapes: the row stride is a size
    # the host unpacked and passed to the kernel, one uniform value, and
    # the rows are stored through the flattened address.
    rows, n = v.shape
    for b in hl.grid(offsets.size(0)):
        start = offsets[b]
        for tile_r in hl.tile(rows):
            for tile_c in hl.tile(n):
                index = (start + tile_r.index)[:, None] * n + tile_c.index[None, :]
                x[index] = v[tile_r, tile_c]
    return x


def test_segment_rows_with_a_dynamic_stride_need_no_barrier() -> None:
    # The column tile's mask bounds the column below the row stride ``n``,
    # a kernel argument the proof takes as a radix, so the rows ``start +
    # r`` are one lane's each whatever value ``n`` has: no barrier and no
    # split of the column loop.
    code = _generate(
        _store_segment_rows_dynamic, _SEGMENT_ARGS, **_DEVICE_LOOP_COLUMNS_CONFIG
    )
    assert not _barriers(_kernel_function(code)), code


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _store_segment_rows_under_the_column_loop(
    offsets: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # The jagged layer norm's form: the columns are the outer loop (lanes
    # over threads), the segment's rows the inner device loop, and the
    # column tile's mask reads the outer loop's offset and lane.
    rows, n = v.shape
    for b in hl.grid(offsets.size(0)):
        start = offsets[b]
        for tile_c in hl.tile(n):
            for tile_r in hl.tile(rows):
                index = (start + tile_r.index)[:, None] * n + tile_c.index[None, :]
                x[index] = v[tile_r, tile_c]
    return x


# The columns (block 0) two lanes over 64 threads, the rows (block 1) two
# lanes over two threads.
_COLUMN_LOOP_OUTSIDE_CONFIG = {
    "block_sizes": [128, 4],
    "num_threads": [64, 2],
    "cute_vector_widths": [1, 1],
}


def test_segment_rows_under_the_column_loop_need_no_barrier() -> None:
    # The inner row loop's store is bounded by the outer column tile's mask,
    # which reads the outer loop's offset and lane variable: the barrier
    # pass of the inner body knows them as the loops around it (nonnegative,
    # one value each throughout the body), so the mask bounds the column
    # below the stride ``n`` and the inner body needs no closing barrier.
    code = _generate(
        _store_segment_rows_under_the_column_loop,
        _SEGMENT_ARGS,
        **_COLUMN_LOOP_OUTSIDE_CONFIG,
    )
    assert not _barriers(_kernel_function(code)), code


def test_gathered_rows_in_a_repeated_device_loop_reject_the_config() -> None:
    # Gathered per row, two lanes' rows may be one element's, in an order
    # the lanes' passes over the column loop interleave.
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _generate(
            _increment_gathered_rows,
            (torch.zeros(8, dtype=torch.int64), *_SEGMENT_ARGS[1:]),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


_OUTER_COLUMN = "indices_2 = cutlass.Int32(cute.arch.thread_idx()[0])\n"
_STRIDE_250_STORE = (
    f"if mask_2:\n    {_pointer('x', 'indices_0 * 250 + indices_2')}.store(c)\n"
)


@pytest.mark.parametrize(
    ("definitions", "body", "accepted"),
    [
        pytest.param(
            "", f"mask_2 = indices_2 < 250\n{_STRIDE_250_STORE}", True, id="own_mask"
        ),
        pytest.param(
            f"{_OUTER_COLUMN}mask_2 = indices_2 < 250\n",
            _STRIDE_250_STORE,
            False,
            id="outer_mask",
        ),
        pytest.param(
            f"{_OUTER_COLUMN}outer_column = indices_2\nmask_2 = outer_column < 250\n",
            _STRIDE_250_STORE,
            False,
            id="outer_mask_through_a_chain",
        ),
    ],
)
def test_a_mask_the_nest_defines_bounds_the_outer_index_not_the_loops(
    definitions: str, body: str, accepted: bool
) -> None:
    # ``mask_2 = indices_2 < 250`` around the loop reads the outer ``indices_2
    # = tid`` (always true there); the loop's own ``indices_2 = tile_offset_2
    # + tid`` runs to 256, so the flattened store with stride 250 reaches the
    # next row's first columns.  The conjunct is read as its definitions
    # left it and bounds nothing of the loop's index: the loop is rejected
    # like an unmasked one.  Its own mask over its own index bounds the
    # store to one row.
    if accepted:
        assert _repeated_loop_nest(body, (), {"_BLOCK_SIZE_2": 64}, definitions) == [
            "stmt"
        ]
    else:
        with pytest.raises(
            helion.exc.BackendUnsupported, match="repeated whole once per lane_0"
        ):
            _repeated_loop_nest(body, (), {"_BLOCK_SIZE_2": 64}, definitions)


@pytest.mark.parametrize(
    ("inner_mask", "expected"),
    [(True, ["stmt"]), (False, ["stmt", "sync"])],
    ids=["own_mask", "outer_mask"],
)
def test_a_mask_the_nest_defines_bounds_nothing_across_the_outer_iterations(
    inner_mask: bool, expected: list[str]
) -> None:
    # The same mask over an outer ``indices_2``, read inside an inner device
    # loop that redefines the name, checked across the iterations of the
    # enclosing row loop (two rows per iteration on two threads): the
    # flattened stores of stride 250 reach the next row, so the inner body
    # closes with a barrier unless the mask is the inner loop's own.
    header = _loop_header(
        "for tile_offset_1 in range(cutlass.Int32(0), cutlass.Int32(1024), cutlass.Int32(2))"
    )
    mask = "    mask_2 = indices_2 < 250\n" if inner_mask else ""
    items: list[ast.AST | lane_loop_distribution.LanePlacement] = list(
        ast.parse(
            "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(256), cutlass.Int32(64)):\n"
            "    indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[0])\n"
            f"{mask}"
            f"    if mask_2:\n        {_pointer('x', 'indices_0 * 250 + indices_2')}.store(v)\n"
        ).body
    )
    lane_loop_distribution.add_thread_barriers(
        items,
        [],
        rename_groups={},
        axis_sizes={0: 64, 1: 2},
        definitions=ast.parse(
            "indices_0 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
            f"{_OUTER_COLUMN}mask_2 = indices_2 < 250\n"
        ).body,
        masks=frozenset(),
        loop_variables=frozenset({"tile_offset_1"}),
        loop_body=items,
        loop_headers=[header],
    )
    assert _kinds(items) == expected, _kinds(items)


_NEXT_COLUMN_LOAD = f"{_pointer('x', 'indices_0 * 250 + indices_2 + 1')}.load()"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            f"mask_2 = indices_2 < 250\nif mask_2 and {_NEXT_COLUMN_LOAD} > 0.0:\n"
            f"    {_pointer('x', 'indices_0 * 250 + indices_2')}.store(c)\n",
            id="branch_test",
        ),
        pytest.param(
            f"mask_2 = indices_2 < 250\nwhile {_NEXT_COLUMN_LOAD} > 0.0:\n"
            + textwrap.indent(_STRIDE_250_STORE, "    "),
            id="while_test",
        ),
        pytest.param(
            f"mask_2 = indices_2 < 250\nfor q in range(cutlass.Int32({_NEXT_COLUMN_LOAD})):\n"
            + textwrap.indent(_STRIDE_250_STORE, "    "),
            id="loop_header",
        ),
    ],
)
def test_a_load_in_a_test_or_header_beside_a_store_of_its_tensor_is_rejected(
    body: str,
) -> None:
    # The next column's element, which the next lane stores, read in a
    # branch's test, a while's test or a loop's header: no body statement
    # holds the load, so the pairs over the body never see it; the loop is
    # rejected like one reading it in a statement.
    with pytest.raises(
        helion.exc.BackendUnsupported, match="in a branch test or a loop header"
    ):
        _repeated_loop_nest(body, (), {"_BLOCK_SIZE_2": 64})


def test_a_load_of_another_tensor_in_the_test_leaves_the_store_alone() -> None:
    body = (
        f"mask_2 = indices_2 < 250\n"
        f"if mask_2 and {_pointer('y', 'indices_0 * 250 + indices_2 + 1')}.load() > 0.0:\n"
        f"    {_pointer('x', 'indices_0 * 250 + indices_2')}.store(c)\n"
    )
    assert _repeated_loop_nest(body, (), {"_BLOCK_SIZE_2": 64}) == ["stmt"]
