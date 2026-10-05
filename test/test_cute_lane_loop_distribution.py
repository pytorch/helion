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
the same way, never redistributed).  A placement is checked like the nest it
replaces: an inner loop nested inside an outer one (by its hoisted packets, or
because a statement needs both and each loop is materialized once) runs its
statements inside the outer loop whether or not their values change with it.
A ``tile.begin`` access carries no lane mask, so on a partial tile it is
placed as on a full one.
"""

from __future__ import annotations

import ast

from examples.concatenate import concat2d_dim1_simple
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.cute import lane_loop_distribution
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


def _loops(node: ast.AST, prefix: str) -> list[ast.For]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.For)
        and isinstance(child.target, ast.Name)
        and child.target.id.startswith(prefix)
    ]


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


def test_tile_uniform_store_in_an_inner_loops_lane_nest_is_checked() -> None:
    # A device loop's lane loops are built around its body before the body
    # exists and are never redistributed, so the nest must be the tile
    # program.  The zeroing store ignores the column lane: before the copy it
    # would be re-applied after the first lane's copy of the same element ...
    args = (torch.empty((8, 512)), torch.empty((8, 512)))
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant store to out would repeat"
    ):
        _generate(
            _zero_first_column_then_copy_in_inner_tile, args, **_INNER_LANES_CONFIG
        )
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
