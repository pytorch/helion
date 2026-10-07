"""A launch axis sized by a kernel argument is launched with its extent and validated at launch.

``hl.tile(n, block_size=bsz)`` with ``bsz`` an int argument of a kernel bound
with ``static_shapes=False`` renders its thread extent as the host constant
``_BLOCK_SIZE_n``.  Static tracking holds a one for that axis, so whenever
another axis had static threads the launch fell back to the static shape
(``block=(1, 32, 1)`` for a row tile over 32 column threads: one row per CTA
written, 35 of 40 rows wrong), and the symbolic shape it emitted otherwise went
unchecked.  The axis now keeps its extent in a
``block=_cute_checked_block_dims((...))`` tuple that the host wrapper validates
at launch; a body that combines threads under the static layout (a warp
reduction beside the symbolic axis, a reduction over the symbolic dim itself)
is declined; a literal block size is still checked at codegen and emitted bare.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.cute.thread_budget import checked_thread_block_dims
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])


def _doubled_rows(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """Row tiles of ``bsz`` rows, a kernel argument: the launch shape is symbolic."""
    out = torch.empty_like(x)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return out


def _doubled_cols(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """The argument-sized tile is the inner (column) loop."""
    out = torch.empty_like(x)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return out


def _doubled_grid(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """One ND grid whose first block is argument-sized and whose second is a literal."""
    out = torch.empty_like(x)
    for tile_b, tile_m in hl.tile([x.size(0), x.size(1)], block_size=[bsz, 32]):
        out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return out


def _row_sums(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """Argument-sized rows; the column loop's sum combines the column threads."""
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        acc = hl.zeros([tile_b], dtype=x.dtype)
        for tile_m in hl.tile(x.size(1)):
            acc += x[tile_b, tile_m].sum(dim=1)
        out[tile_b] = acc
    return out


def _column_sums(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A reduction over the argument-sized dim itself."""
    out = torch.zeros([x.size(1)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].sum(dim=0))
    return out


def _doubled_rows_literal(x: torch.Tensor) -> torch.Tensor:
    """A literal block size: the launch shape is static."""
    out = torch.empty_like(x)
    for tile_b in hl.tile(x.size(0), block_size=8):
        for tile_m in hl.tile(x.size(1)):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return out


def _row_sums_literal(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=8):
        acc = hl.zeros([tile_b], dtype=x.dtype)
        for tile_m in hl.tile(x.size(1)):
            acc += x[tile_b, tile_m].sum(dim=1)
        out[tile_b] = acc
    return out


def _rows_count_columns(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A tile-uniform (along rows) atomic beside a store that reads the row index: once per row tile per column."""
    out = torch.empty_like(x)
    count = torch.zeros([x.size(1)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
            hl.atomic_add(count, [tile_m], 1.0)
    return count


def _rows_count_columns_only(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """The same atomic alone: the body never reads the row index, so the row axis keeps one thread."""
    count = torch.zeros([x.size(1)], dtype=x.dtype, device=x.device)
    for _tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            hl.atomic_add(count, [tile_m], 1.0)
    return count


def _rows_count_all(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A scalar atomic beside the store: once per (row tile, column tile)."""
    out = torch.empty_like(x)
    count = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
            hl.atomic_add(count, [0], 1.0)
    return count


def _rows_count_tiles(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A tile-uniform atomic keyed on ``tile.id`` after the column loop: once per row tile."""
    out = torch.empty_like(x)
    count = torch.zeros([(x.size(0) + bsz - 1) // bsz], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [tile_b.id], 1.0)
    return count


def _column_sums_atomic(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """An atomic indexed by the column whose value varies along the rows: every element is added."""
    out = torch.zeros([x.size(1)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=bsz):
        for tile_m in hl.tile(x.size(1)):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m])
    return out


def _column_sums_atomic_literal(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(1)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=8):
        for tile_m in hl.tile(x.size(1)):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m])
    return out


def _cols_count_rows_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """Static rows, argument-sized columns; a per-row atomic once the column loop has exited."""
    out = torch.empty_like(x)
    count = torch.zeros([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [tile_b], 1.0)
    return count


def _cols_count_rows_before(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """The per-row atomic before the column loop is entered."""
    out = torch.empty_like(x)
    count = torch.zeros([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        hl.atomic_add(count, [tile_b], 1.0)
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return count


def _cols_count_all_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    out = torch.empty_like(x)
    count = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [0], 1.0)
    return count


def _cols_count_all_before(x: torch.Tensor, bsz: int) -> torch.Tensor:
    out = torch.empty_like(x)
    count = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        hl.atomic_add(count, [0], 1.0)
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return count


def _cols_count_tiles_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    out = torch.empty_like(x)
    count = torch.zeros([(x.size(0) + 31) // 32], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [tile_b.id], 1.0)
    return count


def _cols_count_tiles_before(x: torch.Tensor, bsz: int) -> torch.Tensor:
    out = torch.empty_like(x)
    count = torch.zeros([(x.size(0) + 31) // 32], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        hl.atomic_add(count, [tile_b.id], 1.0)
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
    return count


def _cols_count_gathered_after(
    x: torch.Tensor, idx: torch.Tensor, bsz: int
) -> torch.Tensor:
    """A gather-indexed atomic after the column loop: covered along the rows by its index tensor."""
    out = torch.empty_like(x)
    count = torch.zeros([10], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [idx[tile_b]], 1.0)
    return count


def _carried_register_tile(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A ``[rows, bsz]`` register tile carried through the argument-sized column loop: its ``bsz`` dim is a reduction dim sized by a host scalar."""
    out = torch.zeros([x.size(0), bsz], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        acc = hl.zeros([tile_b, bsz], dtype=x.dtype)
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            acc = acc + x[tile_b, tile_m]
        out[tile_b, :] = acc
    return out


def _kernel(fn: Callable[..., torch.Tensor] = _doubled_rows) -> helion.Kernel:
    return helion.kernel(
        fn, backend="cute", static_shapes=False, autotune_effort="none"
    )


def _config(bound: BoundKernel, **overrides: object) -> helion.Config:
    config = dict(bound.config_spec.default_config().config)
    config.update(overrides)
    return helion.Config.from_dict(config)


def _one_thread_config(bound: BoundKernel) -> helion.Config:
    """One column thread: static tracking infers no launch shape, so the row tile's argument-sized extent is emitted symbolically."""
    return _config(bound, num_threads=[1])


def _render(fn: Callable[..., torch.Tensor], *args: object, **overrides: object) -> str:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(fn), args)
        return bound.to_code(_config(bound, **overrides))


def _launch_block(code: str) -> str:
    """The ``block=...`` argument of the launcher call."""
    (line,) = [line for line in code.splitlines() if "_launcher(" in line]
    return line[line.index("block=") : -1]


_CHECKED_IMPORT = (
    "from helion._compiler.cute.thread_budget import checked_thread_block_dims"
    " as _cute_checked_block_dims"
)


def test_an_argument_sized_launch_shape_is_checked_at_launch() -> None:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), (torch.randn(40, 100), 8))
        code = bound.to_code(_one_thread_config(bound))
    assert "block=_cute_checked_block_dims((_BLOCK_SIZE_0, 1, 1))" in code, code
    assert _CHECKED_IMPORT in code, code


def test_a_static_launch_shape_is_checked_at_codegen_and_not_wrapped() -> None:
    code = _render(_doubled_rows_literal, torch.randn(40, 100))
    assert _launch_block(code) == "block=(32, 8, 1)", code
    assert "_cute_checked_block_dims" not in code, code


@pytest.mark.parametrize(
    ("fn", "overrides", "block"),
    [
        pytest.param(_doubled_rows, {}, "(_BLOCK_SIZE_0, 32, 1)", id="rows"),
        pytest.param(_doubled_cols, {}, "(32, _BLOCK_SIZE_1, 1)", id="cols"),
        pytest.param(
            _doubled_rows,
            {"num_threads": [8]},
            "(_BLOCK_SIZE_0, 8, 1)",
            id="rows_lane_loop",
        ),
        pytest.param(_doubled_grid, {}, "(_BLOCK_SIZE_0, 32, 1)", id="grid"),
    ],
)
def test_an_argument_sized_axis_keeps_its_extent_beside_static_threads(
    fn: Callable[..., torch.Tensor], overrides: dict[str, object], block: str
) -> None:
    """The static shape held a one for the argument-sized axis and was launched as is (``block=(1, 32, 1)`` for the row tile: one row per CTA)."""
    code = _render(fn, torch.randn(40, 100), 8, **overrides)
    assert _launch_block(code) == f"block=_cute_checked_block_dims({block})", code
    assert _CHECKED_IMPORT in code, code
    for axis in (0, 1):
        assert f"cute.arch.thread_idx()[{axis}]" in code, code


def test_a_warp_reduction_beside_an_argument_sized_axis_is_declined() -> None:
    """``warp_reduction_sum(threads_in_group=32)`` over the column threads spans one warp only while the row axis has one thread; launched with ``bsz`` row threads it would sum across rows."""
    with pytest.raises(exc.BackendUnsupported, match="warp_reduction_sum"):
        _render(_row_sums, torch.randn(40, 100), 8)


def test_a_serial_column_loop_beside_an_argument_sized_axis_is_launched() -> None:
    """One column thread serializes the sum in a lane loop, which no launch shape can break."""
    code = _render(_row_sums, torch.randn(40, 100), 8, num_threads=[1])
    assert (
        _launch_block(code) == "block=_cute_checked_block_dims((_BLOCK_SIZE_0, 1, 1))"
    )
    assert "for lane_1 in range(32):" in code, code
    assert "warp_reduction" not in code, code


@pytest.mark.parametrize("num_threads", [None, 1], ids=["default", "one_thread"])
def test_a_reduction_over_an_argument_sized_tile_dim_is_declined(
    num_threads: int | None,
) -> None:
    """Static tracking found no thread axis for the dim, so the reduction passed each thread's own element through as the total."""
    overrides = {} if num_threads is None else {"num_threads": [num_threads]}
    with pytest.raises(exc.BackendUnsupported, match="static thread extent"):
        _render(_column_sums, torch.randn(40, 100), 8, **overrides)


_LEADER_GUARD = "cute.arch.thread_idx()[0] == 0"
_LANE_CONFIGS = [
    pytest.param({}, id="default"),
    pytest.param({"num_threads": [1]}, id="one_thread"),
    pytest.param({"num_threads": [8]}, id="lane_loop"),
]
_COUNT_KERNELS = [
    pytest.param(_rows_count_columns, id="columns"),
    pytest.param(_rows_count_all, id="all"),
    pytest.param(_rows_count_tiles, id="tiles"),
]


def _atomic_guard(code: str) -> str:
    """The test of the ``if`` guarding the kernel's one atomic."""
    lines = code.splitlines()
    (position,) = [
        index for index, line in enumerate(lines) if "cute.arch.atomic_add(" in line
    ]
    guard = lines[position - 1].strip()
    assert guard.startswith("if ") and guard.endswith(":"), code
    return guard[3:-1]


@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
@pytest.mark.parametrize("fn", _COUNT_KERNELS)
def test_a_tile_uniform_atomic_beside_an_argument_sized_axis_runs_on_its_leader(
    fn: Callable[..., torch.Tensor], overrides: dict[str, object]
) -> None:
    """The row block had no entry in the grid's thread-axis map (its extent is not static), so the leader predicate skipped the row axis and the atomic ran once per row thread (40 per column at bsz=8 instead of 5); under a lane loop the lane-tile bounds simplifier then erased the guard from the static shape's one."""
    code = _render(fn, torch.randn(40, 100), 8, **overrides)
    assert _LEADER_GUARD in _atomic_guard(code), code
    assert _launch_block(code).startswith(
        "block=_cute_checked_block_dims((_BLOCK_SIZE_0, "
    ), code


def test_a_tile_uniform_atomic_on_a_dead_argument_sized_axis_keeps_one_thread() -> None:
    """The body never reads the row index: the leader guard is not such a read, and the axis keeps its one thread."""
    code = _render(_rows_count_columns_only, torch.randn(40, 100), 8)
    assert _launch_block(code) == "block=(1, 32, 1)", code
    assert _LEADER_GUARD in _atomic_guard(code), code


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        pytest.param(_column_sums_atomic, (8,), id="argument_sized"),
        pytest.param(_column_sums_atomic_literal, (), id="literal"),
    ],
)
def test_an_atomic_whose_value_varies_along_an_uncovered_axis_runs_on_every_thread(
    fn: Callable[..., torch.Tensor], args: tuple[object, ...]
) -> None:
    """``hl.atomic_add(out, [tile_m], x[tile_b, tile_m])`` adds every row's element; the leader predicate collapsed the row axis to its leader and kept the first row of each tile (the literal twin too)."""
    code = _render(fn, torch.randn(40, 100), *args)
    assert _atomic_guard(code) == "mask_0 and mask_1", code


_COUNTS_AROUND_THE_COLUMN_LOOP = [
    pytest.param(_cols_count_rows_after, id="rows_after"),
    pytest.param(_cols_count_rows_before, id="rows_before"),
    pytest.param(_cols_count_all_after, id="all_after"),
    pytest.param(_cols_count_all_before, id="all_before"),
    pytest.param(_cols_count_tiles_after, id="tiles_after"),
    pytest.param(_cols_count_tiles_before, id="tiles_before"),
]


def _symbolic_axis(code: str) -> int:
    """The launch axis the argument-sized column block is launched on."""
    block = _launch_block(code)
    assert block.startswith("block=_cute_checked_block_dims(("), code
    entries = block[len("block=_cute_checked_block_dims((") : -2].split(", ")
    return entries.index("_BLOCK_SIZE_1")


@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
@pytest.mark.parametrize("fn", _COUNTS_AROUND_THE_COLUMN_LOOP)
def test_a_tile_uniform_atomic_around_an_argument_sized_inner_loop_runs_on_its_leader(
    fn: Callable[..., torch.Tensor], overrides: dict[str, object]
) -> None:
    """With the column loop exited or not yet entered its axis is neither active nor, having no recorded size, a ghost of ``max_thread_block_dims``, so the atomic ran once per column thread (8 per row at bsz=8 instead of 1)."""
    code = _render(fn, torch.randn(40, 100), 8, **overrides)
    axis = _symbolic_axis(code)
    assert f"cute.arch.thread_idx()[{axis}] == 0" in _atomic_guard(code), code


def test_a_gathered_atomic_around_an_argument_sized_inner_loop_runs_on_its_leader() -> (
    None
):
    idx = (torch.arange(40) % 10).to(torch.int32)
    code = _render(_cols_count_gathered_after, torch.randn(40, 100), idx, 8)
    axis = _symbolic_axis(code)
    assert f"cute.arch.thread_idx()[{axis}] == 0" in _atomic_guard(code), code


@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
def test_a_register_tile_dim_sized_by_a_kernel_argument_is_declined(
    overrides: dict[str, object],
) -> None:
    """The dim's persistent reduction took its thread count from the bound value's hint (8) and laid the dim one element per thread on it: at another value the elements past the hint were dropped, and the stores of the per-thread column sums raced (every column of ``out`` got one phase's sum)."""
    with pytest.raises(exc.BackendUnsupported, match="int argument bsz"):
        _render(_carried_register_tile, torch.randn(40, 100), 8, **overrides)


def _row_sums_of_an_aliased_width(x: torch.Tensor) -> torch.Tensor:
    """``m, n = x.size()`` names the width: its size symbol then carries the host local's name, and is a tensor size all the same."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def test_a_persistent_reduction_over_an_aliased_tensor_width_renders() -> None:
    """The decline keyed on the symbol's origin class once read the aliased width as a host scalar; the int argument is told by its provenance now."""
    code = _render(_row_sums_of_an_aliased_width, torch.randn(40, 16))
    assert "warp_reduction_sum" in code, code
    assert "_RDIM_SIZE_" in code, code


@pytest.mark.parametrize(
    ("kernel_name", "args"),
    [
        pytest.param("sum_aot", (torch.randn(64, 16),), id="sum_aot"),
        pytest.param(
            "rms_norm_batched",
            (torch.randn(64, 16, dtype=torch.bfloat16), 1e-5),
            id="rms_norm_batched",
        ),
        pytest.param(
            "rms_norm_oneshot",
            (torch.randn(1024, 512, dtype=torch.bfloat16), 1e-5),
            id="rms_norm_oneshot",
        ),
    ],
)
def test_the_aot_example_reductions_render_under_the_default_config(
    kernel_name: str, args: tuple[object, ...]
) -> None:
    """The shipped ``static_shapes=False`` kernels of examples/aot_example.py write ``m, n = x.size()`` and reduce along ``n``: they declined for a round."""
    module = import_path(EXAMPLES_DIR / "aot_example.py")
    with _mock_cuda_unavailable():
        bound = _cpu_bind(getattr(module, kernel_name), args)
        code = bound.to_code(bound.config_spec.default_config())
    assert "_RDIM_SIZE_" in code, code
    assert "_launcher(" in code, code


@pytest.mark.parametrize(
    "dims",
    [(2048, 1, 1), (1, 2048, 1), (1, 2, 128), (32, 64, 1)],
    ids=["x", "y", "z", "total"],
)
def test_checked_thread_block_dims_rejects_a_shape_cuda_cannot_start(
    dims: tuple[int, int, int],
) -> None:
    with pytest.raises(exc.BackendUnsupported, match="launch limit|too large"):
        checked_thread_block_dims(dims)


def test_checked_thread_block_dims_passes_a_valid_shape_through() -> None:
    assert checked_thread_block_dims((256, 2, 1)) == (256, 2, 1)
    assert checked_thread_block_dims((1024, 1, 1)) == (1024, 1, 1)
    assert checked_thread_block_dims((1, 16, 64)) == (1, 16, 64)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("bsz", [8, 32, 40])
@pytest.mark.parametrize(
    ("fn", "num_threads", "static_threads"),
    [
        pytest.param(_doubled_rows, None, 32, id="rows"),
        pytest.param(_doubled_rows, [1], 1, id="rows_one_thread"),
        pytest.param(_doubled_rows, [8], 8, id="rows_lane_loop"),
        pytest.param(_doubled_cols, None, 32, id="cols"),
        pytest.param(_doubled_cols, [1], 1, id="cols_one_thread"),
        pytest.param(_doubled_cols, [8], 8, id="cols_lane_loop"),
        pytest.param(_doubled_grid, None, 32, id="grid"),
    ],
)
def test_an_argument_sized_axis_is_launched_with_its_extent(
    fn: Callable[..., torch.Tensor],
    num_threads: list[int] | None,
    static_threads: int,
    bsz: int,
) -> None:
    """Every row and column is written at 8, 32 and 40 (one CTA) rows per block; a block that would exceed 1024 threads raises at launch instead."""
    torch.manual_seed(bsz)
    x = torch.randn(40, 100, device=DEVICE)
    bound = _kernel(fn).bind((x, 8))
    overrides = {} if num_threads is None else {"num_threads": num_threads}
    compiled = bound.compile_config(_config(bound, **overrides))
    if bsz * static_threads > 1024:
        with pytest.raises(exc.BackendUnsupported, match="too large"):
            compiled(x, bsz)
        return
    out = compiled(x, bsz)
    torch.testing.assert_close(out, x * 2)


@requires_cuda
@pytest.mark.parametrize("bsz", [8, 32, 40])
def test_a_row_sum_beside_an_argument_sized_axis_is_right_on_one_column_thread(
    bsz: int,
) -> None:
    torch.manual_seed(bsz)
    x = torch.randn(40, 100, device=DEVICE)
    bound = _kernel(_row_sums).bind((x, 8))
    compiled = bound.compile_config(_one_thread_config(bound))
    torch.testing.assert_close(compiled(x, bsz), x.sum(dim=1))


@requires_cuda
def test_a_row_sum_beside_an_argument_sized_axis_is_declined_on_column_threads() -> (
    None
):
    x = torch.randn(40, 100, device=DEVICE)
    bound = _kernel(_row_sums).bind((x, 8))
    with pytest.raises(exc.BackendUnsupported, match="warp_reduction_sum"):
        bound.compile_config(bound.config_spec.default_config())


@requires_cuda
@pytest.mark.parametrize("bsz", [8, 32, 40])
@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
@pytest.mark.parametrize(
    "fn",
    [
        *_COUNT_KERNELS,
        pytest.param(_rows_count_columns_only, id="columns_only"),
        pytest.param(_column_sums_atomic, id="column_sums"),
    ],
)
def test_atomics_beside_an_argument_sized_axis_are_right(
    fn: Callable[..., torch.Tensor], overrides: dict[str, object], bsz: int
) -> None:
    """Each count is one per tile (per column, overall, per row tile) and the column sums add every element, at 8, 32 and 40 rows per block under 32, 1 and 8 column threads."""
    torch.manual_seed(bsz)
    x = torch.randn(40, 100, device=DEVICE)
    row_tiles = -(-40 // bsz)
    column_tiles = 4
    expected = {
        _rows_count_columns: torch.full([100], float(row_tiles), device=DEVICE),
        _rows_count_columns_only: torch.full([100], float(row_tiles), device=DEVICE),
        _rows_count_all: torch.tensor([float(row_tiles * column_tiles)], device=DEVICE),
        _rows_count_tiles: torch.ones([row_tiles], device=DEVICE),
        _column_sums_atomic: x.sum(dim=0),
    }[fn]
    bound = _kernel(fn).bind((x, 8))
    compiled = bound.compile_config(_config(bound, **overrides))
    num_threads = overrides.get("num_threads")
    static_threads = 32 if num_threads is None else num_threads[0]  # pyrefly: ignore [index-error]
    if fn is not _rows_count_columns_only and bsz * static_threads > 1024:
        with pytest.raises(exc.BackendUnsupported, match="too large"):
            compiled(x, bsz)
        return
    torch.testing.assert_close(compiled(x, bsz), expected)


@requires_cuda
@pytest.mark.parametrize("bsz", [8, 32, 40])
@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
@pytest.mark.parametrize("fn", _COUNTS_AROUND_THE_COLUMN_LOOP)
def test_atomics_around_an_argument_sized_inner_loop_are_right(
    fn: Callable[..., torch.Tensor], overrides: dict[str, object], bsz: int
) -> None:
    """One per row, one per row tile (two tiles of 32 rows) or one overall, before and after the argument-sized column loop, under 32, 1 and 8 column threads."""
    torch.manual_seed(bsz)
    x = torch.randn(40, 100, device=DEVICE)
    expected = {
        _cols_count_rows_after: torch.ones([40], device=DEVICE),
        _cols_count_rows_before: torch.ones([40], device=DEVICE),
        _cols_count_all_after: torch.tensor([2.0], device=DEVICE),
        _cols_count_all_before: torch.tensor([2.0], device=DEVICE),
        _cols_count_tiles_after: torch.ones([2], device=DEVICE),
        _cols_count_tiles_before: torch.ones([2], device=DEVICE),
    }[fn]
    bound = _kernel(fn).bind((x, 8))
    compiled = bound.compile_config(_config(bound, **overrides))
    num_threads = overrides.get("num_threads")
    static_threads = 32 if num_threads is None else num_threads[0]  # pyrefly: ignore [index-error]
    if bsz * static_threads > 1024:
        with pytest.raises(exc.BackendUnsupported, match="too large"):
            compiled(x, bsz)
        return
    torch.testing.assert_close(compiled(x, bsz), expected)


@requires_cuda
@pytest.mark.parametrize("bsz", [8, 32])
def test_a_gathered_atomic_around_an_argument_sized_inner_loop_is_right(
    bsz: int,
) -> None:
    x = torch.randn(40, 100, device=DEVICE)
    idx = (torch.arange(40, device=DEVICE) % 10).to(torch.int32)
    bound = _kernel(_cols_count_gathered_after).bind((x, idx, 8))
    compiled = bound.compile_config(bound.config_spec.default_config())
    torch.testing.assert_close(
        compiled(x, idx, bsz), torch.full([10], 4.0, device=DEVICE)
    )


@requires_cuda
def test_an_argument_sized_kernel_runs_at_shapes_other_than_the_bound_one() -> None:
    bound = _kernel(_doubled_rows).bind((torch.randn(40, 100, device=DEVICE), 8))
    compiled = bound.compile_config(bound.config_spec.default_config())
    for shape in ((37, 50), (100, 300), (1, 7)):
        x = torch.randn(*shape, device=DEVICE)
        torch.testing.assert_close(compiled(x, 8), x * 2)


@requires_cuda
def test_a_literal_block_size_is_a_static_launch_and_right() -> None:
    x = torch.randn(40, 100, device=DEVICE)
    doubled = _kernel(_doubled_rows_literal).bind((x,))
    assert "block=(32, 8, 1)" in doubled.to_code(doubled.config_spec.default_config())
    torch.testing.assert_close(doubled(x), x * 2)
    column_sums = _kernel(_column_sums_atomic_literal).bind((x,))
    torch.testing.assert_close(column_sums(x), x.sum(dim=0))
    sums = _kernel(_row_sums_literal).bind((x,))
    torch.testing.assert_close(sums(x), x.sum(dim=1))


@requires_cuda
def test_a_too_wide_argument_sized_launch_raises_a_clear_error() -> None:
    x = torch.randn(40, 100, device=DEVICE)
    bound = _kernel().bind((x, 8))
    compiled = bound.compile_config(_one_thread_config(bound))
    torch.testing.assert_close(compiled(x, 8), x * 2)
    with pytest.raises(exc.BackendUnsupported, match="exceeds the launch limit"):
        compiled(x, 2048)
