"""A SIMT launch must retain the threads required by root and inner tiles."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import patch_target_facts
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_collective import collective_config
from helion._testing.cute_collective import collective_kernel
from helion._testing.cute_collective import jagged_inputs
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

CPU_DEVICE = "cpu"

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with cpu_target():
        yield


def _selected_config() -> helion.Config:
    # Full-search winner whose root N axis was omitted from the launch guard.
    return helion.Config(
        block_sizes=[1, 1024, 32, 64],
        num_threads=[1, 1024, 0, 0],
        loop_orders=[[0, 1]],
        load_eviction_policies=["l2_last", "streaming", "", "l2_last", "l2_last"],
        cute_vector_widths=[2, 4, 1, 2],
        cute_lane_layouts=["blocked", "strided", "strided", "strided"],
        cute_collective_mma=False,
        cute_collective_compute="warp",
        cute_collective_native_seeded=True,
        cute_collective_stages=4,
        cute_collective_copy="scalar",
        cute_gathered_mma_n=256,
        cute_gathered_mma_stages=3,
    )


def _launch_shape(source: str) -> tuple[int, int, int]:
    shapes = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "block"
    ]
    assert len(shapes) == 1
    return shapes[0]


def test_selected_jagged_config_cannot_drop_1024_row_threads() -> None:
    inputs = jagged_inputs(batch=256, rows=30856, k=128, n=128)
    bound = collective_kernel(static_shapes=True)._bind_isolated(inputs)
    config = _selected_config()
    before = config.to_json()
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match=r"under-dimension required_thread_block_dims=\(32, 1024, 1\)",
    ):
        bound.to_code(config)
    assert config.to_json() == before


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize(
    ("schedule", "static_launch", "dynamic_launch"),
    [
        ("reference", (16, 1, 1), (1, 1, 1)),
        ("serial_n", (1024, 1, 1), (1024, 1, 1)),
        ("auto_n", (32, 32, 1), (32, 1, 1)),
        ("lane_rows", (32, 4, 1), (4, 32, 1)),
    ],
)
def test_jagged_simt_schedules_keep_row_coverage(
    static_shapes: bool,
    schedule: str,
    static_launch: tuple[int, int, int],
    dynamic_launch: tuple[int, int, int],
) -> None:
    inputs = jagged_inputs(batch=256, rows=30856, k=128, n=128)
    bound = collective_kernel(static_shapes=static_shapes)._bind_isolated(inputs)
    if schedule == "reference":
        config = bound.config_spec.autotune_reference_config()
    else:
        settings = _selected_config().config
        settings["num_threads"] = {
            "serial_n": [1, 1024, 1, 1],
            "auto_n": [1, 32, 0, 0],
            "lane_rows": [1, 4, 32, 1],
        }[schedule]
        if schedule == "lane_rows":
            settings["block_sizes"] = [1, 64, 32, 64]
        if not static_shapes:
            # A dynamic N extent stays inside the row loop, so this binding
            # has no root-axis permutation to configure.
            settings.pop("loop_orders")
        config = helion.Config.from_dict(settings)
    source = bound.to_code(config)
    assert _launch_shape(source) == (static_launch if static_shapes else dynamic_launch)
    assert "cute.gemm(" not in source
    if schedule == "serial_n":
        assert f"for lane_2 in range({32 if static_shapes else 8})" in source
        assert (
            "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])"
            in source
        )
    else:
        assert "for lane_1" in source


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_collective_matmul_keeps_its_cooperative_launch(
    static_shapes: bool, compute: str
) -> None:
    inputs = (torch.empty((3, 130, 78)), torch.empty((3, 78, 70)))
    kernel = collective_kernel(batched=True, static_shapes=static_shapes)
    source = kernel._bind_isolated(inputs).to_code(collective_config(compute=compute))
    assert _launch_shape(source) == (4, 32, 1)
    assert "cute.gemm(" in source
    if compute == "tcgen05":
        assert "tcgen05.CtaGroup.ONE" in source
    else:
        assert "warp.MmaTF32Op((16, 8, 8))" in source


def _add_one_per_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        for a, b, c in hl.tile([x.size(1), x.size(2), x.size(3)]):
            out[row, a, b, c] = x[row, a, b, c] + 1
    return out


@pytest.mark.parametrize(
    "lane_layouts",
    [["blocked", "strided", "strided", "blocked"], ["blocked"] * 4],
    ids=["strided", "blocked"],
)
def test_lane_loop_thread_indices_count_toward_the_launch(
    lane_layouts: list[str],
) -> None:
    """A device loop's nest (its header, the lane loops around its body and
    their setup) is emitted once the loop is closed, but its thread indices
    count on the loop's own axes, whatever the lane layout: two packed tiles
    of 4 x 256 threads need 2048, so the launch, which fell back to the
    packed axis alone, is refused; one tile per CTA still launches its 1024."""
    inputs = (torch.empty((4, 64, 64, 256), dtype=torch.bfloat16, device=CPU_DEVICE),)
    kernel = helion.kernel(
        _add_one_per_row, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated(inputs)

    def config(tiles_per_block: int) -> helion.Config:
        return helion.Config(
            block_sizes=[32, 1, 256],
            num_threads=[4, 0, 256],
            loop_orders=[[2, 1, 0]],
            flatten_loops=[True],
            cute_vector_widths=[4, 8, 4, 4],
            cute_lane_layouts=lane_layouts,
            cute_tiles_per_block=tiles_per_block,
        )

    with pytest.raises(
        helion.exc.BackendUnsupported,
        match=r"required_thread_block_dims=\(2, 256, 4\)",
    ):
        bound.to_code(config(2))
    assert "block=(256, 4, 1)" in bound.to_code(config(1))


def _nested_copy(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    output = torch.empty_like(x)
    for column in hl.tile(columns):
        for row in hl.tile(rows):
            output[row, column] = x[row, column]
    return output


def test_plain_nested_simt_loop_cannot_drop_an_explicit_inner_axis() -> None:
    inputs = (torch.empty((1536, 128), device=CPU_DEVICE),)
    kernel = helion.kernel(
        _nested_copy, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated(inputs)
    with pytest.raises(helion.exc.BackendUnsupported):
        bound.to_code(helion.Config(block_sizes=[32, 1024], num_threads=[0, 1024]))


def _outer_masked_running_max(x: torch.Tensor) -> torch.Tensor:
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=x.dtype, device=x.device)
    block_r = hl.register_block_size(r)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.full([tile_s, block_r, tile_c], float("-inf"), dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            # ``tile_s.index[:, None]`` keeps tile_s's block: no reduction
            # block is allocated for it.
            flat = tile_s.index[:, None] * r + tile_r.index[None, :]
            valid = (flat < s * r)[:, :, None]
            acc = torch.where(valid, torch.maximum(acc, x[tile_s, tile_r, tile_c]), acc)
        out[tile_s, tile_c] = torch.amax(acc, dim=1)
    return out


def _running_max_beside_dead_sum(x: torch.Tensor) -> torch.Tensor:
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=x.dtype, device=x.device)
    block_r = hl.register_block_size(r)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.full([tile_s, block_r, tile_c], float("-inf"), dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            acc = torch.maximum(acc, x[tile_s, tile_r, tile_c])
        result = torch.amax(acc, dim=1)
        if block_r > 64:
            # A full-slice sum on the side this config never takes.
            result = result + x[tile_s, :, tile_c].sum(1)
        out[tile_s, tile_c] = result
    return out


def test_reduction_without_strategy_reserves_no_axis() -> None:
    # The full-slice sum on the side of the branch this config never takes
    # gets no strategy.  Reserving the leading thread axis for it left the row
    # loop's block none: a lane loop folds the running max into one carried
    # scalar per thread, so it was refused.
    kernel = helion.kernel(
        _running_max_beside_dead_sum,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((torch.empty((4, 200, 16), device=CPU_DEVICE),))
    # The amax after the loop combines across threads through shared memory.
    with patch_target_facts(reduction_smem_budget_bytes=232448):
        source = bound.to_code(helion.Config(block_sizes=[16, 4, 16]))
    assert _launch_shape(source) == (4, 16, 16)
    assert "for lane_" not in source


def test_tile_index_slice_beside_a_running_max_reserves_no_axis() -> None:
    # ``tile_s.index[:, None]`` keeps tile_s's block, so it allocates no
    # reduction block and no thread axis is reserved for one: the row loop's
    # block keeps its 16 threads.  Left none, a lane loop would fold the
    # running max into one carried scalar per thread, which is refused.
    kernel = helion.kernel(
        _outer_masked_running_max,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((torch.empty((4, 200, 16), device=CPU_DEVICE),))
    assert not any(block.reduction for block in bound.env.block_sizes)
    with patch_target_facts(reduction_smem_budget_bytes=232448):
        source = bound.to_code(helion.Config(block_sizes=[16, 4, 16]))
    assert _launch_shape(source) == (4, 16, 16)
    assert "for lane_" not in source


def _four_full_slices(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_a, out_b = torch.empty_like(a), torch.empty_like(b)
    out_c, out_d = torch.empty_like(c), torch.empty_like(d)
    for tile_m in hl.tile(a.size(0)):
        out_a[tile_m, :] = a[tile_m, :] * 2
        out_b[tile_m, :] = b[tile_m, :] * 2
        out_c[tile_m, :] = c[tile_m, :] * 2
        out_d[tile_m, :] = d[tile_m, :] * 2
    return out_a, out_b, out_c, out_d


def test_single_thread_reductions_claim_no_thread_axis() -> None:
    # Four full slices, 256, 128, 64 and 32 wide, are four reduction blocks.
    # The widest takes 256 threads and the 4-row tile the rest of the CTA, so
    # the other three are left one live thread each: they walk their extent
    # in a lane loop from the constant 0 and claim no thread axis
    # (``ReductionStrategy._claims_thread_axis``).  Had each claimed one, the
    # launch would have planned the tile on a fifth axis.
    kernel = helion.kernel(
        _four_full_slices, backend="cute", static_shapes=True, autotune_effort="none"
    )
    inputs = tuple(
        torch.empty((64, width), device=CPU_DEVICE) for width in (256, 128, 64, 32)
    )
    source = kernel._bind_isolated(inputs).to_code(helion.Config(block_sizes=[4]))
    assert _launch_shape(source) == (256, 4, 1)
    assert "thread_idx()[3]" not in source
    assert source.count("= cutlass.Int32(0) + cutlass.Int32(synthetic_lane_") == 3


def _outer_masked_row_sums(x: torch.Tensor) -> torch.Tensor:
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=x.dtype, device=x.device)
    block_r = hl.register_block_size(r)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            flat = tile_s.index[:, None] * r + tile_r.index[None, :]
            valid = (flat < s * r)[:, :, None]
            acc = acc + torch.where(valid, x[tile_s, tile_r, tile_c], 0.0).sum(1)
        out[tile_s, tile_c] = acc
    return out


def _row_masked_row_sums(x: torch.Tensor) -> torch.Tensor:
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=x.dtype, device=x.device)
    block_r = hl.register_block_size(r)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            valid = (tile_r.index < r)[None, :, None]
            acc = acc + torch.where(valid, x[tile_s, tile_r, tile_c], 0.0).sum(1)
        out[tile_s, tile_c] = acc
    return out


def _wide_row_source(fn: Callable[[torch.Tensor], torch.Tensor]) -> str:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated((torch.empty((2, 200, 2), device=CPU_DEVICE),))
    with patch_target_facts(reduction_smem_budget_bytes=232448):
        return bound.to_code(helion.Config(block_sizes=[128, 2, 2]))


@pytest.mark.parametrize("fn", [_outer_masked_row_sums, _row_masked_row_sums])
def test_row_tile_wider_than_z_keeps_64_threads(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    # Beside the two grid axes the row loop's tile lands on z, which takes 64
    # threads.  Its 128 automatic threads launched an invalid
    # block=(2, 2, 128).  The tile keeps 64 threads, each walking two lanes,
    # whether its mask reads the outer tile's index (``tile_s.index``, the
    # same on every lane) or its own (``tile_r.index``, which each lane
    # offsets).
    source = _wide_row_source(fn)
    assert _launch_shape(source) == (2, 2, 64)
    assert "for lane_0 in range(2)" in source


def test_row_tile_wider_than_z_refuses_a_carry_over_its_lanes() -> None:
    # The running max carries the row tile's dim: with two lanes per thread
    # it would fold both into one carried scalar, so it is refused.
    with pytest.raises(helion.exc.BackendUnsupported, match="own loop's block"):
        _wide_row_source(_outer_masked_running_max)


def _nested_row_sums(x: torch.Tensor) -> torch.Tensor:
    m, n, _ = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n, :].sum(-1)
    return out


def _running_max_beside_row_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s, r = x.shape
    block_r = hl.register_block_size(r)
    out = torch.empty([s], dtype=x.dtype, device=x.device)
    for tile_s in hl.tile(s):
        bias = y[tile_s, :].sum(1)
        acc = hl.full([tile_s, block_r], float("-inf"), dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            acc = torch.maximum(acc, x[tile_s, tile_r])
        out[tile_s] = acc.amax(1) + bias
    return out


def _sibling_loops_beside_row_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        scale = y[tile_m, :].sum(-1)
        for tile_a in hl.tile(n):
            out[tile_m, tile_a] = x[tile_m, tile_a] * scale[:, None]
        for tile_b in hl.tile(n):
            out[tile_m, tile_b] = out[tile_m, tile_b] + 1.0
    return out


def _row_sums_beside_unread_grid_tile(x: torch.Tensor) -> torch.Tensor:
    s, r = x.shape
    block_r = hl.register_block_size(r)
    out = torch.zeros([s], dtype=x.dtype, device=x.device)
    for tile_s, _tile_c in hl.tile([s, 4]):
        acc = hl.zeros([tile_s], dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            acc = acc + x[tile_s, tile_r].sum(1)
        out[tile_s] = acc
    return out


def _sibling_maxes_over_one_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s, r = x.shape
    block_r = hl.register_block_size(r)
    out = torch.empty([s], dtype=x.dtype, device=x.device)
    for tile_s in hl.tile(s):
        bias = y[tile_s, :].sum(1)
        m1 = hl.full([tile_s, block_r], float("-inf"), dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            m1 = torch.maximum(m1, x[tile_s, tile_r])
        m2 = hl.full([tile_s, block_r], float("-inf"), dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=block_r):
            m2 = torch.maximum(m2, x[tile_s, tile_r] * 2)
        out[tile_s] = m1.amax(1) + m2.amax(1) + bias
    return out


@pytest.mark.parametrize(
    ("fn", "shapes", "config", "launch"),
    [
        # The tiles take the whole CTA, which leaves the sum one thread.
        (_nested_row_sums, [(64, 256, 32)], {"block_sizes": [8, 128]}, (8, 128, 1)),
        (_nested_row_sums, [(64, 256, 32)], {"block_sizes": [4, 256]}, (4, 256, 1)),
        (_nested_row_sums, [(64, 256, 32)], {"block_sizes": [2, 512]}, (2, 512, 1)),
        # The config gives the sum one thread.
        (
            _nested_row_sums,
            [(64, 256, 32)],
            {"block_sizes": [2, 128], "num_threads": [2, 0, 1]},
            (2, 128, 1),
        ),
        (
            _running_max_beside_row_sum,
            [(2, 200), (2, 2)],
            {"block_sizes": [128, 2], "num_threads": [0, 0, 1], "reduction_loops": [2]},
            (2, 128, 1),
        ),
        # The launch widens the 2-wide sibling loop to the other's 128 threads
        # before the sum takes its share, which leaves it one thread.
        (
            _sibling_loops_beside_row_sum,
            [(64, 256), (64, 32)],
            {"block_sizes": [2, 2, 128]},
            (2, 128, 1),
        ),
        # No graph reads the second grid tile, which takes no thread axis.
        (
            _row_sums_beside_unread_grid_tile,
            [(8, 256)],
            {"block_sizes": [128, 2, 2]},
            (2, 128, 1),
        ),
        # The two loops over one registered block run on one strategy, which
        # the launch lists once per loop: the sum's share of the CTA counts
        # its 128 threads twice and leaves it one thread.
        (
            _sibling_maxes_over_one_block,
            [(64, 256), (64, 32)],
            {"block_sizes": [128, 2]},
            (2, 128, 1),
        ),
    ],
)
def test_loop_tile_on_y_keeps_its_threads(
    fn: Callable[..., torch.Tensor],
    shapes: list[tuple[int, ...]],
    config: dict[str, list[int]],
    launch: tuple[int, int, int],
) -> None:
    # Each inner loop's tile launches on y, which takes 1024 threads.  Counted
    # one axis higher, on z, it was capped at 64 threads with a lane loop over
    # the rest: needless beside the sibling loops and the unread grid tile,
    # and refused for the nested sum (its reduction marker in that lane loop)
    # and the running max (a carry over the loop's lanes).
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    inputs = tuple(torch.empty(shape, device=CPU_DEVICE) for shape in shapes)
    # The amax after the loop combines across threads through shared memory.
    with patch_target_facts(reduction_smem_budget_bytes=232448):
        source = kernel._bind_isolated(inputs).to_code(helion.Config(**config))
    assert _launch_shape(source) == launch
    assert "for lane_" not in source


def _three_level_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_s in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            for tile_m in hl.tile(x.size(2)):
                out[tile_s, tile_n, tile_m] = x[tile_s, tile_n, tile_m]
    return out


def test_tile_below_an_enclosing_loop_keeps_64_threads_on_z() -> None:
    # The innermost loop's tile lands on z, above the grid tile's axis and
    # the enclosing loop's, each two threads wide.  Counted without the
    # enclosing loop it looked to land on y and kept its 128 automatic
    # threads: an invalid block=(2, 2, 128).
    kernel = helion.kernel(
        _three_level_copy, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated((torch.empty((4, 64, 256), device=CPU_DEVICE),))
    source = bound.to_code(helion.Config(block_sizes=[2, 2, 128]))
    assert _launch_shape(source) == (2, 2, 64)
    assert "for lane_2 in range(2)" in source


def _root_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_s, tile_n, tile_m in hl.tile(x.size()):
        out[tile_s, tile_n, tile_m] = x[tile_s, tile_n, tile_m]
    return out


def test_root_tile_wider_than_z_keeps_64_threads() -> None:
    # A three-dimensional root tile puts its third block on z, which takes
    # 64 threads: its 128 automatic threads keep 64 and walk the rest in a
    # lane loop.  Uncapped, the root launched an invalid block=(2, 2, 128).
    kernel = helion.kernel(
        _root_copy, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated((torch.empty((4, 64, 256), device=CPU_DEVICE),))
    source = bound.to_code(helion.Config(block_sizes=[2, 2, 128]))
    assert _launch_shape(source) == (2, 2, 64)
    assert "for lane_2 in range(2)" in source


def _short_row_sums(x: torch.Tensor) -> torch.Tensor:
    m, n, _ = x.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n, :].to(torch.float32).sum(-1)
    return out


def test_tile_beside_a_reduction_keeping_threads_keeps_64_threads() -> None:
    # The sum over 32 elements rolls beside the tile's 256 threads, in a
    # chunk of 8 (the flat search space's smallest), and takes x, so the
    # [2, 128] tile lands on y and z: its 128 automatic threads keep 64 and
    # walk the rest in a lane loop.  Counted without the reduction's axis,
    # the tile looked to land on x and y: an invalid block=(4, 2, 128).
    kernel = helion.kernel(
        _short_row_sums, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated((torch.empty((8, 256, 32), device=CPU_DEVICE),))
    source = bound.to_code(helion.Config(block_sizes=[2, 128]))
    assert _launch_shape(source) == (8, 2, 64)


def _copy_rows_beside_another_root(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tile_s, tile_n in hl.tile([x.size(0), x.size(1)]):
        for tile_m in hl.tile(x.size(2)):
            out[tile_s, tile_n, tile_m] = x[tile_s, tile_n, tile_m]
    for tile_k in hl.tile(y.size(0)):
        out_y[tile_k] = y[tile_k] + 1
    return out, out_y


def test_tile_below_its_own_root_keeps_64_threads_on_z() -> None:
    # The inner loop's tile lands on z, above its own root's two grid axes.
    # Counted from the root with the fewest grid axes (the second root, one
    # thread wide), it looked to land on x and kept its 128 automatic
    # threads: an invalid block=(2, 2, 128).
    kernel = helion.kernel(
        _copy_rows_beside_another_root,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(
        (
            torch.empty((8, 8, 256), device=CPU_DEVICE),
            torch.empty((64,), device=CPU_DEVICE),
        )
    )
    source = bound.to_code(helion.Config(block_sizes=[2, 2, 128, 1]))
    assert _launch_shape(source) == (2, 2, 64)
    assert "for lane_2 in range(2)" in source


def _middle_dim_sums(x: torch.Tensor) -> torch.Tensor:
    m, _, n = x.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_n, tile_m in hl.tile([n, m]):
        out[tile_m, tile_n] = x[tile_m, :, tile_n].to(torch.float32).sum(1)
    return out


def test_register_tile_reduction_lays_out_its_tiles() -> None:
    # The register-tile reduction lays the tiles out below its own axis: the
    # vectorized tile's two threads on x, the 128-row tile on y, its own two
    # threads on z.  Counted reduction-first, the row tile looked to land on
    # z and was capped at 64 threads, whose unvectorized lane loop broke the
    # register tile: block=(2, 2, 64) with lane loops instead.
    kernel = helion.kernel(
        _middle_dim_sums, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated((torch.empty((128, 16, 8), device=CPU_DEVICE),))
    config = helion.Config(
        block_sizes=[4, 128],
        num_threads=[2, 0, 2],
        cute_vector_widths=[1, 2, 1],
        reduction_loops=[None],
    )
    with patch_target_facts(reduction_smem_budget_bytes=232448):
        source = bound.to_code(config)
    assert _launch_shape(source) == (2, 128, 2)
    assert "_cute_grouped_reduce_shared_columns" in source
    # Only the vectorized tile's one-vector lane; the row tile walks none.
    assert "lane_1" not in source
