from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.reduction_strategy import ReductionStrategy
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_sibling_roots import row_sums_beside_a_wider_tile
from helion._testing.cute_sibling_roots import sibling_config
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only


def _mean_then_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(x.size(0)):
        means[tile_m] = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = torch.sum(y[tile_r, :].to(torch.float32), dim=-1)
    return means, sums


def _mean_barrier_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(x.size(0)):
        means[tile_m] = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
    hl.barrier()
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = torch.sum(y[tile_r, :].to(torch.float32), dim=-1)
    return means, sums


def _center_rows_then_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(x.size(0)):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = torch.sum(y[tile_r, :].to(torch.float32), dim=-1)
    return out, sums


def _render(
    fn: Callable[..., object],
    shapes: list[tuple[int, int]],
    values: dict[str, object],
) -> tuple[str, dict[int, int]]:
    """The code ``fn`` renders for ``values`` (rows of one by default) on
    bfloat16 inputs of ``shapes``, and the thread count of each reduction."""
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = tuple(torch.randn(*shape, dtype=torch.bfloat16) for shape in shapes)
    counts: dict[int, int] = {}
    add_reductions = TileStrategyDispatch._add_reduction_strategies

    def record(dispatch: TileStrategyDispatch, *args: object) -> None:
        add_reductions(dispatch, *args)
        for strategy in dispatch.strategies:
            if isinstance(strategy, ReductionStrategy):
                counts[strategy.block_index] = strategy._reduction_thread_count()

    config = helion.Config(**{"block_sizes": [1, 1], **values})
    with (
        mock_cuda_unavailable(),
        patch.object(TileStrategyDispatch, "_add_reduction_strategies", record),
    ):
        code = cpu_bind(kernel, args).to_code(config)
    return code, counts


@pytest.mark.parametrize(
    ("fn", "shapes", "values", "launch", "threads"),
    [
        # Two wide rolled reductions: the second walked its row in one thread
        # beside the first root's 1024.
        (
            _mean_then_sum,
            [(64, 8192), (64, 65536)],
            {"reduction_loops": [4096, 4096]},
            "(1024, 1, 1)",
            {1: 1024, 3: 1024},
        ),
        # The same around an hl.barrier(): the second phase ran one thread.
        (
            _mean_barrier_sum,
            [(64, 8192), (64, 65536)],
            {"reduction_loops": [4096, 4096]},
            "(1024, 1, 1)",
            {1: 1024, 3: 1024},
        ),
        # A persistent 512-wide first reduction: the second took 2 threads and
        # the launch's 512 was refused as surplus lanes.
        (
            _mean_then_sum,
            [(64, 512), (64, 65536)],
            {"reduction_loops": [None, 4096]},
            "(512, 1, 1)",
            {1: 512, 3: 512},
        ),
        (
            _mean_barrier_sum,
            [(64, 512), (64, 65536)],
            {"reduction_loops": [None, 4096]},
            "(512, 1, 1)",
            {1: 512, 3: 512},
        ),
        # Beside the first root's 16-thread inner loop the launch leaves the
        # second root's reduction the first's 64 threads, not 1024 / (16 * 64).
        (
            _center_rows_then_sum,
            [(64, 8192), (64, 65536)],
            {"block_sizes": [1, 16, 1], "reduction_loops": [4096, 4096]},
            "(64, 1, 16)",
            {1: 64, 4: 64},
        ),
    ],
)
def test_reduction_of_another_root_takes_the_shared_launch(
    fn: Callable[..., object],
    shapes: list[tuple[int, int]],
    values: dict[str, object],
    launch: str,
    threads: dict[int, int],
) -> None:
    """Root loops and ``hl.barrier()`` phases never share a CTA, so a
    reduction does not divide its threads with another root's tiles and
    reductions; it gets what the launch they share leaves its axis, which is
    exactly the threads another root's reduction runs there."""
    code, counts = _render(fn, shapes, values)
    assert f"block={launch}" in code
    assert counts == threads


def _sums_then_maxes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty([x.size(0)], device=x.device)
    maxes = torch.empty([y.size(0)], device=y.device)
    for tile in hl.tile(x.size(0)):
        sums[tile] = x[tile, :].sum(-1)
    for tile in hl.tile(y.size(0)):
        maxes[tile] = y[tile, :].amax(-1)
    return sums, maxes


def test_wider_reduction_of_another_root_takes_the_narrower_launch_axis() -> None:
    """A 40-wide max beside another root's 8-wide sum takes the sum's eight
    threads (eight lanes each) instead of its own 64, which made the launch
    64 wide and the sum's surplus lanes were refused.  A narrower reduction
    of a later root is still refused
    (``test_reduction_narrower_than_the_shared_launch_is_rejected``)."""
    kernel = helion.kernel(
        _sums_then_maxes, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with mock_cuda_unavailable():
        code = cpu_bind(kernel, (torch.randn(16, 8), torch.randn(8, 40))).to_code(
            helion.Config(block_sizes=[1, 1])
        )
    assert "block=(8, 1, 1)" in code
    assert code.count("threads_in_group=8") == 2


def test_reduction_narrower_than_the_shared_launch_is_rejected() -> None:
    """Each root reduces its rows over its own lane count, but the roots
    share the widest launch: the second root's 16-wide max cannot take the
    32 threads of the first root's sum, and its surplus lanes would read past
    its row, reduce among themselves and race the row's result."""
    kernel = helion.kernel(
        _sums_then_maxes, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with (
        mock_cuda_unavailable(),
        pytest.raises(exc.BackendUnsupported, match="surplus lanes"),
    ):
        cpu_bind(kernel, (torch.randn(64, 32), torch.randn(32, 16))).to_code(
            helion.Config(block_sizes=[1, 1])
        )


@pytest.mark.parametrize("pid_type", ["flat", "persistent_blocked"])
def test_a_tile_reduction_beside_a_wider_sibling_defines_its_indices(
    pid_type: str,
) -> None:
    """The 128-thread column tile runs on axis 1 of the (2, 512, 1) launch
    it shares with the sibling's 512-row tile, whose row sum gets the two
    threads of axis 0, so its load is masked to the tile's threads and the
    combine spans the launch; the index and mask definitions were parked in
    the grid's prefix and never emitted (``NameError: name 'mask_1' is not
    defined``).  ``test_cute_tile_dim_reduction_threads.py`` runs it."""
    kernel = helion.kernel(
        row_sums_beside_a_wider_tile,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (torch.randn(64, 128), torch.randn(64, 1024)))
        code = bound.to_code(sibling_config(bound, pid_type))
    body = "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("#")
    )
    assert "block=(2, 512, 1)" in code, code
    assert "mask_1 = cutlass.Int32(cute.arch.thread_idx()[1]) < 128" in body, code
    assert "pre=2, group_span=1024" in body, code
    assert "threads_in_group=2" in body, code
    (load,) = [line for line in body.splitlines() if "if mask_1 else" in line]
    for name in ("indices_0", "indices_1"):
        assert f"cutlass.Int32({name})" in load, code
        assert body.index(f"{name} = ") < body.index(load), code
    assert body.index("mask_1 = ") < body.index(load), code


def _sum_then_two_sums(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    b = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(x.size(0)):
        a[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    for tile_r in hl.tile(y.size(0)):
        b[tile_r] = y[tile_r, :].to(torch.float32).sum(-1) + z[tile_r, :].to(
            torch.float32
        ).sum(-1)
    return a, b


def _center_rows_then_two_roots(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(w)
    y_sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    z_sums = torch.empty([z.size(0)], dtype=torch.float32, device=z.device)
    for tile_m in hl.tile(x.size(0)):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(w.size(1)):
            out[tile_m, tile_n] = (w[tile_m, tile_n] - mean[:, None]).to(w.dtype)
    for tile_r in hl.tile(y.size(0)):
        y_sums[tile_r] = y[tile_r, :].to(torch.float32).sum(-1)
    for tile_q in hl.tile(z.size(0)):
        z_sums[tile_q] = z[tile_q, :].to(torch.float32).sum(-1)
    return out, y_sums, z_sums


def _center_rows_then_sum_and_max(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(w)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    maxes = torch.empty([z.size(0)], dtype=torch.float32, device=z.device)
    for tile_m in hl.tile(x.size(0)):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(w.size(1)):
            out[tile_m, tile_n] = (w[tile_m, tile_n] - mean[:, None]).to(w.dtype)
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = y[tile_r, :].to(torch.float32).sum(-1)
        maxes[tile_r] = z[tile_r, :].to(torch.float32).amax(-1)
    return out, sums, maxes


@pytest.mark.parametrize(
    ("fn", "shapes", "values", "launch", "threads"),
    [
        # x and y share their reduction block (one size, one block id), which
        # lies in the first root's range: in the second root's CTA it runs on
        # axis 0, where z's sum would have taken 1024 threads.
        (
            _sum_then_two_sums,
            [(64, 8192), (64, 8192), (64, 65536)],
            {"reduction_loops": [4096]},
            "(1024, 1, 1)",
            {1: 1024, 3: 1},
        ),
        # A third root's sum, threaded, would reserve a third axis and push the
        # first root's 16-thread loop off the launch: the loop would run as a
        # lane loop in every thread, the launch (8, 1, 1).
        (
            _center_rows_then_two_roots,
            [(64, 8), (64, 16), (64, 512), (64, 1024)],
            {"block_sizes": [1, 16, 1, 1]},
            "(8, 1, 16)",
            {1: 8, 4: 8, 6: 1},
        ),
        # The second root's max, threaded, would reserve a third axis and push
        # the first root's 16-thread loop off the launch.
        (
            _center_rows_then_sum_and_max,
            [(64, 8), (64, 16), (64, 128), (64, 256)],
            {"block_sizes": [1, 16, 1]},
            "(8, 1, 16)",
            {1: 8, 4: 8, 5: 1},
        ),
        # Taking the first sum's 8 threads would move the 128-row tiles to the
        # z axis, past its 64 threads.
        (
            _mean_then_sum,
            [(256, 8), (256, 16)],
            {"block_sizes": [128, 128], "reduction_loops": [8, 16]},
            "(8, 128, 1)",
            {1: 8, 3: 1},
        ),
    ],
)
def test_reduction_of_another_root_keeps_its_budget_off_the_shared_launch(
    fn: Callable[..., object],
    shapes: list[tuple[int, int]],
    values: dict[str, object],
    launch: str,
    threads: dict[int, int],
) -> None:
    """Where the shared launch cannot take another root's reduction, it keeps
    the budget over every strategy, as before: when the roots share a
    reduction block, and when the axis a threaded reduction reserves would
    push a tile off the launch's three axes or past an axis limit (each
    threaded reduction moves every tile up one axis).  These failed to render
    or dropped a tile from the launch when taking the shared launch."""
    code, counts = _render(fn, shapes, values)
    assert f"block={launch}" in code
    assert counts == threads


def _slices_of_two_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty_like(x)
    b = torch.empty_like(y)
    for tile_i in hl.tile(x.size(0)):
        a[tile_i, :] = x[tile_i, :] + 1
    for tile_j in hl.tile(y.size(0)):
        b[tile_j, :, :] = y[tile_j, :, :] + 1
    return a, b


@pytest.mark.parametrize(
    ("values", "launch", "threads"),
    [
        # Block 3 indexed block 1's axis, and block 4 then took the axis's 8
        # threads too: each row wrote 64 of its 512 elements.
        (
            {"block_sizes": [4, 4], "num_threads": [4, 4, 8, 8, 0]},
            "(8, 1, 4)",
            {1: 8, 3: 8, 4: 1},
        ),
        # Block 4's own 32 threads landed on block 1's axis as well.
        (
            {"num_threads": [1, 1, 8, 8, 0]},
            "(8, 32, 1)",
            {1: 8, 3: 8, 4: 32},
        ),
    ],
)
def test_a_slice_indexes_no_axis_of_another_root(
    values: dict[str, object], launch: str, threads: dict[int, int]
) -> None:
    """A full slice of a pointwise kernel shares the thread axis of a slice
    spelling its thread count (``plan_axis_partner``) only within its root:
    another root's slice never runs in its CTA, and the axis it would index
    there is one its own root's blocks do not count, so the next of them
    took it too.  The two slices of ``b`` now index distinct axes."""
    code, counts = _render(_slices_of_two_roots, [(64, 8), (64, 16, 32)], values)
    assert f"block={launch}" in code, code
    assert counts == threads
    axes = [
        set(re.findall(rf"indices_{block} = .*thread_idx\(\)\[(\d)\]", code))
        for block in (3, 4)
    ]
    assert not axes[0] & axes[1], code


def _atomic_beside_a_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m], x[tile_m].to(torch.float32))
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = y[tile_r, :].to(torch.float32).sum(-1)
    return out, sums


def _sum_beside_an_atomic(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = y[tile_r, :].to(torch.float32).sum(-1)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m], x[tile_m].to(torch.float32))
    return out, sums


_ATOMIC_CONFIG = {"block_sizes": [32, 32], "reduction_loops": [1024]}


@pytest.mark.parametrize(
    ("fn", "reduction"), [(_atomic_beside_a_sum, 2), (_sum_beside_an_atomic, 1)]
)
def test_an_atomic_beside_a_sum_runs_on_its_axis_leader(
    fn: Callable[..., object], reduction: int
) -> None:
    """The sum takes 32 threads on axis 0 of the launch the roots share,
    which moves the atomic's tile to axis 1 (32, 32, 1) in either order of
    the roots.  The atomic's root claims no thread of axis 0, so the atomic
    runs on that axis's leader thread alone."""
    code, counts = _render(fn, [(1024,), (64, 4096)], _ATOMIC_CONFIG)
    assert "block=(32, 32, 1)" in code, code
    assert counts == {reduction: 32}
    assert code.count("cute.arch.atomic_add(") == 1, code
    assert re.search(
        r"if cute\.arch\.thread_idx\(\)\[0\] == 0:\n\s+cute\.arch\.atomic_add\(", code
    ), code


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("fn", [_atomic_beside_a_sum, _sum_beside_an_atomic])
def test_cuda_an_atomic_beside_a_sum_adds_each_element_once(
    fn: Callable[..., object],
) -> None:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    x = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
    bound = kernel.bind((x, y))
    out, sums = bound.compile_config(helion.Config(**_ATOMIC_CONFIG))(x, y)
    torch.testing.assert_close(out, x.float(), rtol=0, atol=0)
    torch.testing.assert_close(sums, y.float().sum(-1), rtol=1e-4, atol=1e-3)


def _sum_beside_a_folded_row(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    b = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_i in hl.tile(x.size(0)):
        a[tile_i] = x[tile_i, :].to(torch.float32).sum(-1)
    for tile_j in hl.tile(y.size(0)):
        b[tile_j] = (
            y[tile_j, :].to(torch.float32).reshape([tile_j, 1, 128]).sum(-1).sum(-1)
        )
    return a, b


def _sums_over_a_shared_block(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    block = hl.register_block_size(x.size(0))
    a = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    b = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile in hl.tile(x.size(0), block_size=block):
        a[tile] = x[tile, :].to(torch.float32).sum(-1)
    for tile in hl.tile(y.size(0), block_size=block):
        b[tile] = y[tile, :].to(torch.float32).sum(-1)
    return a, b


_HELD_BY_THE_BUDGET = [
    # The second root's persistent 128-wide row would take the first root's
    # 64 threads on axis 0 and then fold to a warp of lanes
    # (``plan_synthetic_lanes``), narrower than its axis: refused.
    (
        _sum_beside_a_folded_row,
        [(64, 64), (64, 128)],
        {"block_sizes": [16, 16], "reduction_loops": [64]},
        "(64, 16, 1)",
        {1: 64, 3: 1},
    ),
    # Roots over one registered block size share their grid's block ids,
    # which the root ranges cannot tell apart: the second root's sum took 4
    # threads of axis 1 beside the first root's 64, (64, 4, 4).
    (
        _sums_over_a_shared_block,
        [(64, 512), (64, 8192)],
        {"block_sizes": [4], "reduction_loops": [1024, 1024]},
        "(64, 4, 1)",
        {1: 64, 2: 1},
    ),
]


@pytest.mark.parametrize(
    ("fn", "shapes", "values", "launch", "threads"), _HELD_BY_THE_BUDGET
)
def test_a_reduction_the_shared_launch_cannot_hold_keeps_its_budget(
    fn: Callable[..., object],
    shapes: list[tuple[int, int]],
    values: dict[str, object],
    launch: str,
    threads: dict[int, int],
) -> None:
    """A reduction keeps the backend's budget where the shared launch would
    not hold it: a persistent row that folds below another root's threads
    on its axis, and the reductions of roots over one registered block
    size."""
    code, counts = _render(fn, shapes, values)
    assert f"block={launch}" in code, code
    assert counts == threads


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("fn", "shapes", "values"), [case[:3] for case in _HELD_BY_THE_BUDGET]
)
def test_cuda_a_reduction_held_by_the_budget_matches_eager(
    fn: Callable[..., object],
    shapes: list[tuple[int, int]],
    values: dict[str, object],
) -> None:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, config=helion.Config(**values)
    )
    x, y = (torch.randn(*shape, device=DEVICE) for shape in shapes)
    a, b = kernel(x, y)
    torch.testing.assert_close(a, x.sum(-1), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(b, y.sum(-1), rtol=1e-4, atol=1e-3)


def _two_reductions_beside_one(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    b = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_i in hl.tile(x.size(0)):
        a[tile_i] = x[tile_i, :].to(torch.float32).sum(-1) + w[tile_i, :].to(
            torch.float32
        ).amax(-1)
    for tile_j in hl.tile(y.size(0)):
        b[tile_j] = y[tile_j, :].to(torch.float32).sum(-1)
    return a, b


_TWO_BESIDE_ONE_SHAPES = [(64, 64), (64, 128), (64, 256)]


def test_a_roots_second_reduction_keeps_its_budget() -> None:
    """A root whose first reduction takes the shared launch's first axis
    leaves the next one to the backend's budget: on the second axis, the
    stores of its lane loops would vary along the first reduction's threads,
    which the lane splitter refuses ("lane reduction store aliases a
    value-varying thread axis")."""
    code, counts = _render(
        _two_reductions_beside_one, _TWO_BESIDE_ONE_SHAPES, {"block_sizes": [8, 16]}
    )
    assert "block=(8, 1, 16)" in code, code
    assert counts == {1: 8, 2: 1, 4: 8}


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_a_roots_second_reduction_matches_eager() -> None:
    kernel = helion.kernel(
        _two_reductions_beside_one,
        backend="cute",
        static_shapes=True,
        config=helion.Config(block_sizes=[8, 16]),
    )
    x, w, y = (torch.randn(*shape, device=DEVICE) for shape in _TWO_BESIDE_ONE_SHAPES)
    a, b = kernel(x, w, y)
    torch.testing.assert_close(a, x.sum(-1) + w.amax(-1), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(b, y.sum(-1), rtol=1e-4, atol=1e-3)
