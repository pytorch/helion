"""Device loops that share a registered block size share one tile strategy.

Two ``hl.tile`` loops over one ``hl.register_block_size`` have the same block
ids, and codegen runs both on the strategy it finds by them, the last one
built.  An earlier strategy left in ``TileStrategyDispatch.strategies`` kept
the block's configured threads after codegen demoted the block past the
third launch axis, and the thread-extent queries read it: the reductions over
the demoted block combined each thread's lanes, which already cover the
block, across 32 lanes of unrelated rows, and the lane-merged carry check let
a carry fold the demoted block's lanes.  Only loops over the same block-id
tuple share a strategy.  Device loops of several roots over one registered
block size run on one strategy, which each root's body lays out after its
own grid's axes, and the launch takes it there in each root.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.shared_block_kernels import device_loops_of_two_roots
from helion._testing.shared_block_kernels import root_beside_a_device_loop
from helion._testing.shared_block_kernels import shared_block_softmax
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]


def shared_block_decay(x: torch.Tensor) -> torch.Tensor:
    batch, heads, rows, n = x.shape
    out = torch.empty_like(x)
    block_n = hl.register_block_size(n)
    for tile_b, tile_h, tile_r in hl.tile([batch, heads, rows]):
        acc = hl.zeros([tile_b, tile_h, tile_r, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc * 0.5 + x[tile_b, tile_h, tile_r, tile_n]
        total = acc.sum(-1)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_b, tile_h, tile_r, tile_n] = (
                x[tile_b, tile_h, tile_r, tile_n] - total[:, :, :, None]
            )
    return out


def shared_block_two_roots(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block = hl.register_block_size(x.size(0))
    rows = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    cols = torch.zeros([x.size(1)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0), block_size=block):
        rows[tile] = x[tile, :].to(torch.float32).sum(-1)
    for tile in hl.tile(x.size(1), block_size=block):
        cols[tile] = x[:, tile].to(torch.float32).sum(0)
    return rows, cols


@functools.cache
def _bound(fn: object) -> BoundKernel:
    """Each kernel bound once for the module; its configs share the binding."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    return cpu_bind(kernel, (torch.empty(2, 2, 2, 64),))


def _code(fn: object, num_threads: list[int]) -> str:
    return _bound(fn).to_code(
        helion.Config(block_sizes=[32, 2, 2, 2], num_threads=num_threads)
    )


def test_block_demoted_past_the_third_axis_reduces_within_each_thread() -> None:
    # The grid's three blocks take the launch axes, so the shared block's 32
    # threads are demoted to one thread walking it in a lane loop: its
    # reductions fold that lane loop and combine nothing across threads.
    code = _code(shared_block_softmax, [32, 2, 2, 2])
    assert "block=(2, 2, 2)" in code
    assert "for lane_0 in range(32)" in code
    assert "warp_reduction" not in code
    assert "_cute_grouped_reduce" not in code


def test_block_on_a_thread_axis_reduces_across_its_threads() -> None:
    # With a one-thread grid block the shared block keeps its 32 threads, on
    # the axis above the other two blocks' 4 threads.
    code = _code(shared_block_softmax, [32, 2, 2, 1])
    assert "block=(2, 2, 32)" in code
    assert code.count("pre=4, group_span=128, group_count=1)") == 2


def test_carry_on_a_demoted_block_folds_its_lanes() -> None:
    # Demoted, the block's 32 lanes would fold into one carried scalar per
    # thread, which only a zero-initialized sum survives (not acc * 0.5).
    with pytest.raises(helion.exc.BackendUnsupported, match="one carried scalar"):
        _code(shared_block_decay, [32, 2, 2, 2])
    # On its own thread axis, each thread carries one element of the block.
    assert "block=(2, 2, 32)" in _code(shared_block_decay, [32, 2, 2, 1])


def test_root_loops_sharing_a_block_keep_a_grid_each() -> None:
    # Both roots run on the second root's strategy, which also takes the
    # first one's place in the dispatcher's list: the list's first two
    # entries stay the roots' grids, which the root ranges index.
    kernel = helion.kernel(shared_block_two_roots, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (torch.empty(64, 32, dtype=torch.bfloat16),))
    assert "block=(32, 32, 1)" in bound.to_code(bound.config_spec.default_config())


def test_a_device_loop_takes_no_root_grids_place() -> None:
    # The second root's device loop runs on the strategy codegen finds by
    # the shared block ids, and the first root on its own grid, which the
    # list keeps: the root ranges lay that root out on its own axis instead
    # of on the device loop's, where the body does not place it.
    kernel = helion.kernel(
        root_beside_a_device_loop, backend="cute", static_shapes=True
    )
    bound = cpu_bind(kernel, (torch.empty(64), torch.empty(8, 64)))
    assert "block=(32, 32, 1)" in bound.to_code(bound.config_spec.default_config())


_TWO_ROOT_LOOPS = helion.Config(block_sizes=[16, 8, 4, 4], num_threads=[4, 8, 4, 4])


def test_device_loops_of_two_roots_follow_their_own_grids() -> None:
    # The two roots' device loops over block_n run on one strategy, which
    # each body lays out after its own grid's axes: axis 1 after the 1-D
    # grid, axis 2 after the 2-D one.  One branch's axis for both roots made
    # the launch-layout check refuse the other root.
    kernel = helion.kernel(
        device_loops_of_two_roots, backend="cute", static_shapes=True
    )
    bound = cpu_bind(kernel, (torch.empty(64, 64), torch.empty(16, 16, 64)))
    code = bound.to_code(_TWO_ROOT_LOOPS)
    assert "block=(8, 4, 4)" in code
    axes = re.findall(r"cute\.arch\.thread_idx\(\)\[(\d)\]\) \* 4 \+", code)
    assert axes == ["1", "2"], code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_device_loops_of_two_roots_match_eager() -> None:
    x = torch.randn(64, 64, device=DEVICE)
    y = torch.randn(16, 16, 64, device=DEVICE)
    kernel = helion.kernel(
        device_loops_of_two_roots, backend="cute", static_shapes=True
    )
    shifted, doubled = kernel.bind((x, y)).compile_config(_TWO_ROOT_LOOPS)(x, y)
    torch.testing.assert_close(shifted, x + 1)
    torch.testing.assert_close(doubled, y * 2)
