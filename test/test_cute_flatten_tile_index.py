"""A pure pointwise CuTe kernel that reads ``tile.index`` keeps its flatten
choice (CPU codegen).

A flattened tile decodes every element's per-dim indices from the flat offset,
so ``tile.index`` is still the element's own index there.  These tests pin the
choice, the flattened form and the kernels that keep no flatten choice;
``test_cute_pointwise_vec.py`` checks the results on a GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.backend_registry import get_backend
from helion._compiler.cute.autotune.heuristics.cute import CutePointwiseVecHeuristic
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import kernel_body
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import FlattenLoopSpec
import helion.language as hl

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_render_target():
        yield


def _zero_pad(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # out[y, c] = x[y, c - 3] for 3 <= c < 23, else 0: rows of 20 padded to
    # 27 columns, read as helion.inductor emits a pad (a flat input).
    rows, cols = out.shape
    for tile_y, tile_x in hl.tile([rows, cols]):
        col = tile_x.index
        inside = (col >= 3) & (col < 23)
        vals = hl.load(
            x, [(-3) + col + 20 * tile_y.index.reshape([tile_y, 1])], extra_mask=inside
        )
        out[tile_y, tile_x] = torch.where(inside, vals, 0.0)
    return out


def _row_sum_plus_index(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rows, cols, _ = x.shape
    for tile_y, tile_x in hl.tile([rows, cols]):
        out[tile_y, tile_x] = x[tile_y, tile_x, :].sum(-1) + tile_x.index
    return out


def _shift_by_tile_begin(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_y, tile_x in hl.tile(out.shape):
        out[tile_y, tile_x] = x[tile_y, tile_x] + tile_x.begin
    return out


def _gather_even_columns(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_y, tile_x in hl.tile(out.shape):
        out[tile_y, tile_x] = x[tile_y.index[:, None], tile_x.index[None, :] * 2]
    return out


def _column_in_block(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_d in hl.tile(out.shape):
        out[tile_m, tile_d] = (
            x[tile_m, tile_d] + (tile_d.index % tile_d.block_size)[None, :]
        )
    return out


def _plus_block_size(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_d in hl.tile(out.shape):
        out[tile_m, tile_d] = x[tile_m, tile_d] + tile_d.block_size
    return out


def _plus_column(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_y, tile_x in hl.tile(out.shape):
        out[tile_y, tile_x] = x[tile_y, tile_x] + tile_x.index
    return out


def _plus_one(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_y, tile_x in hl.tile(out.shape):
        out[tile_y, tile_x] = x[tile_y, tile_x] + 1.0
    return out


def _swap_column_pairs(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # out[m, d] = x[m, d] + (d ^ 1): the column indices of each pair swapped
    # through hl.split and hl.join.
    for tile_m, tile_d in hl.tile(out.shape):
        even, odd = hl.split(tile_d.index.view(tile_d.block_size // 2, 2))
        swapped = hl.join(odd, even).view(tile_d.block_size)
        out[tile_m, tile_d] = x[tile_m, tile_d] + swapped[None, :]
    return out


def _regroup_columns(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_d in hl.tile(out.shape):
        pairs = tile_d.index.view(tile_d.block_size // 2, 2)
        out[tile_m, tile_d] = x[tile_m, tile_d] + pairs.view(tile_d.block_size)
    return out


def _pad_args() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(8 * 20, dtype=torch.bfloat16),
        torch.empty(8, 27, dtype=torch.bfloat16),
    )


def _bind(
    fn: Callable[..., torch.Tensor], args: tuple[torch.Tensor, ...]
) -> BoundKernel[object]:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    return cpu_bind(kernel, args)


def _render(bound: BoundKernel[object], **overrides: object) -> str:
    config = bound.config_spec.default_config().config | overrides
    return kernel_body(bound.to_code(helion.Config.from_dict(config)))


# 8-element packets of the flattened [8, 27] tile, 16 threads.
_FLAT = {
    "flatten_loops": [True],
    "block_sizes": [1, 128],
    "num_threads": [1, 16],
    "cute_vector_widths": [1, 8],
}


def test_tile_index_keeps_the_flatten_choice() -> None:
    assert [
        (spec.block_ids, spec.pinned)
        for spec in _bind(_zero_pad, _pad_args()).config_spec.flatten_loops
    ] == [([0, 1], False)]


def test_flattened_tile_index_decodes_each_element() -> None:
    """Each element of a packet decodes its own row and column; the dense
    output, rows of 27 that no packet fits, is stored a packet at a time."""
    body = _render(_bind(_zero_pad, _pad_args()), **_FLAT)
    assert "for vec_lane_0_1 in cutlass.range_constexpr(8):" in body
    assert "indices_1 = offsets_0_1 % 27" in body
    assert "indices_0 = offsets_0_1 // 27" in body
    # col >= 3 reads the element's own column.
    assert "operator.ge(indices_1, " in body
    assert "_cute_store_u16_vec(out.iterator + cutlass.Int32(lane_base_0_1)" in body


def test_codegen_keeps_the_reoffered_choice() -> None:
    """Generating one config leaves the choice for the next: tile.index's
    codegen does not take it from the spec."""
    bound = _bind(_zero_pad, _pad_args())
    for flatten in (True, False, True):
        _render(bound, **(_FLAT | {"flatten_loops": [flatten]}))
    assert len(bound.config_spec.flatten_loops) == 1


def test_kernel_with_a_reduction_keeps_no_flatten_choice() -> None:
    args = (torch.empty(8, 32, 16), torch.empty(8, 32))
    assert len(_bind(_row_sum_plus_index, args).config_spec.flatten_loops) == 0


def test_tile_begin_keeps_no_flatten_choice() -> None:
    args = (torch.empty(8, 32), torch.empty(8, 32))
    assert len(_bind(_shift_by_tile_begin, args).config_spec.flatten_loops) == 0


@pytest.mark.parametrize("fn", [_swap_column_pairs, _regroup_columns])
def test_a_tile_index_rearranged_by_its_block_size_keeps_no_flatten_choice(
    fn: Callable[..., torch.Tensor],
) -> None:
    """hl.split, hl.join and a reshape that moves elements exchange them
    through a buffer of the rectangular tile's shape (``split_smem`` of
    block-size elements), which a flattened tile would index by its flat
    offset; a reshape that only adds a unit dim (the pad's
    ``tile_y.index.reshape([tile_y, 1])``) keeps the choice."""
    args = (torch.empty(8, 64), torch.empty(8, 64))
    assert len(_bind(fn, args).config_spec.flatten_loops) == 0


def test_a_narrow_packet_axis_that_reads_tile_index_keeps_the_tile_seeds() -> None:
    """A stride-1 axis that takes packets but not a CTA of them (128 fp32
    columns: 4-wide packets) flattens its pointwise seed, or gets a flat
    twin of it; where the kernel reads tile.index, the tile seeds stay, with
    no flat twin."""
    args = (torch.empty(64, 128), torch.empty(64, 128))

    def flattened(fn: Callable[..., torch.Tensor]) -> list[bool]:
        bound = _bind(fn, args)
        seeds = CutePointwiseVecHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds
        return [seed.config.get("flatten_loops") == [True] for seed in seeds]

    assert any(flattened(_plus_one))
    assert not any(flattened(_plus_column))


def test_a_partial_access_defers_only_its_loops_flatten_choice() -> None:
    """A partial access of a [bm, bn] loop defers that loop's choice, not
    the one of a [bm, bn, bk] loop that shares its blocks (a pure pointwise
    kernel would get both back); tile.index defers the choices that hold
    its block, but not two that share a block."""

    def deferred(**kwargs: object) -> list[list[int]]:
        spec = ConfigSpec(backend=get_backend("cute"), device=torch.device("cpu"))
        two = FlattenLoopSpec([0, 1])
        spec.flatten_loops.append(FlattenLoopSpec([0, 1, 2]))
        spec.flatten_loops.append(two)
        block = kwargs.pop("block")
        if kwargs.pop("own", False):
            kwargs["spec"] = two
        spec.defer_flatten_loop(block, **kwargs)  # pyrefly: ignore [bad-argument-type]
        assert len(spec.flatten_loops) == 0
        return [list(c.block_ids) for c in spec.cute_reflatten_candidates]

    assert deferred(block=0, own=True) == [[0, 1]]
    assert deferred(block=1, per_element=True) == [[0, 1, 2]]


def test_a_flattened_gather_through_tile_index_masks_each_element() -> None:
    """A gather through tile.index guards its load by each element's own
    indices: a tile flattened over [32, 33] in 64 threads has no coordinate
    along the rows, and ``thread_idx()[0] < 32`` would zero the elements of
    threads 32 to 63."""
    bound = _bind(_gather_even_columns, (torch.empty(32, 66), torch.empty(32, 33)))
    body = _render(bound, flatten_loops=[True], block_sizes=[4, 16])
    load = next(line for line in body.splitlines() if ".load() if " in line)
    assert "cute.arch.thread_idx()[0]) < 32" not in load, load
    assert "cutlass.Int32(indices_0) < 32" in load, load


@pytest.mark.parametrize(
    ("fn", "width"), [(_plus_block_size, 34), (_column_in_block, 64)]
)
def test_a_choice_of_a_loop_that_reads_its_block_size_is_pinned(
    fn: Callable[..., torch.Tensor], width: int
) -> None:
    """A loop that reads tile.block_size as a value gets its flatten choice
    pinned to False (pin_unflattened_loops): flattened over [4, 16], ``x +
    tile_d.block_size`` would add the flat tile's 64, and ``tile_d.index %
    tile_d.block_size`` would take each column modulo 64.  A choice that
    tile.index deferred comes back pinned too.  The slot stays, so configs
    that set it still load."""
    args = (torch.empty(8, width), torch.empty(8, width))
    bound = _bind(fn, args)
    (spec,) = bound.config_spec.flatten_loops
    assert spec.pinned
    config = helion.Config(block_sizes=[4, 16], flatten_loops=[True])
    bound.config_spec.normalize(config)
    assert config.flatten_loops == [False]
