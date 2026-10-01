"""A reduction over a tile dim combines across the threads that hold the dim, and only those.

A block of one holds no threads: its strategy's starting axis is the next
block's, so ``x[tile_b, tile_m].sum(dim=0)`` over a one-row tile combined the
128 column threads and every column received the row's total.  The single
element passes through now.

A block whose loop has exited still lives one element per thread along its
axis.  ``acc.mean(dim=1)`` after the column loop, with one warp per CTA, took
the direct-reduction shortcut that ``codegen_reduction`` routes to the
loop-carried passthrough for a block without a live thread axis, so each
thread kept its own column.  Such a block keeps the strided warp combine now.

A mean over a masked tile dim divides by the tile's extent, ``min(begin +
block, end) - begin``; a masked loop that carries no end variable has no such
extent, and the mean is declined there rather than divided by the block.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler import tile_strategy
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])


def _column_sums_added(x: torch.Tensor) -> torch.Tensor:
    """The column sums of every row tile added into ``out``."""
    b, m = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].sum(dim=0))
    return out


def _column_means_added(x: torch.Tensor) -> torch.Tensor:
    b, m = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].mean(dim=0))
    return out


def _row_means_after_the_column_loop(x: torch.Tensor) -> torch.Tensor:
    """The mean over a registered block dim, taken once its loop has ended."""
    b, m = x.size()
    bs = hl.register_block_size(m)
    out = torch.empty([b], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(b):
        acc = hl.zeros([tile_b, bs], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=bs):
            acc = acc + x[tile_b, tile_m]
        out[tile_b] = acc.mean(dim=1)
    return out


def _row_sums_after_the_column_loop(x: torch.Tensor) -> torch.Tensor:
    b, m = x.size()
    bs = hl.register_block_size(m)
    out = torch.empty([b], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(b):
        acc = hl.zeros([tile_b, bs], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=bs):
            acc = acc + x[tile_b, tile_m]
        out[tile_b] = acc.sum(dim=1)
    return out


def _kernel(
    fn: Callable[..., torch.Tensor], *, static_shapes: bool = True
) -> helion.Kernel:
    return helion.kernel(
        fn, backend="cute", static_shapes=static_shapes, autotune_effort="none"
    )


def _config(
    bound: BoundKernel, block_sizes: list[int], num_threads: list[int]
) -> helion.Config:
    config = dict(bound.config_spec.default_config().config)
    config.update(
        block_sizes=block_sizes,
        num_threads=num_threads,
        cute_vector_widths=[1] * len(block_sizes),
        cute_lane_layouts=["strided"] * len(block_sizes),
    )
    return helion.Config.from_dict(config)


@contextmanager
def _cpu_codegen() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _kernel_body(code: str) -> str:
    return "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("#")
    )


class _WithoutEndVariable(tile_strategy.LoopDimInfo):
    """A loop's dim info with its end variable withheld.

    No shipped strategy hosts a reduction in a masked loop without an end
    variable (a loop with a reduction is never flattened), so the shape is
    modelled rather than reached.
    """

    def __init__(
        self,
        *,
        begin_var_name: str | None = None,
        begin_expr: object = None,
        end_var_name: str | None = None,
        end_expr: object = None,
        mask_has_lower_bound: bool = False,
    ) -> None:
        super().__init__(
            begin_var_name=begin_var_name,
            begin_expr=begin_expr,  # pyrefly: ignore [bad-argument-type]
            end_var_name=None,
            end_expr=end_expr,  # pyrefly: ignore [bad-argument-type]
            mask_has_lower_bound=mask_has_lower_bound,
        )


@pytest.mark.parametrize(
    "num_threads", [[1, 128], [1, 64], [1, 32]], ids=["cta", "lanes", "warp"]
)
def test_a_reduction_over_a_one_row_tile_passes_the_element_through(
    num_threads: list[int],
) -> None:
    """``block_sizes=[1, 128]``: the row block holds one element per thread and no thread axis, so there is nothing to combine; the 128 column threads were folded into one total (every column got the row's sum, 130 times too large over 32 rows)."""
    with _cpu_codegen():
        bound = _cpu_bind(_kernel(_column_sums_added), (torch.randn(32, 128),))
        code = bound.to_code(_config(bound, [1, 128], num_threads))
    body = _kernel_body(code)
    assert "sum_1 = cutlass.Float32(load)" in body, code
    for combine in ("_cute_grouped_reduce", "warp_reduction", "strided_lane"):
        assert combine not in body, code
    assert body.count("cute.arch.atomic_add(") == 1, code


@pytest.mark.parametrize(
    ("num_threads", "combine"),
    [
        pytest.param([32, 1], "_cute_grouped_reduce_warp(acc, 'sum'", id="warp"),
        pytest.param([16, 1], "_cute_grouped_reduce_warp(acc, 'sum'", id="half_warp"),
        pytest.param(
            [128, 1], "_cute_grouped_reduce_shared_two_stage(acc, 'sum'", id="cta"
        ),
    ],
)
def test_a_reduction_after_the_blocks_loop_combines_its_threads(
    num_threads: list[int], combine: str
) -> None:
    """One row per CTA on ``bs`` column threads: ``acc.mean(dim=1)`` after the column loop combines the threads' columns (one warp per CTA took the direct shortcut into the loop-carried passthrough and kept each thread's own column)."""
    with _cpu_codegen():
        bound = _cpu_bind(
            _kernel(_row_means_after_the_column_loop), (torch.randn(37, 100),)
        )
        code = bound.to_code(_config(bound, [num_threads[0], 1], num_threads))
    body = _kernel_body(code)
    assert combine in body, code
    assert f"group_span={num_threads[0]}" in body, code
    assert "mean_extra = cutlass.Float32(acc)" not in body, code
    assert "/ _BLOCK_SIZE_0" in body, code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "num_threads", [[1, 128], [1, 64], [1, 32]], ids=["cta", "lanes", "warp"]
)
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_one_row_tiles_give_the_column_sums(
    num_threads: list[int], static_shapes: bool, reduction: str
) -> None:
    fn = _column_sums_added if reduction == "sum" else _column_means_added
    for shape in ((32, 128), (37, 100)):
        x = torch.randn(shape, device=DEVICE)
        bound = _kernel(fn, static_shapes=static_shapes).bind((x,))
        config = _config(bound, [1, 128], num_threads)
        out = bound.compile_config(config)(x)
        torch.testing.assert_close(out, x.sum(dim=0), rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block", [16, 32, 64, 128])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_a_reduction_after_the_blocks_loop_matches_reference(
    block: int, static_shapes: bool, reduction: str
) -> None:
    """``acc`` holds ``block`` columns, zero past the row's end: the sum is the row's sum and the mean divides it by the block."""
    fn = (
        _row_sums_after_the_column_loop
        if reduction == "sum"
        else _row_means_after_the_column_loop
    )
    x = torch.randn(37, 100, device=DEVICE)
    bound = _kernel(fn, static_shapes=static_shapes).bind((x,))
    config = _config(bound, [block, 1], [block, 1])
    out = bound.compile_config(config)(x)
    expected = x.sum(dim=1) if reduction == "sum" else x.sum(dim=1) / block
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)


def test_a_mean_in_a_masked_loop_without_an_end_variable_is_declined() -> None:
    """The row tile of 37 rows in blocks of 8 is masked, so ``mean(dim=0)`` divides by the tile's extent off the loop's end variable; without one the mean is declined instead of dividing by the block."""
    with _cpu_codegen():
        bound = _cpu_bind(
            _kernel(_column_means_added, static_shapes=False), (torch.randn(37, 100),)
        )
        config = _config(bound, [8, 32], [1, 32])
        body = _kernel_body(bound.to_code(config))
        assert "- tile_offset_0" in body, body
        assert "_helion_inv_div" not in body, body
        bound = _cpu_bind(
            _kernel(_column_means_added, static_shapes=False), (torch.randn(37, 100),)
        )
        with (
            patch("helion._compiler.tile_strategy.LoopDimInfo", _WithoutEndVariable),
            pytest.raises(exc.BackendUnsupported, match="without an end variable"),
        ):
            bound.to_code(config)
