"""A cross-thread reduction spans the launch's threads along its axis.

Sibling loop paths reuse a CUDA axis and the launch takes the widest of
them.  A narrower sibling -- a 16-row K2 ``hl.tile`` beside a 32-thread K
loop, in a kernel whose matmul keeps the SIMT tile strategies at their own
extents -- runs with surplus threads: their tile mask fails, so they hold the
identity, but they execute every collective of the body.  The per-element
strided thread reduction sized its group by the strategy's own extent, so
the surplus threads reduced among themselves: every thread of a row is meant
to hold the K2 column sum, the surplus threads held zero, and the stores all
of them issue to one address raced (2, 0 and 431 wrong elements of 16,384
across three launches of byte-identical code).  The group now spans the
launch's threads along the axis, and a lane-looped reduction over such an
axis is declined.
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
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_then_k2_scaled_rows(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x @ y) * z.sum(1)`` and its row totals; the K2 sum's tile loop is narrower than the K loop's thread axis."""
    m, k = x.size()
    n = hl.specialize(y.size(1))
    k2 = z.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    tot = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        s = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k2 in hl.tile(k2):
            s = s + acc * z[tile_m, tile_k2, :].sum(dim=1)
        total = s.sum(dim=-1)
        out[tile_m, :] = s
        tot[tile_m] = total
    return out, tot


def _inputs(
    device: torch.device | str, *, k2: int = 16, k: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, k, generator=generator)
    y = torch.randn(k, 256, generator=generator)
    z = torch.randn(64, k2, 256, generator=generator)
    return x.to(device), y.to(device), z.to(device)


def _config(
    bound: object, *, block_sizes: list[int], num_threads: list[int]
) -> helion.Config:
    settings = dict(bound.config_spec.default_config().config)  # pyrefly: ignore [missing-attribute]
    settings["block_sizes"] = block_sizes
    settings["num_threads"] = num_threads
    return helion.Config.from_dict(settings)


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


def _k2_reduce_lines(code: str) -> list[str]:
    return [
        line.strip()
        for line in code.splitlines()
        if ("_cute_grouped_reduce" in line or "warp_reduction" in line)
        and "import" not in line
        and "dot_" not in line
        and "total" not in line
    ]


def test_the_k2_reduction_spans_the_launch_threads_of_its_axis() -> None:
    """The K2 tile of 16 rows sits on the K loop's 32-thread axis; its reduction is the cross-warp two-stage combine over all 64 launched threads of the row, not a warp group of the 16 real ones."""
    with _cpu_codegen():
        bound = _cpu_bind(_matmul_then_k2_scaled_rows, _inputs("cpu"))
        code = bound.to_code(
            _config(bound, block_sizes=[16, 64, 16], num_threads=[1, 32])
        )
    assert "block=(2, 32, 1)" in code
    assert "cutlass.Int32(cute.arch.thread_idx()[1]) < _BLOCK_SIZE_3" in code
    (k2_reduce,) = _k2_reduce_lines(code)
    assert "_cute_grouped_reduce_shared_two_stage(" in k2_reduce
    assert "pre=2, group_span=64, group_count=1" in k2_reduce
    assert "_cute_grouped_reduce_warp(" not in code


def test_a_lane_looped_reduction_narrower_than_the_launch_is_declined() -> None:
    """A K2 tile of 32 rows on 16 threads walks a lane loop; the 16 surplus threads of the 32-thread launch walk it too, so neither a group of 16 (their partial races on every owner-guarded store) nor one of 32 (a strided layout counts their aliased elements twice) is a reduction."""
    with _cpu_codegen():
        bound = _cpu_bind(_matmul_then_k2_scaled_rows, _inputs("cpu", k2=32))
        with pytest.raises(
            helion.exc.BackendUnsupported, match="narrower than the launch"
        ):
            bound.to_code(
                _config(bound, block_sizes=[16, 64, 32], num_threads=[1, 32, 16])
            )
        code = bound.to_code(
            _config(bound, block_sizes=[16, 64, 32], num_threads=[1, 32])
        )
    assert "cute.arch.warp_reduction_sum(load_2, threads_in_group=32)" in code


def test_a_thread_shape_beyond_the_launch_limits_is_declined_at_codegen() -> None:
    """Serial reductions leave the z axis free; an actual z extent above 64 rejects."""

    @helion.kernel(backend="cute", static_shapes=True)
    def add(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for i, j, k in hl.tile(x.size()):
            out[i, j, k] = x[i, j, k] + 1
        return out

    with _cpu_codegen():
        bound = _cpu_bind(_matmul_then_k2_scaled_rows, _inputs("cpu", k=256))
        code = bound.to_code(
            _config(bound, block_sizes=[16, 128, 16], num_threads=[2, 128])
        )
        assert "block=(2, 128, 1)" in code

        bound = _cpu_bind(add, (torch.empty(2, 2, 128),))
        with pytest.raises(helion.exc.BackendUnsupported, match="per-axis limits"):
            bound.to_code(
                helion.Config(block_sizes=[2, 2, 128], num_threads=[2, 2, 128])
            )
        code = bound.to_code(
            helion.Config(block_sizes=[2, 2, 128], num_threads=[2, 2, 64])
        )
    assert "block=(2, 2, 64)" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_surplus_threads_hold_the_row_sum_and_launches_agree_bitwise() -> None:
    x, y, z = _inputs(DEVICE)
    bound = _matmul_then_k2_scaled_rows.bind((x, y, z))
    config = _config(bound, block_sizes=[16, 64, 16], num_threads=[1, 32])
    (k2_reduce,) = _k2_reduce_lines(bound.to_code(config))
    assert "group_span=64" in k2_reduce
    fn = bound.compile_config(config)
    first_out, first_tot = fn(x, y, z)
    expected = (x @ y) * z.sum(dim=1)
    torch.testing.assert_close(first_out, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(first_tot, expected.sum(dim=1), rtol=1e-3, atol=1e-2)
    for _ in range(5):
        out, tot = fn(x, y, z)
        assert torch.equal(out, first_out)
        assert torch.equal(tot, first_tot)
