"""Knobs no contraction can act on stay out of the CuTe search surface.

``cute_collective_*`` is read only by the collective marker, which runs from
the addmm/baddbmm lowering and ``hl.dot`` with an accumulator and needs a
tiled K loop or a short static full-slice K; ``epilogue_subtile`` splits the
store of an epilogue chain, which a matmul on the direct/serial path (no
active K tile) does not survive. Both are decided from the DeviceIR at bind
time; normalization keeps recording the collective defaults so configs and
journals do not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_full_slice_matmul import _bind
from test.test_cute_full_slice_matmul import _inputs
from test.test_cute_full_slice_matmul import _se_block_full_slice_fwd

from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FunctionType


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


def _tiled_k_addmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        out[row, column] = acc.to(a.dtype)
    return out


def _tiled_k_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(a.size(1)):
            acc = hl.dot(a[row, inner], b[inner, column], acc=acc)
        out[row, column] = acc.to(a.dtype)
    return out


def _full_slice_addmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        acc = torch.addmm(acc, a[row, :], b[:, column])
        out[row, column] = acc.to(a.dtype)
    return out


def _full_slice_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        out[row, column] = torch.relu(a[row, :] @ b[:, column])
    return out


def _collective_offered(bound: object) -> bool:
    spec = bound.config_spec  # type: ignore[attr-defined]
    fields = spec._flat_fields()
    offered = "cute_collective_mma" in fields
    assert (
        offered
        == ("cute_collective_copy" in fields)
        == ("cute_gathered_mma_n" in fields)
    )
    assert offered is spec.cute_collective_knobs_enabled
    return offered


@pytest.mark.parametrize(
    ("function", "shape", "offered"),
    [
        (_tiled_k_addmm, (64, 64, 256), True),
        (_tiled_k_dot, (64, 64, 256), True),
        # A full-slice K takes the synthetic-lane route only up to 128.
        (_full_slice_addmm, (64, 64, 64), True),
        (_full_slice_addmm, (64, 64, 256), False),
        # aten.mm never calls the marker.
        (_full_slice_mm, (64, 64, 64), False),
    ],
    ids=["tiled_addmm", "tiled_dot", "short_full_slice", "long_full_slice", "mm"],
)
def test_collective_knobs_follow_the_marker_predicate(
    function: FunctionType, shape: tuple[int, int, int], offered: bool
) -> None:
    bound = _bind(function, _inputs(shape), enabled=False)
    assert _collective_offered(bound) is offered
    # Normalization still records the defaults: recorded configs keep their
    # keys whether or not the search may move them.
    config = bound.config_spec.default_config()
    assert config.config.get("cute_collective_mma") is False
    assert "cute_collective_compute" in config.config


def test_se_block_forward_surfaces_before_and_after_the_stripmine() -> None:
    x, w = _inputs((128, 128, 128))
    original = _bind(_se_block_full_slice_fwd, (x, w), enabled=False)
    stripmined = _bind(_se_block_full_slice_fwd, (x, w))
    # The original single-axis kernel keeps a K-loop-less aten.mm: no
    # collective site, and no epilogue store split over the direct path.
    assert _collective_offered(original) is False
    assert original.config_spec.epilogue_subtile_candidate_enabled is False
    # The rewritten kernel has an addmm K loop; both knobs apply again.
    assert _collective_offered(stripmined) is True
    assert stripmined.config_spec.epilogue_subtile_candidate_enabled is True


def test_epilogue_subtile_candidate_needs_an_active_k_tile() -> None:
    direct = _bind(_full_slice_mm, _inputs((64, 64, 64)), enabled=False)
    assert direct.config_spec.epilogue_subtile_candidate_enabled is False
    looped = _bind(_tiled_k_addmm, _inputs((64, 64, 256)), enabled=False)
    assert looped.config_spec.epilogue_subtile_candidate_enabled is True
