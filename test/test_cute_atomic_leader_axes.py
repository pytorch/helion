"""An atomic runs on one thread of every launch axis it does not vary along.

A CuTe atomic's leader guard (``_cute_unindexed_leader_axes`` in
``cute/atomic_ops.py``) restricted the axes of the loops around it and of the
loops emitted before it.  An axis whose loop comes later in the tile body (a
row reduction after the atomic) or belongs to another root loop sharing the
launch ran the atomic on every one of its threads: 1024 additions per element
beside a 1024-thread row sum.  So did the axis of a free ``hl.arange`` dim
emitted after the atomic, which codegen claims only when it reaches it
(``guard_atomics_on_later_thread_axes``).
"""

from __future__ import annotations

import ast
import re

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _add_then_sum(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile in hl.tile(y.size(0)):
        hl.atomic_add(out, [tile], x[tile])
        sums[tile] = y[tile, :].sum(-1)
    return sums


def _add_beside_sum(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m], x[tile_m])
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = y[tile_r, :].sum(-1)
    return sums


def _old_values_then_sum(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    old = torch.empty_like(x)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile in hl.tile(y.size(0)):
        old[tile] = hl.atomic_add(out, [tile], x[tile])
        sums[tile] = y[tile, :].sum(-1)
    return old, sums


def _add_ahead_of_arange(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile], x[tile])
        cols = hl.arange(y.size(1))
        doubled[tile, cols] = y[tile, cols] * 2
    return doubled


def _acquire_add_ahead_of_arange(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile], x[tile], sem="acquire")
        cols = hl.arange(y.size(1))
        doubled[tile, cols] = y[tile, cols] * 2
    return doubled


def _acq_rel_add_ahead_of_arange(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile], x[tile], sem="acq_rel")
        cols = hl.arange(y.size(1))
        doubled[tile, cols] = y[tile, cols] * 2
    return doubled


# The atomic ahead of the hl.arange in each memory order whose result goes
# unused: acquire and acq_rel atomics are emitted with their barriers.
_ARANGE_ADDS = [
    _add_ahead_of_arange,
    _acquire_add_ahead_of_arange,
    _acq_rel_add_ahead_of_arange,
]


def _old_values_ahead_of_arange(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    old = torch.empty_like(x)
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        old[tile] = hl.atomic_add(out, [tile], x[tile])
        cols = hl.arange(y.size(1))
        doubled[tile, cols] = y[tile, cols] * 2
    return old, doubled


# Each kernel, its block sizes, the row counts of x and y, and the launch:
# the atomic ahead of a row sum in its tile body, and in a root loop beside
# another root's row sum, at one row per tile (the sum's 1024 threads on axis
# 0) and at 16-row tiles (the sum's 64 threads per row on axis 0 beside 16
# rows on axis 1).  The sums of 4096 columns roll in 1024-column chunks; a
# persistent sum's block is on the grid state, which kept the atomic on axis
# 0's leader.
_CASES = [
    (_add_then_sum, [1], 64, 64, "(1024, 1, 1)"),
    (_add_beside_sum, [1, 1], 1024, 64, "(1024, 1, 1)"),
    (_add_beside_sum, [16, 16], 1024, 64, "(64, 16, 1)"),
]


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(fn, backend="cute", static_shapes=True, autotune_effort="none")


def _config(blocks: list[int]) -> helion.Config:
    return helion.Config(block_sizes=blocks, reduction_loops=[1024])


def _atomic_conditions(code: str) -> list[set[str]]:
    """The conditions of the ``if`` statements around each atomic, each
    operand of an ``and`` on its own."""
    tree = ast.parse(code)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    conditions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("atomic_add"):
            guards = set()
            child, parent = node, parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.If) and child in parent.body:
                    test = parent.test
                    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
                        guards.update(ast.unparse(value) for value in test.values)
                    else:
                        guards.add(ast.unparse(test))
                child, parent = parent, parents.get(parent)
            conditions.append(guards)
    return conditions


@pytest.mark.parametrize(("fn", "blocks", "rows", "sum_rows", "launch"), _CASES)
def test_an_atomic_runs_on_one_thread_of_a_later_loops_axis(
    fn: object, blocks: list[int], rows: int, sum_rows: int, launch: str
) -> None:
    """The row sum's threads on axis 0 run every statement of the tile
    body, and of the other root's body, including the atomic emitted before
    the sum's loop: only thread 0 of the axis issues it."""
    args = (torch.randn(rows), torch.randn(sum_rows, 4096), torch.zeros(rows))
    with mock_cuda_unavailable():
        code = cpu_bind(_kernel(fn), args).to_code(_config(blocks))
    assert f"block={launch}" in code, code
    (guards,) = _atomic_conditions(code)
    assert "cute.arch.thread_idx()[0] == 0" in guards, code


def test_an_atomic_result_ahead_of_a_later_loop_reaches_every_thread() -> None:
    """The row sum's axis is part of the launch the strategies lay out when
    the atomic is emitted: thread 0 of it issues the atomic and shares the
    old value through shared memory."""
    args = (torch.randn(64), torch.randn(64, 4096), torch.zeros(64))
    with mock_cuda_unavailable():
        code = cpu_bind(_kernel(_old_values_then_sum), args).to_code(_config([1]))
    (guards,) = _atomic_conditions(code)
    assert "cute.arch.thread_idx()[0] == 0" in guards, code
    assert "_atomic_prev_smem" in code, code


def _arange_args(
    device: str = "cpu", rows: int = 64, cols: int = 128
) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(rows, device=device),
        torch.randn(rows, cols, device=device),
        torch.randn(rows, device=device),
    )


@pytest.mark.parametrize("fn", _ARANGE_ADDS, ids=["relaxed", "acquire", "acq_rel"])
def test_an_atomic_runs_on_one_thread_of_a_later_arange_axis(fn: object) -> None:
    """The ``hl.arange`` dim after the atomic gets thread axis 0 (128
    threads) only when codegen reaches it, after the atomic: once codegen is
    done, the atomic runs on thread 0 of it, also with the barriers of an
    acquire or acq_rel atomic whose result nothing reads."""
    with mock_cuda_unavailable():
        code = cpu_bind(_kernel(fn), _arange_args()).to_code(
            helion.Config(block_sizes=[1])
        )
    assert "block=(128, 1, 1)" in code, code
    (guards,) = _atomic_conditions(code)
    assert "cute.arch.thread_idx()[0] == 0" in guards, code


# 64 rows on 16 and on 4 threads walk lanes beside the hl.arange's 128
# columns, and the passes simplify a lane-walking tile's bounds against the
# launch shape they prove.
_LANE_TILES = {16: "(16, 64, 1)", 4: "(4, 128, 1)"}


@pytest.mark.parametrize("threads", list(_LANE_TILES))
def test_a_lane_walking_tile_keeps_the_leader_of_a_later_arange_axis(
    threads: int,
) -> None:
    """The proven launch shape leaves out the axis of the hl.arange claimed
    after the atomic, so the bounds passes cannot drop the atomic's leader
    test on it ("[cutedsl] Keep a free hl.arange axis out of the proven launch
    shape")."""
    with mock_cuda_unavailable():
        code = cpu_bind(_kernel(_add_ahead_of_arange), _arange_args()).to_code(
            helion.Config(block_sizes=[64], num_threads=[threads])
        )
    assert f"block={_LANE_TILES[threads]}" in code, code
    (guards,) = _atomic_conditions(code)
    assert "cute.arch.thread_idx()[1] == 0" in guards, code


def test_a_later_axis_leader_joins_the_atomic_mask_guard() -> None:
    """A partial tile's atomic is already under its mask: the leader test
    of the hl.arange axis claimed after it joins that ``if``."""
    with mock_cuda_unavailable():
        code = cpu_bind(
            _kernel(_add_ahead_of_arange), _arange_args(rows=60, cols=32)
        ).to_code(helion.Config(block_sizes=[16]))
    assert "block=(16, 32, 1)" in code, code
    assert re.search(
        r"if mask_0 and cute\.arch\.thread_idx\(\)\[1\] == 0:\n\s+cute\.arch\.atomic_add\(",
        code,
    ), code


def test_an_atomic_result_ahead_of_an_arange_axis_declines() -> None:
    """Only thread 0 of the arange's axis may issue the atomic, and the old
    value it returns would not reach the axis's other threads."""
    with (
        mock_cuda_unavailable(),
        pytest.raises(exc.BackendUnsupported, match="ahead of an hl.arange dim"),
    ):
        cpu_bind(_kernel(_old_values_ahead_of_arange), _arange_args()).to_code(
            helion.Config(block_sizes=[1])
        )


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteAtomicLeaderAxes(TestCase):
    def test_atomics_beside_a_row_sum_add_once(self) -> None:
        for fn, blocks, rows, sum_rows, _launch in _CASES:
            x = torch.randn(rows, device=DEVICE)
            y = torch.randn(sum_rows, 4096, device=DEVICE)
            out = torch.randn(rows, device=DEVICE)
            expected = out + x
            args = (x, y, out)
            sums = _kernel(fn).bind(args).compile_config(_config(blocks))(*args)
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(sums, y.sum(-1), rtol=1e-4, atol=1e-3)

    def test_an_atomic_result_ahead_of_a_row_sum(self) -> None:
        x = torch.randn(64, device=DEVICE)
        y = torch.randn(64, 4096, device=DEVICE)
        out = torch.randn(64, device=DEVICE)
        before = out.clone()
        args = (x, y, out)
        compiled = _kernel(_old_values_then_sum).bind(args)
        old, sums = compiled.compile_config(_config([1]))(*args)
        torch.testing.assert_close(old, before)
        torch.testing.assert_close(out, before + x)
        torch.testing.assert_close(sums, y.sum(-1), rtol=1e-4, atol=1e-3)

    def test_an_atomic_ahead_of_an_arange_adds_once(self) -> None:
        cases = [(fn, 64, 128, helion.Config(block_sizes=[1])) for fn in _ARANGE_ADDS]
        # The partial last tile's atomic under its mask and the leader test.
        cases.append((_add_ahead_of_arange, 60, 32, helion.Config(block_sizes=[16])))
        cases += [
            (
                _add_ahead_of_arange,
                64,
                128,
                helion.Config(block_sizes=[64], num_threads=[n]),
            )
            for n in _LANE_TILES
        ]
        for fn, rows, cols, config in cases:
            x, y, out = args = _arange_args(DEVICE, rows, cols)
            expected = out + x
            compiled = _kernel(fn).bind(args)
            doubled = compiled.compile_config(config)(*args)
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(doubled, y * 2)
