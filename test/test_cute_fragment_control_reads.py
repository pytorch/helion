"""CTA control reads must finish before another warp may reuse their storage."""

from __future__ import annotations

import ast

import numpy as np
import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import kernel_body
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _conditional_scan(
    x: torch.Tensor,
    control: torch.Tensor,
    tail: torch.Tensor,
    work: torch.Tensor,
    repeat: hl.constexpr,
):
    for i in hl.grid(3):
        if repeat:
            for _j in hl.grid(control[i]):
                work[i, :] = torch.cumsum(x[i, :] + 1, dim=0)
        else:
            if control[i]:
                work[i, :] = torch.cumsum(x[i, :] + 1, dim=0)
        tail[i] = control[(i + 1) % 3]
    return tail, work


@pytest.mark.parametrize("repeat", [False, True], ids=["branch", "zero-trip"])
def test_shared_control_read_finishes_before_reuse(repeat: bool) -> None:
    control = torch.zeros(3, dtype=torch.int32 if repeat else torch.bool)
    x = torch.ones((3, 32))
    with mock_cuda_unavailable():
        bound = cpu_bind(
            _conditional_scan,
            (x, control, torch.empty_like(control), torch.empty_like(x), repeat),
        )
        code = bound.to_code(helion.Config(block_sizes=[]), output_origin_lines=False)
    (kernel,) = [
        n for n in ast.parse(kernel_body(code)).body if isinstance(n, ast.FunctionDef)
    ]
    (branch,) = [
        stmt
        for stmt in kernel.body
        if isinstance(stmt, ast.For if repeat else ast.If)
        and "x.iterator" in ast.unparse(stmt)
    ]
    assert isinstance(branch, (ast.If, ast.For))
    header = branch.iter if isinstance(branch, ast.For) else branch.test
    before = kernel.body[: kernel.body.index(branch)]
    after = kernel.body[kernel.body.index(branch) + 1 :]

    # Identify the two physical publications from the input, without naming
    # the allocation. They reuse one shared cell across the skipped region.
    def published_cell(stmts: list[ast.stmt]) -> str:
        (loading,) = [stmt for stmt in stmts if "control.iterator" in ast.unparse(stmt)]
        (cell,) = [
            a.targets[0]
            for a in ast.walk(loading)
            if isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Subscript)
        ]
        return ast.unparse(cell)

    cell = published_cell(before)
    assert cell == published_cell(after)
    ((load_index, local),) = [
        (i, a.targets[0].id)
        for i, a in enumerate(before)
        if isinstance(a, ast.Assign)
        and isinstance(a.targets[0], ast.Name)
        and any(
            isinstance(n, ast.Subscript) and ast.unparse(n) == cell
            for n in ast.walk(a.value)
        )
    ]
    assert any(isinstance(n, ast.Name) and n.id == local for n in ast.walk(header))
    assert not any(isinstance(n, ast.Subscript) for n in ast.walk(header))
    assert any(
        ast.unparse(stmt) == "cute.arch.sync_threads()"
        for stmt in before[load_index + 1 :]
    )


def test_multidimensional_fragment_header_uses_its_outer_tile():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        width = hl.register_block_size(4)
        for outer in hl.tile(4, 8, block_size=width):
            for rows, cols in hl.tile([4, outer.end], block_size=[width, 4]):
                out[rows, cols] = torch.cumsum(x[rows, cols] + 1, dim=1)

    x = torch.ones((4, 8), dtype=torch.int64)
    out = torch.full_like(x, -73)
    with mock_cuda_unavailable():
        source = cpu_bind(scan, (x, out)).to_code(
            helion.Config(block_sizes=[4]), output_origin_lines=False
        )
    simulate_independent_fragment(source, {"x": x}, {"out": out}, 1)
    expected = (2 * torch.arange(1, 5)).repeat(4, 2)
    torch.testing.assert_close(out, expected)


def test_fragment_loop_preserves_the_configured_int64_induction_value():
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int64)
    def scan(x, lo, out):
        for row in hl.grid(1):
            begin = lo[row]
            for i in hl.grid(begin, begin + 1):
                out[row, :] = torch.cumsum(x[row, :] + i, dim=0)

    x = torch.zeros((1, 4), dtype=torch.int64)
    lo = torch.tensor([2**32])
    out = torch.full_like(x, -73)
    with mock_cuda_unavailable():
        source = cpu_bind(scan, (x, lo, out)).to_code(
            helion.Config(block_sizes=[]), output_origin_lines=False
        )
    simulate_independent_fragment(
        source, {"x": x, "lo": lo}, {"out": out}, 1, int32=np.int32, int64=np.int64
    )
    torch.testing.assert_close(out, (2**32 * torch.arange(1, 5)).reshape(1, 4))
