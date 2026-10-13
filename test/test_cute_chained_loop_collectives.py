from __future__ import annotations

import ast
from dataclasses import replace

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.chained_loop import discover_chained_loop
from helion._compiler.cute.chained_matmul import _classify_chained_graph
from helion._compiler.device_ir import HelperFunctionGraphInfo
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prefix_coefficient_recurrence(
    a: torch.Tensor,
    b: torch.Tensor,
    decay: torch.Tensor,
    delta: torch.Tensor,
    initial: torch.Tensor,
    steps: int,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(steps)
    axis = hl.specialize(axis)
    batches, _, m, k = a.shape
    n = b.size(-1)
    history = torch.empty(
        (batches, max(steps, 1), m, n), dtype=torch.float32, device=a.device
    )
    final = torch.empty_like(initial)
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        state = initial[bi.id, mi, ni]
        for ti in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            row_prefix = hl.cumsum(decay[bi.id, ti.id, mi], dim=0)
            matrix_prefix = hl.cumsum(delta[bi.id, ti.id, mi, kk], dim=axis)
            energy = (matrix_prefix * matrix_prefix).sum(1)
            left = (a[bi.id, ti.id, mi, kk].float() * torch.exp(matrix_prefix)).to(
                a.dtype
            )
            product = hl.dot(left, b[bi.id, ti.id, kk, ni], out_dtype=torch.float32)
            state = state * torch.exp(row_prefix[:, None]) + product / (
                1.0 + energy[:, None]
            )
            history[bi.id, ti.id, mi, ni] = state
        final[bi.id, mi, ni] = state
    return history, final


def _inputs(device: str | torch.device, steps: int, axis: int) -> tuple:
    torch.manual_seed(913)
    return (
        torch.randn((2, max(steps, 1), 16, 16), device=device, dtype=torch.bfloat16)
        * 0.125,
        torch.randn((2, max(steps, 1), 16, 32), device=device, dtype=torch.bfloat16)
        * 0.125,
        torch.randn((2, max(steps, 1), 16), device=device) * 0.015625,
        torch.randn((2, max(steps, 1), 16, 16), device=device) * 0.015625,
        torch.randn((2, 16, 32), device=device) * 0.125,
        steps,
        axis,
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _postdot_recurrence(a, b, initial, steps: int, rank: int, axis: int):
    steps = hl.specialize(steps)
    rank = hl.specialize(rank)
    axis = hl.specialize(axis)
    batches, _, m, k = a.shape
    n = b.size(-1)
    history = torch.empty((batches, steps, m, n), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        state = initial[bi.id, mi, ni]
        for ti in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            product = hl.dot(
                a[bi.id, ti.id, mi, kk],
                b[bi.id, ti.id, kk, ni],
                out_dtype=torch.float32,
            )
            if rank == 1:
                prefix = product + hl.cumsum(product[:, 0], dim=0)[:, None]
            else:
                prefix = hl.cumsum(product, dim=axis)
            state = 0.5 * state + prefix
            history[bi.id, ti.id, mi, ni] = state
        final[bi.id, mi, ni] = state
    return history, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _postdot_chain(a, b, c, rank: int, axis: int):
    rank = hl.specialize(rank)
    axis = hl.specialize(axis)
    m, k = a.shape
    p, n = c.shape
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for mi, ni in hl.tile([m, n], block_size=[128, 32]):
        kk, pp = hl.arange(k), hl.arange(p)
        first = hl.dot(a[mi, kk], b[kk, pp], out_dtype=torch.float32)
        if rank == 1:
            prefix = first + hl.cumsum(first[:, 0], dim=0)[:, None]
        else:
            prefix = hl.cumsum(first, dim=axis)
        second = hl.dot(prefix.to(c.dtype), c[pp, ni], out_dtype=torch.float32)
        if rank == 1:
            terminal = second + hl.cumsum(second[:, 0], dim=0)[:, None]
        else:
            terminal = hl.cumsum(second, dim=axis)
        output[mi, ni] = terminal
    return output


def _postdot_inputs(
    device: str | torch.device, *, loop: bool, rank: int, axis: int
) -> tuple:
    torch.manual_seed(915)
    shapes = (
        ((2, 3, 16, 16), (2, 3, 16, 32), (2, 16, 32))
        if loop
        else ((128, 16), (16, 32), (32, 32))
    )
    tensors = tuple(
        torch.randint(-2, 3, shape, device=device).to(
            torch.float32 if loop and index == 2 else torch.bfloat16
        )
        / 64
        for index, shape in enumerate(shapes)
    )
    return (*tensors, 3, rank, axis) if loop else (*tensors, rank, axis)


def _postdot_source(source: str, *, loop: bool) -> None:
    assert "chain_collective_0" in source
    assert "chain_scan_" not in source
    tree = ast.parse(source)
    if loop:
        body = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and "chain_loop_index" in node.target.id
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "alloc_smem"
            for node in ast.walk(body)
        )
    # Allocations may precede all producers; collective computation cannot.
    compute = source.index("chain_collective_0_acc =")
    # The admitted warp route delegates its ordered K loop to this exact
    # imported helper. Its actual call, not the import/declaration, must precede
    # collective computation; other schedules retain their inline GEMM.
    shared_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "helion._compiler.cute.prepared_warp_contraction"
        for alias in node.names
        if alias.name == "execute_prepared_warp_k"
    }
    issues = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            ast.unparse(node.func) == "cute.gemm"
            or isinstance(node.func, ast.Name)
            and node.func.id in shared_names
        )
    ]
    compute_line = source[:compute].count("\n") + 1
    assert any(node.lineno < compute_line for node in issues)
    if not loop:
        assert compute < source.index("chain_1_mma =")
        assert source.index("chain_1_mma =") < source.index("chain_collective_1_acc =")


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize(("rank", "axis"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_postdot_scans_use_graph_order_cpu(
    loop: bool, rank: int, axis: int, schedule: str
) -> None:
    with _cpu_codegen():
        kernel = _postdot_recurrence if loop else _postdot_chain
        bound = kernel._bind_isolated(
            _postdot_inputs("cpu", loop=loop, rank=rank, axis=axis)
        )
        source = bound.to_code(
            helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
        )
        _postdot_source(source, loop=loop)


def _prefix_reference(value: torch.Tensor, rank: int, axis: int) -> torch.Tensor:
    if rank == 1:
        return value + value[..., 0].cumsum(-1)[..., None]
    return value.cumsum(axis - 2)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize(("rank", "axis"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_postdot_scans_preserve_contraction_results(
    loop: bool, rank: int, axis: int, schedule: str
) -> None:
    args = _postdot_inputs(DEVICE, loop=loop, rank=rank, axis=axis)
    kernel = _postdot_recurrence if loop else _postdot_chain
    bound = kernel._bind_isolated(args)
    config = helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
    _postdot_source(bound.to_code(config), loop=loop)
    actual = bound.compile_config(config)(*args)
    a, b, c = args[:3]
    if loop:
        history, final = actual
        expected = c.clone()
        for step in range(3):
            product = a[:, step].float() @ b[:, step].float()
            expected = 0.5 * expected + _prefix_reference(product, rank, axis)
            torch.testing.assert_close(history[:, step], expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(final, expected, rtol=1e-4, atol=1e-5)
    else:
        prefix = _prefix_reference(a.float() @ b.float(), rank, axis).to(c.dtype)
        expected = _prefix_reference(prefix.float() @ c.float(), rank, axis)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def _check_source(source: str) -> None:
    tree = ast.parse(source)
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and "chain_loop_index" in node.target.id
    ]
    assert len(loops) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "alloc_smem"
        for node in ast.walk(loops[0])
    )
    assert "chain_collective_0" in source
    assert "chain_collective_1" in source
    assert "chain_collective_2" in source
    assert "shuffle_sync_down" in source
    assert "chain_0_mma" in source


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_scan_helpers_admitted_by_shared_loop_codegen(axis: int, schedule: str) -> None:
    with _cpu_codegen():
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 3, axis))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            loop = discover_chained_loop(graphs)
            assert loop is not None
            assert len(loop.region.scans) == 2
            assert len(loop.region.reductions) == 1
            assert any(isinstance(info, HelperFunctionGraphInfo) for info in graphs)
            assert _classify_chained_graph(graphs) is not None
        _check_source(
            bound.to_code(
                helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
            )
        )


def test_unowned_and_missing_scan_helpers_reject_loop() -> None:
    with _cpu_codegen():
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 3, 1))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            helper = next(
                info for info in graphs if isinstance(info, HelperFunctionGraphInfo)
            )
            assert (
                discover_chained_loop([info for info in graphs if info is not helper])
                is None
            )
            assert (
                discover_chained_loop(
                    [
                        *graphs,
                        replace(
                            helper, graph_id=max(info.graph_id for info in graphs) + 1
                        ),
                    ]
                )
                is None
            )


def test_nonadditive_combiner_is_not_claimed_by_shared_loop() -> None:
    with _cpu_codegen():
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 3, 1))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            helper = next(
                info for info in graphs if isinstance(info, HelperFunctionGraphInfo)
            )
            operation = next(
                node for node in helper.graph.nodes if node.op == "call_function"
            )
            operation.target = torch.ops.aten.mul.Tensor
            assert discover_chained_loop(graphs) is not None
            assert _classify_chained_graph(graphs) is None


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_scan_coefficient_loop_matches_reference(
    axis: int, steps: int, schedule: str
) -> None:
    args = _inputs(DEVICE, steps, axis)
    a, b, decay, delta, initial, _, _ = args
    original = initial.clone()
    bound = _prefix_coefficient_recurrence._bind_isolated(args)
    config = helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
    _check_source(bound.to_code(config))
    compiled = bound.compile_config(config)
    history, final = compiled(*args)
    expected = initial.clone()
    for step in range(steps):
        row_prefix = decay[:, step].cumsum(1)
        matrix_prefix = delta[:, step].cumsum(axis + 1)
        energy = (matrix_prefix * matrix_prefix).sum(2)
        left = (a[:, step].float() * torch.exp(matrix_prefix)).to(a.dtype)
        product = left.float() @ b[:, step].float()
        expected = expected * torch.exp(row_prefix[:, :, None]) + product / (
            1.0 + energy[:, :, None]
        )
        torch.testing.assert_close(history[:, step], expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    _, repeated = compiled(*args)
    torch.testing.assert_close(repeated, final, rtol=0, atol=0)
