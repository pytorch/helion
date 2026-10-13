from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute import chained_loop_workspace
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _paired_carries(
    left, right, operand, steps: int, swap: hl.constexpr, remember_old: hl.constexpr
):
    steps = hl.specialize(steps)
    m, n = left.shape
    n = hl.specialize(n)
    history = torch.empty((max(steps, 1), m, n), device=left.device)
    final_left = torch.empty_like(left)
    final_right = torch.empty_like(right)
    for rows, cols in hl.tile([m, n], block_size=[16, 16]):
        a, b = left[rows, cols], right[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = cols
            if remember_old:
                history[step.id, rows, cols] = a + b
            next_a = hl.dot(a.to(operand.dtype), operand[step.id, kk, cols])
            next_b = hl.dot(b.to(operand.dtype), operand[step.id, kk, cols])
            if swap:
                a, b = next_b, next_a
            else:
                a, b = next_a, next_b
            if not remember_old:
                history[step.id, rows, cols] = a + b
        final_left[rows, cols] = a
        final_right[rows, cols] = b
    return history, final_left, final_right


def _args(device, steps=3, swap=False, remember_old=False):
    torch.manual_seed(953)
    return (
        torch.randn((29, 16), device=device) * 0.125,
        torch.randn((29, 16), device=device) * 0.125,
        torch.randn((max(steps, 1), 16, 16), device=device, dtype=torch.bfloat16)
        * 0.125,
        steps,
        swap,
        remember_old,
    )


def _config(warp_rows=0, layout="xor"):
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_warp_mma_rows=warp_rows,
        cute_chained_scratch_layout=layout,
    )


def _allocations(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.alloc_smem"
    ]


def _shared_bytes(source):
    element_bytes = {
        "cutlass.BFloat16": 2,
        "cutlass.Float16": 2,
        "cutlass.Float32": 4,
        "cutlass.Int32": 4,
        "cutlass.Int64": 8,
    }
    return sum(
        (
            ast.literal_eval(call.args[1]) * element_bytes[ast.unparse(call.args[0])]
            + 127
        )
        // 128
        * 128
        for call in _allocations(source)
    )


@pytest.mark.parametrize("layout", ["row_major", "xor"])
@pytest.mark.parametrize("swap", [False, True])
def test_loop_carry_storage_is_hoisted_and_writeback_is_view_exact_cpu(layout, swap):
    args = _args("cpu", swap=swap)
    with _cpu_codegen():
        with patch.object(
            chained_loop_workspace, "plan_loop_workspace", return_value=None
        ):
            old = _paired_carries._bind_isolated(args).to_code(_config(layout=layout))
        with patch.object(
            chained_loop_workspace,
            "plan_loop_workspace",
            side_effect=AssertionError("fitting storage must not need a packed plan"),
        ):
            assert old == _paired_carries._bind_isolated(args).to_code(
                _config(layout=layout)
            )
            with patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=_shared_bytes(old),
            ):
                assert old == _paired_carries._bind_isolated(args).to_code(
                    _config(layout=layout)
                )
        with patch.object(
            CuteTcgen05Config,
            "per_cta_smem_capacity_bytes",
            return_value=_shared_bytes(old) - 128,
        ):
            packed = _paired_carries._bind_isolated(args).to_code(
                _config(layout=layout)
            )
    assert len(_allocations(old)) - len(_allocations(packed)) == 2
    assert "chain_loop_carry_0 = cute.make_tensor(chain_c_workspace + 0," in packed
    assert "chain_loop_carry_1 = cute.make_tensor(chain_c_workspace + 256," in packed
    assert ("_next = cute.make_rmem_tensor" in packed) is swap
    assert "_next = cute.make_rmem_tensor" in old
    # Real writeback keeps both joins; an already published carry needs one.
    assert packed.count("cute.arch.sync_threads()") == (
        old.count("cute.arch.sync_threads()") - int(not swap)
    )
    tree = ast.parse(packed)
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "chain_loop_index"
    )
    assert not _allocations(ast.unparse(loop))
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config,
            "per_cta_smem_capacity_bytes",
            return_value=_shared_bytes(packed) - 128,
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        _paired_carries._bind_isolated(args).to_code(_config(layout=layout))


def test_epilogue_old_carry_reads_keep_separate_storage_cpu():
    args = _args("cpu", remember_old=True)
    with _cpu_codegen():
        packed = _paired_carries._bind_isolated(args).to_code(_config())
    assert "chain_loop_carry_0 = cute.make_tensor(cute.arch.alloc_smem" in packed
    assert "chain_loop_carry_1 = cute.make_tensor(cute.arch.alloc_smem" in packed
    assert "_next = cute.make_rmem_tensor" in packed
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config,
            "per_cta_smem_capacity_bytes",
            return_value=_shared_bytes(packed) - 128,
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        _paired_carries._bind_isolated(args).to_code(_config())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cached_carries(left, right, operand):
    steps = operand.size(0)
    m, n = left.shape
    history = torch.empty((steps, m, n), device=left.device)
    final_left = torch.empty_like(left)
    final_right = torch.empty_like(right)
    for rows, cols in hl.tile([m, n], block_size=[16, 16]):
        a, b = left[rows, cols], right[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = cols
            rhs = torch.exp(
                hl.cumsum(operand[step.id, kk, cols].float(), dim=0) * 0.125
            ).to(operand.dtype)
            a = hl.dot(a.to(operand.dtype), rhs)
            b = hl.dot(b.to(operand.dtype), rhs)
            history[step.id, rows, cols] = a + b
        final_left[rows, cols] = a
        final_right[rows, cols] = b
    return history, final_left, final_right


@pytest.mark.parametrize("budget", [4096, 16384])
def test_actual_cache_and_collectives_decide_capacity_not_requested_budget_cpu(budget):
    args = _args("cpu")[:3]
    config = helion.Config.from_dict(
        {
            **_config().config,
            "cute_chained_pointwise_cache_bytes": budget,
            "cute_chained_scan_schedule": "warp",
        }
    )
    with _cpu_codegen():
        plain = _cached_carries._bind_isolated(args).to_code(_config())
        with patch.object(
            chained_loop_workspace,
            "plan_loop_workspace",
            side_effect=AssertionError("fitting storage must not need a packed plan"),
        ):
            separate = _cached_carries._bind_isolated(args).to_code(config)
            actual_cache_bytes = _shared_bytes(separate) - _shared_bytes(plain)
            assert 0 < actual_cache_bytes < budget
            assert "chain_collective_0" in separate
            assert "chain_pointwise_cache_0" in separate
            with patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=_shared_bytes(separate),
            ):
                assert separate == _cached_carries._bind_isolated(args).to_code(config)
        planner = chained_loop_workspace.plan_loop_workspace
        seen = []

        def plan_with_cache(plan, shapes):
            assert plan.pointwise_cache is not None
            assert plan.pointwise_cache.shared_bytes == actual_cache_bytes
            seen.append(plan)
            return planner(plan, shapes)

        with (
            patch.object(
                chained_loop_workspace, "plan_loop_workspace", plan_with_cache
            ),
            patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=_shared_bytes(plain),
            ),
        ):
            packed = _cached_carries._bind_isolated(args).to_code(config)
        assert seen
        assert "chain_loop_carry_0 = cute.make_tensor(chain_c_workspace" in packed
        assert _shared_bytes(packed) <= _shared_bytes(plain) < _shared_bytes(separate)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "steps,swap,remember_old",
    [
        (0, False, False),
        (1, False, False),
        (3, False, False),
        (3, True, False),
        (3, True, True),
    ],
)
@pytest.mark.parametrize("warp_rows", [0, 16])
def test_loop_workspace_gpu_preserves_aliases_cross_carries_and_zero_trip(
    steps, swap, remember_old, warp_rows
):
    args = _args(DEVICE, steps, swap, remember_old)
    saved = tuple(arg.clone() for arg in args[:3])
    config = _config(warp_rows)
    with patch.object(chained_loop_workspace, "plan_loop_workspace", return_value=None):
        bound = _paired_carries._bind_isolated(args)
        original_source = bound.to_code(config)
        original = bound.compile_config(config)
    with patch.object(
        CuteTcgen05Config,
        "per_cta_smem_capacity_bytes",
        return_value=_shared_bytes(original_source) - (0 if remember_old else 128),
    ):
        bound = _paired_carries._bind_isolated(args)
        packed_source = bound.to_code(config)
        assert (
            "chain_loop_carry_0 = cute.make_tensor(chain_c_workspace" in packed_source
        ) is not remember_old
        packed = bound.compile_config(config)
    expected, actual = original(*args), packed(*args)
    if steps:
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
    a, b = saved[:2]
    for step in range(steps):
        before = a + b
        next_a = torch.mm(a.bfloat16(), saved[2][step], out_dtype=torch.float32)
        next_b = torch.mm(b.bfloat16(), saved[2][step], out_dtype=torch.float32)
        a, b = (next_b, next_a) if swap else (next_a, next_b)
        torch.testing.assert_close(
            actual[0][step], before if remember_old else a + b, rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(actual[1:], (a, b), rtol=1e-4, atol=1e-5)
    repeated = packed(*args)
    if steps:
        torch.testing.assert_close(repeated[0], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(repeated[1:], actual[1:], rtol=0, atol=0)
    torch.testing.assert_close(args[:3], saved, rtol=0, atol=0)
