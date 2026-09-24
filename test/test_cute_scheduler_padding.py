from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy
import torch

from .test_cute_batched_aux_tma import _cpu_codegen
from .test_cute_scheduler_mailbox import _plain_matmul
import helion
from helion import exc
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.program_id import PIDInfo
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_product(
    lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    batch, m, k = lhs.shape
    n = rhs.shape[2]
    for bi, mi, ni in hl.tile([batch, m, n], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        out[bi, mi, ni] = acc.to(out.dtype)
    return out


def _config(order: tuple[int, ...], swizzle: int, mailbox: bool) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [64, 64, 64],
            "loop_orders": [list(order)],
            "pid_type": "persistent_interleaved",
            "tcgen05_ab_stages": 2,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_l2_swizzle_size": swizzle,
            "tcgen05_strategy": "role_local_with_scheduler"
            if mailbox
            else "role_local_monolithic",
            "tcgen05_warp_spec_scheduler_warps": int(mailbox),
            "tcgen05_warp_spec_c_input_warps": 0,
        }
    )


def _predicate_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ".tile_idx[1]" in ast.unparse(node.test)
    )


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("swizzle", [1, 2, 4, 8])
@pytest.mark.parametrize("mailbox", [False, True])
def test_padding_guards_actual_role_and_mailbox_codegen(
    order: tuple[int, ...], swizzle: int, mailbox: bool
) -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((7, 128, 64), (7, 64, 320), (7, 128, 320))
    )
    with _cpu_codegen():
        bound = _batched_product._bind_isolated(args)
        code = bound.to_code(_config(order, swizzle, mailbox))
        with patch.object(
            Tcgen05PersistentProgramIDs,
            "_tcgen05_work_tile_padding_predicate",
            return_value=None,
        ):
            unguarded = bound.to_code(_config(order, swizzle, mailbox))
    logical_counts = [7, 2, 5]
    needs_guard = logical_counts[order[1]] % swizzle != 0
    if not needs_guard:
        assert code == unguarded
        return
    tree = ast.parse(code)
    guards = [node for node in ast.walk(tree) if _predicate_guard(node)]
    assert len(guards) == (1 if mailbox else 3), code
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for guard in guards:
        assert isinstance(guard, ast.If)
        body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
        assert "advance_to_next_work" not in body
        if mailbox:
            assert ".producer_commit(" in body
        else:
            assert "virtual_pid =" in body
        loop = parents[guard]
        assert isinstance(loop, ast.While)
        assert ast.unparse(loop.test).endswith(".is_valid_tile")
        assert "advance_to_next_work" in ast.unparse(loop.body[-2])
        assert "get_current_work" in ast.unparse(loop.body[-1])
    if mailbox:
        # One terminating publish remains outside the producer loop, never
        # an invalid publication for a padding slot in the middle of work.
        assert (
            code.count("tcgen05_work_tile_smem[cutlass.Int32(3)] = cutlass.Int32(0)")
            == 1
        )


@pytest.mark.parametrize("swizzle", [1, 2, 4, 8])
@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("logical_n", [1, 3, 7, 8, 9, 16])
def test_padding_predicate_preserves_partial_cluster_peers(
    swizzle: int, cluster_n: int, logical_n: int
) -> None:
    scheduler = object.__new__(Tcgen05PersistentProgramIDs)
    scheduler.pid_info = [
        PIDInfo("pid0", "1", sympy.Integer(3), 0),
        PIDInfo("pid1", "1", sympy.Integer(logical_n), 1),
    ]
    with (
        patch.object(
            Tcgen05PersistentProgramIDs,
            "_tcgen05_l2_swizzle_size",
            return_value=swizzle,
        ),
        patch.object(
            Tcgen05PersistentProgramIDs, "_tcgen05_cluster_n", return_value=cluster_n
        ),
        patch.object(
            Tcgen05PersistentProgramIDs,
            "_tcgen05_scheduler_tile_dims_expr",
            return_value=["3", str(logical_n), "1"],
        ),
    ):
        predicate = scheduler._tcgen05_work_tile_padding_predicate("work")
    logical_clusters = (logical_n + cluster_n - 1) // cluster_n
    padded_clusters = (logical_clusters + swizzle - 1) // swizzle * swizzle
    assert (predicate is None) == (logical_clusters % swizzle == 0)
    for cluster in range(padded_clusters):
        admitted = []
        for peer in range(cluster_n):
            coord = cluster * cluster_n + peer
            allowed = predicate is None or eval(
                predicate,
                {"__builtins__": {}},
                {
                    "work": SimpleNamespace(tile_idx=(0, coord, 0)),
                    "cutlass": SimpleNamespace(Int32=int),
                },
            )
            admitted.append(allowed)
            assert allowed == (cluster < logical_clusters)
        assert len(set(admitted)) == 1


def test_shared_scheduler_seeks_padding_without_early_sentinel() -> None:
    scheduler = object.__new__(Tcgen05PersistentProgramIDs)
    with patch.object(
        Tcgen05PersistentProgramIDs,
        "_tcgen05_work_tile_padding_predicate",
        return_value="work.tile_idx[1] < 3",
    ):
        statements = scheduler._tcgen05_skip_padding_work("scheduler", "work")
    tree = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))

    class WorkScheduler:
        def __init__(self) -> None:
            self.index = 0
            self.coords = [3, 3, 1, 3]

        def get_current_work(self) -> SimpleNamespace:
            valid = self.index < len(self.coords)
            return SimpleNamespace(
                is_valid_tile=valid,
                tile_idx=(0, self.coords[self.index] if valid else 0, 0),
            )

        def advance_to_next_work(self) -> None:
            self.index += 1

    state = WorkScheduler()
    context = {"scheduler": state, "work": state.get_current_work()}
    exec(compile(tree, "<generated-padding-seek>", "exec"), context)
    assert state.index == 2
    assert context["work"].is_valid_tile
    state.advance_to_next_work()
    context["work"] = state.get_current_work()
    exec(compile(tree, "<generated-padding-seek>", "exec"), context)
    assert state.index == 4
    assert not context["work"].is_valid_tile


def test_dynamic_logical_count_keeps_runtime_guard() -> None:
    scheduler = object.__new__(Tcgen05PersistentProgramIDs)
    scheduler.pid_info = [
        PIDInfo("pid0", "1", sympy.Integer(3), 0),
        PIDInfo("pid1", "1", "logical_n", 1),
    ]
    with (
        patch.object(
            Tcgen05PersistentProgramIDs, "_tcgen05_l2_swizzle_size", return_value=4
        ),
        patch.object(Tcgen05PersistentProgramIDs, "_tcgen05_cluster_n", return_value=1),
        patch.object(
            Tcgen05PersistentProgramIDs,
            "_tcgen05_scheduler_tile_dims_expr",
            return_value=["3", "logical_n", "1"],
        ),
    ):
        predicate = scheduler._tcgen05_work_tile_padding_predicate("work")
    assert predicate == "work.tile_idx[1] < (logical_n)"


@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("swizzle", [2, 4])
def test_clustered_static_scheduler_guards_logical_clusters_codegen(
    cluster_m: int, cluster_n: int, swizzle: int
) -> None:
    args = (
        torch.empty((768, 64), dtype=torch.bfloat16),
        torch.empty((64, 1280), dtype=torch.bfloat16),
    )
    config = helion.Config.from_dict(
        _config((0, 1), swizzle, True).config
        | {
            "block_sizes": [128 * cluster_m, 128, 64],
            "tcgen05_cluster_m": cluster_m,
            "tcgen05_cluster_n": cluster_n,
        }
    )
    with _cpu_codegen():
        if cluster_m == 1 and cluster_n == 2:
            with pytest.raises(exc.InvalidConfig, match="cluster_n=2 requires"):
                _plain_matmul._bind_isolated(args).to_code(config)
            return
        code = _plain_matmul._bind_isolated(args).to_code(config)
    guards = [node for node in ast.walk(ast.parse(code)) if _predicate_guard(node)]
    if (10 // cluster_n) % swizzle == 0:
        assert not guards
    else:
        assert len(guards) == 1, code
        assert isinstance(guards[0], ast.If)
        assert (
            f"cutlass.Int32({cluster_n})" in ast.unparse(guards[0].test)
            or cluster_n == 1
        )


def test_batched_plain_cold_search_does_not_expose_clc() -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((7, 128, 128), (7, 128, 4096), (7, 128, 4096))
    )
    with _cpu_codegen():
        bound = _batched_product._bind_isolated(args)
        config = bound.config_spec._cute_tcgen05_config
        assert config.matmul_has_leading_passthrough
        assert not config._clc_persistence_search_enabled()


@pytest.mark.parametrize("swizzle", [2, 4])
def test_clc_source_is_outside_static_padding_change(swizzle: int) -> None:
    args = (
        torch.empty((1024, 128), dtype=torch.bfloat16),
        torch.empty((128, 1280), dtype=torch.bfloat16),
    )
    config = helion.Config.from_dict(
        _config((0, 1), swizzle, True).config
        | {
            "block_sizes": [256, 128, 64],
            "tcgen05_cluster_m": 2,
            "tcgen05_persistence_model": "clc_persistent",
        }
    )
    with _cpu_codegen():
        bound = _plain_matmul._bind_isolated(args)
        code = bound.to_code(config)
        with patch.object(
            Tcgen05PersistentProgramIDs,
            "_tcgen05_work_tile_padding_predicate",
            return_value=None,
        ):
            old = bound.to_code(config)
    assert "cute.arch.clc_response" in code
    assert code == old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mailbox", [False, True])
@pytest.mark.parametrize("swizzle", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_padding_runtime_batched_full_history(
    mailbox: bool, swizzle: int, dtype: torch.dtype
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 BF16/FP16 support required")
    generator = torch.Generator(device=DEVICE).manual_seed(9023)
    lhs = (
        torch.randn((7, 128, 64), generator=generator, device=DEVICE, dtype=dtype)
        * 0.125
    )
    rhs = (
        torch.randn((7, 64, 4096), generator=generator, device=DEVICE, dtype=dtype)
        * 0.125
    )
    original = lhs.clone(), rhs.clone()
    # 896 logical tiles force multiple waves and repeated AB/ACC/C phase wraps.
    out = torch.empty((7, 128, 4096), device=DEVICE, dtype=dtype)
    bound = _batched_product._bind_isolated((lhs, rhs, out))
    compiled = bound.compile_config(_config((1, 0, 2), swizzle, mailbox))
    expected = torch.bmm(lhs.double(), rhs.double()).to(out.dtype)
    for _ in range(3):
        out.fill_(float("nan"))
        result = compiled(lhs, rhs, out)
        torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(lhs, rhs, out)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    assert torch.equal(lhs, original[0])
    assert torch.equal(rhs, original[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("swizzle", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_padding_runtime_cluster_peers(
    cluster_n: int, swizzle: int, dtype: torch.dtype
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 BF16/FP16 support required")
    generator = torch.Generator(device=DEVICE).manual_seed(29023)
    lhs = (
        torch.randn((768, 64), generator=generator, device=DEVICE, dtype=dtype) * 0.125
    )
    rhs = (
        torch.randn((64, 1280), generator=generator, device=DEVICE, dtype=dtype) * 0.125
    )
    saved = lhs.clone(), rhs.clone()
    config = helion.Config.from_dict(
        _config((0, 1), swizzle, True).config
        | {
            "block_sizes": [256, 128, 64],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": cluster_n,
        }
    )
    compiled = _plain_matmul._bind_isolated((lhs, rhs)).compile_config(config)
    expected = (lhs.double() @ rhs.double()).to(lhs.dtype)
    first = compiled(lhs, rhs)
    second = compiled(lhs, rhs)
    torch.testing.assert_close(first, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(second, expected, atol=0.02, rtol=0.02)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, second)
    assert torch.equal(lhs, saved[0])
    assert torch.equal(rhs, saved[1])
