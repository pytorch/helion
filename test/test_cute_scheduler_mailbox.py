from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _config
from ._cute_aux import _cpu_codegen
from ._cute_aux import _rank_two_aux
from ._cute_aux import _rank_two_code
from ._cute_aux import _rank_two_inputs
from .test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from .test_cute_grouped_gemm_split_sizes import _selected_config
import helion
from helion._compiler.device_function import DeviceFunction
from helion._compiler.program_id import _build_sched_pipeline_consumer_release_block
from helion._compiler.program_id import _build_sched_pipeline_consumer_wait_block
from helion._compiler.program_id import _SchedMailboxSnapshot
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    m, k = lhs.shape
    n = rhs.shape[1]
    out = torch.empty((m, n), dtype=lhs.dtype, device=lhs.device)
    for mi, ni in hl.tile([m, n]):
        acc = hl.zeros([mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.addmm(acc, lhs[mi, ki], rhs[ki, ni])
        out[mi, ni] = acc.to(out.dtype)
    return out


def _assert_release_policy(code: str, *, arrivals: int, leader_only: bool) -> None:
    tree = ast.parse(code)
    parent = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    releases = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "consumer_release"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tcgen05_sched_pipeline"
    ]
    assert len(releases) >= 6  # At least MMA, AB-load, epi; tile + terminal.
    for release in releases:
        ancestor = parent[release]
        has_lane_gate = False
        while ancestor in parent:
            if isinstance(ancestor, ast.If) and "lane_idx" in ast.unparse(
                ancestor.test
            ):
                has_lane_gate = True
            ancestor = parent[ancestor]
        assert has_lane_gate is leader_only, code
    assert (
        "tcgen05_sched_pipeline_consumer_group = "
        "cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, "
        f"cutlass.Int32({arrivals}))"
    ) in code


@pytest.mark.parametrize(
    "wait_mode,expected",
    [
        (
            "normal",
            [
                "pipe.consumer_release(state)",
                "state.advance()",
                "cute.arch.sync_warp()",
            ],
        ),
        (
            "warp_leader",
            [
                "if cute.arch.lane_idx() == cutlass.Int32(0):\n"
                "    pipe.consumer_release(state)",
                "state.advance()",
                "cute.arch.sync_warp()",
            ],
        ),
    ],
)
def test_scheduler_release_helper_matches_wait_policy(
    wait_mode: str, expected: list[str]
) -> None:
    function = SimpleNamespace(config={"tcgen05_sched_consumer_wait_mode": wait_mode})
    with patch.object(DeviceFunction, "current", return_value=function):
        nodes = _build_sched_pipeline_consumer_release_block(
            sched_pipeline="pipe", sched_consumer_state="state"
        )
        other = _build_sched_pipeline_consumer_release_block(
            sched_pipeline="pipe", sched_consumer_state="state"
        )
    assert [ast.unparse(node) for node in nodes] == expected
    assert all(first is not second for first, second in zip(nodes, other, strict=True))


def test_warp_leader_release_follows_snapshot_broadcasts() -> None:
    function = SimpleNamespace(
        config={"tcgen05_sched_consumer_wait_mode": "warp_leader"}
    )
    snapshot = _SchedMailboxSnapshot("snapshot_valid", {0: "snapshot_coord"})
    with patch.object(DeviceFunction, "current", return_value=function):
        nodes = [
            *_build_sched_pipeline_consumer_wait_block(
                sched_pipeline="pipe",
                sched_consumer_state="state",
                work_tile_smem="mailbox",
                valid_var="valid",
                snapshot=snapshot,
            ),
            *_build_sched_pipeline_consumer_release_block(
                sched_pipeline="pipe", sched_consumer_state="state"
            ),
        ]
    source = "\n".join(ast.unparse(node) for node in nodes)
    assert source.rindex("cute.arch.shuffle_sync") < source.index(
        "pipe.consumer_release"
    )
    assert source.index("pipe.consumer_release") < source.index("state.advance()")


@pytest.mark.parametrize("wait_mode", ["normal", "warp_leader"])
@pytest.mark.parametrize("load_mode", ["simt", "tma"])
@pytest.mark.parametrize("tile", [[64, 256, 64], [128, 128, 64]])
def test_scheduler_aux_consumers_use_mode_release_policy(
    wait_mode: str, load_mode: str, tile: list[int]
) -> None:
    code = _rank_two_code(
        _rank_two_inputs(),
        _config(
            block_sizes=tile,
            tcgen05_aux_load_mode=load_mode,
            tcgen05_sched_consumer_wait_mode=wait_mode,
        ),
    )
    # Both rank-two AUX transports stage through the C-input warp.
    arrivals_per_warp = 1 if wait_mode == "warp_leader" else 32
    _assert_release_policy(
        code,
        arrivals=7 * arrivals_per_warp,
        leader_only=wait_mode == "warp_leader",
    )
    # The aux-data pipeline is a distinct lifetime protocol; its per-warp
    # release/count must not be changed by the scheduler-mailbox fix.
    assert "tcgen05_aux_pipeline.consumer_release" in code


@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("clc", [False, True])
@pytest.mark.parametrize("inert_c_input", [0, 1])
@pytest.mark.parametrize("wait_mode", ["normal", "warp_leader"])
def test_scheduler_thread_count_preserves_cluster_routing(
    cluster_n: int, clc: bool, inert_c_input: int, wait_mode: str
) -> None:
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 1024), dtype=torch.bfloat16),
    )
    config = _config(
        block_sizes=[256, 256, 128],
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        tcgen05_warp_spec_c_input_warps=inert_c_input,
        tcgen05_aux_load_mode="simt",
        tcgen05_sched_consumer_wait_mode=wait_mode,
    )
    if clc:
        config.config["tcgen05_persistence_model"] = "clc_persistent"
    with _cpu_codegen():
        _plain_matmul.reset()
        code = _plain_matmul.bind(args).to_code(config)
    arrivals_per_warp = 1 if wait_mode == "warp_leader" else 32
    _assert_release_policy(
        code,
        arrivals=6 * arrivals_per_warp * (2 * cluster_n if clc else 1),
        leader_only=wait_mode == "warp_leader",
    )
    assert ("consumer_mask=cutlass.Int32(0)" in code) == clc
    assert "defer_sync=True" in code
    assert "tcgen05_c_input_warp_valid" not in code
    assert ("cute.arch.clc_response" in code) == clc


@pytest.mark.parametrize("wait_mode", ["normal", "warp_leader"])
def test_scheduler_staged_mailbox_uses_mode_release_policy(wait_mode: str) -> None:
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 1024), dtype=torch.bfloat16),
    )
    config = _config(
        block_sizes=[256, 256, 128],
        tcgen05_cluster_m=2,
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
        tcgen05_persistence_model="clc_persistent",
        tcgen05_sched_stage_count=2,
        tcgen05_sched_consumer_wait_mode=wait_mode,
    )
    with _cpu_codegen():
        _plain_matmul.reset()
        code = _plain_matmul.bind(args).to_code(config)
    arrivals_per_warp = 1 if wait_mode == "warp_leader" else 32
    _assert_release_policy(
        code,
        arrivals=6 * arrivals_per_warp * 2,
        leader_only=wait_mode == "warp_leader",
    )
    assert "tcgen05_sched_pipeline_consumer_state.index]" in code


@pytest.mark.parametrize("wait_mode", ["normal", "warp_leader"])
def test_scheduler_simt_aux_producer_uses_mode_release_policy(wait_mode: str) -> None:
    args = tuple(torch.empty((256, 256), dtype=torch.bfloat16) for _ in range(3))
    with _cpu_codegen():
        _rank_two_aux.reset()
        code = _rank_two_aux.bind(args).to_code(
            _config(
                block_sizes=[128, 128, 64],
                tcgen05_aux_load_mode="simt",
                tcgen05_sched_consumer_wait_mode=wait_mode,
            )
        )
    arrivals_per_warp = 1 if wait_mode == "warp_leader" else 32
    _assert_release_policy(
        code,
        arrivals=7 * arrivals_per_warp,
        leader_only=wait_mode == "warp_leader",
    )
    assert "tcgen05_c_input_warp_valid" in code
    assert "tcgen05_aux_pipeline.consumer_release" in code


def test_grouped_worklist_mailbox_releases_all_lanes() -> None:
    split_sizes = torch.tensor((0, 1, 127, 224, 256, 449, 0, 991), dtype=torch.int32)
    args = (
        torch.empty((2048, 128), dtype=torch.bfloat16),
        torch.empty((8, 224, 128), dtype=torch.bfloat16),
        split_sizes,
    )
    config = _selected_config(128)
    with _cpu_codegen():
        _device_split_sizes_kernel.reset()
        code = _device_split_sizes_kernel.bind(args).to_code(config)
    _assert_release_policy(code, arrivals=6 * 32, leader_only=False)
    assert "tcgen05_grouped_selected_sched" in code
    assert "worklist_metadata" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires TCgen05 GPU")
@pytest.mark.parametrize(
    "cluster_n,clc,stages",
    [(1, False, 1), (2, False, 1), (1, True, 1), (1, True, 2), (2, True, 1)],
    ids=["static-2x1", "static-2x2", "clc-2x1", "clc-2x1-stage2", "clc-2x2"],
)
def test_clustered_scheduler_multiwave_runtime(
    cluster_n: int, clc: bool, stages: int
) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires a TCgen05-capable GPU")
    # 512 logical tiles require multiple waves on SM100/SM103. CLC additionally
    # routes every lane's empty-barrier arrival to the cluster leader.
    config = _config(
        block_sizes=[256, 256, 128],
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
        tcgen05_sched_stage_count=stages,
    )
    if clc:
        config.config["tcgen05_persistence_model"] = "clc_persistent"
    for seed in range(5):
        torch.manual_seed(seed)
        lhs = torch.randn((8192, 256), dtype=torch.bfloat16, device=DEVICE) * 0.1
        rhs = torch.randn((256, 4096), dtype=torch.bfloat16, device=DEVICE) * 0.1
        before = (lhs.clone(), rhs.clone())
        expected = lhs.double() @ rhs.double()
        bound = _plain_matmul.bind((lhs, rhs))
        bound.set_config(config)
        first = bound(lhs, rhs)
        repeated = bound(lhs, rhs)
        assert torch.equal(repeated, first)
        torch.testing.assert_close(first.double(), expected, atol=0.002, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound(lhs, rhs)
        graph.replay()
        assert torch.equal(captured, first)
        for _ in range(3):
            assert torch.equal(bound(lhs, rhs), first)
            captured.fill_(float("nan"))
            graph.replay()
            assert torch.equal(captured, first)
        assert torch.equal(lhs, before[0])
        assert torch.equal(rhs, before[1])
