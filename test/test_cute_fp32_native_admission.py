from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute import cute_mma
from helion._testing import patch_cute_mma_support
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    support = SimpleNamespace(
        universal=True,
        warp_f16bf16=True,
        warpgroup_f16bf16=False,
        tcgen05_f16bf16=True,
        tcgen05_f8=True,
        tcgen05_tf32=True,
        supported_impls=("universal", "warp", "tcgen05"),
    )
    with cpu_codegen_target(), patch_cute_mma_support(support), forbid_cuda_work():
        yield


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", dot_precision="tf32"
)
def _epilogue_matmul(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, mode: hl.constexpr
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        if mode == "identity":
            out[row, col] = acc
        elif mode == "typed_scale":
            out[row, col] = acc * hl.full([], 0.3, dtype=torch.float32)
        elif mode == "cast":
            out[row, col] = acc.to(torch.bfloat16).to(torch.float32)
        elif mode == "relu":
            out[row, col] = torch.relu(acc)
        elif mode == "trailing_bias":
            out[row, col] = acc + bias[col]
        elif mode == "leading_bias":
            out[row, col] = acc + bias[row].reshape([row, 1])
        elif mode == "same_anchor_square":
            out[row, col] = acc * acc
        else:
            out[row, col] = torch.sin(acc)
    return out


def _bind(
    mode: str, *, rhs_col_major: bool = False, lhs_col_major: bool = False
) -> BoundKernel:
    a = torch.empty((256, 256), dtype=torch.float32, device="cpu")
    b = torch.empty_like(a)
    if lhs_col_major:
        a = a.T
    if rhs_col_major:
        b = b.T
    bias = torch.empty((256,), dtype=torch.float32, device="cpu")
    return cpu_bind(_epilogue_matmul, (a, b, bias, mode))


def _candidate(bound: BoundKernel):
    assert bound.host_function is not None
    graphs = bound.host_function.device_ir.graphs
    with bound.env, bound.host_function:
        node = next(
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        candidate = cute_mma.analyze_cute_mma_node(
            node, device_ir=bound.host_function.device_ir
        )
    assert candidate is not None
    return graphs, node, candidate


@pytest.mark.parametrize(
    "mode",
    ["identity", "typed_scale", "cast", "relu", "trailing_bias", "same_anchor_square"],
)
@pytest.mark.parametrize("rhs_col_major", [False, True])
def test_supported_fp32_epilogues_keep_native_search(
    mode: str, rhs_col_major: bool
) -> None:
    bound = _bind(mode, rhs_col_major=rhs_col_major)
    assert bound.config_spec.cute_tcgen05_search_enabled
    fields = bound.config_spec._flat_fields()
    assert "tcgen05_ab_stages" in fields
    assert "cute_collective_mma" not in fields
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert cute_mma.tcgen05_fp32_source_eligible(
            graphs,
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=candidate.operands.rhs,
        )
        assert not cute_mma._tcgen05_fp32_lowering_blocked(
            SimpleNamespace(codegen_graphs=graphs),
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=candidate.operands.rhs,
            config={},
        )
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    ast.parse(source)
    assert "make_trivial_tiled_mma(cutlass.TFloat32" in source
    assert "tcgen05_ab_tma" in source
    if mode == "same_anchor_square":
        assert "tcgen05_dag_step" in source


@pytest.mark.parametrize("mode", ["unsupported_sin", "leading_bias"])
def test_unsupported_fp32_epilogues_retain_round_trip_safe_collective_choices(
    mode: str,
) -> None:
    bound = _bind(mode)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    fields = bound.config_spec._flat_fields()
    assert "tcgen05_ab_stages" not in fields
    assert "cute_collective_mma" in fields
    assert "num_threads" in fields
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_fp32_source_eligible(
            graphs,
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=candidate.operands.rhs,
        )
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds
        assert {seed.get("cute_collective_compute", "warp") for seed in seeds} >= {
            "warp",
            "tcgen05",
        }
        generation = ConfigGeneration(bound.config_spec)
        configs = []
        for compute in ("warp", "tcgen05"):
            seed = next(
                seed
                for seed in seeds
                if seed.get("cute_collective_compute", "warp") == compute
            )
            flat, normalized = generation.canonicalize_flat(generation.flatten(seed))
            flat2, normalized2 = generation.canonicalize_flat(flat)
            assert normalized.get("cute_collective_mma") is True
            assert normalized.get("cute_collective_compute", "warp") == compute
            assert flat == flat2 and normalized == normalized2
            configs.append(normalized)
    for config in configs:
        source = bound.to_code(config)
        ast.parse(source)
        assert "cute.gemm(" in source
        assert "dot_acc = dot_acc +" not in source


def test_column_major_fp32_lhs_does_not_get_a_native_only_surface() -> None:
    bound = _bind("identity", lhs_col_major=True)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert "cute_collective_mma" in bound.config_spec._flat_fields()
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_fp32_source_eligible(
            graphs,
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=candidate.operands.rhs,
        )


@pytest.mark.parametrize("operand", ["lhs", "rhs"])
def test_noncontiguous_matrix_layout_stays_rejected(operand: str) -> None:
    bound = _bind("identity")
    graphs, node, candidate = _candidate(bound)
    info = getattr(candidate.operands, operand)
    tensor = torch.empty((512, 512), dtype=torch.float32, device="cpu")[::2, ::2]
    rejected = replace(info, source_fake=tensor, logical_fake=tensor)
    operands = {
        "lhs_operand": candidate.operands.lhs,
        "rhs_operand": candidate.operands.rhs,
    }
    operands[f"{operand}_operand"] = rejected
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_fp32_source_eligible(graphs, node, **operands)
        assert cute_mma._tcgen05_fp32_lowering_blocked(
            SimpleNamespace(codegen_graphs=graphs), node, **operands, config={}
        )


@pytest.mark.parametrize(
    "field", ["rhs_rank3_grouped_nt", "rhs_segment_group", "rhs_packed_group"]
)
def test_grouped_source_protocols_remain_rejected(field: str) -> None:
    bound = _bind("identity")
    graphs, node, candidate = _candidate(bound)
    rhs = replace(
        candidate.operands.rhs,
        **{field: True if field == "rhs_rank3_grouped_nt" else object()},
    )
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_fp32_source_eligible(
            graphs,
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=rhs,
        )


@pytest.mark.parametrize("subtile", [2, True, 0])
def test_source_eligibility_does_not_relax_epilogue_subtile_config_guards(
    subtile: object,
) -> None:
    bound = _bind("identity")
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert cute_mma._tcgen05_fp32_lowering_blocked(
            SimpleNamespace(codegen_graphs=graphs),
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=candidate.operands.rhs,
            config={"epilogue_subtile": subtile},
        )


def test_source_eligibility_does_not_relax_grouped_mode_config_guard() -> None:
    bound = _bind("identity")
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert cute_mma._tcgen05_fp32_lowering_blocked(
            SimpleNamespace(codegen_graphs=graphs),
            node,
            lhs_operand=candidate.operands.lhs,
            rhs_operand=candidate.operands.rhs,
            config={"tcgen05_grouped_mode": "static"},
        )


@pytest.mark.parametrize("operand", ["lhs", "rhs"])
def test_leading_passthrough_source_protocols_remain_rejected(operand: str) -> None:
    bound = _bind("identity")
    graphs, node, candidate = _candidate(bound)
    info = getattr(candidate.operands, operand)
    rejected = replace(info, block_ids=(99, *info.block_ids))
    operands = {
        "lhs_operand": candidate.operands.lhs,
        "rhs_operand": candidate.operands.rhs,
    }
    operands[f"{operand}_operand"] = rejected
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_fp32_source_eligible(graphs, node, **operands)
        assert cute_mma._tcgen05_fp32_lowering_blocked(
            SimpleNamespace(codegen_graphs=graphs), node, **operands, config={}
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_search_does_not_use_the_fp32_source_gate(
    dtype: torch.dtype,
) -> None:
    a = torch.empty((256, 256), dtype=dtype, device="cpu")
    bias = torch.empty((256,), dtype=dtype, device="cpu")
    with patch.object(
        cute_mma,
        "tcgen05_fp32_source_eligible",
        side_effect=AssertionError("FP32 source gate must not run for half inputs"),
    ):
        bound = cpu_bind(_epilogue_matmul, (a, a.clone(), bias, "identity"))
    assert bound.config_spec.cute_tcgen05_search_enabled


def test_rejected_source_still_publishes_all_planning_guards() -> None:
    from helion._compiler import device_ir

    with patch.object(
        device_ir,
        "_finalize_cute_tcgen05_search_planning",
        wraps=device_ir._finalize_cute_tcgen05_search_planning,
    ) as finalize:
        bound = _bind("unsupported_sin")
    assert not bound.config_spec.cute_tcgen05_search_enabled
    results = finalize.call_args.args[1]
    assert len(results) == 1
    assert results[0].plan is not None
    assert results[0].guards_complete
    assert not finalize.call_args.args[2]


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", dot_precision="tf32"
)
def _mixed_source_matmuls(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    n = b.shape[1]
    first = torch.empty((m, n), dtype=torch.float32, device=a.device)
    second = torch.empty_like(first)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        first[row, col] = acc
        other = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            other = torch.addmm(other, a[row, inner], b[inner, col])
        second[row, col] = torch.sin(other)
    return first, second


def test_a_rejected_fp32_peer_cannot_promote_a_surviving_native_subset() -> None:
    from helion._compiler import device_ir

    a = torch.empty((256, 256), dtype=torch.float32, device="cpu")
    with (
        patch.object(
            cute_mma,
            "tcgen05_fp32_source_eligible",
            wraps=cute_mma.tcgen05_fp32_source_eligible,
        ) as eligibility,
        patch.object(
            device_ir,
            "_finalize_cute_tcgen05_search_planning",
            wraps=device_ir._finalize_cute_tcgen05_search_planning,
        ) as finalize,
    ):
        bound = cpu_bind(_mixed_source_matmuls, (a, a.clone()))
    assert eligibility.call_count == 2
    assert len(finalize.call_args.args[1]) == 2
    assert not finalize.call_args.args[2]
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert "cute_collective_mma" in bound.config_spec._flat_fields()
