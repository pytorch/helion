"""Native half-input preflight/final-store agreement without GPU work."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test.test_cute_fp32_native_admission import _mixed_source_matmuls

import helion
from helion._compiler import device_ir
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute import cute_mma
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _multi_output_epilogue(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    mode: hl.constexpr,
):
    m, k = a.shape
    n = b.shape[1]
    out0 = torch.empty((m, n), dtype=torch.float32, device=a.device)
    out1 = torch.empty_like(out0)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            left = a[row, inner].to(torch.float32).to(a.dtype)
            right = b[inner, col].to(torch.float32).to(b.dtype)
            acc = torch.addmm(acc, left, right)
        value = residual[row, col] + acc
        if mode == "bool_mask":
            value = value * mask[row, col].to(torch.float32)
        elif mode == "sin":
            value = torch.sin(value)
        elif mode == "intermediate_cast":
            value = value.to(torch.bfloat16).to(torch.float32)
        out0[row, col] = value
        out1[row, col] = value * scale[col]
    return out0, out1


def _bind(mode, dtype):
    return cpu_bind(
        _multi_output_epilogue,
        (
            torch.empty((256, 128), dtype=dtype),
            torch.empty((128, 256), dtype=dtype),
            torch.empty((256, 256), dtype=torch.float32),
            torch.empty((256, 256), dtype=torch.bool),
            torch.empty((256,), dtype=torch.float32),
            mode,
        ),
    )


def _candidate(bound):
    graphs = bound.host_function.device_ir.graphs
    node = next(
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    with bound.env, bound.host_function:
        candidate = cute_mma.analyze_cute_mma_node(
            node, device_ir=bound.host_function.device_ir
        )
    assert candidate is not None
    return graphs, node, candidate


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("bool_mask", "sin"))
def test_unsupported_half_final_store_keeps_full_fallback_surface(mode, dtype):
    bound = _bind(mode, dtype)
    spec = bound.config_spec
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_half_source_store_eligible(
            graphs, node, candidate.operands
        )
    assert not spec.cute_tcgen05_search_enabled
    assert "tcgen05_ab_stages" not in spec._flat_fields()
    assert "cute_collective_mma" in spec._flat_fields()
    configs = [spec.default_config(), spec._base_default_config()]
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds
        generation = ConfigGeneration(spec)
        for config in seeds[:3]:
            flat, canonical = generation.canonicalize_flat(generation.flatten(config))
            flat2, canonical2 = generation.canonicalize_flat(flat)
            assert flat == flat2 and canonical == canonical2
            configs.append(canonical)
    for config in configs:
        source = bound.to_code(config)
        assert "tcgen05_ab_tma" not in source
        # A whole-root MMA-DAG owner may independently prove this source;
        # only the rejected ordinary native namespace is excluded here.
        assert (
            "tcgen05_mma(" in source
            or ".mma_bf16(" in source
            or ".mma_f16(" in source
            or ("indices_0" in source and "indices_1" in source)
        )
        # The FX input dtype is unchanged; a whole-root owner may read
        # packed half words directly without spelling a scalar cast type.
        assert candidate.operands.lhs.mma_dtype == dtype
        assert candidate.operands.rhs.mma_dtype == dtype


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("supported", "intermediate_cast"))
def test_supported_half_full_store_retains_native_lowering(mode, dtype):
    bound = _bind(mode, dtype)
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert cute_mma.tcgen05_half_source_store_eligible(
            graphs, node, candidate.operands
        )
    assert bound.config_spec.cute_tcgen05_search_enabled
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_output_dtypes(a: torch.Tensor, b: torch.Tensor):
    m, k = a.shape
    n = b.shape[1]
    half = torch.empty((m, n), dtype=a.dtype, device=a.device)
    wide = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        half[row, col] = acc.to(a.dtype)
        wide[row, col] = (acc * 0.3).to(torch.float32)
    return half, wide


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_mixed_output_dtype_releases_ordinary_claim_and_keeps_dag_incumbent(dtype):
    a = torch.empty((256, 128), dtype=dtype)
    b = torch.empty((128, 256), dtype=dtype)
    bound = cpu_bind(_mixed_output_dtypes, (a, b))
    graphs, node, candidate = _candidate(bound)
    with bound.env, bound.host_function:
        assert not cute_mma.tcgen05_half_source_store_eligible(
            graphs, node, candidate.operands
        )
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert bound.config_spec.cute_mma_dag is not None
    # An explicit existing DAG config remains available; output casts and
    # the two original output dtypes survive the independent whole-root plan.
    source = bound.to_code(
        helion.Config(
            block_sizes=[64, 64, 32],
            cute_mma_dag="warp",
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
        )
    )
    assert ".mma_bf16(" in source if dtype == torch.bfloat16 else ".mma_f16(" in source
    assert "tcgen05_ab_tma" not in source


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_rejected_half_peer_cannot_promote_a_supported_native_subset(dtype):
    a = torch.empty((256, 256), dtype=dtype)
    with patch.object(
        device_ir,
        "_finalize_cute_tcgen05_search_planning",
        wraps=device_ir._finalize_cute_tcgen05_search_planning,
    ) as finalize:
        bound = cpu_bind(_mixed_source_matmuls, (a, a.clone()))
    assert len(finalize.call_args.args[1]) == 2
    assert not finalize.call_args.args[2]
    assert not bound.config_spec.cute_tcgen05_search_enabled
    source = bound.to_code(bound.config_spec.default_config())
    assert "tcgen05_ab_tma" not in source


def test_half_rejection_keeps_planning_guard_publication():
    with patch.object(
        device_ir,
        "_finalize_cute_tcgen05_search_planning",
        wraps=device_ir._finalize_cute_tcgen05_search_planning,
    ) as finalize:
        bound = _bind("bool_mask", torch.bfloat16)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    results = finalize.call_args.args[1]
    assert len(results) == 1 and results[0].plan is not None
    assert results[0].guards_complete
    assert not finalize.call_args.args[2]
