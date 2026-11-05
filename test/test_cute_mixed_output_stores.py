"""Mixed-width native output stores keep FP32 accumulator coordinates."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def mixed_outputs(a, b, bias, narrow, wide):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            left = a[row, inner].to(torch.float32).to(a.dtype)
            right = b[inner, col].to(torch.float32).to(b.dtype)
            acc = torch.addmm(acc, left, right)
        value = acc + bias[col].to(torch.float32)
        narrow[row, col] = value.to(a.dtype)
        wide[row, col] = value.to(torch.float32)


def _arguments(dtype, m=256, n=128, k=128):
    a = torch.empty((m, k), dtype=dtype)
    b = torch.empty((n, k), dtype=dtype).T
    bias = torch.empty(n, dtype=dtype)
    narrow = torch.empty((m, n), dtype=dtype)
    wide = torch.empty((m, n), dtype=torch.float32)
    return a, b, bias, narrow, wide


def _analysis(bound):
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


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_disjoint_mixed_output_source_has_a_complete_native_store_proof(dtype):
    bound = cpu_bind(mixed_outputs, _arguments(dtype))
    graphs, node, candidate = _analysis(bound)
    with bound.env, bound.host_function:
        plan = cute_mma.tcgen05_mixed_precision_store_plan(
            graphs, node, candidate.operands
        )
        assert plan is not None
        assert plan.dtypes == (dtype, torch.float32)
        assert cute_mma.tcgen05_half_source_store_eligible(
            graphs, node, candidate.operands
        )
    assert bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_native_codegen_uses_common_fp32_coordinates_and_independent_typed_stores(
    dtype,
):
    bound = cpu_bind(mixed_outputs, _arguments(dtype))
    captured = []
    original = CuteDeviceFunctionState.register_tcgen05_store_value

    def record(self, name, value):
        captured.append(value)
        return original(self, name, value)

    with patch.object(CuteDeviceFunctionState, "register_tcgen05_store_value", record):
        source = bound.to_code(
            helion.Config(block_sizes=[128, 64, 64], pid_type="flat")
        )
    ast.parse(source)
    assert len(captured) == 1
    value = captured[0]
    assert value.mixed_precision_stores is not None
    assert value.epi_elem_dtype_str == "cutlass.Float32"
    assert (
        value.explicit_epi_tile_m,
        value.explicit_epi_tile_n,
        value.explicit_d_store_box_n,
    ) == (128, 32, 32)
    assert "tcgen05_ab_tma" in source
    assert "tcgen05_fp32_tmem_desc" in source
    assert "'c_dtype': cutlass.Float32" in source
    assert (
        f"'c_dtype': cutlass.{'Float16' if dtype is torch.float16 else 'BFloat16'}"
        in source
    )
    assert "tcgen05_tma_store_atom" in source and "tcgen05_tma_store_atom_1" in source


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("n", (128, 136, 8008))
def test_original_masked_n_and_mn_major_b_keep_mixed_output_geometry(dtype, n):
    bound = cpu_bind(mixed_outputs, _arguments(dtype, m=128, n=n))
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source
    assert "tcgen05_fp32_tmem_desc" in source
    assert "cutlass.Float32" in source
    if n % 64:
        # Both stores retain the same original N bound, not a padded output.
        assert str(n) in source and "tcgen05_full_tile" in source


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("tile_n", (16, 32, 64))
def test_existing_explicit_subtile_controls_keep_one_common_coordinate_tile(
    dtype, tile_n
):
    bound = cpu_bind(mixed_outputs, _arguments(dtype))
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 64, 64],
            pid_type="flat",
            tcgen05_layout_strategy="explicit_epi_tile",
            tcgen05_layout_overrides_epi_tile_m=128,
            tcgen05_layout_overrides_epi_tile_n=tile_n,
            tcgen05_layout_overrides_d_store_box_n=tile_n,
        )
    )
    assert "tcgen05_fp32_tmem_desc" in source
    ast.parse(source)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("alias", ("input", "output", "bias"))
def test_mixed_output_proof_refuses_storage_aliases(dtype, alias):
    a, b, bias, narrow, wide = _arguments(dtype, m=128)
    if alias == "input":
        narrow = a
    elif alias == "output":
        narrow = wide.view(dtype).as_strided((128, 128), (256, 1))
    else:
        bias = narrow[0]
    bound = cpu_bind(mixed_outputs, (a, b, bias, narrow, wide))
    graphs, node, candidate = _analysis(bound)
    with bound.env, bound.host_function:
        assert (
            cute_mma.tcgen05_mixed_precision_store_plan(
                graphs, node, candidate.operands
            )
            is None
        )


@pytest.mark.parametrize("bad_output", ("column_major", "float64", "wrong_extent"))
def test_unknown_or_incompatible_store_geometry_keeps_the_old_loud_backstop(bad_output):
    a, b, bias, narrow, wide = _arguments(torch.bfloat16)
    if bad_output == "column_major":
        wide = torch.empty((128, 256), dtype=torch.float32).T
    elif bad_output == "float64":
        wide = wide.to(torch.float64)
    else:
        wide = torch.empty((257, 128), dtype=torch.float32)
    bound = cpu_bind(mixed_outputs, (a, b, bias, narrow, wide))
    graphs, node, candidate = _analysis(bound)
    with bound.env, bound.host_function:
        assert (
            cute_mma.tcgen05_mixed_precision_store_plan(
                graphs, node, candidate.operands
            )
            is None
        )


def test_external_outputs_require_dispatch_backed_disjointness_not_fake_storage_inequality():
    bound = cpu_bind(mixed_outputs, _arguments(torch.bfloat16))
    graphs, node, candidate = _analysis(bound)
    with (
        bound.env,
        bound.host_function,
        patch.object(bound.env, "bound_runtime_input_specialization_results", {}),
    ):
        assert (
            cute_mma.tcgen05_mixed_precision_store_plan(
                graphs, node, candidate.operands
            )
            is None
        )


def test_default_and_base_reference_configs_generate_actual_mixed_store_source():
    bound = cpu_bind(mixed_outputs, _arguments(torch.bfloat16))
    for config in (
        bound.config_spec.default_config(),
        bound.config_spec._base_default_config(),
    ):
        source = bound.to_code(config)
        if config.get("cute_mma_dag", "off") == "tcgen05":
            assert ".prims.tcgen05_mma(" in source
        elif config.get("cute_mma_dag", "off") == "warp":
            assert ".mma_bf16(" in source
        else:
            # The ordinary owner must use the common unrounded T2R plan.
            assert "tcgen05_ab_tma" in source
            assert "tcgen05_fp32_tmem_desc" in source


def test_source_plan_matches_original_store_identity_and_refuses_an_unknown_node():
    bound = cpu_bind(mixed_outputs, _arguments(torch.bfloat16))
    graphs, node, candidate = _analysis(bound)
    with bound.env, bound.host_function:
        plan = cute_mma.tcgen05_mixed_precision_store_plan(
            graphs, node, candidate.operands
        )
    assert plan is not None
    assert [plan.dtype_for(store) for store in plan.stores] == list(plan.dtypes)
    assert plan.dtype_for(node) is None


def test_constructor_certificate_adds_seeds_without_publishing_or_rereading_bound_facts():
    from helion._compiler.autotuner_heuristics.cute_mixed_output import (
        CuteMixedOutputStoreHeuristic,
    )

    bound = cpu_bind(mixed_outputs, _arguments(torch.bfloat16, m=2048, n=8008, k=2560))
    state = bound.config_spec._cute_tcgen05_config
    assert state.mixed_output_store_plan is not None
    assert "cute_mixed_output_store" in bound.config_spec.autotuner_heuristics
    assert bound.config_spec.compiler_seed_configs
    before = dict(bound.env.bound_runtime_input_specialization_results)
    with bound.env, bound.host_function:
        seeds = CuteMixedOutputStoreHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    assert bound.env.bound_runtime_input_specialization_results == before
    # A seed certificate is not a substitute for the strict late alias proof.
    graphs, node, candidate = _analysis(bound)
    with (
        bound.env,
        bound.host_function,
        patch.object(bound.env, "bound_runtime_input_specialization_results", {}),
    ):
        assert (
            cute_mma.tcgen05_mixed_precision_store_plan(
                graphs, node, candidate.operands
            )
            is None
        )


@pytest.mark.parametrize("capacity", (0, 32 * 1024, 64 * 1024, 232448))
def test_mixed_seed_budget_accounts_for_each_original_typed_output(capacity):
    from helion._compiler.autotuner_heuristics.cute_mixed_output import (
        CuteMixedOutputStoreHeuristic,
    )
    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    bound = cpu_bind(mixed_outputs, _arguments(torch.bfloat16))
    plan = bound.config_spec._cute_tcgen05_config.mixed_output_store_plan
    assert plan is not None
    with (
        bound.env,
        bound.host_function,
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
    ):
        seeds = CuteMixedOutputStoreHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    if capacity == 0:
        assert not seeds
    for config in seeds:
        bm, bn, bk = config["block_sizes"]
        required = plan.seed_smem_upper_bound(
            bm=bm,
            bn=bn,
            bk=bk,
            input_dtype_bytes=2,
            ab_stages=config["tcgen05_ab_stages"],
            c_stages=2,
        )
        assert required is not None and required <= capacity
        assert config["tcgen05_aux_load_placement"] == "post_acc_wait"
        assert config["tcgen05_cluster_m"] == 1


def test_internally_overlapping_outputs_cannot_enter_the_mixed_store_contract():
    a, b, bias, narrow, wide = _arguments(torch.bfloat16)
    wide = wide.as_strided(wide.shape, (1, 1))
    bound = cpu_bind(mixed_outputs, (a, b, bias, narrow, wide))
    graphs = bound.host_function.device_ir.graphs
    node = next(
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    lhs, rhs = node.args[1:3]
    with bound.env, bound.host_function:
        operands = cute_mma._analyze_mma_operands(lhs, rhs, bound.env)
        assert operands is not None
        assert (
            cute_mma.tcgen05_mixed_precision_store_plan(graphs, node, operands) is None
        )
