"""Seed-only complete shared-GEMM coverage; no CUDA compilation or timing."""

from __future__ import annotations

from dataclasses import replace
import json
import random
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import shared_gemm_seeds as seeds
from helion._compiler.cute.shared_gemm_seeds import shared_gemm_seed_configs
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.language import memory_ops

COUNT = hl.constexpr(2)


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="full", fast_math=False
)
def multi(a, b, residual, extra, out0, out1, out2, out3, count: hl.constexpr = COUNT):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        pre = acc + residual[col].to(torch.float32)
        mapped = extra[
            row.index.reshape([row.index.size(0), 1]) // 512,
            row.index.reshape([row.index.size(0), 1]) % 512,
            col.index.reshape([1, col.index.size(0)]),
        ].to(torch.float32)
        product = pre * mapped
        out0[row, col] = pre.to(out0.dtype)
        out1[row, col] = product.to(out1.dtype)
        if count == 3 or count == 4:
            out2[row, col] = (pre + product).to(out2.dtype)
        if count == 4:
            out3[row, col] = (pre - product).to(out3.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def single(a, b, out):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        out[row, col] = acc.to(out.dtype)


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def values(dtype=torch.bfloat16, output_dtype=torch.float32, m=8192, n=384):
    return (
        torch.empty((m, 128), dtype=dtype),
        torch.empty_strided((128, n), (1, 128), dtype=dtype),
        torch.empty(n, dtype=torch.float32),
        torch.empty_strided((m // 512, 512, n), (512 * n, 1, 512), dtype=torch.float32),
        *(torch.empty((m, n), dtype=output_dtype) for _ in range(4)),
    )


def kernel(fn=multi, **settings):
    policy = helion.CuteStructuralPolicy.from_dict(
        {
            "schema": "helion.cute.structural_policy",
            "version": 1,
            "settings": {
                "cute_region_fission": True,
                "cute_full_slice_matmul_tiling": True,
                "cute_segmented_matmul_tiling": True,
                "cute_flatten_nested_reductions": True,
                "cute_materialize_transformed_operands": True,
            },
        }
    )
    return helion.kernel(
        fn.fn, settings=fn.settings.copy(**settings), cute_structural_policy=policy
    )


def bind(fn, args):
    with patch.object(fn, "_bind_isolated", side_effect=fn.bind):
        return cpu_bind(fn, args)


def generated(bound):
    with bound.env, bound.host_function:
        return shared_gemm_seed_configs(bound.env, bound.host_function.device_ir)


def fingerprint(generation):
    return [(type(f).__name__, f.search_values(limit=64)) for f in generation.flat_spec]


@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("output_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("count", (2, 3, 4))
def test_managed_complete_2_to_4_outputs_get_canonical_full_seeds(
    input_dtype, output_dtype, count
):
    bound = bind(kernel(), (*values(input_dtype, output_dtype), count))
    configs = generated(bound)
    assert configs
    assert all(config["tcgen05_epilogue_fanout"] == "shared" for config in configs)
    with bound.env, bound.host_function:
        canonical = ConfigGeneration(bound.config_spec).seed_flat_config_pairs()
        facts = seeds.prove_shared_gemm_seed_facts(
            bound.env, bound.host_function.device_ir
        )
    assert facts is not None and len(facts.whole.stores) == count
    assert facts.whole.output_dtype == output_dtype
    assert bound.config_spec._cute_tcgen05_config.full_tile_schedule_facts is None
    assert all(
        any(config == normalized for _, normalized in canonical) for config in configs
    )
    for config in configs:
        with bound.env, bound.host_function:
            assert bound.config_spec.normalized_config(config) == config
            generation = ConfigGeneration(bound.config_spec)
            _, roundtrip = generation.canonicalize_flat(generation.flatten(config))
            assert roundtrip == config
        assert helion.Config.from_dict(json.loads(json.dumps(config.config))) == config
    # Every config is strictly transferred above. Render one for every typed
    # cohort here; the separate exhaustive test renders every four-output seed.
    for config in configs[:1]:
        code = bound.to_code(config)
        assert code.count("PipelineTmaStore.create(") == count
        assert code.count("producer_commit()") == count
        assert code.count("producer_tail()") == count
        assert code.count("tcgen05_acc_pipeline.consumer_release(") == 1
        assert code.count("tcgen05_acc_consumer_state.advance()") == 1
        assert "workspace" not in code.lower() and "split_k" not in code.lower()


def test_builder_is_idempotent_and_does_not_consume_RNG_or_change_random_fragments():
    bound = bind(kernel(), (*values(), 3))
    generation = ConfigGeneration(bound.config_spec)
    before_fields = fingerprint(generation)
    before_python = random.getstate()
    before_torch = torch.get_rng_state().clone()
    before_default = bound.config_spec.default_config().config.copy()
    initial = list(bound.config_spec.compiler_seed_configs)
    configs = generated(bound)
    with bound.env, bound.host_function:
        seeds.append_shared_gemm_seed_configs(bound.env, bound.host_function.device_ir)
    assert generated(bound) == configs
    assert list(bound.config_spec.compiler_seed_configs) == initial
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert fingerprint(ConfigGeneration(bound.config_spec)) == before_fields
    assert bound.config_spec.default_config().config == before_default


def test_single_output_analysis_defaults_and_seeds_are_not_modified():
    operands = values()[:2]
    args = (
        *operands,
        torch.empty((operands[0].shape[0], operands[1].shape[1]), dtype=torch.bfloat16),
    )
    bound = bind(kernel(single), args)
    before = list(bound.config_spec.compiler_seed_configs)
    default = bound.config_spec.default_config()
    fingerprint_before = bound.config_spec.cache_fingerprint_hash()
    assert generated(bound) == []
    with bound.env, bound.host_function:
        seeds.append_shared_gemm_seed_configs(bound.env, bound.host_function.device_ir)
    assert bound.config_spec.compiler_seed_configs == before
    assert bound.config_spec.default_config() == default
    assert bound.config_spec.cache_fingerprint_hash() == fingerprint_before


@pytest.mark.parametrize("count", (2, 3, 4))
def test_all_output_dtype_and_ring_counts_are_required_by_budget(count):
    bound = bind(kernel(), (*values(output_dtype=torch.bfloat16), count))
    state = bound.config_spec._cute_tcgen05_config
    original = state.rowvec_aux_facts
    assert original is not None
    for field, value in (("output_count", 1), ("output_itemsize", 4)):
        with patch.object(
            state, "rowvec_aux_facts", original._replace(**{field: value})
        ):
            assert generated(bound) == []
    constraints = state.ab_stages_three_search_constraints
    assert constraints is not None
    with patch.object(
        state,
        "ab_stages_three_search_constraints",
        constraints._replace(output_stores=1),
    ):
        assert generated(bound) == []
    with patch.object(state, "ab_stages_three_search_constraints", None):
        assert generated(bound) == []


def test_missing_or_stale_complete_protocol_declines_optional_seed_not_default_bind():
    bound = bind(kernel(), (*values(), 3))
    state = bound.config_spec._cute_tcgen05_config
    plans = state.epilogue_fanout_plans
    with patch.object(state, "epilogue_fanout_plans", ()):
        assert generated(bound) == []
    with patch.object(
        state, "epilogue_fanout_plans", (replace(plans[0], shape=(1, 1)),)
    ):
        assert generated(bound) == []
    assert bound.config_spec.default_config() is not None
    assert generated(bound)


@pytest.mark.parametrize(
    "fault", ("alias", "different_dtype", "different_layout", "different_shape")
)
def test_ineligible_actual_views_have_no_optional_full_seeds(fault):
    args = list(values())
    if fault == "alias":
        args[5] = args[4].view_as(args[4])
    elif fault == "different_dtype":
        args[5] = torch.empty_like(args[5], dtype=torch.float16)
    elif fault == "different_layout":
        args[5] = torch.empty_strided(args[5].shape, (1, args[5].shape[0]))
    else:
        args[5] = torch.empty((args[5].shape[0] + 1, args[5].shape[1]))
    bound = bind(kernel(), (*args, 2))
    assert generated(bound) == []
    assert bound.config_spec.default_config() is not None


def test_missing_registered_disjoint_and_readonly_facts_decline_seeds():
    bound = bind(kernel(), (*values(), 3))
    with patch(
        "helion._compiler.runtime_tensor_guards.runtime_tensors_are_proven_disjoint",
        return_value=False,
    ):
        assert generated(bound) == []
    with patch(
        "helion._compiler.cute.register_loads.host_load_is_readonly", return_value=False
    ):
        assert generated(bound) == []


@pytest.mark.parametrize(
    "fault", ("mask", "foreign_store", "foreign_graph", "side_effect")
)
def test_incomplete_actual_whole_graph_declines_seeds(fault):
    bound = bind(kernel(), (*values(), 3))
    graphs = bound.host_function.device_ir.graphs
    stores = [n for g in graphs for n in g.graph.nodes if n.target is memory_ops.store]
    graph = stores[0].graph
    if fault == "mask":
        original = stores[0].args
        stores[0].args = (*original[:3], True)
        try:
            assert generated(bound) == []
        finally:
            stores[0].args = original
    elif fault == "foreign_graph":
        original = stores[0].args
        foreign = torch.fx.Graph().placeholder("foreign")
        foreign.meta.update(original[2].meta)
        stores[0].args = (*original[:2], foreign, original[3])
        try:
            assert generated(bound) == []
        finally:
            stores[0].args = original
    else:

        def effect():
            return None

        with graph.inserting_before(stores[1]):
            extra = (
                graph.call_function(effect, ())
                if fault == "side_effect"
                else graph.call_function(memory_ops.store, stores[0].args)
            )
        try:
            assert generated(bound) == []
        finally:
            graph.erase_node(extra)
    assert generated(bound)


def test_heuristics_disabled_does_not_add_seeds_or_change_default():
    bound = bind(kernel(disable_autotuner_heuristics=True), (*values(), 3))
    assert generated(bound) == []
    assert bound.config_spec.compiler_seed_configs == []
    assert bound.config_spec.default_config() is not None


def test_cached_managed_bind_preserves_positive_and_alias_refusal_identities():
    fn = kernel()
    args = values()
    first = bind(fn, (*args, 2))
    again = bind(fn, (*args, 2))
    assert first is again and generated(again)
    alias = list(args)
    alias[5] = alias[4].view_as(alias[4])
    changed = bind(fn, (*alias, 2))
    assert changed is not first and generated(changed) == []
