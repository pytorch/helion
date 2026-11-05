"""Exhaustive source and strict provenance/budget tests of shared GEMM seeds."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_shared_gemm_seeds import bind
from test.test_cute_shared_gemm_seeds import cpu_only as cpu_only
from test.test_cute_shared_gemm_seeds import generated
from test.test_cute_shared_gemm_seeds import kernel
from test.test_cute_shared_gemm_seeds import values

from helion._compiler.cute import shared_epilogue_dag as shared
from helion._compiler.cute import shared_gemm_seeds as seeds
from helion.language import memory_ops


def test_exhaustive_four_output_seed_sources_keep_all_pipelines():
    bound = bind(kernel(), (*values(), 4))
    configs = generated(bound)
    assert configs
    for config in configs:
        source = bound.to_code(config)
        assert source.count("PipelineTmaStore.create(") == 4
        assert source.count("producer_commit()") == 4
        assert source.count("producer_tail()") == 4
        assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
        assert source.count("tcgen05_acc_consumer_state.advance()") == 1
        assert "workspace" not in source.lower() and "split_k" not in source.lower()


def test_existing_compiler_seed_population_is_not_extended():
    from test.test_cute_m_vector_broadcast import arguments as m_arguments
    from test.test_cute_m_vector_broadcast import m_vector_epilogue

    from helion._testing.cute_codegen import cpu_bind

    bound = cpu_bind(m_vector_epilogue, m_arguments(torch.bfloat16))
    before = list(bound.config_spec.compiler_seed_configs)
    assert before
    schema = bound.config_spec.cache_fingerprint_hash()
    with bound.env, bound.host_function:
        seeds.append_shared_gemm_seed_configs(bound.env, bound.host_function.device_ir)
    assert bound.config_spec.compiler_seed_configs == before
    assert bound.config_spec.cache_fingerprint_hash() == schema


@pytest.mark.parametrize(
    "fault",
    (
        "index",
        "dtype",
        "registered_cast_keys",
        "registered_index_keys",
        "effect_before_store",
    ),
)
def test_typed_source_identity_and_whole_effect_refusals(fault):
    bound = bind(kernel(), (*values(), 3))
    state = bound.config_spec._cute_tcgen05_config
    graph_infos = bound.host_function.device_ir.graphs
    stores = [
        n
        for info in graph_infos
        for n in info.graph.nodes
        if n.target is memory_ops.store
    ]
    graph = stores[0].graph
    if fault == "index":
        original = stores[0].args
        stores[0].args = (original[0], list(reversed(original[1])), *original[2:])
        try:
            assert generated(bound) == []
        finally:
            stores[0].args = original
    elif fault == "dtype":
        load = next(n for n in graph.nodes if n.target is memory_ops.load)
        original = load.meta.copy()
        load.meta.clear()
        try:
            assert generated(bound) == []
        finally:
            load.meta.update(original)
    elif fault.startswith("registered"):
        plan = state.epilogue_fanout_plans[0]
        if fault == "registered_index_keys":
            bad = replace(plan, block_ids=tuple(reversed(plan.block_ids)))
        else:
            chain = plan.chains[0]
            dag = chain.steps[0].expr
            assert isinstance(dag, shared.epi._PointwiseDagExpr)
            provenance = list(dag.provenance)
            origin, dtype = provenance[-1]
            provenance[-1] = (
                origin,
                torch.float16 if dtype is torch.float32 else torch.float32,
            )
            changed = replace(
                chain,
                steps=(
                    replace(
                        chain.steps[0], expr=replace(dag, provenance=tuple(provenance))
                    ),
                ),
            )
            bad = replace(plan, chains=(changed, *plan.chains[1:]))
        with patch.object(state, "epilogue_fanout_plans", (bad,)):
            assert generated(bound) == []
    else:

        def side_effect():
            return None

        with graph.inserting_before(stores[0]):
            extra = graph.call_function(side_effect, ())
        try:
            assert generated(bound) == []
        finally:
            graph.erase_node(extra)
    assert generated(bound)


def test_a_budget_that_cannot_price_all_four_rings_has_no_seed():
    bound = bind(kernel(), (*values(), 4))
    state = bound.config_spec._cute_tcgen05_config
    constraints = state.ab_stages_three_search_constraints
    assert constraints is not None
    with patch.object(
        state,
        "ab_stages_three_search_constraints",
        constraints._replace(per_cta_smem_budget_bytes=0),
    ):
        assert generated(bound) == []


@pytest.mark.parametrize("count", (2, 3, 4))
def test_fresh_prebinding_JSON_config_roundtrip_keeps_source_and_all_outputs(count):
    import ast
    import json

    from test.test_cute_shared_gemm_seeds import multi

    import helion

    original = kernel()
    args = (*values(), count)
    bound = bind(original, args)
    config = generated(bound)[-1]
    copied = helion.Config.from_dict(json.loads(config.to_json()))
    cold = helion.kernel(
        multi.fn,
        config=copied,
        settings=original.settings,
        cute_structural_policy=original.cute_structural_policy,
    )
    rebound = bind(cold, args)
    assert rebound is not bound
    assert rebound._normalized_config_copy(copied) == config
    before = bound.to_code(config)
    after = rebound.to_code(copied)
    assert ast.dump(ast.parse(before), include_attributes=False) == ast.dump(
        ast.parse(after), include_attributes=False
    )
    assert after.count("PipelineTmaStore.create(") == count
