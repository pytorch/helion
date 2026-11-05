"""Managed coverage discovery cannot use a bound-only store trace as eligibility."""

from __future__ import annotations

import ast
from dataclasses import replace
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._compiler.autotuner_heuristics import cute_epilogue_fanout as coverage
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import cute_mma as mma
from helion._compiler.cute import shared_epilogue_dag as shared
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops

KEY = "tensor_storage_disjoint_matrix_v1"
DEFAULT = hl.constexpr(False)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def prefix_where(
    a,
    b,
    predicate,
    u,
    scalar,
    first,
    second,
    reverse: hl.constexpr = DEFAULT,
    cast_before: hl.constexpr = DEFAULT,
    bad_numeric: hl.constexpr = DEFAULT,
):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        pred = predicate[row, col]
        prefix = (
            u[row].to(torch.float32) + hl.full([], 1.0, dtype=torch.float32)
        ) * hl.full([], 0.5, dtype=torch.float32)
        ordinary = acc + prefix.reshape([row.index.size(0), 1])
        selected_scalar = scalar[()].to(torch.float32)
        if cast_before:
            numeric = pred.to(torch.float32)
        else:
            numeric = pred
        if reverse:
            selected = torch.where(pred, ordinary, selected_scalar)
        else:
            selected = torch.where(pred, selected_scalar, ordinary)
        if bad_numeric:
            second_value = selected * pred
        elif cast_before:
            second_value = selected * numeric
        else:
            second_value = selected * pred.to(torch.float32)
        first[row, col] = selected
        second[row, col] = second_value


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def arguments(dtype=torch.bfloat16, scalar_dtype=torch.bfloat16, m=256):
    return (
        torch.empty((m, 128), dtype=dtype),
        torch.empty_strided((128, 128), (1, 128), dtype=dtype),
        torch.empty((m, 128), dtype=torch.bool),
        torch.empty(m),
        torch.empty((), dtype=scalar_dtype),
        torch.empty((m, 128)),
        torch.empty((m, 128)),
    )


def kernel():
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
        prefix_where.fn, settings=prefix_where.settings, cute_structural_policy=policy
    )


def bind(fn, values, managed=True):
    if not managed:
        return cpu_bind(fn, values)
    with patch.object(fn, "_bind_isolated", side_effect=fn.bind):
        return cpu_bind(fn, values)


def config(shared_mode=True):
    return helion.Config(
        block_sizes=[128, 32, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_epilogue_fanout="shared" if shared_mode else "off",
    )


@pytest.mark.parametrize(
    "dtype,reverse,cast_before",
    tuple(
        itertools.product((torch.float16, torch.bfloat16), (False, True), (False, True))
    ),
)
@pytest.mark.parametrize("scalar_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_real_managed_bind_retains_same_complete_shared_plan_and_code_as_isolated(
    dtype, reverse, cast_before, scalar_dtype
):
    args = (*arguments(dtype, scalar_dtype), reverse, cast_before, False)
    isolated = bind(kernel(), args, managed=False)
    managed = bind(kernel(), args)
    state = managed.config_spec._cute_tcgen05_config
    assert managed.config_spec.cute_tcgen05_search_enabled
    assert len(state.epilogue_fanout_plans) == 1
    assert state.epilogue_fanout_search_enabled
    assert managed.to_code(config()) == isolated.to_code(config())
    assert managed.to_code(config(False)) == isolated.to_code(config(False))
    assert (
        managed.config_spec.cache_fingerprint_hash()
        == isolated.config_spec.cache_fingerprint_hash()
    )
    source = managed.to_code(config())
    t2r = [
        call
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "cute.copy"
        and call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id.startswith("tcgen05_tiled_copy_t2r")
    ]
    assert len(t2r) == 1
    assert source.count("PipelineTmaStore.create(") == 2
    assert source.count("producer_commit()") == 2
    assert source.count("producer_tail()") == 2
    assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
    assert source.count("tcgen05_acc_consumer_state.advance()") == 1
    assert "cute.where" in source and "!= cutlass.Uint8(0)" in source
    assert "Float32(1.0)" in source and "Float32(0.5)" in source
    plan = state.epilogue_fanout_plans[0]
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    dag = plan.chains[0].steps[0].expr
    assert isinstance(dag, epi._PointwiseDagExpr)
    raw_pred = [
        n for n in dag.nodes if isinstance(n, epi._ReadonlyBoolPredicateLoadExpr)
    ]
    assert raw_pred
    assert (
        state._readonly_bool_aux_seed_subject is not None
    )  # Original second-output explicit numeric cast, not the predicate tag.


@pytest.mark.parametrize(
    "reverse,cast_before", tuple(itertools.product((False, True), (False, True)))
)
def test_managed_shared_chain_keeps_prefix_where_numeric_cast_IEEE_and_byte_normalization(
    reverse, cast_before
):
    from test.test_cute_epilogue_dag import _evaluate

    bound = bind(kernel(), (*arguments(), reverse, cast_before, False))
    (plan,) = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    acc = torch.linspace(-2, 2, 128).reshape(8, 16)
    pred = (
        torch.tensor([0, 1, 2, 128, 255, 0, 2, 255], dtype=torch.uint8)
        .repeat(16)
        .reshape(8, 16)
    )
    vector = (
        torch.tensor(
            [-0.0, 0.0, -1.0, 2.0, float("inf"), -float("inf"), float("nan"), 2**-126]
        )
        .reshape(8, 1)
        .expand(8, 16)
    )
    scalar = torch.tensor(-0.0, dtype=torch.float32)
    ordinary = acc + (
        (vector + torch.tensor(1.0, dtype=torch.float32))
        * torch.tensor(0.5, dtype=torch.float32)
    )
    selected = (
        torch.where(pred.ne(0), ordinary, scalar)
        if reverse
        else torch.where(pred.ne(0), scalar, ordinary)
    )
    expected = (selected, selected * pred.ne(0).float())
    memo = {}
    counter = itertools.count()
    full_source = ""
    for chain, wanted in zip(plan.chains, expected, strict=True):
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        bindings: epi._LeafBindings = {}
        for leaf in chain.auxiliary_tensor_loads:
            bindings[leaf] = (
                "raw_u"
                if isinstance(leaf, epi._MVectorScalarPrefixLoadExpr)
                else "raw_pred"
            )
        for leaf in chain.runtime_scalars:
            bindings[leaf] = "raw_scalar"
        source, expr = shared.render_dag(
            dag,
            "original_acc",
            bindings,
            lambda name: f"{name}_{next(counter)}",
            memo,
            "",
        )
        full_source += source
        actual = _evaluate(
            full_source,
            expr,
            acc,
            raw_u=vector,
            raw_pred=pred,
            raw_scalar=scalar,
            cutlass=SimpleNamespace(
                Float32=torch.float32,
                Float16=torch.float16,
                BFloat16=torch.bfloat16,
                Uint8=torch.uint8,
            ),
        )
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(wanted)
        assert torch.equal(
            actual[finite].view(torch.int32), wanted[finite].view(torch.int32)
        )
        nonnan = ~torch.isnan(wanted)
        assert torch.equal(torch.signbit(actual[nonnan]), torch.signbit(wanted[nonnan]))


def test_store_inventory_discovery_reaches_registered_live_proof_before_snapshot_without_publication():
    observed = []
    original = coverage.prove_epilogue_fanout

    def prove(env, host, graphs, stores, anchors, **kwargs):
        facts = dict(env.bound_runtime_input_specialization_results)
        registered = dict(env.runtime_input_specializations)
        scope = env.early_mma_metadata_scope_active
        target = next(iter(anchors))
        trace = mma._trace_mma_to_stores(target, graphs)
        answer = original(env, host, graphs, stores, anchors, **kwargs)
        observed.append((facts, trace, kwargs, answer))
        assert facts == env.bound_runtime_input_specialization_results
        assert registered == env.runtime_input_specializations
        assert env.early_mma_metadata_scope_active == scope is False
        assert set(stores) == {
            n for g in graphs for n in g.graph.nodes if n.target is memory_ops.store
        }
        return answer

    with patch.object(coverage, "prove_epilogue_fanout", prove):
        bound = bind(kernel(), arguments())
    assert observed
    assert any(
        not facts
        and trace is None
        and kwargs == {"allow_unbound": True}
        and isinstance(plan, shared.SharedDagFanoutPlan)
        for facts, trace, kwargs, plan in observed
    )
    assert KEY in bound.env.bound_runtime_input_specialization_results
    assert set(bound.env.bound_runtime_input_specialization_results) == set(
        bound.env.runtime_input_specializations
    )
    assert bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans


@pytest.mark.parametrize(
    "fault",
    (
        "no_registration",
        "fake_args",
        "missing_args",
        "alias_destinations",
        "alias_input",
        "bad_classifier",
        "missing_reusable_storage_property",
        "contradictory_bound_fact",
    ),
)
def test_discovered_store_candidates_are_not_an_alias_proof_or_partial_snapshot_bypass(
    fault,
):
    fn = kernel()
    args = arguments()
    bound = bind(fn, args)
    env = bound.env
    state = bound.config_spec._cute_tcgen05_config
    facts = dict(env.bound_runtime_input_specialization_results)
    classifiers = dict(env.runtime_input_specializations)
    values = dict(zip(fn.signature.parameters, fn.normalize_args(*args), strict=True))
    env.bound_runtime_input_specialization_results = {}
    if fault == "no_registration":
        env.runtime_input_specializations.pop(KEY)
    elif fault == "fake_args":
        with FakeTensorMode() as mode:
            values = {
                k: mode.from_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in values.items()
            }
    elif fault == "missing_args":
        values = {}
    elif fault == "alias_destinations":
        values["second"] = values["first"]
    elif fault == "alias_input":
        values["u"] = args[-2].view(-1)[:256]
    elif fault == "bad_classifier":
        env.runtime_input_specializations[KEY] = replace(
            classifiers[KEY], classifier_identity=("unproved",)
        )
    elif fault == "missing_reusable_storage_property":
        env.runtime_input_specializations[KEY] = replace(
            classifiers[KEY], reusable_tensor_properties=frozenset()
        )
    else:
        env.bound_runtime_input_specialization_results[KEY] = tuple(
            False for _ in facts[KEY]
        )
    try:
        with env, bound.host_function, env.use_runtime_arg_values(values):
            before = dict(env.bound_runtime_input_specialization_results)
            registration = dict(env.runtime_input_specializations)
            coverage.register_epilogue_fanout_coverage(
                env, bound.host_function.device_ir
            )
            assert state.epilogue_fanout_plans == ()
            assert before == env.bound_runtime_input_specialization_results
            assert registration == env.runtime_input_specializations
    finally:
        env.bound_runtime_input_specialization_results = facts
        env.runtime_input_specializations = classifiers


@pytest.mark.parametrize("order", (False, True))
def test_raw_predicate_numeric_use_is_rejected_in_both_memo_orders(order):
    bound = bind(kernel(), (*arguments(), False, order, True))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans


@pytest.mark.parametrize("changed", ("destination_alias", "input_alias"))
def test_managed_cache_rebinding_cannot_reuse_prior_fanout_proof(changed):
    fn = kernel()
    args = arguments()
    original = bind(fn, args)
    alias = list(args)
    if changed == "destination_alias":
        alias[-1] = alias[-2]
    else:
        alias[3] = args[-2].view(-1)[:256]
    rebound = bind(fn, tuple(alias))
    assert rebound is not original
    assert original.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert not rebound.config_spec._cute_tcgen05_config.epilogue_fanout_plans


@pytest.mark.parametrize("missing", ("bound", "registered"))
def test_final_codegen_remains_bound_only_and_does_not_repair_missing_runtime_facts(
    missing,
):
    bound = bind(kernel(), arguments())
    env = bound.env
    table = (
        env.bound_runtime_input_specialization_results
        if missing == "bound"
        else env.runtime_input_specializations
    )
    saved = table.pop(KEY)
    try:
        before = dict(table)
        with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
            bound.to_code(config())
        assert table == before
    finally:
        table[KEY] = saved


@pytest.mark.parametrize(
    "fault",
    (
        "nonstore_effect",
        "device_if",
        "foreign_store_value",
        "foreign_graph_store",
        "cyclic_prefix",
        "over_budget",
    ),
)
def test_inventory_fallback_retains_complete_anchor_effect_interval_and_budget_guards(
    fault,
):
    fn = kernel()
    args = arguments()
    bound = bind(fn, args)
    env = bound.env
    graphs = bound.host_function.device_ir.graphs
    state = bound.config_spec._cute_tcgen05_config
    plan = state.epilogue_fanout_plans[0]
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    graph = plan.stores[0].graph
    value = plan.stores[0].args[2]
    assert isinstance(value, torch.fx.Node)
    if fault in ("nonstore_effect", "device_if"):
        with graph.inserting_before(plan.stores[-1]):
            graph.call_function(
                _tracing_ops._if if fault == "device_if" else hl.barrier, ()
            )
    elif fault == "foreign_store_value":
        scalar = next(
            n
            for n in graph.nodes
            if n.target is memory_ops.load and n.meta["val"].ndim == 0
        )
        store = plan.stores[-1]
        store.args = (store.args[0], store.args[1], scalar, None)
    elif fault == "foreign_graph_store":
        other = graphs[0].graph
        with other.inserting_before(next(n for n in other.nodes if n.op == "output")):
            other.call_function(memory_ops.store, plan.stores[0].args)
    else:
        prefix = next(
            leaf
            for chain in plan.chains
            for leaf in chain.auxiliary_tensor_loads
            if isinstance(leaf, epi._MVectorScalarPrefixLoadExpr)
        )
        add = next(
            origin
            for origin, _dtype in prefix.prefix.provenance
            if origin.target is torch.ops.aten.add.Tensor
        )
        if fault == "cyclic_prefix":
            add.args = (add, add.args[1])
        else:
            original = add.args[0]
            assert isinstance(original, torch.fx.Node)
            with graph.inserting_before(add):
                for _ in range(65):
                    node = graph.call_function(torch.ops.aten.neg.default, (original,))
                    node.meta["val"] = original.meta["val"]
                    original = node
            add.args = (original, add.args[1])
    values = dict(zip(fn.signature.parameters, fn.normalize_args(*args), strict=True))
    with env, bound.host_function, env.use_runtime_arg_values(values):
        coverage.register_epilogue_fanout_coverage(env, bound.host_function.device_ir)
    assert not state.epilogue_fanout_plans
