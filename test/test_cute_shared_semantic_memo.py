"""Shared semantic memo is optimization-only; original proof keys stay strict."""

from __future__ import annotations

import ast
from dataclasses import replace
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_epilogue_dag import _evaluate
from test.test_cute_where_shared_managed_coverage import prefix_where

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import shared_epilogue_dag as shared
from helion._compiler.cute import shared_semantic_memo as semantics
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl
from helion.language import memory_ops

DEFAULT = hl.constexpr("same")
KEY = "tensor_storage_disjoint_matrix_v1"


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="full", fast_math=False
)
def repeated_prefix(
    a, b, u, other_u, dense, nvec, out0, out1, out2, out3, mode: hl.constexpr = DEFAULT
):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        out0[row, col] = acc
        first = u[row] + hl.full([], 1e-6, dtype=torch.float32)
        if mode == "association":
            first = (u[row] + hl.full([], 1.0, dtype=torch.float32)) * hl.full(
                [], 0.5, dtype=torch.float32
            )
        if mode == "zero":
            first = u[row] + hl.full([], 0.0, dtype=torch.float32)
        q1 = acc / first.reshape([row.index.size(0), 1])
        out1[row, col] = q1 * dense[row, col]
        if mode == "other_input":
            second = other_u[row] + hl.full([], 1e-6, dtype=torch.float32)
        elif mode == "epsilon":
            second = u[row] + hl.full([], 2e-6, dtype=torch.float32)
        elif mode == "zero":
            second = u[row] + hl.full([], -0.0, dtype=torch.float32)
        elif mode == "halfcast":
            second = (
                (u[row] + hl.full([], 1e-6, dtype=torch.float32))
                .to(torch.float16)
                .to(torch.float32)
            )
        elif mode == "bfcast":
            second = (
                (u[row] + hl.full([], 1e-6, dtype=torch.float32))
                .to(torch.bfloat16)
                .to(torch.float32)
            )
        elif mode == "association":
            second = u[row] * hl.full([], 0.5, dtype=torch.float32) + hl.full(
                [], 0.5, dtype=torch.float32
            )
        elif mode == "offset":
            second = u[row.index + 1] + hl.full([], 1e-6, dtype=torch.float32)
        elif mode == "huge_weak":
            second = u[row] + 16777217
        else:
            second = u[row] + hl.full([], 1e-6, dtype=torch.float32)
        q2 = acc / second.reshape([row.index.size(0), 1])
        out2[row, col] = q2 * nvec[col]
        last = u[row] + hl.full([], 1e-6, dtype=torch.float32)
        if mode == "association":
            last = (u[row] + hl.full([], 1.0, dtype=torch.float32)) * hl.full(
                [], 0.5, dtype=torch.float32
            )
        if mode == "zero":
            last = u[row] + hl.full([], 0.0, dtype=torch.float32)
        pair = (dense[row, col] * nvec[col]) / last.reshape([row.index.size(0), 1])
        pair = pair / last.reshape([row.index.size(0), 1])
        out3[row, col] = (-acc) * pair


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def arguments(dtype=torch.bfloat16):
    return (
        torch.empty((256, 128), dtype=dtype),
        torch.empty_strided((128, 128), (1, 128), dtype=dtype),
        torch.empty(256),
        torch.empty(256),
        torch.empty((256, 128)),
        torch.empty(128),
        *[torch.empty((256, 128)) for _ in range(4)],
    )


def bind(args=None, mode="same"):
    values = arguments() if args is None else args
    kernel = helion.kernel(repeated_prefix.fn, settings=repeated_prefix.settings)
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, (*values, mode))
    return bound, kernel, (*values, mode)


def configuration(shared_mode=True):
    return helion.Config(
        block_sizes=[128, 32, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_epilogue_fanout="shared" if shared_mode else "off",
    )


def plan(bound):
    (result,) = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert isinstance(result, shared.SharedDagFanoutPlan)
    return result


def count_div(source):
    return sum(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
        for n in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_complete_managed_shared_plan_reuses_only_proved_prefix_and_common_quotient(
    dtype,
):
    bound, _, _ = bind(arguments(dtype))
    proved = plan(bound)
    memo = proved.semantic_memo
    assert memo is not None
    assert count_div(bound.to_code(configuration())) == 3
    assert count_div(bound.to_code(configuration(False))) == 4
    code = bound.to_code(configuration())
    assert code.count("PipelineTmaStore.create(") == 4
    assert code.count("producer_commit()") == code.count("producer_tail()") == 4
    assert (
        code.count("tcgen05_acc_pipeline.consumer_release(")
        == code.count("tcgen05_acc_consumer_state.advance()")
        == 1
    )
    copies = [
        n
        for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "cute.copy"
        and n.args
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id.startswith("tcgen05_tiled_copy_t2r")
    ]
    assert len(copies) == 1
    divisions = []
    for dag_index, chain in enumerate(proved.chains):
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        strict = shared.dag_keys(dag)
        assert strict == memo.strict_keys[dag_index]
        for i, (node, (origin, _dtype)) in enumerate(
            zip(dag.nodes, dag.provenance, strict=True)
        ):
            if (
                isinstance(node, epi._BinaryTensorExpr)
                and origin.target is torch.ops.aten.div.Tensor
            ):
                divisions.append((origin, strict[i], memo.value_keys[dag_index][i]))
    q1 = next(t for t in divisions if t[0].name == "q1")
    q2 = next(t for t in divisions if t[0].name == "q2")
    assert q1[0] is not q2[0] and q1[1] != q2[1] and q1[2] == q2[2]
    assert any(q1[0] in group and q2[0] in group for group in memo.origins_by_value)
    # The original sequential pair has distinct ordered numerator children.
    pair = [t for t in divisions if t[0].name.startswith("pair")]
    assert len(pair) == 2 and pair[0][2] != pair[1][2]


@pytest.mark.parametrize(
    "mode", ("other_input", "epsilon", "zero", "halfcast", "bfcast", "association")
)
def test_distinct_input_literal_cast_and_association_never_share_common_quotient(mode):
    bound, _, _ = bind(mode=mode)
    proved = plan(bound)
    assert count_div(bound.to_code(configuration())) == 4
    if proved.semantic_memo is not None:
        keys = {}
        for k, chain in enumerate(proved.chains):
            dag = chain.steps[0].expr
            assert isinstance(dag, epi._PointwiseDagExpr)
            for i, (_node, (origin, _dtype)) in enumerate(
                zip(dag.nodes, dag.provenance, strict=True)
            ):
                if origin.name in ("q1", "q2"):
                    keys[origin.name] = proved.semantic_memo.value_keys[k][i]
        assert keys["q1"] != keys["q2"]


@pytest.mark.parametrize("mode", ("offset", "huge_weak"))
def test_unproved_index_and_scalar_promotion_are_not_repaired_by_semantic_cse(mode):
    bound, _, _ = bind(mode=mode)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans


def test_duplicate_input_pointer_and_equal_samples_do_not_replace_input_identity():
    values = list(arguments())
    values[3] = values[2]
    bound, _, _ = bind(tuple(values), "other_input")
    assert count_div(bound.to_code(configuration())) == 4


def test_semantic_render_preserves_ordered_IEEE_values_signedzeros_and_cse_pair():
    bound, _, _ = bind()
    proved = plan(bound)
    memo = proved.semantic_memo
    assert memo is not None
    acc = torch.linspace(-2, 2, 128).reshape(8, 16)
    u = (
        torch.tensor(
            [-0.0, 0.0, -1e-6, 1e-6, 2**-126, float("inf"), -float("inf"), float("nan")]
        )
        .reshape(8, 1)
        .expand(8, 16)
    )
    dense = torch.linspace(-1, 1, 128).reshape(8, 16)
    nvec = torch.linspace(-1, 1, 16).reshape(1, 16).expand(8, 16)
    denominator = u + torch.tensor(1e-6, dtype=torch.float32)
    q = acc / denominator
    pair = ((dense * nvec) / denominator) / denominator
    expected = (acc, q * dense, q * nvec, (-acc) * pair)
    state = {}
    counter = itertools.count()
    source = ""
    for k, (chain, wanted) in enumerate(zip(proved.chains, expected, strict=True)):
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        bindings: epi._LeafBindings = {}
        for leaf in chain.auxiliary_tensor_loads:
            host = leaf.load_node.args[0]
            assert isinstance(host, torch.fx.Node)
            bindings[leaf] = {"u": "raw_u", "dense": "raw_dense", "nvec": "raw_nvec"}[
                host.name
            ]
        generated, expression = shared.render_dag(
            dag,
            "original_acc",
            bindings,
            lambda name: f"{name}_{next(counter)}",
            state,
            "",
            semantic_keys=memo.value_keys[k],
        )
        source += generated
        got = _evaluate(
            source,
            expression,
            acc,
            raw_u=u,
            raw_dense=dense,
            raw_nvec=nvec,
            cutlass=SimpleNamespace(
                Float32=torch.float32,
                Float16=torch.float16,
                BFloat16=torch.bfloat16,
                Uint8=torch.uint8,
            ),
        )
        assert isinstance(got, torch.Tensor)
        torch.testing.assert_close(got, wanted, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(wanted)
        valid = ~torch.isnan(wanted)
        assert torch.equal(
            got[finite].view(torch.int32), wanted[finite].view(torch.int32)
        )
        assert torch.equal(torch.signbit(got[valid]), torch.signbit(wanted[valid]))
    assert count_div(source) == 3


@pytest.mark.parametrize(
    "fault",
    (
        "missing_bound",
        "missing_registration",
        "false_bound",
        "destination_alias",
        "input_alias",
        "fake_args",
        "missing_args",
        "bad_classifier",
    ),
)
def test_lost_or_invalid_facts_never_use_semantic_keys_as_admission(fault):
    bound, kernel, values = bind()
    env = bound.env
    proved = plan(bound)
    saved = dict(env.bound_runtime_input_specialization_results)
    registered = dict(env.runtime_input_specializations)
    if fault in ("missing_bound", "missing_registration", "false_bound"):
        table = (
            env.runtime_input_specializations
            if fault == "missing_registration"
            else env.bound_runtime_input_specialization_results
        )
        if fault == "false_bound":
            table[KEY] = tuple(False for _ in saved[KEY])
        else:
            table.pop(KEY)
        try:
            with pytest.raises(
                (helion.exc.InvalidConfig, helion.exc.BackendUnsupported)
            ):
                bound.to_code(configuration())
        finally:
            env.bound_runtime_input_specialization_results = saved
            env.runtime_input_specializations = registered
        return
    actual = dict(zip(kernel.signature.parameters, values, strict=True))
    env.bound_runtime_input_specialization_results = {}
    if fault == "destination_alias":
        actual["out2"] = actual["out1"]
    elif fault == "input_alias":
        output = values[6]
        assert isinstance(output, torch.Tensor)
        actual["u"] = output.view(-1)[:256]
    elif fault == "fake_args":
        with FakeTensorMode() as mode:
            actual = {
                k: mode.from_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in actual.items()
            }
    elif fault == "missing_args":
        actual = {}
    else:
        env.runtime_input_specializations[KEY] = replace(
            registered[KEY], classifier_identity=("unproved",)
        )
    try:
        with env, bound.host_function, env.use_runtime_arg_values(actual):
            graphs = bound.host_function.device_ir.graphs
            anchors = {
                n
                for g in graphs
                for n in g.graph.nodes
                if n.target is torch.ops.aten.addmm.default
            }
            assert (
                shared.prove_shared_dag_fanout(
                    env,
                    bound.host_function,
                    graphs,
                    proved.stores,
                    anchors,
                    allow_unbound=True,
                )
                is None
            )
            assert not env.bound_runtime_input_specialization_results
    finally:
        env.bound_runtime_input_specialization_results = saved
        env.runtime_input_specializations = registered


@pytest.mark.parametrize(
    "fault",
    (
        "mask",
        "wrong_index",
        "foreign_graph",
        "different_dtype",
        "different_layout",
        "different_device",
    ),
)
def test_raw_semantic_access_checks_keep_all_typed_layout_coordinate_dimensions(fault):
    bound, _, _ = bind()
    proved = plan(bound)
    memo = proved.semantic_memo
    assert memo is not None
    leaf = next(
        candidate
        for candidate in proved.chains[1].auxiliary_tensor_loads
        if isinstance(candidate, epi._MVectorScalarPrefixLoadExpr)
    )
    original = semantics.load_signature(leaf)
    load = leaf.load_node
    old_args = load.args
    old_value = load.meta["val"]
    with bound.env:
        if fault == "mask":
            load.args = (*old_args[:2], True, None)
        elif fault == "wrong_index":
            other = torch.fx.Graph().placeholder("different_index")
            load.args = (old_args[0], [other], *old_args[2:])
        elif fault == "foreign_graph":
            host = torch.fx.Graph().placeholder("other_host")
            old_host = old_args[0]
            assert isinstance(old_host, torch.fx.Node)
            host.meta["val"] = old_host.meta["val"]
            load.args = (host, old_args[1], *old_args[2:])
        elif fault == "different_dtype":
            load.meta["val"] = torch.empty(
                old_value.shape, dtype=torch.float16, device=old_value.device
            )
        elif fault == "different_layout":
            load.meta["val"] = torch.empty_strided(
                old_value.shape, (2,), dtype=old_value.dtype, device=old_value.device
            )
        else:
            load.meta["val"] = torch.empty(
                old_value.shape, dtype=old_value.dtype, device="cpu"
            )
        try:
            if fault in ("mask", "wrong_index", "foreign_graph"):
                with pytest.raises(ValueError):
                    semantics.load_signature(leaf)
            else:
                assert semantics.load_signature(leaf) != original
        finally:
            load.args = old_args
            load.meta["val"] = old_value


def test_context_isolation_bool_predicate_and_numeric_cast_keep_distinct_value_keys():
    from test.test_cute_where_shared_managed_coverage import arguments

    values = (*arguments(), False, False, False)
    kernel = helion.kernel(prefix_where.fn, settings=prefix_where.settings)
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, values)
    proved = plan(bound)
    pool = semantics._ValuePool(
        proved.stores[0],
        next(
            n
            for g in bound.host_function.device_ir.graphs
            for n in g.graph.nodes
            if n.target is torch.ops.aten.addmm.default
        ),
    )
    tagged = {}
    for chain in proved.chains:
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        keys = pool.dag(dag)
        for n, k in zip(dag.nodes, keys, strict=True):
            if isinstance(
                n, (epi._ReadonlyBoolPredicateLoadExpr, epi._ReadonlyBoolAuxLoadExpr)
            ):
                tagged[type(n)] = k
    assert (
        tagged[epi._ReadonlyBoolPredicateLoadExpr]
        != tagged[epi._ReadonlyBoolAuxLoadExpr]
    )


@pytest.mark.parametrize(
    "fault", ("strict_origin", "load_signature", "forged_value_keys", "foreign_scope")
)
def test_final_rendering_rejects_tampered_sources_even_when_semantic_values_match(
    fault,
):
    bound, _, _ = bind()
    captured = []
    original = shared.combine_dag_stores

    def observe(stores):
        captured.extend(stores)
        return original(stores)

    with patch.object(shared, "combine_dag_stores", observe):
        bound.to_code(configuration())
    stores = list(captured[-4:])
    assert len(stores) == 4
    proved = stores[0].plan
    assert isinstance(proved, shared.SharedDagFanoutPlan)
    memo = proved.semantic_memo
    assert memo is not None
    target = stores[2]
    actual = target.iteration.chain.chain.steps[0].expr
    assert isinstance(actual, epi._PointwiseDagExpr)
    if fault == "strict_origin":
        old, dtype = actual.provenance[-1]
        assert callable(old.target)
        clone = old.graph.call_function(old.target, old.args, old.kwargs)
        clone.meta["val"] = old.meta["val"]
        changed = replace(actual, provenance=(*actual.provenance[:-1], (clone, dtype)))
        # Equal pure value is not equal source/proof identity.
        pool = semantics._ValuePool(proved.stores[0], actual.provenance[0][0])
        assert pool.dag(actual) == pool.dag(changed)
        assert shared.dag_keys(actual) != shared.dag_keys(changed)
        chain = replace(
            target.iteration.chain.chain,
            steps=(replace(target.iteration.chain.chain.steps[0], expr=changed),),
        )
        rendered = replace(target.iteration.chain, chain=chain)
        stores[2] = replace(target, iteration=replace(target.iteration, chain=rendered))
    elif fault == "load_signature":
        leaf = next(
            candidate
            for candidate in actual.nodes
            if isinstance(candidate, epi._MVectorScalarPrefixLoadExpr)
        )
        old = leaf.load_node.args[0]
        assert isinstance(old, torch.fx.Node)
        clone = old.graph.placeholder("same_metadata_not_same_input")
        clone.meta["val"] = old.meta["val"]
        old_args = leaf.load_node.args
        leaf.load_node.args = (clone, *old_args[1:])
    else:
        if fault == "forged_value_keys":
            altered = replace(
                memo,
                value_keys=tuple(
                    tuple(keys[0] for _ in keys) for keys in memo.value_keys
                ),
            )
        else:
            other = proved.stores[0].graph.placeholder("unproved_scope")
            altered = replace(memo, scope=other)
        changed_plan = replace(proved, semantic_memo=altered)
        stores = [replace(store, plan=changed_plan) for store in stores]
    with pytest.raises(helion.exc.BackendUnsupported):
        shared.combine_dag_stores(stores)


@pytest.mark.parametrize(
    "kind", ("runtime_scalar", "half_constant", "signed_zero", "literal_rounding")
)
def test_semantic_value_atoms_do_not_infer_scalar_or_storage_rounding_equivalence(kind):
    graph = torch.fx.Graph()
    anchor = graph.placeholder("anchor")
    scope = graph.call_function(memory_ops.store, ())
    one = graph.placeholder("one")
    two = graph.placeholder("two")
    pool = semantics._ValuePool(scope, anchor)
    if kind == "runtime_scalar":
        left = epi._RuntimeScalarExpr(one)
        right = epi._RuntimeScalarExpr(two)
    elif kind == "half_constant":
        left = epi._ScalarConstantExpr("1.0", torch.float16)
        right = epi._ScalarConstantExpr("1.0", torch.bfloat16)
    elif kind == "signed_zero":
        left = epi._ScalarConstantExpr("0.0", torch.float32)
        right = epi._ScalarConstantExpr("-0.0", torch.float32)
    else:
        # Same rounded storage value, but no new double/weak-literal inference.
        left = epi._ScalarConstantExpr("1.0", torch.float32)
        right = epi._ScalarConstantExpr("1.0000000596046448", torch.float32)
    first = epi._PointwiseDagExpr((left,), ((one, torch.float32),))
    second = epi._PointwiseDagExpr((right,), ((two, torch.float32),))
    assert pool.dag(first) != pool.dag(second)


@pytest.mark.parametrize(
    "fault", ("cycle", "large_graph", "effect", "strict_origin_mismatch")
)
def test_semantic_merging_cannot_repair_source_provenance_effect_or_budget_faults(
    fault,
):
    bound, _, _ = bind()
    proved = plan(bound)
    graphs = bound.host_function.device_ir.graphs
    graph = proved.stores[0].graph
    anchors = {
        n
        for g in graphs
        for n in g.graph.nodes
        if n.target is torch.ops.aten.addmm.default
    }
    if fault == "effect":
        with graph.inserting_before(proved.stores[-1]):
            graph.call_function(hl.barrier, ())
    elif fault == "strict_origin_mismatch":
        dag = proved.chains[1].steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        origin, dtype = dag.provenance[-1]
        other = graph.placeholder("unproved_origin")
        other.meta["val"] = origin.meta["val"]
        changed = replace(dag, provenance=(*dag.provenance[:-1], (other, dtype)))
        assert shared.dag_keys(changed) != shared.dag_keys(dag)
        return
    else:
        leaf = next(
            candidate
            for candidate in proved.chains[1].auxiliary_tensor_loads
            if isinstance(candidate, epi._MVectorScalarPrefixLoadExpr)
        )
        add = next(
            n
            for n, _ in leaf.prefix.provenance
            if n.target is torch.ops.aten.add.Tensor
        )
        if fault == "cycle":
            add.args = (add, add.args[1])
        else:
            source = add.args[0]
            assert isinstance(source, torch.fx.Node)
            with graph.inserting_before(add):
                for _ in range(65):
                    clone = graph.call_function(torch.ops.aten.neg.default, (source,))
                    clone.meta["val"] = source.meta["val"]
                    source = clone
            add.args = (source, add.args[1])
    with bound.env, bound.host_function:
        assert (
            shared.prove_shared_dag_fanout(
                bound.env, bound.host_function, graphs, proved.stores, anchors
            )
            is None
        )
