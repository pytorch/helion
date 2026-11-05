"""An identity MMA output is admitted only inside a complete shared fanout."""

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

import helion
from helion._compiler.autotuner_heuristics import cute_epilogue_fanout as coverage
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import shared_epilogue_dag as shared
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops

DEFAULT_POSITION = hl.constexpr(0)
DEFAULT_FLAG = hl.constexpr(False)
KEY = "tensor_storage_disjoint_matrix_v1"


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="full", fast_math=False
)
def identity_fanout(
    a,
    b,
    predicate,
    u,
    scalar,
    identity,
    first,
    second,
    position: hl.constexpr = DEFAULT_POSITION,
    duplicate_identity: hl.constexpr = DEFAULT_FLAG,
):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        pred = predicate[row, col]
        prefix = (u[row] + hl.full([], 1.0, dtype=torch.float32)) * hl.full(
            [], 0.5, dtype=torch.float32
        )
        selected = torch.where(
            pred,
            scalar[()].to(torch.float32),
            acc + prefix.reshape([row.index.size(0), 1]),
        )
        rounded = selected.to(a.dtype).to(torch.float32)
        computed = rounded * pred.to(torch.float32)
        if duplicate_identity:
            selected = acc
        if position == 0:
            identity[row, col] = acc
            first[row, col] = selected
            second[row, col] = computed
        elif position == 1:
            first[row, col] = selected
            identity[row, col] = acc
            second[row, col] = computed
        else:
            first[row, col] = selected
            second[row, col] = computed
            identity[row, col] = acc


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def standalone_identity(a, b, out):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        out[row, col] = acc


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def args(dtype=torch.bfloat16, output_dtype=torch.float32):
    return (
        torch.empty((256, 128), dtype=dtype),
        torch.empty_strided((128, 128), (1, 128), dtype=dtype),
        torch.empty((256, 128), dtype=torch.bool),
        torch.empty(256),
        torch.empty((), dtype=dtype),
        *[torch.empty((256, 128), dtype=output_dtype) for _ in range(3)],
    )


def make_kernel():
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
        identity_fanout.fn,
        settings=identity_fanout.settings,
        cute_structural_policy=policy,
    )


def bind(fn, values, managed=True):
    if not managed:
        return cpu_bind(fn, values)
    with patch.object(fn, "_bind_isolated", side_effect=fn.bind):
        return cpu_bind(fn, values)


def cfg(shared_mode=True):
    return helion.Config(
        block_sizes=[128, 32, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_epilogue_fanout="shared" if shared_mode else "off",
    )


def plan_for(bound):
    graph_infos = bound.host_function.device_ir.graphs
    anchors = {
        n
        for g in graph_infos
        for n in g.graph.nodes
        if n.target is torch.ops.aten.addmm.default
    }
    stores = [
        n for g in graph_infos for n in g.graph.nodes if n.target is memory_ops.store
    ]
    with bound.env, bound.host_function:
        plan = shared.prove_shared_dag_fanout(
            bound.env, bound.host_function, graph_infos, stores, anchors
        )
    return plan, graph_infos, anchors


@pytest.mark.parametrize(
    "input_dtype,output_dtype,position",
    tuple(
        itertools.product(
            (torch.float16, torch.bfloat16),
            (torch.float16, torch.bfloat16, torch.float32),
            (0, 1, 2),
        )
    ),
)
def test_shared_identity_first_middle_last_preserves_all_outputs_casts_and_final_lifetime(
    input_dtype, output_dtype, position
):
    values = (*args(input_dtype, output_dtype), position, False)
    managed = bind(make_kernel(), values)
    isolated = bind(make_kernel(), values, managed=False)
    assert managed.config_spec.cute_tcgen05_search_enabled
    plan, graphs, anchors = plan_for(managed)
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    identity_indices = []
    for i, chain in enumerate(plan.chains):
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        if isinstance(dag.nodes[-1], epi._CurrentTensorExpr):
            identity_indices.append(i)
    # Half stores already own an explicit typed terminal cast, not a pure
    # identity root. This option changes only the FP32 identity contract.
    assert identity_indices == ([position] if output_dtype is torch.float32 else [])
    assert len(plan.stores) == 3
    identity_value = plan.stores[position].args[2]
    assert isinstance(identity_value, torch.fx.Node)
    with managed.env, managed.host_function:
        standalone = epi._analyze_tcgen05_pointwise_dag(
            identity_value, graphs, anchors, plan.shape
        )
        assert (standalone is None) == (output_dtype is torch.float32)
        result = epi._analyze_tcgen05_pointwise_dag(
            identity_value, graphs, anchors, plan.shape, allow_identity_output=True
        )
        assert result is not None
    assert managed.to_code(cfg()) == isolated.to_code(cfg())
    assert managed.to_code(cfg(False)) == isolated.to_code(cfg(False))
    assert (
        managed.config_spec.cache_fingerprint_hash()
        == isolated.config_spec.cache_fingerprint_hash()
    )
    claims = []
    original = CuteDeviceFunctionState.claim_tcgen05_store_site

    def claim(state, value, store):
        answer = original(state, value, store)
        claims.append((store.name, answer[1]))
        return answer

    with patch.object(CuteDeviceFunctionState, "claim_tcgen05_store_site", claim):
        source = managed.to_code(cfg())
    assert claims and [final for _, final in claims[-3:]] == [False, False, True]
    assert source.count("PipelineTmaStore.create(") == 3
    assert source.count("producer_commit()") == source.count("producer_tail()") == 3
    assert (
        source.count("tcgen05_acc_pipeline.consumer_release(")
        == source.count("tcgen05_acc_consumer_state.advance()")
        == 1
    )
    copies = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "cute.copy"
        and n.args
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id.startswith("tcgen05_tiled_copy_t2r")
    ]
    assert len(copies) == 1
    assert "cute.where" in source and "!= cutlass.Uint8(0)" in source


def test_standalone_identity_default_contract_and_even_private_optin_stay_declined():
    fn = helion.kernel(standalone_identity.fn, settings=standalone_identity.settings)
    values = (
        torch.empty((256, 128), dtype=torch.bfloat16),
        torch.empty_strided((128, 128), (1, 128), dtype=torch.bfloat16),
        torch.empty((256, 128)),
    )
    bound = bind(fn, values)
    graphs = bound.host_function.device_ir.graphs
    anchors = {
        n
        for g in graphs
        for n in g.graph.nodes
        if n.target is torch.ops.aten.addmm.default
    }
    store = next(
        n for g in graphs for n in g.graph.nodes if n.target is memory_ops.store
    )
    value = store.args[2]
    assert isinstance(value, torch.fx.Node)
    with bound.env, bound.host_function:
        assert (
            epi._analyze_tcgen05_pointwise_dag(value, graphs, anchors, (256, 128))
            is None
        )
        assert (
            epi._analyze_tcgen05_pointwise_dag(
                value, graphs, anchors, (256, 128), allow_identity_output=True
            )
            is None
        )
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans


@pytest.mark.parametrize("position", (0, 1, 2))
def test_identity_and_computed_cast_IEEE_rounding_shared_memo_is_exact(position):
    bound = bind(make_kernel(), (*args(), position, False))
    plan, _, _ = plan_for(bound)
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    acc = (
        torch.tensor(
            [
                -0.0,
                0.0,
                1.00390625,
                -2.0,
                2**-126,
                float("inf"),
                -float("inf"),
                float("nan"),
            ]
        )
        .repeat(16)
        .reshape(8, 16)
    )
    raw_pred = (
        torch.tensor([0, 1, 2, 128, 255, 0, 2, 255], dtype=torch.uint8)
        .repeat(16)
        .reshape(8, 16)
    )
    raw_u = (
        torch.tensor(
            [0.0, -0.0, -1.0, 1.0, 2**-126, float("inf"), -float("inf"), float("nan")]
        )
        .reshape(8, 1)
        .expand(8, 16)
    )
    raw_scalar = torch.tensor(-0.0, dtype=torch.bfloat16)
    selected = torch.where(
        raw_pred.ne(0),
        raw_scalar.float(),
        acc
        + (raw_u + torch.tensor(1.0, dtype=torch.float32))
        * torch.tensor(0.5, dtype=torch.float32),
    )
    computed = selected.to(torch.bfloat16).float() * raw_pred.ne(0).float()
    expected = [selected, computed]
    expected.insert(position, acc)
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
            raw_u=raw_u,
            raw_pred=raw_pred,
            raw_scalar=raw_scalar,
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
        valid = ~torch.isnan(wanted)
        assert torch.equal(torch.signbit(actual[valid]), torch.signbit(wanted[valid]))


def test_duplicate_identity_outputs_are_not_a_new_shared_pure_gemm_subject():
    bound = bind(make_kernel(), (*args(), 0, True))
    plan, _, _ = plan_for(bound)
    assert plan is None
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans


@pytest.mark.parametrize(
    "fault",
    (
        "destination_alias",
        "input_alias",
        "no_registration",
        "fake_args",
        "missing_args",
        "bad_classifier",
        "bad_reusable_properties",
        "false_bound_fact",
    ),
)
def test_identity_optin_preserves_registered_readonly_and_disjoint_proof(fault):
    fn = make_kernel()
    values = args()
    bound = bind(fn, values)
    env = bound.env
    facts = dict(env.bound_runtime_input_specialization_results)
    classifiers = dict(env.runtime_input_specializations)
    runtime = dict(
        zip(fn.signature.parameters, fn.normalize_args(*values), strict=True)
    )
    env.bound_runtime_input_specialization_results = {}
    if fault == "destination_alias":
        runtime["second"] = runtime["identity"]
    elif fault == "input_alias":
        runtime["u"] = values[-3].view(-1)[:256]
    elif fault == "no_registration":
        env.runtime_input_specializations.pop(KEY)
    elif fault == "fake_args":
        with FakeTensorMode() as mode:
            runtime = {
                k: mode.from_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in runtime.items()
            }
    elif fault == "missing_args":
        runtime = {}
    elif fault == "bad_classifier":
        env.runtime_input_specializations[KEY] = replace(
            classifiers[KEY], classifier_identity=("unproved",)
        )
    elif fault == "bad_reusable_properties":
        env.runtime_input_specializations[KEY] = replace(
            classifiers[KEY], reusable_tensor_properties=frozenset()
        )
    else:
        env.bound_runtime_input_specialization_results[KEY] = tuple(
            False for _ in facts[KEY]
        )
    try:
        with env, bound.host_function, env.use_runtime_arg_values(runtime):
            before = dict(env.bound_runtime_input_specialization_results)
            registered = dict(env.runtime_input_specializations)
            coverage.register_epilogue_fanout_coverage(
                env, bound.host_function.device_ir
            )
            assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
            assert env.bound_runtime_input_specialization_results == before
            assert env.runtime_input_specializations == registered
    finally:
        env.bound_runtime_input_specialization_results = facts
        env.runtime_input_specializations = classifiers


def test_final_bound_proof_loss_is_refused_without_live_facts_or_early_identity_release():
    bound = bind(make_kernel(), args())
    saved = bound.env.bound_runtime_input_specialization_results.pop(KEY)
    try:
        with pytest.raises((helion.exc.BackendUnsupported, helion.exc.InvalidConfig)):
            bound.to_code(cfg())
        assert KEY not in bound.env.bound_runtime_input_specialization_results
    finally:
        bound.env.bound_runtime_input_specialization_results[KEY] = saved


@pytest.mark.parametrize(
    "fault",
    (
        "effect_gap",
        "device_if",
        "foreign_store",
        "foreign_graph",
        "mixed_layout",
        "mixed_dtype",
        "cycle",
        "budget",
    ),
)
def test_identity_boundary_never_proves_bad_effect_type_geometry_or_dag(fault):
    bound = bind(make_kernel(), args())
    plan, graphs, anchors = plan_for(bound)
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    graph = plan.stores[0].graph
    if fault in ("effect_gap", "device_if"):
        with graph.inserting_before(plan.stores[-1]):
            graph.call_function(
                _tracing_ops._if if fault == "device_if" else hl.barrier, ()
            )
    elif fault == "foreign_store":
        store = plan.stores[-1]
        scalar = next(
            n
            for n in graph.nodes
            if n.target is memory_ops.load and n.meta["val"].ndim == 0
        )
        store.args = (store.args[0], store.args[1], scalar, None)
    elif fault == "foreign_graph":
        foreign = graphs[0].graph
        with foreign.inserting_before(
            next(n for n in foreign.nodes if n.op == "output")
        ):
            foreign.call_function(memory_ops.store, plan.stores[0].args)
    elif fault in ("mixed_layout", "mixed_dtype"):
        tensor = plan.stores[-1].args[0]
        assert isinstance(tensor, torch.fx.Node)
        old = tensor.meta["val"]
        with bound.env:
            tensor.meta["val"] = (
                torch.empty_strided(
                    old.shape, (1, old.shape[0]), dtype=old.dtype, device=old.device
                )
                if fault == "mixed_layout"
                else torch.empty(old.shape, dtype=torch.float16, device=old.device)
            )
    else:
        prefix = next(
            leaf
            for chain in plan.chains
            for leaf in chain.auxiliary_tensor_loads
            if isinstance(leaf, epi._MVectorScalarPrefixLoadExpr)
        )
        add = next(
            n
            for n, _ in prefix.prefix.provenance
            if n.target is torch.ops.aten.add.Tensor
        )
        if fault == "cycle":
            add.args = (add, add.args[1])
        else:
            original = add.args[0]
            assert isinstance(original, torch.fx.Node)
            with graph.inserting_before(add):
                for _ in range(65):
                    current = graph.call_function(
                        torch.ops.aten.neg.default, (original,)
                    )
                    current.meta["val"] = original.meta["val"]
                    original = current
            add.args = (original, add.args[1])
    with bound.env, bound.host_function:
        stores = [
            n for g in graphs for n in g.graph.nodes if n.target is memory_ops.store
        ]
        assert (
            shared.prove_shared_dag_fanout(
                bound.env, bound.host_function, graphs, stores, anchors
            )
            is None
        )


@pytest.mark.parametrize("mode", ("flat", "grouped", "warp", "partial_tile"))
def test_identity_does_not_relax_shared_role_geometry_or_full_tile_guards(mode):
    bound = bind(make_kernel(), args())
    config = cfg()
    if mode == "flat":
        config.config["pid_type"] = "flat"
    elif mode == "grouped":
        config.config["tcgen05_grouped_mode"] = True
    elif mode == "warp":
        config.config["tcgen05_num_epi_warps"] = 0
    else:
        config.config["block_sizes"] = [128, 96, 64]
    with pytest.raises((helion.exc.BackendUnsupported, helion.exc.InvalidConfig)):
        bound.to_code(config)
