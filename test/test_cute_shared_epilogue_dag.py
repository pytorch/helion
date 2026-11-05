"""CPU-only proof and source checks of complete same-layout shared DAG fanout."""

from __future__ import annotations

import ast
from dataclasses import replace
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_bool_aux_native import _args
from test.test_cute_bool_aux_native import _chains
from test.test_cute_bool_aux_native import _external_bool_epilogue
from test.test_cute_epilogue_dag import _analyze
from test.test_cute_epilogue_dag import _call
from test.test_cute_epilogue_dag import _constant
from test.test_cute_epilogue_dag import _evaluate
from test.test_cute_epilogue_dag import _input

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.epilogue_fanout import FanoutStore
from helion._compiler.cute.shared_epilogue_dag import SharedDagFanoutPlan
from helion._compiler.cute.shared_epilogue_dag import dag_keys
from helion._compiler.cute.shared_epilogue_dag import prove_shared_dag_fanout
from helion._compiler.cute.shared_epilogue_dag import raw_aux_key
from helion._compiler.cute.shared_epilogue_dag import render_dag
from helion._compiler.cute.shared_epilogue_dag import schedule_supported
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.exc import InvalidConfig
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _external_two(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    extra: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor,
) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        rounded = (acc + residual[row, col]).to(torch.bfloat16).to(torch.float32)
        branch = rounded * extra[row, col]
        out0[row, col] = rounded
        out1[row, col] = branch


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _external_three(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    extra: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor,
) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        rounded = (acc + residual[row, col]).to(torch.bfloat16).to(torch.float32)
        branch = rounded * extra[row, col]
        out0[row, col] = rounded
        out1[row, col] = branch
        out2[row, col] = rounded + branch


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _external_four(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    extra: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor,
) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        rounded = (acc + residual[row, col]).to(torch.bfloat16).to(torch.float32)
        branch = rounded * extra[row, col]
        out0[row, col] = rounded
        out1[row, col] = branch
        out2[row, col] = rounded + branch
        out3[row, col] = branch * 3.0


@pytest.mark.parametrize("count", (2, 3, 4))
def test_two_to_four_external_stores_retain_original_casts_and_complete_scope(count):
    shape = (128, 128)
    args = (
        torch.empty(shape, dtype=torch.bfloat16),
        torch.empty(shape, dtype=torch.bfloat16),
        torch.empty(shape, dtype=torch.float32),
        torch.empty(shape, dtype=torch.float32),
        *(torch.empty(shape, dtype=torch.float32) for _ in range(4)),
    )
    kernel = {2: _external_two, 3: _external_three, 4: _external_four}[count]
    bound = cpu_bind(kernel, args)
    plans = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert len(plans) == 1 and isinstance(plans[0], SharedDagFanoutPlan)
    assert len(plans[0].stores) == count
    state = bound.config_spec._cute_tcgen05_config
    # Nonbool 2–4 external subjects discover sharing from their own complete
    # graph/witnesses; the tagged-bool compact-seed producer stays uninvolved.
    assert state._readonly_bool_aux_seed_subject is None
    assert state._multi_output_seed_configs() == []
    assert state.epilogue_fanout_search_enabled
    groups = [
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == FANOUT_CONFIG_KEY
    ]
    assert len(groups) == 1 and tuple(groups[0].domain) == ("off", "shared")
    assert {witness.value for witness in groups[0].witnesses} == {"off", "shared"}
    raw = configuration()
    raw.config["tcgen05_aux_load_placement"] = "post_acc_wait"
    source = bound.to_code(raw)
    ast.parse(source)
    assert source.count("'kind': 'tcgen05_d_tma'") == count
    copies = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.copy"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id.startswith("tcgen05_tiled_copy_t2r")
    ]
    assert len(copies) == 1
    assert source.count("PipelineTmaStore.create(") == count
    assert source.count(".to(cutlass.BFloat16).to(cutlass.Float32)") >= 1


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def bound_original(**kwargs):
    return cpu_bind(_external_bool_epilogue, _args(**kwargs))


def plan_for(bound):
    graphs, anchor, chains = _chains(bound)
    assert chains is not None
    with bound.env, bound.host_function:
        plan = prove_shared_dag_fanout(
            bound.env,
            bound.host_function,
            graphs,
            [store for store, _ in chains],
            {anchor},
        )
    return plan, graphs, anchor


def configuration(*, shared=True, scheduler=True):
    config = helion.Config(
        block_sizes=[128, 32, 64],
        pid_type="persistent_blocked",
        tcgen05_strategy="role_local_with_scheduler"
        if scheduler
        else "role_local_monolithic",
        tcgen05_warp_spec_scheduler_warps=1 if scheduler else 0,
        tcgen05_ab_stages=6,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    if shared:
        config.config[FANOUT_CONFIG_KEY] = "shared"
    return config


def test_exact_four_output_external_source_one_accumulator_unique_aux_all_pipelines():
    bound = bound_original()
    plan, _, _ = plan_for(bound)
    assert isinstance(plan, SharedDagFanoutPlan)
    assert len(plan.stores) == 4 and plan.output_dtype is torch.float32
    assert plan.pre_wait_aux_safe
    off = bound.to_code(configuration(shared=False))
    shared = bound.to_code(configuration())
    ast.parse(shared)
    tmem_copies = [
        node
        for node in ast.walk(ast.parse(shared))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.copy"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id.startswith("tcgen05_tiled_copy_t2r")
    ]
    assert len(tmem_copies) == 1
    assert shared.count("cute.autovec_copy(") == 5
    assert shared.count("cutlass.Uint8(0)") == 1
    assert off.count("cutlass.Uint8(0)") == 4
    for token, count in (
        ("PipelineTmaStore.create(", 4),
        ("producer_commit()", 4),
        ("producer_tail()", 4),
        ("'kind': 'tcgen05_d_tma'", 4),
        (".consumer_release(tcgen05_acc_consumer_state)", 1),
        ("tcgen05_acc_consumer_state.advance()", 1),
    ):
        assert shared.count(token) == count, token
        assert shared.count(token) == off.count(token)
    # The original per-output visibility barriers and D copies remain distinct.
    assert shared.count(".arrive_and_wait()") == off.count(".arrive_and_wait()")
    assert shared.count("cute.arch.fence_view_async_shared()") == off.count(
        "cute.arch.fence_view_async_shared()"
    )
    assert "cutlass.Float32(768.0)" in shared
    assert "cutlass.Float32(1.1111111111111112)" in shared


def _copy_store(store: FanoutStore) -> FanoutStore:
    main = ast.parse(ast.unparse(store.main)).body[0]
    assert isinstance(main, ast.If)
    return replace(store, main=main)


@pytest.fixture(scope="module")
def shared_store_records():
    from helion._compiler.cute import shared_epilogue_dag as shared

    records = []
    original = shared.combine_dag_stores

    def collect(stores):
        records.extend(_copy_store(store) for store in stores)
        return original(stores)

    with cpu_codegen_target(), forbid_cuda_work():
        bound = bound_original()
        with patch.object(shared, "combine_dag_stores", collect):
            bound.to_code(configuration())
    assert len(records) == 4
    return records


@pytest.mark.parametrize(
    "fault",
    (
        "empty",
        "missing",
        "reversed",
        "commit",
        "same_commit",
        "acquire",
        "r2s",
        "tma",
        "early_release",
        "early_advance",
        "late_advance",
        "stages",
        "aux_source",
        "provenance",
    ),
)
def test_complete_pipeline_lifetime_and_source_proof_failure_agreement(
    shared_store_records, fault
):
    from helion._compiler.cute import shared_epilogue_dag as shared
    from helion.exc import BackendUnsupported

    stores = [_copy_store(store) for store in shared_store_records]
    if fault == "empty":
        stores = []
    elif fault == "missing":
        stores.pop()
    elif fault == "reversed":
        stores.reverse()
    elif fault == "commit":
        stores[1] = replace(stores[1], commit="")
    elif fault == "acquire":
        stores[1] = replace(stores[1], acquire="")
    elif fault == "r2s":
        stores[1] = replace(stores[1], r2s="")
    elif fault == "tma":
        stores[1] = replace(stores[1], tma="")
    elif fault == "same_commit":
        stores[1] = replace(stores[1], commit=stores[0].commit)
    elif fault == "early_release":
        stores[0] = replace(
            stores[0],
            iteration=replace(
                stores[0].iteration, release=stores[-1].iteration.release
            ),
        )
    elif fault == "early_advance":
        stores[0] = replace(stores[0], advance=stores[-1].advance)
    elif fault == "late_advance":
        stores[-1] = replace(stores[-1], advance="")
    elif fault == "stages":
        stores[1] = replace(stores[1], c_stages=4)
    elif fault == "aux_source":
        stores[1] = replace(
            stores[1], iteration=replace(stores[1].iteration, auxiliary_load_sources=())
        )
    elif fault == "provenance":
        rendered = stores[1].iteration.chain
        dag = rendered.chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        step = replace(rendered.chain.steps[0], expr=replace(dag, provenance=()))
        bad_chain = replace(rendered, chain=replace(rendered.chain, steps=(step,)))
        stores[1] = replace(
            stores[1], iteration=replace(stores[1].iteration, chain=bad_chain)
        )
    with pytest.raises(BackendUnsupported):
        shared.combine_dag_stores(stores)


@pytest.mark.parametrize("shared", (False, True))
def test_no_fit_and_ragged_backstops_remain_strict(shared):
    bound = bound_original()
    ragged = configuration(shared=shared)
    ragged.config["block_sizes"] = [96, 32, 64]
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(ragged)
    oversized = configuration(shared=shared)
    oversized.config.update(
        block_sizes=[128, 128, 64], tcgen05_ab_stages=6, tcgen05_c_stages=4
    )
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(oversized)


@pytest.mark.parametrize(
    "field,value",
    (
        ("tcgen05_epi_warp_groups", 2),
        ("tcgen05_cluster_m", 2),
        ("tcgen05_aux_load_mode", "tma"),
        ("tcgen05_aux_prefetch_subtiles", 1),
        ("tcgen05_aux_fragment_hoist", True),
        ("tcgen05_c_store_mode", "direct"),
        ("tcgen05_grouped_mode", "worklist_nm"),
    ),
)
def test_new_plan_declines_uncomposed_protocols(field, value):
    bound = bound_original()
    plan, _, _ = plan_for(bound)
    assert plan is not None
    config = configuration()
    config.config[field] = value
    assert not schedule_supported(plan, config.config)
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(config)


def test_second_host_grid_root_cannot_escape_complete_effect_scope():
    bound = bound_original()
    plan, graphs, anchor = plan_for(bound)
    assert plan is not None
    host = bound.host_function
    body = list(host.body)
    root = next(node for node in body if isinstance(node, ast.For))
    try:
        host.body.append(root)
        with bound.env, host:
            assert (
                prove_shared_dag_fanout(bound.env, host, graphs, plan.stores, {anchor})
                is None
            )
    finally:
        host.body[:] = body


@pytest.mark.parametrize(
    "fault",
    (
        "device_if",
        "nested_loop",
        "barrier",
        "non_store_effect",
        "foreign_ssa",
        "other_root",
    ),
)
def test_between_store_control_effect_and_foreign_ssa_scope_declines(fault):
    from helion._compiler.cute import shared_epilogue_dag as shared
    from helion.language import _tracing_ops

    bound = bound_original()
    plan, graphs, anchor = plan_for(bound)
    assert plan is not None
    graph = plan.stores[0].graph
    assert shared._store_interval_is_pure(graph, plan.stores)
    if fault == "other_root":
        value = plan.stores[-1].args[2]
        assert isinstance(value, torch.fx.Node)
        foreign = torch.fx.Graph()
        other = foreign.call_function(torch.ops.aten.neg.default, (value,))
        other.meta.update(value.meta)
        args = list(plan.stores[-1].args)
        args[2] = other
        plan.stores[-1].args = tuple(args)
    else:
        with graph.inserting_before(plan.stores[1]):
            if fault == "nested_loop":
                loop = next(
                    node
                    for node in graph.nodes
                    if _tracing_ops.is_for_loop_target(node.target)
                )
                added = graph.call_function(loop.target, loop.args, loop.kwargs)
                added.meta.update(loop.meta)
            elif fault == "device_if":
                graph.call_function(_tracing_ops._if, ())
            elif fault == "barrier":
                graph.call_function(hl.barrier, ())
            elif fault == "non_store_effect":
                graph.call_function(torch.ops.aten._assert_async.msg, ())
            elif fault == "foreign_ssa":
                foreign = torch.fx.Graph()
                other = foreign.placeholder("foreign")
                value = plan.stores[0].args[2]
                assert isinstance(value, torch.fx.Node)
                other.meta.update(value.meta)
                added = graph.call_function(torch.ops.aten.add.Tensor, (value, other))
                added.meta.update(value.meta)
        assert not shared._store_interval_is_pure(graph, plan.stores)
    with bound.env, bound.host_function:
        assert (
            prove_shared_dag_fanout(
                bound.env, bound.host_function, graphs, plan.stores, {anchor}
            )
            is None
        )


def test_nested_expression_fails_closed_with_a_provenance_decline():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    square = _call(graph, torch.ops.aten.mul.Tensor, (anchor, anchor))
    analysis = _analyze(square, anchor)
    assert analysis is not None
    dag = analysis[0].steps[0].expr
    assert isinstance(dag, epi._PointwiseDagExpr)
    nested = epi._PointwiseDagExpr((dag,), ((anchor, torch.float32),))
    with pytest.raises(ValueError, match="nested"):
        dag_keys(nested)


@pytest.mark.parametrize("scheduler", (False, True))
def test_structural_off_shared_surface_and_strict_normalization(scheduler):
    bound = bound_original()
    plan, _, _ = plan_for(bound)
    assert plan is not None
    assert schedule_supported(plan, configuration(scheduler=scheduler).config)
    assert (
        bound.config_spec.normalized_config(configuration(scheduler=scheduler)).config[
            FANOUT_CONFIG_KEY
        ]
        == "shared"
    )
    # Coverage is discovered by the complete graph, not a case ID or coefficient.
    state = bound.config_spec._cute_tcgen05_config
    assert state.epilogue_fanout_plans and state.epilogue_fanout_search_enabled
    groups = bound.config_spec.compiler_coverage_groups
    assert any(
        group.key == FANOUT_CONFIG_KEY and tuple(group.domain) == ("off", "shared")
        for group in groups
    )


@pytest.mark.parametrize("fault", ("outputs", "residual", "boolean_output"))
def test_real_overlapping_storage_declines_all_output_sharing(fault):
    bound = bound_original(alias=fault)
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(configuration())


def test_fake_only_and_missing_or_contradictory_bound_alias_facts_fail_closed():
    args = _args()
    with FakeTensorMode() as mode:
        fake = tuple(mode.from_tensor(value) for value in args)
    bound = cpu_bind(_external_bool_epilogue, fake)
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    bound = cpu_bind(_external_bool_epilogue, args)
    plan, graphs, anchor = plan_for(bound)
    assert plan is not None
    key = "tensor_storage_disjoint_matrix_v1"
    original = bound.env.bound_runtime_input_specialization_results.pop(key)
    try:
        with bound.env, bound.host_function:
            assert (
                prove_shared_dag_fanout(
                    bound.env, bound.host_function, graphs, plan.stores, {anchor}
                )
                is None
            )
        from helion.exc import BackendUnsupported

        with pytest.raises(BackendUnsupported):
            bound.to_code(configuration())
        bound.env.bound_runtime_input_specialization_results[key] = tuple(
            False for _ in original
        )
        with bound.env, bound.host_function:
            assert (
                prove_shared_dag_fanout(
                    bound.env, bound.host_function, graphs, plan.stores, {anchor}
                )
                is None
            )
    finally:
        bound.env.bound_runtime_input_specialization_results[key] = original
    with bound.env, bound.host_function:
        assert (
            prove_shared_dag_fanout(
                bound.env, bound.host_function, graphs, plan.stores, {anchor}
            )
            is not None
        )


@pytest.mark.parametrize(
    "mode",
    ("extra_mask", "offset_index", "transpose", "bool_half", "float_bool", "sin"),
)
def test_unproved_access_cast_effect_and_operation_never_gain_shared_mode(mode):
    bound = cpu_bind(_external_bool_epilogue, (*_args(), mode))
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(configuration())


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "duplicate",
        "mask",
        "dtype",
        "shape",
        "extra_effect",
        "depth",
        "whole_node_bound",
    ),
)
def test_complete_store_identity_shape_effect_and_whole_dag_bounds(fault):
    from helion.language import memory_ops

    bound = bound_original()
    plan, graphs, anchor = plan_for(bound)
    assert plan is not None
    stores = list(plan.stores)
    graph = stores[0].graph
    if fault == "missing":
        stores.pop()
    elif fault == "duplicate":
        stores[-1] = stores[0]
    elif fault == "mask":
        stores[0].kwargs = {"extra_mask": True}
    elif fault in ("dtype", "shape"):
        output = stores[-1].args[0]
        assert isinstance(output, torch.fx.Node)
        shape = plan.shape if fault == "dtype" else (plan.shape[0], plan.shape[1] + 1)
        output.meta["val"] = torch.empty(
            shape, dtype=torch.float16 if fault == "dtype" else torch.float32
        )
    elif fault == "extra_effect":
        with graph.inserting_before(stores[0]):
            effect = graph.call_function(memory_ops.store, stores[0].args)
        effect.meta.update(stores[0].meta)
    elif fault in ("depth", "whole_node_bound"):
        for store in stores[:1] if fault == "depth" else stores:
            value = store.args[2]
            assert isinstance(value, torch.fx.Node)
            with graph.inserting_before(store):
                for _ in range(40 if fault == "depth" else 20):
                    following = graph.call_function(
                        torch.ops.aten.neg.default, (value,)
                    )
                    following.meta.update(value.meta)
                    value = following
            args = list(store.args)
            args[2] = value
            store.args = tuple(args)
    with bound.env, bound.host_function:
        assert (
            prove_shared_dag_fanout(
                bound.env, bound.host_function, graphs, stores, {anchor}
            )
            is None
        )


def test_typed_fx_provenance_child_refs_missing_origins_and_signed_zero():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    zero = _constant(graph, -0.0)
    product = _call(graph, torch.ops.aten.mul.Tensor, (anchor, zero))
    square = _call(graph, torch.ops.aten.mul.Tensor, (product, product))
    analysis = _analyze(square, anchor)
    assert analysis is not None
    dag = analysis[0].steps[0].expr
    assert isinstance(dag, epi._PointwiseDagExpr)
    keys = dag_keys(dag)
    assert len(keys) == len(dag.nodes)
    with pytest.raises(ValueError, match="provenance"):
        dag_keys(replace(dag, provenance=()))
    origins = list(dag.provenance)
    origins[-1] = (origins[-1][0], torch.bfloat16)
    assert keys != dag_keys(replace(dag, provenance=tuple(origins)))
    # A separately originating identical printed operation is not the same FX value.
    other = _call(graph, torch.ops.aten.mul.Tensor, (product, product))
    other_analysis = _analyze(other, anchor)
    assert other_analysis is not None
    other_dag = other_analysis[0].steps[0].expr
    assert isinstance(other_dag, epi._PointwiseDagExpr)
    assert dag_keys(dag)[-1] != dag_keys(other_dag)[-1]


@pytest.mark.parametrize("byte", (0, 1, 2, 128, 255))
def test_four_original_outputs_exact_fp32_association_ieee_and_nonzero_bytes(byte):
    bound = bound_original()
    plan, _, _ = plan_for(bound)
    assert plan is not None
    acc = torch.linspace(-2, 2, 128 * 128).reshape(128, 128)
    acc[0, :4] = torch.tensor([-0.0, float("inf"), -float("inf"), float("nan")])
    residual0 = torch.full_like(acc, -0.0)
    residual1 = torch.full_like(acc, -0.0)
    residual2 = torch.full_like(acc, 0.3)
    row = torch.linspace(0.5, 1.5, 128)
    raw = torch.full(acc.shape, byte, dtype=torch.uint8)
    leaves = {}
    for chain in plan.chains:
        for leaf in chain.auxiliary_tensor_loads:
            leaves.setdefault(raw_aux_key(leaf), leaf)
    assert len(leaves) == 5
    values = {}
    names = {}
    # Exact original raw load sites: source argument identities, not text CSE.
    for leaf in leaves.values():
        host_node = leaf.load_node.args[0]
        assert isinstance(host_node, torch.fx.Node)
        host_name = host_node.args[0]
        assert isinstance(host_name, str)
        name = f"leaf_{len(names)}"
        names[raw_aux_key(leaf)] = name
        values[name] = {
            "residual0": residual0,
            "residual1": residual1,
            "boolean": raw,
            "scale": row,
            "residual2": residual2,
        }[host_name]
    factory_count = itertools.count()

    def factory(prefix):
        return f"{prefix}_{next(factory_count)}"

    memo = {}
    source = ""
    results = []
    for chain in plan.chains:
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        aux: epi._LeafBindings = {
            leaf: names[raw_aux_key(leaf)] for leaf in chain.auxiliary_tensor_loads
        }
        prelude, result = render_dag(dag, "original_acc", aux, factory, memo, "")
        source += prelude
        results.append(result)
    assert source.count("cutlass.Uint8(0)") == 1
    masked = ((residual0 + acc) + residual1) * (
        (raw != 0).float() * torch.tensor(1.1111111111111112, dtype=torch.float32)
    )
    expected = (
        masked,
        masked * row,
        (masked * row) * torch.tensor(768.0, dtype=torch.float32),
        masked * residual2,
    )
    for expression, reference in zip(results, expected, strict=True):
        actual = _evaluate(
            source,
            expression,
            acc,
            cutlass=SimpleNamespace(
                Float32=torch.float32,
                Float16=torch.float16,
                BFloat16=torch.bfloat16,
                Uint8=torch.uint8,
            ),
            **values,
        )
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(reference)
        assert torch.equal(
            actual[finite].view(torch.int32), reference[finite].view(torch.int32)
        )
