from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import math
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import _Matcher
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_bounded import bounded_merge_args
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_bounded import bounded_partial_merge

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(query_len):
    kernel, args = bounded_partial_fixture(query_len, reverse_pages=query_len != 1)
    args = (*args[:7], 31 if query_len == 4 else None, *args[8:])
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return bound, program


@pytest.mark.parametrize("query_len,columns", [(1, 8), (4, 32), (5, 40)])
def test_direct_capped_raw_statistics_have_paired_source_route(query_len, columns):
    bound, program = _bound(query_len)
    assert bound.host_function is not None
    plan = match_paged_attention(program)
    assert plan is not None and plan.coefficients_capped
    assert plan.partial_statistics is not None and plan.lse is None
    assert plan.score_base2 and plan.score_reference is None
    assert plan.output.dtype == torch.float32
    assert (plan.coordinates.reflected is not None) == (query_len != 1)
    assert (plan.key.post_scale_log2, plan.value.post_scale_log2) == (-6, -22)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
        with bound.env, bound.host_function:
            transport = plan_paged_attention_transport(bound.env, plan)
            assert transport is not None and transport.direct_io is not None
            assert transport.direct_io.partial_statistics == (
                (query_len * 16, 16, 1),
                (query_len * 16, 16, 1),
            )
            config.config["cute_mma_dag_pass_warps"] = "attention"
            with pytest.raises(helion.exc.InvalidConfig, match="paired"):
                bound.config_spec.normalized_config(config)
    assert "paged_attention_paired_body(" in source
    for fragment in (
        f"query_columns={columns}",
        "base2_scores=True",
        "coefficients_capped=True",
        "maximum_output=maximum_storage",
        "denominator_output=denominator_storage",
        "statistic_layouts=",
    ):
        assert fragment in source
    assert "reference_scores=True" not in source


def test_raw_statistics_capture_contract_retains_all_endpoint_reads_and_stores():
    bound, _ = _bound(1)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[8],
        cute_mma_dag_pass_warps="attention_paired",
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    (plans,) = [
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    ]
    assert plans[0]["capture_pdl_contract"] == "paired-output-stores-v1"
    alias = plans[1]
    assert {
        name
        for name, loaded in zip(
            alias["tensor_names"], alias["capture_loaded"], strict=True
        )
        if loaded
    } == {
        "query_storage",
        "k_subpages",
        "v_subpages",
        "k_scale_subpages",
        "v_scale_subpages",
        "page_table",
        "lengths",
        "segment_begin",
        "segment_end",
    }
    assert {
        name
        for name, stored in zip(alias["tensor_names"], alias["stored"], strict=True)
        if stored
    } == {"output_storage", "maximum_storage", "denominator_storage"}


def test_raw_statistics_decline_unproved_packed_row_layouts():
    kernel, args = optional_lse_fixture("per_key", True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            plan = match_paged_attention(program)
            assert plan is not None and plan.lse is not None
            assert plan.query.fake.ndim == 3 and plan.query.index_map is None
            assert plan_paged_attention_transport(bound.env, plan) is not None
            raw = dataclasses.replace(
                plan,
                output=dataclasses.replace(plan.output, fake=plan.output.fake.float()),
                lse=None,
                coefficients_capped=True,
                score_base2=True,
                partial_statistics=(plan.lse, plan.lse),
            )
            # The packed query, output and row layouts satisfy all the older
            # transport checks; only this unimplemented raw-mode gate declines.
            assert plan_paged_attention_transport(bound.env, raw) is None


def test_raw_statistics_same_destination_preserves_source_store_order_by_declining():
    kernel, args = bounded_partial_fixture(
        1, alias_statistics=True, reverse_pages=False
    )
    args = (*args[:7], None, *args[8:])
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            program = _Matcher(graphs, None).match()
            assert program is not None
            assert match_paged_attention(program) is None
            # Public extraction includes this semantic gate for mapped I/O.
            assert match_mma_dag(graphs) is None
    stores = [
        target.value.id
        for node in ast.walk(ast.parse(inspect.getsource(kernel.fn)))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id in ("maximum_storage", "denominator_storage")
    ]
    assert stores == ["denominator_storage", "maximum_storage"]


@pytest.mark.parametrize("alias", ["raw_maximum", "raw_denominator", "statistics"])
def test_raw_statistics_require_three_independent_output_storages(alias):
    _, original = _bound(1)
    program = copy_paged_attention_program(original)
    stores = list(program.vec_stores)
    statistic = 0 if alias == "raw_maximum" else 1
    access = stores[statistic].access
    storage = (
        stores[0].access.fake
        if alias == "statistics"
        else program.stores[0].access.fake
    )
    view = storage.as_strided(access.fake.shape, access.fake.stride())
    stores[statistic] = dataclasses.replace(
        stores[statistic], access=dataclasses.replace(access, fake=view)
    )
    program = dataclasses.replace(program, vec_stores=tuple(stores))
    assert match_paged_attention(program) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "p_bound",
        "alpha_bound",
        "p_dtype",
        "alpha_dtype",
        "p_unclamped",
        "alpha_unclamped",
        "raw_identity",
        "maximum_identity",
        "denominator_identity",
        "extra_store",
        "stat_dtype",
        "stat_axes",
    ],
)
def test_raw_statistics_proof_declines_nearby_unimplemented_sources(mutation):
    # Independent mutations cover coefficient value/rounding, each output
    # identity, complete write effects, and row-coordinate/type contracts.
    _, original = _bound(5)
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None and plan.partial_statistics is not None
    probability = plan.probability
    alpha = next(
        node
        for node in probability.graph.nodes
        if node.target is torch.ops.aten.clamp_max.default
        and node.meta["val"].ndim == 1
    )
    if mutation in ("p_bound", "alpha_bound"):
        node = probability if mutation == "p_bound" else alpha
        node.args = (node.args[0], 2.0)
    elif mutation in ("p_dtype", "alpha_dtype"):
        node = probability if mutation == "p_dtype" else alpha
        node.meta["val"] = node.meta["val"].to(torch.bfloat16)
    elif mutation in ("p_unclamped", "alpha_unclamped"):
        node = probability if mutation == "p_unclamped" else alpha
        assert isinstance(node.args[0], torch.fx.Node)
        node.replace_all_uses_with(node.args[0])
    elif mutation == "raw_identity":
        program = dataclasses.replace(
            program,
            stores=(
                dataclasses.replace(program.stores[0], value=plan.denominator.phi),
            ),
        )
    elif mutation == "maximum_identity":
        program = dataclasses.replace(
            program,
            vec_stores=tuple(
                dataclasses.replace(store, value=plan.denominator_final)
                if store.access is plan.partial_statistics[0]
                else store
                for store in program.vec_stores
            ),
        )
    elif mutation == "denominator_identity":
        plan.denominator_final.args = (plan.denominator_final.args[0], [0])
    elif mutation == "extra_store":
        program = dataclasses.replace(
            program, vec_stores=(*program.vec_stores, program.vec_stores[0])
        )
    elif mutation in ("stat_dtype", "stat_axes"):
        store = program.vec_stores[0]
        access = store.access
        if mutation == "stat_dtype":
            access = dataclasses.replace(access, fake=access.fake.to(torch.bfloat16))
        else:
            assert access.index_map is not None
            access = dataclasses.replace(
                access,
                index_map=dataclasses.replace(access.index_map, axes=()),
            )
        program = dataclasses.replace(
            program,
            vec_stores=(
                dataclasses.replace(store, access=access),
                *program.vec_stores[1:],
            ),
        )
    else:
        raise AssertionError(mutation)
    assert match_paged_attention(program) is None


def test_written_partial_reference_executes_empty_full_attention_segments():
    kernel, initial = bounded_partial_fixture(1)
    args = (*initial[:7], None, *initial[8:])
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    q.zero_()
    k.zero_()
    v.fill_(64)
    ks.fill_(127)
    vs.fill_(127)
    pages.zero_()
    lengths.copy_(lengths.new_tensor([256, 257]))
    # CPU has no aten.mm.dtype implementation. Empty segments exercise the
    # actual reference context and source outputs without replacing that op;
    # nonempty numerical agreement is covered on CUDA.
    cast("torch.Tensor", args[11]).fill_(3)
    with cpu_target():
        eager = helion.kernel(
            kernel.fn, backend="cute", ref_mode=helion.RefMode.EAGER
        ).bind(args)
        raw, maximum, denominator = eager.run_ref(*args)
    assert torch.isneginf(maximum).all()
    torch.testing.assert_close(
        denominator, torch.zeros_like(denominator), rtol=0, atol=0
    )
    torch.testing.assert_close(raw, torch.zeros_like(raw), rtol=0, atol=0)


def test_written_merge_retains_partition_mass_and_exceptional_arithmetic():
    # The first two partial LSEs round to the same FP32 number, despite their
    # different masses. Separate m/l retains the exact weighted result 16.
    args = bounded_merge_args()
    raw0, maximum0, denominator0, raw1, maximum1, denominator1 = args
    partial_lse0 = maximum0[0] * math.log(2) + torch.log(denominator0[0])
    partial_lse1 = maximum1[0] * math.log(2) + torch.log(denominator1[0])
    torch.testing.assert_close(partial_lse0, partial_lse1, rtol=0, atol=0)
    with cpu_target():
        eager = helion.kernel(
            bounded_partial_merge.fn, backend="cute", ref_mode=helion.RefMode.EAGER
        ).bind(args)
        output, lse = eager.run_ref(*args)
    maximum_decoded = 254.0 * 2.0**120
    expected = (
        torch.tensor(
            [16, 0, math.nan, math.inf, maximum_decoded, -maximum_decoded, math.nan],
            dtype=torch.bfloat16,
        )
        .reshape(7, 1, 1, 1)
        .expand_as(output)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.isneginf(lse[1]).all()
