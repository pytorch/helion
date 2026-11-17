"""Check paired retirement configuration and allocation-wide synchronization."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import helion
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._testing import cpu_render_target
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_config

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_early_tmem_retirement"
SOURCE = (
    Path(__file__).parents[1]
    / "helion/_compiler/cute/_paged_attention_paired_pipeline.py"
)


def _specialize_retirement(enabled):
    class Specialize(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test == "cutlass.const_expr(early_tmem_retirement)":
                return node.body if enabled else node.orelse
            if test == "cutlass.const_expr(not early_tmem_retirement)":
                return node.orelse if enabled else node.body
            return self.generic_visit(node)

    return ast.fix_missing_locations(Specialize().visit(ast.parse(SOURCE.read_text())))


def calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == name
    ]


def expression_matches(node, source):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(source, mode="eval").body, include_attributes=False
    )


def final_join_before(body, free):
    joins = [
        node
        for node in body.body
        if isinstance(node, ast.Expr)
        and expression_matches(node.value, "cute.arch.barrier()")
    ]
    assert joins
    return max(node.lineno for node in joins) < free.lineno


def assert_protocol(source):
    tree = ast.parse(source)
    compute = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_compute_head"
    )
    body = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "paged_attention_paired_body"
    )
    arrivals = calls(tree, "barrier_cta_arrive")
    assert len(arrivals) == 1
    arrive = arrivals[0]
    assert ast.unparse(arrive) == "prims.barrier_cta_arrive(5, thread_count=288)"
    arrival_index = next(
        i
        for i, n in enumerate(compute.body)
        if isinstance(n, ast.Expr) and n.value is arrive
    )
    result_index = next(
        i
        for i, n in enumerate(compute.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
    )
    assert result_index < arrival_index
    assert calls(compute.body[result_index], "load_query_panel")
    assert "Tcgen05Wait.LOAD" in ast.unparse(compute.body[result_index + 1])
    assert "Tcgen05Fence.BEFORE_THREAD_SYNC" in ast.unparse(
        compute.body[arrival_index - 1]
    )
    # The actual nonserial final-PV drain must dominate the final O load.
    drains = [
        n
        for n in compute.body[:result_index]
        if isinstance(n, ast.If) and "probability_schedule !=" in ast.unparse(n.test)
    ]
    assert len(drains) == 1
    assert any(
        expression_matches(c, "_wait_non_mma(pv_done, (count - 1) & 1, sleep_waits)")
        for c in calls(drains[0], "_wait_non_mma")
    )
    # The seeded serial schedule drains each PV inside the loop before the
    # final output-panel read. Non-reference softmax uses this same path.
    loops = [
        node
        for node in compute.body[:result_index]
        if isinstance(node, ast.For) and calls(node, "online_softmax_per_key")
    ]
    assert len(loops) == 1
    serial = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.If)
        and expression_matches(
            node.test, 'cutlass.const_expr(probability_schedule == "serial")'
        )
    ]
    assert len(serial) == 1 and calls(loops[0], "online_softmax_per_key")
    assert any(
        expression_matches(c, "_wait_non_mma(pv_done, step & 1, sleep_waits)")
        for c in calls(serial[0], "_wait_non_mma")
    )
    tail = ast.Module(body=compute.body[arrival_index + 1 :], type_ignores=[])
    assert not any(
        calls(tail, name)
        for name in [
            "load_query_panel",
            "store_query_panel",
            "tcgen05_ld",
            "tcgen05_st",
            "tcgen05_mma",
        ]
    )
    wait = calls(tail, "griddepcontrol_wait")
    stores = calls(tail, "store_output_element") + calls(tail, "store_row_element")
    assert len(wait) == 1 and stores and all(wait[0].lineno < n.lineno for n in stores)
    owner = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "warp < 8"
    )
    assert len(calls(owner, "_compute_head")) == 1
    assert "head = cute.arch.make_warp_uniform(warp // 4)" in ast.unparse(owner)
    frees = calls(tree, "tcgen05_dealloc")
    assert len(frees) == 1 and len(calls(tree, "tcgen05_alloc")) == 1
    issuer = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "warp == 16"
        and calls(n, "tcgen05_dealloc")
    )
    assert (
        isinstance(issuer.body[0], ast.For)
        and ast.unparse(issuer.body[0].iter) == "cutlass.range(count)"
    )
    sync = next(n for n in issuer.body if calls(n, "barrier_cta_sync"))
    assert ast.unparse(sync) == "prims.barrier_cta_sync(5, thread_count=288)"
    assert sync.lineno < frees[0].lineno
    assert not final_join_before(body, frees[0])
    assert (
        ast.unparse(frees[0])
        == "prims.tcgen05_dealloc(cutlass.inttoptr(base, 6, cutlass.Int32), 512, group=prims.CTAGroup.CTA_1)"
    )


def test_enabled_retirement_preserves_complete_protocol():
    assert_protocol(ast.unparse(_specialize_retirement(True)))


def test_default_retirement_keeps_final_join_before_deallocation():
    tree = _specialize_retirement(False)
    assert not calls(tree, "barrier_cta_arrive")
    assert not calls(tree, "barrier_cta_sync")
    body = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "paged_attention_paired_body"
    )
    defaults = dict(
        zip(
            (arg.arg for arg in body.args.args[-len(body.args.defaults) :]),
            body.args.defaults,
            strict=True,
        )
    )
    assert ast.literal_eval(defaults["early_tmem_retirement"]) is False
    frees = calls(tree, "tcgen05_dealloc")
    assert len(frees) == 1 and final_join_before(body, frees[0])
    owners = [
        node
        for node in body.body
        if isinstance(node, ast.If)
        and expression_matches(node.test, "warp == 16")
        and frees[0] in ast.walk(node)
    ]
    assert len(owners) == 1
    assert (
        ast.unparse(frees[0])
        == "prims.tcgen05_dealloc(cutlass.inttoptr(base, 6, cutlass.Int32), 512, group=prims.CTAGroup.CTA_1)"
    )


def test_missing_retirement_and_explicit_false_share_config_and_source():
    bound = paired_v_lag_bound()
    absent = paired_v_lag_config(0)
    explicit = helion.Config.from_dict({**absent.config, KEY: False})
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        assert bound.config_spec.normalized_config(
            absent
        ) == bound.config_spec.normalized_config(explicit)
    with cpu_render_target(fake_device_sm_count=148):
        assert bound.to_code(absent) == bound.to_code(explicit)


def test_retirement_seeds_are_four_independent_paired_counterparts():
    bound = paired_v_lag_bound()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        planner = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        )
        seeds = planner.configs(planner.ranked())
        early = [s for s in seeds if s.get(KEY, False)]
        assert len(early) == 4 and seeds[-4:] == early
        assert [s.get("cute_mma_dag_v_lag", 0) for s in early] == [0, 1, 2, 3]
        for seed in early:
            assert seed.get("cute_mma_dag_pass_warps") == "attention_paired"
            assert seed.get("cute_pdl") is False
            assert not any(seed.get("load_eviction_policies"))
            counterpart = helion.Config.from_dict(
                {k: v for k, v in seed.config.items() if k != KEY}
            )
            assert counterpart in seeds[:-4]
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(KEY) is True
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
