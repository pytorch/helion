from __future__ import annotations

import ast
from copy import deepcopy
import itertools
import random
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from test.test_cute_grouped_coverage_search import _search
from test.test_cute_grouped_k128_full_coverage import _bound
from test.test_cute_grouped_row_union_cluster4 import cpu_only as cpu_only
from test.test_cute_grouped_row_union_cluster4 import wrapper
from test.test_cute_shared_rhs_grouped import _plans

from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_paired_clc_carrier,
)
from helion._compiler.cute.grouped_row_union import CONFIG_KEY
from helion._compiler.cute.grouped_row_union import PAIRED_CLC
from helion._compiler.cute.grouped_row_union import PAIRED_CLC_SCHEDULE
from helion._compiler.cute.grouped_row_union import SCHEDULE_KEY
from helion._compiler.cute.grouped_row_union import STARTUP_PREFILL_KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED
from helion._compiler.cute.grouped_row_union import GroupedRowUnionPlan
from helion._compiler.cute.tcgen05_constants import TCGEN05_CLC_QUERY_PACING_CONFIG_KEY
from helion._compiler.program_id import ProgramIDs
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig


@pytest.fixture(scope="module")
def case():
    bound, args = _bound(10240, 1024, 1024, 16)
    with bound.env:
        config = grouped_row_union_paired_clc_carrier(
            bound.env, bound.host_function.device_ir
        )
    assert config is not None
    config.config[STARTUP_PREFILL_KEY] = True
    source = bound.to_code(config)
    return bound, args, config, source


def device(source):
    return next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )


@skipUnlessBackends(["cute"])
def test_original_ordinary_caller_profile_and_queues(case):
    bound, _, config, source = case
    before = deepcopy(config)
    normalized = bound.config_spec.normalized_config(config)
    assert normalized[STARTUP_PREFILL_KEY] is True
    assert normalized[SCHEDULE_KEY] == PAIRED_CLC_SCHEDULE
    with bound.env:
        projected = bound.env.backend.codegen_config(normalized)
    assert projected.block_sizes == [192, 256, 64]
    assert config == before
    ab, out = _plans(source)
    assert (ab["bm"], ab["bn"], ab["bk"], ab["ab_stage_count"]) == (256, 192, 64, 7)
    assert (ab["lhs_idx"], ab["rhs_idx"], out["d_idx"]) == (2, 1, 3)
    assert (out["epi_tile_m"], out["epi_tile_n"], out["c_stage_count"]) == (128, 32, 2)
    rendered = ast.unparse(wrapper(source))
    assert "make_tiled_tma_atom_A" in rendered and "make_tiled_tma_atom_B" in rendered
    assert ast.unparse(ast.parse(source)).count("out = torch.empty(") == 1
    assert "block=(32, 8, 1)" in source
    assert "cutlass.Int32(384)" in source
    assert "setmaxregister_" not in source
    assert "StaticPersistentTileScheduler" not in source
    # Both issuer queues commit even when the final row packet is empty.
    commits = [
        n
        for n in ast.walk(device(source))
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Expr) and ".producer_commit()" in ast.unparse(x)
            for x in n.body
        )
    ]
    assert len(commits) == 1
    issuer = commits[0]
    assert "Int32(0)" in ast.unparse(issuer.test) and "Int32(2)" in ast.unparse(
        issuer.test
    )
    assert isinstance(issuer.body[0], ast.If)
    assert "Int32(10240)" in ast.unparse(issuer.body[0].test)
    assert "Int32(32)" in ast.unparse(issuer.body[0].test)
    assert source.count("tcgen05_c_pipeline.producer_tail()") == 1


@skipUnlessBackends(["cute"])
def test_startup_is_private_before_publication_and_resume_is_once(case):
    fn = device(case[3])
    prefix = next(
        n
        for n in fn.body
        if isinstance(n, ast.If) and "tcgen05_prefill_state =" in ast.unparse(n)
    )
    # The prologue publication is a named barrier the TMA-load warp skips
    # (the startup prefill above runs ahead of it on that warp anyway).
    publish = next(
        n
        for n in fn.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not tcgen05_tma_warp"
        and "tcgen05_row_union_publication_barrier.arrive_and_wait()"
        in ast.unparse(n.body[0])
    )
    assert not any(
        isinstance(n, ast.Expr) and ast.unparse(n) == "cute.arch.sync_threads()"
        for n in fn.body
    )
    assert fn.body.index(prefix) < fn.body.index(publish)
    assert "griddepcontrol_wait" in ast.unparse(prefix.body[0])
    # The throttled CLC scheduler's credit for the prefilled first tile follows
    # the dependent-launch wait; the private prefill state comes after it.
    assert ast.unparse(prefix.body[1]) == (
        "if cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == cutlass.Int32(0):"
        "\n    tcgen05_clc_throttle_barrier.arrive()"
    )
    assert ast.unparse(prefix.body[2]).endswith("tcgen05_ab_producer_state.clone()")
    writes = {
        n.id
        for n in ast.walk(prefix)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    assert writes and all(n.startswith("tcgen05_prefill_") for n in writes)
    loops = [n for n in prefix.body if isinstance(n, ast.For)]
    assert len(loops) == 1 and "range(7, unroll=1)" in ast.unparse(loops[0].iter)
    original_advances = [
        n for n in prefix.body if "tcgen05_ab_producer_state.advance" in ast.unparse(n)
    ]
    assert not original_advances
    producer = next(
        n
        for n in fn.body
        if isinstance(n, ast.If)
        and "tcgen05_prefilled_first_record =" in ast.unparse(n)
    )
    resumed = next(n for n in producer.body if isinstance(n, ast.For))
    assert "range(7, unroll_full=True)" in ast.unparse(resumed.iter)
    work = next(n for n in producer.body if isinstance(n, ast.While))
    gated = [
        n
        for n in work.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not tcgen05_prefilled_first_record"
    ]
    # The throttled scheduler's per-tile credit, then the resumed prefetch.
    assert len(gated) == 2
    assert ast.unparse(gated[0].body[0]).endswith(
        "tcgen05_clc_throttle_barrier.arrive()"
    )
    first = gated[1]
    assert len(first.body) == 1 and isinstance(first.body[0], ast.For)
    latch = work.body[work.body.index(first) + 1]
    assert (
        ast.unparse(latch) == "tcgen05_prefilled_first_record = cutlass.Boolean(False)"
    )
    # No shared-memory read or output bounds predicate chooses prefill count.
    assert not any(isinstance(n, ast.While) for n in ast.walk(prefix))
    assert (
        sum(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "copy"
            for n in ast.walk(loops[0])
        )
        == 2
    )


@skipUnlessBackends(["cute"])
def test_prefill_false_and_absent_are_byte_identical(case):
    bound, _, config, _ = case
    off = deepcopy(config)
    off.config.pop(STARTUP_PREFILL_KEY)
    absent = bound.to_code(off)
    off.config[STARTUP_PREFILL_KEY] = False
    assert bound.to_code(off) == absent
    assert STARTUP_PREFILL_KEY not in bound.config_spec.normalized_config(off)
    assert "tcgen05_prefill_state" not in absent
    assert "range(7, unroll=1)" in absent


@skipUnlessBackends(["cute"])
def test_steady_producer_has_only_remaining_packets(case):
    source = case[3]
    steady = next(
        n
        for n in ast.walk(device(source))
        if isinstance(n, ast.For) and ast.unparse(n.target) == "tcgen05_remaining_k"
    )
    assert ast.unparse(steady.iter) == "cutlass.range(7, 16, unroll=1)"
    assert not any(isinstance(n, ast.If) for n in ast.walk(steady))
    assert len(steady.body) == 7
    rendered = [ast.unparse(n) for n in steady.body]
    assert ".producer_try_acquire(" in rendered[0]
    assert ".producer_acquire(" in rendered[1]
    assert "tcgen05_ab_producer_try_token" in rendered[1]
    assert ".producer_get_barrier(" in rendered[2]
    assert all("tcgen05_remaining_k" in n for n in rendered[3:5])
    assert ".producer_commit(" in rendered[5]
    assert rendered[6] == "tcgen05_ab_producer_state.advance()"


@pytest.mark.parametrize("order", list(itertools.permutations((17, 23, 31))))
def test_linear_records_preserve_all_semantic_pid_orders(order):
    groups, m, n = 16, 10240, 1024
    rows = (m + 191) // 192
    union = GroupedRowUnionPlan(
        groups, m, n, 1024, 17, 23, 31, "o", "u", schedule=PAIRED_CLC
    )
    plan = SimpleNamespace(row_union=union, cluster_m=2)
    extents = {17: groups, 23: rows, 31: n // 256}
    strategy: Any = SimpleNamespace(
        pid_info=[
            SimpleNamespace(
                block_id=b,
                pid_var=f"pid_{b}",
                pack=1,
                pack_axis=0,
                num_pids_expr=lambda *, is_device, value=extents[b]: str(value),
            )
            for b in order
        ],
        _tcgen05_plan=lambda: plan,
        _tcgen05_logical_m_coord_expr=lambda value: f"({value}) // 2",
    )
    expression = (
        Tcgen05PersistentProgramIDs._tcgen05_linear_virtual_pid_from_coords_expr(
            strategy, ["unused_m", "unused_n", "record"]
        )
    )
    if order == (17, 23, 31):
        assert expression == "(record) * cutlass.Int32(16)"
    state: Any = SimpleNamespace(
        device_function=SimpleNamespace(new_var=lambda name: name)
    )
    code = compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    *ast.parse("virtual = " + expression).body,
                    *ProgramIDs._decompose_pid_to_statements(
                        strategy, "virtual", state
                    ),
                ],
                type_ignores=[],
            )
        ),
        "<paired-clc-pid>",
        "exec",
    )
    for record in range(rows * (n // 256)):
        scope: dict[str, Any] = {
            "record": record,
            "cutlass": SimpleNamespace(Int32=int),
        }
        exec(code, scope)
        assert (scope["pid_17"], scope["pid_23"], scope["pid_31"]) == (
            0,
            record % rows,
            record // rows,
        )


@pytest.mark.parametrize(
    "m,n,k,g", [(640, 512, 512, 1), (1280, 768, 1024, 31), (768, 256, 512, 2)]
)
@skipUnlessBackends(["cute"])
def test_non_original_domains_emit(m, n, k, g):
    assert PAIRED_CLC.index_domain(g, m, n, k)
    bound, _ = _bound(m, n, k, g)
    with bound.env:
        config = grouped_row_union_paired_clc_carrier(
            bound.env, bound.host_function.device_ir
        )
    assert config is not None
    config.config[STARTUP_PREFILL_KEY] = True
    source = bound.to_code(config)
    assert f"Int32({g + 1})" in source
    assert f"Int32({m})" in source
    assert "tcgen05_prefill_state" in source


@pytest.mark.parametrize(
    "m,n,k,g",
    [
        (10240, 1024, 1024, 32),
        (10208, 1024, 1024, 16),
        (10240, 960, 1024, 16),
        (10240, 1024, 384, 16),
        (10240, 1024, 16448, 16),
        (0, 1024, 1024, 16),
    ],
)
def test_paired_proof_rejects_uncovered_domains(m, n, k, g):
    assert not PAIRED_CLC.index_domain(g, m, n, k)


def test_legacy_cluster4_whole_tile_domain_stays_separate():
    assert not TRANSPOSED.index_domain(2, 768, 256, 512)
    assert TRANSPOSED.index_domain(8, 2560, 512, 512)
    assert PAIRED_CLC.index_domain(16, 10240, 1024, 1024)
    assert PAIRED_CLC.shared_upper_bound == 219136


@pytest.mark.parametrize(
    "updates",
    [
        {SCHEDULE_KEY: "legacy"},
        {CONFIG_KEY: False},
        {STARTUP_PREFILL_KEY: 1},
        {STARTUP_PREFILL_KEY: "true"},
        {"tcgen05_warp_spec_c_input_warps": 1},
        {"tcgen05_c_acquire_placement": "first_in_loop"},
        {"tcgen05_acc_wait_placement": "before_subtile_loop"},
        {"tcgen05_grouped_ctas_per_sm": 2},
        {"block_sizes": [128, 64, 64]},
    ],
)
@skipUnlessBackends(["cute"])
def test_invalid_explicit_startup_or_profile_rejects(case, updates):
    config = deepcopy(case[2])
    config.config.update(updates)
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        case[0].to_code(config)


@skipUnlessBackends(["cute"])
def test_full_population_keeps_both_profile_siblings(case):
    bound, args, _, _ = case
    seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    assert seeds[-2][SCHEDULE_KEY] == seeds[-1][SCHEDULE_KEY] == PAIRED_CLC_SCHEDULE
    assert not seeds[-2].get(STARTUP_PREFILL_KEY, False)
    assert seeds[-1][STARTUP_PREFILL_KEY] is True
    search = _search(bound, args)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026092151)
        population = search._generate_initial_population_flat()
    configs = [search.config_gen.unflatten(row) for row in population]
    outcomes = [
        x
        for x in search.compiler_coverage_outcomes
        if x.mechanism == "cute.ab_startup_prefill"
    ]
    assert len(outcomes) == 1
    assert outcomes[0].outcome in ("added", "already_present")
    assert outcomes[0].effective in configs
    assert outcomes[0].effective[STARTUP_PREFILL_KEY] is True
    assert bound.config_spec.compiler_seed_configs == seeds


def _clc_throttle_markers(source: str) -> tuple[int, int, int]:
    """(barrier objects, TMA-load credits, scheduler waits) in ``source``."""
    return (
        source.count(
            "tcgen05_clc_throttle_barrier = cutlass.pipeline.NamedBarrier("
            "barrier_id=5, num_threads=64)"
        ),
        source.count("tcgen05_clc_throttle_barrier.arrive()"),
        source.count("tcgen05_clc_throttle_barrier.arrive_and_wait()"),
    )


@skipUnlessBackends(["cute"])
def test_clc_query_pacing_throttles_the_scheduler_to_tile_starts(case):
    bound, _, config, source = case
    # Default pacing ("tile"): one barrier, the leader TMA-load warp's credit
    # for the prefilled first tile and one per later tile, the scheduler's
    # wait right before its single query.
    assert _clc_throttle_markers(source) == (1, 2, 1)
    fn = device(source)
    credit = (
        "if cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == cutlass.Int32(0):"
        "\n    tcgen05_clc_throttle_barrier.arrive()"
    )
    per_tile = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not tcgen05_prefilled_first_record"
        and ast.unparse(n.body[0]) == credit
    ]
    assert len(per_tile) == 1
    # The per-tile credit leads the tile's AB prefetch (so the next tile's
    # query overlaps these loads) and the wait leads the query.
    rendered = ast.unparse(fn)
    prologue_credit = rendered.index("tcgen05_clc_throttle_barrier.arrive()")
    assert prologue_credit < rendered.index("tcgen05_prefill_state =")
    assert rendered.index(
        "tcgen05_clc_throttle_barrier.arrive()", prologue_credit + 1
    ) < rendered.index("tcgen05_initial_k in cutlass.range(")
    wait = rendered.index("tcgen05_clc_throttle_barrier.arrive_and_wait()")
    assert wait < rendered.index("_cute_issue_clc_query_nomulticast(")
    assert rendered.rindex("cute.arch.sync_warp()", 0, wait) > rendered.rindex(
        "tcgen05_sched_pipeline_producer_state.advance()", 0, wait
    )
    # The default pacing is implied: normalization never carries it, so a
    # config spelling it out and one leaving it implicit read alike.
    normalized = bound.config_spec.normalized_config(config)
    assert TCGEN05_CLC_QUERY_PACING_CONFIG_KEY not in normalized
    spelled = deepcopy(config)
    spelled.config[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY] = "tile"
    assert bound.config_spec.normalized_config(spelled) == normalized
    assert bound.to_code(spelled) == source
    # Eager pacing: no barrier, no credits, no wait; the knob survives.
    eager = deepcopy(config)
    eager.config[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY] = "eager"
    eager_source = bound.to_code(eager)
    assert _clc_throttle_markers(eager_source) == (0, 0, 0)
    assert eager_source.count("_cute_issue_clc_query_nomulticast(") == 1
    assert (
        bound.config_spec.normalized_config(eager)[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY]
        == "eager"
    )


@skipUnlessBackends(["cute"])
def test_clc_query_pacing_needs_a_clc_scheduler(case):
    bound, _, config, _ = case
    # The transposed profile has no scheduler warp: the inert default pacing
    # leaves the config silently, a requested non-default pacing is rejected
    # strictly and repaired by the search normalization.
    other = deepcopy(config)
    other.config[SCHEDULE_KEY] = TRANSPOSED.name
    other.config.pop(STARTUP_PREFILL_KEY)
    other.config[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY] = "tile"
    assert (
        TCGEN05_CLC_QUERY_PACING_CONFIG_KEY
        not in bound.config_spec.normalized_config(other)
    )
    other.config[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY] = "eager"
    with pytest.raises(InvalidConfig, match="tcgen05_clc_query_pacing"):
        bound.config_spec.normalized_config(other)
    repaired = deepcopy(other)
    bound.config_spec.normalize(repaired, _fix_invalid=True)
    assert TCGEN05_CLC_QUERY_PACING_CONFIG_KEY not in repaired.config
    assert "tcgen05_clc_throttle_barrier" not in bound.to_code(repaired)
