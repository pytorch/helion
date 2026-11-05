from __future__ import annotations

import ast
from copy import deepcopy
import itertools
import random
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _bound
from test.test_cute_grouped_coverage_search import _search
from test.test_cute_grouped_k128_full_coverage import _bound as _generic_bound
from test.test_cute_native_schedule_cuda import _row_union_bound
from test.test_cute_shared_rhs_grouped import _OffsetPointer
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.grouped_row_union import AB_STAGES_KEY
from helion._compiler.cute.grouped_row_union import CONFIG_KEY
from helion._compiler.cute.grouped_row_union import PAIRED_CLC_SCHEDULE
from helion._compiler.cute.grouped_row_union import ROUND_SUBTILES_KEY
from helion._compiler.cute.grouped_row_union import SCHEDULE_KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED
from helion._compiler.cute.grouped_row_union import TRANSPOSED_SCHEDULE
from helion._compiler.cute.grouped_row_union import GroupedRowUnionPlan
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.program_id import ProgramIDs
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _append_cute_wrapper_plan


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    initialized = torch.cuda.is_initialized()
    torch.set_num_threads(1)
    try:
        with (
            _mock_cuda_unavailable(),
            patch(
                "torch.cuda.current_device", side_effect=AssertionError("CPU only")
            ) as current_device,
            patch(
                "torch.cuda.get_device_capability",
                side_effect=AssertionError("CPU only"),
            ) as capability,
            patch(
                "torch.cuda.get_device_properties",
                side_effect=AssertionError("CPU only"),
            ) as properties,
            patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")
            ) as initialization,
            patch(
                "torch.cuda.synchronize", side_effect=AssertionError("GPU forbidden")
            ) as synchronize,
            _forbid_native_compile() as native_compilation,
            _target(),
            patch(
                "helion._compiler.cute.cute_mma.split_barrier_init_supported",
                return_value=True,
            ),
        ):
            yield
            for prohibited in (
                current_device,
                capability,
                properties,
                initialization,
                synchronize,
                *native_compilation,
            ):
                prohibited.assert_not_called()
            assert torch.cuda.is_initialized() == initialized
    finally:
        torch.set_num_threads(previous)


def selected(bound):
    group = next(
        g for g in bound.config_spec.compiler_coverage_groups if g.key == SCHEDULE_KEY
    )
    config = deepcopy(group.witnesses[0].carrier)
    config.config[SCHEDULE_KEY] = TRANSPOSED_SCHEDULE
    return config


@pytest.fixture(scope="module")
def case():
    bound, args = _bound(1)
    config = selected(bound)
    saved = deepcopy(config)
    source = bound.to_code(config)
    assert config == saved
    return bound, args, config, source


def wrapper(source):
    body = []
    for plan in _plans(source):
        _append_cute_wrapper_plan(body, [], plan)
    result = ast.parse("def wrapper():\n" + "\n".join(body)).body[0]
    assert isinstance(result, ast.FunctionDef)
    return result


@skipUnlessBackends(["cute"])
def test_ordinary_profile_and_original_allocation(case):
    bound, _args, config, source = case
    source = ast.unparse(ast.parse(source))
    assert config.block_sizes == [128, 64, 128]
    assert (
        bound.config_spec.normalized_config(config)[SCHEDULE_KEY] == TRANSPOSED_SCHEDULE
    )
    with bound.env:
        projected = bound.env.backend.codegen_config(config)
    assert projected.block_sizes == [80, 256, 64]
    assert config.block_sizes == [128, 64, 128]
    assert source.count("StaticPersistentTileScheduler.create(") == 3
    assert "tcgen05_work_tile_smem" not in source
    assert "block=(32, 6, 1)" in source
    assert "_MAX_ACTIVE_CLUSTERS" in source and "_NUM_SM // 4" not in source
    assert source.count("out = torch.empty(") == 1
    ab, out = _plans(source)
    assert (ab["bm"], ab["bn"], ab["bk"], ab["ab_stage_count"]) == (256, 80, 64, 10)
    assert ab["row_union_schedule"] == TRANSPOSED_SCHEDULE
    assert ab["a_producer_partition"] == (2, 1, 1)
    assert (ab["lhs_idx"], ab["rhs_idx"], out["d_idx"]) == (2, 1, 3)
    assert (out["epi_tile_m"], out["epi_tile_n"], out["d_store_box_n"]) == (128, 16, 16)
    assert "row_union_all_rows and" in source
    assert "row_union_store_pred_copy" in source
    assert source.count("fence_view_async_tmem_load()") == 2
    assert source.count("tcgen05_acc_pipeline.consumer_release(") == 2
    assert source.count("tcgen05_c_pipeline.producer_tail()") == 1
    assert (
        "row_union_output = cute.make_tensor(out.iterator, cute.make_layout((512, 2560), stride=(1, 512)))"
        in source
    )
    rendered = ast.unparse(wrapper(source))
    assert "cute.make_layout((2, 1, 1))" in rendered
    assert "cluster_shape_to_tma_atom_A((2, 2, 1)" in rendered
    assert "cute.select(tma_atom_b_smem_layout, mode=[0, 1, 2])" in rendered


def test_transposed_row_union_advances_accumulator_once_per_output_tile():
    args = (
        torch.empty((5120, 512), dtype=torch.bfloat16),
        torch.empty((512, 512), dtype=torch.bfloat16),
        torch.tensor([0, 13, 13, 479, 1280, 1280, 3819, 5001, 5120], dtype=torch.int32),
    )
    bound, config = _row_union_bound(args, "cluster4")
    source = bound.to_code(config)
    ab = _plans(source)[0]
    assert (ab["bm"], ab["bn"], ab["bk"]) == (256, 80, 64)
    assert (ab["cluster_m"], ab["cluster_n"]) == (2, 2)
    assert config.block_sizes == [128, 64, 128]
    # Logical projected M=80 is not a physical M subtile of the 256-row MMA.
    # Its producer must still progress through the accumulator ring per tile.
    work_loops = []
    for loop in ast.walk(ast.parse(source)):
        if not isinstance(loop, ast.While):
            continue
        commits = [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "producer_commit"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id.endswith("acc_producer_state")
        ]
        if commits:
            assert len(commits) == 1
            work_loops.append((loop, commits[0]))
    assert len(work_loops) == 1
    loop, commit = work_loops[0]
    producer_state = commit.args[0]
    assert isinstance(producer_state, ast.Name)
    state = producer_state.id
    commit_index = next(
        index for index, stmt in enumerate(loop.body) if commit in ast.walk(stmt)
    )
    advance_stmt = loop.body[commit_index + 1]
    assert isinstance(advance_stmt, ast.Expr)
    assert ast.unparse(advance_stmt.value) == f"{state}.advance()"
    assert (
        sum(
            isinstance(node, ast.Call) and ast.unparse(node) == f"{state}.advance()"
            for node in ast.walk(loop)
        )
        == 1
    )


@pytest.mark.parametrize(
    "updates",
    [
        {CONFIG_KEY: False},
        {"block_sizes": [64, 64, 64]},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_ab_stages": 10},
        {"tcgen05_grouped_ctas_per_sm": 2},
        {SCHEDULE_KEY: "unknown"},
        {"tcgen05_cluster_m": 1.0},
        {"tcgen05_cluster_m": True},
        {"block_sizes": [128.0, 64, 128]},
    ],
)
@skipUnlessBackends(["cute"])
def test_incoherent_explicit_schedule_rejects(case, updates):
    config = deepcopy(case[2])
    config.config.update(updates)
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        case[0].to_code(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("a_producer_partition", (2, 2, 1)),
        ("cluster_n", 1),
        ("ab_stage_count", 9),
        ("bn", 64),
        ("b_k_major", False),
        ("row_union_schedule", "unknown"),
    ],
)
@skipUnlessBackends(["cute"])
def test_wrapper_rejects_descriptor_and_schedule_mutations(case, field, value):
    plan = deepcopy(_plans(case[3])[0])
    plan[field] = value
    with pytest.raises(BackendUnsupported, match="physical descriptor"):
        _append_cute_wrapper_plan([], [], plan)


@pytest.mark.parametrize("m,n,k,groups", [(640, 256, 128, 2), (1280, 768, 1024, 5)])
@skipUnlessBackends(["cute"])
def test_general_aligned_domains_emit(m, n, k, groups):
    bound, args = _generic_bound(m, n, k, groups)
    source = bound.to_code(selected(bound))
    plan = _plans(source)[0]
    assert (plan["m_size"], plan["n_size"], plan["k_total_size"]) == (n, m, k)
    assert source.count("StaticPersistentTileScheduler.create(") == 3
    assert f"make_layout(({n}, {m}), stride=(1, {n}))" in source


@pytest.mark.parametrize(
    "m,n,k", [(319, 256, 64), (320, 255, 64), (320, 256, 63), (320, 256, 16448)]
)
@skipUnlessBackends(["cute"])
def test_shape_proof_declines_spatial_k_tails_and_loop_limit(m, n, k):
    assert not TRANSPOSED.index_domain(2, m, n, k)
    bound, _ = _generic_bound(m, n, k, 2)
    assert not any(
        g.key == SCHEDULE_KEY for g in bound.config_spec.compiler_coverage_groups
    )


@skipUnlessBackends(["cute"])
def test_actual_full_population_transfers_dependent_witness(case):
    bound, args = case[:2]
    seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    previous_seeds = [
        seed for seed in seeds if seed.get(SCHEDULE_KEY) != PAIRED_CLC_SCHEDULE
    ]
    # The transposed carrier, its measured twin (6-stage ring, epilogue
    # rounds of two subtiles) and that twin without the setmaxnreg pair close
    # the non-paired prefix.
    assert [seed.get(SCHEDULE_KEY) for seed in previous_seeds[-3:]] == [
        TRANSPOSED_SCHEDULE,
        TRANSPOSED_SCHEDULE,
        TRANSPOSED_SCHEDULE,
    ]
    for seed in previous_seeds[-2:]:
        assert seed[AB_STAGES_KEY] == 6
        assert seed[ROUND_SUBTILES_KEY] == 2
    assert "tcgen05_warp_register_reallocation" not in previous_seeds[-2]
    assert previous_seeds[-1]["tcgen05_warp_register_reallocation"] is False
    assert "tcgen05_consumer_regs" not in previous_seeds[-1]
    assert all(SCHEDULE_KEY not in seed for seed in previous_seeds[:-3])
    search = _search(bound, args)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091951)
        rows = search._generate_initial_population_flat()
    configs = [search.config_gen.unflatten(row) for row in rows]
    outcomes = [
        entry
        for entry in search.compiler_coverage_outcomes
        if entry.mechanism == "cute.grouped_row_union_schedule"
        and entry.requested.get(SCHEDULE_KEY) == TRANSPOSED_SCHEDULE
    ]
    assert len(outcomes) == 1
    assert outcomes[0].outcome in ("added", "already_present"), outcomes
    assert outcomes[0].effective in configs
    assert outcomes[0].effective is not None
    assert outcomes[0].effective[CONFIG_KEY] is True
    assert outcomes[0].effective[SCHEDULE_KEY] == TRANSPOSED_SCHEDULE
    assert seeds == bound.config_spec.compiler_seed_configs


@skipUnlessBackends(["cute"])
def test_off_override_blocks_physical_seed():
    bound, args = _bound(1, autotune_config_overrides={CONFIG_KEY: False})
    search = _search(bound, args)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        rows = search._generate_initial_population_flat()
    assert all(SCHEDULE_KEY not in search.config_gen.unflatten(row) for row in rows)


@pytest.mark.parametrize("disable_heuristics", [False, True])
@skipUnlessBackends(["cute"])
def test_random_full_population_retains_physical_witness(disable_heuristics):
    bound, args = _bound(
        1,
        disable_autotuner_heuristics=disable_heuristics,
        autotune_initial_population_strategy="from_random",
    )
    search = _search(bound, args)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        rows = search._generate_initial_population_flat()
    if disable_heuristics:
        assert not bound.config_spec.compiler_seed_configs
        assert "compiler_coverage_outcomes" not in vars(search)
        assert any(
            group.key == SCHEDULE_KEY
            for group in bound.config_spec.compiler_coverage_groups
        )
        with bound.env:
            _flat, effective = search.config_gen.strict_config_pair(selected(bound))
        assert effective[SCHEDULE_KEY] == TRANSPOSED_SCHEDULE
        return
    outcome = next(
        item
        for item in search.compiler_coverage_outcomes
        if item.mechanism == "cute.grouped_row_union_schedule"
    )
    assert outcome.outcome in ("added", "already_present"), outcome
    assert outcome.effective in [search.config_gen.unflatten(row) for row in rows]
    assert outcome.effective in search._pinned_finalist_configs


def test_thread_coordinate_omission_does_not_drop_observable_work():
    cls = Tcgen05PersistentProgramIDs
    assert cls._tcgen05_shared_stmt_safe_to_omit(
        ast.parse("x = cute.arch.thread_idx()[0] * 2").body[0]
    )
    for source in (
        "cute.copy(a, b, c)",
        "out[i] = x",
        "barrier.wait()",
        "x = unknown()",
    ):
        assert not cls._tcgen05_shared_stmt_safe_to_omit(ast.parse(source).body[0])


@skipUnlessBackends(["cute"])
def test_late_shared_capacity_failure_cannot_fall_back_to_scalar(case):
    with (
        patch.object(
            CuteTcgen05Config,
            "per_cta_smem_capacity_bytes",
            return_value=TRANSPOSED.shared_upper_bound - 1,
        ),
        pytest.raises(BackendUnsupported, match="row-union"),
    ):
        case[0].to_code(case[2])


@pytest.mark.parametrize("stride", [1, 2])
@skipUnlessBackends(["cute"])
def test_actual_interval_normalization_and_dense_guard(case, stride):
    device = next(
        n
        for n in ast.parse(case[3]).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    # The interval setup is epilogue-role prelude: it sits inside the
    # epilogue warps' role block, behind that role's dependent-launch wait
    # and ahead of its persistent loop (so a ``cute_pdl`` kernel needs no
    # entry wait and its prologue overlaps the previous kernel).
    assert not any(
        isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).startswith("row_union_")
        for node in device.body
    )
    role_block = next(
        node
        for node in device.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.For)
            and ast.unparse(child.target) == "row_union_group"
            for child in node.body
        )
    )
    assert ast.unparse(role_block.test).endswith("< cutlass.Int32(4)")
    statements: list[ast.stmt] = [
        node
        for node in role_block.body
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0])
            in {"row_union_starts", "row_union_ends", "row_union_all_rows"}
        )
        or (
            isinstance(node, ast.For)
            and ast.unparse(node.target)
            in {"row_union_group", "row_union_coverage_group"}
        )
    ]
    assert [type(node).__name__ for node in statements] == [
        "Assign",
        "Assign",
        "For",
        "Assign",
        "For",
    ]
    assert all(
        role_block.body.index(node) > role_block.body.index(wait)
        for wait in role_block.body
        if ast.unparse(wait) == "cute.arch.griddepcontrol_wait()"
        for node in statements
    )
    assert all(
        role_block.body.index(node) < role_block.body.index(loop)
        for loop in role_block.body
        if isinstance(loop, ast.While)
        for node in statements
    )
    code = compile(
        ast.Module(body=statements, type_ignores=[]), "<actual-interval-setup>", "exec"
    )
    m, groups = 2560, 8
    corpus = [
        list(range(0, m + 1, m // groups)),
        [1280, 2560, 0, 1280, 1280, 1280, 1280, 1280, 1280],
        [-17, 193, 2577, 2577, 2577, 2577, 2577, 2577, 2577],
        [320] * 9,
        [-(1 << 31), (1 << 31) - 1, 0, m, m, m, m, m, m],
        [(1 << 31) - 1, -(1 << 31), 0, m, m, m, m, m, m],
    ]
    rng = random.Random(2026091967)
    values = [-(1 << 31), -17, 0, 1, 193, 1280, 2559, 2560, 2577, (1 << 31) - 1]
    corpus.extend(rng.choices(values, k=9) for _ in range(64))

    def convert(value, dtype):
        return (
            value.to(dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=dtype)
        )

    def wrap(value):
        return (value + (1 << 31)) % (1 << 32) - (1 << 31)

    for offsets in corpus:
        data = torch.full((len(offsets) * stride,), 17, dtype=torch.int32)
        data[::stride] = torch.tensor(offsets, dtype=torch.int32)
        scope: dict[str, Any] = {
            "cutlass": SimpleNamespace(
                Int32=lambda x: convert(x, torch.int32),
                Int64=lambda x: convert(x, torch.int64),
                Boolean=bool,
                range=lambda n, **kwargs: range(n),
            ),
            "cute": SimpleNamespace(
                make_layout=lambda x: x,
                make_rmem_tensor=lambda shape, dtype: torch.empty(
                    shape, dtype=torch.int32
                ),
            ),
            "group_offsets": SimpleNamespace(
                iterator=_OffsetPointer(data), layout=SimpleNamespace(stride=(stride,))
            ),
            "tcgen05_epi_active": True,
        }
        exec(code, scope)
        intervals = []
        covered = set()
        for start, end in itertools.pairwise(offsets):
            extent = max(wrap(end - start), 0)
            first = min(max(start, 0), m)
            length = min(max(wrap(extent + min(start, 0)), 0), m - first)
            intervals.append((first, first + length))
            covered.update(range(first, first + length))
        assert (
            list(
                zip(
                    scope["row_union_starts"].tolist(),
                    scope["row_union_ends"].tolist(),
                    strict=True,
                )
            )
            == intervals
        )
        expected_dense = (
            intervals[0][0] == 0
            and intervals[-1][1] == m
            and all(
                left[1] == right[0] for left, right in itertools.pairwise(intervals)
            )
        )
        assert bool(scope["row_union_all_rows"]) == expected_dense
        if scope["row_union_all_rows"]:
            assert covered == set(range(m))


@pytest.mark.parametrize("k", [128, 512, 768, 16384])
@skipUnlessBackends(["cute"])
def test_actual_producer_all_ranks_and_ring_wrap(k):
    # The prologue runtime imports cutlass at module level: CPU-only runners skip.
    _helion_tc_pro = pytest.importorskip(
        "helion._compiler.cute._tcgen05_prologue_runtime"
    )
    bound, _ = _generic_bound(640, 256, k, 4)
    source = bound.to_code(selected(bound))
    device = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    producer = next(
        node
        for node in device.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test).endswith("== cutlass.Int32(5)")
    )
    work = next(node for node in producer.body if isinstance(node, ast.While))
    start = next(
        i
        for i, node in enumerate(work.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "tcgen05_tma_initial_full_tile"
    )
    end = next(
        i
        for i, node in enumerate(work.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func).endswith("advance_to_next_work")
    )
    code = compile(
        ast.Module(body=work.body[start:end], type_ignores=[]),
        "<actual-producer>",
        "exec",
    )

    class State:
        def __init__(self):
            self.index, self.phase = 0, 1

        def advance(self):
            self.index += 1
            if self.index == 10:
                self.index = 0
                self.phase ^= 1

    class Partition:
        def __getitem__(self, key):
            return key[-1]

    for rank in range(4):
        state, copies, acquisitions, commits = State(), [], [], []

        def copy(atom, source, destination, *, tma_bar_ptr, mcast_mask, copies=copies):
            assert destination == tma_bar_ptr
            copies.append((atom, source, destination, mcast_mask))

        pipeline = SimpleNamespace(
            producer_acquire=lambda state, *args, acquisitions=acquisitions: (
                acquisitions.append((state.index, state.phase))
            ),
            producer_get_barrier=lambda state: state.index,
            producer_commit=lambda state, commits=commits: commits.append(
                (state.index, state.phase)
            ),
            producer_try_acquire=lambda state: True,
        )
        scope: dict[str, Any] = {
            "cutlass": SimpleNamespace(
                Int32=int, Boolean=bool, range=lambda *args, **kwargs: range(*args)
            ),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(
                    make_warp_uniform=lambda x: x,
                    block_idx_in_cluster=lambda rank=rank: rank,
                    # the dependent-launch wait + release sit between stage
                    # 0's acquire and its loads
                    griddepcontrol_wait=lambda: None,
                    griddepcontrol_launch_dependents=lambda: None,
                ),
                copy=copy,
            ),
            "tile_offset_1": (rank // 2) * 80,
            "tile_offset_2": 0,
            "symnode_0": k,
            "_BLOCK_SIZE_3": 64,
            "tcgen05_ab_pipeline": pipeline,
            "tcgen05_ab_producer_state": state,
            "tma_atom_a": "A",
            "tma_atom_b": "B",
            "tma_gA": Partition(),
            "tma_gB": Partition(),
            "tma_sA": Partition(),
            "tma_sB": Partition(),
            "tcgen05_a_mcast_mask": 5 << (rank % 2),
            "tcgen05_b_mcast_mask": 1 << rank,
            "_helion_tc_pro": _helion_tc_pro,
        }
        exec(code, scope)
        count = k // 64
        assert (
            acquisitions
            == commits
            == [(i % 10, 1 ^ ((i // 10) % 2)) for i in range(count)]
        )
        assert [item[1] for item in copies if item[0] == "A"] == (
            list(range(count)) if rank < 2 else []
        )
        assert [item[1] for item in copies if item[0] == "B"] == list(range(count))
        assert all(stage == tile % 10 for _, tile, stage, _ in copies)
        assert (state.index, state.phase) == (count % 10, 1 ^ ((count // 10) % 2))


@pytest.mark.parametrize("order", list(itertools.permutations((17, 23, 31))))
def test_physical_pid_maps_to_original_rows_and_columns(order):
    groups, m, n = 5, 1280, 768
    union = GroupedRowUnionPlan(
        groups, m, n, 64, 17, 23, 31, "o", "u", schedule=TRANSPOSED
    )
    plan = SimpleNamespace(row_union=union)
    extents = {17: groups, 23: m // 80, 31: n // 256}
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
            strategy, ["pm", "pn", "0"]
        )
    )
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
        "<physical-pid>",
        "exec",
    )
    for pm, pn in itertools.product(range(2 * (n // 256)), range(m // 80)):
        scope: dict[str, Any] = {
            "pm": pm,
            "pn": pn,
            "cutlass": SimpleNamespace(Int32=int),
        }
        exec(code, scope)
        assert (scope["pid_17"], scope["pid_23"], scope["pid_31"]) == (0, pn, pm // 2)


@skipUnlessBackends(["cute"])
def test_actual_wrapper_descriptor_and_device_copy_coremlir(case, tmp_path):
    initialized = torch.cuda.is_initialized()
    import cutlass
    from cutlass._mlir import ir as mlir_ir
    from cutlass._mlir import passmanager as mlir_passmanager
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    import helion._compiler.cute._tcgen05_prologue_runtime as _helion_tc_pro

    # Dynamically exported pybind types are absent from the SDK stubs.
    ir: Any = mlir_ir
    passmanager: Any = mlir_passmanager

    def forbidden(*args, **kwargs):
        raise AssertionError("native/GPU forbidden")

    def execute(node, scope):
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "<actual-emission>", "exec"
            ),
            scope,
        )

    device = next(
        n
        for n in ast.parse(case[3]).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    producer = next(
        n
        for n in device.body
        if isinstance(n, ast.If) and ast.unparse(n.test).endswith("== cutlass.Int32(5)")
    )
    work = next(n for n in producer.body if isinstance(n, ast.While))
    assignments = {
        n.targets[0].id: n
        for n in work.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    # The TMA partitions of the whole tiled operands run once per CTA in the
    # role's prelude ahead of its cluster join; the loop slices the tile.
    prelude = producer.body[: producer.body.index(work)]
    partitions = [
        n
        for n in prelude
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
    ]
    assert len(partitions) == 2
    assert [ast.unparse(n.targets[0]) for n in partitions] == [
        "(tma_sA, tma_gA_all)",
        "(tma_sB, tma_gB_all)",
    ]
    cluster_wait = next(
        i
        for i, n in enumerate(prelude)
        if ast.unparse(n).startswith("cutlass.pipeline.pipeline_init_wait(")
    )
    assert all(prelude.index(n) < cluster_wait for n in partitions)
    assert ast.unparse(assignments["tma_gA"].value) == (
        "tma_gA_all[None, _helion_tc_pro.floordiv_nonneg(tile_offset_2, 256), None]"
    )
    assert ast.unparse(assignments["tma_gB"].value) == (
        "tma_gB_all[None, _helion_tc_pro.floordiv_nonneg(tile_offset_1, 80), None]"
    )
    assert not any("tma_partition(" in ast.unparse(n) for n in work.body)
    first = next(
        n for n in work.body if isinstance(n, ast.If) and "cute.copy(" in ast.unparse(n)
    )
    copies = {
        ast.unparse(n.value.args[0]): n
        for n in ast.walk(first)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "cute.copy"
    }
    assert set(copies) == {"tma_atom_a", "tma_atom_b"}
    copy_parent = next(
        n
        for n in ast.walk(first)
        if isinstance(n, ast.If) and copies["tma_atom_a"] in n.body
    )
    assert (
        ast.unparse(copy_parent.test)
        == "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) < cutlass.Int32(2)"
    )

    with (
        patch.object(cute, "compile", forbidden),
        patch.object(cutlass, "compile", forbidden),
    ):
        for kind in ("descriptor", "copy"):
            with ir.Context(), ir.Location.unknown():
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp(kind, ([], []))
                with ir.InsertionPoint(function.add_entry_block()):
                    scope: dict[str, Any] = {
                        "cute": cute,
                        "cutlass": cutlass,
                        "_helion_tc_pro": _helion_tc_pro,
                    }
                    for index, shape, stride, dtype in [
                        (0, (9,), (1,), cutlass.Int32),
                        (1, (2560, 512), (512, 1), cutlass.BFloat16),
                        (2, (512, 512), (512, 1), cutlass.BFloat16),
                        (3, (2560, 512), (512, 1), cutlass.BFloat16),
                    ]:
                        scope[f"arg{index}"] = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (index + 1) * 2**26,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(shape, stride=stride),
                        )
                        for dim, value in enumerate(shape):
                            scope[f"arg{index}_shape{dim}"] = value
                            scope[f"arg{index}_stride{dim}"] = stride[dim]
                    for node in wrapper(case[3]).body:
                        execute(node, scope)
                    atom = scope["tma_atom_a"]
                    if kind == "descriptor":
                        function.attributes["function_type"] = ir.TypeAttr.get(
                            ir.FunctionType.get([], [atom._trait.value.type])
                        )
                        func.ReturnOp([atom._trait.value])
                    else:
                        mma = scope["tma_atom_a_tiled_mma"]
                        la, lb = (
                            scope["tma_atom_a_smem_layout"],
                            scope["tma_atom_b_smem_layout"],
                        )
                        assert int(cute.cosize(la.outer)) * 2 == 163840
                        assert int(cute.cosize(lb.outer)) * 2 == 51200
                        scope.update(
                            tma_thr_mma=mma.get_slice(0),
                            tma_a_cta_coord=0,
                            tma_b_cta_coord=0,
                            tma_a_cta_layout=cute.make_layout(1),
                            tma_b_cta_layout=cute.make_layout(1),
                            tile_offset_1=0,
                            tile_offset_2=0,
                        )
                        for name, layout, address in (
                            ("sA", la, 0),
                            ("sB", lb, 163840),
                        ):
                            scope[name] = cute.make_tensor(
                                cute.recast_ptr(
                                    cute.make_ptr(
                                        cutlass.BFloat16,
                                        address,
                                        cute.AddressSpace.smem,
                                        assumed_align=128,
                                    ),
                                    layout.inner,
                                    dtype=cutlass.BFloat16,
                                ),
                                layout.outer,
                            )
                        # The prelude's once-per-CTA tiled views and
                        # partitions, then the tile's slices.
                        for node in prelude:
                            if isinstance(node, ast.Assign) and "tma_g" in ast.unparse(
                                node.targets[0]
                            ):
                                execute(node, scope)
                        for name in ("tma_gA", "tma_gB"):
                            execute(assignments[name], scope)
                        assert (
                            str(scope["tma_sA"].layout)
                            == "((4096,2),10):((1,4096),8192)"
                        )
                        assert (
                            str(scope["tma_sB"].layout) == "((2560,1),10):((1,0),2560)"
                        )
                        scope.update(
                            tcgen05_ab_producer_state=SimpleNamespace(index=0),
                            tcgen05_tma_barrier=cute.make_ptr(
                                cutlass.Int64,
                                215040,
                                cute.AddressSpace.smem,
                                assumed_align=8,
                            ),
                            tcgen05_a_mcast_mask=cutlass.Int16(5),
                            tcgen05_b_mcast_mask=cutlass.Int16(1),
                        )
                        execute(copies["tma_atom_a"], scope)
                        execute(copies["tma_atom_b"], scope)
                        func.ReturnOp([])
                pipeline = (
                    "builtin.module(canonicalize,cute-desugar,cute-expand-ops,cute-fold-static,canonicalize"
                    + (
                        ",convert-cute-to-core,canonicalize)"
                        if kind == "descriptor"
                        else ")"
                    )
                )
                passmanager.PassManager.parse(pipeline).run(module.operation)
                text = str(module)
                (tmp_path / f"{kind}.mlir").write_text(text)
                if kind == "descriptor":
                    constants = {
                        name: int(value)
                        for name, value in re.findall(
                            r"(%[\w]+) = arith.constant (-?\d+) : i64", text
                        )
                    }
                    pointers = {
                        name: int(index)
                        for name, index in re.findall(
                            r"(%[\w]+) = llvm.getelementptr %\w+\[0, (\d+)\]", text
                        )
                    }
                    words = {
                        pointers[p]: constants[v]
                        for v, p in re.findall(
                            r"llvm.store (%\w+), (%\w+) : i64, !llvm.ptr", text
                        )
                        if p in pointers and v in constants
                    }
                    assert words[7] == 63  # Former underfilled A64x32 encoded 31.
                else:
                    lines = [
                        line
                        for line in text.splitlines()
                        if "cute_nvgpu.arch.copy.SM100.tma_load" in line
                    ]
                    assert len(lines) == 3, lines
                    assert all("mask = %c5_i16" in line for line in lines[:2])
                    assert "mask =" not in lines[2]
                    assert "copy_bits = 65536" in text and "copy_bits = 40960" in text
    assert torch.cuda.is_initialized() == initialized


@skipUnlessBackends(["cute"])
def test_actual_epilogue_cross_lane_publication(case):
    import cutlass
    from cutlass._mlir import ir as mlir_ir
    import cutlass.cute as cute
    from cutlass.utils.gemm import sm100

    ir: Any = mlir_ir
    epi: Any = sm100

    source = case[3]
    device = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    nodes = {
        n.targets[0].id: n
        for n in device.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    full = next(
        n
        for n in ast.walk(device)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "tcgen05_full_tile"
    )
    fnodes = {
        n.targets[0].id: n
        for n in full.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            scope: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "mma_slice_tidx": 0,
            }

            def execute(name, table=nodes):
                exec(
                    compile(
                        ast.Module(body=[table[name]], type_ignores=[]),
                        "<ordinary-device>",
                        "exec",
                    ),
                    scope,
                )

            for name in [
                "tiled_mma",
                "thr_mma",
                "sB_layout",
                "sB_tma_layout",
                "acc_frag_base",
                "tcgen05_epilogue_rest_mode",
            ]:
                execute(name)
            for name in [
                "tcgen05_kernel_desc",
                "tcgen05_store_epi_tile",
                "tcgen05_sD_layout",
            ]:
                execute(name, fnodes)
            lb, selected = scope["sB_layout"], scope["sB_tma_layout"]
            stage_offsets = [selected(i) for i in range(cute.size(selected.outer))]
            assert stage_offsets == [lb(i) for i in range(2560)]
            assert set(stage_offsets) == set(range(2560))
            # Also reconstruct the same layout using only an outer-layout slice.
            rebuilt = cute.make_composed_layout(
                lb.inner, lb.offset, cute.slice_(lb.outer, (None, None, None, 0))
            )
            assert str(rebuilt) == str(selected)
            mma, frag = scope["tiled_mma"], scope["acc_frag_base"]
            tacc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
                ),
                frag.layout,
            )
            tacc = epi.transform_partitioned_tensor_layout(tacc)
            ptr = cute.make_ptr(
                cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16
            )
            global_c = cute.make_tensor(
                ptr, cute.make_layout((256, 80), stride=(1, 512))
            )
            gc = epi.transform_partitioned_tensor_layout(
                mma.get_slice(0).partition_C(global_c)
            )
            rest = scope["tcgen05_epilogue_rest_mode"]
            planned = cute.make_tensor(
                gc.iterator,
                cute.append(cute.append(cute.append(gc.layout, rest), rest), rest),
            )
            ep, ld, desc = (
                scope["tcgen05_store_epi_tile"],
                scope["tcgen05_sD_layout"],
                scope["tcgen05_kernel_desc"],
            )
            tc, pt, rr = epi.epilogue_tmem_copy_and_partition(
                desc, 0, tacc, planned, ep, True
            )
            rd = cute.make_rmem_tensor(rr.shape, cutlass.BFloat16)
            sd = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16, 0, cute.AddressSpace.smem, assumed_align=1024
                    ),
                    ld.inner,
                    dtype=cutlass.BFloat16,
                ),
                ld.outer,
            )
            r2s, rs, ds = epi.epilogue_smem_copy_and_partition(desc, tc, rd, 0, sd)
            coordinate = epi.transform_partitioned_tensor_layout(
                mma.get_slice(0).partition_C(cute.make_identity_tensor((256, 80)))
            )
            divided = cute.flat_divide(coordinate, ep)
            staged_identity = cute.make_identity_tensor((128, 16, 2))
            threads = cute.size(r2s.thr_id.shape)
            values = cute.size(r2s.layout_src_tv.shape[1])
            assert cute.size(r2s.layout_dst_tv.shape[1]) == values
            source_values, destinations = {}, [{}, {}]
            for tid in range(128):
                part = tc.get_slice(tid).partition_D(divided)[None, None, None, 0, 0]
                reg = r2s.retile(part)
                dst = r2s.get_slice(tid).partition_D(staged_identity)
                count = cute.size(reg.shape)
                assert count % values == 0
                for repeat in range(count // values):
                    for value in range(values):
                        index = value + values * repeat
                        skey = (
                            tid // threads,
                            repeat,
                            r2s.layout_src_tv((tid % threads, value)),
                        )
                        dkey = (
                            tid // threads,
                            repeat,
                            r2s.layout_dst_tv((tid % threads, value)),
                        )
                        assert skey not in source_values
                        source_values[skey] = tuple(reg[index])
                        for stage in range(2):
                            dpart = dst[None, None, None, stage]
                            assert cute.size(dpart.shape) == count
                            assert dkey not in destinations[stage]
                            destinations[stage][dkey] = tuple(dpart[index])
            assert len(source_values) == 2048
            assert set(source_values.values()) == {
                (m, n) for m in range(128) for n in range(16)
            }
            for stage, destination in enumerate(destinations):
                assert destination.keys() == source_values.keys()
                for key, origin in source_values.items():
                    assert destination[key] == (*origin, stage)
            # Both MMA CTAs and all five column subtiles cover the whole
            # physical tile exactly once through the actual TMEM copy layout.
            output_coords = set()
            for rank in range(2):
                identity = epi.transform_partitioned_tensor_layout(
                    mma.get_slice(rank).partition_C(
                        cute.make_identity_tensor((256, 80))
                    )
                )
                divided = cute.flat_divide(identity, ep)
                for column_subtile in range(5):
                    coordinates = []
                    for tid in range(128):
                        part = tc.get_slice(tid).partition_D(divided)[
                            None, None, None, 0, column_subtile
                        ]
                        fragment = r2s.retile(part)
                        coordinates.extend(
                            tuple(fragment[i]) for i in range(cute.size(fragment.shape))
                        )
                    assert len(coordinates) == len(set(coordinates)) == 2048
                    assert set(coordinates) == {
                        (m, n)
                        for m in range(rank * 128, (rank + 1) * 128)
                        for n in range(column_subtile * 16, (column_subtile + 1) * 16)
                    }
                    assert not output_coords.intersection(coordinates)
                    output_coords.update(coordinates)
            assert output_coords == set(itertools.product(range(256), range(80)))


def _kernel_body(source: str) -> list[ast.stmt]:
    return next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    ).body


def _top_level_lines(source: str) -> list[str]:
    return [ast.unparse(stmt) for stmt in _kernel_body(source)]


def _launched_warps(source: str) -> int:
    """Warps of the kernel launch (``block=(x, y, z)`` in the wrapper)."""
    match = re.search(r"block=\((\d+), (\d+), (\d+)\)", source)
    assert match is not None, source
    x, y, z = (int(value) for value in match.groups())
    assert x * y * z % 32 == 0
    return x * y * z // 32


@skipUnlessBackends(["cute"])
def test_tma_warp_skips_prologue_publication(case):
    """The row-union prologue publishes the TMEM allocation (and, on the
    paired profile, the coverage tables) through a named barrier the
    TMA-load warp does not join: it issues its first loads while the
    epilogue warps are still inside ``tcgen05.alloc``."""
    bound, _args, config, source = case
    lines = _top_level_lines(source)
    barrier = next(
        line for line in lines if "tcgen05_row_union_publication_barrier = " in line
    )
    assert barrier.endswith("NamedBarrier(barrier_id=4, num_threads=160)")
    # The count is derived from the launch itself: every launched warp but
    # the one TMA-load warp.
    assert _launched_warps(source) == 6
    skip = lines.index(
        "if not tcgen05_tma_warp:\n    tcgen05_row_union_publication_barrier.arrive_and_wait()"
    )
    allocate = next(i for i, line in enumerate(lines) if ".allocate(" in line)
    retrieve = next(i for i, line in enumerate(lines) if "wait_for_alloc()" in line)
    assert allocate < skip < retrieve
    assert not any("sync_threads()" in line for line in lines[allocate:retrieve])
    # The paired profile launches eight warps (coverage warp 7 writes the
    # SMEM tables the epilogue warps read behind the same barrier).
    paired = deepcopy(config)
    paired.config[SCHEDULE_KEY] = PAIRED_CLC_SCHEDULE
    paired_source = bound.to_code(paired)
    paired_lines = _top_level_lines(paired_source)
    assert any(
        line.endswith("NamedBarrier(barrier_id=4, num_threads=224)")
        for line in paired_lines
    )
    assert _launched_warps(paired_source) == 8
    assert (
        "if not tcgen05_tma_warp:\n    tcgen05_row_union_publication_barrier.arrive_and_wait()"
        in paired_lines
    )
    assert not any(line == "cute.arch.sync_threads()" for line in paired_lines)


@skipUnlessBackends(["cute"])
def test_dependent_launch_needs_no_entry_wait(case):
    """Under ``cute_pdl`` the group offsets are read inside the epilogue
    role behind its own ``griddepcontrol_wait``; nothing reads global memory
    ahead of a role wait, so the kernel gets no entry wait and its prologue
    (barrier init, cluster sync, TMEM allocation) overlaps the previous
    kernel in the stream."""
    from helion._compiler.cute.pdl_coverage import waits_cover_global_reads

    bound, _args, config, _source = case
    pdl = deepcopy(config)
    pdl.config["cute_pdl"] = True
    assert bound.config_spec.normalized_config(pdl)["cute_pdl"] is True
    source = bound.to_code(pdl)
    body = _kernel_body(source)
    assert ast.unparse(body[0]) != "cute.arch.griddepcontrol_wait()"
    assert waits_cover_global_reads(body)
    assert "'use_pdl': True" in source
    epilogue = next(
        stmt
        for stmt in body
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test).endswith("< cutlass.Int32(4)")
    )
    role_lines = [ast.unparse(stmt) for stmt in epilogue.body]
    assert role_lines[0] == "cute.arch.griddepcontrol_wait()"
    offsets_read = next(
        i for i, line in enumerate(role_lines) if "group_offsets.iterator" in line
    )
    persistent_loop = next(
        i for i, stmt in enumerate(epilogue.body) if isinstance(stmt, ast.While)
    )
    assert 0 < offsets_read < persistent_loop
    # Without the knob the epilogue role carries no wait (the two-CTA TMA-load
    # role keeps its unconditional one).  (``cute_pdl`` defaults to True on
    # the sm_90+ tcgen05 surface; the knob-off render pins it off.)
    without_knob = deepcopy(config)
    without_knob.config["cute_pdl"] = False
    plain_epilogue = next(
        stmt
        for stmt in _kernel_body(bound.to_code(without_knob))
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test).endswith("< cutlass.Int32(4)")
    )
    assert "griddepcontrol_wait" not in ast.unparse(plain_epilogue)


@skipUnlessBackends(["cute"])
def test_tma_load_role_is_hoisted_ahead_of_the_cluster_wait(case):
    """The transposed profile's TMA-load role block sits right after the
    cluster ``pipeline_init_arrive``: its scheduler setup overlaps the
    rendezvous and the warp joins it once, in its prelude ahead of the
    persistent loop, while every other warp waits in the prefix."""
    bound, _args, config, source = case
    body = _kernel_body(source)
    lines = [ast.unparse(stmt) for stmt in body]
    arrive = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("cutlass.pipeline.pipeline_init_arrive(")
    )
    cluster_wait = "cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=tcgen05_cluster_layout_vmnk)"
    prefix_wait = lines.index(f"if not tcgen05_tma_warp:\n    {cluster_wait}")
    tma_role = next(
        i
        for i, stmt in enumerate(body)
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test).endswith("== cutlass.Int32(5)")
    )
    assert arrive < tma_role < prefix_wait
    role_lines = [ast.unparse(stmt) for stmt in body[tma_role].body]
    scheduler = next(
        i for i, line in enumerate(role_lines) if "initial_work_tile_info()" in line
    )
    loop = next(
        i for i, stmt in enumerate(body[tma_role].body) if isinstance(stmt, ast.While)
    )
    # Between the scheduler setup and the join: the once-per-CTA TMA
    # partitions of the whole tiled operands (pure address math that
    # overlaps the other warps' barrier inits and the peers' arrival).
    joined = role_lines.index(cluster_wait)
    assert scheduler < joined < loop
    between = role_lines[scheduler + 1 : joined]
    assert sum("tma_partition(" in line for line in between) == 2
    assert all("tma_" in line or line.startswith("#") for line in between), between
    # No unguarded cluster wait remains at the top level.
    assert not any(
        line.startswith("cutlass.pipeline.pipeline_init_wait(") for line in lines
    )
    # The CLC-scheduled paired profile keeps the plain prefix wait.
    paired = deepcopy(config)
    paired.config[SCHEDULE_KEY] = PAIRED_CLC_SCHEDULE
    paired_lines = _top_level_lines(bound.to_code(paired))
    assert cluster_wait in paired_lines
    assert not any(
        line.startswith("if not tcgen05_tma_warp:\n    cutlass.pipeline")
        for line in paired_lines
    )


@skipUnlessBackends(["cute"])
def test_ab_stage_knob_normalizes_projects_and_launches(case):
    """``tcgen05_grouped_row_union_ab_stages`` picks the physical AB ring depth
    of a named profile (``None`` = its own, deepest fitting ring); it is
    searched beside the schedule, rejected strictly / repaired to ``None``
    outside a profile's range or without a physical profile, and carried by
    the wrapper plan so the launcher's descriptor contract follows it."""
    from helion._compiler.cute.grouped_row_union import AB_STAGES_CHOICES
    from helion._compiler.cute.grouped_row_union import AB_STAGES_KEY
    from helion._compiler.cute.grouped_row_union import physical_schedule

    bound, _args, config, source = case
    spec = bound.config_spec
    fields = spec._cute_tcgen05_config.optional_fragments(for_search=True)
    assert tuple(fields[AB_STAGES_KEY].choices) == AB_STAGES_CHOICES
    assert AB_STAGES_KEY not in spec.normalized_config(config).config
    for stages in (4, 6, 8, 10):
        with_knob = deepcopy(config)
        with_knob.config[AB_STAGES_KEY] = stages
        normalized = spec.normalized_config(with_knob)
        assert normalized.config.get(AB_STAGES_KEY) == stages
        assert physical_schedule(normalized.config).ab_stages == stages
    for bad in (
        {AB_STAGES_KEY: 12},
        # Two (one stage of look-ahead) is neither searched nor accepted.
        {AB_STAGES_KEY: 2},
        {AB_STAGES_KEY: 1},
        {AB_STAGES_KEY: 6, SCHEDULE_KEY: "legacy"},
        {AB_STAGES_KEY: 6, CONFIG_KEY: False},
    ):
        invalid = deepcopy(config)
        invalid.config.update(bad)
        with pytest.raises(InvalidConfig):
            spec.normalized_config(invalid)
        repaired = deepcopy(invalid.config)
        spec.normalize(repaired, _fix_invalid=True)
        assert AB_STAGES_KEY not in repaired
    # The paired profile's own ring is 7 deep: 8 is out of range for it.
    paired = deepcopy(config)
    paired.config[SCHEDULE_KEY] = PAIRED_CLC_SCHEDULE
    paired.config[AB_STAGES_KEY] = 8
    with pytest.raises(InvalidConfig):
        spec.normalized_config(paired)
    repaired = deepcopy(paired.config)
    spec.normalize(repaired, _fix_invalid=True)
    assert AB_STAGES_KEY not in repaired
    paired.config[AB_STAGES_KEY] = 6
    assert spec.normalized_config(paired).config[AB_STAGES_KEY] == 6
    # Codegen with six stages: the ring, its producer/consumer states, the
    # mbarrier storage and the wrapper plan all follow the knob; the plan
    # passes the launcher's descriptor contract.
    six = deepcopy(config)
    six.config[AB_STAGES_KEY] = 6
    six_source = bound.to_code(six)
    assert "make_tma_umma_pipeline(num_stages=6," in six_source
    assert "PipelineUserType.Producer, 6)" in six_source
    assert "cutlass.BFloat16, 6, is_k_major=" in six_source
    assert "alloc_smem(cutlass.Int64, cutlass.Int32(12))" in six_source
    plan = _plans(six_source)[0]
    assert (plan["ab_stage_count"], plan["row_union_ab_stages"]) == (6, 6)
    body: list[str] = []
    _append_cute_wrapper_plan(body, [], plan)
    assert any("6" in line and "smem_layout" in line for line in body)
    tampered = dict(plan)
    for depth in (12, 2):
        tampered["row_union_ab_stages"] = depth
        with pytest.raises(BackendUnsupported):
            _append_cute_wrapper_plan([], [], tampered)
    tampered["row_union_ab_stages"] = 6
    tampered["ab_stage_count"] = 8
    with pytest.raises(BackendUnsupported):
        _append_cute_wrapper_plan([], [], tampered)
    # Ten is the profile's own depth: byte-identical to the knob's absence.
    ten = deepcopy(config)
    ten.config[AB_STAGES_KEY] = 10
    assert bound.to_code(ten) == source


@skipUnlessBackends(["cute"])
def test_tma_role_hoist_fails_closed_without_exactly_one_role_block():
    """The hoist moves the one TMA-load role block behind its anchor; a
    setup without it, or with a second one (a scheduler full/edge phase
    split), is reported as unsupported instead of asserted."""
    pids = Tcgen05PersistentProgramIDs.__new__(Tcgen05PersistentProgramIDs)
    predicate = "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)"
    anchor = ast.parse("anchor()").body[0]
    other = ast.parse("x = 1").body[0]
    role = ast.parse(f"if {predicate}:\n    pass").body[0]
    with patch.object(
        Tcgen05PersistentProgramIDs,
        "_tcgen05_plan",
        return_value=SimpleNamespace(tma_warp_id=5),
    ):
        setup = [anchor, other, role]
        pids._hoist_tcgen05_tma_role_block(setup, anchor)
        assert setup == [anchor, role, other]
        for bad in ([anchor, other], [anchor, role, deepcopy(role)]):
            with pytest.raises(
                BackendUnsupported, match="exactly one TMA-load role block"
            ):
                pids._hoist_tcgen05_tma_role_block(bad, anchor)


@skipUnlessBackends(["cute"])
def test_two_cta_row_union_drains_stores_after_the_tmem_free(case):
    """The two-CTA row-union teardown issues the store warp's bulk-store drain
    after the TMEM dealloc handshake so the two overlap (one tile per CTA at
    the catalogue shapes puts the whole teardown on the critical path); the
    dense two-CTA kernels keep the drain-then-dealloc order
    (test_cute_tcgen05_fixed_chain)."""
    bound, _args, config, source = case
    lines = _top_level_lines(source)
    drain = lines.index(
        "if tcgen05_warp_idx == cutlass.Int32(0):\n    tcgen05_c_pipeline.producer_tail()"
    )
    free = next(
        i for i, line in enumerate(lines) if "tcgen05_tmem_allocator.free(" in line
    )
    assert free < drain == len(lines) - 1
    paired = deepcopy(config)
    paired.config[SCHEDULE_KEY] = PAIRED_CLC_SCHEDULE
    paired_lines = _top_level_lines(bound.to_code(paired))
    paired_drain = next(
        i
        for i, line in enumerate(paired_lines)
        if "tcgen05_c_pipeline.producer_tail()" in line
    )
    paired_free = next(
        i
        for i, line in enumerate(paired_lines)
        if "tcgen05_tmem_allocator.free(" in line
    )
    assert paired_free < paired_drain


@skipUnlessBackends(["cute"])
def test_mma_warp_owns_tmem_on_the_transposed_profile(case):
    """The transposed profile's MMA warp allocates TMEM and frees it right
    after its accumulator ``producer_tail`` (every epilogue release follows a
    ``fence_view_async_tmem_load``), so the epilogue warps go straight from
    their last TMEM read to their stores and never meet the MMA warp again;
    the paired CLC profile keeps the epilogue-warp allocation."""
    bound, _args, config, source = case
    lines = _top_level_lines(source)
    allocators = [line for line in lines if "cutlass.utils.TmemAllocator(" in line]
    assert len(allocators) == 2
    assert all("allocator_warp_id=4," in line for line in allocators)
    allocate = next(i for i, line in enumerate(lines) if ".allocate(" in line)
    assert lines[allocate].startswith("if tcgen05_exec_active:\n")
    assert "relinquish_alloc_permit()" in lines[allocate]
    publication = lines.index(
        "if not tcgen05_tma_warp:\n    tcgen05_row_union_publication_barrier.arrive_and_wait()"
    )
    retrieve = next(i for i, line in enumerate(lines) if "wait_for_alloc()" in line)
    assert allocate < publication < retrieve
    acc_tail = next(
        i
        for i, line in enumerate(lines)
        if "tcgen05_acc_pipeline.producer_tail(" in line
    )
    free = next(
        i for i, line in enumerate(lines) if "tcgen05_tmem_allocator.free(" in line
    )
    assert lines[free] == (
        "if tcgen05_exec_active:\n    tcgen05_tmem_allocator.free(tcgen05_exec_acc_tmem_ptr)"
    )
    assert retrieve < acc_tail < free
    # No alloc-barrier meet in the teardown: the only arrive_and_wait on it is
    # the prologue's ``wait_for_alloc``.
    assert not any(
        "tcgen05_tmem_alloc_barrier.arrive" in line for line in lines[acc_tail:]
    )
    paired = deepcopy(config)
    paired.config[SCHEDULE_KEY] = PAIRED_CLC_SCHEDULE
    paired_lines = _top_level_lines(bound.to_code(paired))
    assert all(
        "allocator_warp_id=0," in line
        for line in paired_lines
        if "cutlass.utils.TmemAllocator(" in line
    )
    assert any(
        line
        == "if tcgen05_epi_active:\n    tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)"
        for line in paired_lines
    )


@skipUnlessBackends(["cute"])
def test_tma_load_role_waits_per_tile_ahead_of_its_first_stage(case):
    """Under ``cute_pdl`` the TMA-load role's dependent-launch wait sits inside
    its persistent loop, after the tile's address math (scheduler step, local
    tiles, TMA partitions, the stage-0 gate and the stage-0 acquire: its
    empty-barrier wait and ``expect_tx`` touch only shared memory) and right
    ahead of the stage-0 loads, so only the TMA issue itself follows the
    release; the prelude carries no wait and the release follows the wait.
    Per-CTA stamps on the one-wave grouped kernel put the first TMA issue
    250-300 ns after the release with the prelude wait; waiting ahead of the
    first stage read 3.26 -> 3.15 us per call, waiting behind its acquire
    (with the TMEM allocation overlapping the cluster join) another 35 ns."""
    from helion._compiler.cute.pdl_coverage import waits_cover_global_reads

    bound, _args, config, _source = case
    pdl = deepcopy(config)
    pdl.config["cute_pdl"] = True
    body = _kernel_body(bound.to_code(pdl))
    assert waits_cover_global_reads(body)
    tma_role = next(
        stmt
        for stmt in body
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test).endswith("== cutlass.Int32(5)")
    )
    prelude = [ast.unparse(stmt) for stmt in tma_role.body]
    assert "cute.arch.griddepcontrol_wait()" not in prelude
    loop = next(stmt for stmt in tma_role.body if isinstance(stmt, ast.While))
    lines = [ast.unparse(stmt) for stmt in loop.body]
    wait = lines.index("cute.arch.griddepcontrol_wait()")
    assert lines[wait + 1] == "cute.arch.griddepcontrol_launch_dependents()"
    # The tile's operand views are slices of the once-per-CTA partitions
    # (the role prelude), taken ahead of the wait.
    slices = [i for i, line in enumerate(lines) if "_all[" in line]
    assert len(slices) == 2 and max(slices) < wait
    assert not any("tma_partition(" in line for line in lines)
    assert sum("tma_partition(" in line for line in prelude) == 2
    gate = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("tcgen05_tma_initial_full_tile =")
    )
    acquire = lines.index(
        "if tcgen05_tma_initial_full_tile:\n"
        "    tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state)"
    )
    assert gate < acquire < wait
    # The stage-0 block itself carries no acquire any more.
    assert "producer_acquire" not in lines[wait + 2]
    assert lines[wait + 2].startswith("if tcgen05_tma_initial_full_tile:")
    assert "cute.copy(tma_atom_a" in lines[wait + 2]
    assert not any("cute.copy(" in line for line in lines[:wait])
    # Only the two role waits remain (TMA-load and epilogue); no entry wait.
    source = bound.to_code(pdl)
    assert source.count("cute.arch.griddepcontrol_wait()") == 2
    assert ast.unparse(body[0]) != "cute.arch.griddepcontrol_wait()"


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("rounds", [2, 4])
def test_epilogue_round_knob_renders_rounds_and_sizes_the_c_ring(case, rounds):
    """``tcgen05_epi_round_subtiles`` on the transposed profile: the five
    128x16 subtiles of a tile are staged and stored in rounds of ``rounds``
    (2, 2, 1 or 4, 1), one async-proxy fence, one named barrier and one bulk
    commit per round; the C ring holds a buffer per subtile (whole rounds),
    the bulk-group pipeline a stage per round, and the acquire that guards a
    buffer's reuse is taken only from the second tile of a CTA on."""
    bound, _args, config, _source = case
    knob = deepcopy(config)
    knob.config[AB_STAGES_KEY] = 6
    knob.config[ROUND_SUBTILES_KEY] = rounds
    normalized = bound.config_spec.normalized_config(knob)
    assert normalized[ROUND_SUBTILES_KEY] == rounds
    assert normalized[AB_STAGES_KEY] == 6
    projected = bound.config_spec._cute_tcgen05_config.project_row_union_codegen(
        normalized
    )
    subtiles = TRANSPOSED.epilogue_subtiles
    assert subtiles == 5
    n_rounds = -(-subtiles // rounds)
    c_stages = rounds * n_rounds
    assert projected["tcgen05_c_stages"] == c_stages
    source = bound.to_code(knob)
    kernel = source[source.index("@cute.kernel") :]
    full_tile = kernel[kernel.index("if tcgen05_full_tile:") : kernel.index("else:")]
    assert full_tile.count("cute.arch.fence_view_async_shared()") == n_rounds
    assert full_tile.count("tcgen05_c_pipeline.producer_commit()") == n_rounds
    assert full_tile.count("cute.copy(tcgen05_tma_store_atom,") == subtiles
    assert full_tile.count("cute.copy(tcgen05_tiled_copy_t2r,") == subtiles
    assert full_tile.count("consumer_release(tcgen05_acc_consumer_state)") == 1
    assert full_tile.count("fence_view_async_tmem_load()") == 1
    # The reuse guard: an acquire plus a barrier per round, only from the
    # second tile of the CTA on (the first tile's buffers are fresh).
    assert (
        full_tile.count("if tcgen05_tma_store_role_tile >= cutlass.Int32(1):")
        == n_rounds
    )
    # (plus the per-tile first acquire ahead of the setup, a no-op on a ring
    # with a stage per round)
    assert full_tile.count("tcgen05_c_pipeline.producer_acquire()") == n_rounds + 1
    assert (
        full_tile.count("tcgen05_epilog_sync_barrier.arrive_and_wait()") == 2 * n_rounds
    )
    assert f"% cutlass.Int32({c_stages})" in full_tile
    assert f"tcgen05_store_epi_tile, {c_stages})" in full_tile
    assert f"'num_c_stage': cutlass.Int32({c_stages})" in full_tile
    assert f"PipelineTmaStore.create(num_stages={n_rounds}," in kernel
    # Slot fragments for the round's subtiles beyond the first.
    assert ("tcgen05_tTR_rAcc_r1 = cute.make_rmem_tensor" in full_tile) == (rounds > 1)
    assert ("tcgen05_tTR_rAcc_r3 = cute.make_rmem_tensor" in full_tile) == (rounds > 3)
    assert "for _tcgen05_subtile in" not in full_tile
    # The wrapper's store contract carries the ring depth.
    d_plan = next(plan for plan in _plans(source) if plan["kind"] == "tcgen05_d_tma")
    assert d_plan["c_stage_count"] == c_stages
    # The knob-off render is unchanged: five per-subtile iterations.
    plain = bound.to_code(config)
    assert "for _tcgen05_subtile in cutlass.range(tcgen05_subtile_count" in plain
    assert "PipelineTmaStore.create(num_stages=2," in plain


@skipUnlessBackends(["cute"])
def test_epilogue_round_knob_normalizes_against_the_profile(case):
    """The round must be a choice no wider than the tile's subtiles and its C
    ring must fit beside the physical AB ring; the paired CLC profile keeps
    its own store tail.  Strict normalization refuses, the repair drops the
    knob (1 is canonical as absent)."""
    bound, _args, config, _source = case
    spec = bound.config_spec
    fields = spec._cute_tcgen05_config.optional_fragments(for_search=True)
    assert tuple(fields[ROUND_SUBTILES_KEY].choices) == (1, 2, 4)
    assert ROUND_SUBTILES_KEY not in spec.normalized_config(config).config
    one = deepcopy(config)
    one.config[ROUND_SUBTILES_KEY] = 1
    assert ROUND_SUBTILES_KEY not in spec.normalized_config(one).config
    for stages, rounds in ((8, 2), (6, 4), (7, 2), (8, 4)):
        with_knob = deepcopy(config)
        with_knob.config.update({AB_STAGES_KEY: stages, ROUND_SUBTILES_KEY: rounds})
        assert spec.normalized_config(with_knob).config[ROUND_SUBTILES_KEY] == rounds
    for bad in (
        # The profile's own 10-stage ring leaves no room for the wider C ring.
        {ROUND_SUBTILES_KEY: 2},
        {ROUND_SUBTILES_KEY: 2, AB_STAGES_KEY: 10},
        {ROUND_SUBTILES_KEY: 4, AB_STAGES_KEY: 10},
        {ROUND_SUBTILES_KEY: 3, AB_STAGES_KEY: 6},
        # the legacy one-CTA row-union schedule keeps the per-subtile loop
        {ROUND_SUBTILES_KEY: 2, SCHEDULE_KEY: "legacy", AB_STAGES_KEY: None},
        {ROUND_SUBTILES_KEY: 8, AB_STAGES_KEY: 6},
        {ROUND_SUBTILES_KEY: "2", AB_STAGES_KEY: 6},
        {ROUND_SUBTILES_KEY: 0, AB_STAGES_KEY: 6},
        {ROUND_SUBTILES_KEY: 2, SCHEDULE_KEY: PAIRED_CLC_SCHEDULE},
    ):
        invalid = deepcopy(config)
        invalid.config.update(bad)
        with pytest.raises(InvalidConfig):
            spec.normalized_config(invalid)
        repaired = deepcopy(invalid.config)
        spec.normalize(repaired, _fix_invalid=True)
        assert ROUND_SUBTILES_KEY not in repaired
        expected_schedule = bad.get(SCHEDULE_KEY, TRANSPOSED_SCHEDULE)
        # (the legacy schedule is canonical as absent)
        assert repaired.get(SCHEDULE_KEY) == (
            None if expected_schedule == "legacy" else expected_schedule
        )
        assert repaired[CONFIG_KEY] is True


@skipUnlessBackends(["cute"])
def test_prologue_trims_on_the_transposed_profile(case):
    """The TMA-load warp's prologue is the critical path of the CTAs the
    hardware dispatches last (after the dependent-launch release): the
    clustered kernel builds its pipelines without the DSL's serial warp-0
    mbarrier init and spreads the inits over the idle epilogue warps behind
    ONE guard the TMA-load warp skips (with the two-CTA TMEM dealloc barrier
    init and epilogue warp 0's store descriptor prefetch) ahead of the
    cluster arrive, masks the cluster rank to its
    static range, and divides the (non-negative) tile indices, offsets and
    pids with shifts / masks instead of floor-division fix-up chains."""
    _bound, _args, _config, source = case
    assert (
        "import helion._compiler.cute._tcgen05_prologue_runtime as _helion_tc_pro"
        in source
    )
    lines = _top_level_lines(source)
    acc = next(i for i, line in enumerate(lines) if "tcgen05_acc_pipeline = " in line)
    ab = next(i for i, line in enumerate(lines) if "tcgen05_ab_pipeline = " in line)
    assert "_helion_tc_pro.make_umma_async_pipeline(num_stages=2, " in lines[acc]
    assert "_helion_tc_pro.make_tma_umma_pipeline(num_stages=" in lines[ab]
    assert "defer_sync" not in lines[acc] and "defer_sync" not in lines[ab]
    assert not any("PipelineTmaUmma.create(" in line for line in lines)
    assert not any("PipelineUmmaAsync.create(" in line for line in lines)
    arrive = lines.index(
        "cutlass.pipeline.pipeline_init_arrive(cluster_shape_mn=tcgen05_cluster_layout_vmnk, is_relaxed=True)"
    )
    (inits,) = [
        i
        for i, line in enumerate(lines)
        if line.startswith("if not tcgen05_tma_warp:\n")
        and "init_pipeline_barriers" in line
    ]
    assert max(acc, ab) < inits < arrive
    block = lines[inits]
    dealloc_init = (
        "if tcgen05_warp_idx == cutlass.Int32(4):\n        with cute.arch.elect_one():"
        "\n            cute.arch.mbarrier_init(tcgen05_tmem_dealloc_mbar_ptr, 32)"
    )
    store_prefetch = (
        "if tcgen05_warp_idx == cutlass.Int32(0):"
        "\n        cute.nvgpu.cpasync.prefetch_descriptor(tcgen05_tma_store_atom)"
    )
    pieces = (
        dealloc_init,
        "_helion_tc_pro.init_pipeline_barriers(tcgen05_warp_idx, (0,), tcgen05_acc_pipeline)",
        "_helion_tc_pro.init_pipeline_barriers_pair(tcgen05_warp_idx, 1, tcgen05_ab_pipeline)",
        store_prefetch,
    )
    assert all(piece in block for piece in pieces)
    # Nothing of it is emitted anywhere else, and the allocator constructor
    # no longer inits the dealloc barrier itself.
    assert not any(
        "init_pipeline_barriers" in line
        or "prefetch_descriptor(tcgen05_tma_store_atom)" in line
        or "mbarrier_init(tcgen05_tmem_dealloc_mbar_ptr" in line
        for i, line in enumerate(lines)
        if i != inits
    )
    allocator = next(
        line for line in lines if line.startswith("tcgen05_tmem_allocator = ")
    )
    assert allocator.endswith(", initialize_mbarrier=False)")
    # The cluster rank is masked to its static range (a no-op the compiler
    # folds the coordinate math with) and the MMA slice uses the mask form.
    assert (
        "tcgen05_cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) & cutlass.Int32(3)"
        in lines
    )
    assert (
        "mma_slice_tidx = _helion_tc_pro.mod_nonneg(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()), 2)"
        in lines
    )
    # The TMA role's per-tile head: pids by shift / mask, slices by the
    # non-negative division helper; no floor division or modulo remains.
    tma_role = next(
        stmt
        for stmt in _kernel_body(source)
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test).endswith("== cutlass.Int32(5)")
    )
    loop = next(stmt for stmt in tma_role.body if isinstance(stmt, ast.While))
    head = [ast.unparse(stmt) for stmt in loop.body]
    pid_lines = [line for line in head if line.startswith("pid_")]
    assert pid_lines and all(
        (" >> cutlass.Int32(" in line or " & cutlass.Int32(" in line)
        and " // " not in line
        and " % " not in line
        for line in pid_lines
    ), pid_lines
    virtual_pid = next(line for line in head if line.startswith("virtual_pid = "))
    assert "tile_idx[0] >> cutlass.Int32(1))" in virtual_pid
    assert "tile_idx[0] // " not in virtual_pid
    assert any(
        line
        == "tma_gA = tma_gA_all[None, _helion_tc_pro.floordiv_nonneg(tile_offset_2, 256), None]"
        for line in head
    )
    assert any(
        line
        == "tma_gB = tma_gB_all[None, _helion_tc_pro.floordiv_nonneg(tile_offset_1, 80), None]"
        for line in head
    )
    assert not any(
        " // " in line or " % " in line
        for line in head
        if line.startswith(("pid_", "tile_offset_", "tma_g"))
    )
    assert (
        "tcgen05_tma_k_tile = _helion_tc_pro.floordiv_nonneg(tile_offset_3, 64)"
        in source
    )


@skipUnlessBackends(["cute"])
def test_register_reallocation_optional_on_the_transposed_profile(case):
    """At one CTA per SM the transposed profile's six warps hold their
    registers from launch, so ``tcgen05_warp_register_reallocation=False``
    (no ``setmaxnreg`` pair, a shorter TMA-load warp prologue) is a searchable
    value of that profile; the consumer-register pin stays incompatible with
    it, and the legacy one-CTA schedule and the paired profile keep the pair."""
    bound, _args, config, source = case
    key = "tcgen05_warp_register_reallocation"
    spec = bound.config_spec
    assert "cute.arch.setmaxregister_decrease(" in source
    off = deepcopy(config)
    off.config.pop("tcgen05_consumer_regs", None)
    off.config[key] = False
    assert spec.normalized_config(off)[key] is False
    off_source = bound.to_code(off)
    assert "setmaxregister" not in off_source
    assert "_helion_tc_pro.init_pipeline_barriers_pair(" in off_source
    # The default consumer budget is inert without the pair (a full config
    # materializes it); an explicit other value is refused / repaired.
    default_regs = deepcopy(off)
    default_regs.config["tcgen05_consumer_regs"] = 256
    normalized = spec.normalized_config(default_regs)
    assert normalized[key] is False and "tcgen05_consumer_regs" not in normalized
    pinned = deepcopy(off)
    pinned.config["tcgen05_consumer_regs"] = 240
    with pytest.raises(InvalidConfig):
        spec.normalized_config(pinned)
    repaired = deepcopy(pinned.config)
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.get(key) is False and "tcgen05_consumer_regs" not in repaired
    for schedule in ("legacy", PAIRED_CLC_SCHEDULE):
        other = deepcopy(off)
        other.config[SCHEDULE_KEY] = schedule
        with pytest.raises(InvalidConfig):
            spec.normalized_config(other)
        repaired = deepcopy(other.config)
        spec.normalize(repaired, _fix_invalid=True)
        assert key not in repaired


@skipUnlessBackends(["cute"])
def test_split_init_falls_back_to_the_dsl_constructors(case):
    """When the installed DSL is not the release the runtime's no-init
    constructors mirror, the clustered kernel keeps the DSL constructors
    (``defer_sync=True``, the serial warp-0 inits) and emits no split init."""
    bound, _args, config, _source = case
    with patch(
        "helion._compiler.cute.cute_mma.split_barrier_init_supported",
        return_value=False,
    ):
        source = bound.to_code(config)
    lines = _top_level_lines(source)
    acc = next(line for line in lines if line.startswith("tcgen05_acc_pipeline = "))
    ab = next(line for line in lines if line.startswith("tcgen05_ab_pipeline = "))
    assert acc.startswith(
        "tcgen05_acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create("
    )
    assert ab.startswith(
        "tcgen05_ab_pipeline = cutlass.pipeline.PipelineTmaUmma.create("
    )
    assert acc.endswith("defer_sync=True)") and ab.endswith("defer_sync=True)")
    assert not any("init_pipeline_barriers" in line for line in lines)
    assert (
        "cutlass.pipeline.pipeline_init_arrive(cluster_shape_mn=tcgen05_cluster_layout_vmnk, is_relaxed=True)"
        in lines
    )
