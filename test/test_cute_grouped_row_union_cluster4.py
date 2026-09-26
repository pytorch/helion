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

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _bound
from test.test_cute_grouped_coverage_search import _search
from test.test_cute_grouped_k128_full_coverage import _bound as _generic_bound
from test.test_cute_shared_rhs_grouped import _OffsetPointer
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.grouped_row_union import CONFIG_KEY
from helion._compiler.cute.grouped_row_union import PAIRED_CLC_SCHEDULE
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
            patch(
                "cutlass.cute.compile", side_effect=AssertionError("native forbidden")
            ) as cute_compilation,
            patch(
                "cutlass.compile", side_effect=AssertionError("native forbidden")
            ) as cutlass_compilation,
            _target(),
        ):
            yield
            for prohibited in (
                current_device,
                capability,
                properties,
                initialization,
                synchronize,
                cute_compilation,
                cutlass_compilation,
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
    assert "_NUM_SM // 4" in source
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
    assert previous_seeds[-1][SCHEDULE_KEY] == TRANSPOSED_SCHEDULE
    assert all(SCHEDULE_KEY not in seed for seed in previous_seeds[:-1])
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
    statements: list[ast.stmt] = [
        node
        for node in device.body
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0])
            in {"row_union_starts", "row_union_ends", "row_union_all_rows"}
        )
        or (
            isinstance(node, ast.If)
            and any(
                isinstance(child, ast.For)
                and ast.unparse(child.target) == "row_union_group"
                for child in node.body
            )
        )
    ]
    assert len(statements) == 4
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
    partitions = [
        n
        for n in work.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
    ]
    assert len(partitions) == 2
    first = next(n for n in work.body if isinstance(n, ast.If))
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
                    scope: dict[str, Any] = {"cute": cute, "cutlass": cutlass}
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
                        for name in ("gA_tma", "gB_tma", "gA_tma_part", "gB_tma_part"):
                            execute(assignments[name], scope)
                        for node in partitions:
                            execute(node, scope)
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
