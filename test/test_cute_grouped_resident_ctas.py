from __future__ import annotations

import ast
from copy import deepcopy
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _bound
from test.test_cute_grouped_coverage_search import _search
from test.test_cute_grouped_dense_row_union import _selected
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.grouped_row_union import CONFIG_KEY
from helion._compiler.cute.grouped_row_union import RESIDENT_CTAS_KEY
from helion._compiler.cute.grouped_row_union import ROW_UNION_SHARED_UPPER_BOUND
from helion._compiler.cute.grouped_row_union import ROW_UNION_TMEM_COLUMNS
from helion._compiler.cute.grouped_row_union import GroupedRowUnionPlan
from helion._compiler.cute.grouped_row_union import resident_ctas_supported
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        _target(),
    ):
        yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def case():
    bound, args = _bound(1)
    control = _selected(bound)
    multi = deepcopy(control)
    multi.config[RESIDENT_CTAS_KEY] = 2
    return bound, args, control, multi, bound.to_code(control), bound.to_code(multi)


def calls(tree, suffix):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(suffix)
    ]


def test_only_grid_and_permit_position_change(case):
    control, multi = (ast.parse(text) for text in case[-2:])
    old_kernel, new_kernel = [
        next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        for tree in (control, multi)
    ]
    old_release = calls(old_kernel, ".relinquish_alloc_permit")
    new_release = calls(new_kernel, ".relinquish_alloc_permit")
    assert len(old_release) == len(new_release) == 1
    alloc = calls(new_kernel, ".allocate")
    assert len(alloc) == 1 and len(calls(new_kernel, ".free")) == 1
    old_release_if = next(
        node for node in old_kernel.body if old_release[0] in list(ast.walk(node))
    )
    new_alloc_if = next(
        node for node in new_kernel.body if alloc[0] in list(ast.walk(node))
    )
    assert isinstance(old_release_if, ast.If) and isinstance(new_alloc_if, ast.If)
    assert ast.dump(old_release_if.test) == ast.dump(new_alloc_if.test)
    assert len(new_alloc_if.body) == 2
    assert new_alloc_if.body[0].value is alloc[0]
    assert new_alloc_if.body[1].value is new_release[0]
    assert new_alloc_if in new_kernel.body  # one setup allocation outside all loops
    new_alloc_if.body.pop()
    pos = old_kernel.body.index(old_release_if)
    new_kernel.body.insert(pos, deepcopy(old_release_if))
    # This compares all math, interval masks, publication barriers, ring states,
    # role loops and final free, rather than weakening individual counts.
    assert ast.dump(old_kernel) == ast.dump(new_kernel)
    old_launch = calls(control, "_launcher")
    new_launch = calls(multi, "_launcher")
    assert len(old_launch) == len(new_launch) == 1
    assert "_NUM_SM * 2" in ast.unparse(new_launch[0].args[1])
    new_launch[0].args[1] = deepcopy(old_launch[0].args[1])
    assert ast.dump(control) == ast.dump(multi)


def test_actual_SDK_layouts_bound_two_resident_allocations(case):
    kernel = next(
        node for node in ast.parse(case[-1]).body if isinstance(node, ast.FunctionDef)
    )
    names = {
        "tiled_mma",
        "sA_layout",
        "sB_layout",
        "acc_frag_base",
        "tcgen05_acc_tmem_cols",
    }
    body = [
        node
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert len(body) == len(names)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            ns = {"cutlass": cutlass, "cute": cute}
            exec(
                compile(
                    ast.Module(body=body, type_ignores=[]),
                    "<actual-emitted-layouts>",
                    "exec",
                ),
                ns,
            )
            assert cute.cosize(ns["sA_layout"].outer) == 128 * 128 * 2
            assert cute.cosize(ns["sB_layout"].outer) == 64 * 128 * 2
            assert ns["tcgen05_acc_tmem_cols"] == ROW_UNION_TMEM_COLUMNS == 128
            byte_sizes = []
            for name, rows in (("sA_layout", 128), ("sB_layout", 64)):
                layout = ns[name]
                assert (
                    cute.size_in_bytes(cutlass.BFloat16, layout) == rows * 128 * 2 * 2
                )
                # The emitted outer-layout staging dimension partitions the
                # full ring into two disjoint, contiguous stage spans.
                stage = cute.slice_(layout, (None, None, None, 0))
                stage_bytes = cute.size_in_bytes(cutlass.BFloat16, stage)
                assert stage_bytes == rows * 128 * 2
                byte_sizes.append(2 * stage_bytes)
            assert sum(byte_sizes) == 98304
            assert sum(byte_sizes) + 2048 == ROW_UNION_SHARED_UPPER_BOUND
            assert 2 * ROW_UNION_SHARED_UPPER_BOUND <= 232448
            assert 2 * ROW_UNION_TMEM_COLUMNS <= 512


@pytest.mark.parametrize(
    "sm_count,tile_count",
    list(itertools.product((1, 2, 80, 148, 200), (1, 2, 10, 160, 296, 297))),
)
def test_real_grid_expression_has_complete_disjoint_role_coverage(sm_count, tile_count):
    # The driver reports at most ``sm_count // 4`` co-resident four-CTA
    # clusters (33 on a 148-SM B200, whose GPCs hold uneven TPC counts).
    max_active_clusters = max(1, sm_count * 33 // 148)
    for ctas in (1, 2):
        plan = SimpleNamespace(
            row_union=GroupedRowUnionPlan(
                8, 2560, 512, 512, 0, 1, 2, "offsets", "union", ctas
            )
        )
        strategy = _grid_strategy_stub(cluster_m=1, cluster_n=1, plan=plan, pid_axes=3)
        grid = _grid_work_clusters(strategy, tile_count, sm_count, max_active_clusters)
        assert grid == min(tile_count, sm_count * ctas)
        _assert_complete_disjoint_role_coverage(grid, tile_count)
        if (sm_count, tile_count, ctas) == (148, 160, 2):
            assert all(len(work) == 1 for work in _role_work(grid, tile_count)[0])
    # Batched cluster_n=2 grids (no row-union plan) are capped in whole 4-CTA
    # clusters, at the device's co-resident cluster count, so the persistent
    # CTAs recycle their pipelines across tiles; the coverage stays complete
    # and disjoint under that cap.
    batched = _grid_strategy_stub(cluster_m=2, cluster_n=2, plan=None, pid_axes=3)
    grid = _grid_work_clusters(batched, tile_count, sm_count, max_active_clusters)
    assert grid == min(tile_count, max_active_clusters)
    _assert_complete_disjoint_role_coverage(grid, tile_count)
    # The CtaGroup.TWO pair keeps the exact ``_NUM_SM // 2`` capacity.
    pair = _grid_strategy_stub(cluster_m=2, cluster_n=1, plan=None, pid_axes=3)
    grid = _grid_work_clusters(pair, tile_count, sm_count, max_active_clusters)
    assert grid == min(tile_count, max(1, sm_count // 2))
    _assert_complete_disjoint_role_coverage(grid, tile_count)


def _grid_strategy_stub(
    *, cluster_m: int, cluster_n: int, plan: object, pid_axes: int
) -> SimpleNamespace:
    """A ``Tcgen05PersistentProgramIDs`` stand-in for the grid-capacity helpers.

    The grid-shape helpers are run unbound on this namespace, so it carries
    the attributes they read (``pid_info`` is only measured for its length)
    and the real batched-grid predicates; the host constexpr for the
    co-resident cluster count is named directly since it needs a device
    function to be defined.
    """
    strategy = SimpleNamespace(
        _tcgen05_cluster_m=lambda: cluster_m,
        _tcgen05_cluster_n=lambda: cluster_n,
        _tcgen05_plan=lambda: plan,
        _tcgen05_max_active_clusters_var=lambda: "_MAX_ACTIVE_CLUSTERS",
        pid_info=(None,) * pid_axes,
        grid_size_expr="_NUM_SM",
    )
    for name in (
        "_tcgen05_batched_grid",
        "_tcgen05_batched_cluster_n2",
        "_tcgen05_whole_cluster_capacity_expr",
    ):
        method = getattr(Tcgen05PersistentProgramIDs, name)
        setattr(
            strategy,
            name,
            lambda *args, _method=method, **kwargs: _method(strategy, *args, **kwargs),
        )
    return strategy


def _grid_work_clusters(
    strategy, tile_count: int, sm_count: int, max_active_clusters: int
) -> int:
    capacity = Tcgen05PersistentProgramIDs._tcgen05_max_persistent_work_clusters_expr(
        strategy
    )
    strategy._tcgen05_max_persistent_work_clusters_expr = lambda value=capacity: value
    expr = Tcgen05PersistentProgramIDs._tcgen05_grid_work_clusters_expr(
        strategy, str(tile_count)
    )
    return eval(
        expr, {"_NUM_SM": sm_count, "_MAX_ACTIVE_CLUSTERS": max_active_clusters}
    )


def _role_work(grid: int, tile_count: int) -> list[list[list[int]]]:
    return [
        [list(range(block, tile_count, grid)) for block in range(grid)]
        for _ in range(3)
    ]


def _assert_complete_disjoint_role_coverage(grid: int, tile_count: int) -> None:
    role_work = _role_work(grid, tile_count)
    assert role_work[0] == role_work[1] == role_work[2]
    flat = [i for work in role_work[0] for i in work]
    assert len(flat) == len(set(flat)) == tile_count
    assert sorted(flat) == list(range(tile_count))


@pytest.mark.parametrize("value", (None, True, False, 0, -1, 3, 4, 2.0, "2"))
def test_invalid_residency_rejects(case, value):
    bound, _args, control = case[:3]
    invalid = deepcopy(control)
    invalid.config[RESIDENT_CTAS_KEY] = value
    with pytest.raises(InvalidConfig):
        bound.to_code(invalid)


def test_default_normalization_and_unsupported_domain(case):
    bound, _args, control, multi, old = case[:5]
    explicit = deepcopy(control)
    explicit.config[RESIDENT_CTAS_KEY] = 1
    assert bound.to_code(explicit) == old
    assert RESIDENT_CTAS_KEY not in bound.config_spec.normalized_config(explicit)
    for updates in (
        {CONFIG_KEY: False},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_ab_stages": 3},
        {"num_sm_multiplier": 2},
        {"block_sizes": [128, 128, 128]},
    ):
        invalid = deepcopy(multi)
        invalid.config.update(updates)
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(invalid)


def test_capacity_is_checked_again_at_codegen(case):
    bound, _args, _control, multi = case[:4]
    with (
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            return_value=2 * ROW_UNION_SHARED_UPPER_BOUND - 1,
        ),
        pytest.raises(BackendUnsupported, match="dense row-union requires"),
    ):
        bound.to_code(multi)


def test_resource_bound_and_normal_search_coordinate(case):
    assert resident_ctas_supported(2, 2 * ROW_UNION_SHARED_UPPER_BOUND)
    assert not resident_ctas_supported(2, 2 * ROW_UNION_SHARED_UPPER_BOUND - 1)
    assert not resident_ctas_supported(2, True)
    bound, args, _control, multi = case[:4]
    group = next(
        g for g in bound.config_spec.compiler_coverage_groups if g.key == CONFIG_KEY
    )
    assert [w.carrier.get(RESIDENT_CTAS_KEY, 1) for w in group.witnesses] == [1]
    resident = next(
        g
        for g in bound.config_spec.compiler_coverage_groups
        if g.key == RESIDENT_CTAS_KEY
    )
    assert len(resident.witnesses) == 1 and resident.witnesses[0].value == 2
    (dependency,) = resident.dependencies
    assert (dependency.mechanism, dependency.key, dependency.value) == (
        group.mechanism,
        CONFIG_KEY,
        True,
    )
    assert resident.witnesses[0].carrier[CONFIG_KEY] is True
    assert RESIDENT_CTAS_KEY not in resident.witnesses[0].carrier
    assert (
        group.witnesses[0].carrier["block_sizes"]
        is not resident.witnesses[0].carrier["block_sizes"]
    )
    fields = bound.config_spec._cute_tcgen05_config.flat_fields()
    assert fields[RESIDENT_CTAS_KEY].choices == (1, 2)
    search = _search(bound, args)
    with bound.env:
        _flat, effective = search.config_gen.strict_config_pair(multi)
    assert effective.get(RESIDENT_CTAS_KEY) == 2
