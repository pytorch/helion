from __future__ import annotations

import ast
from copy import deepcopy
from itertools import count
from types import SimpleNamespace

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_ab_k import _unsupported_k_loop
from test.test_cute_grouped_d_shift import _args
from test.test_cute_grouped_d_shift import _config as _d_shift_config
from test.test_cute_grouped_d_shift import _plans
from test.test_cute_grouped_d_shift import _unsafe_output
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel

from helion._compiler.cute.grouped_balanced_workers import GroupedBalancedWorkersPlan
from helion._compiler.cute.grouped_balanced_workers import balanced_workers_index_domain
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes

KEY = "tcgen05_grouped_balanced_workers"


def _config(source_m: int = 128):
    config = _d_shift_config("mn")
    config.config["tcgen05_grouped_worklist_source_m_tile"] = source_m
    config.config[KEY] = True
    return config


def _is_call(node: ast.AST, path: str) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == path


def _source(kind="offsets", dtype=torch.int32, source_m=128, *, shifted=True):
    kernel = _device_offsets_kernel if kind == "offsets" else _device_split_sizes_kernel
    args = _args(kind, dtype)
    config = _config(source_m)
    config.config["tcgen05_grouped_d_shift"] = shifted
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        assert bound.config_spec._cute_tcgen05_config.grouped_balanced_workers_supported
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.block_sizes[2] == 64
        source = bound.to_code(normalized)
    return source, args


@pytest.mark.parametrize("kind", ["offsets", "sizes"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("source_m", [128, 256])
def test_clipped_device_metadata_controls_all_peers_and_scheduler_stride(
    kind: str, dtype: torch.dtype, source_m: int
) -> None:
    source, args = _source(kind, dtype, source_m)
    tree = ast.parse(source)
    grouped, ab, output = _plans(source)
    assert grouped["balanced_workers"] is True
    assert grouped["bm"] == source_m
    assert grouped["bn"] == 256
    assert grouped["cluster_m"] == 2 and grouped["cluster_n"] == 1
    assert grouped["bounded_output_shift"] is True
    _validate_tcgen05_grouped_device_split_sizes(grouped, args[2])

    grid = next(
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("tcgen05_grouped_virtual_grid")
    )
    creates = [
        call
        for call in ast.walk(tree)
        if _is_call(call, "cutlass.utils.StaticPersistentGroupTileScheduler.create")
    ]
    assert creates
    for call in creates:
        assert isinstance(call.args[2], ast.Name) and call.args[2].id == grid
        assert ast.literal_eval(call.args[3]) == (source_m, 256, 64)

    admission = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "tcgen05_grouped_total_tiles" in ast.unparse(node.test)
        and "block_idx()[2]" in ast.unparse(node.test)
    )
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    )
    assert not any(isinstance(node, ast.Return) for node in ast.walk(kernel))
    assert "block_idx()[2]" in ast.unparse(admission.test)
    assert "block_idx()[0]" not in ast.unparse(admission.test)
    assert "block_idx_in_cluster" not in ast.unparse(admission.test)
    assert admission.lineno > source[: source.index("warp_reduction_sum")].count("\n")
    setup_calls = [
        node
        for node in ast.walk(tree)
        if any(
            _is_call(node, path)
            for path in (
                "cutlass.utils.TmemAllocator",
                "cutlass.pipeline.NamedBarrier",
                "cutlass.pipeline.PipelineUmmaAsync.create",
                "cutlass.pipeline.PipelineTmaUmma.create",
                "cute.arch.setmaxregister_decrease",
                "cute.arch.setmaxregister_increase",
            )
        )
    ]
    assert setup_calls
    assert all(call.lineno > admission.lineno for call in setup_calls)
    guarded_nodes = {id(node) for node in ast.walk(admission)}
    assert all(id(call) in guarded_nodes for call in setup_calls)
    tails = [
        call
        for call in ast.walk(kernel)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("producer_tail", "relinquish_alloc_permit", "free")
    ]
    assert tails and all(id(call) in guarded_nodes for call in tails)
    reduction = next(
        call
        for call in ast.walk(tree)
        if _is_call(call, "cute.arch.warp_reduction_sum")
    )
    assert id(reduction) not in guarded_nodes
    publications = [
        call
        for call in ast.walk(tree)
        if _is_call(call, "cute.arch.sync_threads") and call.lineno < reduction.lineno
    ]
    assert len(publications) == 1
    assert source.count("warp_reduction_sum") == 1
    assert "TensorMapManager" not in source

    # Host launch bounds remain static and do not depend on CUDA routing values.
    body, call_args = [], []
    _append_cute_wrapper_plan(body, call_args, grouped, num_sm=152)
    assert not any(".item(" in line or ".tolist(" in line for line in body)
    assert not any("warp_reduction" in line for line in body)
    assert grouped["sched_params_arg"] in call_args
    assert ab["full_allocation_a"] and ab["full_allocation_b"]
    assert output["bounded_output_shift"]


@pytest.mark.parametrize("source_m", [128, 256])
def test_worker_count_is_independent_of_mutable_or_shifted_output_maps(
    source_m,
) -> None:
    source, args = _source(source_m=source_m, shifted=False)
    grouped = _plans(source)[0]
    assert grouped["balanced_workers"]
    assert grouped["dynamic_d_tensormap"]
    _validate_tcgen05_grouped_device_split_sizes(grouped, args[2])
    assert "tcgen05_grouped_virtual_grid" in source


class _Sizes:
    def __init__(self, rows):
        self.rows = rows
        self.accesses = []

    def __getitem__(self, index):
        self.accesses.append(index)
        return self.rows[index[0]][index[1]]


def _model(rows, *, collective_m=128, collective_n=256, maximum=76, cluster_n=1):
    serial = count()
    plan = GroupedBalancedWorkersPlan(
        "total", "workers", "grid", collective_m, collective_n, 2, cluster_n
    )
    statements = plan.setup_statements(
        "sizes", len(rows), lambda name: f"{name}_{next(serial)}"
    )
    state = SimpleNamespace(lane=0, peer=0, peer_n=0, cluster=0, total=0)
    sizes = _Sizes(rows)
    arch = SimpleNamespace(
        lane_idx=lambda: state.lane,
        grid_dim=lambda: (2, cluster_n, maximum),
        block_idx=lambda: (state.peer, state.peer_n, state.cluster),
        warp_reduction_sum=lambda partial, **kwargs: state.total,
    )
    env = {
        "sizes": sizes,
        "cute": SimpleNamespace(arch=arch),
        "cutlass": SimpleNamespace(
            Int32=int, range=lambda *args, **kwargs: range(*args)
        ),
    }
    prefix = compile(
        ast.fix_missing_locations(ast.Module(body=statements[:2], type_ignores=[])),
        "<device tile count>",
        "exec",
    )
    partial_name = statements[0].targets[0].id
    partials = []
    for lane in range(32):
        state.lane = lane
        values = dict(env)
        exec(prefix, values)
        partials.append(values[partial_name])
    state.total = sum(partials)
    functions = []
    for name in ("launch", "count_workers"):
        result = ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id=field, ctx=ast.Load())
                    for field in ("total", "workers", "grid")
                ],
                ctx=ast.Load(),
            )
        )
        body = deepcopy(statements[2:])
        if name == "launch":
            body.append(
                ast.If(
                    test=ast.parse(plan.active_predicate, mode="eval").body,
                    body=[result],
                    orelse=[ast.Return()],
                )
            )
        else:
            body.append(result)
        function = ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=partial_name)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "<device worker admission>",
                "exec",
            ),
            env,
        )
        functions.append(env[name])
    return state, sizes, partials, functions


@pytest.mark.parametrize(
    "rows, expected_total, expected_workers",
    [
        ([(128, 1536)] * 16, 96, 48),
        ([(2048, 1536), *[(1, 1536)] * 6, *[(0, 1536)] * 9], 132, 66),
        ([(0, 1536)] * 16, 0, 1),
        ([(1, 1)], 1, 1),
        ([(129, 257), (0, 257), (127, 257)], 6, 6),
        ([(1, 256)] * 65, 65, 65),
    ],
)
def test_emitted_count_handles_empty_groups_tails_and_multiple_warp_windows(
    rows, expected_total, expected_workers
) -> None:
    state, sizes, partials, (launch, compute) = _model(rows)
    assert state.total == expected_total
    assert sorted(sizes.accesses) == sorted(
        (group, dim) for group in range(len(rows)) for dim in (0, 1)
    )
    assert compute(partials[0]) == (
        expected_total,
        expected_workers,
        (2, 1, expected_workers),
    )
    for cluster in range(76):
        state.cluster = cluster
        for peer in (0, 1):
            state.peer = peer
            result = launch(partials[0])
            active = expected_total > 0 and cluster < expected_workers
            assert (result is not None) is active
            if active:
                assert result == (
                    expected_total,
                    expected_workers,
                    (2, 1, expected_workers),
                )


@pytest.mark.parametrize("maximum", [1, 2, 7, 32, 76])
def test_balancing_preserves_every_tile_with_the_minimum_number_of_waves(
    maximum,
) -> None:
    state, sizes, partials, (launch, compute) = _model([], maximum=maximum)
    for total in (*range(maximum * 4 + 1), (1 << 31) - 1):
        state.total = total
        actual_total, workers, grid = compute(0)
        assert actual_total == total and grid == (2, 1, workers)
        assert 1 <= workers <= maximum
        if total == 0:
            assert launch(0) is None
            continue
        waves = (total + maximum - 1) // maximum
        assert (total + workers - 1) // workers == waves
        assert workers == 1 or (total + workers - 2) // (workers - 1) > waves
        if total < 1000:
            assignments = [
                list(range(worker, total, workers)) for worker in range(workers)
            ]
            assert sorted(tile for tiles in assignments for tile in tiles) == list(
                range(total)
            )
            lengths = [len(tiles) for tiles in assignments]
            assert max(lengths) - min(lengths) <= 1
    assert sizes.accesses == []


@pytest.mark.parametrize("value", [False, None])
def test_default_off_is_canonical_and_keeps_source_unchanged(value) -> None:
    args = _args("offsets", torch.int32)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_device_offsets_kernel, args)
        control = _d_shift_config("mn")
        source = bound.to_code(bound.config_spec.normalized_config(control))
        control.config[KEY] = value
        config = bound.config_spec.normalized_config(control)
        assert KEY not in config.config
        assert bound.to_code(config) == source
    assert "balanced_workers" not in source
    assert "virtual_grid" not in source


@pytest.mark.parametrize(
    "updates",
    [
        {"tcgen05_grouped_worklist_orientation": "nm"},
        {"tcgen05_cluster_n": 3},
        {"num_sm_multiplier": 2},
        {KEY: 1},
        {KEY: "true"},
    ],
)
def test_unsupported_profiles_and_config_values_fail_closed(updates) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_device_offsets_kernel, _args("offsets", torch.int32))
        config = _config()
        config.config.update(updates)
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(config)


@pytest.mark.parametrize("variant", range(4))
def test_noncanonical_k_effects_do_not_gain_a_worker_balance_domain(variant) -> None:
    args = (*_args("offsets", torch.int32), variant)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsupported_k_loop, args)
        assert not bound.config_spec._cute_tcgen05_config.grouped_balanced_workers_supported
        config = _config()
        config.config["tcgen05_grouped_d_shift"] = False
        config.config["tcgen05_grouped_full_allocation_a"] = False
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(config)


@pytest.mark.parametrize("variant", [0, 1])
def test_alias_or_strided_output_cannot_reassign_persistent_workers(variant) -> None:
    a, b, offsets = _args("offsets", torch.int64)
    b = torch.empty((8, 256, 256), dtype=a.dtype)
    args = (a, b, offsets, variant)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsafe_output, args)
        assert not bound.config_spec._cute_tcgen05_config.grouped_balanced_workers_supported
        config = _config()
        config.config["tcgen05_grouped_d_shift"] = False
        config.config["tcgen05_grouped_full_allocation_a"] = False
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(config)


@pytest.mark.parametrize(
    "domain",
    [
        (0, 128, 256, 128, 256),
        (1, (1 << 31) - 1, 256, 128, 256),
        (1, 128, (1 << 31) - 1, 128, 256),
        ((1 << 31) - 1, 129, 256, 128, 256),
        (1 << 20, 1 << 20, 1 << 20, 128, 256),
    ],
)
def test_signed_sdk_prefix_sum_and_padded_coordinate_limits(domain) -> None:
    assert not balanced_workers_index_domain(*domain)


def test_runtime_plan_rejects_changed_cluster_geometry() -> None:
    source, args = _source()
    plan = _plans(source)[0]
    plan["cluster_n"] = 3
    with pytest.raises(BackendUnsupported):
        _validate_tcgen05_grouped_device_split_sizes(plan, args[2])


def _quad_source(kind="offsets", dtype=torch.int32, source_m=128, n_size=768):
    a, b, metadata = _args(kind, dtype)
    args = (a, torch.empty((8, n_size, 256), dtype=a.dtype), metadata)
    config = _config(source_m)
    config.config["tcgen05_cluster_n"] = 2
    config.config["tcgen05_grouped_worklist_mma_n"] = 192
    kernel = _device_offsets_kernel if kind == "offsets" else _device_split_sizes_kernel
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        source = bound.to_code(bound.config_spec.normalized_config(config))
    return source, args


@pytest.mark.parametrize("kind", ["offsets", "sizes"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("source_m", [128, 256])
def test_complete_n_quad_uses_collective_n_width_and_whole_cluster_grid(
    kind: str, dtype: torch.dtype, source_m: int
) -> None:
    source, args = _quad_source(kind, dtype, source_m)
    tree = ast.parse(source)
    plan = _plans(source)[0]
    assert plan["balanced_workers"] and plan["cluster_n"] == 2
    assert plan["bn"] == 192
    _validate_tcgen05_grouped_device_split_sizes(plan, args[2])
    grid = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("tcgen05_grouped_virtual_grid")
    )
    assert isinstance(grid, ast.Tuple)
    assert [ast.literal_eval(value.args[0]) for value in grid.elts[:2]] == [2, 2]
    creates = [
        call
        for call in ast.walk(tree)
        if _is_call(call, "cutlass.utils.StaticPersistentGroupTileScheduler.create")
    ]
    assert creates
    assert all(
        ast.literal_eval(call.args[3]) == (source_m, 384, 64) for call in creates
    )
    columns = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("tcgen05_balance_partial")
        and isinstance(node.value, ast.BinOp)
    )
    assert "cutlass.Int32(383)" in ast.unparse(columns)
    assert "// cutlass.Int32(384)" in ast.unparse(columns)
    admission = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "tcgen05_grouped_total_tiles" in ast.unparse(node.test)
    )
    assert "block_idx()[2]" in ast.unparse(admission.test)
    assert "block_idx()[0]" not in ast.unparse(admission.test)
    assert "block_idx()[1]" not in ast.unparse(admission.test)
    body, call_args = [], []
    _append_cute_wrapper_plan(body, call_args, plan, num_sm=152)
    assert "cutlass.Int32(38)" in "\n".join(body)


@pytest.mark.parametrize(
    "rows, total, workers",
    [
        ([(128, 1536)] * 16, 64, 32),
        ([(2048, 1536), *[(1, 1536)] * 6, *[(0, 1536)] * 9], 88, 30),
        ([(0, 1536)] * 16, 0, 1),
    ],
)
def test_quad_wave_count_and_admission_agree_for_all_four_peers(rows, total, workers):
    state, sizes, partials, (launch, compute) = _model(
        rows, collective_n=384, maximum=38, cluster_n=2
    )
    assert state.total == total
    assert compute(partials[0]) == (total, workers, (2, 2, workers))
    for cluster in range(38):
        state.cluster = cluster
        for peer_m in (0, 1):
            state.peer = peer_m
            for peer_n in (0, 1):
                state.peer_n = peer_n
                result = launch(partials[0])
                assert (result is not None) is (total > 0 and cluster < workers)
                if result is not None:
                    assert result == (total, workers, (2, 2, workers))


def test_quad_feature_tail_is_rejected_at_codegen_and_runtime_admission() -> None:
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        _quad_source(n_size=576)
    source, args = _quad_source()
    plan = _plans(source)[0]
    plan["n_size"] = 576
    with pytest.raises(BackendUnsupported, match="balanced"):
        _validate_tcgen05_grouped_device_split_sizes(plan, args[2])
