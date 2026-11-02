from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_d_shift import _plans
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from test.test_cute_grouped_mn_224 import _args
from test.test_cute_grouped_mn_224 import _grouped
from test.test_cute_grouped_mn_224 import _profile

from helion._compiler.cute import cute_mma
from helion._compiler.cute.grouped_panel_raster import grouped_device_panel_index_domain
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes


def _config(swizzle=8, cluster_n=1):
    config = _profile(depth=2)
    config.config["tcgen05_l2_swizzle_size"] = swizzle
    config.config["tcgen05_cluster_n"] = cluster_n
    if cluster_n == 2:
        config.config["tcgen05_grouped_worklist_mma_n"] = 128
    return config


def _emit(config, args=None):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        if args is None:
            args = _args()
            if config.get("tcgen05_cluster_n", 1) == 2:
                args = (
                    torch.empty((768, 256), dtype=torch.bfloat16),
                    args[1],
                    torch.arange(9, dtype=torch.int64) * 96,
                )
        bound = _cpu_bind(_grouped, args)
        assert (
            bound.config_spec._cute_tcgen05_config.grouped_device_mn_swizzle_supported
        )
        normalized = bound.config_spec.normalized_config(config)
        assert (
            normalized["tcgen05_l2_swizzle_size"] == config["tcgen05_l2_swizzle_size"]
        )
        assert normalized.block_sizes[2] == 64
        return bound.to_code(normalized)


def _panel_block(source):
    tree = ast.parse(source)
    local_name = "tcgen05_grouped_panel_local_index"
    leader = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == local_name
                for target in statement.targets
            )
            for statement in node.body
        )
    )
    start = next(
        index
        for index, node in enumerate(leader.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == local_name
            for target in node.targets
        )
    )
    stop = next(
        index
        for index in range(start, len(leader.body))
        if isinstance(leader.body[index], ast.Assign)
        and isinstance(leader.body[index].targets[0], ast.Subscript)
    )
    # Panel decoding is dominated by the SDK's valid matched-group loop and
    # happens before the first mailbox write, so zero-tile groups never divide.
    assert any(
        isinstance(node, ast.While)
        and ast.unparse(node.test).endswith(".is_valid_tile")
        and leader in node.body
        for node in ast.walk(tree)
    )
    return ast.Module(body=leader.body[start:stop], type_ignores=[])


@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("swizzle", [2, 4, 8, 16])
def test_actual_decoder_is_a_complete_peer_preserving_bijection(cluster_n, swizzle):
    original = cute_mma.CuteTcgen05GroupedPlan

    def dynamic_plan(*args, **kwargs):
        return replace(original(*args, **kwargs), device_n_size=None)

    with patch.object(cute_mma, "CuteTcgen05GroupedPlan", side_effect=dynamic_plan):
        source = _emit(_config(swizzle, cluster_n))
    block = compile(_panel_block(source), "<actual bounded panel>", "exec")
    bm, bn = 256, 128 if cluster_n == 2 else 224
    # Includes a single panel, a short last panel, a one-row tail, and multiple
    # empty groups between live groups. Nonzero prefixes test SDK state reuse.
    for m_clusters in (1, 2, 3, 7, 16):
        for n_clusters in (1, 2, 3, 7, 9, 17):
            previous = 193
            rows = (m_clusters - 1) * bm + 1
            columns = n_clusters * bn * cluster_n
            observed = set()
            for local in range(m_clusters * n_clusters):
                for peer in range(cluster_n):
                    values = {
                        "cutlass": SimpleNamespace(Int32=int, min=min),
                        "tcgen05_grouped_selected_sched": SimpleNamespace(
                            _current_work_linear_idx=previous + local,
                            search_state=SimpleNamespace(
                                tile_count_prev_group=previous
                            ),
                        ),
                        "tcgen05_grouped_selected_group_search_result": SimpleNamespace(
                            problem_shape_m=rows, problem_shape_n=columns
                        ),
                        "tcgen05_grouped_selected_cta_tile_idx_m": local % m_clusters,
                        "tcgen05_grouped_selected_cta_tile_idx_n": (local // m_clusters)
                        * cluster_n
                        + peer,
                    }
                    exec(block, values)
                    m = values["tcgen05_grouped_selected_cta_tile_idx_m"]
                    n = values["tcgen05_grouped_selected_cta_tile_idx_n"]
                    assert 0 <= m < m_clusters and 0 <= n < n_clusters * cluster_n
                    assert n % cluster_n == peer
                    assert (m, n) not in observed
                    observed.add((m, n))
            assert observed == {
                (m, n) for m in range(m_clusters) for n in range(n_clusters * cluster_n)
            }


def test_uniform_two_by_seven_order_matches_baseline_along_n_panel():
    block = compile(
        _panel_block(_emit(_config(), _args(n=1536))), "<uniform panel>", "exec"
    )
    order = []
    for local in range(14):
        values = {
            "cutlass": SimpleNamespace(Int32=int, min=min),
            "tcgen05_grouped_selected_sched": SimpleNamespace(
                _current_work_linear_idx=local,
                search_state=SimpleNamespace(tile_count_prev_group=0),
            ),
            "tcgen05_grouped_selected_group_search_result": SimpleNamespace(
                problem_shape_m=512, problem_shape_n=1536
            ),
            "tcgen05_grouped_selected_cta_tile_idx_m": local % 2,
            "tcgen05_grouped_selected_cta_tile_idx_n": local // 2,
        }
        exec(block, values)
        order.append(
            (
                values["tcgen05_grouped_selected_cta_tile_idx_m"],
                values["tcgen05_grouped_selected_cta_tile_idx_n"],
            )
        )
    assert order == [(m, n) for m in range(2) for n in range(7)]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_live_offset_metadata_remains_device_read_and_runtime_plan_is_checked(dtype):
    args = _args(dtype=dtype)
    source = _emit(_config(), args)
    plan, ab, output = _plans(source)
    assert plan["device_panel_swizzle"] is True
    assert plan["immutable_ab_tensormaps"] is True
    assert ab["full_allocation_b_rank3"] is True
    assert output["bounded_output_shift"] is True
    assert "StaticPersistentGroupTileScheduler.create" in source
    assert "tcgen05_grouped_problem_sizes_smem_ptr" in source
    with _forbid_cuda_work():
        _validate_tcgen05_grouped_device_split_sizes(plan, args[2])
        for bad in (
            {"device_panel_swizzle": False},
            {"cluster_n": 2},
            {"orientation": "nm"},
        ):
            with pytest.raises(BackendUnsupported):
                _validate_tcgen05_grouped_device_split_sizes(plan | bad, args[2])


def test_default_swizzle_one_preserves_the_original_emission():
    config = _config(swizzle=1)
    explicit = _emit(config)
    config.config.pop("tcgen05_l2_swizzle_size")
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args())
        omitted = bound.to_code(config)
    assert omitted == explicit
    assert "tcgen05_grouped_panel_local_index" not in omitted
    assert "device_panel_swizzle" not in omitted


def test_empty_and_skewed_offsets_remain_live_inputs_to_the_same_generated_code():
    config = _config()
    uniform = _args(dtype=torch.int64)
    empty = (*uniform[:2], torch.zeros(9, dtype=torch.int64))
    skewed = (
        *uniform[:2],
        torch.tensor([0, 0, 1, 129, 257, 385, 513, 639, 640], dtype=torch.int64),
    )
    assert _emit(config, uniform) == _emit(config, empty) == _emit(config, skewed)


def test_quad_global_m_edge_keeps_the_existing_split_scheduler_guard():
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args())
        with pytest.raises(BackendUnsupported, match="split full/fringe scheduler"):
            bound.to_code(_config(cluster_n=2))


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_noncanonical_loop_cannot_claim_device_swizzle(variant):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, (*_args(), variant))
        assert not bound.config_spec._cute_tcgen05_config.grouped_device_mn_swizzle_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config())


def test_split_sizes_and_nm_stay_on_their_existing_raster_routes():
    args = list(_args())
    args[2] = torch.full((8,), 80, dtype=torch.int32)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_device_split_sizes_kernel, tuple(args))
        assert not bound.config_spec._cute_tcgen05_config.grouped_device_mn_swizzle_supported
        with pytest.raises(InvalidConfig):
            bound.to_code(_config())
        bound = _cpu_bind(_grouped, _args())
        config = _config()
        config.config["tcgen05_grouped_worklist_orientation"] = "nm"
        with pytest.raises(InvalidConfig):
            bound.to_code(config)


def test_incomplete_cluster_n_and_invalid_index_domains_are_rejected():
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(
            _grouped,
            (
                torch.empty((768, 256), dtype=torch.bfloat16),
                torch.empty((8, 160, 256), dtype=torch.bfloat16),
                torch.arange(9, dtype=torch.int32) * 96,
            ),
        )
        with pytest.raises(BackendUnsupported, match="mn_complete_n_clusters"):
            bound.to_code(_config(cluster_n=2))
        bound = _cpu_bind(_grouped, _args())
        with (
            patch.object(
                cute_mma, "grouped_device_panel_index_domain", return_value=False
            ),
            pytest.raises(BackendUnsupported, match="bounded Int32 cluster prefixes"),
        ):
            bound.to_code(_config())
    assert grouped_device_panel_index_domain(16, 8192, 1536, 256, 224, 1)
    for values in (
        (0, 640, 512, 256, 224, 1),
        (8, 0, 512, 256, 224, 1),
        (1, 2**31 - 1, 1, 256, 224, 1),
        (2**31 - 1, 1, 2, 1, 1, 1),
        (8, 2**20, 2**12, 256, 224, 1),
    ):
        assert not grouped_device_panel_index_domain(*values)
