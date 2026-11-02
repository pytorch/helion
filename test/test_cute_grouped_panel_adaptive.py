from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_d_shift import _plans
from test.test_cute_grouped_device_panel import _config
from test.test_cute_grouped_device_panel import _emit
from test.test_cute_grouped_mn_224 import _args
from test.test_cute_grouped_mn_224 import _grouped

from helion._compiler.cute import cute_mma
from helion._compiler.cute.grouped_panel_raster import (
    GROUPED_PANEL_ALTERNATE_KEY as ALT,
)
from helion._compiler.cute.grouped_panel_raster import GROUPED_PANEL_RATIO_KEY as RATIO
from helion._compiler.cute.grouped_panel_raster import grouped_panel_ratio_index_domain
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes


def _adaptive(ratio=1, *, primary=8, alternate=4, cluster_n=1):
    config = _config(primary, cluster_n)
    config.config.update({ALT: alternate, RATIO: ratio})
    return config


def _choice_block(source):
    tree = ast.parse(source)
    name = "tcgen05_grouped_panel_choice_m_clusters"
    leader = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            )
            for statement in node.body
        )
    )
    start = next(
        i
        for i, node in enumerate(leader.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )
    stop = next(
        i
        for i in range(start, len(leader.body))
        if isinstance(leader.body[i], ast.Assign)
        and isinstance(leader.body[i].targets[0], ast.Subscript)
    )
    assert any(
        isinstance(node, ast.While)
        and ast.unparse(node.test).endswith(".is_valid_tile")
        and leader in node.body
        for node in ast.walk(tree)
    )
    return ast.Module(body=leader.body[start:stop], type_ignores=[])


def _order(m, n, width):
    if width == 1:
        return [(row, column) for column in range(n) for row in range(m)]
    result = []
    for panel, start in enumerate(range(0, n, width)):
        rows = range(m) if panel % 2 == 0 else reversed(range(m))
        for row in rows:
            result.extend(
                (row, column) for column in range(start, min(start + width, n))
            )
    return result


@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("n_clusters", [1, 7, 17])
@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_actual_adaptive_decoder_matches_each_selected_order_and_peer_bijection(
    cluster_n, n_clusters, ratio
):
    bn = 224 if cluster_n == 1 else 128
    columns = n_clusters * bn * cluster_n
    args = (
        torch.empty((8192, 256), dtype=torch.bfloat16),
        torch.empty((8, columns, 256), dtype=torch.bfloat16),
        torch.arange(9, dtype=torch.int64) * 1024,
    )
    source = _emit(_adaptive(ratio, cluster_n=cluster_n), args)
    block = compile(_choice_block(source), "<actual adaptive panel>", "exec")
    for m_clusters in (1, 2, 7, 8, 14, 16, 32):
        width = 8 if m_clusters <= ratio * n_clusters else 4
        expected = _order(m_clusters, n_clusters, width)
        observed = set()
        for local, pair in enumerate(expected):
            for peer in range(cluster_n):
                values = {
                    "cutlass": SimpleNamespace(Int32=int, min=min),
                    "tcgen05_grouped_selected_sched": SimpleNamespace(
                        _current_work_linear_idx=117 + local,
                        search_state=SimpleNamespace(tile_count_prev_group=117),
                    ),
                    "tcgen05_grouped_selected_group_search_result": SimpleNamespace(
                        problem_shape_m=(m_clusters - 1) * 256 + 1,
                        problem_shape_n=columns,
                    ),
                    "tcgen05_grouped_selected_cta_tile_idx_m": local % m_clusters,
                    "tcgen05_grouped_selected_cta_tile_idx_n": (local // m_clusters)
                    * cluster_n
                    + peer,
                }
                exec(block, values)
                result = (
                    values["tcgen05_grouped_selected_cta_tile_idx_m"],
                    values["tcgen05_grouped_selected_cta_tile_idx_n"],
                )
                assert result == (pair[0], pair[1] * cluster_n + peer)
                assert result not in observed
                observed.add(result)
        assert observed == {
            (m, n) for m in range(m_clusters) for n in range(n_clusters * cluster_n)
        }


def test_absent_and_explicit_none_keep_static_source_and_remove_optional_keys():
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args())
        original = _config()
        config = _config()
        config.config.update({ALT: None, RATIO: None})
        normalized = bound.config_spec.normalized_config(config)
        assert ALT not in normalized.config and RATIO not in normalized.config
        assert bound.to_code(normalized) == bound.to_code(original)


@pytest.mark.parametrize(
    "alternate,ratio",
    [
        (None, 1),
        (4, None),
        (0, 1),
        (3, 1),
        (4, 0),
        (4, -1),
        (True, 1),
        (4, True),
        (4, 1.5),
    ],
)
def test_invalid_or_partial_adaptation_fails_closed(alternate, ratio):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args())
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(_adaptive(ratio, alternate=alternate))


def test_adaptive_metadata_is_live_and_runtime_threshold_is_checked():
    args = _args(dtype=torch.int64)
    empty = (*args[:2], torch.zeros(9, dtype=torch.int64))
    source = _emit(_adaptive(), args)
    assert source == _emit(_adaptive(), empty)
    plan = _plans(source)[0]
    assert plan[ALT] == 4 and plan[RATIO] == 1
    _validate_tcgen05_grouped_device_split_sizes(plan, args[2])
    for bad in ({ALT: 3}, {RATIO: None}, {RATIO: 0}, {"device_panel_swizzle": False}):
        with pytest.raises(BackendUnsupported):
            _validate_tcgen05_grouped_device_split_sizes(plan | bad, args[2])
    assert grouped_panel_ratio_index_domain(1536, 224, 1, 1)
    assert not grouped_panel_ratio_index_domain(257, 128, 1, 2**31 - 1)
    assert not grouped_panel_ratio_index_domain(1536, 224, 1, 0)


def test_threshold_guard_is_repeated_at_codegen():
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args())
        with (
            patch.object(
                cute_mma, "grouped_panel_ratio_index_domain", return_value=False
            ),
            pytest.raises(BackendUnsupported, match="signed Int32 threshold"),
        ):
            bound.to_code(_adaptive())


def test_legacy_width_can_be_selected_without_relaxing_quad_edge_guard():
    source = _emit(_adaptive(alternate=1))
    assert "tcgen05_grouped_panel_choice_m_clusters" in source
    source = _emit(_adaptive(primary=1))
    assert _plans(source)[0]["l2_swizzle_size"] == 1
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args())
        with pytest.raises(BackendUnsupported, match="split full/fringe scheduler"):
            bound.to_code(_adaptive(primary=1, cluster_n=2))
