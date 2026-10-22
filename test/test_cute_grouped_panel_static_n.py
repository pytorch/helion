from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

from test.test_cute_grouped_device_panel import _config
from test.test_cute_grouped_device_panel import _emit
from test.test_cute_grouped_device_panel import _panel_block


@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize(
    "n_clusters,swizzle",
    [(1, 8), (2, 8), (3, 2), (7, 4), (7, 8), (9, 8), (16, 8), (17, 8)],
)
def test_static_panel_paths_preserve_the_full_cluster_coordinate_set(
    cluster_n, n_clusters, swizzle
):
    bn = 128 if cluster_n == 2 else 224
    columns = n_clusters * bn * cluster_n
    args = (
        torch.empty((4096, 256), dtype=torch.bfloat16),
        torch.empty((8, columns, 256), dtype=torch.bfloat16),
        torch.arange(9, dtype=torch.int64) * 512,
    )
    source = _emit(_config(swizzle, cluster_n), args)
    block = _panel_block(source)
    code = compile(block, "<static panel paths>", "exec")
    text = ast.unparse(block)
    if n_clusters <= swizzle:
        assert "tcgen05_grouped_panel_m_clusters" not in text
        assert "tcgen05_grouped_panel_span" not in text
    elif n_clusters <= 2 * swizzle:
        assert "tcgen05_grouped_panel_span" not in text
        assert "tcgen05_grouped_panel_tail_index" in text
    else:
        assert "tcgen05_grouped_panel_span" in text
    for m_clusters in (1, 2, 7, 16):
        observed = set()
        for local in range(m_clusters * n_clusters):
            for peer in range(cluster_n):
                values = {
                    "cutlass": SimpleNamespace(Int32=int, min=min),
                    "tcgen05_grouped_selected_sched": SimpleNamespace(
                        _current_work_linear_idx=291 + local,
                        search_state=SimpleNamespace(tile_count_prev_group=291),
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
                exec(code, values)
                result = (
                    values["tcgen05_grouped_selected_cta_tile_idx_m"],
                    values["tcgen05_grouped_selected_cta_tile_idx_n"],
                )
                assert result not in observed
                assert result[1] % cluster_n == peer
                observed.add(result)
        assert observed == {
            (m, n) for m in range(m_clusters) for n in range(n_clusters * cluster_n)
        }
