from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import sympy

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import LoopedReductionStrategy


@pytest.mark.parametrize(
    "extent,chunk,cluster,expected_cluster",
    [
        (7168, 1024, 8, 8),
        (8196, 2048, 2, 2),
        (2**31 - 2049, 1024, 2, 2),
        (2**31 - 1, 1024, 2, 1),
        (1024, 2**28, 8, 1),
    ],
)
def test_static_padded_cluster_bounds_fit_int32(
    extent: int, chunk: int, cluster: int, expected_cluster: int
) -> None:
    numel = sympy.Integer(extent)
    env = SimpleNamespace(
        backend=SimpleNamespace(name="cute"),
        shape_env=SimpleNamespace(size_hint=lambda expr: int(expr)),
        block_sizes=[SimpleNamespace(numel=numel)],
        config_spec=SimpleNamespace(cute_indexed_reduction_block_ids=set()),
        specialize_expr=lambda expr: expr,
        specialized_multiple=lambda expr, multiple: expr % multiple == 0,
    )
    fn = SimpleNamespace(
        config=helion.Config(cute_cluster_n=cluster),
        cute_state=SimpleNamespace(simt_cluster_n=1),
    )
    strategy: Any = object.__new__(LoopedReductionStrategy)
    strategy._fn = lambda: fn
    strategy.block_ids = [0]
    strategy._thread_count = 128
    strategy._loop_block_size = chunk
    strategy._mask_var = "mask_0"
    strategy._cute_reduction_vec_width = 1
    strategy._cute_rolled_cluster_n = 1
    strategy._cute_rolled_cluster_slice_len = 0
    strategy._cute_rolled_cluster_checked = False
    state = SimpleNamespace(
        codegen=SimpleNamespace(
            current_grid_state=SimpleNamespace(thread_axis_sizes={})
        )
    )
    host = SimpleNamespace(device_ir=SimpleNamespace(has_atomic_ops=lambda: False))
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch.object(HostFunction, "current", return_value=host),
    ):
        strategy._maybe_apply_cute_rolled_cluster(state)
    assert strategy._cute_rolled_cluster_n == expected_cluster
    if expected_cluster > 1:
        assert strategy._cute_rolled_cluster_slice_len * cluster <= 2**31 - 1
    else:
        assert strategy._cute_rolled_cluster_slice_len == 0
    assert strategy._mask_var == "mask_0"
