from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import sympy
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.flydsl.backend import FlyDSLBackend
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.reduction_strategy import PersistentReductionStrategy

if TYPE_CHECKING:
    from helion._compiler.backend import Backend


@pytest.mark.parametrize("backend_type", [MetalBackend, FlyDSLBackend])
@pytest.mark.parametrize("tile_size", [4, 8, 16])
@pytest.mark.parametrize("graph_reduction", [False, True])
def test_merged_reduction_preserves_backend_thread_coverage(
    backend_type: type[Backend], tile_size: int, graph_reduction: bool
) -> None:
    # Exercise the real backend hooks without requiring Metal or ROCm devices.
    # Small merged reductions must not acquire CuTe lanes; larger reductions
    # must retain their backend's thread count instead of being capped at 32.
    backend = backend_type()
    first, second = sympy.symbols("first second", integer=True, positive=True)
    block_ids = {first: 1, second: 2}
    env = SimpleNamespace(
        backend=backend,
        block_sizes=[SimpleNamespace(numel=first * second)],
        specialize_expr=lambda expr: expr,
        shape_env=ShapeEnv(),
        get_block_id=block_ids.get,
    )
    graph = Mock(spec=ReductionLoopGraphInfo, block_ids=[0])
    fn = Mock(
        spec=DeviceFunction,
        block_size_var_cache={},
        tile_strategy=None,
        codegen=SimpleNamespace(codegen_graphs=[graph] if graph_reduction else []),
        resolved_block_size=Mock(return_value=tile_size),
        new_var=Mock(side_effect=lambda name, **kwargs: name),
    )
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch(
            "helion._compiler.reduction_strategy.find_block_size_symbols",
            return_value=(block_ids, set()),
        ),
    ):
        strategy = PersistentReductionStrategy(fn, 0)

    expected_threads = min(tile_size**2, 64 if backend.name == "flydsl" else 1024)
    assert strategy._reduction_thread_count() == expected_threads
    assert strategy._synthetic_cute_lane_var is None
