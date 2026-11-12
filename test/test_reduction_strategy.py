from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._testing import import_example
from helion.exc import ReductionOnNonTile
import helion.language as hl

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from helion._compiler.backend import Backend


class _ThreadLevel64Backend(MetalBackend):
    # A thread-level backend whose reductions use at most 64 threads (one AMD
    # wavefront), like the out-of-tree FlyDSL backend.
    @property
    def name(self) -> str:
        return "_test_thread_level_64"

    def max_reduction_threads(self) -> int | None:
        return 64


@pytest.mark.parametrize(
    ("backend_type", "max_threads"),
    [(MetalBackend, 1024), (_ThreadLevel64Backend, 64)],
)
@pytest.mark.parametrize("tile_size", [4, 8, 16])
@pytest.mark.parametrize("graph_reduction", [False, True])
def test_merged_reduction_preserves_backend_thread_coverage(
    backend_type: type[Backend],
    max_threads: int,
    tile_size: int,
    graph_reduction: bool,
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

    expected_threads = min(tile_size**2, max_threads)
    assert strategy._reduction_thread_count() == expected_threads
    assert strategy._synthetic_cute_lane_var is None


def test_split_k_barrier_binds_on_pallas() -> None:
    # The full slice over the hl.register_tunable-sized split dim needs a
    # reduction dim even though Pallas does not pad factory tensors.
    kernel = helion.kernel(
        import_example("split_k_barrier").split_k_matmul.fn,
        backend="pallas",
        static_shapes=True,
        ref_mode=helion.RefMode.OFF,
    )
    bound = kernel.bind((torch.randn(150, 200), torch.randn(200, 6)))
    assert any(info.reduction for info in bound.env.block_sizes)


@pytest.mark.parametrize("backend", ["pallas", "triton"])
def test_reduction_over_runtime_extent_is_rejected(backend: str) -> None:
    @helion.kernel(backend=backend, static_shapes=True, ref_mode=helion.RefMode.OFF)
    def kernel(x: torch.Tensor, extent: int) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            values = hl.full([tile, extent], 1.0, x.dtype)
            out[tile] = values.sum(-1)
        return out

    with pytest.raises(ReductionOnNonTile):
        kernel.bind((torch.randn(17), 32))


def test_pallas_block_size_slice_keeps_its_extent() -> None:
    # Guards the scope of the tunable-extent fix: only tunable extents take
    # a reduction dim on Pallas; a full slice of a block-size-sized buffer
    # keeps its extent as before.
    @helion.kernel(backend="pallas", static_shapes=True, ref_mode=helion.RefMode.OFF)
    def kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        bm = hl.register_block_size(m)
        scratch = torch.zeros([bm, n], dtype=x.dtype, device=x.device)
        out = torch.empty_like(x)
        for tile in hl.tile(m, block_size=bm):
            out[tile, :] = x[tile, :] + scratch[:, :]
        return out

    bound = kernel.bind((torch.randn(32, 16),))
    assert not any(info.reduction for info in bound.env.block_sizes)
