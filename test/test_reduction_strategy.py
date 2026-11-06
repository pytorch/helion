from __future__ import annotations

import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.compile_environment import BlockSizeInfo
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import LoopSpecBlockSizeSource
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._testing import import_example
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import ReductionOnNonTile
import helion.language as hl

pytestmark = pytest.mark.cpu_only

# Settings for kernels these tests only bind, on CPU under any HELION_BACKEND,
# to see which block sizes become reductions. Pallas refuses
# debug_dtype_asserts (it emits Triton checks), which the dtype-asserts CI job
# turns on through HELION_DEBUG_DTYPE_ASSERTS, so it is pinned off.
_BIND_ONLY: dict[str, Any] = {
    "static_shapes": True,
    "ref_mode": helion.RefMode.OFF,
    "debug_dtype_asserts": False,
}

if TYPE_CHECKING:
    from helion._compiler.backend import Backend


class _ThreadLevel64Backend(MetalBackend):
    # A thread-level backend whose reductions use at most 64 threads (one AMD
    # wavefront); stands in for FlyDSL's thread cap only, so this test does not
    # import FlyDSL (test_flydsl_launcher.py pins the cap to the real backend).
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
    shape_env = ShapeEnv()
    first_var = shape_env.create_unbacked_symint()
    second_var = shape_env.create_unbacked_symint()
    first, second = first_var.node.expr, second_var.node.expr
    block_ids = {first: 1, second: 2}
    config = helion.Config(block_sizes=[tile_size, tile_size])
    config_spec = ConfigSpec(
        backend=backend, device=torch.device("cpu"), target_device_capability=None
    )
    # The trace hint differs from every selected size, so only actual config
    # resolution can produce the expected thread coverage below.
    config_spec.block_sizes.extend(
        BlockSizeSpec(block_id=block_id, size_hint=64) for block_id in (1, 2)
    )
    tile_infos = [
        BlockSizeInfo(
            block_id=block_id,
            size=64,
            var=var,
            reduction=False,
            block_size_source=LoopSpecBlockSizeSource(),
        )
        for block_id, var in ((1, first_var), (2, second_var))
    ]
    env = SimpleNamespace(
        backend=backend,
        block_sizes=[SimpleNamespace(numel=first * second), *tile_infos],
        config_spec=config_spec,
        specialize_expr=lambda expr: expr,
        shape_env=shape_env,
        get_block_id=block_ids.get,
        known_equal=operator.eq,
    )
    graph = Mock(spec=ReductionLoopGraphInfo, block_ids=[0])
    fn = Mock(
        spec=DeviceFunction,
        config=config,
        block_size_var_cache={},
        tile_strategy=None,
        codegen=SimpleNamespace(codegen_graphs=[graph] if graph_reduction else []),
        new_var=Mock(side_effect=lambda name, **kwargs: name),
    )
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch(
            "helion._compiler.device_function.find_block_size_symbols",
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
        **_BIND_ONLY,
    )
    bound = kernel.bind((torch.randn(150, 200), torch.randn(200, 6)))
    assert any(info.reduction for info in bound.env.block_sizes)


@pytest.mark.parametrize("backend", ["pallas", "triton"])
def test_reduction_over_runtime_extent_is_rejected(backend: str) -> None:
    @helion.kernel(backend=backend, **_BIND_ONLY)
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
    @helion.kernel(backend="pallas", **_BIND_ONLY)
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
