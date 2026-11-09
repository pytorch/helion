"""The shared per-thread tile planner's tensor-core contraction candidate.

``helion/_compiler/per_thread_planning.py`` plans the tile loops of every
per-thread backend.  Whether a multi-dimensional device loop that contains a
matmul is a tensor-core contraction candidate decides its thread budget: a
candidate keeps the full 1024 threads, its configured thread counts and every
axis active; a loop with a matmul that is no candidate is capped at 256 threads
and its unobserved axes are demoted.  Metal has no detector of its own and
relies on the core default (``Backend.per_thread_mma_candidate_loop``), which
answers from the matmul's operands.
"""

from __future__ import annotations

from contextlib import contextmanager
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.compile_environment import BlockSizeInfo
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import LoopSpecBlockSizeSource
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.host_function import HostFunction
from helion._compiler.lane_loop_context import LaneLoopContext
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.per_thread_planning import plan_per_thread_loop_strategy
from helion._compiler.tile_strategy import PerThreadNDTileStrategy
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.language import _tracing_ops
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.backend import Backend
    from helion._compiler.device_function import DeviceFunction

# Block ids 0 and 1 are the root grid (M, N); 2 and 3 the device loop (K and
# an axis no statement observes).  The block sizes double as the extents.  The
# grid axes and J carry one explicit thread each so the launch stays within
# 1024 threads (the planner demotes every automatic device-loop axis of a
# wider launch before the budget is even consulted) and K is the one automatic
# axis the thread budget resolves.
_EXTENTS = (16, 32, 512, 8)
_NUM_THREADS = [1, 1, 0, 1]
_BLOCK_ID_BY_SIZE = {extent: block_id for block_id, extent in enumerate(_EXTENTS)}


def _k_loop_ir(matmul_dtype: torch.dtype | None) -> DeviceIR:
    """A root grid over (M, N) with a two-dimensional device loop over (K, J)
    containing ``acc = acc + x[m, k] @ y[k, n]`` of ``matmul_dtype`` (no matmul
    for None)."""
    ir = DeviceIR()
    root, loop = torch.fx.Graph(), torch.fx.Graph()
    root.call_function(_tracing_ops._for_loop, (1, [0, 0], [512, 8], []))
    if matmul_dtype is not None:
        lhs = loop.call_function(_tracing_ops._host_tensor, ("x",))
        lhs.meta["val"] = torch.empty((16, 512), dtype=matmul_dtype)
        rhs = loop.call_function(_tracing_ops._host_tensor, ("y",))
        rhs.meta["val"] = torch.empty((512, 32), dtype=matmul_dtype)
        acc = loop.call_function(_tracing_ops._host_tensor, ("acc",))
        acc.meta["val"] = torch.empty((16, 32), dtype=torch.float32)
        mm = loop.call_function(torch.ops.aten.mm.default, (lhs, rhs))
        mm.meta["val"] = torch.empty((16, 32), dtype=matmul_dtype)
        add = loop.call_function(torch.ops.aten.add.Tensor, (acc, mm))
        add.meta["val"] = torch.empty((16, 32), dtype=torch.float32)
    ir.graphs = [RootGraphInfo(0, root), ForLoopGraphInfo(1, loop, [], [2, 3])]
    ir.root_ids = [0]
    ir.grid_block_ids = [[0, 1]]
    return ir


class _Function:
    def __init__(self, ir: DeviceIR, config: Config) -> None:
        self.config = config
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}
        self.lane_contexts: dict[int, LaneLoopContext] = {}
        self.backend_state = None
        self.codegen = SimpleNamespace(
            codegen_graphs=ir.graphs,
            current_root_graph_info=ir.graphs[ir.root_ids[0]],
        )

    def new_var(self, name: str, *, dce: bool = False) -> str:
        return name

    def register_lane(
        self, block_id: int, *, flat_multi: bool = False
    ) -> LaneLoopContext:
        context = LaneLoopContext(block_id, flat_multi=flat_multi)
        self.lane_contexts[block_id] = context
        return context


class _MetalWithBoundaryAxis(MetalBackend):
    """Metal whose planner policy marks block 3 as a boundary-only axis, so
    the loop over (K, J) starts with J inactive."""

    def boundary_only_grid_block_ids(self, fn: DeviceFunction) -> set[int]:
        return {3}


@contextmanager
def _context(
    ir: DeviceIR, backend: Backend, config: Config
) -> Iterator[tuple[CompileEnvironment, DeviceFunction]]:
    spec = ConfigSpec(backend=backend, device=torch.device("cpu"))
    for block_id, extent in enumerate(_EXTENTS):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=extent))
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=extent))
    env = cast(
        "CompileEnvironment",
        SimpleNamespace(
            backend=backend,
            backend_name=backend.name,
            config_spec=spec,
            known_equal=operator.eq,
            specialize_expr=lambda expr: expr,
            shape_env=SimpleNamespace(replace=lambda expr: expr),
            is_jagged_tile=lambda block_id: False,
            get_block_id=lambda size: None,
            resolve_block_id=lambda size: (
                _BLOCK_ID_BY_SIZE.get(int(size)) if isinstance(size, int) else None
            ),
            canonical_block_id=lambda block_id: block_id,
            block_sizes=[
                BlockSizeInfo(
                    block_id=block_id,
                    size=extent,
                    var=cast("torch.SymInt", block_id),
                    reduction=False,
                    block_size_source=LoopSpecBlockSizeSource(),
                )
                for block_id, extent in enumerate(_EXTENTS)
            ],
        ),
    )
    fn = _Function(ir, config)
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch.object(
            HostFunction, "current", return_value=SimpleNamespace(device_ir=ir)
        ),
    ):
        yield env, cast("DeviceFunction", fn)


def _plan(matmul_dtype: torch.dtype | None) -> PerThreadNDTileStrategy:
    backend = _MetalWithBoundaryAxis()
    config = Config(block_sizes=list(_EXTENTS), num_threads=list(_NUM_THREADS))
    with _context(_k_loop_ir(matmul_dtype), backend, config) as (_env, fn):
        strategy = plan_per_thread_loop_strategy(backend, fn, [2, 3], config)
    assert isinstance(strategy, PerThreadNDTileStrategy)
    return strategy


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_metal_contraction_loop_keeps_budget_threads_and_axes(
    dtype: torch.dtype,
) -> None:
    """The K loop of a tensor-core dtype matmul whose output is the root grid:
    the full 1024-thread budget (K keeps its automatic 512 lanes instead of
    shrinking to the 256 cap), the configured thread counts and no inactive
    axis, as the planner decided before it moved out of the CuTe backend and
    ran CuTe's detector for Metal too."""
    strategy = _plan(dtype)
    assert strategy.inactive_block_ids == set()
    assert strategy.num_threads == [0, 1]
    assert strategy.mma_mode is False


def test_metal_non_contraction_matmul_loop_is_capped() -> None:
    """A matmul the tensor cores cannot take (fp64) keeps the 256-thread cap of
    a matmul-heavy loop without an MMA path (K's 512 lanes shrink to 256) and
    the boundary axis demoted."""
    strategy = _plan(torch.float64)
    assert strategy.inactive_block_ids == {3}
    assert strategy.num_threads == [256, 1]


def test_metal_loop_without_matmul_keeps_the_full_budget() -> None:
    """No matmul: the 1024-thread budget with the boundary axis demoted."""
    strategy = _plan(None)
    assert strategy.inactive_block_ids == {3}
    assert strategy.num_threads == [0, 1]


def test_metal_contraction_reserve_is_the_backend_hook() -> None:
    """The planner takes the contraction reserve from the backend
    (``Backend.matmul_contraction_thread_reserve``): Metal's MPP ops contract
    their own K, so its hook returns 1 and a tile beside a reserving K axis
    keeps the full budget even when the core reserve would claim 64 threads
    (the shared planner once called the core function directly and halved
    Metal's tiles)."""
    from helion._compiler import per_thread_planning

    with patch.object(
        per_thread_planning, "matmul_contraction_thread_reserve", return_value=64
    ):
        strategy = _plan(None)
        backend = _MetalWithBoundaryAxis()
        config = Config(block_sizes=list(_EXTENTS), num_threads=list(_NUM_THREADS))
        with _context(_k_loop_ir(None), backend, config) as (_env, fn):
            assert backend.matmul_contraction_thread_reserve(fn, [2, 3]) == 1
    assert strategy.inactive_block_ids == {3}
    assert strategy.num_threads == [0, 1]


def _plan_root(matmul_dtype: torch.dtype | None) -> PerThreadNDTileStrategy:
    """The root (M, N) tile of ``_k_loop_ir`` with automatic thread counts
    (512 threads at full extent)."""
    backend = _MetalWithBoundaryAxis()
    config = Config(block_sizes=list(_EXTENTS), num_threads=[0, 0, 0, 1])
    with _context(_k_loop_ir(matmul_dtype), backend, config) as (_env, fn):
        strategy = plan_per_thread_loop_strategy(backend, fn, [0, 1], config)
    assert isinstance(strategy, PerThreadNDTileStrategy)
    return strategy


def test_metal_root_tile_beside_a_tensor_core_k_loop_keeps_the_full_budget() -> None:
    """A matmul kernel's root tile keeps the 1024-thread limit when its K
    loop is a tensor-core candidate (``per_thread_matmul_loop_has_mma_path``
    answers through the candidate detector, as the CuTe probe did for
    Metal); only a matmul without such a path caps the tile at 256 threads
    (the 32-wide axis shrinks to 16), which would otherwise serialize a tile
    a column sum reduces over into a lane loop."""
    assert _plan_root(torch.float32).num_threads == [0, 0]
    assert _plan_root(torch.float64).num_threads == [0, 16]
