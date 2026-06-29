from __future__ import annotations

from typing import Any
from typing import cast
from unittest.mock import patch

import sympy

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.indexing_strategy import BlockedSubscriptIndexing
from helion._compiler.tile_strategy import LoopDimInfo


class _FakeBase:
    ndim = 1

    def __init__(self, size: sympy.Expr) -> None:
        self._size = size

    def size(self, dim: int) -> sympy.Expr:
        assert dim == 0
        return self._size


class _FakeStrategy:
    def mask_var(self, block_id: int) -> None:
        assert block_id == 0
        return None


class _FakeLoopState:
    strategy = _FakeStrategy()

    def __init__(self, loop_info: LoopDimInfo) -> None:
        self.block_id_to_info = {0: loop_info}


class _FakeCodegen:
    def __init__(self, loop_info: LoopDimInfo) -> None:
        self.active_device_loops = {0: [_FakeLoopState(loop_info)]}

    def offset_var(self, block_id: int) -> str:
        assert block_id == 0
        return "offs"


class _FakeState:
    fx_node = None

    def __init__(self, loop_info: LoopDimInfo) -> None:
        self.codegen = _FakeCodegen(loop_info)


class _FakeEnv:
    shape_env = object()

    def get_block_id(self, size: object) -> int:
        return 0

    def resolve_codegen_block_id(
        self, block_id: int, codegen: _FakeCodegen, graph: object
    ) -> int:
        assert block_id == 0
        return block_id


def _boundary_check_for(tensor_size: sympy.Expr, loop_end: sympy.Expr) -> str:
    block_size = sympy.Symbol("block_size", integer=True)
    loop_info = LoopDimInfo(end_expr=loop_end)
    indexing = BlockedSubscriptIndexing(
        cast("Any", _FakeBase(tensor_size)),
        offsets=["offs"],
        block_shape=cast("Any", [block_size]),
    )
    with (
        patch.object(CompileEnvironment, "current", return_value=_FakeEnv()),
        patch("helion._compiler.tile_strategy.shape_env_size_hint", return_value=64),
    ):
        return indexing.boundary_check(cast("Any", _FakeState(loop_info)))


def test_block_ptr_boundary_check_kept_for_distinct_dynamic_extents() -> None:
    tensor_size = sympy.Symbol("s0", integer=True)
    loop_end = sympy.Symbol("s1", integer=True)

    assert _boundary_check_for(tensor_size, loop_end) == "[0]"


def test_block_ptr_boundary_check_elided_for_exact_dynamic_extent() -> None:
    tensor_size = sympy.Symbol("s0", integer=True)

    assert _boundary_check_for(tensor_size, tensor_size) == "None"
