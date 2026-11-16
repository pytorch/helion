"""The device-IR backend hooks: stage coverage, call order and the CuTe dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion._compiler.backend import DeviceIRStage
from helion._compiler.cute import ir_passes
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.triton.backend import TritonBackend
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sum(x: torch.Tensor) -> torch.Tensor:
    rows, _columns = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    for row in hl.tile(rows):
        out[row] = x[row, :].sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] + 1
    return out


def _record_hooks(kernel: Kernel[torch.Tensor], *args: object) -> list[object]:
    """Bind ``kernel`` on CPU and record every device-IR hook call in order."""
    calls: list[object] = []
    rewrite = CuteBackend.rewrite_device_ir
    register = CuteBackend.register_search_surface

    def record_rewrite(self, stage, func, device_ir, analysis=None):
        calls.append(stage)
        return rewrite(self, stage, func, device_ir, analysis)

    def record_register(self, device_ir, config_spec):
        calls.append("register_search_surface")
        return register(self, device_ir, config_spec)

    with (
        patch.object(CuteBackend, "rewrite_device_ir", record_rewrite),
        patch.object(CuteBackend, "register_search_surface", record_register),
        mock_cuda_unavailable(),
    ):
        cpu_bind(kernel, args)
    return calls


def _expected_order(*, rolls_reductions: bool) -> list[object]:
    order: list[object] = list(DeviceIRStage)
    order.insert(
        order.index(DeviceIRStage.AFTER_MULTI_ROOT_PID_RESTRICTION),
        "register_search_surface",
    )
    if not rolls_reductions:
        # ``register_rollable_reductions`` returns before the roller runs.
        order.remove(DeviceIRStage.REDUCTION_ROLLING_ANALYZED)
        order.remove(DeviceIRStage.REDUCTION_LOOPS_REGISTERED)
    return order


def test_stage_table_covers_every_stage_in_pipeline_order() -> None:
    assert list(ir_passes._STAGE_PASSES) == list(DeviceIRStage)


def test_reduction_kernel_runs_every_hook_once_in_order() -> None:
    calls = _record_hooks(_row_sum, torch.randn(64, 128))
    assert calls == _expected_order(rolls_reductions=True)


def test_grid_kernel_skips_the_rolling_stages() -> None:
    calls = _record_hooks(_add_one, torch.randn(64, 128))
    assert calls == _expected_order(rolls_reductions=False)


def test_decomp_table_overrides_pop_the_numeric_extrema_only_on_cute() -> None:
    table = {
        torch.ops.aten.fmin.default: object(),
        torch.ops.aten.fmax.default: object(),
        torch.ops.aten.stack.default: object(),
    }
    default = dict(table)
    TritonBackend().decomp_table_overrides(default)
    assert default == table
    cute = dict(table)
    CuteBackend().decomp_table_overrides(cute)
    assert set(cute) == {torch.ops.aten.stack.default}
