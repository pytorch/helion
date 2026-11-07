"""Alias-only MMA plans specialize guarded tensor layouts in the launch schema."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_launcher import mock_cute_launcher_runtime
import helion.language as hl
from helion.runtime.cute import launcher
from helion.runtime.cute.launch_descriptor import bake_tensor_shapes

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only


def _state_recurrence(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batches, steps, _, depth = k.shape
    value_depth = v.size(3)
    out = torch.empty(
        (batches, depth, value_depth), dtype=torch.float32, device=k.device
    )
    for batch, rows, columns in hl.tile(
        (batches, depth, value_depth), block_size=(1, None, None)
    ):
        acc = hl.zeros((rows, columns), dtype=torch.float32)
        for step in hl.grid(steps):
            left = k[batch.id, step, :, rows]
            right = v[batch.id, step, :, columns]
            acc = hl.dot(left.transpose(-2, -1), right, acc=acc)
        out[batch.id, rows, columns] = acc
    return out


def test_alias_guard_preserves_partial_mma_runtime_layout_requirement():
    for rows, expected in ((250, False), (256, True)):
        kernel = SimpleNamespace(
            _helion_cute_wrapper_plans=[
                {
                    "kind": "tcgen05_ab_tma",
                    "m_size": rows,
                    "n_size": 128,
                    "k_total_size": 128,
                    "bm": 128,
                    "bn": 128,
                    "bk": 64,
                },
                {"kind": "helion_mma_dag_alias"},
            ]
        )
        assert bake_tensor_shapes(kernel) is expected


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def pointer(dtype, address, space, *, assumed_align):
        return address, assumed_align

    mock_cute_launcher_runtime(monkeypatch, pointer)


def test_actual_alias_only_recurrence_specializes_guarded_layouts(
    fake_runtime, tmp_path: Path
) -> None:
    initialized = torch.cuda.is_initialized()
    inputs = tuple(torch.empty((1, 2, 32, 32), dtype=torch.bfloat16) for _ in range(2))
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_state_recurrence, backend="cute", static_shapes=True),
            inputs,
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[32, 32],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_loads="lane",
            )
        )
        path = tmp_path / "alias_recurrence.py"
        path.write_text(source)
        module = load_generated_module(path)
    kernels = [
        value
        for value in vars(module).values()
        if hasattr(value, "_helion_cute_wrapper_plans")
    ]
    assert len(kernels) == 1
    kernel = kernels[0]
    plans = kernel._helion_cute_wrapper_plans
    assert [plan["kind"] for plan in plans] == ["helion_mma_dag_alias"]
    assert plans[0]["stored"] == [False, False, True]
    assert not getattr(kernel, "_helion_cute_disable_bake_tensor_shapes", False)
    assert bake_tensor_shapes(kernel)

    args = (*inputs, torch.empty((1, 32, 32), dtype=torch.float32))
    grid = (1, 1, 1)
    original = launcher._build_cute_schema_and_args(kernel, args, grid)
    assert [entry[3:] for entry in original.schema] == [
        (tuple(tensor.shape), tensor.stride()) for tensor in args
    ]
    assert original.launch_args[len(args) :] == grid
    guard = launcher._cute_last_launch_arg_guard(kernel, args, grid)
    assert guard.matches(kernel, args, grid)
    for index in (0, 2):
        changed = list(args)
        changed[index] = args[index].transpose(-2, -1)
        assert changed[index].shape == args[index].shape
        assert changed[index].data_ptr() == args[index].data_ptr()
        assert launcher._build_cute_schema_and_args(kernel, changed, grid).schema != (
            original.schema
        )
        assert not guard.matches(kernel, changed, grid)
    assert torch.cuda.is_initialized() == initialized
