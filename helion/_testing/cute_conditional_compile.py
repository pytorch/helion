"""Compile the normal conditional-root entry using its actual CPU host arguments."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import torch

from .cute_codegen import cpu_target
from .cute_codegen import load_generated_module

if TYPE_CHECKING:
    from pathlib import Path


@cute.jit
def _launch_entry(
    entry: cutlass.Constexpr,
    tensors: tuple,
    grid: cutlass.Constexpr,
    block: cutlass.Constexpr,
    stream: object,
) -> None:
    entry(*tensors).launch(grid=grid, block=block, stream=stream)


def compile_conditional_entries(
    source: Path, inputs: tuple[object, ...], *, complete: bool = False
) -> list[str]:
    """Run only the CPU host factory; fake arguments retain its view metadata."""
    cuda = importlib.import_module("cuda.bindings.driver")
    module = load_generated_module(source)
    calls: list[
        tuple[object, tuple[int, ...], tuple[torch.Tensor, ...], tuple[int, ...]]
    ] = []

    def record(
        entry: object,
        grid: tuple[int, ...],
        *args: torch.Tensor,
        block: tuple[int, ...],
    ) -> None:
        assert all(
            isinstance(arg, torch.Tensor) and arg.device.type == "cpu" for arg in args
        )
        calls.append((entry, grid, args, block))

    host = (
        module.written_conditional_attention
        if complete
        else module.conditional_online_attention_prefix
    )
    outputs = host(*inputs, _launcher=record)
    assert len(calls) == (2 if complete else 1) and len(outputs) == 2
    dtypes = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int8: cutlass.Int8,
        torch.uint8: cutlass.Uint8,
        torch.int32: cutlass.Int32,
    }
    modules = []
    for entry, grid, args, block in calls:
        tensors = tuple(
            make_fake_tensor(
                dtypes[arg.dtype],
                tuple(arg.shape),
                tuple(arg.stride()),
                assumed_align=16,
            )
            for arg in args
        )
        # The entry is exactly the normal generated kernel; only the host launch
        # trampoline uses fake arguments and suppresses JIT-engine/device loading.
        with cpu_target(forbid_native=False):
            compiled = cute.compile(
                _launch_entry,
                entry,
                tensors,
                (*grid, *([1] * (3 - len(grid)))),
                block,
                cuda.CUstream(0),
                no_jit_engine=True,
                no_cache=True,
            )
        modules.append(str(compiled.ir_module))
    return modules
