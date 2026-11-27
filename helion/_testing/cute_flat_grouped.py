"""The flat grouped kernel bound for a CUDA target on the CPU, and what the
flat grouped, grouped descriptor and TMA RN test modules read from it."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion._compiler.cute.tcgen05_flat_grouped_config import K_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import WARPS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_rna_schedule
from helion._testing import import_example
from helion._testing.cute_codegen import cuda_fake
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.cute.tcgen05_flat_grouped_plan import FlatGroupedRnaSchedule
    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel


def bound_host_function(bound: BoundKernel) -> HostFunction:
    host = bound.host_function
    assert host is not None
    return host


def flat_contract(
    routing: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    addend: torch.Tensor,
    mode: hl.constexpr,
    scalar_group: hl.constexpr,
) -> torch.Tensor:
    rows, reduction = left.shape
    groups, reduction, columns = right.shape
    if mode == "partial_groups":
        groups = groups - 1
    left = left.view(-1)
    result = torch.empty((rows, columns), dtype=right.dtype, device=right.device).view(
        -1
    )
    for group in hl.tile(groups, block_size=1 if scalar_group else None):
        begin = routing[group]
        end = routing[group.index + 1]
        length = end - begin
        for row in hl.jagged_tile(length):
            raw_indices = begin[:, None] + row.index[None, :]
            if mode == "narrow_rows":
                indices = raw_indices.to(torch.int32)
            else:
                indices = raw_indices
            for column in hl.tile(0, columns):
                carry = hl.zeros(
                    [group, row, column], dtype=torch.float32, device=right.device
                )
                for inner in hl.tile(0, reduction):
                    if mode == "reset_carry":
                        carry = hl.zeros(
                            [group, row, column],
                            dtype=torch.float32,
                            device=right.device,
                        )
                    if mode == "wrong_stride":
                        x = hl.load(
                            left,
                            [
                                indices[:, :, None] * (reduction + 4)
                                + inner.index[None, None, :]
                            ],
                        )
                    else:
                        x = hl.load(
                            left,
                            [
                                indices[:, :, None] * reduction
                                + inner.index[None, None, :]
                            ],
                        )
                    y = right[group, inner, column]
                    carry = carry + torch.matmul(x, y)
                if mode == "shift_bias":
                    z = addend[group, column.index + 1]
                else:
                    z = addend[group, column]
                carry = carry + z.unsqueeze(1)
                if mode == "extra_math":
                    carry = carry * 2.0
                hl.store(
                    result,
                    [indices[:, :, None] * columns + column.index[None, None, :]],
                    carry,
                )
    return result.reshape(rows, columns)


@contextmanager
def cuda_binding(
    *,
    static: bool = False,
    bits: int = 64,
    mode: str = "ordinary",
    groups: int = 3,
    rows: int = 273,
    k: int = 64,
    n: int = 132,
    dtype: torch.dtype = torch.float32,
    scalar_group: bool = False,
    original: bool = False,
    misalign: int | None = None,
    offsets_stride: int = 1,
    dot_precision: str = "tf32",
) -> Iterator[BoundKernel]:
    inputs = [
        torch.empty_strided(
            (groups + 1,),
            (offsets_stride,),
            dtype=torch.int32 if bits == 32 else torch.int64,
        ),
        torch.empty((rows, k), dtype=dtype, requires_grad=True),
        torch.empty((groups, k, n), dtype=dtype, requires_grad=True),
        torch.empty((groups, n), dtype=dtype, requires_grad=True),
    ]
    if misalign is not None:
        original_input = inputs[misalign]
        assert isinstance(original_input, torch.Tensor)
        shifted = torch.empty(original_input.numel() + 1, dtype=original_input.dtype)[
            1:
        ].view(original_input.shape)
        shifted.requires_grad_(original_input.requires_grad)
        inputs[misalign] = shifted
    flags = {
        f"HELION_CUTE_{name}": "1"
        for name in (
            "REGION_FISSION",
            "FULL_SLICE_MATMUL_TILING",
            "SEGMENTED_MATMUL_TILING",
            "FLATTEN_NESTED_REDUCTIONS",
            "MATERIALIZE_TRANSFORMED_OPERANDS",
        )
    }
    with (
        _target(),
        patch.dict(os.environ, flags),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        mock_cuda_unavailable(),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch(
            "helion.runtime.kernel._find_device", return_value=torch.device("cuda:0")
        ),
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=cuda_fake,
        ),
        patch(
            "torch._inductor.runtime.hints.DeviceProperties.create",
            return_value=SimpleNamespace(
                type="cuda",
                index=0,
                cc=100,
                warp_size=32,
                multi_processor_count=148,
            ),
        ),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        function = flat_contract
        arguments = (*inputs, mode, scalar_group)
        if original:
            jagged_dense_bmm = import_example("jagged_dense_bmm").jagged_dense_bmm

            function = jagged_dense_bmm.fn
            arguments = tuple(inputs)
        kernel = helion.kernel(
            function,
            backend="cute",
            static_shapes=static,
            autotune_effort="none",
            dot_precision=dot_precision,
        )
        bound = kernel.bind(arguments)
        assert all(tensor.requires_grad for tensor in inputs[1:4])
        assert bound.env.device.type == "cuda"
        assert all(tensor.device.type == "cuda" for tensor in bound.fake_args[:4])
        with bound.env, bound_host_function(bound):
            yield bound


def flat_rna_schedule(
    bound: BoundKernel, warps: int = 4, bk: int = 32
) -> FlatGroupedRnaSchedule | None:
    proof = prove_flat_grouped_rna(bound.env, bound_host_function(bound).device_ir)
    assert proof is not None
    return flat_grouped_rna_schedule(
        bound.env,
        proof,
        converter_warps=warps,
        block_k=bk,
        ab_stages={32: 6, 64: 3}[bk],
        shared_capacity=232448,
    )


def flat_converter_config(
    bound: BoundKernel, warps: int = 4, block_k: int = 32
) -> helion.Config:
    return helion.Config.from_dict(
        bound.config_spec._base_default_config().config
        | {WARPS_KEY: warps, K_KEY: block_k}
    )


def compiled_source(bound: BoundKernel, config: helion.Config) -> str:
    # This is the public compiler entry. There is no emitter/launcher patch.
    with bound.env.suspend():
        return bound.to_code(config)


def native_kernel_source(source: str) -> str:
    candidates = [
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and "def _helion_native_grouped_rna(" in node.args[0].value
    ]
    assert len(candidates) == 1
    return candidates[0]
