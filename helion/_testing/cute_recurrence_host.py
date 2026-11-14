"""Shared fake inputs and host-wrapper plans for recurrence tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import cutlass.cute as cute
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .._compiler.cute.gdn_recurrence_geometry import gdn_smem_bytes
from .._compiler.cute.gdn_recurrence_geometry import gdn_tmem_layout
from ..runtime.cute.launcher import _create_cute_wrapper
from . import DEVICE

if TYPE_CHECKING:
    from collections.abc import Callable

BT, DK, DV = 16, 128, 128


def chunk_fake_inputs(
    *,
    tokens: int = 128,
    heads: int = 12,
    sequences: int = 2,
    chunks: int = 8,
    packed_workspace: bool = True,
) -> tuple[torch.Tensor, ...]:
    rows = chunks * BT
    with FakeTensorMode():
        vector_bytes = heads * rows * DK * 2
        aq_bytes = heads * chunks * BT * BT * 2
        gt_bytes = heads * chunks * DK * 4
        if packed_workspace:
            workspace = torch.empty(
                (3 * vector_bytes + aq_bytes + gt_bytes,),
                device=DEVICE,
                dtype=torch.uint8,
            )
            kd = workspace[:vector_bytes].view(torch.bfloat16).view(1, heads, rows, DK)
            qd = (
                workspace[vector_bytes : 2 * vector_bytes]
                .view(torch.bfloat16)
                .view(1, heads, rows, DK)
            )
            ak = (
                workspace[2 * vector_bytes : 3 * vector_bytes]
                .view(torch.bfloat16)
                .view(1, heads, rows, DK)
            )
            aq = (
                workspace[3 * vector_bytes : 3 * vector_bytes + aq_bytes]
                .view(torch.bfloat16)
                .view(1, heads, chunks, BT * BT)
            )
            g_total = (
                workspace[3 * vector_bytes + aq_bytes :]
                .view(torch.float32)
                .view(1, heads, chunks, DK)
            )
        else:
            kd = torch.empty((1, heads, rows, DK), device=DEVICE, dtype=torch.bfloat16)
            qd = torch.empty_like(kd)
            ak = torch.empty_like(kd)
            aq = torch.empty(
                (1, heads, chunks, BT * BT),
                device=DEVICE,
                dtype=torch.bfloat16,
            )
            g_total = torch.empty(
                (1, heads, chunks, DK), device=DEVICE, dtype=torch.float32
            )
        values = torch.empty(
            (1, tokens, heads, DV), device=DEVICE, dtype=torch.bfloat16
        )
        output = torch.empty_like(values)
        state = torch.empty(
            (sequences, heads, DV, DK), device=DEVICE, dtype=torch.bfloat16
        )
        cu_seqlens = torch.empty((sequences + 1,), device=DEVICE, dtype=torch.int32)
        cu_chunks = torch.empty_like(cu_seqlens)
    return kd, qd, ak, aq, g_total, values, output, state, cu_seqlens, cu_chunks


def chunk_runtime_plan(
    *, tokens: int = 128, heads: int = 12, sequences: int = 2, chunks: int = 8
) -> dict[str, object]:
    plan: dict[str, object] = {
        "kind": "chunk_recurrence_warp_dv4",
        "workspace_layout_version": 2,
        "outputs_scaled": False,
        "factor_key_xor": 8,
        "chunk_size": BT,
        "key_size": DK,
        "value_size": DV,
        "total_tokens": tokens,
        "total_chunks": chunks,
        "heads": heads,
        "sequences": sequences,
    }
    for index, name in enumerate(
        (
            "kd",
            "qd",
            "ak",
            "aq",
            "gt",
            "v",
            "out",
            "state",
            "cu_seqlens",
            "cu_chunks",
        )
    ):
        plan[f"{name}_idx"] = index
    return plan


def chunk_runtime_view_args() -> tuple[torch.Tensor, ...]:
    args = chunk_fake_inputs()
    return (
        args[0].view(-1, DK),
        args[1].view(-1, DK),
        args[2].view(-1, DK),
        args[3].view(-1, BT * BT),
        args[4].view(-1, DK),
        args[5].view(-1, DV),
        args[6].view(-1, DV),
        args[7],
        args[8],
        args[9],
    )


def gdn_wrapper_plan(**overrides: object) -> dict[str, object]:
    layout = gdn_tmem_layout(64, 128)
    assert layout is not None
    plan: dict[str, object] = {
        "kind": "gdn_recurrence_sm100",
        "k_idx": 0,
        "w_idx": 1,
        "u_idx": 2,
        "g_idx": 3,
        "h_idx": 4,
        "batch": 2,
        "heads": 8,
        "seqlen": 1024,
        "chunk": 128,
        "dhead": 64,
        "dstate": 128,
        "num_chunks": 8,
        "block_v": 128,
        "epilogue_warps": 8,
        "stages": 3,
        "token_groups": 2,
        "mma_m": 128,
        "threads": 288,
        "tmem_cols": layout.alloc_cols,
        "smem_bytes": gdn_smem_bytes(128, 64, 128, 3, 128),
        "state_col": layout.state_col,
        "state_image_col": layout.state_image_col,
        "acc_col": layout.acc_col,
        "update_image_col": layout.update_image_col,
        "device_abi": 5,
    }
    plan.update(overrides)
    return plan


def tensor_schema(dtype: torch.dtype, shape: tuple[int, ...]) -> tuple[object, ...]:
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return ("tensor", str(dtype), len(shape), shape, tuple(reversed(strides)))


def gdn_test_wrapper() -> Callable[..., object]:
    def dummy_kernel(*_args: object) -> None:
        pass

    dummy_kernel._helion_cute_wrapper_plans = [gdn_wrapper_plan()]  # type: ignore[attr-defined]
    schema = (
        tensor_schema(torch.bfloat16, (2, 1024, 8, 64)),
        tensor_schema(torch.bfloat16, (2, 1024, 8, 64)),
        tensor_schema(torch.bfloat16, (2, 1024, 8, 128)),
        tensor_schema(torch.float32, (2, 1024, 8)),
        tensor_schema(torch.bfloat16, (2, 8, 8, 64, 128)),
        ("scalar_constexpr", "int", 128, 128),
    )
    return cast(
        "Callable[..., object]", _create_cute_wrapper(dummy_kernel, schema, (288, 1, 1))
    )


def compile_host_offline(function: object, *args: object) -> Any:  # noqa: ANN401
    """Lower the real host without a native execution engine or compile cache."""
    return cute.compile(
        function, *args, options="--enable-tvm-ffi", no_jit_engine=True, no_cache=True
    )
