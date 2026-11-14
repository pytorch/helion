"""Alignment facts for paired native loads into the existing conditional SS panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..memory_ops import tensor_has_specialized_base_alignment

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from .conditional_attention_access import NativeGatherPanel


def can_pack_native_panel(panel: NativeGatherPanel, env: CompileEnvironment) -> bool:
    """Keep scalar staging unless the actual input pointer and both bytes are proved.

    The access certificate already owns the clamped row, common page/head,
    full channel iota, group32 scale and positive-zero row mask. Read the bound
    input's cache-key-backed pointer residue: a normalized fake storage offset
    alone does not prove the physical input pointer's alignment.
    """
    raw, scale = panel.raw, panel.scale
    value = raw.host.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == raw.dtype == torch.int8
        and scale.dtype == torch.uint8
        and panel.page_size == panel.depth == 128
        and panel.group_size == 32
        and panel.shifted_row.meta["val"].dtype in (torch.int32, torch.int64)
        and len(raw.shape) == len(raw.strides) == len(scale.strides) == 4
        and all(type(size) is int and size > 0 for size in raw.shape)
        and raw.shape[1] == raw.shape[3] == 128
        and scale.shape == (*raw.shape[:3], 4)
        and all(type(stride) is int and stride >= 0 for stride in raw.strides)
        and all(type(stride) is int and stride >= 0 for stride in scale.strides)
        and raw.strides[-1] == 1
        and all(type(stride) is int and stride % 2 == 0 for stride in raw.strides[:-1])
        and tensor_has_specialized_base_alignment(env, value, 2)
    )
