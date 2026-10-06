"""Fused tcgen05 body for gated (softmax-free) jagged attention.

CPU tests drive codegen with the tcgen05 support probe patched on; the device
kernel is additionally compiled through the CuTe DSL (no GPU needed). GPU tests
check numerics at jagged, non-multiple-of-128 sequence lengths, for the example
kernel and for variants that exercise the matcher's corner cases.
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import os
import re
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash_gated
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


def _hstu_module() -> Any:
    return import_path(EXAMPLES_DIR / "jagged_hstu_attn.py")


def _hstu_args(
    *,
    batch: int = 8,
    seq: int = 512,
    heads: int = 8,
    head_dim: int = 64,
    total_rows: int = 3044,
    device: str = "cpu",
) -> tuple[Any, ...]:
    q = torch.empty((total_rows, heads, head_dim), dtype=torch.bfloat16, device=device)
    seq_offsets = torch.zeros((batch + 1,), dtype=torch.int64, device=device)
    return (
        seq,
        1.0 / head_dim**2,
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        seq_offsets,
    )


def _hstu2_module() -> Any:
    return import_path(EXAMPLES_DIR / "jagged_hstu_attn_2.py")


def _hstu2_args(
    *,
    heads: int = 4,
    head_dim: int = 64,
    total_rows: int = 3913,
    sequences: int = 64,
    max_seq_len: int = 128,
    device: str = "cpu",
) -> tuple[Any, ...]:
    """Arguments of ``jagged_hstu_attention`` (fp32, all heads per tile)."""
    q = torch.empty((total_rows, heads, head_dim), dtype=torch.float32, device=device)
    seq_offsets = torch.zeros((sequences + 1,), dtype=torch.int32, device=device)
    return (
        max_seq_len,
        1.0 / head_dim**2,
        1.0 / max_seq_len,
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        seq_offsets,
    )


@contextlib.contextmanager
def _cpu_codegen_patches() -> Any:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
    ):
        yield


def _bind_hstu(args: tuple[Any, ...]) -> Any:
    kernel = _hstu_module()._helion_jagged_attention_kernel
    kernel.reset()
    return kernel.bind(args)


def _gated_markers(code: str) -> bool:
    return (
        "'kind': 'helion_flash_gated'" in code and "flash_gated_shared_storage" in code
    )


def _match(bound: Any) -> cute_flash_gated.GatedAttentionMatch | None:
    """Run the config-independent matcher on a bound kernel's device IR."""
    env, host_function = bound.env, bound.host_function
    assert env is not None and host_function is not None
    with env, host_function:
        return cute_flash_gated.match_gated_attention(host_function.device_ir)


# ---------------------------------------------------------------------------
# Variant kernels: the example's dataflow with one mutation each.
# ---------------------------------------------------------------------------


@helion.kernel()
def _tile_end_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Sequence end read through ``tile_b.end`` (which is ``tile_b.begin + 1``)."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.end]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _no_if_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """No root ``if``: every query tile runs, masked rows are skipped at the store."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        mask_q = tile_q.index < seq_len
        q_blk = q[tile_q.index + starts, tile_h.begin, :]
        acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
        for tile_kv in hl.tile(0, tile_q.end, block_size=None):
            mask_kv = tile_kv.index < seq_len
            k_blk = k[tile_kv.index + starts, tile_h.begin, :]
            v_blk = v[tile_kv.index + starts, tile_h.begin, :]
            scores = (
                torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha) * scale
            )
            scores = torch.where(
                (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                & mask_q[:, None]
                & mask_kv[None, :],
                scores,
                0.0,
            )
            acc += torch.matmul(scores.to(v.dtype), v_blk)
        out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _row_only_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Row mask that is not the jagged length: the frontend stores zeros there."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    # Poisoned output: a skipped store is visible as NaN.
    out = torch.full_like(v, float("nan"))
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len - 1
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, seq_len, block_size=None):
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(mask_q[:, None], scores, 0.0)
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _band_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Band mask through ``%`` (torch floor-mod semantics below the diagonal)."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                band = (
                    (tile_kv.index.unsqueeze(0) - tile_q.index.unsqueeze(1)) % 3
                ) == 1
                scores = torch.where(
                    band & mask_q[:, None] & mask_kv[None, :], scores, 0.0
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _pair_table_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Start/end pairs ``off[2b]``, ``off[2b + 1]``: not the jagged-length form,
    so the gap rows after each sequence store zeros like the frontend."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) // 2)
    dimV = hl.specialize(v.size(2))
    # Poisoned output: a skipped store is visible as NaN.
    out = torch.full_like(v, float("nan"))
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin * 2]
        ends = seq_offsets[tile_b.begin * 2 + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _atomic_loop_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    stats: torch.Tensor,
) -> torch.Tensor:
    """Rejected: a side effect (atomic) inside the KV loop, nothing else."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                hl.atomic_add(stats, [tile_b.begin], 1.0)
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _atomic_root_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    stats: torch.Tensor,
) -> torch.Tensor:
    """Rejected: a side effect (atomic) in the root ``if`` body, nothing else."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            hl.atomic_add(stats, [tile_b.begin], 1.0)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _erf_gate_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Rejected: a gate op outside the supported elementwise table."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = torch.erf(torch.matmul(q_blk, k_blk.T) * alpha) * scale
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _row_bias_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Rejected: the row mask reads a per-row tensor, not a scalar program."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = bias[tile_q.index + starts] > 0
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _split_offsets_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
) -> torch.Tensor:
    """Rejected: K/V rows use a different offset than Q/O."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        kv_starts = kv_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + kv_starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + kv_starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _head_mismatch_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Rejected: K is read at a fixed head while Q/V/O use the head tile."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, 0, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _store_target_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Rejected: the output rows are not offset like the input rows."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _row_scaled_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Accepted: the column compare is against a row-derived scalar that is
    not ``row +/- c`` (``2 * row > col``), so it cannot bound the KV range."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    ((tile_q.index * 2).unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


_ACCEPTED_VARIANTS: dict[str, Any] = {
    "tile_end": _tile_end_kernel,
    "row_scaled": _row_scaled_kernel,
    "no_if": _no_if_kernel,
    "row_only": _row_only_kernel,
    "band": _band_kernel,
    "pair_table": _pair_table_kernel,
}


def _rejected_variant(name: str) -> tuple[Any, tuple[Any, ...]]:
    args = _hstu_args()
    total_rows = args[2].shape[0]
    if name == "atomic_loop":
        return _atomic_loop_kernel, (*args, torch.zeros((8,), dtype=torch.float32))
    if name == "atomic_root":
        return _atomic_root_kernel, (*args, torch.zeros((8,), dtype=torch.float32))
    if name == "erf_gate":
        return _erf_gate_kernel, args
    if name == "row_bias":
        return _row_bias_kernel, (*args, torch.ones((total_rows,)))
    if name == "split_offsets":
        return _split_offsets_kernel, (*args, torch.zeros_like(args[5]))
    if name == "head_mismatch":
        return _head_mismatch_kernel, args
    if name == "store_target":
        return _store_target_kernel, args
    assert name == "head_dim_96"
    return _hstu_module()._helion_jagged_attention_kernel, _hstu_args(head_dim=96)


# ---------------------------------------------------------------------------
# CPU codegen tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_hstu_selects_gated_body_at_fused_tiles(head_dim: int) -> None:
    with _cpu_codegen_patches():
        bound = _bind_hstu(_hstu_args(head_dim=head_dim))
        spec = bound.config_spec
        assert spec.cute_flash_gated_search_enabled
        assert not spec.cute_flash_search_enabled
        # The fused body owns the root: no tcgen05 matmul fields hide the knob.
        assert not spec.cute_tcgen05_search_enabled
        assert list(spec._flat_fields()) == [
            "block_sizes",
            "cute_flash_kv_stage",
            "cute_flash_gate_warpgroups",
        ]
        seeds = [seed.config for seed in spec.autotune_seed_configs()]
        assert seeds[0]["block_sizes"] == [128, 128]
        assert seeds[0]["cute_flash_gate_warpgroups"] == 2
        # Every fused tile shape (both query tile heights, every KV tile width)
        # at every legal ring depth and every gate warpgroup count above one is
        # seeded (a count must split a thread's columns into chunks of at least
        # 16 elements; one warpgroup stays a search choice but never won).
        assert {tuple(seed["block_sizes"]) for seed in seeds} == {
            (q_tile, kv_tile)
            for q_tile in cute_flash_gated.GATED_Q_TILE_CHOICES
            for kv_tile in cute_flash_gated.GATED_KV_TILE_CHOICES
        }
        # ... at every ring depth that fits shared memory for that tile.
        assert {
            (seed["block_sizes"][1], seed["cute_flash_kv_stage"]) for seed in seeds
        } == {
            (kv_tile, stage)
            for kv_tile in cute_flash_gated.GATED_KV_TILE_CHOICES
            for stage in cute_flash_gated.gated_kv_stage_choices(
                head_dim, kv_tile, torch.bfloat16, smem_capacity=232448
            )
        }
        assert {
            (*seed["block_sizes"], seed["cute_flash_gate_warpgroups"]) for seed in seeds
        } == {
            (128, 128, 2),
            (128, 128, 4),
            (128, 64, 2),
            (128, 64, 4),
            (64, 128, 2),
            (64, 128, 4),
            (64, 64, 2),
        }
        assert all(seed["cute_flash_gate_warpgroups"] > 1 for seed in seeds)
        # One seed per (tile shape, legal ring depth, warpgroup count above one):
        # 21 at head_dim 32 and 64, 17 at 128 (the 128-wide KV tile fits two
        # ring depths there).
        expected = sum(
            len(
                cute_flash_gated.gated_kv_stage_choices(
                    head_dim,
                    kv_tile,
                    torch.bfloat16,
                    smem_capacity=232448,
                    q_tile=q_tile,
                )
            )
            * len(
                [
                    g
                    for g in cute_flash_gated.gated_gate_warpgroup_choices(
                        q_tile, kv_tile
                    )
                    if g > 1
                ]
            )
            for q_tile in cute_flash_gated.GATED_Q_TILE_CHOICES
            for kv_tile in cute_flash_gated.GATED_KV_TILE_CHOICES
        )
        assert len(seeds) == expected == (17 if head_dim == 128 else 21)
        code = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
    assert _gated_markers(code)
    assert (
        f"flash_gated_shared_storage({head_dim}, 128, 2, cutlass.BFloat16, 128)" in code
    )
    # Jagged rows: TMA coordinate offsets and a runtime KV trip count.
    assert "cute.domain_offset((flash_row_base, 0, 0), _flash_mQt)" in code
    # One work item per CTA in the root form; the Q tile is indexed per item.
    assert "flash_n_items = cutlass.Int32(1)" in code
    assert "for flash_item in cutlass.range(flash_n_items, unroll=1):" in code
    assert "cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile], " in code
    # The strict causal term bounds the KV range at the tile's last row.
    assert (
        "flash_kv_end = cutlass.min(cutlass.min(flash_q_row_end, flash_g3 - "
        "cutlass.Int32(0)), flash_q_row_end - cutlass.Int32(1) + cutlass.Int32(0) - "
        "cutlass.Int32(0))"
    ) in code
    assert "cute.domain_offset((0, flash_row_base, 0), _flash_mVt)" in code
    assert "for flash_active_kv in cutlass.range(flash_num_kv_active, unroll=1)" in code
    # The gate program replays the frontend chain with its bf16 rounding points
    # on 32-wide vectors: the score, the alpha product, the quotient and the
    # scale product are rounded (paired conversions); the selects between
    # bf16-exact values and the zero constant are not re-rounded.
    assert "cute.math.exp2(" in code
    assert (
        "flash_g10 = _helion_flash_rt.gated_round_vec(flash_s_v, cutlass.BFloat16)"
        in code
    )
    assert code.count("_helion_flash_rt.gated_round_vec(") == 4
    assert ".to(cutlass.BFloat16).to(cutlass.Float32)" not in code
    # Column-index compares are per-chunk bitmasks: ``row > col`` is the set
    # of columns below ``row - chunk_col0``; the row-only term gates the mask.
    assert re.search(
        r"flash_g6 = cutlass\.Int32\(\(cutlass\.Int64\(1\) << cutlass\.Int64\("
        r"cutlass\.min\(cutlass\.max\(flash_gate_row - cutlass\.Int32\(0\) - "
        r"flash_chunk_base, cutlass\.Int32\(0\)\), cutlass\.Int32\(32\)\)\)\) - "
        r"cutlass\.Int64\(1\)\)",
        code,
    )
    assert "flash_g7 = cutlass.select_(flash_g5, flash_g6, cutlass.Int32(0))" in code
    assert "flash_g9 = flash_g7 & flash_g8" in code
    # ``exp`` feeding ``1 + e`` uses the flush-to-zero ex2 (identical result);
    # the division stays IEEE.
    assert re.search(
        r"cute\.math\.exp2\(flash_g\d+ \* 1\.4426950408889634, fastmath=True\)", code
    )
    assert re.search(r"flash_g\d+ = flash_g\d+ / flash_g\d+$", code, re.MULTILINE)
    # The P handoff ring: warp 0 waits for P, issues the PV MMAs and releases
    # the stage (the gate's second acquire of a stage waits on that release).
    assert "flash_p_full = flash_p_ready_cons.wait_and_advance()" in code
    assert code.index(
        "flash_p_full = flash_p_ready_cons.wait_and_advance()"
    ) < code.index("flash_p_full.release()")
    # P: one select on the AND of the in-tile and gate masks (the nested
    # ``where`` / ``_mask_to`` selects with the same zero fill are merged), and
    # zero past the tile whatever the gate computes there.  The mask is
    # materialized once, ahead of the scores, as an Int32 vector of all-ones /
    # zero lanes; both selects are bit selects on the fp32 patterns.  The
    # masked elements' scores are replaced by 1.0 so zero-filled rows never
    # take the IEEE division's slow path; the mask statements are hoisted
    # above the first score use.
    assert (
        "flash_p_keep_v = cute.where(cute.full((32,), flash_elem_in_tile_m & flash_g9, "
        "cutlass.Int32) & flash_bit_iota != cutlass.Int32(0), cute.full((32,), "
        "cutlass.Int32(-1), cutlass.Int32), cute.full((32,), cutlass.Int32(0), "
        "cutlass.Int32))"
    ) in code
    assert (
        "flash_s_v = (flash_s_raw_v.bitcast(cutlass.Int32) & flash_p_keep_v | "
        "(flash_p_keep_v ^ cute.full((32,), cutlass.Int32(-1), cutlass.Int32)) & "
        "cute.full((32,), cutlass.Int32(1065353216), cutlass.Int32))"
        ".bitcast(cutlass.Float32)"
    ) in code
    assert code.index("flash_g9 = flash_g7 & flash_g8") < code.index("flash_s_v = ")
    assert code.index("flash_s_v = ") < code.index("flash_g10 = ")
    # The last bf16 rounding (``flash_g20``) is performed by the P store's own
    # conversion: P is the pre-rounding product masked to zero.
    assert (
        "flash_p_v = ((flash_g18 * cute.full((32,), flash_g19, cutlass.Float32))"
        ".bitcast(cutlass.Int32) & flash_p_keep_v).bitcast(cutlass.Float32)"
    ) in code
    assert "tSTrS_e.store(flash_p_v.to(cutlass.BFloat16))" in code
    # Prologue: deferred mbarrier syncs published by one fence; warp 1 owns the
    # TMEM allocation (overlapping warp 0's mbarrier setup and first loads) and
    # hands the permit back at allocation.
    assert code.count("defer_sync=True).make_participants()") == 6
    assert code.count("cute.arch.mbarrier_init_fence()") == 1
    assert "barrier_for_retrieve=flash_tmem_bar, allocator_warp_id=1)" in code
    assert re.search(
        r"if warp_idx == 1:\n\s+flash_tmem\.relinquish_alloc_permit\(\)", code
    )
    assert code.index("flash_tmem.relinquish_alloc_permit()") < code.index(
        "flash_q_empty = flash_q_prod.acquire_and_advance()"
    )
    # The load-independent setup (smem, TMEM allocation, mbarriers, their
    # fence) is emitted before the scalar loads (jagged offsets), behind a
    # CTA-scope acquire-release fence that keeps ptxas from hoisting the loads
    # (and the stall at their first use) above it; the row base, item 0's
    # coordinates and TMA loads follow the loads, and only then the TMEM
    # allocation barrier and the TMEM partitions.
    first_load = code.index(".load()")
    assert code.index("smem = cutlass_utils_flash.SmemAllocator()") < first_load
    assert code.index("cute.arch.mbarrier_init_fence()") < code.index(
        "cute.arch.fence_acq_rel_cta()"
    )
    assert code.count("cute.arch.fence_acq_rel_cta()") == 1
    assert code.index("cute.arch.fence_acq_rel_cta()") < first_load
    assert re.search(
        r"flash_g0_raw = \(seq_offsets\.iterator \+ .*\)\.load\(\)\n", code
    )
    assert first_load < code.index("flash_g0 = cutlass.Int32(flash_g0_raw)")
    assert code.index("flash_row_base = ") < code.index("flash_tmem.wait_for_alloc()")
    assert code.index(
        "cute.domain_offset((flash_row_base, 0, 0), _flash_mQt)"
    ) < code.index("if flash_body_active:")
    assert code.count("if flash_body_active:") == 2
    assert code.index("if flash_body_active:") < code.index(
        "flash_q_empty = flash_q_prod.acquire_and_advance()"
    )
    assert code.index(
        "flash_q_empty = flash_q_prod.acquire_and_advance()"
    ) < code.index("flash_tmem.wait_for_alloc()")
    assert code.index("flash_tmem.wait_for_alloc()") < code.index(
        "tSTcS = flash_thr_st0.partition_S(tScS_P)"
    )
    assert code.index("tSTcS = flash_thr_st0.partition_S(tScS_P)") < code.index(
        "if warp_idx == 0:\n            flash_nk = "
    )
    assert "cute.arch.sync_warp()" not in code
    # An inactive CTA still frees the TMEM it allocated: its gate warps arrive
    # on the teardown barrier (``ast.unparse`` folds the ``else: if`` into an
    # ``elif``).
    assert re.search(
        r"elif warp_idx >= 4:\n\s+_helion_flash_rt\.tcgen05_fence_before_thread_sync\(\)\n"
        r"\s+cute\.arch\.barrier_arrive\(barrier_id=2",
        code,
    )
    assert re.search(
        r"if warp_idx == 0:\n\s+flash_nk = cute\.size\(tSrQ, mode=\[2\]\)", code
    )
    # Query tiles are walked from the last (heaviest under a causal bound) to
    # the first.
    assert (
        "flash_m_tile = (flash_q_extent + cutlass.Int32(127)) // cutlass.Int32(128) "
        "- cutlass.Int32(1) - flash_pid // "
    ) in code
    # Teardown: no CTA-wide barrier.  The gate warps arrive on a named barrier
    # right after their last TMEM read (the last item's O load in the epilogue
    # warpgroup; the end of the item loop for the others or with no item) and
    # go on to store O while warp 1 syncs on it and frees TMEM.
    assert code.count("cute.arch.barrier()") == 0
    arrive = "cute.arch.barrier_arrive(barrier_id=2, number_of_threads=288)"
    fence = "_helion_flash_rt.tcgen05_fence_before_thread_sync()"
    assert code.count(arrive) == 3
    assert re.search(
        r"flash_o_full\.release\(\)\n\s+if flash_item == flash_n_items - "
        r"cutlass\.Int32\(1\):\n\s+"
        + re.escape(fence)
        + r"\n\s+"
        + re.escape(arrive)
        + r"\n\s+flash_rego\.store",
        code,
    )
    assert re.search(
        r"if \(flash_gate_wg != cutlass\.Int32\(0\)\) \| \(flash_n_items == "
        r"cutlass\.Int32\(0\)\):\n\s+"
        + re.escape(fence)
        + r"\n\s+"
        + re.escape(arrive)
        + "\n",
        code,
    )
    assert re.search(
        r"if warp_idx == 1:\n\s+cute\.arch\.barrier\(barrier_id=2, "
        r"number_of_threads=288\)\n\s+_helion_flash_rt\.tcgen05_fence_after_thread_sync\(\)"
        r"\n\s+flash_tmem\.free\(flash_tmem_ptr\)",
        code,
    )
    assert code.count("flash_tmem.free(") == 1
    # tcgen05 fences: the allocating warp publishes the TMEM address before the
    # allocation barrier and every thread orders after it; the gate warps fence
    # before each teardown arrive and warp 1 after the teardown barrier.
    assert code.count("_helion_flash_rt.tcgen05_fence_before_thread_sync()") == 4
    assert code.count("_helion_flash_rt.tcgen05_fence_after_thread_sync()") == 2
    assert re.search(
        r"flash_tmem\.relinquish_alloc_permit\(\)\n\s+"
        r"_helion_flash_rt\.tcgen05_fence_before_thread_sync\(\)",
        code,
    )
    assert re.search(
        r"flash_tmem\.wait_for_alloc\(\)\n\s+_helion_flash_rt\.tcgen05_fence_after_thread_sync\(\)\n"
        r"\s+flash_tmem_ptr = flash_tmem\.retrieve_ptr",
        code,
    )
    # Packed bf16 P has two TMEM buffers of its own after O (S0 @ 0, S1 @ 128,
    # O @ 256): a chunk's P words never alias S columns another warpgroup reads.
    assert f"flash_p_base = flash_tmem_ptr + {256 + head_dim}" in code
    assert "tStS_P0 = cute.make_tensor(flash_p_base, flash_P_layout)" in code
    assert "tStS_P1 = cute.make_tensor(flash_p_base + 64, flash_P_layout)" in code
    # The MMA reads P through the base fragment's iterator, moved by the CTA's
    # TMEM allocation base column plus the buffer offset (``make_fragment_A``
    # carries neither: the second gated CTA on an SM has a nonzero base).
    assert "tP = cute.make_tensor(flash_tmem_ptr, _flash_ptl.outer)" in code
    assert (
        "flash_tmem_col0 = cutlass.Int32(flash_tmem_ptr.toint()) & cutlass.Int32(65535)"
        in code
    )
    assert (
        "tOrP0 = cute.make_tensor(tOrP_base.iterator + cutlass.Float32.width // "
        f"cutlass.BFloat16.width * (flash_tmem_col0 + cutlass.Int32({256 + head_dim})), "
        "tOrP_base.layout)"
    ) in code
    assert (
        "tOrP1 = cute.make_tensor(tOrP_base.iterator + cutlass.Float32.width // "
        f"cutlass.BFloat16.width * (flash_tmem_col0 + cutlass.Int32({256 + head_dim + 64})), "
        "tOrP_base.layout)"
    ) in code
    assert re.search(
        r"cute\.arch\.fence_view_async_tmem_load\(\)\n\s+else:\n\s+flash_reg\.fill",
        code,
    )
    # 64-bit element offsets for the scalar loads and the output tile.
    assert re.search(
        r"cutlass\.Int64\(flash_axis0\) \* cutlass\.Int64\(\w+\.layout\.stride\[0\]\)",
        code,
    )
    assert (
        "cutlass.Int64(flash_o_row) * cutlass.Int64(_flash_mOt.layout.stride[0])"
        in code
    )
    # The output row pointer carries a 16-byte divisibility assumption so the
    # per-thread row store vectorizes; the row is stored in one copy.
    assert "flash_o_row_ptr = _flash_mOt.iterator + cute.assume(" in code
    assert "divby=8)" in code
    assert "cute.autovec_copy(flash_rego_row, gO_row)" in code
    # No online-softmax machinery.
    assert "flash_row_max" not in code
    assert "rescale_o_tmem" not in code
    # Default: two gate warpgroups (plus the producer warpgroup) split the
    # four 32-column chunks of a KV tile; only warpgroup 0 drains O.
    assert "block=(384, 1, 1)" in code
    assert "flash_gate_wg = warp_idx // cutlass.Int32(4) - cutlass.Int32(1)" in code
    assert "for flash_cq in cutlass.range(2, unroll=1)" in code
    assert "flash_ci = flash_cq * cutlass.Int32(2) + flash_gate_wg" in code
    assert "if flash_gate_wg == 0:" in code
    assert code.count("cutlass_pipeline_flash.Agent.Thread, 256)") == 2
    assert "NamedBarrier(barrier_id=1, num_threads=384)" in code
    # The store skips rows masked by the jagged-length term only.
    assert "flash_store_row_ok = flash_row_in_tile & " in code


@pytest.mark.parametrize("head_dim", [64, 128])
def test_hstu2_sequence_form_selects_gated_body(head_dim: int) -> None:
    """``jagged_hstu_attn_2``: a grid over sequences, data-dependent query and
    KV tile loops sharing the sequence bounds, fp32 rows batched over the heads.
    The fused body runs one CTA per (sequence, head) with a per-CTA loop over
    the sequence's query tiles and the tf32 MMA."""
    mod = _hstu2_module()
    kernel = mod.jagged_hstu_attention
    args = _hstu2_args(head_dim=head_dim)
    with _cpu_codegen_patches():
        kernel.reset()
        bound = kernel.bind(args)
        match = _match(bound)
        assert match is not None
        assert match.form == "sequence"
        assert match.io_dtype is torch.float32
        assert match.head_dim == head_dim
        assert match.lane_extent == 4
        assert match.store_skip_terms == ()
        assert all(access.lane_all for access in (match.q, match.k, match.v, match.o))
        spec = bound.config_spec
        assert spec.cute_flash_gated_search_enabled
        seeds = [seed.config for seed in spec.autotune_seed_configs()]
        # fp32 rows: a 128-wide KV tile fits two ring stages at head_dim 64 and
        # none at head_dim 128, where only the 64-wide tile is a choice.
        tiles = {tuple(seed["block_sizes"]) for seed in seeds}
        if head_dim == 64:
            assert tiles == {(128, 128), (128, 64), (64, 128), (64, 64)}
            kv_tile = 128
        else:
            assert tiles == {(128, 64), (64, 64)}
            kv_tile = 64
        for seed in seeds:
            assert (
                cute_flash_gated.gated_smem_bytes(
                    head_dim,
                    seed["block_sizes"][1],
                    seed["cute_flash_kv_stage"],
                    torch.float32,
                    seed["block_sizes"][0],
                )
                <= 232448
            )
        # The estimate is per tile shape: fp32 rows at head_dim 128 fit a
        # three-deep 64-wide ring under a 64-row Q tile but not under 128 rows.
        stages = {
            (tuple(seed["block_sizes"]), seed["cute_flash_kv_stage"]) for seed in seeds
        }
        if head_dim == 128:
            assert ((64, 64), 3) in stages
            assert ((128, 64), 3) not in stages
        config = helion.Config(
            block_sizes=[128, kv_tile],
            cute_flash_kv_stage=2,
            cute_flash_gate_warpgroups=2,
        )
        code = bound.to_code(config)
    assert _gated_markers(code)
    # One CTA per (sequence, head): the grid is multiplied by the head count and
    # the head is the fastest coordinate.
    assert "[0] * 4,)" in code
    assert "flash_lane = flash_pid % cutlass.Int32(4)" in code
    assert "flash_axis0 = flash_pid // cutlass.Int32(4)" in code
    # Rows are absolute: the CTA's row base is the sequence start and the
    # query tiles of the sequence are the work items (heaviest first).
    assert "flash_row_base = flash_g0" in code
    assert (
        "flash_q_extent = cutlass.max(flash_g2 - flash_row_base, cutlass.Int32(0))"
        in code
    )
    assert (
        "flash_n_items = (flash_q_extent + cutlass.Int32(127)) // cutlass.Int32(128)"
    ) in code
    assert "flash_m_tile = flash_n_items - cutlass.Int32(1) - flash_item" in code
    # ``tile_q.index >= tile_kv.index`` bounds the KV range at the tile's end.
    assert re.search(
        r"flash_kv_end = cutlass\.min\(flash_q_extent, flash_q_row_end - cutlass\.Int32\(1\)"
        r" \+ flash_row_base \+ cutlass\.Int32\(1\) - flash_row_base\)",
        code,
    )
    # fp32 io: tf32 MMA operands, no narrow rounding, P stored as 32-bit words.
    assert (
        f"flash_gated_shared_storage({head_dim}, {kv_tile}, 2, cutlass.TFloat32, 128)"
        in code
    )
    assert "'mma_dtype': 'cutlass.TFloat32'" in code
    assert "gated_round_vec" not in code
    # 32-bit P overwrites the thread's own S columns in place: P0 = S0, P1 = S1.
    assert "flash_p_base = flash_tmem_ptr + 0" in code
    assert (
        f"tStS_P1 = cute.make_tensor(flash_p_base + {kv_tile}, flash_P_layout)" in code
    )
    assert "tSTrS.store(flash_p_v)" in code
    assert "St32x32bOp(cute_tcgen05_flash.Repetition(32))" in code
    assert re.search(r"flash_g\d+ = flash_g\d+ / flash_g\d+$", code, re.MULTILINE)
    assert "block=(384, 1, 1)" in code


def test_hstu2_wide_tile_rejected_at_head_dim_128() -> None:
    """fp32 rows at head_dim 128: a 128-wide KV tile has no legal ring depth."""
    mod = _hstu2_module()
    kernel = mod.jagged_hstu_attention
    with _cpu_codegen_patches():
        kernel.reset()
        bound = kernel.bind(_hstu2_args(head_dim=128))
        spec = bound.config_spec
        with pytest.raises(InvalidConfig, match="cute_flash_kv_stage"):
            spec.normalize(helion.Config(block_sizes=[128, 64], cute_flash_kv_stage=3))
        default = spec.default_config().config
        assert default["block_sizes"] == [128, 64]
        assert default["cute_flash_kv_stage"] == 2


def test_hstu_other_tiles_keep_scalar_path_and_validate_knob() -> None:
    with _cpu_codegen_patches():
        bound = _bind_hstu(_hstu_args())
        code = bound.to_code(helion.Config(block_sizes=[16, 16]))
        assert not _gated_markers(code)
        assert "_flash_qk_mma" not in code
        with pytest.raises(InvalidConfig, match="cute_flash_kv_stage"):
            bound.config_spec.normalize(
                helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=99)
            )
        # Normalization in autotune (fix) mode keeps the pinned tiles and a
        # legal depth.
        default = bound.config_spec.default_config().config
        assert default["block_sizes"] in ([128, 128], [128, 64])
        assert default["cute_flash_kv_stage"] in (2, 3, 4)
        # The narrow KV tile is a real choice: it halves the TMEM footprint.
        narrow = bound.to_code(
            helion.Config(block_sizes=[128, 64], cute_flash_kv_stage=2)
        )
        assert _gated_markers(narrow)
        assert "flash_gated_shared_storage(64, 64, 2, cutlass.BFloat16, 128)" in narrow
        assert "flash_tmem.allocate(256)" in narrow
        # Two chunks over two gate warpgroups: one chunk each per KV tile.
        assert "for flash_cq in cutlass.range(1, unroll=1)" in narrow
        # Four warpgroups split the narrow tile into 16-element chunks.
        narrow4 = bound.to_code(
            helion.Config(
                block_sizes=[128, 64],
                cute_flash_kv_stage=2,
                cute_flash_gate_warpgroups=4,
            )
        )
        assert "Ld32x32bOp(cute_tcgen05_flash.Repetition(16))" in narrow4
        assert "St32x32bOp(cute_tcgen05_flash.Repetition(8))" in narrow4
        assert "flash_chunk_layout = cute.make_layout((16,), stride=(1,))" in narrow4
        assert "for flash_cq in cutlass.range(1, unroll=1)" in narrow4
        # At the 64-row tile a thread owns half a row (32 columns of the narrow
        # tile): at most two warpgroups of 16-element chunks.
        with pytest.raises(InvalidConfig, match="cute_flash_gate_warpgroups"):
            bound.config_spec.normalize(
                helion.Config(
                    block_sizes=[64, 64],
                    cute_flash_kv_stage=2,
                    cute_flash_gate_warpgroups=4,
                )
            )
        for warpgroups in (1, 4):
            wide = bound.to_code(
                helion.Config(
                    block_sizes=[128, 128],
                    cute_flash_kv_stage=2,
                    cute_flash_gate_warpgroups=warpgroups,
                )
            )
            assert _gated_markers(wide)
            assert f"block=({128 * (1 + warpgroups)}, 1, 1)" in wide
            # The teardown barrier counts the TMEM warp and every gate warp
            # (two arrive sites in the gate role, one for an inactive CTA).
            assert (
                wide.count(
                    "cute.arch.barrier_arrive(barrier_id=2, "
                    f"number_of_threads={32 + 128 * warpgroups})"
                )
                == 3
            )
            assert (
                "cute.arch.barrier(barrier_id=2, "
                f"number_of_threads={32 + 128 * warpgroups})"
            ) in wide
            assert f"for flash_cq in cutlass.range({4 // warpgroups}, unroll=1)" in wide
            if warpgroups == 1:
                assert "flash_ci = flash_cq\n" in wide
                assert "if flash_gate_wg == 0:" not in wide
            else:
                assert (
                    f"flash_ci = flash_cq * cutlass.Int32({warpgroups}) + flash_gate_wg"
                    in wide
                )
                assert (
                    f"cutlass_pipeline_flash.Agent.Thread, {128 * warpgroups})" in wide
                )


@pytest.mark.parametrize(
    ("kv_tile", "warpgroups", "chunk_cols", "chunk_iters"),
    [(128, 1, 32, 2), (128, 2, 32, 1), (128, 4, 16, 1), (64, 1, 32, 1), (64, 2, 16, 1)],
)
def test_hstu_64_row_tile_codegen(
    kv_tile: int, warpgroups: int, chunk_cols: int, chunk_iters: int
) -> None:
    """The 64-row query tile: M=64 tcgen05 shapes, 16x64b TMEM copies, the
    lane-pair P and O exchanges and stride-2 column bitmasks."""
    with _cpu_codegen_patches():
        bound = _bind_hstu(_hstu_args(head_dim=64))
        code = bound.to_code(
            helion.Config(
                block_sizes=[64, kv_tile],
                cute_flash_kv_stage=2,
                cute_flash_gate_warpgroups=warpgroups,
            )
        )
    assert _gated_markers(code)
    geometry = cute_flash_gated.gated_chunk_geometry(64, kv_tile, warpgroups)
    assert geometry == cute_flash_gated.GatedChunkGeometry(chunk_cols, 2, chunk_iters)
    # The wrapper plan carries the tile height (the host builds 64-row tiled
    # MMAs and TMA boxes from it); the kernel uses a 64-row Q tile throughout.
    assert "'q_tile': 64" in code
    assert f"flash_gated_shared_storage(64, {kv_tile}, 2, cutlass.BFloat16, 64)" in code
    assert f"flash_qk_acc_shape = flash_qkt.partition_shape_C((64, {kv_tile}))" in code
    assert f"cS = cute.make_identity_tensor((64, {kv_tile}))" in code
    assert "flash_q_row0 = flash_m_tile * cutlass.Int32(64)" in code
    assert (
        "flash_m_tile = (flash_q_extent + cutlass.Int32(63)) // cutlass.Int32(64) "
        "- cutlass.Int32(1) - flash_pid // "
    ) in code
    # 16x64b copies: ``chunk_cols`` elements two columns apart per thread, the
    # packed P words (half as many) stored with the same shape.
    assert (
        f"cute_tcgen05_flash.Ld16x64bOp(cute_tcgen05_flash.Repetition({chunk_cols}))"
        in code
    )
    assert (
        f"cute_tcgen05_flash.St16x64bOp(cute_tcgen05_flash.Repetition({chunk_cols // 2}))"
        in code
    )
    assert (
        f"flash_chunk_layout = cute.make_layout(({chunk_cols},), stride=(1,))" in code
    )
    assert "flash_iota_frag[flash_j] = cutlass.Int32(2 * flash_j)" in code
    assert f"for flash_cq in cutlass.range({chunk_iters}, unroll=1)" in code
    assert (
        f"flash_chunk_col0 = flash_kv_col0 + flash_ci * cutlass.Int32({2 * chunk_cols})"
        in code
    )
    # Thread -> (row, column parity) of the 16x64b shapes; the chunk base and
    # the output row follow the mapping.
    assert (
        "flash_col_parity = flash_local_tidx // cutlass.Int32(2) % cutlass.Int32(2)"
        in code
    )
    assert (
        "flash_gate_row = flash_q_row0 + cutlass.Int32(16) * (flash_local_tidx // "
        "cutlass.Int32(32)) + flash_local_tidx % cutlass.Int32(32) // cutlass.Int32(4) "
        "+ cutlass.Int32(8) * (flash_local_tidx % cutlass.Int32(2))"
    ) in code
    assert "flash_chunk_base = flash_chunk_col0 + flash_col_parity" in code
    assert "flash_o_row = flash_row_base + flash_gate_row" in code
    # Stride-2 column bitmasks: ``2j < n`` is ``j < ceil(n / 2)``.
    assert (
        "flash_g6 = cutlass.Int32((cutlass.Int64(1) << cutlass.Int64(cutlass.min("
        "cutlass.max(flash_gate_row - cutlass.Int32(0) - flash_chunk_base + "
        "cutlass.Int32(1) >> cutlass.Int32(1), cutlass.Int32(0)), "
        f"cutlass.Int32({chunk_cols})))) - cutlass.Int64(1))"
    ) in code
    # Lane-pair exchanges: packed P words per stored chunk, and the thread's
    # contiguous half row of O (32 of 64 columns) stored at its parity's offset.
    assert (
        "cute.make_tensor(tSTrS.iterator, flash_word_layout).store("
        "_helion_flash_rt.gated_m64_p_words(flash_p_v, cutlass.BFloat16, "
        "flash_col_parity).bitcast(cutlass.Float32))"
    ) in code
    assert (
        "flash_rego_row.store(_helion_flash_rt.gated_m64_o_half_row("
        "flash_reg_row.load(), cutlass.BFloat16, flash_col_parity))"
    ) in code
    assert "cute_tcgen05_flash.Ld16x64bOp(cute_tcgen05_flash.Repetition(32))" in code
    assert "+ cutlass.Int64(flash_col_parity) * cutlass.Int64(32), divby=8)" in code
    assert (
        "gO_row = cute.make_tensor(flash_o_row_ptr, cute.make_layout((32,), stride=(1,)))"
        in code
    )
    assert "cute.autovec_copy(flash_rego_row, gO_row)" in code
    assert (
        "flash_tmem.allocate(512)" in code
        if kv_tile == 128
        else "flash_tmem.allocate(256)" in code
    )


def test_fast_math_setting_selects_approximate_gate_math() -> None:
    """The existing fast_math setting (never a config knob) switches the gate's
    division and exp2 to the approximate forms; the default stays IEEE."""
    mod = _hstu_module()
    exact_kernel = mod._helion_jagged_attention_kernel
    fast_kernel = helion.kernel(fast_math=True)(exact_kernel.fn)
    config = helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
    args = _hstu_args()
    with _cpu_codegen_patches():
        exact_kernel.reset()
        exact = exact_kernel.bind(args).to_code(config)
        fast = fast_kernel.bind(args).to_code(config)
    assert _gated_markers(exact) and _gated_markers(fast)
    assert "approx=True" not in exact
    # The only fast-math exp2 of the exact path is the ``1 + exp(v)`` site,
    # where the flush-to-zero form is provably identical.
    assert exact.count("fastmath=True") == 1
    assert re.search(r"flash_g\d+ = flash_g\d+ / flash_g\d+$", exact, re.MULTILINE)
    assert "cute.math.rcp(" in fast and "approx=True, ftz=True" in fast
    assert "fastmath=True" in fast
    assert not re.search(r"flash_g\d+ = flash_g\d+ / flash_g\d+$", fast, re.MULTILINE)


def test_gated_match_facts_and_online_softmax_rejection() -> None:
    from test.test_cute_backend import cute_causal_biased_attention

    with _cpu_codegen_patches():
        bound = _bind_hstu(_hstu_args())
        match = _match(bound)
        assert match is not None
        assert match.head_dim == 64
        assert match.io_dtype is torch.bfloat16
        assert match.score_dtype is torch.bfloat16
        assert len(match.leading_block_ids) == 2
        assert match.q_block_id == match.root_block_ids[-1]
        assert match.if_node is not None
        assert match.row_offset is not None
        # ``tile_q.index < seq_offsets[b + 1] - seq_offsets[b]`` skips the store.
        assert len(match.store_skip_terms) == 1
        assert (match.q.row_dim, match.q.lane_dim) == (0, 1)
        assert match.q.lane_block_id == match.root_block_ids[1]

        values = [torch.empty((1, 2, 256, 64), dtype=torch.float16) for _ in range(3)]
        values.append(torch.empty((1, 2, 256, 256), dtype=torch.float16))
        cute_causal_biased_attention.reset()
        softmax_bound = cute_causal_biased_attention.bind(tuple(values))
        assert _match(softmax_bound) is None
        assert softmax_bound.config_spec.cute_flash_search_enabled
        assert not softmax_bound.config_spec.cute_flash_gated_search_enabled
        code = softmax_bound.to_code(
            helion.Config(block_sizes=[1, 128, 128], cute_flash_topology="ws_overlap")
        )
    assert "'kind': 'helion_flash'" in code
    assert not _gated_markers(code)


@pytest.mark.parametrize("name", sorted(_ACCEPTED_VARIANTS))
def test_variant_kernels_match_and_lower(name: str) -> None:
    kernel = _ACCEPTED_VARIANTS[name]
    with _cpu_codegen_patches():
        kernel.reset()
        bound = kernel.bind(_hstu_args())
        assert bound.config_spec.cute_flash_gated_search_enabled
        match = _match(bound)
        assert match is not None
        code = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
    assert _gated_markers(code)
    structural_predicate = (
        "flash_store_row_ok = flash_row_in_tile & "
        "(flash_o_row < cutlass.Int32(_flash_mOt.layout.shape[0]))"
    )
    if name == "row_scaled":
        # ``2 * row > col`` is a per-row bound that is not ``row +/- c``: the
        # mask still uses it (a per-thread column bitmask), the KV range does
        # not (the term's scalar only exists in the gate threads).
        assert re.search(
            r"flash_kv_end = cutlass\.min\(flash_q_row_end, flash_g\d+ - cutlass\.Int32\(0\)\)\n",
            code,
        )
        assert re.search(
            r"flash_g\d+ = cutlass\.Int32\(\(cutlass\.Int64\(1\) << cutlass\.Int64\(cutlass\.min\("
            r"cutlass\.max\(flash_g\d+ - cutlass\.Int32\(0\) - flash_chunk_base, ",
            code,
        )
    elif name == "tile_end":
        # ``seq_offsets[tile_b.begin]`` and ``seq_offsets[tile_b.end]`` read
        # consecutive entries: the leading axis runs at block size 1.
        assert re.search(
            r"cutlass\.Int64\(flash_axis0\) \* cutlass\.Int64\(\w+\.layout\.stride\[0\]\)",
            code,
        )
        assert re.search(
            r"cutlass\.Int64\(flash_axis0 \+ cutlass\.Int32\(1\)\) \* "
            r"cutlass\.Int64\(\w+\.layout\.stride\[0\]\)",
            code,
        )
        assert len(match.store_skip_terms) == 1
    elif name == "no_if":
        assert match.if_node is None
        assert "flash_body_active = cutlass.Boolean(True)" in code
        assert len(match.store_skip_terms) == 1
    elif name == "row_only":
        # ``tile_q.index < seq_len - 1`` is not the jagged length: the gate
        # zeroes P and the zero row is stored like the frontend does.
        assert match.store_skip_terms == ()
        assert structural_predicate in code
    elif name == "pair_table":
        # ``off[2b + 1] - off[2b]`` reads consecutive entries, but ``2b`` is not
        # a leading-axis coordinate: the gap rows after a sequence are stored
        # as zeros rather than skipped.
        assert match.store_skip_terms == ()
        assert structural_predicate in code
        assert re.search(r"= flash_axis0 \* cutlass\.Int32\(2\)$", code, re.MULTILINE)
    else:
        assert name == "band"
        assert len(match.store_skip_terms) == 1
        # ``%`` follows torch (sign of the divisor), not the DSL's truncation.
        assert re.search(
            r"cute\.where\(\((flash_g\d+) < cute\.full\(\(32,\), cutlass\.Int32\(0\), "
            r"cutlass\.Int32\)\) & cute\.full\(\(32,\), cutlass\.Int32\(3\) > "
            r"cutlass\.Int32\(0\), cutlass\.Boolean\) \| \(\1 > cute\.full\(\(32,\), "
            r"cutlass\.Int32\(0\), cutlass\.Int32\)\) & cute\.full\(\(32,\), "
            r"cutlass\.Int32\(3\) < cutlass\.Int32\(0\), cutlass\.Boolean\), "
            r"\1 \+ cute\.full\(\(32,\), cutlass\.Int32\(3\), cutlass\.Int32\), \1\)",
            code,
        )


@pytest.mark.parametrize(
    "name",
    [
        "atomic_loop",
        "atomic_root",
        "erf_gate",
        "row_bias",
        "split_offsets",
        "head_mismatch",
        "store_target",
        "head_dim_96",
    ],
)
def test_mutations_are_rejected(name: str) -> None:
    kernel, args = _rejected_variant(name)
    with _cpu_codegen_patches():
        kernel.reset()
        bound = kernel.bind(args)
        spec = bound.config_spec
        assert not spec.cute_flash_gated_search_enabled
        assert _match(bound) is None
        # Without a flash surface the ring-depth knob has no consumer: a
        # pinned config keeps working on the scalar path, minus the key.
        normalized = spec.normalized_config(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
        assert "cute_flash_kv_stage" not in normalized.config
        assert normalized.config["block_sizes"] == [128, 128]


@pytest.mark.parametrize("reason", ["seq_192", "flash_gate_off"])
def test_pinned_flash_config_runs_on_scalar_path(reason: str) -> None:
    """A forward-flash config pinned at one shape must still compile where the
    fused path is off (another sequence length, ``HELION_CUTE_FLASH=0``, a GPU
    without tcgen05): the ring-depth knob is dropped, not rejected."""
    from test.test_cute_backend import cute_causal_biased_attention

    seq = 192 if reason == "seq_192" else 256
    env_patch = {"HELION_CUTE_FLASH": "0"} if reason == "flash_gate_off" else {}
    values = [torch.empty((1, 2, seq, 64), dtype=torch.float16) for _ in range(3)]
    values.append(torch.empty((1, 2, seq, seq), dtype=torch.float16))
    config = helion.Config(
        block_sizes=[1, 128, 128], cute_flash_kv_stage=2, cute_flash_s_stage=2
    )
    with _cpu_codegen_patches(), patch.dict(os.environ, env_patch):
        cute_causal_biased_attention.reset()
        bound = cute_causal_biased_attention.bind(tuple(values))
        spec = bound.config_spec
        assert not spec.cute_flash_search_enabled
        assert not spec.cute_flash_gated_search_enabled
        normalized = spec.normalized_config(config)
        assert "cute_flash_kv_stage" not in normalized.config
        # Other flash keys keep their pre-existing pass-through behavior.
        assert normalized.config["cute_flash_s_stage"] == 2
        code = bound.to_code(config)
    assert "'kind': 'helion_flash'" not in code
    assert not _gated_markers(code)


@pytest.mark.parametrize(
    ("head_dim", "kv_tile", "kv_stage", "q_tile"),
    [
        (32, 64, 2, 128),
        (64, 128, 3, 128),
        (128, 128, 3, 128),
        (128, 128, 4, 128),
        (64, 128, 4, 64),
        (128, 128, 3, 64),
    ],
)
def test_gated_smem_estimate_matches_struct(
    head_dim: int, kv_tile: int, kv_stage: int, q_tile: int
) -> None:
    import cutlass

    from helion._compiler.cute._flash_runtime import flash_gated_shared_storage

    actual = flash_gated_shared_storage(
        head_dim, kv_tile, kv_stage, cutlass.BFloat16, q_tile
    ).size_in_bytes()
    assert (
        cute_flash_gated.gated_smem_bytes(
            head_dim, kv_tile, kv_stage, torch.bfloat16, q_tile
        )
        == actual
    )
    fits = actual <= 232448
    assert (
        kv_stage
        in cute_flash_gated.gated_kv_stage_choices(
            head_dim, kv_tile, torch.bfloat16, smem_capacity=232448, q_tile=q_tile
        )
    ) is fits


def test_gated_smem_estimate_is_per_query_tile() -> None:
    # fp32 rows, head_dim 128, 64-wide KV tile: a three-deep ring fits only
    # under the 64-row Q tile (225 KB vs 257 KB under 128 rows).
    assert cute_flash_gated.gated_kv_stage_choices(
        128, 64, torch.float32, smem_capacity=232448
    ) == (2,)
    assert cute_flash_gated.gated_kv_stage_choices(
        128, 64, torch.float32, smem_capacity=232448, q_tile=64
    ) == (2, 3)


def _compile_only_launcher(
    cute_kernel: object, grid: tuple[int, ...], *args: object, **kwargs: object
) -> None:
    import cutlass.cute as cute

    from helion.runtime.cute import launcher as cute_launcher

    cuda_driver = importlib.import_module("cuda.bindings.driver")

    block = kwargs.pop("block", (256, 1, 1))
    assert isinstance(block, tuple)
    grid_xyz = (int(grid[0]), 1, 1)
    with patch.object(
        cute_launcher, "_validate_cute_launcher_tensor", lambda arg: None
    ):
        launch = cute_launcher._build_cute_schema_and_args(
            cute_kernel, tuple(args), grid_xyz
        )
    jit_func = cute_launcher._create_cute_wrapper(
        cute_kernel, launch.schema, block, num_sm=148
    )
    cute.compile(jit_func, *launch.launch_args, cuda_driver.CUstream(0))


@pytest.mark.parametrize(
    ("head_dim", "kv_tile", "kv_stage", "fast_math", "q_tile", "warpgroups"),
    [
        (64, 128, 3, False, 128, 2),
        (32, 64, 2, False, 128, 2),
        (64, 64, 2, True, 128, 2),
        (64, 128, 3, False, 64, 4),
        (128, 64, 2, False, 64, 2),
    ],
)
def test_gated_kernel_compiles_without_gpu(
    head_dim: int,
    kv_tile: int,
    kv_stage: int,
    fast_math: bool,
    q_tile: int,
    warpgroups: int,
    tmp_path: Path,
) -> None:
    """Compile the generated host wrapper and device kernel through the CuTe DSL."""
    env_patch = (
        {}
        if torch.cuda.is_available() or "CUTE_DSL_ARCH" in os.environ
        else {"CUTE_DSL_ARCH": "sm_100a"}
    )
    args = _hstu_args(head_dim=head_dim)
    mod = _hstu_module()
    kernel = mod._helion_jagged_attention_kernel
    if fast_math:
        kernel = helion.kernel(fast_math=True)(kernel.fn)
    with _cpu_codegen_patches(), patch.dict(os.environ, env_patch):
        kernel.reset()
        bound = kernel.bind(args)
        config = helion.Config(
            block_sizes=[q_tile, kv_tile],
            cute_flash_kv_stage=kv_stage,
            cute_flash_gate_warpgroups=warpgroups,
        )
        code = bound.to_code(config)
        assert _gated_markers(code)
        assert ("approx=True" in code) is fast_math
        # The DSL re-reads the kernel source, so the module must live on disk.
        source_path = tmp_path / "gated_hstu.py"
        source_path.write_text(code)
        namespace: dict[str, object] = {}
        exec(compile(code, str(source_path), "exec"), namespace)
        host_fn = namespace["_helion_jagged_attention_kernel"]
        assert callable(host_fn)
        host_fn(*args, _launcher=_compile_only_launcher)


@pytest.mark.parametrize(
    ("head_dim", "kv_tile", "q_tile"), [(64, 128, 128), (128, 64, 128), (64, 128, 64)]
)
def test_hstu2_kernel_compiles_without_gpu(
    head_dim: int, kv_tile: int, q_tile: int, tmp_path: Path
) -> None:
    """Compile the sequence-form (fp32 / tf32) kernel through the CuTe DSL."""
    env_patch = (
        {}
        if torch.cuda.is_available() or "CUTE_DSL_ARCH" in os.environ
        else {"CUTE_DSL_ARCH": "sm_100a"}
    )
    mod = _hstu2_module()
    kernel = mod.jagged_hstu_attention
    args = _hstu2_args(head_dim=head_dim)
    with _cpu_codegen_patches(), patch.dict(os.environ, env_patch):
        kernel.reset()
        bound = kernel.bind(args)
        config = helion.Config(
            block_sizes=[q_tile, kv_tile],
            cute_flash_kv_stage=2,
            cute_flash_gate_warpgroups=2,
        )
        code = bound.to_code(config)
        assert _gated_markers(code)
        source_path = tmp_path / "gated_hstu2.py"
        source_path.write_text(code)
        namespace: dict[str, object] = {}
        exec(compile(code, str(source_path), "exec"), namespace)
        host_fn = namespace["jagged_hstu_attention"]
        assert callable(host_fn)
        host_fn(*args, _launcher=_compile_only_launcher)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _causal_keep(r: torch.Tensor, c: torch.Tensor, length: int) -> torch.Tensor:
    return r > c


def _band_keep(r: torch.Tensor, c: torch.Tensor, length: int) -> torch.Tensor:
    return (torch.remainder(c - r, 3) == 1) & (r < length) & (c < length)


def _row_scaled_keep(r: torch.Tensor, c: torch.Tensor, length: int) -> torch.Tensor:
    return 2 * r > c


def _row_only_keep(r: torch.Tensor, c: torch.Tensor, length: int) -> torch.Tensor:
    return (r < length - 1) & (c >= 0)


def _sequence_bounds(
    seq_offsets: torch.Tensor, *, pair_table: bool = False
) -> list[tuple[int, int]]:
    values = seq_offsets.tolist()
    if pair_table:
        return list(zip(values[0::2], values[1::2], strict=True))
    return list(itertools.pairwise(values))


def _stored_rows(length: int, max_seq_len: int) -> int:
    """Rows a kernel with a root ``if`` and no store skip writes per sequence."""
    return min(-(-length // 128) * 128, max_seq_len)


def _gated_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    *,
    alpha: float,
    max_seq_len: int,
    keep: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    kv_end: str,
    pair_table: bool = False,
) -> torch.Tensor:
    """fp32 reference with the kernels' semantics.

    ``keep(r, c, length)`` is the ``where`` condition for row ``r`` and column
    ``c`` of a sequence; ``kv_end`` is ``"tile_end"`` when the KV loop runs to
    ``tile_q.end`` (128-row query tiles) or ``"seq_len"``. Rows the kernels
    never store keep the output's initial value (zero here); the example's
    PyTorch reference keeps the diagonal, which the kernel excludes. With
    ``pair_table`` the offsets are ``(start, end)`` pairs, the output starts as
    NaN and every row of an active query tile is stored (zeros past the length).
    """
    initial = float("nan") if pair_table else 0.0
    out = torch.full(v.shape, initial, dtype=torch.float32, device=v.device)
    scale = 1.0 / max_seq_len
    for start, end in _sequence_bounds(seq_offsets, pair_table=pair_table):
        length = end - start
        if pair_table:
            out[start : start + _stored_rows(length, max_seq_len)] = 0.0
        qb, kb, vb = (t[start:end].float().transpose(0, 1) for t in (q, k, v))
        scores = torch.nn.functional.silu(qb @ kb.transpose(1, 2) * alpha) * scale
        r = torch.arange(length, device=v.device)[:, None]
        c = torch.arange(length, device=v.device)[None, :]
        mask = keep(r, c, length)
        if kv_end == "tile_end":
            mask = mask & (c < torch.clamp((r // 128 + 1) * 128, max=max_seq_len))
        scores = torch.where(mask, scores, 0.0)
        out[start:end] = (scores @ vb).transpose(0, 1)
    return out


def _jagged_inputs(
    lengths: list[int], heads: int, head_dim: int, *, padding: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device=DEVICE
    )
    total = int(seq_offsets[-1]) + padding
    q = torch.randn((total, heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    return q, torch.randn_like(q), torch.randn_like(q), seq_offsets


def _pair_table_inputs(
    lengths: list[int], heads: int, head_dim: int, *, max_seq_len: int, tail: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Each sequence sits in a slot padded to the rows its tiles store, so the
    zero rows after it alias no other sequence; ``tail`` rows follow the slots."""
    bounds: list[int] = []
    slots_end = 0
    for length in lengths:
        bounds.extend((slots_end, slots_end + length))
        slots_end += _stored_rows(length, max_seq_len)
    seq_offsets = torch.tensor(bounds, dtype=torch.int64, device=DEVICE)
    total = slots_end + tail
    q = torch.randn((total, heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    return q, torch.randn_like(q), torch.randn_like(q), seq_offsets, slots_end


def _run_gated(
    kernel: Any, args: tuple[Any, ...], config: helion.Config
) -> torch.Tensor:
    from helion._compiler.cute.mma_support import get_cute_mma_support

    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires tcgen05")
    kernel.reset()
    bound = kernel.bind(args)
    bound.set_config(config)
    assert _gated_markers(bound.to_code(config))
    out = bound(*args)
    torch.cuda.synchronize()
    return out


def _assert_matches(out: torch.Tensor, expected: torch.Tensor, tol: float) -> None:
    assert torch.isfinite(out).all()
    scale = expected.abs().max().item()
    torch.testing.assert_close(out.float(), expected, atol=tol * scale, rtol=tol)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("head_dim", "kv_stage", "kv_tile", "q_tile", "warpgroups"),
    [
        (64, 3, 128, 128, 2),
        (64, 2, 64, 128, 2),
        (32, 3, 128, 128, 2),
        (128, 2, 128, 128, 2),
        (128, 2, 64, 128, 2),
        (128, 2, 64, 128, 4),
        # 64-row tile: 16x64b TMEM shapes, lane-pair P/O exchanges, 32- and
        # 16-wide gate vectors.
        (64, 3, 128, 64, 1),
        (64, 3, 128, 64, 2),
        (64, 3, 128, 64, 4),
        (64, 2, 64, 64, 1),
        (64, 2, 64, 64, 2),
        (32, 3, 128, 64, 4),
        (128, 2, 128, 64, 4),
        (128, 2, 64, 64, 2),
    ],
)
def test_gated_hstu_matches_reference_at_jagged_lengths(
    head_dim: int, kv_stage: int, kv_tile: int, q_tile: int, warpgroups: int
) -> None:
    torch.manual_seed(2026)
    max_seq_len = 300
    q, k, v, seq_offsets = _jagged_inputs([300, 173, 261, 5], 2, head_dim)
    # ``alpha`` scales the gate argument to O(1) so the check is meaningful; the
    # example's default puts outputs near 1e-3.
    args = (max_seq_len, 0.05, q, k, v, seq_offsets)
    expected = _gated_reference(
        q,
        k,
        v,
        seq_offsets,
        alpha=0.05,
        max_seq_len=max_seq_len,
        keep=_causal_keep,
        kv_end="tile_end",
    )
    out = _run_gated(
        _hstu_module()._helion_jagged_attention_kernel,
        args,
        helion.Config(
            block_sizes=[q_tile, kv_tile],
            cute_flash_kv_stage=kv_stage,
            cute_flash_gate_warpgroups=warpgroups,
        ),
    )
    _assert_matches(out, expected, 2e-2)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "name", ["tile_end", "no_if", "band", "row_only", "pair_table", "row_scaled"]
)
def test_gated_variants_match_reference_on_gpu(name: str) -> None:
    torch.manual_seed(7)
    kernel = _ACCEPTED_VARIANTS[name]
    slots_end = 0
    if name == "row_only":
        # Rows past a sequence's length alias the next sequence, which is a
        # store race in the frontend as well; keep them in padding instead.
        max_seq_len, lengths, padding = 256, [256, 128, 173], 83
        keep, kv_end = _row_only_keep, "seq_len"
    else:
        max_seq_len, lengths, padding = 300, [300, 173, 261, 5], 0
        keep = {"band": _band_keep, "row_scaled": _row_scaled_keep}.get(
            name, _causal_keep
        )
        kv_end = "tile_end"
    if name == "pair_table":
        q, k, v, seq_offsets, slots_end = _pair_table_inputs(
            lengths, 2, 64, max_seq_len=max_seq_len, tail=16
        )
    else:
        q, k, v, seq_offsets = _jagged_inputs(lengths, 2, 64, padding=padding)
    args = (max_seq_len, 0.05, q, k, v, seq_offsets)
    expected = _gated_reference(
        q,
        k,
        v,
        seq_offsets,
        alpha=0.05,
        max_seq_len=max_seq_len,
        keep=keep,
        kv_end=kv_end,
        pair_table=name == "pair_table",
    )
    out = _run_gated(
        kernel, args, helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=3)
    )
    if name == "row_only":
        # Every row is written: the last row of each sequence and the padding
        # rows are stored as zeros, never skipped (the output starts as NaN).
        assert not torch.isnan(out).any()
        for _start, end in _sequence_bounds(seq_offsets):
            assert (out[end - 1] == 0).all()
    elif name == "pair_table":
        # The gap rows after each sequence are stored as zeros (no store skip
        # for a pair table); rows no query tile covers keep the NaN poison.
        assert torch.isnan(out[slots_end:]).all()
        for start, end in _sequence_bounds(seq_offsets, pair_table=True):
            gap = out[end : start + _stored_rows(end - start, max_seq_len)]
            assert (gap == 0).all()
        out, expected = out[:slots_end], expected[:slots_end]
    _assert_matches(out, expected, 2e-2)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("q_tile", [128, 64])
def test_gated_long_sequence_bitwise_across_warpgroups(q_tile: int) -> None:
    """A 3000-row sequence (24 KV tiles) lets the gate warpgroups drift a whole
    tile apart.  Packed P has its own TMEM buffers, so no warpgroup's P store
    aliases S columns another has yet to load: four warpgroups produce the
    one-warpgroup result bit for bit."""
    torch.manual_seed(11)
    q, k, v, seq_offsets = _jagged_inputs([3000], 2, 64)
    args = (3000, 0.05, q, k, v, seq_offsets)
    kernel = _hstu_module()._helion_jagged_attention_kernel

    def run(warpgroups: int) -> torch.Tensor:
        return _run_gated(
            kernel,
            args,
            helion.Config(
                block_sizes=[q_tile, 128],
                cute_flash_kv_stage=2,
                cute_flash_gate_warpgroups=warpgroups,
            ),
        )

    base = run(1)
    assert torch.isfinite(base).all()
    assert torch.equal(run(4), base)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("q_tile", [128, 64])
def test_gated_oversubscribed_grid_is_deterministic(q_tile: int) -> None:
    """kv tile 64 with one gate warpgroup (256 TMEM columns, 256 threads) lets
    two gated CTAs share an SM once the grid exceeds the SM count; the TMEM
    address the allocating warp publishes must be fenced or the CTAs alias
    each other's TMEM.  Three launches agree bit for bit with each other and
    with the one-CTA-per-SM kv 128 configuration."""
    torch.manual_seed(12)
    lengths = [64, 128, 192, 256, 320, 384, 448, 512] * 3
    q, k, v, seq_offsets = _jagged_inputs(lengths, 8, 64)
    args = (512, 0.05, q, k, v, seq_offsets)
    kernel = _hstu_module()._helion_jagged_attention_kernel
    reference = _run_gated(
        kernel,
        args,
        helion.Config(
            block_sizes=[q_tile, 128],
            cute_flash_kv_stage=2,
            cute_flash_gate_warpgroups=2,
        ),
    )
    assert torch.isfinite(reference).all()
    narrow = helion.Config(
        block_sizes=[q_tile, 64], cute_flash_kv_stage=2, cute_flash_gate_warpgroups=1
    )
    for _ in range(3):
        assert torch.equal(_run_gated(kernel, args, narrow), reference)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gated_fast_math_matches_reference_on_gpu() -> None:
    torch.manual_seed(11)
    max_seq_len = 300
    q, k, v, seq_offsets = _jagged_inputs([300, 173, 261, 5], 2, 64)
    args = (max_seq_len, 0.05, q, k, v, seq_offsets)
    expected = _gated_reference(
        q,
        k,
        v,
        seq_offsets,
        alpha=0.05,
        max_seq_len=max_seq_len,
        keep=_causal_keep,
        kv_end="tile_end",
    )
    kernel = helion.kernel(fast_math=True)(
        _hstu_module()._helion_jagged_attention_kernel.fn
    )
    config = helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=3)
    out = _run_gated(kernel, args, config)
    assert "approx=True" in kernel.bind(args).to_code(config)
    _assert_matches(out, expected, 3e-2)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("head_dim", "kv_tile", "warpgroups", "q_tile"),
    [
        (64, 128, 2, 128),
        (64, 128, 4, 128),
        (128, 64, 2, 128),
        # 64-row tile with 32-bit P (no lane-pair exchange for P).
        (64, 128, 4, 64),
        (64, 128, 1, 64),
        (128, 64, 2, 64),
    ],
)
def test_gated_hstu2_matches_reference_on_gpu(
    head_dim: int, kv_tile: int, warpgroups: int, q_tile: int
) -> None:
    """Sequence form (``jagged_hstu_attn_2``, fp32 rows, tf32 MMA) at jagged
    lengths, including an empty sequence and lengths past one query tile."""
    torch.manual_seed(5)
    mod = _hstu2_module()
    lengths = [300, 173, 0, 261, 5, 128]
    heads = 4
    seq_offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    total = int(seq_offsets[-1])
    q = torch.randn((total, heads, head_dim), dtype=torch.float32, device=DEVICE)
    k, v = torch.randn_like(q), torch.randn_like(q)
    max_seq_len = 300
    # ``alpha`` puts the gate argument at O(1) so the check is meaningful.
    args = (max_seq_len, 0.05, 1.0 / max_seq_len, q, k, v, seq_offsets)
    expected = mod.reference_jagged_hstu_attention(*args)
    out = _run_gated(
        mod.jagged_hstu_attention,
        args,
        helion.Config(
            block_sizes=[q_tile, kv_tile],
            cute_flash_kv_stage=2,
            cute_flash_gate_warpgroups=warpgroups,
        ),
    )
    assert out.dtype is torch.float32
    # tf32 operands (Helion's default dot precision): ~1e-3 relative.
    _assert_matches(out, expected, 1e-2)
