"""Whole-render goldens for the CuTe flash emitters and resolver (CPU).

The flash refactors (the fa4 / ws forward bodies and their persistent
scaffolds, the backward bodies, the resolver and the fragment surface) are
"generated code byte-identical" changes.  The other CPU tests pin fragments
of the renders; this module pins whole renders as SHA-256 digests in
``test/data/cute_flash_render_golden.json`` for the surfaces no other CPU
test renders as a whole (the four ``helion_flash_bwd`` plan shapes with
their host wrapper bodies, ``ws_2cta`` with and without the persistent
cluster loop, the separate load warp, fa4 under ``persistent_loop`` while /
counted, the CLC scheduler, the flex list walk's persistent / split-KV / tail
forms, the MLA pair, the segment tiling), plus the resolver's full field tuple
and ``flash_effective_config_values`` over the probe inputs and the fragment
surfaces of representative shapes (verbatim, so a diff names the knob).

Renders are normalized the way ``dumpdiff`` compares them: ``# src[...]``
comments and the embedded source hash are not generated code.  On a mismatch
the test writes the actual render next to the failure (``tmp_path``) so the
diff against a base worktree's render can be read.

Regenerate after an intentional change (from the repository root; the
module form puts the root on ``sys.path`` for the ``test._cute_aux`` import)::

    CUDA_VISIBLE_DEVICES="" HELION_BACKEND=cute \\
        python -m test.test_cute_flash_render_golden --accept [--dump DIR] [names...]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
import helion.language as hl
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import FLEX_SCORE_KIND
from helion._compiler.cute.attention_plan import ROW_KV_BEGIN_MASK_KIND
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._testing.cute_aux import cpu_codegen
from helion._testing.cute_aux import flash_env

GOLDEN = Path(__file__).parent / "data" / "cute_flash_render_golden.json"

_SRC_COMMENT = re.compile(r"# src\[[^\]]*\]")
_SOURCE_HASH = re.compile(r"(_helion_cute_source_hash = )'[0-9a-f]+'")


def _normalize(code: str) -> str:
    code = _SRC_COMMENT.sub("# src[...]", code)
    return _SOURCE_HASH.sub(r"\1'<HASH>'", code)


def _digest(code: str) -> dict[str, object]:
    normalized = _normalize(code)
    return {
        "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "lines": normalized.count("\n"),
    }


def _jsonable(value: object) -> object:
    """JSON form of resolver / fragment values (tuples as lists, the rest by
    ``repr`` when JSON has no form for it)."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
# These copy the attention kernels other test files define on purpose: the
# digests below must not move when another file edits its copy.


@helion.kernel(backend="cute", static_shapes=True)
def _dense_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention_dv(q_in, k_in, v_in):
    """The causal envelope whose V/O head dim differs from Q/K's (the MLA pair)."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    v_dim = hl.specialize(v_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, v_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty(
        [q_view.size(0), m_dim, v_dim], dtype=q_in.dtype, device=q_in.device
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, v_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view([q_in.size(0), q_in.size(1), m_dim, v_dim])


@helion.kernel(backend="cute", static_shapes=True)
def _row_kv_begin_attention(q_in, k_in, v_in, kv_begin):
    """``key >= kv_begin[batch, query]`` with the causal bound (ROW_KV_BEGIN)."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    heads = hl.specialize(q_in.size(1))
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    begin_view = kv_begin.reshape([-1, m_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            begin_batch = tile_b.index // heads
            row_begin = begin_view[begin_batch, tile_m]
            causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            in_range = tile_n.index[None, None, :] >= row_begin[:, :, None]
            qk = torch.where(in_range & causal, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@functools.cache
def _causal_attention_fast_math() -> Kernel:
    """The causal kernel under ``fast_math`` (the resident softmax packets)."""
    return helion.kernel(
        _causal_attention.fn, backend="cute", static_shapes=True, fast_math=True
    )


@functools.cache
def _attention_example() -> Any:
    return import_path(EXAMPLES_DIR / "attention.py")


@functools.cache
def _flex_example() -> Any:
    return import_path(EXAMPLES_DIR / "flex_attention.py")


@functools.cache
def _blackwell_example() -> Any:
    return import_path(EXAMPLES_DIR / "blackwell_attention.py")


def _cute_kernel(kernel: Kernel) -> Kernel:
    return Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute"))


@functools.cache
def _flex_forward_kernel() -> Kernel:
    return _cute_kernel(_flex_example().helion_flex_attention_kernel)


@functools.cache
def _flex_backward_kernel() -> Kernel:
    return _cute_kernel(_flex_example().helion_flex_attention_backward_kernel)


@functools.cache
def _attention_backward_kernel() -> Kernel:
    example = _attention_example()
    return helion.kernel(
        example.attention_backward.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _render(kernel: Kernel, args: tuple[object, ...], values: dict[str, object]) -> str:
    """``kernel`` bound on CPU tensors, rendered at ``values`` (normalized by
    the bound kernel's config spec, as the autotuner's configs are)."""
    kernel.reset()
    with flash_env(), cpu_codegen():
        bound = kernel.bind(args)
        config = helion.Config(**values)
        return bound.to_code(bound._normalized_config_copy(config))


def _qkv(
    shape: tuple[int, int, int, int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim_v: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    batch, heads, seq, head_dim = shape
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim_v or head_dim, dtype=dtype)
    return q, k, v


def _forward(
    kernel: Kernel,
    shape: tuple[int, int, int, int],
    values: dict[str, object],
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim_v: int | None = None,
) -> str:
    return _render(
        kernel,
        _qkv(shape, dtype=dtype, head_dim_v=head_dim_v),
        {"block_sizes": [1, 128, 128], **values},
    )


# The example registers Triton AutoWS tunables; a config completed by the
# autotuner carries them (the default config's values).
_BLACKWELL_TUNABLES: dict[str, object] = {
    "_triton_range_id_data_partition_factor": 0,
    "_triton_range_value_data_partition_factor": 2,
    "_triton_config_maxRegAutoWS": 152,
}


def _blackwell_forward(
    values: dict[str, object], *, shape: tuple[int, int, int, int] = (1, 4, 256, 64)
) -> str:
    """``examples/blackwell_attention.py``'s forward: the flattened-rows form
    (one root axis over ``B*H*M`` rows, 2-D operands, a runtime softmax scale
    applied after the row max, a packed ``mul.f32x2`` rescale) on the dense
    LSE surface."""
    kernel = _blackwell_example().blackwell_attention_kernel
    kernel = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )
    q, k, v = _qkv(shape)
    return _render(
        kernel,
        (q, k, v, 1.0 / math.sqrt(shape[-1])),
        {"block_sizes": [128, 128], **_BLACKWELL_TUNABLES, **values},
    )


def _document_kv_begin(lengths_per_batch: list[list[int]]) -> torch.Tensor:
    rows = []
    for lengths in lengths_per_batch:
        starts: list[int] = []
        start = 0
        for length in lengths:
            starts.extend([start] * length)
            start += length
        rows.append(torch.tensor(starts, dtype=torch.int32))
    return torch.stack(rows).contiguous()


def _row_kv_begin(values: dict[str, object]) -> str:
    documents = _document_kv_begin(
        [[100, 300, 212, 412], [512, 250, 262], [1024], [40, 90, 130, 764]]
    )
    q, k, v = _qkv((4, 2, 1024, 64), dtype=torch.float16)
    return _render(
        _row_kv_begin_attention,
        (q, k, v, documents),
        {"block_sizes": [1, 128, 128], "cute_flash_pipeline_family": "fa4", **values},
    )


def _causal_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> torch.Tensor:
    return q_idx >= kv_idx


def _identity(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return score


# The flex list-walk structure pinned by ``test_cute_flex_attention_gpu.py``.
_FLEX_CONFIG: dict[str, object] = {
    "block_sizes": [128, 128],
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
    cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
    cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
    cute_flash.FLASH_S_LOAD_REP_KEY: 16,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_DISC_PIPE_KEY: 2,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    cute_flash.FLASH_EPI_TMA_KEY: True,
    cute_flash.FLASH_KV_SPLIT_KEY: 1,
}


def _flex_forward(
    values: dict[str, object], *, head_dim: int = 64, seq: int = 1024
) -> str:
    from torch.nn.attention.flex_attention import create_block_mask

    q, k, v = _qkv((2, 4, seq, head_dim))
    block_mask = create_block_mask(
        _causal_mask, 2, 4, seq, seq, device=torch.device("cpu")
    )
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    assert full_num is not None and full_idx is not None
    args = (
        q,
        k,
        v,
        _identity,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        1.0 / math.sqrt(head_dim),
        False,
    )
    return _render(_flex_forward_kernel(), args, {**_FLEX_CONFIG, **values})


def _flex_backward(
    values: dict[str, object], *, head_dim: int = 64, seq: int = 512
) -> str:
    from torch.nn.attention.flex_attention import create_block_mask

    torch.manual_seed(0)
    batch, heads = 2, 4
    dtype = torch.bfloat16
    q, k, v, do, out = (
        torch.randn(batch, heads, seq, head_dim, dtype=dtype) for _ in range(5)
    )
    lse = torch.randn(batch, heads, seq, dtype=torch.float32)
    block_mask = create_block_mask(
        _causal_mask, None, None, seq, seq, device=torch.device("cpu"), BLOCK_SIZE=128
    )
    args = _flex_example().flex_attention_backward_args(
        q, k, v, do, lse, out, None, None, block_mask, 0.125, False
    )
    kernel = _flex_backward_kernel()
    kernel.reset()
    with (
        flash_env(),
        cpu_codegen(),
        patch(
            "helion._compiler.cute.analysis.attention_match._attention_flash_supported",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.analysis.attention_match._attention_flash_gate_enabled",
            return_value=True,
        ),
    ):
        bound = kernel.bind(args)
        config = bound.config_spec.default_config()
        if values:
            config = helion.Config(**{**config.config, **values})
        return bound.to_code(config)


def _dense_backward(
    values: dict[str, object], *, head_dim: int = 64, seq: int = 512
) -> str:
    torch.manual_seed(0)
    shape = (1, 2, seq, head_dim)
    q, k, v, o, do = (torch.randn(shape, dtype=torch.bfloat16) for _ in range(5))
    lse = torch.randn(shape[:-1], dtype=torch.float32)
    delta = torch.randn(shape[:-1], dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(head_dim)
    args = (q, k * (sm_scale * 1.4426950408889634), v, o, lse, do, delta, sm_scale)
    return _render(
        _attention_backward_kernel(), args, {"block_sizes": [128, 128], **values}
    )


_FA4 = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
_WS = {
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
    cute_flash.FLASH_S_STAGE_KEY: 2,
    cute_flash.FLASH_KV_STAGE_KEY: 2,
}
_WS_2CTA = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
_LOADER_WARP = {
    cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
    cute_flash.FLASH_WS_MASK_PHASES_KEY: True,
    cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: 1,
}
_P = cute_flash.FLASH_PERSISTENT_KEY

RENDER_CASES: dict[str, Callable[[], str]] = {
    # fa4 dense / causal: flat, the persistent loop forms, the schedulers.
    "fa4_dense_hd64_flat": lambda: _forward(
        _dense_attention, (2, 4, 1024, 64), {**_FA4, _P: False}
    ),
    # The flattened-rows form (blackwell_attention): the runtime scale
    # reaches the fa4 body, the ws_overlap body and the row programs.
    "fa4_flattened_rows_hd64_persistent": lambda: _blackwell_forward(
        {**_FA4, _P: True}, shape=(2, 8, 1024, 64)
    ),
    "ws_flattened_rows_hd64_flat": lambda: _blackwell_forward({**_WS, _P: False}),
    "row_mma_flattened_rows_hd128": lambda: _blackwell_forward(
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "row_mma"}, shape=(1, 4, 256, 128)
    ),
    "fa4_dense_hd128_persistent_while": lambda: _forward(
        _dense_attention,
        (2, 4, 2048, 128),
        {**_FA4, _P: True, cute_flash.FLASH_PERSISTENT_LOOP_KEY: "while"},
    ),
    "fa4_dense_hd128_persistent_counted": lambda: _forward(
        _dense_attention,
        (2, 4, 2048, 128),
        {**_FA4, _P: True, cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted"},
    ),
    "fa4_dense_hd128_persistent_recompute": lambda: _forward(
        _dense_attention,
        (2, 4, 2048, 128),
        {**_FA4, _P: True, cute_flash.FLASH_RECOMPUTE_TILE_COORDS_KEY: True},
    ),
    "fa4_dense_hd64_fp16_persistent": lambda: _forward(
        _dense_attention, (2, 8, 1024, 64), {**_FA4, _P: True}, dtype=torch.float16
    ),
    "fa4_causal_hd64_flat": lambda: _forward(
        _causal_attention, (2, 4, 1024, 64), {**_FA4, _P: False}
    ),
    "fa4_causal_hd128_default": lambda: _forward(
        _causal_attention, (2, 4, 2048, 128), {}
    ),
    "fa4_causal_hd128_fast_math_resident": lambda: _forward(
        _causal_attention_fast_math(),
        (2, 4, 2048, 128),
        {
            cute_flash.FLASH_EXP2_PACKET_KEY: (
                "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"
            )
        },
    ),
    "fa4_clc_local_tma_4d": lambda: _forward(
        _dense_attention,
        (1, 2, 512, 64),
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc_local_tma_4d",
            cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY: 2,
            cute_flash.FLASH_CLC_PDL_KEY: True,
            cute_flash.FLASH_CLC_STAGES_KEY: 3,
        },
        dtype=torch.float16,
    ),
    "fa4_cga2_local_tma_4d": lambda: _forward(
        _dense_attention,
        (1, 2, 512, 64),
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_cga2_local_tma_4d"},
        dtype=torch.float16,
    ),
    "fa4_local_tma_hd128": lambda: _forward(
        _dense_attention,
        (2, 4, 2048, 128),
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_local_tma"},
    ),
    "fa4_2cta_hd128": lambda: _forward(
        _dense_attention,
        (2, 4, 2048, 128),
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
    ),
    "fa4_2cta_causal_hd64": lambda: _forward(
        _causal_attention,
        (2, 4, 2048, 64),
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_causal",
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
            cute_flash.FLASH_P_STORE_REP_KEY: 16,
        },
        dtype=torch.float16,
    ),
    "fa4_deep_1cta_hd64": lambda: _forward(
        _dense_attention,
        (2, 4, 1024, 64),
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_deep_1cta"},
    ),
    # The two-warpgroup ws body: flat, persistent, the separate load warp.
    "ws_overlap_hd64_flat": lambda: _forward(
        _dense_attention, (2, 4, 1024, 64), {**_WS, _P: False}
    ),
    "ws_overlap_hd64_persistent": lambda: _forward(
        _dense_attention, (2, 4, 1024, 64), {**_WS, _P: True}
    ),
    "ws_overlap_causal_hd64_persistent": lambda: _forward(
        _causal_attention, (2, 4, 1024, 64), {**_WS, _P: True}
    ),
    "ws_overlap_hd64_loader_warp_persistent": lambda: _forward(
        _dense_attention, (2, 4, 1024, 64), {**_WS, _P: True, **_LOADER_WARP}
    ),
    "ws_overlap_m64_one_pass": lambda: _forward(
        _dense_attention,
        (1, 4, 256, 64),
        {
            **_WS,
            _P: False,
            cute_flash.FLASH_Q_TILE_M_KEY: 64,
            cute_flash.FLASH_WS_ONE_PASS_KEY: True,
        },
    ),
    # The single-tile body's cooperative pair.
    "ws_2cta_hd256_flat": lambda: _forward(
        _dense_attention, (2, 4, 1024, 256), {**_WS_2CTA, _P: False}
    ),
    "ws_2cta_hd256_persistent": lambda: _forward(
        _dense_attention, (2, 4, 1024, 256), {**_WS_2CTA, _P: True}
    ),
    "ws_2cta_causal_hd256_loader_warp_flat": lambda: _forward(
        _causal_attention, (2, 4, 1024, 256), {**_WS_2CTA, _P: False, **_LOADER_WARP}
    ),
    "ws_2cta_causal_hd256_loader_warp_persistent": lambda: _forward(
        _causal_attention, (2, 4, 1024, 256), {**_WS_2CTA, _P: True, **_LOADER_WARP}
    ),
    # The MLA pair: the uneven ring and the Q-staged output.
    "mla_192_128_default": lambda: _forward(
        _causal_attention_dv, (2, 4, 1024, 192), {}, head_dim_v=128
    ),
    "mla_192_128_uneven_o_over_q": lambda: _forward(
        _causal_attention_dv,
        (2, 4, 1024, 192),
        {
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_O_OVER_Q_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
        },
        head_dim_v=128,
    ),
    # The per-row KV lower bound: the global pair grid and the segment tiling.
    "row_kv_begin_global": lambda: _row_kv_begin(
        {cute_flash.FLASH_ROW_BOUND_TILING_KEY: "global"}
    ),
    "row_kv_begin_segment": lambda: _row_kv_begin(
        {
            cute_flash.FLASH_ROW_BOUND_TILING_KEY: "segment",
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
        }
    ),
    # The flex list walk: one item per CTA, the persistent item table (with
    # the hoisted next-item head), the split-KV walk and the tail split.
    "flex_fa4_hd64_flat": lambda: _flex_forward({_P: False}),
    "flex_fa4_hd64_persistent": lambda: _flex_forward({_P: True}),
    "flex_fa4_hd64_kv_split2": lambda: _flex_forward(
        {_P: False, cute_flash.FLASH_KV_SPLIT_KEY: 2}
    ),
    "flex_fa4_hd64_tail_split": lambda: _flex_forward(
        {_P: False, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4}
    ),
    "flex_fa4_hd128_persistent": lambda: _flex_forward({_P: True}, head_dim=128),
    # The fused backward: the four ``helion_flash_bwd`` plan shapes (plain,
    # two_cta, flex_two_cta, flex_pair) with their host wrapper bodies.
    "bwd_dense_hd64": lambda: _dense_backward({}),
    "bwd_dense_hd128": lambda: _dense_backward({}, head_dim=128),
    "bwd_dense_hd128_persistent": lambda: _dense_backward(
        {"cute_flash_bwd_persistent": 1}, head_dim=128
    ),
    "bwd_dense_hd128_two_cta": lambda: _dense_backward(
        {"cute_flash_bwd_two_cta": 1}, head_dim=128
    ),
    "bwd_dense_hd128_two_cta_persistent": lambda: _dense_backward(
        {"cute_flash_bwd_two_cta": 1, "cute_flash_bwd_persistent": 1}, head_dim=128
    ),
    "bwd_dense_hd64_persistent": lambda: _dense_backward(
        {"cute_flash_bwd_persistent": 1}
    ),
    "bwd_flex_hd64": lambda: _flex_backward({}),
    "bwd_flex_hd64_persistent": lambda: _flex_backward(
        {"cute_flash_bwd_persistent": 1}
    ),
    "bwd_flex_hd128_two_cta": lambda: _flex_backward(
        {"cute_flash_bwd_two_cta": 1}, head_dim=128
    ),
    "bwd_flex_hd256_pair": lambda: _flex_backward({}, head_dim=256),
}


# --------------------------------------------------------------------------
# the resolver and the fragment surface
# --------------------------------------------------------------------------

_SLIDING_WINDOW = {
    "has_kv_tile_pruning": True,
    "standard_causal_output": True,
    "has_score_modifiers": True,
    "score_modifier_kinds": (SLIDING_WINDOW_MASK_KIND,),
    "plain_row_body": False,
}
_ROW_KV_BEGIN = {**_SLIDING_WINDOW, "score_modifier_kinds": (ROW_KV_BEGIN_MASK_KIND,)}
_FLEX = {
    "has_list_walk": True,
    "score_modifier_kinds": (FLEX_SCORE_KIND,),
    "standard_dense_output": False,
}

# (head_dim, num_kv, config, resolver keywords): the surfaces the fragment
# builder and its probes resolve.
RESOLVER_CASES: dict[str, tuple[int, int, dict[str, object] | None, dict[str, Any]]] = {
    "dense_hd64_nk8_default": (64, 8, None, {"standard_dense_output": True}),
    "dense_hd64_nk8_fa4": (64, 8, {**_FA4}, {"standard_dense_output": True}),
    "dense_hd64_nk8_ws": (64, 8, {**_WS}, {"standard_dense_output": True}),
    "dense_hd64_nk8_ws_loader": (
        64,
        8,
        {**_WS, _P: True, **_LOADER_WARP},
        {"standard_dense_output": True},
    ),
    "dense_hd64_nk1": (64, 1, None, {"standard_dense_output": True}),
    "dense_hd128_nk16_default": (128, 16, None, {"standard_dense_output": True}),
    "dense_hd128_nk16_counted": (
        128,
        16,
        {**_FA4, _P: True, cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted"},
        {"standard_dense_output": True},
    ),
    "dense_hd128_nk16_clc": (
        128,
        16,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc",
            cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY: 2,
        },
        {"standard_dense_output": True, "num_bh": 8},
    ),
    "dense_hd128_nk16_lse": (
        128,
        16,
        None,
        {"standard_dense_output": True, "dense_lse_output": True},
    ),
    "dense_hd256_nk8_default": (256, 8, None, {"standard_dense_output": True}),
    "dense_hd256_nk8_ws_2cta_persistent": (
        256,
        8,
        {**_WS_2CTA, _P: True, **_LOADER_WARP},
        {"standard_dense_output": True},
    ),
    "causal_hd64_nk8_default": (
        64,
        8,
        None,
        {"is_causal": True, "standard_causal_output": True},
    ),
    "causal_hd128_nk16_default": (
        128,
        16,
        None,
        {"is_causal": True, "standard_causal_output": True, "num_bh": 8},
    ),
    "causal_hd128_nk16_fast_math": (
        128,
        16,
        None,
        {
            "is_causal": True,
            "standard_causal_output": True,
            "num_bh": 8,
            "fast_math": True,
        },
    ),
    "causal_hd64_nk16_2cta": (
        64,
        16,
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_causal"},
        {"is_causal": True, "standard_causal_output": True},
    ),
    "mla_192_128_nk8_causal": (
        192,
        8,
        None,
        {"is_causal": True, "standard_causal_output": True, "head_dim_v": 128},
    ),
    "mla_192_128_nk8_uneven": (
        192,
        8,
        {
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_O_OVER_Q_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
        },
        {"is_causal": True, "standard_causal_output": True, "head_dim_v": 128},
    ),
    "sliding_window_hd64_nk8": (64, 8, None, {**_SLIDING_WINDOW}),
    "row_kv_begin_hd64_nk8_segment": (
        64,
        8,
        {cute_flash.FLASH_ROW_BOUND_TILING_KEY: "segment"},
        {**_ROW_KV_BEGIN},
    ),
    "flex_hd64_nk8_default": (64, 8, None, {**_FLEX, "num_bh": 8}),
    "flex_hd64_nk8_persistent": (64, 8, {_P: True}, {**_FLEX, "num_bh": 8}),
    "flex_hd64_nk8_kv_split2": (
        64,
        8,
        {cute_flash.FLASH_KV_SPLIT_KEY: 2},
        {**_FLEX, "num_bh": 8},
    ),
    "fp8_hd128_nk16_dense": (
        128,
        16,
        None,
        {"standard_dense_output": True, "dtype": torch.float8_e4m3fn},
    ),
}

# (head_dim, num_kv, fragment keywords): the surfaces whose fragments are
# pinned verbatim (defaults, choices, search choices).
FRAGMENT_CASES: dict[str, tuple[int, int, dict[str, Any]]] = {
    "dense_hd64_nk8": (64, 8, {"standard_dense_output": True, "num_bh": 8}),
    "dense_hd64_nk1": (64, 1, {"standard_dense_output": True, "num_bh": 8}),
    "dense_hd128_nk16": (128, 16, {"standard_dense_output": True, "num_bh": 32}),
    "dense_hd128_nk16_lse": (
        128,
        16,
        {"standard_dense_output": True, "dense_lse_output": True, "num_bh": 32},
    ),
    "dense_hd256_nk8": (256, 8, {"standard_dense_output": True, "num_bh": 8}),
    "causal_hd64_nk8": (
        64,
        8,
        {"is_causal": True, "standard_causal_output": True, "num_bh": 8},
    ),
    "causal_hd128_nk16_fast_math": (
        128,
        16,
        {
            "is_causal": True,
            "standard_causal_output": True,
            "num_bh": 32,
            "fast_math": True,
        },
    ),
    "mla_192_128_nk8_causal": (
        192,
        8,
        {
            "is_causal": True,
            "standard_causal_output": True,
            "head_dim_v": 128,
            "num_bh": 8,
        },
    ),
    "sliding_window_hd64_nk8": (64, 8, {**_SLIDING_WINDOW, "num_bh": 8}),
    "flex_hd64_nk8": (
        64,
        8,
        {**_FLEX, "num_bh": 8, "flex_has_mask_region": True, "q_seq": 1024},
    ),
    "fp8_hd128_nk16": (
        128,
        16,
        {"standard_dense_output": True, "dtype": torch.float8_e4m3fn, "num_bh": 32},
    ),
}


def _resolver_snapshot(name: str) -> dict[str, object]:
    head_dim, num_kv, config, kwargs = RESOLVER_CASES[name]
    with flash_env():
        resolved = cute_flash.resolve_flash_config(head_dim, num_kv, config, **kwargs)
    return {
        "fields": _jsonable(dataclasses.asdict(resolved)),
        "effective": _jsonable(cute_flash.flash_effective_config_values(resolved)),
    }


def _fragment_snapshot(name: str) -> dict[str, object]:
    head_dim, num_kv, kwargs = FRAGMENT_CASES[name]
    with flash_env():
        fragments = cute_flash.flash_autotune_fragments(head_dim, num_kv, **kwargs)
    return {
        key: {
            "type": type(fragment).__name__,
            "default": _jsonable(fragment.default()),
            **_jsonable(vars(fragment)),  # pyrefly: ignore[bad-argument-type]
        }
        for key, fragment in fragments.items()
    }


# --------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------


@functools.cache
def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", sorted(RENDER_CASES))
def test_render_matches_golden(name: str, tmp_path: Path) -> None:
    code = RENDER_CASES[name]()
    actual = _digest(code)
    expected = _golden()["renders"][name]
    if actual != expected:
        dump = tmp_path / f"{name}.py"
        dump.write_text(_normalize(code))
        pytest.fail(
            f"render {name!r} changed: {actual} != {expected} (golden). The actual "
            f"render is at {dump}; diff it against a base worktree's render or "
            f"regenerate the golden after an intentional change."
        )


@pytest.mark.parametrize("name", sorted(RESOLVER_CASES))
def test_resolver_matches_golden(name: str) -> None:
    assert _resolver_snapshot(name) == _golden()["resolver"][name]


@pytest.mark.parametrize("name", sorted(FRAGMENT_CASES))
def test_fragments_match_golden(name: str) -> None:
    assert _fragment_snapshot(name) == _golden()["fragments"][name]


def test_golden_covers_every_case() -> None:
    golden = _golden()
    assert set(golden["renders"]) == set(RENDER_CASES)
    assert set(golden["resolver"]) == set(RESOLVER_CASES)
    assert set(golden["fragments"]) == set(FRAGMENT_CASES)


# --------------------------------------------------------------------------
# regeneration
# --------------------------------------------------------------------------


def _cases(selection: list[str], available: Iterator[str]) -> list[str]:
    names = sorted(available)
    return names if not selection else [name for name in names if name in selection]


def _check_selection(selection: list[str]) -> None:
    known = set(RENDER_CASES) | set(RESOLVER_CASES) | set(FRAGMENT_CASES)
    unknown = sorted(set(selection) - known)
    if unknown:
        raise SystemExit(f"--accept: unknown case names {unknown}")


def _accept(selection: list[str], *, dump_dir: Path | None) -> None:
    _check_selection(selection)
    # Named cases replace their entries in the existing golden; without names
    # the golden is rebuilt from the tables, so a removed case leaves the file.
    existing: dict[str, Any] = (
        json.loads(GOLDEN.read_text()) if selection and GOLDEN.exists() else {}
    )
    renders = dict(existing.get("renders", {}))
    for name in _cases(selection, iter(RENDER_CASES)):
        code = RENDER_CASES[name]()
        renders[name] = _digest(code)
        if dump_dir is not None:
            (dump_dir / f"{name}.py").write_text(_normalize(code))
        print(f"{name}: {renders[name]}", file=sys.stderr)
    resolver = dict(existing.get("resolver", {}))
    for name in _cases(selection, iter(RESOLVER_CASES)):
        resolver[name] = _resolver_snapshot(name)
    fragments = dict(existing.get("fragments", {}))
    for name in _cases(selection, iter(FRAGMENT_CASES)):
        fragments[name] = _fragment_snapshot(name)
    comment = (
        "Generated by test_cute_flash_render_golden.py --accept (CPU, fake GB300 "
        "target); renders are SHA-256 digests of the normalized to_code() text."
    )
    GOLDEN.write_text(_golden_text(comment, renders, resolver, fragments))


def _golden_text(
    comment: str,
    renders: dict[str, Any],
    resolver: dict[str, Any],
    fragments: dict[str, Any],
) -> str:
    """The golden's JSON: one line per render, per resolver field tuple and
    per fragment key, so a diff names the case and the knob that moved."""
    one = functools.partial(json.dumps, separators=(", ", ": "))

    def section(name: str, entries: dict[str, str]) -> str:
        body = ",\n".join(f"  {one(key)}: {text}" for key, text in entries.items())
        return f" {one(name)}: {{\n{body}\n }}"

    def nested(name: str, cases: dict[str, dict[str, Any]]) -> str:
        parts = []
        for case, values in sorted(cases.items()):
            lines = ",\n".join(
                f"   {one(key)}: {one(value)}" for key, value in values.items()
            )
            parts.append(f"  {one(case)}: {{\n{lines}\n  }}")
        return f" {one(name)}: {{\n" + ",\n".join(parts) + "\n }"

    return (
        "{\n"
        f' "_comment": {one(comment)},\n'
        + section("renders", {k: one(v) for k, v in sorted(renders.items())})
        + ",\n"
        + nested("resolver", resolver)
        + ",\n"
        + nested("fragments", fragments)
        + "\n}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept", action="store_true", help="rewrite the golden")
    parser.add_argument(
        "--dump", type=Path, default=None, help="also write the renders here"
    )
    parser.add_argument("names", nargs="*", help="case names (default: all)")
    args = parser.parse_args()
    if not args.accept:
        parser.error("nothing to do without --accept")
    if args.dump is not None:
        args.dump.mkdir(parents=True, exist_ok=True)
    _accept(args.names, dump_dir=args.dump)


if __name__ == "__main__":
    main()
