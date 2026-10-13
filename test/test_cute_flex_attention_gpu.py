"""GPU correctness of the fused FlexAttention envelope on the CuTe backend.

``examples/flex_attention.py::helion_flex_attention_kernel`` with real
``create_block_mask`` BlockMasks runs on the fa4 list-walk body (matcher
``flex_match``, renderer ``flash_score_mod``, emitter ``cute_flash``). Every
case pins the same configuration (the pruned-fa4 campaign structure) so the
resolved body is identical across shapes, checks the generated code for the
list-walk markers and compares against eager ``flex_attention`` (its math
path upcasts to fp32: output ``0`` and LSE ``-inf`` for a row without a
visible key). The all-full noop BlockMask is also compared with the dense
fa4 kernel at the same configuration.
"""

from __future__ import annotations

import functools
import math
import re
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import noop_mask

import helion
from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
import helion.language as hl
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.nn.attention.flex_attention import BlockMask

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

# The pruned-fa4 campaign structure (8x2 packets at the 16/8 cadence, the
# chunked body with a two-deep PASS2 pipe, split P arrivals, the descending
# walk, the TMA epilogue); the resolver pins the rest for the list walk.
FLEX_CONFIG: dict[str, object] = {
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
}
# The dense comparison body: the same structure plus the pins the resolver
# applies to the list walk, requested explicitly for the dense plan.
DENSE_CONFIG: dict[str, object] = {
    **{
        key: value
        for key, value in FLEX_CONFIG.items()
        # The dense plan walks its KV range by ``kv_order``; the causal
        # order knob is the interval and list walks' alone.
        if key != cute_flash.FLASH_CAUSAL_KV_ORDER_KEY
    },
    "block_sizes": [1, 128, 128],
    cute_flash.FLASH_PERSISTENT_KEY: False,
    cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
    cute_flash.FLASH_KV_ORDER_KEY: "descending",
}


@functools.cache
def _flex_kernel() -> Kernel:
    """The example kernel on the CuTe backend (a separate ``Kernel`` object,
    so the example module's singleton keeps its own configs)."""
    module = import_path(EXAMPLES_DIR / "flex_attention.py")
    kernel = module.helion_flex_attention_kernel
    return Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute"))


def _identity(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return score


@helion.kernel(backend="cute", static_shapes=True)
def _dense_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain attention on the fa4 body (base-2 LSE), the bitwise reference of
    the all-full noop BlockMask."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty(
        [q_view.size(0), m_dim, 1], device=q_in.device, dtype=torch.float32
    )
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
        lse[tile_b, tile_m, :] = (m_i + torch.log2(l_i))[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.reshape(q_in.size()[:-1])


# -- mask_mods (shared by the kernel and the reference) ---------------------


def _causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return q_idx >= kv_idx


def _sliding_causal(window: int) -> Callable[..., Any]:
    def sliding_causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= window)

    return sliding_causal


def _hide_first_rows(rows: int) -> Callable[..., Any]:
    """Causal, but the first ``rows`` query rows see no key at all."""

    def hide_first_rows(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= rows) & (q_idx >= kv_idx)

    return hide_first_rows


def _global_window_masks(
    is_global: torch.Tensor, window: int
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Longformer: a window or a global row or a global column, the bool
    ``[S]`` table read once per row (item-invariant) and once per column
    (lane-contiguous: a 16-lane byte vector load). Kernel form and twin."""

    def helion_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        window_keep = torch.abs(q_idx - kv_idx) <= window
        return window_keep | hl.load(is_global, [q_idx]) | hl.load(is_global, [kv_idx])

    def torch_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        window_keep = torch.abs(q_idx - kv_idx) <= window
        return window_keep | is_global[q_idx] | is_global[kv_idx]

    return helion_mask, torch_mask


# -- score_mods: the kernel form (``hl.load``) and the torch twin ------------


def _rel_bias_mods(
    table: torch.Tensor, offset: int
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """A relative-position bias ``table[kv - q + offset]`` (``offset`` is a
    closure int: a host scalar of the plan)."""

    def helion_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(table, [kv_idx - q_idx + offset])

    def torch_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + table[kv_idx - q_idx + offset]

    return helion_mod, torch_mod


def _zeros_like_mods(
    table: torch.Tensor,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Accumulate into ``torch.zeros_like(score)`` (attention-gym
    graphormer-edge): the ``full`` is sized by ``sym_size_int`` metadata."""

    def helion_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        total = torch.zeros_like(score)
        total = total + hl.load(table, [kv_idx])
        return score + total

    def torch_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        total = torch.zeros_like(score)
        total = total + table[kv_idx]
        return score + total

    return helion_mod, torch_mod


def _column_table_mods(
    table: torch.Tensor,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """A per-key bias ``table[kv_idx]`` (lane-contiguous: a 16-byte vector
    load of ``16 // itemsize`` lanes; bf16 gives 8, int32 gives 4)."""

    def helion_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(table, [kv_idx]).to(torch.float32)

    def torch_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + table[kv_idx].to(torch.float32)

    return helion_mod, torch_mod


def _gather_mods(
    bias: torch.Tensor,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """A per-(batch, row, column) gather ``bias[b, q, kv]``."""

    def helion_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(bias, [b, q_idx, kv_idx])

    def torch_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + bias[b, q_idx, kv_idx]

    return helion_mod, torch_mod


# -- running -------------------------------------------------------------------


def _qkv(
    batch: int,
    heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    kv_heads = heads if kv_heads is None else kv_heads
    q = torch.randn(batch, heads, q_seq, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=dtype, device=DEVICE)
    return q, k, v


def _block_mask(
    mask_mod: Callable[..., Any],
    batch: int | None,
    heads: int | None,
    q_seq: int,
    kv_seq: int,
    *,
    block_size: int | tuple[int, int] = 128,
) -> BlockMask:
    return create_block_mask(
        mask_mod, batch, heads, q_seq, kv_seq, BLOCK_SIZE=block_size, device=DEVICE
    )


def _kernel_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: Callable[..., Any],
    block_mask: BlockMask,
    *,
    scale: float,
    enable_gqa: bool,
    mask_mod: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    """The kernel arguments the example wrapper builds (both lists always).
    ``mask_mod`` overrides the BlockMask's own (a table-reading mask needs its
    ``hl.load`` form in the kernel and its indexing twin in the reference)."""
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = torch.zeros_like(block_mask.kv_num_blocks)
        full_idx = torch.zeros_like(block_mask.kv_indices)
    return (
        q,
        k,
        v,
        score_mod,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod if mask_mod is None else mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        scale,
        enable_gqa,
    )


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
    *,
    mods: tuple[Callable[..., Any], Callable[..., Any]] | None = None,
    enable_gqa: bool = False,
    kernel_mask_mod: Callable[..., Any] | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fused kernel at ``FLEX_CONFIG`` and the eager reference:
    ``(code, out, lse, ref_out, ref_lse)``."""
    scale = 1.0 / math.sqrt(q.size(-1))
    helion_mod, torch_mod = mods if mods is not None else (_identity, None)
    args = _kernel_args(
        q,
        k,
        v,
        helion_mod,
        block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        mask_mod=kernel_mask_mod,
    )
    code, (out, lse) = code_and_output(_flex_kernel(), args, **FLEX_CONFIG)
    ref_out, ref_lse = flex_attention(
        q,
        k,
        v,
        score_mod=torch_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=True,
    )
    return code, out, lse, ref_out, ref_lse


def _assert_list_walk(code: str, *, mask: bool, score: bool) -> None:
    """The fa4 list-walk body: the union bitmask, the per-step status and
    the hooks the plan's regions call for."""
    assert "_flash_mFlexPartCnt = " in code
    assert "flash_flex_u0 = " in code
    assert "flash_flex_status = " in code
    assert ("flash_flex_needs_mask" in code) == mask
    assert ("flash_flex0s_" in code) == score
    assert ("flash_flex0m_" in code) == mask


def _assert_sub_tile_store(code: str, q_seq: int) -> None:
    """An output shorter than one 128-row tile must not use the TMA store
    (its tiled view has no whole tile and the box lands nowhere: M=64 left
    the output unwritten); the row-guarded staged store serves it. Longer
    ragged lengths may use either store (the numerics decide)."""
    if q_seq < 128:
        assert "_flash_tma_o" not in code
        assert "flash_live_rows0" in code


def _assert_no_table_reads(code: str) -> None:
    """A trivial walk (one BlockMask KV block covering the sequence) never
    binds or reads the tables, whichever way the emitter renders its
    constant per-tile status."""
    assert "_flash_mFlexPartCnt" not in code
    assert "_flash_mFlexFullIdx" not in code
    assert "flash_flex_cp0" not in code


def _assert_close(
    out: torch.Tensor,
    lse: torch.Tensor,
    ref_out: torch.Tensor,
    ref_lse: torch.Tensor,
) -> None:
    assert not torch.isnan(out).any().item()
    assert not torch.isnan(lse).any().item()
    atol = 3e-2 if out.dtype is torch.bfloat16 else 2e-2
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=2e-2)
    empty = torch.isinf(ref_lse)
    assert torch.equal(torch.isinf(lse), empty)
    torch.testing.assert_close(lse[~empty], ref_lse[~empty], atol=2e-3, rtol=1e-3)


# -- (i) the all-full noop BlockMask against the dense fa4 kernel --------------


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("head_dim", "dtype"), [(64, torch.bfloat16), (128, torch.float16)]
)
def test_noop_block_mask_matches_the_dense_kernel(
    head_dim: int, dtype: torch.dtype
) -> None:
    """An all-full BlockMask with noop mods is the dense walk; against the
    dense fa4 body at the same configuration the output is bitwise at
    ``head_dim == 64`` (the runtime scale ``1/8`` folds exactly) and the
    base-2 LSE converts exactly."""
    q, k, v = _qkv(2, 4, 1024, 1024, head_dim, dtype)
    # torch's ``noop_mask`` (a constant-true predicate the matcher drops; a
    # ``q_idx >= 0`` would be a real mask region on every full tile).
    block_mask = _block_mask(noop_mask, 2, 4, 1024, 1024)
    assert int(block_mask.kv_num_blocks.sum()) == 0
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=False, score=False)
    _assert_close(out, lse, ref_out, ref_lse)
    dense_code, (dense_out, dense_lse) = code_and_output(
        _dense_attention, (q, k, v), **DENSE_CONFIG
    )
    assert "flash_flex" not in dense_code
    if head_dim == 64:
        assert torch.equal(out, dense_out)
    else:
        torch.testing.assert_close(out, dense_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, dense_lse * math.log(2.0), atol=1e-4, rtol=1e-5)


@onlyBackends(["cute"])
def test_block_mask_none_is_the_trivial_dense_walk() -> None:
    """``block_mask=None`` (``_create_empty_block_mask``: one KV block over
    the whole sequence) is a trivial walk: the same body over every tile
    (a constant full status) and no table reads."""
    from torch.nn.attention.flex_attention import _create_empty_block_mask

    q, k, v = _qkv(2, 4, 256, 256, 64, torch.bfloat16)
    block_mask = _create_empty_block_mask(q, k)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_no_table_reads(code)
    assert "flash_flex_needs_mask" not in code
    _assert_close(out, lse, ref_out, ref_lse)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=2e-2)


# -- (ii), (iii) mask_mods ------------------------------------------------------


@onlyBackends(["cute"])
def test_causal_mask() -> None:
    q, k, v = _qkv(2, 4, 1024, 1024, 64, torch.bfloat16)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_sliding_causal_mask() -> None:
    q, k, v = _qkv(2, 4, 2048, 2048, 128, torch.float16)
    block_mask = _block_mask(_sliding_causal(256), 2, 4, 2048, 2048)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_bool_table_mask_mod_by_row_and_column() -> None:
    """A bool ``[S]`` table read by ``q_idx`` and by ``kv_idx`` (attention-gym
    global-sw): the per-column read is a 16-lane vector load, which must load
    bytes (a ``vector<16xi1>`` load trapped the DSL compile with SIGILL)."""
    q, k, v = _qkv(2, 4, 1024, 1024, 64, torch.bfloat16)
    is_global = torch.zeros(1024, dtype=torch.bool, device=DEVICE)
    is_global[:64] = True
    is_global[::256] = True
    helion_mask, torch_mask = _global_window_masks(is_global, 128)
    block_mask = _block_mask(torch_mask, 2, 4, 1024, 1024)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, kernel_mask_mod=helion_mask
    )
    _assert_list_walk(code, mask=True, score=False)
    assert "_cute_flex_load_vec(" in code
    assert "16, cutlass.Int8)" in code
    assert "16, cutlass.Boolean)" not in code
    _assert_close(out, lse, ref_out, ref_lse)


# -- (iv), (v) score_mods -------------------------------------------------------


@onlyBackends(["cute"])
def test_relative_bias_table_score_mod_with_causal_mask() -> None:
    """A closure table read through ``hl.load`` by ``kv - q + offset`` (the
    offset is a closure int: a host scalar kernel param)."""
    q, k, v = _qkv(2, 4, 1024, 1024, 64, torch.bfloat16)
    table = torch.randn(2048, dtype=torch.float32, device=DEVICE)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, mods=_rel_bias_mods(table, 1024)
    )
    _assert_list_walk(code, mask=True, score=True)
    assert "_flash_aux0 = " in code
    assert "_flash_hs" in code
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_3d_gather_score_mod() -> None:
    """``hl.load(t, [b, q_idx, kv_idx])``: a per-element gather."""
    q, k, v = _qkv(2, 2, 512, 512, 64, torch.float16)
    bias = torch.randn(2, 512, 512, dtype=torch.float32, device=DEVICE)
    block_mask = _block_mask(_causal, 2, 2, 512, 512)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, mods=_gather_mods(bias)
    )
    _assert_list_walk(code, mask=True, score=True)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_zeros_like_score_mod() -> None:
    q, k, v = _qkv(2, 2, 512, 512, 64, torch.float16)
    table = torch.randn(512, dtype=torch.float32, device=DEVICE)
    block_mask = _block_mask(_causal, 2, 2, 512, 512)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, mods=_zeros_like_mods(table)
    )
    _assert_list_walk(code, mask=True, score=True)
    assert "sym_size" not in code
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
@pytest.mark.parametrize(("dtype", "lanes"), [(torch.bfloat16, 8), (torch.int32, 4)])
def test_column_table_vector_loads(dtype: torch.dtype, lanes: int) -> None:
    """A table read by ``kv_idx`` alone takes the vector-load form at the
    dtype's 16-byte lane count (bf16 8, int32 4; bool 16 is the global-sw
    case above)."""
    q, k, v = _qkv(2, 2, 512, 512, 64, torch.float16)
    if dtype is torch.int32:
        table = torch.randint(-3, 4, (512,), dtype=dtype, device=DEVICE)
    else:
        table = torch.randn(512, device=DEVICE).to(dtype)
    block_mask = _block_mask(_causal, 2, 2, 512, 512)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, mods=_column_table_mods(table)
    )
    _assert_list_walk(code, mask=True, score=True)
    vec_call = re.search(r"_cute_flex_load_vec\(.*, (\d+), cutlass\.(\w+)\)", code)
    assert vec_call is not None and int(vec_call.group(1)) == lanes
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_kv_length_over_32_tiles_uses_two_bitmask_words() -> None:
    """``N = 4352`` (34 KV tiles): the per-row tile bitmasks span two 32-bit
    words (the word-select chains of the walk step and the status test)."""
    q, k, v = _qkv(1, 2, 4352, 4352, 64, torch.bfloat16)
    block_mask = _block_mask(_sliding_causal(1024), 1, 2, 4352, 4352)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    assert "flash_flex_u1 = " in code
    assert "flash_flex_rem1 = flash_flex_u1" in code
    _assert_close(out, lse, ref_out, ref_lse)


# -- (vi) rows without a visible key -------------------------------------------


@onlyBackends(["cute"])
def test_rows_without_keys_give_zero_and_minus_inf() -> None:
    q, k, v = _qkv(2, 2, 512, 512, 64, torch.bfloat16)
    block_mask = _block_mask(_hide_first_rows(128), 2, 2, 512, 512)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    assert torch.equal(out[:, :, :128], torch.zeros_like(out[:, :, :128]))
    assert torch.isinf(lse[:, :, :128]).all().item()
    assert (lse[:, :, :128] < 0).all().item()
    _assert_close(out, lse, ref_out, ref_lse)


# -- (vii), (viii), (ix) lengths --------------------------------------------------


@onlyBackends(["cute"])
@pytest.mark.parametrize(("q_seq", "kv_seq"), [(1024, 512), (512, 1024)])
def test_unequal_lengths(q_seq: int, kv_seq: int) -> None:
    q, k, v = _qkv(2, 4, q_seq, kv_seq, 64, torch.bfloat16)
    block_mask = _block_mask(_causal, 2, 4, q_seq, kv_seq)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "dtype"),
    [(1152, 1152, torch.bfloat16), (64, 256, torch.float16)],
)
def test_ragged_query_length(q_seq: int, kv_seq: int, dtype: torch.dtype) -> None:
    """``M % 256 == 128`` (a half-live last pair) and ``M == 64`` (a half-live
    single stage)."""
    q, k, v = _qkv(2, 4, q_seq, kv_seq, 64, dtype)
    block_mask = _block_mask(_sliding_causal(192), 2, 4, q_seq, kv_seq)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    assert "flash_live_rows1" in code
    _assert_sub_tile_store(code, q_seq)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_trivial_walk_with_a_mask_region() -> None:
    """``M == N == 64``: the 128-wide KV block covers the sequence, so the
    BlockMask walk is trivial and every tile takes the masked body (a
    constant partial status)."""
    q, k, v = _qkv(2, 4, 64, 64, 64, torch.bfloat16)
    block_mask = _block_mask(_causal, 2, 4, 64, 64)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_no_table_reads(code)
    assert "flash_flex_needs_mask" in code
    _assert_sub_tile_store(code, 64)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "mask_mod"),
    [(1024, 1088, noop_mask), (1088, 1088, _causal)],
)
def test_ragged_kv_length(
    q_seq: int, kv_seq: int, mask_mod: Callable[..., Any]
) -> None:
    """``N % 128 == 64``: the last KV tile's zero-filled tail columns are
    masked in whichever list the tile appears (the dense noop walk keeps the
    tile in the full list)."""
    q, k, v = _qkv(2, 4, q_seq, kv_seq, 64, torch.bfloat16)
    block_mask = _block_mask(mask_mod, 2, 4, q_seq, kv_seq)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    assert (
        "_flash_kv_tail_cols = cutlass.Int32(cutlass.select_("
        "flash_kv == _flash_last_tile, cutlass.Int32(64), cutlass.Int32(128)))"
    ) in code
    assert "flash_kv == _flash_last_tile" in code
    _assert_sub_tile_store(code, q_seq)
    _assert_close(out, lse, ref_out, ref_lse)


# -- (x) grouped-query heads ----------------------------------------------------


@onlyBackends(["cute"])
def test_grouped_query_heads() -> None:
    q, k, v = _qkv(2, 4, 1024, 1024, 64, torch.bfloat16, kv_heads=2)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask, enable_gqa=True)
    _assert_list_walk(code, mask=True, score=False)
    _assert_close(out, lse, ref_out, ref_lse)


# -- (xi) a 256-row BlockMask ------------------------------------------------------


@onlyBackends(["cute"])
def test_256_row_block_mask() -> None:
    q, k, v = _qkv(2, 4, 1024, 1024, 64, torch.bfloat16)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024, block_size=(256, 128))
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    assert "flash_flex_row0 = flash_m_pair" in code
    _assert_close(out, lse, ref_out, ref_lse)


# -- (xii) a large grid with broadcast and expanded tables ----------------------


@onlyBackends(["cute"])
@pytest.mark.parametrize("expanded", [False, True])
def test_large_grid_broadcast_tables(expanded: bool) -> None:
    """``B=4, H=16``: a ``(1, 1, R)`` BlockMask indexed modulo its sizes, and
    the same mask with per-(batch, head) tables."""
    q, k, v = _qkv(4, 16, 512, 512, 64, torch.float16)
    if expanded:
        block_mask = _block_mask(_sliding_causal(128), 4, 16, 512, 512)
    else:
        block_mask = _block_mask(_sliding_causal(128), None, None, 512, 512)
    assert tuple(block_mask.kv_num_blocks.shape[:2]) == (
        (4, 16) if expanded else (1, 1)
    )
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    if expanded:
        assert "flash_flex_b = flash_bh // _flash_heads" in code
    else:
        assert "flash_flex_b = cutlass.Int32(0)" in code
    _assert_close(out, lse, ref_out, ref_lse)
