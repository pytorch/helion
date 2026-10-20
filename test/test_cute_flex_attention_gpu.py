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
    # The seeded default splits a small grid's KV walk over several CTAs
    # (``cute_flash_kv_split``); these tests pin the one-CTA walk (the dense
    # kernel comparisons are bitwise only there).
    cute_flash.FLASH_KV_SPLIT_KEY: 1,
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
    overrides: dict[str, object] | None = None,
    extra_config: dict[str, object] | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fused kernel at ``FLEX_CONFIG`` (``overrides`` / ``extra_config`` on top) and the
    eager reference: ``(code, out, lse, ref_out, ref_lse)``."""
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
    config = {**FLEX_CONFIG, **(overrides or {}), **(extra_config or {})}
    code, (out, lse) = code_and_output(_flex_kernel(), args, **config)
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
    """The fa4 list-walk body: the union bitmask (from the host-built item
    words by default, or rebuilt from the count tables), the per-step status
    and the hooks the plan's regions call for."""
    assert "_flash_mFlexWords = " in code or "_flash_mFlexPartCnt = " in code
    # The union of the item's lists: register words, or the cursor walk's
    # header (two words or more).
    assert "flash_flex_u0 = " in code or "flash_flex_total = " in code
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
    words. The host words table is walked through the cursor (the header's
    population and word range, one word at a time); the device-built
    register masks (still searched at two words) keep the word-select
    chains of the walk step and the status test. Both agree bitwise."""
    q, k, v = _qkv(1, 2, 4352, 4352, 64, torch.bfloat16)
    block_mask = _block_mask(_sliding_causal(1024), 1, 2, 4352, 4352)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    _assert_cursor_walk(code)
    _assert_close(out, lse, ref_out, ref_lse)
    code_r, out_r, lse_r, _, _ = _run(
        q, k, v, block_mask, overrides={cute_flash.FLASH_FLEX_ITEM_WORDS_KEY: False}
    )
    assert "flash_flex_u1 = " in code_r
    assert "flash_flex_rem1 = flash_flex_u1" in code_r
    assert torch.equal(out_r, out) and torch.equal(lse_r, lse)


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
    # One 256-row list per pair: two words per item (partial, full), no
    # second-stage words.
    assert ") * cutlass.Int32(2)" in code
    assert "flash_flex_f0_0 = " in code and "flash_flex_p1_0" not in code
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


# -- (xiii) strided operands ------------------------------------------------------


def _as_bshd(t: torch.Tensor) -> torch.Tensor:
    """``t``'s values in a ``[B, S, H, D]`` allocation viewed ``[B, H, S, D]``
    (the layout every model produces)."""
    batch, heads, seq, head_dim = t.shape
    return (
        torch.empty(batch, seq, heads, head_dim, dtype=t.dtype, device=t.device)
        .transpose(1, 2)
        .copy_(t)
    )


def _assert_rank4_maps(code: str, *, kv_group: int = 1) -> None:
    """The strided render: rank-4 ``(S, D, H, B)`` TMA maps built from the
    recorded strides, indexed by head and batch."""
    assert "'operand_strides': (" in code
    assert "tQgQ = tQgQ_qdl[None, None, 0, flash_head, flash_batch]" in code
    kv_head = "flash_head" if kv_group == 1 else f"flash_head // {kv_group}"
    assert f"tKgK = tKgK_kdl[None, None, 0, {kv_head}, flash_batch]" in code
    assert "tQgQ_qdl[None, None, 0, flash_bh]" not in code


def _assert_rank3_maps(code: str) -> None:
    assert "'operand_strides'" not in code
    assert "flash_head, flash_batch]" not in code


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("head_dim", "dtype", "kv_heads", "q_seq", "kv_seq"),
    [
        (64, torch.bfloat16, 4, 1024, 1024),
        (128, torch.float16, 4, 1024, 1024),
        (64, torch.bfloat16, 2, 1024, 1024),
        (64, torch.bfloat16, 4, 1024, 512),
        (64, torch.bfloat16, 4, 1152, 1152),
    ],
)
def test_transposed_views_match_the_contiguous_fused_path(
    head_dim: int, dtype: torch.dtype, kv_heads: int, q_seq: int, kv_seq: int
) -> None:
    """``[B, S, H, D].transpose(1, 2)`` Q/K/V run the fused list walk through
    rank-4 TMA maps (the O ``empty_like`` keeps the layout): bitwise the
    contiguous fused result at the same configuration (TMA epilogue), and
    within tolerance of the eager reference. Covers both dtypes and head
    dims, grouped heads, ``M != N`` and a ragged query length."""
    q, k, v = _qkv(2, 4, q_seq, kv_seq, head_dim, dtype, kv_heads=kv_heads)
    block_mask = _block_mask(_causal, 2, 4, q_seq, kv_seq)
    enable_gqa = kv_heads != 4
    code_c, out_c, lse_c, _, _ = _run(q, k, v, block_mask, enable_gqa=enable_gqa)
    _assert_rank3_maps(code_c)
    qt, kt, vt = (_as_bshd(t) for t in (q, k, v))
    assert not qt.is_contiguous()
    code, out, lse, ref_out, ref_lse = _run(
        qt, kt, vt, block_mask, enable_gqa=enable_gqa
    )
    _assert_rank4_maps(code, kv_group=4 // kv_heads)
    _assert_list_walk(code, mask=True, score=False)
    assert out.stride() == qt.stride()
    assert torch.equal(out, out_c)
    assert torch.equal(lse, lse_c)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "overrides",
    [
        {cute_flash.FLASH_EPI_TMA_KEY: False, cute_flash.FLASH_EPI_STG_KEY: True},
        {
            cute_flash.FLASH_EPI_TMA_KEY: False,
            cute_flash.FLASH_EPI_STG_KEY: True,
            cute_flash.FLASH_KV_SPLIT_KEY: 2,
        },
        {cute_flash.FLASH_KV_SPLIT_KEY: 2},
        {cute_flash.FLASH_PERSISTENT_KEY: False},
    ],
    ids=["staged_store", "split_kv_merge", "split_kv_tma_merge", "non_persistent"],
)
def test_transposed_views_every_output_path(overrides: dict[str, object]) -> None:
    """The staged epilogue store, the split-KV merge store (which addresses
    O by its strides, through ``_flash_mOt`` with the staged store and
    ``_flash_mO`` with the TMA epilogue) and the non-persistent grid all
    write the transposed O bitwise the contiguous run of the same
    configuration."""
    q, k, v = _qkv(2, 4, 1152, 1152, 64, torch.bfloat16)
    block_mask = _block_mask(_sliding_causal(192), 2, 4, 1152, 1152)
    _, out_c, lse_c, _, _ = _run(q, k, v, block_mask, overrides=overrides)
    qt, kt, vt = (_as_bshd(t) for t in (q, k, v))
    code, out, lse, ref_out, ref_lse = _run(qt, kt, vt, block_mask, overrides=overrides)
    _assert_rank4_maps(code)
    if overrides.get(cute_flash.FLASH_KV_SPLIT_KEY, 1) > 1:
        tma = overrides.get(cute_flash.FLASH_EPI_TMA_KEY, True)
        view = "_flash_mO" if tma else "_flash_mOt"
        assert (
            f"flash_split_gO = cute.make_tensor({view}.iterator + cute.assume(flash_batch * "
            in code
        )
    assert torch.equal(out, out_c)
    assert torch.equal(lse, lse_c)
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_packed_qkv_slices() -> None:
    """Q/K/V as head-dim slices of one packed ``[B, S, H, 3 * D]`` buffer
    (outer strides ``3 * D``): fused, bitwise the contiguous result."""
    q, k, v = _qkv(2, 4, 1024, 1024, 64, torch.bfloat16)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    _, out_c, lse_c, _, _ = _run(q, k, v, block_mask)
    qkv = torch.empty(2, 1024, 4, 192, dtype=q.dtype, device=DEVICE)
    qp = qkv[..., :64].transpose(1, 2).copy_(q)
    kp = qkv[..., 64:128].transpose(1, 2).copy_(k)
    vp = qkv[..., 128:].transpose(1, 2).copy_(v)
    assert qp.stride() == (786432, 192, 768, 1)
    code, out, lse, ref_out, ref_lse = _run(qp, kp, vp, block_mask)
    _assert_rank4_maps(code)
    assert torch.equal(out, out_c)
    assert torch.equal(lse, lse_c)
    _assert_close(out, lse, ref_out, ref_lse)


# -- (xiv) head dims beyond 64 and 128 -----------------------------------------
#
# Every multiple of 16 from 16 to 128 runs the same fa4 list-walk body at the
# exact width: the CuTe DSL picks the SMEM swizzle the width admits (64 B at
# multiples of 32, 32 B otherwise), the TMA boxes and the MMA extents are exact
# and the two TMEM output accumulators take ``head_dim`` columns each.


def _assert_close_to_fp32(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
    *,
    torch_mod: Callable[..., Any] | None = None,
    enable_gqa: bool = False,
) -> None:
    """The output against the fp32 eager reference (the tolerance is the
    I/O dtype's own rounding of a unit-variance output, which the bf16 /
    fp16 eager paths reach themselves)."""
    ref32 = flex_attention(
        q.float(),
        k.float(),
        v.float(),
        score_mod=torch_mod,
        block_mask=block_mask,
        scale=1.0 / math.sqrt(q.size(-1)),
        enable_gqa=enable_gqa,
    )
    atol = 2e-2 if out.dtype is torch.bfloat16 else 4e-3
    torch.testing.assert_close(out.float(), ref32, atol=atol, rtol=0.0)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("head_dim", "dtype"),
    [
        pytest.param(16, torch.bfloat16, id="d16-bf16"),
        pytest.param(32, torch.bfloat16, id="d32-bf16"),
        pytest.param(80, torch.bfloat16, id="d80-bf16"),
        pytest.param(96, torch.float16, id="d96-fp16"),
        pytest.param(112, torch.bfloat16, id="d112-bf16"),
    ],
)
def test_head_dim_causal(head_dim: int, dtype: torch.dtype) -> None:
    q, k, v = _qkv(2, 4, 1024, 1024, head_dim, dtype)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_list_walk(code, mask=True, score=False)
    assert f"'head_dim': {head_dim}," in code
    assert f"flash_pvt.partition_shape_C((128, {head_dim}))" in code
    _assert_close(out, lse, ref_out, ref_lse)
    _assert_close_to_fp32(out, q, k, v, block_mask)


@onlyBackends(["cute"])
@pytest.mark.parametrize("head_dim", [32, 80, 96])
def test_head_dim_score_mod_and_grouped_heads(head_dim: int) -> None:
    """A relative-position bias score_mod over the causal mask, two K/V heads
    shared by four query heads."""
    q, k, v = _qkv(2, 4, 1024, 1024, head_dim, torch.bfloat16, kv_heads=2)
    torch.manual_seed(1)
    table = torch.randn(2 * 1024, device=DEVICE, dtype=torch.float32)
    mods = _rel_bias_mods(table, 1024)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, mods=mods, enable_gqa=True
    )
    _assert_list_walk(code, mask=True, score=True)
    _assert_close(out, lse, ref_out, ref_lse)
    _assert_close_to_fp32(out, q, k, v, block_mask, torch_mod=mods[1], enable_gqa=True)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("head_dim", "q_seq", "dtype"),
    [
        pytest.param(16, 1024, torch.bfloat16, id="d16"),
        pytest.param(80, 1024, torch.bfloat16, id="d80"),
        pytest.param(96, 960, torch.float16, id="d96-ragged"),
        pytest.param(112, 64, torch.bfloat16, id="d112-sub-tile"),
        pytest.param(48, 1024, torch.bfloat16, id="d48"),
    ],
)
def test_head_dim_staged_vector_store(
    head_dim: int, q_seq: int, dtype: torch.dtype
) -> None:
    """The row-guarded staged store (the TMA store off): a width whose
    16-byte packets per row do not divide the epilogue warp spreads one pass
    over a row slice instead of whole rows (threads past the layout would
    otherwise address outside the tile)."""
    q, k, v = _qkv(2, 4, q_seq, 1024, head_dim, dtype)
    block_mask = _block_mask(_causal, 2, 4, q_seq, 1024)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, extra_config={cute_flash.FLASH_EPI_TMA_KEY: False}
    )
    assert "_flash_tma_o" not in code
    # The whole-tile store, or its row-guarded sibling for a ragged length.
    assert "_helion_flash_rt.fa4_store_o_smem_to_gmem" in code
    threads_per_row = math.gcd(head_dim // 8, 32)
    if threads_per_row == head_dim // 8:
        # Whole rows per pass: the historical layout text.
        assert (
            f"flash_o_tO_shape_dim_1 = {head_dim} // flash_o_async_copy_elems" in code
        )
    else:
        assert f"flash_o_tO_shape_dim_1 = {threads_per_row}\n" in code
    _assert_close(out, lse, ref_out, ref_lse)
    _assert_close_to_fp32(out, q, k, v, block_mask)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("head_dim", "kv_stage", "packet", "schedule"),
    [
        pytest.param(16, 51, "8x1", "16/4", id="d16-depth51"),
        pytest.param(32, 24, "4x2", "16/4", id="d32-depth24"),
        pytest.param(80, 7, "8x2", "16/8", id="d80-depth7"),
        pytest.param(96, 5, "1x1", "8/2", id="d96-depth5"),
    ],
)
def test_head_dim_deepest_staged_ring(
    head_dim: int, kv_stage: int, packet: str, schedule: str
) -> None:
    """The deepest K/V ring the byte model allows beside the staged output at
    the width (hd16 reaches the depth where the per-stage barrier pair pushes
    the storage header past 3 KB), with the plain exp2 packets."""
    q, k, v = _qkv(2, 4, 2048, 2048, head_dim, torch.bfloat16)
    block_mask = _block_mask(_sliding_causal(512), 2, 4, 2048, 2048)
    code, out, lse, ref_out, ref_lse = _run(
        q,
        k,
        v,
        block_mask,
        extra_config={
            cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
            cute_flash.FLASH_EXP2_PACKET_KEY: packet,
            cute_flash.FLASH_E2E_SCHEDULE_KEY: schedule,
        },
    )
    assert f"'kv_stage': {kv_stage}," in code
    _assert_list_walk(code, mask=True, score=False)
    _assert_close(out, lse, ref_out, ref_lse)
    _assert_close_to_fp32(out, q, k, v, block_mask)


# -- (xv) the (192, 128) MLA pair on the list walk ---------------------------


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("with_score_mod", "epi_tma"),
    [
        pytest.param(False, True, id="causal-tma"),
        pytest.param(True, True, id="score-tma"),
        pytest.param(True, False, id="score-staged"),
    ],
)
def test_mla_pair_list_walk_stages_output_in_the_q_stages(
    with_score_mod: bool, epi_tma: bool
) -> None:
    """(192, 128) through the example kernel (its accumulator and output take
    V's width): two staged 128 x 128 output tiles do not fit beside two
    192-wide Q tiles and a two-deep ring, and the list walk has no direct
    store, so its output is staged in the dead Q stages of the
    one-item-per-CTA body (the resolver pins ``o_over_q`` on and
    ``persistent`` off; a persistent request resolves the same way)."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 1024, 192, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(2, 4, 1024, 192, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(2, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    mods = None
    if with_score_mod:
        table = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        mods = _column_table_mods(table)
    code, out, lse, ref_out, ref_lse = _run(
        q,
        k,
        v,
        block_mask,
        mods=mods,
        # The pair's packets are the exact cadence and the resident family
        # (keyed on its 128-wide value tiles); the shared structure's 8x2
        # plain packet is not among them.
        extra_config={
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_EXP2_PACKET_KEY: "1x1",
            cute_flash.FLASH_EPI_TMA_KEY: epi_tma,
        },
    )
    _assert_list_walk(code, mask=True, score=with_score_mod)
    assert "'head_dim': 192, 'head_dim_v': 128," in code
    assert "'persistent': False," in code
    assert f"'epi_tma': {epi_tma}," in code
    assert f"'epi_stg': {not epi_tma}," in code
    # The staged tiles alias the Q stages: no separate ``sO`` bytes.
    assert "flash_fa4_shared_storage(192, 2, 2, 2, cutlass.BFloat16, False, " in code
    _assert_close(out, lse, ref_out, ref_lse)
    _assert_close_to_fp32(
        out, q, k, v, block_mask, torch_mod=None if mods is None else mods[1]
    )


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("aux_budget", "kv_stage"),
    [
        pytest.param(0, 3, id="uneven-ring"),
        pytest.param(16384, 2, id="aux-tables-decline-the-uneven-ring"),
    ],
)
def test_mla_pair_list_walk_uneven_ring_against_the_aux_tables(
    aux_budget: int, kv_stage: int
) -> None:
    """The pair's uneven three-slot ring fills shared memory exactly; a
    relative-bias score_mod whose staged table (``cute_flash_flex_aux_smem``,
    8 KB here) takes bytes beyond the header padding declines it back to the
    two-deep ring instead of overflowing the budget at launch."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 1024, 192, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(2, 4, 1024, 192, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(2, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    block_mask = _block_mask(_causal, 2, 4, 1024, 1024)
    torch.manual_seed(1)
    table = torch.randn(2 * 1024, device=DEVICE, dtype=torch.float32)
    mods = _rel_bias_mods(table, 1024)
    code, out, lse, ref_out, ref_lse = _run(
        q,
        k,
        v,
        block_mask,
        mods=mods,
        extra_config={
            cute_flash.FLASH_EXP2_PACKET_KEY: "1x1",
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_FLEX_AUX_SMEM_KEY: aux_budget,
        },
    )
    _assert_list_walk(code, mask=True, score=True)
    assert f"'kv_stage': {kv_stage}," in code
    assert ("flex_aux_bytes=" in code) == (aux_budget > 0)
    _assert_close(out, lse, ref_out, ref_lse)
    _assert_close_to_fp32(out, q, k, v, block_mask, torch_mod=mods[1])


# -- (xvi) long KV lengths: the cursor walk over the words table -----------------


def _assert_cursor_walk(code: str) -> None:
    """From two bitmask words on the roles walk the host words table through
    a cursor: no per-word registers or select chains (code independent of
    the KV length)."""
    assert (
        "flash_flex_total = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow])" in code
    )
    assert "while flash_flex_rem == 0 and flash_flex_cw" in code
    assert "flash_flex_rem1" not in code and "flash_flex_u0" not in code


def _half_rows_causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    """Causal, but only the first 128 rows of every 256-row pair see keys:
    every pair's second stage is dead (the dead-stage split fires)."""
    return (q_idx % 256 < 128) & (q_idx >= kv_idx)


def _global_tile_and_window(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    """The first KV tile plus a causal window: every item's union has empty
    words between its two runs (the cursor's hole scan)."""
    return (kv_idx < 128) | ((q_idx >= kv_idx) & (q_idx - kv_idx <= 640))


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "mask_mod", "overrides"),
    [
        (16384, 16384, _sliding_causal(1024), {}),
        (16384, 16384, _causal, {cute_flash.FLASH_KV_SPLIT_KEY: 2}),
        (
            16384,
            16384,
            _sliding_causal(1024),
            {cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending"},
        ),
        (4160, 16384, _causal, {}),
        (
            16384,
            16384,
            _half_rows_causal,
            {
                cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True,
                cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY: True,
            },
        ),
        (
            16384,
            16384,
            _half_rows_causal,
            {
                cute_flash.FLASH_KV_SPLIT_KEY: 2,
                cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True,
                cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY: True,
            },
        ),
        (16384, 16384, _global_tile_and_window, {}),
        (
            16384,
            16384,
            _global_tile_and_window,
            {
                cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
                cute_flash.FLASH_KV_SPLIT_KEY: 2,
            },
        ),
    ],
    ids=[
        "window",
        "split_kv",
        "ascending",
        "ragged_m",
        "dead_split",
        "dead_split_split_kv",
        "hole",
        "hole_ascending_split_kv",
    ],
)
def test_long_kv_cursor_walk(
    q_seq: int, kv_seq: int, mask_mod: Callable[..., Any], overrides: dict[str, object]
) -> None:
    """16K keys (four bitmask words): the descending and ascending cursor
    walks, the split-KV slice, a ragged query length, the dead-stage split
    (alone and under the split-KV walk) and a union with empty words
    between its runs (the hole scan, descending and ascending under the
    split-KV walk) match the eager reference; the render carries no
    per-word state."""
    q, k, v = _qkv(1, 2, q_seq, kv_seq, 64, torch.bfloat16)
    block_mask = _block_mask(mask_mod, 1, 2, q_seq, kv_seq)
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask, overrides=overrides)
    _assert_cursor_walk(code)
    _assert_list_walk(code, mask=True, score=False)
    if overrides.get(cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY):
        assert "flash_flex_split = cutlass.Int32(" in code
    if overrides.get(cute_flash.FLASH_KV_SPLIT_KEY, 1) > 1:
        # The slice's leading tiles go whole words at a time.
        assert "flash_flex_left = flash_flex_lo" in code
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
def test_long_kv_cursor_walk_two_list_rows() -> None:
    """A 256-row BlockMask at 16K: one row (two lists) per pair item, the
    cursor's union over ``[partial, full]`` only."""
    q, k, v = _qkv(1, 2, 16384, 16384, 64, torch.bfloat16)
    block_mask = _block_mask(
        _sliding_causal(1024), 1, 2, 16384, 16384, block_size=(256, 128)
    )
    code, out, lse, ref_out, ref_lse = _run(q, k, v, block_mask)
    _assert_cursor_walk(code)
    assert "flash_flex_rem = flash_flex_wp0 | flash_flex_wf0\n" in code
    assert "flash_flex_wp1" not in code
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "overrides",
    [
        {
            cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: False,
            cute_flash.FLASH_FLEX_COALESCE_KEY: 128,
        },
        {
            cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: False,
            cute_flash.FLASH_FLEX_PREFETCH_KEY: 32,
        },
    ],
    ids=["coalesce", "prefetch"],
)
def test_long_kv_cursor_peek(overrides: dict[str, object]) -> None:
    """A per-key table score_mod at 16K whose hook issues the next tile's
    table rows ahead: the peek scans the cursor's words on temporaries."""
    q, k, v = _qkv(1, 2, 16384, 16384, 64, torch.bfloat16)
    table = torch.randn(16384, dtype=torch.bfloat16, device=DEVICE)
    block_mask = _block_mask(_sliding_causal(1024), 1, 2, 16384, 16384)
    code, out, lse, ref_out, ref_lse = _run(
        q, k, v, block_mask, mods=_column_table_mods(table), overrides=overrides
    )
    _assert_cursor_walk(code)
    assert "flash_flex_pkw = flash_flex_cw" in code
    assert "flash_kv_peek = cutlass.Int32(cutlass.select_(flash_flex_prem != 0" in code
    _assert_close(out, lse, ref_out, ref_lse)


@onlyBackends(["cute"])
@pytest.mark.parametrize(("seq", "words"), [(2048, 1), (8320, 3)])
def test_item_words_build_is_cuda_graph_capturable(seq: int, words: int) -> None:
    """The host words table is rebuilt inside every captured launch (no
    cache under capture): its build must issue device kernels only (no
    host constant copied to the device); the replay reproduces the eager
    result bitwise, with the register walk (one word) and the cursor."""
    q, k, v = _qkv(1, 2, seq, seq, 64, torch.bfloat16)
    block_mask = _block_mask(_sliding_causal(1024), 1, 2, seq, seq)
    args = _kernel_args(
        q, k, v, _identity, block_mask, scale=1.0 / math.sqrt(64), enable_gqa=False
    )
    bound = _flex_kernel().bind(args)
    config = helion.Config(
        **{**bound.config_spec.default_config().config, **FLEX_CONFIG}
    )
    bound.set_config(config)
    code = bound.to_code(config)
    assert "_helion_flash_flex_item_words(" in code
    assert ("flash_flex_total = " in code) == (words > 1)
    eager_out, eager_lse = bound(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = bound(*args)
    for _ in range(2):
        out.fill_(0)
        lse.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager_out)
        assert torch.equal(lse, eager_lse)


@onlyBackends(["cute"])
def test_register_and_cursor_walks_agree_at_two_words() -> None:
    """8K keys (two words): the device-built register masks
    (``cute_flash_flex_item_words=False``, still searched there) and the
    cursor walk over the host words give the same bits."""
    q, k, v = _qkv(1, 2, 8192, 8192, 64, torch.bfloat16)
    block_mask = _block_mask(_sliding_causal(2048), 1, 2, 8192, 8192)
    code_r, out_r, lse_r, ref_out, ref_lse = _run(
        q, k, v, block_mask, overrides={cute_flash.FLASH_FLEX_ITEM_WORDS_KEY: False}
    )
    assert "flash_flex_rem1" in code_r and "flash_flex_cw" not in code_r
    code_c, out_c, lse_c, _, _ = _run(q, k, v, block_mask)
    _assert_cursor_walk(code_c)
    assert torch.equal(out_c, out_r)
    assert torch.equal(lse_c, lse_r)
    _assert_close(out_c, lse_c, ref_out, ref_lse)
