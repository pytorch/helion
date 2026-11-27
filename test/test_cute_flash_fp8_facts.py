"""Operand facts of the fused flash matcher (CPU).

The attention matcher (``attention_match._attention_softmax_pattern_head_dim``)
reads
two source forms: the 3-D ``bmm`` / ``baddbmm`` pair of ``examples/attention.py``
and the 2-D ``hl.dot`` pair over a scalar batch-head index (the form
``examples/fp8_attention.py`` takes once its query-tile loop is a launch
axis). It records the softmax scale from either Q or the scores, V's index
form (``AttentionScorePlan.v_k_major``) and fp8 operands; the output plan
reads V's storage from its strides and the 4-D ``out[bh // H, bh % H, tile_m,
:]`` store. The fa4 pair body hosts every fact; fp8 runs on it where it fits
(an even KV tile count, a value width of at most 128, a score plan the pair
walks) and on the single-tile family elsewhere
(``cute_flash.flash_fp8_pipeline_family``), gated by
``cute_flash.flash_operand_facts_lowered`` for the search surface and the
codegen detector together. Every test here binds on the CPU
and never compiles or launches CuTe (``test_cute_flash_fp8_body.py`` renders
and runs the bodies).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute.analysis import attention_match as _attention_match
from helion._compiler.cute.attention_plan import CAUSAL_MASK_KIND
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._compiler.cute.autotune.flash_surface import enable_cute_flash_search
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import skipUnlessBackends
from helion._testing.cute_aux import as_sm100
from helion._testing.cute_flash_kernels import dot_attention_k_major
from helion._testing.cute_flash_kernels import dot_attention_rows
from helion._testing.cute_flash_kernels import fp8_dot_attention_k_major
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

LOG2E = math.log2(math.e)


# ---------------------------------------------------------------------------
# The 2-D ``hl.dot`` attention forms (CPU-bound only; never launched).
# ---------------------------------------------------------------------------
@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_wrong_scale(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_attention_k_major`` with a score scale that is not the softmax
    scale of the head dim: not an attention body."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            k_tile = k[bh.begin, tile_n, :]
            qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * 0.5
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, None]
            v_tile = v[bh.begin, :, tile_n]
            acc = hl.dot(p.to(v.dtype), v_tile.t(), acc=acc)
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_grid(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """The example's own loop shape (``hl.grid`` + a serial query-tile loop):
    the KV body matches, the 1-axis root gives it no flash surface."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    sm_scale = 0.18033688
    for bh in hl.grid(batch_heads):
        for tile_m in hl.tile(seq_len):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q_tile = q[bh, tile_m, :]
            for tile_n in hl.tile(seq_len):
                k_tile = k[bh, tile_n, :]
                qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * sm_scale
                m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
                p = torch.exp2(qk - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, dim=-1)
                acc = acc * alpha[:, None]
                v_tile = v[bh, :, tile_n]
                acc = hl.dot(p.to(v.dtype), v_tile.t(), acc=acc)
                m_i = m_new
            out[bh, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_nonunit_batch_tile(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_attention_rows`` over an autotuned batch tile: ``bh.begin`` is
    the tile's first row only, not a scalar batch index."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    sm_scale = 0.18033688
    for bh, tile_m in hl.tile([batch_heads, seq_len]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            k_tile = k[bh.begin, tile_n, :]
            qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * sm_scale
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, None]
            v_tile = v[bh.begin, tile_n, :]
            acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_grid_step2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_attention_grid`` over every other batch head: a grid axis of
    step 2 skips batch heads, which the fused body cannot do."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    sm_scale = 0.18033688
    for bh in hl.grid(batch_heads, step=2):
        for tile_m in hl.tile(seq_len):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q_tile = q[bh, tile_m, :]
            for tile_n in hl.tile(seq_len):
                k_tile = k[bh, tile_n, :]
                qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * sm_scale
                m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
                p = torch.exp2(qk - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, dim=-1)
                acc = acc * alpha[:, None]
                v_tile = v[bh, tile_n, :]
                acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
                m_i = m_new
            out[bh, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_kv_batch_inner_loop(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """K/V read at the scalar index of an inner unit loop over the K/V batch
    heads: a unit block, but not the root batch block the Q tile is read at
    (every query head attends to every K/V head)."""
    batch_heads = q.size(0)
    kv_batch_heads = k.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty(
        [batch_heads, kv_batch_heads, seq_len, head_dim],
        dtype=q.dtype,
        device=q.device,
    )
    sm_scale = 0.18033688
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        q_tile = q[bh.begin, tile_m, :]
        for kv_bh in hl.tile(kv_batch_heads, block_size=1):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            for tile_n in hl.tile(seq_len):
                k_tile = k[kv_bh.begin, tile_n, :]
                qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * sm_scale
                m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
                p = torch.exp2(qk - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, dim=-1)
                acc = acc * alpha[:, None]
                v_tile = v[kv_bh.begin, tile_n, :]
                acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
                m_i = m_new
            out[bh.begin, kv_bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def bmm_attention_post_dot_scale(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """The 3-D ``bmm`` form of ``examples/attention.py`` with the softmax
    scale on the scores instead of on Q."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty_like(q)
    sm_scale = 0.18033688
    for tile_b, tile_m in hl.tile([batch_heads, seq_len]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_b, tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q_tile = q[tile_b, tile_m, :]
        for tile_n in hl.tile(seq_len):
            k_tile = k[tile_b, tile_n, :]
            qk = torch.bmm(q_tile, k_tile.transpose(1, 2), torch.float32) * sm_scale
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, :, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, :, None]
            v_tile = v[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(v.dtype), v_tile)
            m_i = m_new
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_from(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    qk_scale: float,
    m_start: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The flattened-rows form of ``examples/blackwell_attention.py`` with
    the scale on the scores and a running max that starts at ``m_start``. A
    compile-time constant (``hl.constexpr``) traces to the literal a kernel
    spells, for the start and for the scale alike; a ``float`` scale is the
    runtime scale."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(MM):
        m_i = hl.full([tile_m], m_start, dtype=torch.float32)
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32) * qk_scale
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _bind(kernel: Kernel, args: tuple[Any, ...]) -> BoundKernel:
    with as_sm100():
        return kernel._bind_isolated(args)


def _qkv(
    dtype: torch.dtype, *, bh: int = 4, seq: int = 256, head_dim: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(bh, seq, head_dim).to(dtype)
    k = torch.randn(bh, seq, head_dim).to(dtype)
    v = torch.randn(bh, seq, head_dim).to(dtype)
    return q, k, v


def _kv_loop_graph(bound: BoundKernel) -> ForLoopGraphInfo:
    """The one device loop whose body holds the two matmuls (the KV loop)."""
    device_ir = bound.host_function.device_ir
    root = {bid for ids in device_ir.grid_block_ids for bid in ids}
    graphs = [
        graph_info
        for graph_info in device_ir.graphs
        if isinstance(graph_info, ForLoopGraphInfo)
        and len(graph_info.block_ids) == 1
        and not (set(graph_info.block_ids) & root)
        and any(
            _attention_match._attention_matmul_form(node) is not None
            for node in graph_info.graph.nodes
        )
    ]
    assert len(graphs) == 1, [type(g).__name__ for g in device_ir.graphs]
    return graphs[0]


def _pattern(bound: BoundKernel) -> _attention_match.AttentionSoftmaxPattern | None:
    graph_info = _kv_loop_graph(bound)
    with bound.env, bound.host_function:
        return _attention_match._attention_softmax_pattern_head_dim(
            graph_info.graph, kv_block_id=graph_info.block_ids[0]
        )


def _surface(bound: BoundKernel) -> _attention_match.FlashSearchSurface | None:
    device_ir = bound.host_function.device_ir
    with as_sm100(), bound.env, bound.host_function:
        return _attention_match.detect_flash_search_surface(device_ir)


def _standard_example() -> Kernel:
    module = import_path(EXAMPLES_DIR / "attention.py")
    kernel = module.attention
    return Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute"))


# ---------------------------------------------------------------------------
# Matcher facts.
# ---------------------------------------------------------------------------
def test_fp8_dot_form_facts() -> None:
    """The promoted fp8 example: 2-D ``hl.dot`` scores, the scale on the
    scores, a pre-transposed V, fp8 operands and a 4-D fp8 output."""
    q, k, v = _qkv(torch.float8_e4m3fn)
    v = v.transpose(1, 2).contiguous()
    bound = _bind(fp8_dot_attention_k_major, (q, k, v, 1, 4))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.io_dtype is torch.float8_e4m3fn
    assert pattern.v_k_major
    plan = pattern.score_plan
    assert (plan.head_dim, plan.head_dim_v, plan.kv_group) == (64, None, 1)
    assert plan.modifier_kinds == ("dense",)
    assert math.isclose(plan.qk_scale_log2, LOG2E / 8, rel_tol=1e-6)
    assert plan.lse_scale == 1.0
    assert not pattern.is_causal


def test_bf16_dot_form_k_major_v() -> None:
    q, k, v = _qkv(torch.bfloat16)
    v = v.transpose(1, 2).contiguous()
    pattern = _pattern(_bind(dot_attention_k_major, (q, k, v)))
    assert pattern is not None
    assert pattern.io_dtype is torch.bfloat16
    assert pattern.v_k_major
    assert pattern.head_dim == 64


def test_checked_attention_rows_keeps_the_ordinary_fragment_default() -> None:
    args = _qkv(torch.bfloat16)
    with as_sm100():
        plain = _bind(dot_attention_rows, args)
        checked = _bind(dot_attention_rows, (*args, True))
        assert plain.config_spec.cute_flash_search_enabled
        assert plain.config_spec.autotune_seed_configs()
        spec = checked.config_spec
        assert not spec.cute_flash_search_enabled
        assert not spec.cute_flash_gated_search_enabled
        assert not spec.tcgen05.search_enabled
        errors: list[str] = []
        seeds = spec.create_config_generation().seed_flat_config_pairs(errors.append)
        assert not errors
        assert not spec.autotune_seed_configs()
        assert seeds
        # Render a real compiler seed, after filling missing values as the
        # tuner does, as well as the default used without a tuning run.
        for config in (spec.default_config(), seeds[0][1]):
            assert (
                checked.to_code(config, output_origin_lines=False).count(
                    "call __assertfail"
                )
                == 1
            )


def test_bf16_dot_form_rows_v_and_scaled_q() -> None:
    """The 2-D form with the canonical V tile and the scale on Q records the
    same plan as the 3-D example."""
    q, k, v = _qkv(torch.bfloat16)
    pattern = _pattern(_bind(dot_attention_rows, (q, k, v)))
    assert pattern is not None
    assert pattern.io_dtype is torch.bfloat16
    assert not pattern.v_k_major
    assert math.isclose(pattern.score_plan.qk_scale_log2, LOG2E / 8, rel_tol=1e-6)


def test_fp16_dot_form_matches_fp8_form_plan() -> None:
    """The same body with fp16 operands differs from the fp8 one only in the
    operand dtype."""
    q, k, v = _qkv(torch.float16)
    v = v.transpose(1, 2).contiguous()
    fp16_pattern = _pattern(_bind(dot_attention_k_major, (q, k, v)))
    q8, k8, v8 = (t.to(torch.float8_e4m3fn) for t in (q, k, v))
    fp8_pattern = _pattern(_bind(fp8_dot_attention_k_major, (q8, k8, v8, 1, 4)))
    assert fp16_pattern is not None and fp8_pattern is not None
    assert fp16_pattern.score_plan == fp8_pattern.score_plan
    assert (fp16_pattern.io_dtype, fp8_pattern.io_dtype) == (
        torch.float16,
        torch.float8_e4m3fn,
    )


def test_wrong_post_dot_scale_declines() -> None:
    q, k, v = _qkv(torch.bfloat16)
    v = v.transpose(1, 2).contiguous()
    assert _pattern(_bind(dot_attention_wrong_scale, (q, k, v))) is None


def test_grid_form_promotes_to_a_flash_root() -> None:
    """The example's own ``hl.grid`` + serial query-tile shape: the query
    tile loop partitions a fresh output, so ``promote_output_axis`` makes it a
    launch axis (a 2-axis root whose batch block is the unit grid axis), the KV
    body is the attention body with the pre-transposed V, and the surface is
    built (the fused bodies host the 2-D root and either V layout)."""
    q, k, v = _qkv(torch.bfloat16)
    v = v.transpose(1, 2).contiguous()
    bound = _bind(dot_attention_grid, (q, k, v))
    assert bound.host_function.device_ir.grid_block_ids == [[0, 1]]
    pattern = _pattern(bound)
    assert pattern is not None and pattern.v_k_major
    assert pattern.batch_block_id == 0
    surface = _surface(bound)
    assert surface is not None
    assert surface.io_dtype is torch.bfloat16
    # The unit grid axis has no searchable fragment: only the query and KV
    # tiles are targeted (as the flex surface leaves out its b/h axes).
    assert set(surface.block_size_targets.values()) == {128}
    assert 0 not in surface.block_size_targets


def test_standard_example_facts_unchanged() -> None:
    """``examples/attention.py``: the 3-D form keeps its plan and surface."""
    args = tuple(torch.randn(2, 8, 1024, 64, dtype=torch.float16) for _ in range(3))
    bound = _bind(_standard_example(), args)
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.io_dtype is torch.float16
    assert not pattern.v_k_major
    assert pattern.score_plan.modifier_kinds == ("dense",)
    surface = _surface(bound)
    assert surface is not None
    assert surface.io_dtype is torch.float16
    assert surface.head_dim == 64


# ---------------------------------------------------------------------------
# The operand-facts gate: 16-bit operands in either V index form and root
# form; fp8 at a static KV tile count and a width its MMA tiles, on the
# family the geometry picks.
# ---------------------------------------------------------------------------
def test_operand_facts_gate() -> None:
    lowered = cute_flash.flash_operand_facts_lowered
    assert lowered(torch.float16)
    assert lowered(torch.bfloat16)
    assert lowered(torch.float16, v_k_major=True)
    assert lowered(torch.bfloat16, v_k_major=True, score_rank=2)
    assert lowered(torch.float16, score_rank=2)
    # fp8 has one family per geometry, so the KV tile count must be known,
    # and its QK MMA steps 32 elements, so the Q/K width must be a multiple
    # of 32; the V index form does not matter (the storage decides the MMA's
    # major mode, both are hosted).
    fp8 = torch.float8_e4m3fn
    assert not lowered(fp8)
    assert not lowered(fp8, v_k_major=True)
    assert not lowered(fp8, num_kv=2)
    assert not lowered(fp8, head_dim=64)
    assert lowered(fp8, v_k_major=True, num_kv=1, head_dim=64)
    assert lowered(fp8, v_k_major=True, num_kv=3, head_dim=64)
    assert lowered(fp8, v_k_major=True, num_kv=2, head_dim=128)
    assert lowered(fp8, v_k_major=False, num_kv=8, score_rank=2, head_dim=32)
    assert cute_flash.FLASH_FP8_MMA_K == 32
    for width in (16, 48, 80, 112, 144):
        # A width the 16-bit MMAs' 16-element step tiles and the fp8 MMA's
        # does not: fp8 keeps the generic lowering at either tile parity.
        assert lowered(torch.bfloat16, num_kv=1, head_dim=width)
        assert not lowered(fp8, num_kv=1, head_dim=width)
        assert not lowered(fp8, num_kv=2, head_dim=width)
    # The fa4 pair where it fits (an even tile count for its two query
    # tiles, a value width of at most 128 for its two accumulators), else
    # the single-tile family.
    family = cute_flash.flash_fp8_pipeline_family
    assert family(2, 64) == "fa4"
    assert family(1, 64) == "ws_overlap"
    assert family(3, 128) == "ws_overlap"
    assert family(2, 192) == "ws_overlap"
    assert family(4, 256) == "ws_overlap"
    assert family(2, 192, 128) == "fa4"
    # A plan bound to the single-tile body (more than one mask modifier,
    # which leaves its per-tile predicate) takes it at either parity.
    assert family(2, 64, requires_ws_overlap=True) == "ws_overlap"
    assert family(3, 64, requires_ws_overlap=True) == "ws_overlap"
    assert family(2, 192, 128, requires_ws_overlap=True) == "ws_overlap"
    assert not lowered(torch.float32, num_kv=2, head_dim=64)
    assert torch.float8_e4m3fn in cute_flash.FLASH_MATCHED_IO_DTYPES
    assert cute_flash._flash_io_dtype_str(torch.float8_e4m3fn) == "cutlass.Float8E4M3FN"
    assert cute_flash._flash_dtype_bytes("cutlass.Float8E4M3FN") == 1
    assert cute_flash._flash_dtype_bytes("cutlass.Float16") == 2
    assert cute_flash._flash_dtype_bits("cutlass.Float8E4M3FN") == 8


def test_fp8_emitter_facts() -> None:
    """The per-dtype text of the fa4 body: the PTX MMA kind arguments, the
    TMEM P-store repetition in 32-bit words and the output store packet."""
    assert cute_flash._flash_mma_kind_args("cutlass.Float16") == ("", "")
    assert cute_flash._flash_mma_kind_args("cutlass.BFloat16") == ("", "")
    assert cute_flash._flash_mma_kind_args("cutlass.Float8E4M3FN") == (
        ', mma_kind="f8f6f4"',
        ', mma_kind="f8f6f4", a_width=8',
    )
    rep = cute_flash._flash_p_store_tmem_repetition
    assert (rep(16, "cutlass.Float16"), rep(32, "cutlass.BFloat16")) == (16, 32)
    assert (rep(16, "cutlass.Float8E4M3FN"), rep(32, "cutlass.Float8E4M3FN")) == (
        8,
        16,
    )
    lines = cute_flash._flash_o_store_thread_layout_lines(
        64, 32, dim_name="d", elems_name="e", elems_per_packet=16
    )
    # 64 fp8 columns are four 16-byte packets: 8 rows x 4 packets per pass.
    assert "d = 64 // e" in lines and "(32 // d, d)" in lines
    lines16 = cute_flash._flash_o_store_thread_layout_lines(
        64, 32, dim_name="d", elems_name="e"
    )
    assert "d = 64 // e" in lines16


@pytest.mark.parametrize(
    ("kernel", "dtype", "extra_args", "k_major"),
    [
        (fp8_dot_attention_k_major, torch.float8_e4m3fn, (1, 4), True),
        (dot_attention_k_major, torch.bfloat16, (), True),
        (dot_attention_rows, torch.bfloat16, (), False),
    ],
    ids=["fp8_k_major", "bf16_k_major", "bf16_rows_2d_root"],
)
def test_hosted_facts_build_a_surface(
    kernel: Kernel, dtype: torch.dtype, extra_args: tuple[int, ...], k_major: bool
) -> None:
    """A 2-axis root whose body matches with fp8 operands, a pre-transposed V
    or the 2-D ``hl.dot`` root has a fused body: the search surface is built
    over the query and KV tiles alone (the unit batch block has no fragment),
    the spec enables the flash search and the seeds pin ``[128, 128]``."""
    q, k, v = _qkv(dtype)
    if k_major:
        v = v.transpose(1, 2).contiguous()
    bound = _bind(kernel, (q, k, v, *extra_args))
    root = bound.host_function.device_ir.grid_block_ids[0]
    assert len(root) == 2
    assert _pattern(bound) is not None
    surface = _surface(bound)
    assert surface is not None
    assert surface.io_dtype is dtype
    assert surface.block_size_targets == {root[1]: 128, surface_kv_block(bound): 128}
    assert bound.config_spec.cute_flash_search_enabled
    config = bound.config_spec.default_config()
    assert config.config["block_sizes"] == [128, 128]
    assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"


def surface_kv_block(bound: BoundKernel) -> int:
    return _kv_loop_graph(bound).block_ids[0]


def test_fp8_keeps_the_128_row_query_tile() -> None:
    """The 64-row query tile's P store hands each thread two rows of 16-bit
    elements: on the body where 16-bit operands take the tile, an explicit
    request resolves to the 128-row tile for fp8."""
    request: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
        cute_flash.FLASH_Q_TILE_M_KEY: 64,
        cute_flash.FLASH_S_STAGE_KEY: 2,
        cute_flash.FLASH_PERSISTENT_KEY: False,
        cute_flash.FLASH_EPI_STG_KEY: True,
    }

    def q_tile_m(dtype: torch.dtype) -> int:
        return cute_flash.resolve_flash_config(
            64,
            3,
            request,
            dtype=dtype,
            num_bh=8,
            standard_dense_output=True,
            plain_row_body=True,
        ).q_tile_m

    assert q_tile_m(torch.bfloat16) == 64
    assert q_tile_m(torch.float8_e4m3fn) == 128


@pytest.mark.parametrize("seq", [128, 256])
def test_fp8_width_the_mma_does_not_tile_keeps_the_generic_lowering(seq: int) -> None:
    """A Q/K width of 48 is three steps of the 16-bit MMAs and one and a half
    of the fp8 QK MMA: the example still matches as an attention body, and
    the gate leaves it to the generic lowering at either tile parity."""
    module = import_path(EXAMPLES_DIR / "fp8_attention.py")
    kernel = Kernel(
        module.fp8_attention_kernel.fn,
        settings=module.fp8_attention_kernel.settings.copy(backend="cute"),
    )
    q, k, v = (torch.randn(1, 4, seq, 48, dtype=torch.float16) for _ in range(3))
    q8, k8, v8 = module.preprocess_fp8_attention_inputs(q, k, v)
    bound = _bind(kernel, (q8, k8, v8, 1, 4))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.score_plan.head_dim == 48
    assert _surface(bound) is None
    assert not bound.config_spec.cute_flash_search_enabled


@pytest.mark.parametrize("seq", [128, 384])
def test_fp8_odd_kv_tile_count_takes_the_single_tile_family(seq: int) -> None:
    """The fa4 pair needs an even count of 128-key tiles; at an odd count fp8
    operands take the single-tile family, as the 16-bit form of the same
    shape does: the surface is built and its default config resolves to that
    family over ``[128, 128]`` tiles."""
    q, k, v = _qkv(torch.float8_e4m3fn, seq=seq)
    v = v.transpose(1, 2).contiguous()
    bound = _bind(fp8_dot_attention_k_major, (q, k, v, 1, 4))
    assert _pattern(bound) is not None
    surface = _surface(bound)
    assert surface is not None
    assert surface.io_dtype is torch.float8_e4m3fn
    assert surface.num_kv == seq // 128
    assert bound.config_spec.cute_flash_search_enabled
    config = bound.config_spec.default_config()
    assert config.config["block_sizes"] == [128, 128]
    assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
    q, k, v = _qkv(torch.bfloat16, seq=seq)
    v = v.transpose(1, 2).contiguous()
    assert _surface(_bind(dot_attention_k_major, (q, k, v))) is not None


@pytest.mark.parametrize(
    "scale",
    [0.125, hl.constexpr(0.125)],
    ids=["runtime_scale", "literal_scale"],
)
def test_fp8_sentinel_start_keeps_the_generic_lowering_on_the_single_tile_family(
    scale: object,
) -> None:
    """The single-tile family takes fp8 operands from a running max that
    starts at ``-inf``. A finite sentinel (``-1e30``) keeps the generic
    lowering there, on the search surface and in codegen together: the fused
    bodies start every row from ``-inf``, and a runtime scale puts the scores
    of fp8 operands below any sentinel, where the kernel keeps its start
    (``KNOWN_GAPS.md``, cute-flash-sentinel-start-taken-for-minus-inf). The
    fa4 pair, which an even count takes, keeps the sentinel start it took,
    and 16-bit operands keep it on every family."""

    def bind(
        dtype: torch.dtype,
        seq: int,
        m_start: float,
        *,
        batch: int = 1,
        device: str = "cpu",
    ) -> BoundKernel:
        q, k, v = (
            torch.empty(batch, 4, seq, 64, dtype=dtype, device=device) for _ in range(3)
        )
        return _bind(flattened_attention_from, (q, k, v, scale, hl.constexpr(m_start)))

    def fused(bound: BoundKernel) -> bool:
        """Whether codegen takes the fused body at the block sizes it fuses."""
        with as_sm100():
            code = bound.to_code(helion.Config(block_sizes=[128, 128]))
        return "'kind': 'helion_flash'" in code

    fp8 = torch.float8_e4m3fn
    for seq in (128, 384):
        assert _surface(bind(fp8, seq, -math.inf)) is not None
        bound = bind(fp8, seq, -1e30)
        assert _pattern(bound) is not None
        assert _surface(bound) is None
        assert not bound.config_spec.cute_flash_search_enabled
        assert _surface(bind(torch.bfloat16, seq, -1e30)) is not None
    # The codegen detector declines on its own: the same config fuses the
    # ``-inf`` start and not the sentinel.
    assert fused(bind(fp8, 128, -math.inf))
    assert not fused(bind(fp8, 128, -1e30))
    surface = _surface(bind(fp8, 256, -1e30))
    assert surface is not None
    assert surface.io_dtype is fp8
    # An output of more than 2**31 elements on the single-tile family's
    # geometries is handed to the generic search with its block-size bounds
    # first, whatever the start.
    for m_start in (-math.inf, -1e30):
        large = bind(
            fp8,
            384,
            m_start,
            batch=32768,
            device="meta",  # @ignore-device-lint
        )
        spec = large.config_spec
        assert spec.cute_attention_generic_fallback_enabled
        assert spec.default_config().config["block_sizes"] == [64, 64]


def test_rows_v_2d_form_records_its_rank() -> None:
    """The 2-D form records ``score_rank == 2``; the emitters index programs
    by root block, so the gate hosts it like the 3-D form."""
    q, k, v = _qkv(torch.bfloat16)
    bound = _bind(dot_attention_rows, (q, k, v))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.score_rank == 2
    assert cute_flash.flash_operand_facts_lowered(
        pattern.io_dtype, v_k_major=pattern.v_k_major, score_rank=pattern.score_rank
    )


# ---------------------------------------------------------------------------
# Output-plan facts (``cute_flash._flash_graph_output_plan_from_graphs``): the
# scalar batch root, the dot operand names and V's host layout.
# ---------------------------------------------------------------------------
def _output_plan(bound: BoundKernel) -> cute_flash.FlashGraphOutputPlan | None:
    pattern = _pattern(bound)
    assert pattern is not None
    graph_info = _kv_loop_graph(bound)
    device_ir = bound.host_function.device_ir
    with bound.env, bound.host_function:
        return cute_flash._flash_graph_output_plan_from_graphs(
            device_ir.graphs,
            root_block_ids=device_ir.grid_block_ids[0],
            kv_block_id=graph_info.block_ids[0],
            score_plan=pattern.score_plan,
        )


def _operand_names(bound: BoundKernel) -> cute_flash.FlashOperandNames | None:
    graph_info = _kv_loop_graph(bound)
    device_ir = bound.host_function.device_ir
    with bound.env, bound.host_function:
        return cute_flash._flash_attention_operand_names(
            device_ir.graphs,
            root_block_ids=device_ir.grid_block_ids[0],
            kv_block_id=graph_info.block_ids[0],
        )


def test_output_plan_2d_rows_form() -> None:
    """``q[bh.begin, tile_m, :]`` is a root Q load; the dot operands name K
    and V; the 3-D ``out[bh.begin, tile_m, :]`` store is the output."""
    q, k, v = _qkv(torch.bfloat16)
    bound = _bind(dot_attention_rows, (q, k, v))
    plan = _output_plan(bound)
    assert plan is not None
    assert (plan.q_name, plan.k_name, plan.v_name, plan.o_name) == (
        "q",
        "k",
        "v",
        "out",
    )
    assert (plan.batch, plan.seq, plan.head_dim, plan.head_dim_v) == (4, 256, 64, None)
    assert plan.dtype is torch.bfloat16
    assert not plan.v_k_major
    assert plan.lse_name is None
    assert plan.output_epilogue == "identity"


def test_output_plan_2d_k_major_v() -> None:
    """The pre-transposed ``v[bh, :, tile_n]`` names V with its
    ``[kv_batch, head_dim, seq]`` host shape."""
    q, k, v = _qkv(torch.bfloat16)
    v = v.transpose(1, 2).contiguous()
    bound = _bind(dot_attention_k_major, (q, k, v))
    names = _operand_names(bound)
    assert names == cute_flash.FlashOperandNames("q", "k", "v", 1, v_k_major=True)
    plan = _output_plan(bound)
    assert plan is not None
    assert plan.v_k_major
    assert plan.dtype is torch.bfloat16
    assert (plan.batch, plan.seq, plan.head_dim) == (4, 256, 64)


def test_output_plan_fp8_4d_output_split() -> None:
    """The fp8 example's 4-D ``out[bh // heads, bh % heads, tile_m, :]``
    store is the collapsed output's flat row block ``bh``: the plan records
    the split (``o_batch_heads_split``), takes the batch-head count from Q
    and the tile geometry from the output's last dims, and reads V's storage
    from its strides (a contiguous ``[bh, d, s]`` buffer is K-major)."""
    q, k, v = _qkv(torch.float8_e4m3fn)
    v = v.transpose(1, 2).contiguous()
    bound = _bind(fp8_dot_attention_k_major, (q, k, v, 1, 4))
    names = _operand_names(bound)
    assert names == cute_flash.FlashOperandNames("q", "k", "v", 1, v_k_major=True)
    plan = _output_plan(bound)
    assert plan is not None
    assert (plan.q_name, plan.k_name, plan.v_name, plan.o_name) == (
        "q",
        "k",
        "v",
        "out",
    )
    assert plan.o_batch_heads_split
    assert (plan.batch, plan.seq, plan.head_dim, plan.head_dim_v) == (4, 256, 64, None)
    assert plan.dtype is torch.float8_e4m3fn
    assert plan.v_k_major and plan.v_storage_k_major
    assert plan.output_epilogue == "identity"


def test_output_plan_v_storage_follows_the_strides() -> None:
    """``examples/fp8_attention.py`` hands the kernel ``v.permute(...).to(fp8)
    .reshape(...)``: a view of the canonical ``[bh, s, d]`` buffer, read as
    ``v[bh, :, tile_n]``. The index form is K-major, the storage MN-major: the
    plan records both, so the host builds the MN-major PV operand the buffer
    needs. The dense transposed view of a ``[bh, d, s]`` buffer read in the
    canonical form is the mirror image."""
    q, k, v = _qkv(torch.float8_e4m3fn)
    view = v.transpose(1, 2)
    assert not view.is_contiguous()
    bound = _bind(fp8_dot_attention_k_major, (q, k, view, 1, 4))
    plan = _output_plan(bound)
    assert plan is not None
    assert plan.v_k_major and not plan.v_storage_k_major
    assert (plan.batch, plan.seq, plan.head_dim) == (4, 256, 64)
    q, k, v = _qkv(torch.bfloat16)
    k_major_buffer = v.transpose(1, 2).contiguous()
    rows_view = k_major_buffer.transpose(1, 2)
    plan = _output_plan(_bind(dot_attention_rows, (q, k, rows_view)))
    assert plan is not None
    assert not plan.v_k_major and plan.v_storage_k_major
    assert _surface(_bind(dot_attention_rows, (q, k, rows_view))) is not None


def test_v_storage_facts() -> None:
    """The storage helpers on the four dense layouts and a strided one."""
    q, k, v = _qkv(torch.bfloat16)
    bound = _bind(dot_attention_rows, (q, k, v))
    with bound.env:
        contiguous_dim = cute_flash._flash_3d_contiguous_dim
        storage = cute_flash._flash_v_storage_k_major
        canonical = torch.empty(4, 256, 64)
        transposed = canonical.transpose(1, 2)
        assert contiguous_dim(canonical) == 2
        assert contiguous_dim(transposed) == 1
        assert contiguous_dim(canonical[:, ::2]) is None
        assert contiguous_dim(torch.empty(4, 512, 64)[:, :256]) is None
        assert storage(canonical, v_k_major=False) is False
        assert storage(transposed, v_k_major=True) is False
        assert storage(transposed.contiguous(), v_k_major=True) is True
        assert storage(transposed.contiguous().transpose(1, 2), v_k_major=False) is (
            True
        )
        assert storage(canonical[:, ::2], v_k_major=False) is None


def test_example_promoted_form_surfaces() -> None:
    """``examples/fp8_attention.py`` as the campaign runs it: the serial query
    tile loop promoted to a launch axis, the ``hl.grid`` batch index, the 4-D
    output and the transposed-view V. The surface is built, the spec enables
    the flash search, and the default config is the fa4 pair body over
    ``[128, 128]`` tiles."""
    module = import_path(EXAMPLES_DIR / "fp8_attention.py")
    kernel = Kernel(
        module.fp8_attention_kernel.fn,
        settings=module.fp8_attention_kernel.settings.copy(backend="cute"),
    )
    q, k, v = (torch.randn(2, 4, 256, 64, dtype=torch.float16) for _ in range(3))
    q8, k8, v8 = module.preprocess_fp8_attention_inputs(q, k, v)
    assert not v8.is_contiguous()
    bound = _bind(kernel, (q8, k8, v8, 2, 4))
    assert bound.host_function.device_ir.grid_block_ids == [[0, 1]]
    pattern = _pattern(bound)
    assert pattern is not None
    assert (pattern.io_dtype, pattern.v_k_major, pattern.score_rank) == (
        torch.float8_e4m3fn,
        True,
        2,
    )
    surface = _surface(bound)
    assert surface is not None
    assert (surface.head_dim, surface.num_kv, surface.num_bh) == (64, 2, 8)
    plan = _output_plan(bound)
    assert plan is not None
    assert plan.o_batch_heads_split and plan.v_k_major
    assert not plan.v_storage_k_major
    assert bound.config_spec.cute_flash_search_enabled
    config = bound.config_spec.default_config()
    assert config.config["block_sizes"] == [128, 128]
    assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"


def test_output_plan_standard_example_unchanged() -> None:
    args = tuple(torch.randn(2, 8, 1024, 64, dtype=torch.float16) for _ in range(3))
    bound = _bind(_standard_example(), args)
    plan = _output_plan(bound)
    assert plan is not None
    assert plan.dtype is torch.float16
    assert not plan.v_k_major
    assert (plan.q_name, plan.o_name, plan.lse_name) == ("q_view", "out", "lse")
    assert (plan.batch, plan.seq, plan.head_dim) == (16, 1024, 64)


# ---------------------------------------------------------------------------
# The scalar batch index: a unit block, and the root's (review fixes N1/N2).
# ---------------------------------------------------------------------------
def test_scalar_batch_index_requires_a_unit_block() -> None:
    """A batch tile must be pinned to one row, and the root must visit
    consecutive batch heads for the fused kernel to implement the loop."""
    q, k, v = _qkv(torch.bfloat16)
    bound = _bind(dot_attention_nonunit_batch_tile, (q, k, v))
    assert bound.host_function.device_ir.grid_block_ids == [[0, 1]]
    assert _pattern(bound) is None
    assert _surface(bound) is None
    assert not bound.config_spec.cute_flash_search_enabled
    strided = _bind(dot_attention_grid_step2, (q, k, v))
    # Each program still reads one batch head; the grid skips every other one.
    assert _pattern(strided) is not None
    assert _surface(strided) is None
    assert not strided.config_spec.cute_flash_search_enabled


def test_scalar_batch_block_is_recorded() -> None:
    """The 2-D forms record the unit block their K/V batch scalar belongs to
    (the root batch block); the 3-D form records none."""
    q, k, v = _qkv(torch.bfloat16)
    bound = _bind(dot_attention_rows, (q, k, v))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.batch_block_id == bound.host_function.device_ir.grid_block_ids[0][0]
    v_t = v.transpose(1, 2).contiguous()
    grid_pattern = _pattern(_bind(dot_attention_grid, (q, k, v_t)))
    assert grid_pattern is not None
    assert grid_pattern.batch_block_id == 0
    args = tuple(torch.randn(2, 8, 1024, 64, dtype=torch.float16) for _ in range(3))
    standard = _pattern(_bind(_standard_example(), args))
    assert standard is not None
    assert standard.batch_block_id is None


def test_kv_scalar_batch_block_must_be_the_root_batch_block() -> None:
    """K/V read at the scalar of an inner unit loop match the body (the
    block is recorded), but the surface, the codegen detector and the output
    plan all require that block to be the root batch block the Q tile is
    read at."""
    q, k, v = _qkv(torch.bfloat16)
    bound = _bind(dot_attention_kv_batch_inner_loop, (q, k, v))
    root = bound.host_function.device_ir.grid_block_ids[0]
    assert len(root) == 2
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.score_rank == 2
    assert pattern.batch_block_id is not None
    assert pattern.batch_block_id != root[0]
    fits_root = _attention_match._attention_pattern_fits_root
    # With every operand fact hosted, the off-root block alone declines.
    with patch.object(cute_flash, "flash_operand_facts_lowered", return_value=True):
        assert not fits_root(pattern, root)
        assert fits_root(pattern._replace(batch_block_id=root[0]), root)
        assert fits_root(pattern._replace(batch_block_id=None), root)
    assert _surface(bound) is None
    assert _operand_names(bound) is None


# ---------------------------------------------------------------------------
# The 3-D ``bmm`` form with the scale on the scores: the one form the facts
# open to the fused bodies today (review fix N3; GPU numerics in
# test_cute_backend.py::test_flash_attention_post_dot_scale_fires_and_matches_sdpa).
# ---------------------------------------------------------------------------
def test_3d_post_dot_scale_form_matches_and_surfaces() -> None:
    q, k, v = _qkv(torch.float16)
    bound = _bind(bmm_attention_post_dot_scale, (q, k, v))
    pattern = _pattern(bound)
    assert pattern is not None
    assert (pattern.score_rank, pattern.batch_block_id, pattern.v_k_major) == (
        3,
        None,
        False,
    )
    assert pattern.io_dtype is torch.float16
    assert math.isclose(pattern.score_plan.qk_scale_log2, LOG2E / 8, rel_tol=1e-6)
    assert pattern.score_plan.modifier_kinds == ("dense",)
    surface = _surface(bound)
    assert surface is not None
    assert (surface.head_dim, surface.num_bh, surface.io_dtype) == (
        64,
        4,
        torch.float16,
    )
    assert bound.config_spec.cute_flash_search_enabled
    plan = _output_plan(bound)
    assert plan is not None
    assert (plan.q_name, plan.k_name, plan.v_name, plan.o_name) == (
        "q",
        "k",
        "v",
        "out",
    )


# ---------------------------------------------------------------------------
# The fp8 search surface: pinned to one family (the fa4 pair where it fits,
# else the single-tile family: an odd KV tile count, a value width above 128,
# a plan bound to the single-tile body), every searched knob value
# realized by a coverage config (the resolver pins the 16-bit-only knobs off,
# so they must not be searched: a value searched both ways but always pinned
# off fails the structural coverage design, as the role chain did).
# ---------------------------------------------------------------------------
def _flash_generation(
    head_dim: int, num_kv: int, num_bh: int, dtype: torch.dtype, **facts: Any
) -> ConfigGeneration:
    """The search over the surface of ``facts`` (the plain dense plan's when
    none are given)."""
    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in ((1, 128), (3, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    enable_cute_flash_search(
        spec,
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=num_bh,
        dtype=dtype,
        block_size_targets={1: 128, 3: 128},
        **(facts or {"standard_dense_output": True}),
    )
    spec.compiler_seed_configs = []
    return ConfigGeneration(spec)


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "num_bh", "max_kv_stage"),
    [(64, 2, 4, 26), (128, 16, 32, 12)],
    ids=["d64", "d128"],
)
def test_fp8_surface_coverage_design_is_complete(
    head_dim: int, num_kv: int, num_bh: int, max_kv_stage: int
) -> None:
    generation = _flash_generation(head_dim, num_kv, num_bh, torch.float8_e4m3fn)
    configs = generation.extension.deterministic_population_configs()
    assert configs
    assert generation.extension.flash_structural_coverage_uncovered_interactions() == []
    assert {config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] for config in configs} == {
        "fa4"
    }
    assert all(not config[cute_flash.FLASH_ROLE_CHAIN_KEY] for config in configs)
    # The alpha-pin thresholds rescale P before its 8-bit packing, which the
    # fp8 one-ulp accuracy rule rejects: only the unpinned value is searched.
    assert all(
        config[cute_flash.FLASH_RESCALE_THRESHOLD_KEY] == 0.0 for config in configs
    )
    assert cute_flash._flash_safe_rescale_threshold_values(torch.float8_e4m3fn) == (
        0.0,
    )
    assert 8.0 in cute_flash._flash_safe_rescale_threshold_values(torch.bfloat16)
    # The byte model: fp8 tiles are half the 16-bit bytes, so the ring's
    # searched depths reach past the 16-bit surface's.
    fp8_max_kv_stage = max(config[cute_flash.FLASH_KV_STAGE_KEY] for config in configs)
    assert fp8_max_kv_stage == max_kv_stage
    bf16_configs = _flash_generation(
        head_dim, num_kv, num_bh, torch.bfloat16
    ).extension.deterministic_population_configs()
    assert (
        max(config[cute_flash.FLASH_KV_STAGE_KEY] for config in bf16_configs)
        < fp8_max_kv_stage
    )


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "num_bh"),
    [(64, 1, 192), (64, 3, 4), (128, 5, 32), (192, 2, 4)],
    ids=["d64_one_tile", "d64_three_tiles", "d128_five_tiles", "d192_two_tiles"],
)
def test_fp8_single_tile_surface_coverage_design_is_complete(
    head_dim: int, num_kv: int, num_bh: int
) -> None:
    """At an odd KV tile count, or a width above the pair's (head dim 192,
    where pinning fa4 left the surface without a legal family), the fp8
    surface is the single-tile family's: its coverage design is complete and
    it searches no pinned rescale threshold and no role chain, as the pair's
    surface does not."""
    generation = _flash_generation(head_dim, num_kv, num_bh, torch.float8_e4m3fn)
    configs = generation.extension.deterministic_population_configs()
    assert configs
    assert generation.extension.flash_structural_coverage_uncovered_interactions() == []
    assert {config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] for config in configs} == {
        "ws_overlap"
    }
    assert all(not config[cute_flash.FLASH_ROLE_CHAIN_KEY] for config in configs)
    assert all(
        config[cute_flash.FLASH_RESCALE_THRESHOLD_KEY] == 0.0 for config in configs
    )


def _two_mask_facts(num_kv: int) -> dict[str, Any]:
    """The surface facts of a plan with two composed masks (the causal mask
    ahead of a sliding window, each its own ``torch.where``) at ``num_kv``
    tiles, as the detector builds them. The range helpers express one mask
    modifier as an interval of KV tiles per query tile; a plan with two is
    bound to the single-tile body's per-tile predicate whatever the masks
    compose to (``AttentionScorePlan.has_contiguous_kv_interval`` counts the
    mask modifiers)."""
    return {
        "is_causal": True,
        "has_kv_tile_pruning": True,
        "requires_ws_overlap": True,
        "plain_row_body": False,
        "has_score_modifiers": True,
        "score_modifier_kinds": (CAUSAL_MASK_KIND, SLIDING_WINDOW_MASK_KIND),
        "tmem_rowmax_compatible": False,
        # The detector's ``q_seq % 256 == 0``.
        "causal_resident_compatible": num_kv % 2 == 0,
    }


@pytest.mark.parametrize("num_kv", [2, 3], ids=["even", "odd"])
def test_fp8_plan_bound_to_the_single_tile_body_takes_it(num_kv: int) -> None:
    """A plan with two mask modifiers runs on the single-tile body's per-tile
    predicate alone (``AttentionScorePlan.requires_ws_overlap``). The fp8
    family follows the plan at an even tile count as well, where the pinned
    pair left the surface without a legal family
    (``flash_autotune_fragments`` raised ``InvalidConfig``) and the resolver
    handed the plan to the pair whatever the config asked for: the surface
    is the single-tile family's, as the 16-bit surface of the same plan is,
    and the resolver keeps a request for the pair on it.

    Only a CPU bind reaches this today: the masks match in the 3-D ``bmm``
    form alone, which does not trace with fp8 operands on a device
    (``torch.bmm`` refuses their ``out_dtype``), and the 2-D ``hl.dot`` form
    fp8 kernels are written in matches no mask.
    """
    fp8 = torch.float8_e4m3fn
    for dtype in (fp8, torch.bfloat16):
        generation = _flash_generation(64, num_kv, 4, dtype, **_two_mask_facts(num_kv))
        configs = generation.extension.deterministic_population_configs()
        assert configs
        assert {config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] for config in configs} == {
            "ws_overlap"
        }
        assert (
            generation.extension.flash_structural_coverage_uncovered_interactions()
            == []
        )
    for request in (None, "fa4", "ws_overlap"):
        resolved = cute_flash.resolve_flash_config(
            64,
            num_kv,
            None
            if request is None
            else {cute_flash.FLASH_PIPELINE_FAMILY_KEY: request},
            dtype=fp8,
            num_bh=4,
            **_two_mask_facts(num_kv),
        )
        assert resolved.topology == "ws_overlap", request
