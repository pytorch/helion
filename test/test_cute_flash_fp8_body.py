"""The fp8 (e4m3) fused flash forward body and the storage-driven V operand.

CPU render pins of the fa4 body's per-dtype text (the ``kind::f8f6f4`` MMA
arguments, the 8-bit TMEM P store repetition, the P stage stride, the V
descriptor's major mode, the byte-sized shared storage and schedule specs)
and GPU numerics of ``examples/fp8_attention.py`` (its promoted root, 4-D
output and transposed view V) and of the 2-D ``hl.dot`` forms with a dense
K-major V buffer on the pair and the alternating-warpgroup bodies.
"""

from __future__ import annotations

import math
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test._cute_aux import _cpu_codegen
from test.test_cute_flash_fp8_facts import dot_attention_k_major
from test.test_cute_flash_fp8_facts import dot_attention_rows
from test.test_cute_flash_fp8_facts import fp8_dot_attention_k_major

import helion
from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_k_major_any_head_dim(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_attention_k_major`` with the softmax scale derived from the head
    dim (the hd128 renders: the alternating-warpgroup family's head dim)."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / math.sqrt(head_dim) * 1.44269504
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
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
            v_tile = v[bh.begin, :, tile_n]
            acc = hl.dot(p.to(v.dtype), v_tile.t(), acc=acc)
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_rows_any_head_dim(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_attention_rows`` (the canonical ``v[bh, tile_n, :]``) with the
    softmax scale derived from the head dim."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / math.sqrt(head_dim) * 1.44269504
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
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


def _example_module() -> Any:
    return import_path(EXAMPLES_DIR / "fp8_attention.py")


def _example_kernel() -> Kernel:
    module = _example_module()
    return Kernel(
        module.fp8_attention_kernel.fn,
        settings=module.fp8_attention_kernel.settings.copy(backend="cute"),
    )


def _example_args(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """The example's fp16 inputs and its preprocessed fp8 kernel arguments
    (``v`` is the transposed view of the canonical buffer)."""
    torch.manual_seed(42)
    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)
        for _ in range(3)
    )
    q8, k8, v8 = _example_module().preprocess_fp8_attention_inputs(q, k, v)
    return (q, k, v), (q8, k8, v8, batch, heads)


def _render(
    kernel: Kernel, args: tuple[Any, ...], overrides: dict[str, object] | None = None
) -> str:
    """CPU render of the default flash config (plus ``overrides``)."""
    with (
        _cpu_codegen(),
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
    ):
        bound = kernel._bind_isolated(args)
        assert bound.config_spec.cute_flash_search_enabled
        config = bound.config_spec.default_config()
        if overrides:
            config = helion.Config(**{**config.config, **overrides})
        return bound.to_code(config)


# ---------------------------------------------------------------------------
# CPU render pins.
# ---------------------------------------------------------------------------
def test_fp8_example_render_pins() -> None:
    """The example at (1, 4, 256, 64): the fa4 pair body over fp8 operands."""
    _, args = _example_args(1, 4, 256, 64)
    code = _render(_example_kernel(), args)
    assert "_helion_flash_rt" in code
    # Byte-sized shared storage and TMA transaction counts.
    assert "flash_fa4_shared_storage(64, 3, 2, 2, cutlass.Float8E4M3FN" in code
    assert "cute.size_in_bytes(cutlass.Float8E4M3FN, cute.select(_flash_qsl" in code
    # The PTX MMAs: kind::f8f6f4 on every issue, an 8-bit TMEM P on the PV.
    pv_issues = code.count("_helion_flash_ptx.gemm_ptx_precomputed_pv_ts(")
    assert pv_issues > 0
    assert code.count("mma_kind='f8f6f4', a_width=8") == pv_issues
    qk_issues = code.count("_helion_flash_ptx.gemm_ptx_precomputed_qk(")
    assert qk_issues > 0
    assert len(
        re.findall(r"smem_offset=-?512, zero_init=True, mma_kind='f8f6f4'", code)
    ) == (qk_issues)
    # The P-over-S alias: 128 f32 columns of S1 are 512 fp8 units; the TMEM
    # P store repeats 8 words per 32-column score chunk (16 at 16 bits).
    assert "tOrP1 = cute.make_tensor(tOrP0.iterator + 512, tOrP0.layout)" in code
    assert (
        "flash_tilePlikeFP32 = 128 // cutlass.Float32.width * cutlass.Float8E4M3FN.width"
        in code
    )
    assert "St32x32bOp(cute_tcgen05_flash.Repetition(8))" in code
    assert "St32x32bOp(cute_tcgen05_flash.Repetition(16))" not in code
    # The example's V is the transposed view of the canonical buffer: an
    # MN-major B operand (no K-major plan key).
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.MN)" in code
    assert "'dtype': 'cutlass.Float8E4M3FN'" in code
    assert "'v_k_major'" not in code
    # The output store converts the rescaled fp32 tile to fp8.
    assert "flash_reg0.load().to(cutlass.Float8E4M3FN)" in code


def test_fp8_k_major_storage_render_pins() -> None:
    """A dense ``[bh, d, s]`` V buffer: the K-major descriptor and plan key."""
    q, k, v = (torch.randn(4, 256, 64).to(torch.float8_e4m3fn) for _ in range(3))
    v = v.transpose(1, 2).contiguous()
    code = _render(fp8_dot_attention_k_major, (q, k, v, 1, 4))
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)" in code
    assert "'v_k_major': True" in code
    assert "mma_kind='f8f6f4', a_width=8" in code


def test_fp8_staged_output_render_pins() -> None:
    """The staged output stores (TMA and vector) over fp8 tiles."""
    _, args = _example_args(1, 4, 256, 64)
    code = _render(_example_kernel(), args, {cute_flash.FLASH_EPI_STG_KEY: True})
    assert (
        "flash_o_async_copy_elems = flash_o_store_bits // cutlass.Float8E4M3FN.width"
        in code
    )
    assert "flash_o_tO_shape_dim_1 = 64 // flash_o_async_copy_elems" in code
    code = _render(_example_kernel(), args, {cute_flash.FLASH_EPI_TMA_KEY: True})
    # The host wrapper builds the fp8 O TMA store atom from the plan's dtype.
    assert "_flash_tma_o" in code
    assert "'epi_tma': True" in code and "'dtype': 'cutlass.Float8E4M3FN'" in code


def test_16bit_renders_spell_no_fp8_text() -> None:
    """The 16-bit bodies keep the helpers' defaults: no kind or width
    argument, the 16-word P store, the 256-unit P stage stride; a dense
    K-major V buffer takes the K-major descriptor, the canonical V the
    MN-major one."""
    q, k, v = (torch.randn(4, 256, 64, dtype=torch.bfloat16) for _ in range(3))
    code = _render(dot_attention_k_major, (q, k, v.transpose(1, 2).contiguous()))
    assert "mma_kind" not in code and "a_width" not in code
    assert "St32x32bOp(cute_tcgen05_flash.Repetition(16))" in code
    assert "tOrP1 = cute.make_tensor(tOrP0.iterator + 256, tOrP0.layout)" in code
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)" in code
    assert "'v_k_major': True" in code
    assert "'dtype': 'cutlass.BFloat16'" in code
    code = _render(dot_attention_rows, (q, k, v))
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.MN)" in code
    assert "'v_k_major'" not in code
    assert "mma_kind" not in code


_ALT_FAMILY = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_alt"}
_O_OVER_Q_KNOBS: dict[str, object] = {
    cute_flash.FLASH_O_OVER_Q_KEY: True,
    cute_flash.FLASH_EPI_TMA_KEY: True,
    cute_flash.FLASH_PERSISTENT_KEY: False,
}


def test_alt_family_v_descriptor_follows_storage_render_pins() -> None:
    """The alternating-warpgroup body (hd128) builds its V descriptor from
    V's storage like the fa4 pair body: a dense K-major buffer takes
    ``Major.K`` (the host's K-major PV MMA and V staging layout), the
    canonical buffer ``Major.MN``."""
    q, k, v = (torch.randn(4, 256, 128, dtype=torch.bfloat16) for _ in range(3))
    code = _render(
        dot_attention_k_major_any_head_dim,
        (q, k, v.transpose(1, 2).contiguous()),
        _ALT_FAMILY,
    )
    assert "flash_fa4_alt_shared_storage(128, 3, 2, cutlass.BFloat16)" in code
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)" in code
    assert "Major.MN" not in code
    assert "'v_k_major': True" in code
    code = _render(dot_attention_rows_any_head_dim, (q, k, v), _ALT_FAMILY)
    assert "flash_fa4_alt_shared_storage(128, 3, 2, cutlass.BFloat16)" in code
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.MN)" in code
    assert "'v_k_major'" not in code


def test_fp8_verified_schedule_sizes_tiles_by_element_render_pins() -> None:
    """The fa4 body's verified schedule specs (its o_over_q, asymmetric,
    2-CTA and separate-ring branches) size Q, the K/V ring and the staged
    output by the operand element: the fp8 example's deepest o_over_q ring
    (12 slots at hd128, the resolver's fp8 cap) renders at the capacity
    exactly and the storage-size assert carries the byte-sized total (the
    2-byte model raised ``schedule exceeds shared-memory capacity`` at 12
    and asserted twice the struct's bytes at 3); the 16-bit total is the
    same as before."""
    _, args = _example_args(1, 2, 256, 128)
    code = _render(
        _example_kernel(), args, {**_O_OVER_Q_KNOBS, cute_flash.FLASH_KV_STAGE_KEY: 12}
    )
    assert "flash_fa4_shared_storage(128, 12, 2, 2, cutlass.Float8E4M3FN" in code
    assert "assert _flash_storage_cls.size_in_bytes() == 232448" in code
    code = _render(
        _example_kernel(), args, {**_O_OVER_Q_KNOBS, cute_flash.FLASH_KV_STAGE_KEY: 3}
    )
    assert "flash_fa4_shared_storage(128, 3, 2, 2, cutlass.Float8E4M3FN" in code
    assert "assert _flash_storage_cls.size_in_bytes() == 84992" in code
    q, k, v = (torch.randn(4, 256, 128, dtype=torch.bfloat16) for _ in range(3))
    code = _render(
        dot_attention_k_major_any_head_dim,
        (q, k, v.transpose(1, 2).contiguous()),
        {**_O_OVER_Q_KNOBS, cute_flash.FLASH_KV_STAGE_KEY: 3},
    )
    assert "flash_fa4_shared_storage(128, 3, 2, 2, cutlass.BFloat16" in code
    assert "assert _flash_storage_cls.size_in_bytes() == 166912" in code


# ---------------------------------------------------------------------------
# GPU numerics (sm100).
# ---------------------------------------------------------------------------
def _require_sm100() -> None:
    if DEVICE.type != "cuda" or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the fused tcgen05 flash bodies need an sm100 device")


def _compile_default(
    kernel: Kernel, args: tuple[Any, ...], overrides: dict[str, object] | None = None
) -> tuple[Any, str]:
    bound = kernel.bind(args)
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    config = spec.default_config()
    if overrides:
        config = helion.Config(**{**config.config, **overrides})
    code = bound.to_code(config)
    assert "_helion_flash_rt" in code, code
    return bound.compile_config(config), code


@pytest.mark.parametrize(
    ("batch", "heads", "seq", "head_dim"),
    [(2, 4, 256, 64), (1, 2, 256, 128)],
    ids=["d64", "d128"],
)
def test_fp8_example_matches_reference_on_gpu(
    batch: int, heads: int, seq: int, head_dim: int
) -> None:
    """``examples/fp8_attention.py`` on the fused fp8 body: within the
    example's own tolerance of its ``torch._scaled_mm`` reference and of fp16
    SDPA (the fp8 output's one-bucket error)."""
    _require_sm100()
    (q, k, v), args = _example_args(batch, heads, seq, head_dim, device=DEVICE)
    fn, _ = _compile_default(_example_kernel(), args)
    out = fn(*args)
    reference = _example_module().fp8_attention_pytorch(q, k, v)()
    torch.testing.assert_close(out.float(), reference.float(), atol=0.1, rtol=0.1)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out.float(), sdpa.float(), atol=0.1, rtol=0.1)


@pytest.mark.parametrize(
    "overrides",
    [
        {cute_flash.FLASH_EPI_TMA_KEY: True},
        {cute_flash.FLASH_EPI_STG_KEY: True},
        {
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
            cute_flash.FLASH_P_STORE_REP_KEY: 32,
        },
        {**_O_OVER_Q_KNOBS, cute_flash.FLASH_KV_STAGE_KEY: 12},
    ],
    ids=["epi_tma", "epi_stg", "split_p_rep32", "o_over_q_deep_ring"],
)
def test_fp8_example_knobs_match_reference_on_gpu(overrides: dict[str, object]) -> None:
    """The staged output stores, the paired P store and the deepest
    o_over_q ring (the byte-sized verified schedule) over fp8 tiles."""
    _require_sm100()
    (q, k, v), args = _example_args(1, 2, 256, 128, device=DEVICE)
    fn, _ = _compile_default(_example_kernel(), args, overrides)
    out = fn(*args)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out.float(), sdpa.float(), atol=0.1, rtol=0.1)


def test_fp8_k_major_storage_matches_sdpa_on_gpu() -> None:
    """A dense ``[bh, d, s]`` fp8 V buffer: the K-major PV operand."""
    _require_sm100()
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(4, 256, 64, device=DEVICE).to(torch.float8_e4m3fn) for _ in range(3)
    )
    args = (q, k, v.transpose(1, 2).contiguous(), 1, 4)
    fn, code = _compile_default(fp8_dot_attention_k_major, args)
    assert "Major.K" in code
    out = fn(*args).float().reshape(4, 256, 64)
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )
    torch.testing.assert_close(out, sdpa, atol=0.1, rtol=0.1)


def test_alt_family_k_major_storage_matches_sdpa_on_gpu() -> None:
    """The alternating-warpgroup body over a dense K-major bf16 V buffer
    (hd128): its V descriptor agrees with the host's K-major PV operand."""
    _require_sm100()
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(4, 256, 128, device=DEVICE).to(torch.bfloat16) for _ in range(3)
    )
    args = (q, k, v.transpose(1, 2).contiguous())
    fn, code = _compile_default(dot_attention_k_major_any_head_dim, args, _ALT_FAMILY)
    assert "flash_fa4_alt_shared_storage(" in code
    assert "smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)" in code
    out = fn(*args).float()
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )
    torch.testing.assert_close(out, sdpa, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("kernel", "dtype", "k_major_storage"),
    [
        (dot_attention_k_major, torch.bfloat16, True),
        (dot_attention_rows, torch.bfloat16, False),
        (dot_attention_rows, torch.float16, False),
    ],
    ids=["bf16_k_major_storage", "bf16_rows_2d_root", "fp16_rows_2d_root"],
)
def test_16bit_2d_root_forms_match_sdpa_on_gpu(
    kernel: Kernel, dtype: torch.dtype, k_major_storage: bool
) -> None:
    """The 2-D ``hl.dot`` root (a scalar batch-head index) and the dense
    K-major V buffer on the 16-bit bodies."""
    _require_sm100()
    torch.manual_seed(0)
    q, k, v = (torch.randn(4, 256, 64, device=DEVICE).to(dtype) for _ in range(3))
    args = (q, k, v.transpose(1, 2).contiguous() if k_major_storage else v)
    fn, _ = _compile_default(kernel, args)
    out = fn(*args).float()
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )
    torch.testing.assert_close(out, sdpa, atol=2e-2, rtol=2e-2)
