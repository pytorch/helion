"""Fused row-local output epilogues on the CuTe flash-attention path."""

from __future__ import annotations

import ast
import math
import operator
import types
from typing import Callable

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_row_epilogue
from helion._compiler.cute.dtypes import round_scalar
from helion._compiler.cute.flash_row_epilogue import ROW
from helion._compiler.cute.flash_row_epilogue import VEC
from helion._compiler.cute.flash_row_epilogue import FlashRowEpilogueProgram
from helion._compiler.cute.flash_row_epilogue import RowEpilogueOp
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_aux import cpu_codegen
from helion._testing.flash_row_programs import match_row_program
from helion._testing.flash_row_programs import row_accumulator
from helion._testing.flash_row_programs import row_value
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _xsa_like(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Attention whose output row is projected off the normalized value row."""
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
        o = acc / l_i[:, :, None]
        v_self = v_view[tile_b, tile_m, :].to(torch.float32)
        v_norm = torch.sqrt(torch.sum(v_self * v_self, dim=-1, keepdim=True))
        vn = v_self / torch.clamp(v_norm, min=eps)
        proj = torch.sum(o * vn, dim=-1, keepdim=True)
        out[tile_b, tile_m, :] = (o - proj * vn).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _amin_normalized_rows(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, eps: float
) -> torch.Tensor:
    """``out = o / max(amin(|o|), eps)``: a min reduction over head_dim."""
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
        o = acc / l_i[:, :, None]
        scale = torch.clamp(torch.amin(torch.abs(o), dim=-1, keepdim=True), min=eps)
        out[tile_b, tile_m, :] = (o / scale).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _unsupported_epilogue(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """A cumulative sum over head_dim is not a row-local program."""
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
        o = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = torch.cumsum(o, dim=-1).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _gated_output(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """The output is a single-use product: ``o * g``."""
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
        o = acc / l_i[:, :, None]
        g = v_view[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = (o * g).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _two_products_added(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Both operands of the output add are single-use products."""
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
        o = acc / l_i[:, :, None]
        g = v_view[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = (o * g + o * (g * g)).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _projection_residual(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """A product feeds a reduction in one pass and the store pass's sub."""
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
        o = acc / l_i[:, :, None]
        g = v_view[tile_b, tile_m, :].to(torch.float32)
        proj = torch.sum(o * g, dim=-1, keepdim=True)
        out[tile_b, tile_m, :] = (o * g - proj * g).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _bf16_scaled_gate(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """``o * (v * 0.3)`` with the gate computed in bf16, which rounds it."""
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
        o = acc / l_i[:, :, None]
        g = v_view[tile_b, tile_m, :] * 0.3
        out[tile_b, tile_m, :] = (o * g).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _bf16_gate_into_store(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """``o.to(bf16) * v``: a bf16 product whose rounding is the store's."""
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
        o = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = o.to(out.dtype) * v_view[tile_b, tile_m, :]
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _bf16_scaled_sum(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """``o * sum(v * 0.3)`` in bf16: a rounded gate feeds a rounded sum."""
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
        o = acc / l_i[:, :, None]
        g = v_view[tile_b, tile_m, :] * 0.3
        s = torch.sum(g, dim=-1, keepdim=True)
        out[tile_b, tile_m, :] = (o * s).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _bf16_literal_rows(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Literals beside bf16 ``v``: compares against 0.3 and 0.0, a where
    selecting 0.3 and a bf16 ``full`` of 0.3, and ``o`` times a compare."""
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
        o = acc / l_i[:, :, None]
        v = v_view[tile_b, tile_m, :]
        g = torch.where(v > 0.0, v, 0.3) * (v + torch.full_like(v, 0.3))
        out[tile_b, tile_m, :] = (torch.where(v > 0.3, o, 0.0) + o * (v < 0.0) + g).to(
            out.dtype
        )
    return out.view(q_in.size())


def _ref_gated(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    return (y * v.float()).to(q.dtype)


def _ref_two_products(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    g = v.float()
    return (y * g + y * (g * g)).to(q.dtype)


def _ref_projection_residual(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    g = v.float()
    return (y * g - (y * g).sum(dim=-1, keepdim=True) * g).to(q.dtype)


def _ref_bf16_scaled_gate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    return (y * (v * 0.3).float()).to(q.dtype)


def _ref_bf16_scaled_sum(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    return (y * (v * 0.3).sum(dim=-1, keepdim=True).float()).to(q.dtype)


def _ref_bf16_gate_into_store(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v) * v


def _ref_bf16_literal_rows(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    g = torch.where(v > 0.0, v, 0.3) * (v + torch.full_like(v, 0.3))
    return (torch.where(v > 0.3, y, 0.0) + y * (v < 0.0) + g).to(q.dtype)


def _ref_xsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    vn = torch.nn.functional.normalize(v.float(), dim=-1, eps=eps)
    return (y - (y * vn).sum(dim=-1, keepdim=True) * vn).to(q.dtype)


def _meta_args(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(*shape, dtype=torch.bfloat16, device="meta")  # @ignore-device-lint
        for _ in range(3)
    )


def _graph_plan(
    bound: helion.runtime.kernel.BoundKernel,
) -> cute_flash.FlashGraphOutputPlan:
    with bound.env:
        device_ir = bound.host_function.device_ir
        root_block_ids = device_ir.grid_block_ids[0]
        loop = next(
            graph
            for graph in device_ir.graphs
            if type(graph).__name__ == "ForLoopGraphInfo"
        )
        plan = cute_flash._flash_graph_output_plan_from_graphs(
            device_ir.graphs,
            root_block_ids=root_block_ids,
            kv_block_id=loop.block_ids[0],
            score_plan=None,
        )
    assert plan is not None
    return plan


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
def test_row_program_detected_for_xsa_like_kernel() -> None:
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    assert bound.config_spec.cute_flash_search_enabled
    plan = _graph_plan(bound)
    assert plan.output_epilogue == flash_row_epilogue.FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM
    program = plan.row_epilogue
    assert program is not None
    assert program.aux_names == ("v_view",)
    assert len(program.scalar_exprs) == 1  # eps
    assert program.passes == 3
    reductions = [op for op in program.ops if op.op == "reduce"]
    assert [op.reduce_pass for op in reductions] == [0, 1]
    # The value-norm pass never reads the accumulator, so it can run before the loop.
    assert flash_row_epilogue.hoistable_passes(program) == 1
    assert program.op(program.output).op == "sub"


def _expression(program: FlashRowEpilogueProgram, name: str) -> str:
    """The program's value ``name`` as nested ops, a cast spelled as its
    dtype (``bfloat16(...)``) and a reduction as its kind."""
    op = program.op(name)
    args = ", ".join(_expression(program, arg) for arg in op.inputs)
    if op.op == "cast":
        return f"{str(op.attrs[0]).removeprefix('torch.')}({args})"
    if op.op == "const":
        return repr(op.attrs[0])
    if op.op == "reduce":
        return f"{op.attrs[0]}({args})"
    return f"{op.op}({args})" if args else op.op


@pytest.mark.cpu_only
def test_row_program_rounds_arithmetic_in_bf16() -> None:
    # A bf16 product in the gate (``v * 0.3``) and a bf16 sum are rounded to
    # bf16 as torch rounds them; the product feeding the store
    # (``o.to(bf16) * v``) is left to the store's rounding.
    expected = {
        _bf16_scaled_gate: "mul(o_norm, bfloat16(mul(aux, 0.3)))",
        _bf16_scaled_sum: "mul(o_norm, bfloat16(sum(bfloat16(mul(aux, 0.3)))))",
        _bf16_gate_into_store: "mul(bfloat16(o_norm), aux)",
        _gated_output: "mul(o_norm, float32(aux))",
    }
    with cpu_codegen():
        for kernel, expression in expected.items():
            bound = kernel.bind(_meta_args((2, 8, 1024, 64)))
            assert bound.config_spec.cute_flash_search_enabled, kernel
            program = _graph_plan(bound).row_epilogue
            assert program is not None
            assert _expression(program, program.output) == expression


@pytest.mark.cpu_only
def test_row_program_rounds_literals_while_binding() -> None:
    # The matcher runs while the kernel binds, under fake-tensor mode with
    # the arguments' device as the default, where a tensor's ``item()`` is
    # an unbacked symbol.  The compare's literal, the where's literal branch
    # and the bf16 constant are rounded there all the same, to
    # bf16(0.3) = 0.30078125, and the compare ``o`` multiplies by is 0.0 or
    # 1.0 in fp32.
    with cpu_codegen():
        bound = _bf16_literal_rows.bind(_meta_args((2, 8, 1024, 64)))
        assert bound.config_spec.cute_flash_search_enabled
        program = _graph_plan(bound).row_epilogue
    assert program is not None
    consts = [op.attrs[0] for op in program.ops if op.op == "const"]
    assert consts.count(0.30078125) == 3 and 0.3 not in consts
    (mask,) = [op.name for op in program.ops if op.op == "lt"]
    assert [
        op.attrs for op in program.ops if op.op == "cast" and op.inputs == (mask,)
    ] == [(torch.float32,)]


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
def test_row_program_rejects_non_row_local_epilogue() -> None:
    bound = _unsupported_epilogue.bind(_meta_args((2, 8, 1024, 64)))
    assert not bound.config_spec.cute_flash_search_enabled


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
@pytest.mark.parametrize(
    "config",
    (
        {"cute_flash_pipeline_family": "fa4"},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_stg": True},
        {"cute_flash_pipeline_family": "fa4_2cta", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "ws_overlap"},
        {"cute_flash_pipeline_family": "ws_overlap", "cute_flash_s_stage": 1},
    ),
)
def test_row_program_codegen_every_route(config: dict[str, object]) -> None:
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    full = {"block_sizes": [1, 128, 128], "cute_flash_persistent": False, **config}
    code = bound.to_code(bound._normalized_config_copy(helion.Config(**full)))
    ast.parse(code)
    assert "_flash_mEpiAux0" in code
    assert "_ep0_v" in code or "_ep_v" in code
    # Chunk indices are literal so the DSL unrolls every fragment access.
    assert "range_constexpr(cute.size" not in code
    if config["cute_flash_pipeline_family"].startswith("fa4"):
        # The program runs on the softmax warpgroups (``cute_flash_row_epilogue_warps``
        # defaults to ``softmax``): the value-norm pass sits ahead of the softmax
        # KV loop when the aux rows come from gmem, and right after the
        # ``aux_full`` wait when the aux tile is staged through sO.
        softmax0 = code.split("if warp_idx < 4:", 1)[1].split("\n    if ", 1)[0]
        correction = code.split("(warp_idx >= 8) & (warp_idx < 12)", 1)[1]
        assert "cute.math.sqrt" in softmax0 and "cute.math.sqrt" not in correction
        staged = config.get("cute_flash_epi_tma") or config.get("cute_flash_epi_stg")
        anchor = (
            "mbar_spin_wait(flash_aux_full_ptr + 0"
            if staged
            else "for flash_kv in cutlass.range"
        )
        assert (softmax0.index("cute.math.sqrt") > softmax0.index(anchor)) is bool(
            staged
        )


def test_row_mma_completes_a_min_reduction_with_the_lane_group_min() -> None:
    # The row program's ``amin`` completes over the lanes sharing a row with
    # ``lane_group_min``, a helper the emitter names by composition
    # (``lane_group_{sum,max,min}``); render it so the name stays pinned.
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 256, 64, dtype=torch.bfloat16) for _ in range(3))
    with cpu_codegen():
        bound = _amin_normalized_rows.bind((q, k, v, 1e-6))
        config = {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "row_mma",
        }
        code = bound.to_code(bound._normalized_config_copy(helion.Config(**config)))
    assert "'kind': 'helion_flash_row_mma'" in code
    assert "_helion_flash_rowmma.lane_group_min(rm_ep_v2, 8)" in code
    assert "cute.math.min(rm_ep_v2_0_p0, rm_ep_v1_p0, propagate_nan=True)" in code
    assert "lane_group_max" not in code and "lane_group_sum" not in code


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
def test_row_epilogue_emitter_hoists_and_unrolls() -> None:
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    program = _graph_plan(bound).row_epilogue
    assert program is not None
    prologue, epilogue = flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=4,
        chunk_width=16,
        elem_var="j",
        o_alloc="{o} = alloc_o()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc_a()"],
        aux_loads=[["load_a({i}, {a})"]],
        aux_elems=["cutlass.Float32({a}[{j}])"],
        out_alloc="{out} = alloc_out()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=["eps"],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
    )
    assert "cute.math.sqrt" in prologue and "load_o" not in prologue
    assert "load_a(0, e_a0_0)" in prologue and "load_a(1, e_a0_1)" in prologue
    assert "load_o(0, e_o0)" in epilogue and "store_out(3, e_out1)" in epilogue
    # Element pairs per step; the two chunk-parity accumulators and both lanes
    # are folded once the pass closes.
    assert epilogue.count("for j in cutlass.range_constexpr(0, 16, 2):") == 8
    assert "e_v4_0_p0 + e_v4_0_p1 + e_v4_1_p0 + e_v4_1_p1" in prologue
    assert "cute.arch.fma_packed_f32x2(" in epilogue
    scalar_prologue, scalar_epilogue = flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=4,
        chunk_width=16,
        elem_var="j",
        o_alloc="{o} = alloc_o()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc_a()"],
        aux_loads=[["load_a({i}, {a})"]],
        aux_elems=["cutlass.Float32({a}[{j}])"],
        out_alloc="{out} = alloc_out()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=["eps"],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
        packed=False,
    )
    # The scalar form walks the same pairs with the same accumulators and folds
    # the same products (one scalar fma per lane of a packed fma).
    assert scalar_epilogue.count("for j in cutlass.range_constexpr(0, 16, 2):") == 8
    assert "e_v4_0_p0 + e_v4_0_p1 + e_v4_1_p0 + e_v4_1_p1" in scalar_prologue
    assert "_packed_f32x2(" not in scalar_epilogue
    assert scalar_epilogue.count("cute.math.fma(") == 2 * epilogue.count(
        "cute.arch.fma_packed_f32x2("
    )
    assert scalar_prologue.count("cute.math.fma(") == 2 * prologue.count(
        "cute.arch.fma_packed_f32x2("
    )


# --------------------------------------------------------------------------
# Emitter semantics: run the emitted source on plain Python floats
# --------------------------------------------------------------------------

_HEAD_DIM = 16
_CHUNK = 8


def _program(
    ops: list[RowEpilogueOp], output: str, aux_count: int
) -> FlashRowEpilogueProgram:
    scheduled = flash_row_epilogue._schedule(ops)
    return FlashRowEpilogueProgram(
        tuple(scheduled),
        output,
        tuple(f"aux{i}" for i in range(aux_count)),
        (torch.bfloat16,) * aux_count,
        (),
        flash_row_epilogue._lookup(scheduled, output).avail,
    )


def _emit(program: FlashRowEpilogueProgram, packed: bool) -> tuple[str, str]:
    return flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=_HEAD_DIM // _CHUNK,
        chunk_width=_CHUNK,
        elem_var="j",
        o_alloc="{o} = alloc()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc()"] * len(program.aux_names),
        aux_loads=[
            [f"load_aux({k}, {{i}}, {{a}})"] for k in range(len(program.aux_names))
        ],
        aux_elems=["cutlass.Float32({a}[{j}])"] * len(program.aux_names),
        out_alloc="{out} = alloc()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=[],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
        packed=packed,
    )


class _Float32:
    inf = math.inf

    def __call__(self, value: object) -> float:
        return float(value)  # pyrefly: ignore [bad-argument-type]


class _Narrow(float):
    """A 16-bit register value of the emitted code."""


def _narrow(dtype: torch.dtype) -> Callable[[object], _Narrow]:
    """The ``cutlass.BFloat16`` / ``Float16`` conversion: rounds to ``dtype``."""
    return lambda value: _Narrow(round_scalar(float(value), dtype))  # pyrefly: ignore [bad-argument-type]


def _fp32_lanes(*operands: tuple[float, ...]) -> None:
    """The packed ``f32x2`` ops take fp32 registers only: the DSL fails to
    verify the instruction on a 16-bit or a Boolean operand."""
    for operand in operands:
        assert not any(isinstance(lane, (_Narrow, bool)) for lane in operand), operand


def _run_emitted(
    program: FlashRowEpilogueProgram,
    o_row: list[float],
    aux_rows: list[list[float]],
    packed: bool,
) -> list[float]:
    """Execute the emitted passes with DSL stand-ins on Python floats."""
    result = [math.nan] * _HEAD_DIM

    def load_o(i: int, o: list[float]) -> None:
        o[:] = o_row[i * _CHUNK : (i + 1) * _CHUNK]

    def load_aux(k: int, i: int, a: list[float]) -> None:
        a[:] = aux_rows[k][i * _CHUNK : (i + 1) * _CHUNK]

    def store_out(i: int, out: list[float]) -> None:
        result[i * _CHUNK : (i + 1) * _CHUNK] = out

    def packed_op(
        fn: Callable[[float, float], float],
    ) -> Callable[..., tuple[float, float]]:
        def op(x: tuple[float, float], y: tuple[float, float]) -> tuple[float, float]:
            _fp32_lanes(x, y)
            return fn(x[0], y[0]), fn(x[1], y[1])

        return op

    def packed_fma(
        x: tuple[float, float], y: tuple[float, float], c: tuple[float, float]
    ) -> tuple[float, float]:
        _fp32_lanes(x, y, c)
        return x[0] * y[0] + c[0], x[1] * y[1] + c[1]

    namespace: dict[str, object] = {
        "alloc": lambda: [math.nan] * _CHUNK,
        "load_o": load_o,
        "scale": lambda o: None,
        "load_aux": load_aux,
        "store_out": store_out,
        "cutlass": types.SimpleNamespace(
            Float32=_Float32(),
            BFloat16=_narrow(torch.bfloat16),
            Float16=_narrow(torch.float16),
            range_constexpr=range,
            select_=lambda c, a, b: a if c else b,
        ),
        "cute": types.SimpleNamespace(
            math=types.SimpleNamespace(
                fma=lambda a, b, c: a * b + c,
                max=lambda a, b, propagate_nan=True: max(a, b),
                min=lambda a, b, propagate_nan=True: min(a, b),
                sqrt=math.sqrt,
                rsqrt=lambda x: 1.0 / math.sqrt(x),
                exp=math.exp,
                exp2=lambda x, fastmath=False: 2.0**x,
                log=math.log,
                log2=math.log2,
                tanh=math.tanh,
                abs=abs,
                rcp=lambda x, approx=False, ftz=False: 1.0 / x,
            ),
            arch=types.SimpleNamespace(
                fma_packed_f32x2=packed_fma,
                add_packed_f32x2=packed_op(operator.add),
                sub_packed_f32x2=packed_op(operator.sub),
                mul_packed_f32x2=packed_op(operator.mul),
            ),
        ),
    }
    prologue, epilogue = _emit(program, packed)
    exec(prologue + "\n" + epilogue, namespace)
    return result


_O = RowEpilogueOp("o", VEC, "o_norm")
_A0 = RowEpilogueOp("a0", VEC, "aux", (), (0,))
_A1 = RowEpilogueOp("a1", VEC, "aux", (), (1,))


def _mul(name: str, a: str, b: str) -> RowEpilogueOp:
    return RowEpilogueOp(name, VEC, "mul", (a, b))


def _reduce_sum(name: str, source: str) -> RowEpilogueOp:
    return RowEpilogueOp(name, ROW, "reduce", (source,), ("sum",))


_EMITTER_CASES: dict[
    str,
    tuple[
        list[RowEpilogueOp],
        str,
        int,
        Callable[[list[float], list[list[float]]], list[float]],
        int,
    ],
] = {
    # The output itself is a single-use product: nothing to fold into.
    "output_is_mul": (
        [_O, _A0, _mul("m", "o", "a0")],
        "m",
        1,
        lambda o, aux: [x * a for x, a in zip(o, aux[0], strict=True)],
        0,
    ),
    # Both add operands are single-use products: one folds, the other stays.
    "two_muls_into_add": (
        [
            _O,
            _A0,
            _A1,
            _mul("m0", "o", "a0"),
            _mul("m1", "o", "a1"),
            RowEpilogueOp("s", VEC, "add", ("m0", "m1")),
        ],
        "s",
        2,
        lambda o, aux: [
            x * a + x * b for x, a, b in zip(o, aux[0], aux[1], strict=True)
        ],
        1,
    ),
    "two_muls_into_sub": (
        [
            _O,
            _A0,
            _A1,
            _mul("m0", "o", "a0"),
            _mul("m1", "a0", "a1"),
            RowEpilogueOp("s", VEC, "sub", ("m0", "m1")),
        ],
        "s",
        2,
        lambda o, aux: [
            x * a - a * b for x, a, b in zip(o, aux[0], aux[1], strict=True)
        ],
        1,
    ),
    # A product feeds the first pass's reduction and the store pass's sub,
    # whose other operand is a product as well.
    "mul_into_sum_and_output": (
        [
            _O,
            _A0,
            _mul("m", "o", "a0"),
            _reduce_sum("r", "m"),
            _mul("pa", "r", "a0"),
            RowEpilogueOp("s", VEC, "sub", ("m", "pa")),
        ],
        "s",
        1,
        lambda o, aux: [
            x * a - sum(y * b for y, b in zip(o, aux[0], strict=True)) * a
            for x, a in zip(o, aux[0], strict=True)
        ],
        2,
    ),
    # XSA's shape: the projection folds into the sum, the residual into the sub.
    "projection_off_the_aux_row": (
        [
            _O,
            _A0,
            _mul("m", "o", "a0"),
            _reduce_sum("r", "m"),
            _mul("pa", "r", "a0"),
            RowEpilogueOp("s", VEC, "sub", ("o", "pa")),
        ],
        "s",
        1,
        lambda o, aux: [
            x - sum(y * b for y, b in zip(o, aux[0], strict=True)) * a
            for x, a in zip(o, aux[0], strict=True)
        ],
        2,
    ),
}


@pytest.mark.parametrize("case", sorted(_EMITTER_CASES))
@pytest.mark.parametrize("packed", (True, False))
def test_row_epilogue_emitter_folds_one_product_per_consumer(
    case: str, packed: bool
) -> None:
    ops, output, aux_count, reference, fused_per_step = _EMITTER_CASES[case]
    program = _program(ops, output, aux_count)
    torch.manual_seed(0)
    o_row = torch.randn(_HEAD_DIM, dtype=torch.float64).tolist()
    aux_rows = [
        torch.randn(_HEAD_DIM, dtype=torch.float64).tolist() for _ in range(aux_count)
    ]
    # Every name the emitted source reads is defined (the packed emitter used to
    # skip a folded product's definition while its consumer still read it, or
    # to look up the consumer of the output) and the arithmetic is right.
    result = _run_emitted(program, o_row, aux_rows, packed)
    expected = reference(o_row, aux_rows)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)
    prologue, epilogue = _emit(program, packed)
    code = prologue + "\n" + epilogue
    # Products fold exactly where the program has a consumer for them: the same
    # set in both instruction forms (a packed fma covers two lanes).
    fma_lines = code.count(
        "cute.arch.fma_packed_f32x2(" if packed else "cute.math.fma("
    )
    steps = _HEAD_DIM // _CHUNK
    assert fma_lines == fused_per_step * steps * (1 if packed else 2)
    if packed:
        assert "cute.math.fma(" not in code
    else:
        assert "_packed_f32x2(" not in code
    if case.startswith("two_muls"):
        # The unfolded product is still materialized.
        assert ("mul_packed_f32x2(" in code) is packed
        assert ("e_m0_p0 = (e_o_p0 * e_a0_p0)" in code) is not packed
    if case == "output_is_mul":
        assert "fma" not in code


@pytest.mark.cpu_only
@pytest.mark.parametrize("packed", (True, False))
def test_row_epilogue_roundings_keep_fp32_lanes(packed: bool) -> None:
    # o * bf16(a0 * 0.3) + bf16(sum(bf16(a0 * 0.3))): a cast to bf16 rounds
    # and widens back to fp32, so the packed f32x2 products and the sum's
    # packed accumulators read fp32 lanes holding the rounded values.
    ops = [
        _O,
        _A0,
        RowEpilogueOp("c", ROW, "const", (), (0.3,)),
        _mul("g", "a0", "c"),
        RowEpilogueOp("gr", VEC, "cast", ("g",), (torch.bfloat16,)),
        _mul("m", "o", "gr"),
        _reduce_sum("r", "gr"),
        RowEpilogueOp("rr", ROW, "cast", ("r",), (torch.bfloat16,)),
        RowEpilogueOp("s", VEC, "add", ("m", "rr")),
    ]
    program = _program(ops, "s", 1)
    torch.manual_seed(0)
    o_row = torch.randn(_HEAD_DIM, dtype=torch.float64).tolist()
    aux_row = torch.randn(_HEAD_DIM, dtype=torch.float64).tolist()
    result = _run_emitted(program, o_row, [aux_row], packed)
    gates = [round_scalar(a * 0.3, torch.bfloat16) for a in aux_row]
    total = round_scalar(sum(gates), torch.bfloat16)
    expected = [x * g + total for x, g in zip(o_row, gates, strict=True)]
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)
    prologue, epilogue = _emit(program, packed)
    code = prologue + "\n" + epilogue
    assert code.count("cutlass.BFloat16(") == code.count(
        "cutlass.Float32(cutlass.BFloat16("
    )
    assert ("add_packed_f32x2(" in code) is packed


@pytest.mark.cpu_only
@pytest.mark.parametrize("packed", (True, False))
def test_row_epilogue_compare_operand_keeps_fp32_lanes(packed: bool) -> None:
    # o * (o > 0.3): the matcher converts the compare to fp32 (0.0 or 1.0),
    # so the packed product reads fp32 lanes rather than Booleans.
    graph = torch.fx.Graph()
    o = row_accumulator(graph)
    mask = row_value(graph, torch.ops.aten.gt.Scalar, (o, 0.3), torch.bool)
    value = row_value(graph, torch.ops.aten.mul.Tensor, (o, mask), torch.float32)
    program = match_row_program(graph, o, value)
    assert program is not None
    torch.manual_seed(0)
    o_row = torch.randn(_HEAD_DIM, dtype=torch.float64).tolist()
    assert _run_emitted(program, o_row, [], packed) == [x * (x > 0.3) for x in o_row]


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
def test_row_epilogue_emitter_forms_agree_bitwise_on_the_xsa_program() -> None:
    # Same pairs, same accumulators, same folds: the two instruction forms
    # differ only in the instruction spelling.
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    program = _graph_plan(bound).row_epilogue
    assert program is not None
    packed_prologue, packed_epilogue = flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=4,
        chunk_width=16,
        elem_var="j",
        o_alloc="{o} = alloc_o()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc_a()"],
        aux_loads=[["load_a({i}, {a})"]],
        aux_elems=["cutlass.Float32({a}[{j}])"],
        out_alloc="{out} = alloc_out()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=["eps"],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
        packed=True,
    )
    scalar_prologue, scalar_epilogue = flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=4,
        chunk_width=16,
        elem_var="j",
        o_alloc="{o} = alloc_o()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc_a()"],
        aux_loads=[["load_a({i}, {a})"]],
        aux_elems=["cutlass.Float32({a}[{j}])"],
        out_alloc="{out} = alloc_out()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=["eps"],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
        packed=False,
    )

    # Outside the arithmetic lines the two emissions are identical.
    def arithmetic_free(text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if "_packed_f32x2(" not in line
            and "cute.math.fma(" not in line
            and not any(op in line for op in (" + ", " - ", " * "))
        ]

    assert arithmetic_free(packed_prologue) == arithmetic_free(scalar_prologue)
    assert arithmetic_free(packed_epilogue) == arithmetic_free(scalar_epilogue)


@pytest.mark.parametrize(
    "kernel, reference",
    (
        pytest.param(_gated_output, _ref_gated, id="output_is_mul"),
        pytest.param(_two_products_added, _ref_two_products, id="two_muls_into_add"),
        pytest.param(
            _projection_residual, _ref_projection_residual, id="mul_into_sum_and_sub"
        ),
    ),
)
def test_row_programs_with_folded_products_match_reference(
    kernel: helion.Kernel, reference: Callable[..., torch.Tensor]
) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bound = kernel.bind((q, k, v))
    assert bound.config_spec.cute_flash_search_enabled
    config = bound._normalized_config_copy(
        helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_persistent=True,
            cute_flash_epi_tma=True,
        )
    )
    code = bound.to_code(config)
    assert "cute.arch.fma_packed_f32x2(" in code or kernel is _gated_output
    out = bound.compile_config(config)(q, k, v)
    torch.testing.assert_close(
        out.float(), reference(q, k, v).float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "kernel, reference",
    (
        pytest.param(_bf16_scaled_gate, _ref_bf16_scaled_gate, id="rounded_gate"),
        pytest.param(_bf16_scaled_sum, _ref_bf16_scaled_sum, id="rounded_sum"),
        pytest.param(
            _bf16_gate_into_store, _ref_bf16_gate_into_store, id="rounded_operand"
        ),
        pytest.param(_bf16_literal_rows, _ref_bf16_literal_rows, id="literals"),
    ),
)
@pytest.mark.parametrize("family", ("fa4", "row_mma"))
def test_row_programs_that_round_to_bf16_match_reference(
    kernel: helion.Kernel, reference: Callable[..., torch.Tensor], family: str
) -> None:
    # The FA4 output epilogue and the register-MMA row programs evaluate the
    # program on packed f32x2 pairs: a rounding keeps its lanes fp32, which
    # those ops take (a bf16 register fails the DSL's verifier there).
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bound = kernel.bind((q, k, v))
    assert bound.config_spec.cute_flash_search_enabled
    persistence = (
        {"cute_flash_persistent": True, "cute_flash_epi_tma": True}
        if family == "fa4"
        else {"cute_flash_persistent": False}
    )
    config = bound._normalized_config_copy(
        helion.Config(
            block_sizes=[1, 128, 128], cute_flash_pipeline_family=family, **persistence
        )
    )
    code = bound.to_code(config)
    assert "cutlass.Float32(cutlass.BFloat16(" in code
    assert "cute.arch.mul_packed_f32x2(" in code
    out = bound.compile_config(config)(q, k, v)
    torch.testing.assert_close(
        out.float(), reference(q, k, v).float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "config",
    (
        {"cute_flash_pipeline_family": "fa4"},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "ws_overlap"},
        # The register-MMA row programs evaluate the program in their combine
        # epilogue (here behind a double-buffered chunk loop over 256 keys per
        # warp); explicit configs are legal on any grid.
        {"cute_flash_pipeline_family": "row_mma"},
    ),
)
def test_row_program_runtime_matches_reference(config: dict[str, object]) -> None:
    torch.manual_seed(0)
    outputs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        q = torch.randn(2, 8, 1024, 64, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        bound = _xsa_like.bind((q, k, v, 1e-6))
        full = {"block_sizes": [1, 128, 128], "cute_flash_persistent": False, **config}
        fn = bound.compile_config(bound._normalized_config_copy(helion.Config(**full)))
        out = fn(q, k, v, 1e-6)
        torch.testing.assert_close(
            out.float(), _ref_xsa(q, k, v, 1e-6).float(), rtol=1e-2, atol=5e-2
        )
        outputs.append(out)
    assert not torch.equal(outputs[0], outputs[1])
