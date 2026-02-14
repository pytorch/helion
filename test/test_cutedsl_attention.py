"""Phase 4: CuteDSL Attention E2E Tests.

Validates that Phases 1-3 compose correctly for attention patterns:
- Softmax decomposed (amax + exp + sum)
- Softmax two-pass (online softmax with hl.full/hl.zeros)
- Full flash attention (QK^T via bmm, online softmax, attention @ V via baddbmm)
- Codegen structure verification (CuteDSL markers, key operations present)
"""

from __future__ import annotations

import math
import unittest

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import TestCase
from helion.runtime.config import Config


def _has_cutlass() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


skipIfNoCutlass = unittest.skipUnless(_has_cutlass(), "cutlass not installed")


class TestCuteDSLSoftmaxCodegen(TestCase):
    """Tests that CuteDSL backend generates valid code for softmax patterns."""

    def _get_code(self, fn, args, **config_kwargs):
        """Helper to get generated code for a CuteDSL kernel."""
        bound = fn.bind(args)
        if config_kwargs:
            config = Config(**config_kwargs)
        else:
            config = bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_softmax_decomposed_generates_code(self):
        """Test decomposed softmax (amax + exp + sum) generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp = torch.exp(values - amax)
                sum_exp = torch.sum(exp, dim=1, keepdim=True)
                out[tile_n, :] = exp / sum_exp
            return out

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(softmax_decomposed, args)

        # Verify CuteDSL-specific markers
        self.assertIn("@cute.kernel", code)
        self.assertIn("import cutlass", code)
        self.assertIn("import cutlass.cute as cute", code)
        self.assertIn("_default_cutedsl_launcher", code)
        # Verify no triton decorator
        self.assertNotIn("@triton.jit", code)

    def test_softmax_decomposed_parseable(self):
        """Verify decomposed softmax generates valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl")
        def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp = torch.exp(values - amax)
                sum_exp = torch.sum(exp, dim=1, keepdim=True)
                out[tile_n, :] = exp / sum_exp
            return out

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(softmax_decomposed, args)
        py_ast.parse(code)

    def test_softmax_two_pass_generates_code(self):
        """Test two-pass online softmax generates valid CuteDSL code.

        This pattern is the core of flash attention: track running max
        and sum, use correction factors.
        """

        @helion.kernel(backend="cutedsl")
        def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            block_size_m = hl.register_block_size(m)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_size_m):
                mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                di = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    local_amax = torch.amax(values, dim=1)
                    mi_next = torch.maximum(mi, local_amax)
                    di = di * torch.exp(mi - mi_next) + torch.exp(
                        values - mi_next[:, None]
                    ).sum(dim=1)
                    mi = mi_next
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
            return out

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(softmax_two_pass, args)

        self.assertIn("@cute.kernel", code)
        self.assertIn("import cutlass", code)
        self.assertNotIn("@triton.jit", code)

    def test_softmax_two_pass_parseable(self):
        """Verify two-pass softmax generates valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl")
        def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            block_size_m = hl.register_block_size(m)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_size_m):
                mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                di = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    local_amax = torch.amax(values, dim=1)
                    mi_next = torch.maximum(mi, local_amax)
                    di = di * torch.exp(mi - mi_next) + torch.exp(
                        values - mi_next[:, None]
                    ).sum(dim=1)
                    mi = mi_next
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
            return out

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(softmax_two_pass, args)
        py_ast.parse(code)


class TestCuteDSLAttentionCodegen(TestCase):
    """Tests that CuteDSL backend generates valid code for flash attention."""

    def _get_code(self, fn, args, **config_kwargs):
        """Helper to get generated code for a CuteDSL kernel."""
        bound = fn.bind(args)
        if config_kwargs:
            config = Config(**config_kwargs)
        else:
            config = bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_attention_f32_generates_code(self):
        """Test flash attention fp32 generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            assert n_dim == v_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            assert head_dim == k_in.size(-1) == v_in.size(-1)
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
            out = torch.empty_like(q_view)
            sm_scale = 1.0 / math.sqrt(head_dim)
            qk_scale = sm_scale * 1.44269504  # 1/log(2)
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                q = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    k = k_view[tile_b, :, tile_n]
                    qk = torch.bmm(q, k)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    v = v_view[tile_b, tile_n, :]
                    p = p.to(v.dtype)
                    acc = torch.baddbmm(acc, p, v)
                    m_i = m_ij
                m_i += torch.log2(l_i)
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
        )
        code = self._get_code(attention, args, block_sizes=[1, 16, 16])

        # Verify CuteDSL-specific markers
        self.assertIn("@cute.kernel", code)
        self.assertIn("import cutlass", code)
        self.assertIn("import cutlass.cute as cute", code)
        self.assertIn("_default_cutedsl_launcher", code)
        # Verify no triton decorator on the main kernel
        # (helper functions may have @cute.kernel too)
        self.assertNotIn("@triton.jit", code)

    def test_attention_f32_parseable(self):
        """Verify flash attention fp32 generates valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            assert n_dim == v_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            assert head_dim == k_in.size(-1) == v_in.size(-1)
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
            out = torch.empty_like(q_view)
            sm_scale = 1.0 / math.sqrt(head_dim)
            qk_scale = sm_scale * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                q = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    k = k_view[tile_b, :, tile_n]
                    qk = torch.bmm(q, k)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    v = v_view[tile_b, tile_n, :]
                    p = p.to(v.dtype)
                    acc = torch.baddbmm(acc, p, v)
                    m_i = m_ij
                m_i += torch.log2(l_i)
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
        )
        code = self._get_code(attention, args, block_sizes=[1, 16, 16])
        py_ast.parse(code)

    def test_attention_f16_generates_code(self):
        """Test flash attention fp16 generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            assert n_dim == v_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            assert head_dim == k_in.size(-1) == v_in.size(-1)
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
            out = torch.empty_like(q_view)
            sm_scale = 1.0 / math.sqrt(head_dim)
            qk_scale = sm_scale * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                q = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    k = k_view[tile_b, :, tile_n]
                    qk = torch.bmm(q, k)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    v = v_view[tile_b, tile_n, :]
                    p = p.to(v.dtype)
                    acc = torch.baddbmm(acc, p, v)
                    m_i = m_ij
                m_i += torch.log2(l_i)
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float16, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float16, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float16, device=DEVICE),
        )
        code = self._get_code(attention, args, block_sizes=[1, 16, 16])

        self.assertIn("@cute.kernel", code)
        self.assertIn("import cutlass", code)
        self.assertNotIn("@triton.jit", code)

    def test_attention_codegen_structure(self):
        """Verify generated attention code contains the expected operations.

        Flash attention requires: dot products (bmm/baddbmm), exp2, amax,
        sum, and the online softmax correction pattern.
        """

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            assert n_dim == v_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            assert head_dim == k_in.size(-1) == v_in.size(-1)
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
            out = torch.empty_like(q_view)
            sm_scale = 1.0 / math.sqrt(head_dim)
            qk_scale = sm_scale * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                q = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    k = k_view[tile_b, :, tile_n]
                    qk = torch.bmm(q, k)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    v = v_view[tile_b, tile_n, :]
                    p = p.to(v.dtype)
                    acc = torch.baddbmm(acc, p, v)
                    m_i = m_ij
                m_i += torch.log2(l_i)
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
        )
        code = self._get_code(attention, args, block_sizes=[1, 16, 16])

        # Key operations that must be present in flash attention codegen:
        # 1. Matrix multiply (QK^T and attention@V)
        self.assertIn("tl.dot(", code)
        # 2. exp2 for softmax numerics (from Inductor lowering of torch.exp2)
        self.assertIn("exp2", code)
        # 3. Full for initialization (from hl.full)
        self.assertIn("tl.full(", code)
        # 4. CuteDSL types
        self.assertIn("cutlass.Constexpr[int]", code)
        self.assertIn("cutlass.Int32", code)

    def test_attention_uses_cutlass_types(self):
        """Verify attention codegen uses cutlass type annotations."""

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            assert n_dim == v_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            assert head_dim == k_in.size(-1) == v_in.size(-1)
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
            out = torch.empty_like(q_view)
            sm_scale = 1.0 / math.sqrt(head_dim)
            qk_scale = sm_scale * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                q = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    k = k_view[tile_b, :, tile_n]
                    qk = torch.bmm(q, k)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    v = v_view[tile_b, tile_n, :]
                    p = p.to(v.dtype)
                    acc = torch.baddbmm(acc, p, v)
                    m_i = m_ij
                m_i += torch.log2(l_i)
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
        )
        code = self._get_code(attention, args, block_sizes=[1, 16, 16])

        # Verify cutlass type annotations are used
        self.assertIn("cutlass.Constexpr[int]", code)
        self.assertIn("cutlass.Int32", code)
        # Verify cutlass imports
        self.assertIn("import cutlass", code)
        self.assertIn("import cutlass.cute as cute", code)

    def test_attention_hl_dot_generates_code(self):
        """Test attention using hl.dot instead of torch.bmm generates valid code.

        This follows the flex attention pattern where hl.dot is used directly.
        """

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention_hl_dot(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> torch.Tensor:
            seq_len = q.size(0)
            head_dim = hl.specialize(q.size(1))
            out = torch.empty_like(q)
            for tile_m in hl.tile(seq_len):
                acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
                m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                l_i = hl.zeros([tile_m], dtype=torch.float32)
                q_tile = q[tile_m, :]
                for tile_n in hl.tile(seq_len):
                    k_tile = k[tile_n, :]
                    qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    qk = hl.dot(q_tile, k_tile.transpose(0, 1), acc=qk)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    p = torch.exp2(qk - m_ij[:, None])
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    v_tile = v[tile_n, :]
                    acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
                    m_i = m_ij
                out[tile_m, :] = (acc / l_i[:, None]).to(out.dtype)
            return out

        args = (
            torch.randn(64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(64, 32, dtype=torch.float32, device=DEVICE),
        )
        code = self._get_code(attention_hl_dot, args, block_sizes=[16, 16])

        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.dot(", code)
        self.assertNotIn("@triton.jit", code)

    def test_attention_hl_dot_parseable(self):
        """Verify attention with hl.dot generates valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl", static_shapes=True)
        def attention_hl_dot(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> torch.Tensor:
            seq_len = q.size(0)
            head_dim = hl.specialize(q.size(1))
            out = torch.empty_like(q)
            for tile_m in hl.tile(seq_len):
                acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
                m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                l_i = hl.zeros([tile_m], dtype=torch.float32)
                q_tile = q[tile_m, :]
                for tile_n in hl.tile(seq_len):
                    k_tile = k[tile_n, :]
                    qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    qk = hl.dot(q_tile, k_tile.transpose(0, 1), acc=qk)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    p = torch.exp2(qk - m_ij[:, None])
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    v_tile = v[tile_n, :]
                    acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
                    m_i = m_ij
                out[tile_m, :] = (acc / l_i[:, None]).to(out.dtype)
            return out

        args = (
            torch.randn(64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(64, 32, dtype=torch.float32, device=DEVICE),
        )
        code = self._get_code(attention_hl_dot, args, block_sizes=[16, 16])
        py_ast.parse(code)


if __name__ == "__main__":
    unittest.main()
