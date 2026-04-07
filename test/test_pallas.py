from __future__ import annotations

import math
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallas
import helion.language as hl


@helion.kernel(backend="pallas", static_shapes=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.relu(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sin(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_pointwise_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(torch.sin(torch.relu(x[tile] * y[tile])))
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_affine_scalar_args(
    x: torch.Tensor,
    scale: int,
    bias: float,
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * scale + bias
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_matmul_broadcast_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], device=x.device, dtype=torch.promote_types(x.dtype, y.dtype)
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc + bias[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sum_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_max_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_tile_begin_end(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + tile.begin - tile.end
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inplace_add(x: torch.Tensor, y: torch.Tensor) -> None:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(out.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_arange_add(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        offsets = hl.arange(m)
        out[tile_n, :] = x[tile_n, :] + offsets[None, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inner_loop_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel with an outer grid loop and an inner device loop."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    v_view = v_in.reshape([-1, n_dim, head_dim])
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


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_reduce_non_pow2(x: torch.Tensor) -> torch.Tensor:
    """Softmax over a non-power-of-2 reduction dim.

    Uses amax + exp + sum which forces explicit index/mask generation,
    exercising the RDIM_SIZE code path.
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row = x[tile_n, :]
        max_val = torch.amax(row, dim=-1, keepdim=True)
        exp_val = torch.exp(row - max_val)
        out[tile_n, :] = exp_val / torch.sum(exp_val, dim=-1, keepdim=True)
    return out


@onlyBackends(["triton", "pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallas(TestCase):
    def test_add_1d(self) -> None:
        args = (torch.randn(1024, device=DEVICE), torch.randn(1024, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=256)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_large(self) -> None:
        args = (torch.randn(4096, device=DEVICE), torch.randn(4096, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=512)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_does_not_donate_inputs(self) -> None:
        """Verify that read-only inputs are not donated by the kernel.

        Regression test: the codegen used to mark all tensor args as outputs
        (including read-only inputs rebound by broadcast_tensors), causing JAX
        to donate their buffers.  Any external reference to the inputs would
        then fail with "Buffer has been deleted or donated".
        """
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        # Save copies to compare against after the kernel call.
        x_copy = x.clone()
        y_copy = y.clone()
        code, result = code_and_output(add_kernel, (x, y), block_size=256)
        torch.testing.assert_close(result, x_copy + y_copy)
        # Only the output (index 2) should be in _output_indices, not inputs.
        self.assertIn("_output_indices=[2]", code)
        # The original inputs must still be accessible (not donated).
        torch.testing.assert_close(x, x_copy)
        torch.testing.assert_close(y, y_copy)

    def test_add_2d(self) -> None:
        args = (
            torch.randn(64, 512, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 512, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(pallas_add_2d, args, block_sizes=[8, 512])
        torch.testing.assert_close(result, args[0] + args[1])

    def test_arange(self) -> None:
        x = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
        offsets = torch.arange(64, device=DEVICE, dtype=torch.int32).float()
        code, result = code_and_output(pallas_arange_add, (x,), block_size=8)
        torch.testing.assert_close(result, x + offsets[None, :])
        self.assertIn("jnp.arange", code)

    def test_inplace_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        expected = x + y
        # Use block_size=1024 so grid=1; with grid>1 the full-array
        # access pattern causes inplace mutations to accumulate.
        code, result = code_and_output(pallas_inplace_add, (x, y), block_size=1024)
        # x should be mutated in place
        torch.testing.assert_close(x, expected)

    def test_pointwise_mul(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(pallas_mul, args, block_size=256)
        x, y = args
        torch.testing.assert_close(out, x * y)

    def test_pointwise_relu(self) -> None:
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_relu, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.relu(x))

    def test_pointwise_sin(self) -> None:
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_sin, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.sin(x))

    def test_pointwise_sigmoid(self) -> None:
        # float16 is not supported by TPU Pallas Mosaic lowering
        # ("Not implemented: offset not aligned to sublanes")
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_sigmoid, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.sigmoid(x), rtol=1e-5, atol=1e-5)

    def test_pointwise_chain(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(pallas_pointwise_chain, args, block_size=256)
        x, y = args
        expected = torch.sigmoid(torch.sin(torch.relu(x * y)))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_args(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            3,
            1.25,
        )
        code, out = code_and_output(pallas_affine_scalar_args, args, block_size=256)
        x, scale, bias = args
        torch.testing.assert_close(out, x * scale + bias, rtol=1e-5, atol=1e-5)

    def test_sum_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_sum_reduction, (x,), block_size=16)
        self.assertIn("jnp.sum", code)
        torch.testing.assert_close(result, x.sum(-1), rtol=1e-4, atol=1e-4)

    def test_max_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_max_reduction, (x,), block_size=16)
        self.assertIn("jnp.max", code)
        torch.testing.assert_close(result, torch.amax(x, dim=-1), rtol=1e-4, atol=1e-4)

    def test_tile_begin_end(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        from helion.runtime.config import Config

        bound = pallas_tile_begin_end.bind((x,))
        code = bound.to_triton_code(Config(block_size=256))
        self.assertIn("pl.program_id", code)

    def test_dynamic_scalar_no_recompile(self) -> None:
        """Verify that changing dynamic scalar values does not trigger recompilation."""
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        pallas_affine_scalar_args.reset()

        # First call - triggers compilation
        result1 = pallas_affine_scalar_args(x, 3, 1.25)
        self.assertEqual(len(pallas_affine_scalar_args._bound_kernels), 1)

        # Second call with different scalar values - should NOT recompile
        result2 = pallas_affine_scalar_args(x, 5, 2.5)
        self.assertEqual(len(pallas_affine_scalar_args._bound_kernels), 1)

        # Verify correctness
        torch.testing.assert_close(result1, x * 3 + 1.25, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result2, x * 5 + 2.5, rtol=1e-5, atol=1e-5)

    def test_inner_loop_add(self) -> None:
        """Test kernel with outer grid loop and inner device loop."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add, args, block_sizes=[8, 128]
        )
        self.assertIn("for ", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_matmul_broadcast_bias(self) -> None:
        """Regression: bias [1, N] must not iterate grid dim 0.

        Without the dim_size <= block_size guard in _compute_block_spec_info,
        the bias BlockSpec maps grid dim i to its 1-row axis, causing an
        out-of-bounds DMA read that crashes the TPU.
        """
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_broadcast_bias, (x, y, bias), block_sizes=[64, 128, 128]
        )
        expected = (x.float() @ y.float() + bias.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # The bias block_spec_info must have None for dim 0 (not a grid index).
        self.assertIn("(None, 1)", code)

    def test_bmm(self) -> None:
        """Test BMM with default config — exercises size_matches fix.

        Without the size_matches fix, adjust_block_size_constraints cannot
        match block dims to tensor dims (4 block dims vs 3D tensors), causing
        the default config to pick block sizes that violate TPU alignment.
        """
        a = torch.randn(4, 128, 256, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4, 256, 128, device=DEVICE, dtype=torch.bfloat16)
        # No explicit block_sizes — uses default_config() which runs
        # adjust_block_size_constraints and depends on size_matches.
        code, result = code_and_output(pallas_bmm, (a, b))
        expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_emit_pipeline_codegen(self) -> None:
        """Test that pallas_loop_type='emit_pipeline' generates correct emit_pipeline code."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("pl.BlockSpec", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_fori_loop_codegen(self) -> None:
        """Test that pallas_loop_type='fori_loop' generates correct fori_loop code."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn("pltpu.make_async_copy", code)
        self.assertNotIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_attention_default_fp32(self) -> None:
        """Test attention with default (for-loop) inner loop."""
        query = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        key = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        val = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        args = (query, key, val)
        _code, result = code_and_output(pallas_attention, args, block_sizes=[1, 32, 32])
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_attention_emit_pipeline_correctness(self) -> None:
        """Test emit_pipeline attention with loop-carried state."""
        query = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        key = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        val = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        _code, result = code_and_output(
            pallas_attention,
            (query, key, val),
            block_sizes=[4, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_attention_fori_loop_correctness(self) -> None:
        """Test fori_loop attention with loop-carried state."""
        query = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        key = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        val = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        args = (query, key, val)
        code, result = code_and_output(
            pallas_attention,
            args,
            block_sizes=[4, 128, 128],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn("pltpu.make_async_copy", code)
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_attention_emit_pipeline_non_divisible(self) -> None:
        """Test emit_pipeline with seq_kv not divisible by block_k.

        Uses _explicit_indices to pass iteration index into body for
        proper mask computation on partial tiles.
        """
        # seq=384, block_k=256 -> 2 tiles, last is partial (128/256)
        query = torch.randn(1, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        key = torch.randn(1, 2, 384, 128, dtype=torch.float32, device=DEVICE)
        val = torch.randn(1, 2, 384, 128, dtype=torch.float32, device=DEVICE)
        _code, result = code_and_output(
            pallas_attention,
            (query, key, val),
            block_sizes=[2, 128, 256],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("_explicit_indices=True", _code)
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_emit_pipeline_loop_order(self) -> None:
        """Test emit_pipeline with loop_order reordering.

        Without the fix, program_id mapping uses logical grid_block_ids
        order instead of pid_info order (which reflects loop_order),
        producing wrong results.
        """
        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_broadcast_bias,
            (x, y, bias),
            block_sizes=[16, 128, 64],
            loop_orders=[[1, 0]],
            pallas_loop_type="emit_pipeline",
        )
        expected = (x.float() @ y.float() + bias.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @xfailIfPallas("RDIM_SIZE rounded to next power of 2 causes shape mismatch")
    def test_reduce_non_pow2(self) -> None:
        """Reduction over non-power-of-2 dim should use exact size, not rounded."""
        x = torch.randn(128, 1000, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_reduce_non_pow2, (x,), block_size=128)
        expected = torch.nn.functional.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
