from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import helion
from helion import Config
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTensorDescriptor
import helion.language as hl
from helion.runtime.settings import _get_backend

_orig_cudnn_fp32_precision: str = "none"
_orig_float32_matmul_precision: str = "none"


def setUpModule() -> None:
    global _orig_cudnn_fp32_precision, _orig_float32_matmul_precision
    cudnn_conv = torch.backends.cudnn.conv  # pyrefly: ignore[missing-attribute]
    _orig_cudnn_fp32_precision = cudnn_conv.fp32_precision
    _orig_float32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    cudnn_conv.fp32_precision = "tf32"


def tearDownModule() -> None:
    cudnn_conv = torch.backends.cudnn.conv  # pyrefly: ignore[missing-attribute]
    torch.set_float32_matmul_precision(_orig_float32_matmul_precision)
    cudnn_conv.fp32_precision = _orig_cudnn_fp32_precision


examples_dir = Path(__file__).parent.parent / "examples"


def _get_examples_matmul():
    """Lazy accessor to avoid CUDA init during pytest-xdist collection."""
    return import_path(examples_dir / "matmul.py").matmul


@helion.kernel
def matmul_with_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel
def matmul_without_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def matmul_fp8_bf16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@onlyBackends(["triton", "cute"])
class TestMatmul(RefEagerTestBase, TestCase):
    def test_matmul0(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_without_addmm,
            args,
            block_sizes=[16, 16, 16],
            num_threads=[16, 16, 1],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul1(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            _get_examples_matmul(),
            args,
            block_sizes=[16, 16, 16],
            loop_order=[1, 0],
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul3(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_addmm,
            args,
            block_sizes=[16, 16, 16],
            num_threads=[16, 16, 1],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_transposed_rhs_fallback(self):
        @helion.kernel
        def matmul_transposed_rhs(
            x: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            n, k2 = weight.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, weight.dtype),
                device=x.device,
            )
            for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
                out[tile_m, tile_n] = torch.mm(
                    x[tile_m, tile_k],
                    weight[tile_n, tile_k].T,
                )
            return out

        args = (
            torch.randn([4, 8], device=DEVICE, dtype=torch.float32),
            torch.randn([6, 8], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_transposed_rhs,
            args,
            block_sizes=[1, 1, 8],
            num_threads=[1, 1, 8],
        )
        expected = args[0] @ args[1].T
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-3)
        if _get_backend() == "cute":
            self.assertNotIn("cute.gemm", code)
            self.assertNotIn("permute_smem", code)

    def test_addmm_transposed_rhs_fallback(self):
        @helion.kernel
        def addmm_transposed_rhs(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            n, k2 = weight.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty([m, n], dtype=bias.dtype, device=x.device)
            for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
                out[tile_m, tile_n] = torch.addmm(
                    bias[tile_m, tile_n],
                    x[tile_m, tile_k],
                    weight[tile_n, tile_k].T,
                )
            return out

        args = (
            torch.randn([4, 8], device=DEVICE, dtype=torch.float32),
            torch.randn([6, 8], device=DEVICE, dtype=torch.float32),
            torch.randn([4, 6], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            addmm_transposed_rhs,
            args,
            block_sizes=[1, 1, 8],
            num_threads=[1, 1, 8],
        )
        expected = torch.addmm(args[2], args[0], args[1].T)
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-3)
        if _get_backend() == "cute":
            self.assertNotIn("cute.gemm", code)
            self.assertNotIn("permute_smem", code)

    def test_tcgen05_fused_colvec_scale_emits_scalar_read(self):
        """End-to-end: a per-row column-vector scale spelled explicitly as
        ``scale_a.unsqueeze(1).expand(m, n)`` (a full ``(M, N)`` view with
        trailing stride 0) classifies as ``("broadcast", 2)`` and the
        generated cute epilogue reads it as a SCALAR per subtile
        (``aux_loaded = ttr_aux_grouped[(0, 0, 0, subtile)]``) instead of a
        redundant N-wide vector ``.load()`` — matching CUTLASS's
        ``sa = tTR_gSA[(0,0,0,subtile)]``. Numerics are checked on every
        backend; the scalar-read codegen pin is cute-only.
        """
        if _get_backend() != "cute":
            self.skipTest("colvec scalar-read codegen is cute-specific")

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_colvec_scale(
            x: torch.Tensor, y: torch.Tensor, scale_a: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                # Explicit per-row column-vector broadcast over N.
                out[tile_m, tile_n] = (acc * scale_a[tile_m, tile_n]).to(x.dtype)
            return out

        x = torch.randn(128, 128, dtype=torch.bfloat16, device=DEVICE)
        y = torch.randn(128, 128, dtype=torch.bfloat16, device=DEVICE)
        scale_a_vec = torch.randn(128, dtype=torch.float32, device=DEVICE)
        # ``(M,) -> (M, N)`` with trailing stride 0: the colvec form.
        scale_a = scale_a_vec.unsqueeze(1).expand(128, 128)
        self.assertEqual(scale_a.stride(), (1, 0))

        code, output = code_and_output(
            cute_matmul_colvec_scale,
            (x, y, scale_a),
            block_sizes=[128, 128, 32],
            pid_type="persistent_interleaved",
        )

        # Codegen pin: the colvec aux is read as a scalar (T2R index
        # (0, 0, 0) plus the subtile) with NO trailing ``.load()`` — that
        # is the whole point of the ``("broadcast", 2)`` classification.
        scalar_read = (
            "tcgen05_aux_loaded_0 = tcgen05_tTR_gAux_grouped_0"
            "[0, 0, 0, cutlass.Int32(_tcgen05_subtile)]"
        )
        self.assertIn(scalar_read, code)
        # It must NOT fall back to the N-wide vector load the rowvec /
        # exact forms use.
        self.assertNotIn(
            "tcgen05_aux_loaded_0 = tcgen05_tTR_gAux_grouped_0.load()", code
        )

        # Numerics: the scalar broadcast must compute acc * scale_a[m].
        ref = (x.float() @ y.float()) * scale_a_vec.unsqueeze(1)
        torch.testing.assert_close(output.float(), ref, atol=1.0, rtol=1e-1)

    @skipIfNotTriton("block_ptr is triton-only")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfRefEager("checks generated Triton code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_matmul_block_ptr(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            _get_examples_matmul(),
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
            indexing="block_ptr",
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertIn("tl.make_block_ptr", code)
        self.assertNotIn("boundary_check=None", code)
        block_ptr_load_lines = [
            line for line in code.splitlines() if "tl.load(tl.make_block_ptr" in line
        ]
        self.assertGreater(len(block_ptr_load_lines), 0)
        for line in block_ptr_load_lines:
            self.assertNotIn("boundary_check=", line)
        block_ptr_store_lines = [
            line for line in code.splitlines() if "tl.store(tl.make_block_ptr" in line
        ]
        self.assertGreater(len(block_ptr_store_lines), 0)
        for line in block_ptr_store_lines:
            self.assertNotIn("boundary_check=", line)

    @skipIfNotTriton("block_ptr is triton-only")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_fp8_matmul_block_ptr_omits_zero_padding(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            torch.randn([128, 256], device=DEVICE, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .T.contiguous()
            .T,
        )
        code, output = code_and_output(
            matmul_fp8_bf16,
            args,
            block_sizes=[128, 256, 128],
            indexing="block_ptr",
        )
        expected = (args[0].to(torch.float32) @ args[1].to(torch.float32)).to(
            torch.bfloat16
        )
        torch.testing.assert_close(output, expected, atol=1.0, rtol=1e-1)
        self.assertIn("tl.make_block_ptr", code)
        self.assertNotIn("padding_option='zero'", code)

    @skipIfNotTriton("block_ptr is triton-only")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_fp8_matmul_block_ptr_uses_pointer_for_padded_edges(self):
        args = (
            torch.randn([130, 128], device=DEVICE, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            torch.randn([128, 256], device=DEVICE, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .T.contiguous()
            .T,
        )
        code, output = code_and_output(
            matmul_fp8_bf16,
            args,
            block_sizes=[128, 256, 128],
            indexing="block_ptr",
        )
        expected = (args[0].to(torch.float32) @ args[1].to(torch.float32)).to(
            torch.bfloat16
        )
        torch.testing.assert_close(output, expected, atol=1.0, rtol=1e-1)
        self.assertIn("tl.make_block_ptr", code)
        self.assertIn("other=0.0", code)
        self.assertNotIn("padding_option='zero'", code)

    @skipIfNotTriton("tensor_descriptor is triton-only")
    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @skipIfRefEager("to_triton_code is not supported in ref eager mode")
    def test_matmul_tensor_descriptor(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        config = Config(
            block_sizes=[16, 16, 16],
            l2_grouping=4,
            indexing="tensor_descriptor",
        )
        # Note TensorDescriptor doesn't run on older cards
        code = _get_examples_matmul().bind(args).to_triton_code(config)
        self.assertIn("make_tensor_descriptor", code)

    @skipIfNotTriton("indexing='pointer' is triton-only")
    def test_matmul_static_shapes0(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
            indexing="pointer",
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_static_shapes1(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            num_threads=[16, 16, 1],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_static_shapes2(self):
        args = (
            torch.randn([128, 127], device=DEVICE, dtype=torch.float32),
            torch.randn([127, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            num_threads=[16, 16, 1],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_static_shapes3(self):
        args = (
            torch.randn([127, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 127], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            num_threads=[16, 16, 1],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_packed_int4_block_size_constexpr(self):
        torch.manual_seed(0)
        M = N = K = 32

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def matmul_bf16_packed_int4(
            A: torch.Tensor, B_packed: torch.Tensor, C: torch.Tensor
        ) -> torch.Tensor:
            M0, K0 = A.shape
            _, N0 = B_packed.shape

            block_n = hl.register_block_size(N0)
            block_k = hl.register_block_size(K0)

            for tile_m in hl.tile(M0):
                for tile_n in hl.tile(N0, block_size=block_n):
                    acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)

                    for tile_k in hl.tile(K0, block_size=block_k):
                        tile_k_begin = tile_k.begin
                        b_tile = B_packed[
                            tile_k_begin // 2 : tile_k_begin // 2 + block_k // 2,
                            tile_n,
                        ]
                        shift = hl.full((1,), 4, dtype=torch.int8)
                        b_lo = (b_tile << shift) >> shift
                        b_hi = b_tile >> shift
                        stacked = torch.stack(
                            (b_lo.to(A.dtype), b_hi.to(A.dtype)), dim=2
                        )
                        stacked = stacked.permute(0, 2, 1)
                        b_block = stacked.reshape([block_k, block_n])
                        acc = hl.dot(A[tile_m, tile_k], b_block, acc=acc)

                    C[tile_m, tile_n] = acc

            return C

        A = torch.randn((M, K), dtype=torch.bfloat16, device=DEVICE)
        B_packed = torch.randint(0, 16, (K // 2, N), dtype=torch.int8, device=DEVICE)
        C = torch.zeros((M, N), dtype=torch.float32, device=DEVICE)

        matmul_bf16_packed_int4(A, B_packed, C)
        torch.accelerator.synchronize()

        self.assertTrue(torch.isfinite(C).all())
        self.assertFalse(torch.allclose(C, torch.zeros_like(C)))

    def test_matmul_split_k(self):
        @helion.kernel(dot_precision="ieee")
        def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n, outer_k in hl.tile([m, n, k]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for inner_k in hl.tile(outer_k.begin, outer_k.end):
                    acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
                hl.atomic_add(out, [tile_m, tile_n], acc)
            return out

        x = torch.randn([32, 2000], device=DEVICE, dtype=torch.float32)
        y = torch.randn([2000, 32], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            matmul_split_k,
            (x, y),
            block_sizes=[16, 16, 256, 32],
            indexing="pointer",
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)
        if _get_backend() == "cute":
            self.assertIn("cute.arch.atomic_add", code)
        else:
            self.assertIn("tl.atomic_add", code)

    @skipIfRefEager("config_spec is not supported in ref eager mode")
    def test_matmul_config_reuse_with_unit_dim(self):
        torch.manual_seed(0)
        big_args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        big_bound = matmul_with_addmm.bind(big_args)
        big_spec = big_bound.config_spec
        self.assertEqual(len(big_spec.block_sizes), 3)
        big_config = big_spec.default_config()

        small_args = (
            torch.randn([1, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        small_bound = matmul_with_addmm.bind(small_args)
        small_spec = small_bound.config_spec
        self.assertEqual(len(small_spec.block_sizes), 3)

        # Previously raised when reusing configs tuned on larger shapes.
        small_bound.set_config(big_config)
        result = small_bound(*small_args)
        expected = small_args[0] @ small_args[1]
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)

    def test_matmul_packed_rhs(self):
        @helion.kernel(static_shapes=False)
        def matmul_with_packed_b(
            A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
        ) -> None:
            M, K = A.shape
            _, N = B.shape

            block_size_k = hl.register_block_size(K // 2)

            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=A.dtype)

                for tile_k in hl.tile(K // 2, block_size=block_size_k):
                    lhs = A[
                        tile_m,
                        tile_k.begin * 2 : tile_k.begin * 2 + tile_k.block_size * 2,
                    ]
                    packed = B[tile_k, tile_n]
                    rhs = torch.stack([packed, packed], dim=1).reshape(
                        tile_k.block_size * 2, tile_n.block_size
                    )
                    acc = torch.addmm(acc, lhs, rhs)

                C[tile_m, tile_n] = acc

        M, K, N = 32, 70, 32
        A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        B = torch.randn(K // 2, N, device=DEVICE, dtype=torch.float32)
        C = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
        code, _ = code_and_output(matmul_with_packed_b, (A, B, C))
        B_unpacked = torch.stack([B, B], dim=1).reshape(K, N)
        expected = A @ B_unpacked
        torch.testing.assert_close(C, expected, atol=5e-2, rtol=1e-3)

    def test_addmm_under_autocast(self):
        """Test torch.addmm with float32 accumulator under active autocast.

        In mixed-precision training:
        - autocast(bf16) is active from the model's forward pass
        - x is bf16 (autocast output), weights are fp32 (nn.Linear params)
        - x_tile = x.to(weight.dtype) casts to fp32
        - addmm(f32_acc, f32_x_tile, f32_weight) under autocast may
          incorrectly return bf16 during Helion's type propagation,
          because autocast leaks into propagate_types
        """

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def matmul_with_cast(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            n, k2 = weight.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    x_tile = x[tile_m, tile_k].to(weight.dtype)
                    acc = torch.addmm(acc, x_tile, weight[tile_n, tile_k].T)
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        # x is bf16 (from autocast), weight is fp32 (nn.Linear parameter)
        x = torch.randn(128, 64, device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)

        # Call the kernel under autocast, simulating the model's forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            code, result = code_and_output(matmul_with_cast, (x, weight))

        expected = (x.float() @ weight.T).to(x.dtype)
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
