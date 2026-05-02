from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipIfMTIA
import helion.language as hl


@helion.kernel
def _matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@onlyBackends(["triton", "cute"])
class TestDotRequirements(RefEagerTestDisabled, TestCase):
    @patch.object(_compat, "_min_dot_size", lambda *args: (2, 8, 16))
    def test_hl_dot_sets_min_size(self) -> None:
        @helion.kernel
        def k_small(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += hl.dot(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        m, k, n = 32, 4, 16
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        spec = k_small.bind(args).config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [2, 8, 16])

    @patch.object(_compat, "_min_dot_size", lambda *args: (2, 8, 16))
    def test_matmul_sets_min_size(self) -> None:
        m, k, n = 32, 4, 16
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        spec = _matmul_kernel.bind(args).config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [2, 8, 16])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_matmul_constrains_search_space(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [128, 8, 16])
        # tile_k upper bound was previously hardcoded to 16; the cute tcgen05
        # path now allows multiples of 16 up to min(128, static_k) so the
        # autotuner can pack more cute.gemm instructions per K iteration.
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 128, 64])
        default_block_sizes = spec.default_config().config["block_sizes"]
        self.assertGreaterEqual(default_block_sizes[2], 16)
        self.assertLessEqual(default_block_sizes[2], 64)
        self.assertGreaterEqual(default_block_sizes[0], 128)
        self.assertLessEqual(default_block_sizes[0], 256)
        self.assertGreaterEqual(default_block_sizes[1], 8)
        self.assertLessEqual(default_block_sizes[1], 128)
        self.assertEqual(spec.default_config().config["l2_groupings"], [1])
        # cluster_m default is now 1 (cluster_m=2 has runtime issues); the
        # autotuner search space restricts the choice accordingly.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 1)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_equal_dims_keep_default_within_max_bound(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [128, 8, 16])
        # tile_k upper bound is now 128 (the static_k=8192 case; capped at 128
        # to keep AB SMEM staging budget sane).
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 256, 128])
        default_block_sizes = spec.default_config().config["block_sizes"]
        self.assertGreaterEqual(default_block_sizes[2], 16)
        self.assertLessEqual(default_block_sizes[2], 128)
        self.assertGreaterEqual(default_block_sizes[0], 128)
        self.assertLessEqual(default_block_sizes[0], 256)
        self.assertGreaterEqual(default_block_sizes[1], 8)
        self.assertLessEqual(default_block_sizes[1], 256)
        self.assertEqual(spec.default_config().config["l2_groupings"], [1])
        # cluster_m default is now 1 (cluster_m=2 has runtime issues); the
        # autotuner search space restricts the choice accordingly.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 1)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_widened_default_stays_on_tcgen05_path(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
            config = bound.config_spec.default_config()
            code = bound.to_triton_code(config)
        self.assertEqual(config.config["block_sizes"][2], 16)
        self.assertGreaterEqual(config.config["block_sizes"][0], 128)
        self.assertLessEqual(config.config["block_sizes"][0], 256)
        self.assertGreaterEqual(config.config["block_sizes"][1], 8)
        self.assertLessEqual(config.config["block_sizes"][1], 256)
        self.assertIn("make_trivial_tiled_mma", code)
        self.assertIn(f"_BLOCK_SIZE_0 = {config.config['block_sizes'][0]}", code)
        self.assertIn(f"_BLOCK_SIZE_1 = {config.config['block_sizes'][1]}", code)

    @skipIfMTIA("MTIA requires tl.dot initial value stride >= 128 bytes")
    def test_matmul_smaller_than_min_dot_size(self) -> None:
        """Test matmul where K and N are smaller than min_dot_size (16 on CUDA).

        If update_min_block() promotes block sizes beyond the tensor dimensions,
        this will fail with shape mismatches.
        """
        m, k, n = 32, 8, 8
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        _, result = code_and_output(_matmul_kernel, args, block_sizes=[32, 8, 8])
        ref = args[0].float() @ args[1].float()
        torch.testing.assert_close(result, ref, atol=1e-1, rtol=1e-2)

    @skipIfMTIA("MTIA backend does not support 3D dot reshape patterns")
    def test_bmm_constrains_batch_block_to_one(self) -> None:
        """Triton warp-spec only stably supports 2D tl.dot.
        For batched matmul (baddbmm/bmm), the batch dimension block size must
        be constrained to 1 so the codegen an squeeze the 3D operands to 2D
        before emitting tl.dot.

        Without this constraint the autotuner may pick batch block sizes > 1,
        producing a 3D tl.dot that crashes in Triton's LLVM backend with
        "Unsupported DotOp found when converting TritonGPU to LLVM".
        """

        @helion.kernel(static_shapes=True)
        def bmm_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            b, m, k = A.size()
            b, k, n = B.size()
            out = torch.empty(
                [b, m, n],
                device=A.device,
                dtype=torch.promote_types(A.dtype, B.dtype),
            )
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.baddbmm(
                        acc,
                        A[tile_b, tile_m, tile_k],
                        B[tile_b, tile_k, tile_n],
                    )
                out[tile_b, tile_m, tile_n] = acc
            return out

        b, m, k, n = 16, 512, 768, 1024
        args = (
            torch.randn([b, m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([b, k, n], device=DEVICE, dtype=HALF_DTYPE),
        )

        # Use the spec's batch max_size as block_sizes[0], combined with
        # autotuner parameters that trigger a Triton crash when batch > 1.
        # Without the fix, max_size = 16 (full batch dim) and the 3D tl.dot
        # hits "Unsupported DotOp" → RuntimeError: PassManager::run failed.
        # With the fix, max_size = 1 and the codegen squeezes to a 2D tl.dot.
        bound = bmm_kernel.bind(args)
        batch_max = bound.config_spec.block_sizes[0].max_size
        code, result = code_and_output(
            bmm_kernel,
            args,
            block_sizes=[batch_max, 1, 128, 16],
            indexing=["pointer", "pointer", "tensor_descriptor"],
            num_warps=2,
            num_stages=5,
            pid_type="flat",
        )
        expected = torch.bmm(args[0], args[1])
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_validated_autotune_narrowing(self) -> None:
        """``narrow_tcgen05_autotune_to_validated_configs`` consolidates the
        three known tcgen05 limitations into a single config_spec call.

        Pin the resulting state so any future change to the helper has to
        update the test as well: persistent pid types are dropped from the
        autotune search and the cluster_m search is narrowed to ``(1,)``.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        # Persistent pid types miscompute multi-tile silently today; they
        # are dropped from the autotune pid_type search.
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        # cluster_m=2 currently CUDA-launch-fails on B200; the autotune
        # search is narrowed to cluster_m=1.
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        # The validated narrowing leaves cluster_m=2 still accepted as a
        # legal value for an explicit user-supplied helion.Config.
        validation_fragments = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation_fragments["tcgen05_cluster_m"].choices, (1, 2))

    def test_narrow_tcgen05_autotune_to_validated_configs_helper(self) -> None:
        """Direct unit test for the narrowing helper that does not depend
        on the dot-requirements bind path. The helper only manipulates the
        autotune search state on the receiver and is safe to invoke on any
        ``ConfigSpec`` instance."""

        @helion.kernel
        def stub(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        args = (torch.randn([1024], device=DEVICE),)
        spec = stub.bind(args).config_spec
        before_pid = set(spec.allowed_pid_types)
        spec.narrow_tcgen05_autotune_to_validated_configs()
        # Both persistent types are dropped (idempotently if they were
        # already absent).
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        # Other pid types are preserved.
        for pid_type in before_pid - {"persistent_blocked", "persistent_interleaved"}:
            self.assertIn(pid_type, spec.allowed_pid_types)
        # The cluster_m search is now narrowed to (1,).
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        # Calling it twice is idempotent.
        spec.narrow_tcgen05_autotune_to_validated_configs()
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))


@onlyBackends(["pallas"])
class TestDotRequirementsPallas(RefEagerTestDisabled, TestCase):
    def test_tpu_min_dot_size_constrains_matmul(self) -> None:
        """Verify that TPU min_dot_size (8, 128, 128) is applied to matmul block sizes."""
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),
        )
        spec = _matmul_kernel.bind(args).config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [8, 128, 128])


if __name__ == "__main__":
    unittest.main()
