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
from helion.exc import InvalidConfig
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


def _cute_two_matmuls_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    m2, k2 = x2.size()
    _, n2 = y2.size()
    out2 = torch.empty([m2, n2], dtype=x2.dtype, device=x2.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)

    for tile_m2, tile_n2 in hl.tile([m2, n2]):
        acc2 = hl.zeros([tile_m2, tile_n2], dtype=torch.float32)
        for tile_k2 in hl.tile(k2):
            acc2 = torch.addmm(
                acc2,
                x2[tile_m2, tile_k2],
                y2[tile_k2, tile_n2],
            )
        out2[tile_m2, tile_n2] = acc2.to(x2.dtype)
    return out, out2


_cute_two_matmuls_kernel = helion.kernel(_cute_two_matmuls_impl, backend="cute")
_cute_two_matmuls_force_persistent_kernel = helion.kernel(
    _cute_two_matmuls_impl,
    backend="cute",
    autotune_force_persistent=True,
)


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
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)

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
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)

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
        tcgen05 limitations into a single config_spec call.

        Pin the resulting state so any future change to the helper has to
        update the test as well: persistent pid types stay in the autotune
        search for validated static full-tile shapes, the cluster_m search
        is narrowed to ``(1,)``, and the num_epi_warps search is narrowed
        to ``(4,)``.
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
        # Every candidate M/N/K block size divides this static problem, so
        # role-local persistent pid types are admitted back into autotune.
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        # cluster_m=2 currently CUDA-launch-fails on B200; the autotune
        # search is narrowed to cluster_m=1.
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        # num_epi_warps != 4 currently produces wrong output on B200
        # (only 4 epi warps lowers correctly today). The autotune search
        # is narrowed to num_epi_warps=4 so the autotuner does not
        # converge on a wrong-output config.
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        # The validated narrowing leaves cluster_m=2 still accepted as a
        # legal value for an explicit user-supplied helion.Config
        # (CUDA-launch-failure is loud and won't silently miscompute).
        validation_fragments = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation_fragments["tcgen05_cluster_m"].choices, (1, 2))
        # num_epi_warps is the exception: validation is also tightened
        # to (4,) because non-4 values silently produce wrong output, so
        # an explicit user-supplied helion.Config must be rejected
        # rather than allowed to miscompute.
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        self.assertEqual(validation_fragments["tcgen05_num_epi_warps"].choices, (4,))
        # The search view exposes the same narrowed EnumFragment.
        search_fragments = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search_fragments["tcgen05_num_epi_warps"].choices, (4,))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_partial_tile_search_keeps_persistent_pid_types_out(
        self,
    ) -> None:
        """Autotune excludes persistent pid types when the search can sample
        block sizes that produce partial tiles."""

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
            torch.randn([64, 192], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 128, 64])
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_search_keeps_persistent_pid_types_out(
        self,
    ) -> None:
        """Multi-root tcgen05 kernels keep persistent pid types out of autotune
        until the persistent scheduler/grid spans every root case."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_two_matmuls_kernel.bind(args)
        spec = bound.config_spec
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_forced_persistent_raises_invalid_config(
        self,
    ) -> None:
        """Forced-persistent multi-root tcgen05 has no valid pid search choice."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with (
            patch_cute_mma_support(),
            self.assertRaisesRegex(
                InvalidConfig,
                "CuTe tcgen05 multi-root kernels do not support persistent pid types",
            ),
        ):
            _cute_two_matmuls_force_persistent_kernel.bind(args)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_distributed_raises_invalid_config(
        self,
    ) -> None:
        """Distributed mode also makes the pid search persistent-only."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with (
            patch_cute_mma_support(),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.distributed_c10d.get_rank", return_value=0),
            patch("torch.distributed.distributed_c10d.get_world_size", return_value=1),
            patch("torch._logging._internal.dist.get_rank", return_value=0),
            patch(
                "torch.fx.experimental.symbolic_shapes.trace_structured",
                lambda *args, **kwargs: None,
            ),
            patch(
                "helion.runtime.kernel._find_process_group_name",
                return_value="world",
            ),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            self.assertRaisesRegex(
                InvalidConfig,
                "CuTe tcgen05 multi-root kernels do not support persistent pid types",
            ),
        ):
            _cute_two_matmuls_kernel.bind(args)

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
        # The num_epi_warps search is now narrowed to (4,) -- the only
        # currently-correct value on B200 (1 and 2 are directly verified
        # to produce wrong output, 3 is unsafe by extension).
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        # Validation is also tightened for num_epi_warps because the
        # failure mode is silent wrong output.
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        # Calling it twice is idempotent.
        spec.narrow_tcgen05_autotune_to_validated_configs()
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))

        spec = stub.bind(args).config_spec
        spec.allowed_pid_types = (
            "flat",
            "xyz",
            "persistent_blocked",
            "persistent_interleaved",
        )
        spec.narrow_tcgen05_autotune_to_validated_configs(
            allow_persistent_pid_types=True
        )
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))

    def test_restrict_tcgen05_num_epi_warps_search_helper(self) -> None:
        """Direct unit test for ``restrict_tcgen05_num_epi_warps_search``.

        The helper sets the per-instance search-only override and never
        affects the validation view returned by
        ``_tcgen05_optional_fragments(for_search=False)``. The test
        exercises the override on its own (i.e. without going through
        the full ``narrow_tcgen05_autotune_to_validated_configs``
        consolidation) so any future regression to the helper itself is
        caught here directly.
        """

        @helion.kernel
        def stub(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        args = (torch.randn([1024], device=DEVICE),)
        spec = stub.bind(args).config_spec
        # Default state: no override is set, so the search uses the
        # default IntegerFragment range and the validation view keeps
        # the same range.
        self.assertIsNone(spec._tcgen05_num_epi_warps_search_choices)
        default_search = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(default_search["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(default_search["tcgen05_num_epi_warps"].high, 4)

        spec.restrict_tcgen05_num_epi_warps_search((1, 2))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (1, 2))
        narrowed_search = spec._tcgen05_optional_fragments(for_search=True)
        # Narrowing flips the search view to an EnumFragment so the
        # autotuner samples only the listed values.
        self.assertEqual(narrowed_search["tcgen05_num_epi_warps"].choices, (1, 2))
        # Validation view is unaffected by the search-only helper:
        # user-supplied helion.Config values in [1, 4] still round-trip
        # through normalize() unless ``restrict_tcgen05_num_epi_warps_validation``
        # is also called (see ``test_restrict_tcgen05_num_epi_warps_validation_helper``).
        validation = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(validation["tcgen05_num_epi_warps"].high, 4)

        # Empty override raises (a misuse: every search must allow at
        # least one value).
        with self.assertRaises(AssertionError):
            spec.restrict_tcgen05_num_epi_warps_search(())

    def test_restrict_tcgen05_num_epi_warps_validation_helper(self) -> None:
        """Direct unit test for ``restrict_tcgen05_num_epi_warps_validation``.

        Unlike the search-only sibling, this helper tightens what
        ``normalize()`` accepts so user-supplied configs with bad
        values are rejected with ``InvalidConfig`` rather than silently
        accepted. Used by the BF16/FP16 matmul path because non-4
        epi-warp counts produce silent wrong output.
        """

        @helion.kernel
        def stub(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        args = (torch.randn([1024], device=DEVICE),)
        spec = stub.bind(args).config_spec
        # Default state: validation view is the full IntegerFragment.
        self.assertIsNone(spec._tcgen05_num_epi_warps_validation_choices)
        default_validation = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(default_validation["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(default_validation["tcgen05_num_epi_warps"].high, 4)

        spec.restrict_tcgen05_num_epi_warps_validation((4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        narrowed_validation = spec._tcgen05_optional_fragments(for_search=False)
        # Validation view flipped to EnumFragment with the restricted choices.
        self.assertEqual(narrowed_validation["tcgen05_num_epi_warps"].choices, (4,))
        # Search view unaffected by the validation-only helper.
        search = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(search["tcgen05_num_epi_warps"].high, 4)

        # Empty override raises.
        with self.assertRaises(AssertionError):
            spec.restrict_tcgen05_num_epi_warps_validation(())

    @onlyBackends(["cute"])
    def test_cute_tcgen05_num_epi_warps_search_routes_through_flat_fields(
        self,
    ) -> None:
        """End-to-end check that the narrowed num_epi_warps search shows
        up in ``_flat_fields()`` (the autotuner's single source of truth
        for the search space). Without this routing, the narrow_helper
        would only flip the per-instance flag while the autotuner kept
        sampling the full IntegerFragment range.
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
        # cute_tcgen05_search_enabled gates the inclusion of the tcgen05
        # optional fragments in _flat_fields(); enforce_dot_requirements
        # set it during bind, so the narrowed search view should appear.
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        flat_fields = spec._flat_fields()
        self.assertIn("tcgen05_num_epi_warps", flat_fields)
        # The matmul-side narrowing collapses the search to (4,);
        # _flat_fields exposes that as an EnumFragment with a single
        # choice rather than the default IntegerFragment(1, 4, 4).
        self.assertEqual(flat_fields["tcgen05_num_epi_warps"].choices, (4,))
        # cluster_m is similarly narrowed via the same helper.
        self.assertEqual(flat_fields["tcgen05_cluster_m"].choices, (1,))
        self.assertIn("persistent_blocked", flat_fields["pid_type"].choices)
        self.assertIn("persistent_interleaved", flat_fields["pid_type"].choices)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_user_config_num_epi_warps_validation(self) -> None:
        """A user-supplied ``helion.Config(..., tcgen05_num_epi_warps=N)``
        must be rejected by ``normalize()`` for any N != 4 once the
        matmul path has narrowed the validation accept-set to ``(4,)``.
        ``num_epi_warps != 4`` produces silent wrong output today, so
        accepting an explicit user value would silently miscompute —
        the validation tightening is the only loud signal for a user
        bypassing autotune. The legal value 4 must still round-trip.
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
        # Both the search and validation accept-sets are narrowed to (4,).
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        # Non-4 values are rejected: silent wrong output on the
        # current SIMT-store epilogue.
        for n_epi in (1, 2, 3):
            cfg = helion.Config(
                block_sizes=[128, 16, 16],
                tcgen05_num_epi_warps=n_epi,
            )
            with self.assertRaises(InvalidConfig):
                spec.normalize(cfg)
        # The validated value still round-trips unchanged.
        cfg = helion.Config(
            block_sizes=[128, 16, 16],
            tcgen05_num_epi_warps=4,
        )
        spec.normalize(cfg)
        self.assertEqual(cfg.config["tcgen05_num_epi_warps"], 4)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_minimize_normalize_round_trip(self) -> None:
        """The autotuner minimizes the winning config by stripping values
        that match ``default_config()`` (built from the *search* view),
        and the cached/minimized config is later re-expanded by
        ``normalize()``. If the fill-missing branch in normalize() used
        the validation-view default instead of the search-view default,
        the narrowed ``tcgen05_num_epi_warps=4`` choice would silently
        round-trip back to ``4`` only by accident (the validation
        IntegerFragment default also happens to be 4 today). Pin the
        search-view default routing so that, when the search view's
        default later diverges from the validation-view default again
        (e.g. when item 2 lifts the narrowing back to a smaller value),
        normalize() picks up the search-view default instead of the
        validation default.
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
        # The narrowed search default is what default_config() exposes.
        default_cfg = spec.default_config()
        self.assertEqual(default_cfg.config["tcgen05_num_epi_warps"], 4)
        # Simulate the autotuner's minimize step: a winning config of 4
        # matches the search-view default and gets stripped.
        winning = helion.Config(**default_cfg.config)
        minimized = winning.minimize(spec)
        self.assertNotIn("tcgen05_num_epi_warps", minimized.config)
        # Re-normalizing the minimized config (what happens on the next
        # to_code() call after a cache reload) must restore the same
        # effective value via the search-view fill-missing branch.
        spec.normalize(minimized)
        self.assertEqual(minimized.config["tcgen05_num_epi_warps"], 4)
        # Now simulate a future state where the search-view default
        # diverges from the validation-view default. Restrict the
        # search to (2,) (interior of the validation range) and confirm
        # that the fill-missing branch picks up the search-view default
        # of 2 rather than the validation-view default of 4. To do
        # this we must also lift the validation narrowing so that 2 is
        # a legal user-supplied value (otherwise constructing the
        # ``helion.Config(tcgen05_num_epi_warps=2)`` below would be
        # rejected by ``normalize``'s validation pass).
        spec._tcgen05_num_epi_warps_validation_choices = None
        spec.restrict_tcgen05_num_epi_warps_search((2,))
        new_default = spec.default_config()
        self.assertEqual(new_default.config["tcgen05_num_epi_warps"], 2)
        winning_2 = helion.Config(**new_default.config)
        minimized_2 = winning_2.minimize(spec)
        self.assertNotIn("tcgen05_num_epi_warps", minimized_2.config)
        spec.normalize(minimized_2)
        self.assertEqual(minimized_2.config["tcgen05_num_epi_warps"], 2)


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
