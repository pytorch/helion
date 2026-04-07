from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


def _skip_unless_sm90(reason: str = "Requires SM90+ for TLX DSM"):
    """Skip test if GPU compute capability < 9.0."""
    if not torch.cuda.is_available():
        return unittest.skip("No CUDA device available")
    cc = torch.cuda.get_device_capability()
    if cc[0] < 9:
        return unittest.skip(reason)
    return lambda fn: fn


@onlyBackends(["triton"])
class TestMultiCTAReduction(TestCase):
    """Tests for automatic multi-CTA DSM reduction lowering."""

    def test_single_cta_no_tlx(self) -> None:
        """num_reduction_ctas=1 must produce identical code to default (no TLX)."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[8],
                num_reduction_ctas=1,
            ),
        )
        def layer_norm_fwd(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                mean_val = torch.sum(acc, dim=-1) / n
                centered = acc - mean_val[:, None]
                var_val = torch.sum(centered * centered, dim=-1) / n
                rstd_val = torch.rsqrt(var_val + eps)
                out[tile_m, :] = (centered * rstd_val[:, None]).to(x.dtype)
            return out

        x = torch.randn(128, 1024, device=DEVICE, dtype=torch.float16)
        code, result = code_and_output(layer_norm_fwd, (x,))

        # Must NOT contain any TLX calls
        self.assertNotIn("tlx.", code)
        self.assertNotIn("cluster_cta_rank", code)
        self.assertNotIn("async_remote_shmem_store", code)
        self.assertNotIn("ctas_per_cga", code)

        # Correctness check
        expected = torch.nn.functional.layer_norm(x.float(), [1024]).to(x.dtype)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_persistent_reduction_multi_cta_codegen(self) -> None:
        """num_reduction_ctas=2 on persistent reduction emits full TLX DSM pattern."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[8],
                num_reduction_ctas=2,
            ),
        )
        def layer_norm_fwd(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                mean_val = torch.sum(acc, dim=-1) / n
                centered = acc - mean_val[:, None]
                var_val = torch.sum(centered * centered, dim=-1) / n
                rstd_val = torch.rsqrt(var_val + eps)
                out[tile_m, :] = (centered * rstd_val[:, None]).to(x.dtype)
            return out

        x = torch.randn(128, 1024, device=DEVICE, dtype=torch.float16)
        bound = layer_norm_fwd.bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[8], num_reduction_ctas=2))

        # Assert TLX import
        self.assertIn("import triton.language.extra.tlx as tlx", code)

        # Assert TLX prologue
        self.assertIn("tlx.cluster_cta_rank()", code)
        self.assertIn("tlx.alloc_barriers(", code)
        self.assertIn("tlx.barrier_expect_bytes(", code)
        self.assertIn("tlx.cluster_barrier()", code)

        # Assert DSM reduction pattern (for each of 2 reductions: mean, var)
        self.assertIn("tlx.local_alloc(", code)
        self.assertIn("tlx.local_store(", code)
        self.assertIn("tlx.async_remote_shmem_store(", code)
        self.assertIn("tlx.barrier_wait(", code)
        self.assertIn("tlx.local_view(", code)
        self.assertIn("tlx.local_load(", code)

        # Assert grid includes cluster dimension
        self.assertIn("ctas_per_cga=", code)

        # Golden file comparison for full code structure
        self.assertExpectedJournal(code)

    @_skip_unless_sm90()
    def test_persistent_reduction_multi_cta_correctness(self) -> None:
        """Multi-CTA layernorm fwd produces correct results."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[8],
                num_reduction_ctas=2,
            ),
        )
        def layer_norm_fwd(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                mean_val = torch.sum(acc, dim=-1) / n
                centered = acc - mean_val[:, None]
                var_val = torch.sum(centered * centered, dim=-1) / n
                rstd_val = torch.rsqrt(var_val + eps)
                out[tile_m, :] = (centered * rstd_val[:, None]).to(x.dtype)
            return out

        for shape in [(128, 1024), (256, 2048), (64, 4096)]:
            x = torch.randn(shape, device=DEVICE, dtype=torch.float16)
            code, result = code_and_output(layer_norm_fwd, (x,))
            expected = torch.nn.functional.layer_norm(
                x.float(), [shape[1]]
            ).to(x.dtype)
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_looped_reduction_multi_cta_codegen(self) -> None:
        """Looped reduction with num_reduction_ctas=2 emits double-buffered DSM."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[8],
                reduction_loops=[1024],
                num_reduction_ctas=2,
            ),
        )
        def rms_norm_fwd(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                ss = torch.sum(acc * acc, dim=-1) / n
                rstd = torch.rsqrt(ss + eps)
                out[tile_m, :] = (acc * rstd[:, None]).to(x.dtype)
            return out

        x = torch.randn(128, 4096, device=DEVICE, dtype=torch.float16)
        bound = rms_norm_fwd.bind((x,))
        code = bound.to_triton_code(helion.Config(
            block_sizes=[8], reduction_loops=[1024], num_reduction_ctas=2,
        ))

        # Assert double-buffered barrier pattern
        self.assertIn("tlx.alloc_barriers(", code)
        self.assertIn("tlx.barrier_arrive(", code)

        # Assert phase tracking
        self.assertIn("phase", code)

        # Assert DSM core pattern
        self.assertIn("tlx.local_alloc(", code)
        self.assertIn("tlx.async_remote_shmem_store(", code)
        self.assertIn("tlx.barrier_wait(", code)

        # Golden file comparison
        self.assertExpectedJournal(code)

    @_skip_unless_sm90()
    def test_looped_reduction_multi_cta_correctness(self) -> None:
        """Multi-CTA RMSNorm with looped reduction produces correct results."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[4],
                reduction_loops=[2048],
                num_reduction_ctas=2,
            ),
        )
        def rms_norm_fwd(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                ss = torch.sum(acc * acc, dim=-1) / n
                rstd = torch.rsqrt(ss + eps)
                out[tile_m, :] = (acc * rstd[:, None]).to(x.dtype)
            return out

        for shape in [(64, 4096), (128, 8192)]:
            x = torch.randn(shape, device=DEVICE, dtype=torch.float16)
            code, result = code_and_output(rms_norm_fwd, (x,))
            rms = torch.sqrt(
                torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5
            )
            expected = (x.float() / rms).to(x.dtype)
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_invalid_num_reduction_ctas_rejected(self) -> None:
        """Non-power-of-2 num_reduction_ctas raises InvalidConfig."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(block_sizes=[8], num_reduction_ctas=3),
        )
        def simple_reduce(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m], device=x.device, dtype=torch.float32)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.sum(
                    x[tile_m, :].to(torch.float32), dim=-1
                )
            return out

        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig, r"num_reduction_ctas"
        ):
            code_and_output(simple_reduce, (x,))

    def test_barrier_count_matches_reduction_count(self) -> None:
        """Layernorm fwd has 2 reductions -> 2 barriers allocated."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(block_sizes=[8], num_reduction_ctas=2),
        )
        def layer_norm_fwd(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                mean_val = torch.sum(acc, dim=-1) / n
                centered = acc - mean_val[:, None]
                var_val = torch.sum(centered * centered, dim=-1) / n
                rstd_val = torch.rsqrt(var_val + eps)
                out[tile_m, :] = (centered * rstd_val[:, None]).to(x.dtype)
            return out

        x = torch.randn(128, 1024, device=DEVICE, dtype=torch.float16)
        bound = layer_norm_fwd.bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[8], num_reduction_ctas=2))

        # 1 barrier per reduction dimension (reductions on same dim share a barrier)
        self.assertIn("num_barriers=1", code)

    def test_rmsnorm_single_barrier(self) -> None:
        """RMSNorm has 1 reduction -> 1 barrier."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(block_sizes=[8], num_reduction_ctas=2),
        )
        def rms_norm(
            x: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                ss = torch.sum(acc * acc, dim=-1) / n
                rstd = torch.rsqrt(ss + eps)
                out[tile_m, :] = (acc * rstd[:, None]).to(x.dtype)
            return out

        x = torch.randn(128, 1024, device=DEVICE, dtype=torch.float16)
        bound = rms_norm.bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[8], num_reduction_ctas=2))

        self.assertIn("num_barriers=1", code)

    def test_grid_has_cluster_dimension(self) -> None:
        """Grid becomes 2D with cluster axis when num_reduction_ctas > 1."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(block_sizes=[8], num_reduction_ctas=4),
        )
        def simple_reduce(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m], device=x.device, dtype=torch.float32)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.sum(
                    x[tile_m, :].to(torch.float32), dim=-1
                )
            return out

        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.float16)
        bound = simple_reduce.bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[8], num_reduction_ctas=4))

        # Grid should include ctas_per_cga for cluster
        self.assertIn("ctas_per_cga=(1, 4, 1)", code)
        self.assertExpectedJournal(code)

    def test_dsm_helper_function_emitted(self) -> None:
        """The _dsm_reduce helper is emitted as a separate @triton.jit function."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(block_sizes=[8], num_reduction_ctas=2),
        )
        def simple_reduce(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m], device=x.device, dtype=torch.float32)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.sum(
                    x[tile_m, :].to(torch.float32), dim=-1
                )
            return out

        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        bound = simple_reduce.bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[8], num_reduction_ctas=2))

        # Helper function should be defined before the main kernel
        self.assertIn("def _dsm_reduce(", code)
        self.assertIn("@triton.jit", code)

    def test_config_num_reduction_ctas_property(self) -> None:
        """Config.num_reduction_ctas property works correctly."""
        config = helion.Config(num_reduction_ctas=4)
        self.assertEqual(config.num_reduction_ctas, 4)

        config_default = helion.Config()
        self.assertEqual(config_default.num_reduction_ctas, 1)

        config_explicit_one = helion.Config(num_reduction_ctas=1)
        self.assertEqual(config_explicit_one.num_reduction_ctas, 1)


if __name__ == "__main__":
    unittest.main()
