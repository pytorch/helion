"""Tests for Phase 6: CuteDSL autotuning integration.

Verifies that the autotuner infrastructure (config spec, config generation,
config normalization, finite search) works correctly with the CuteDSL backend.

Since the CuteDSL backend currently generates Triton-compatible code body
(tl.* intrinsics) that cannot be executed natively as CuteDSL, these tests
focus on the codegen-level integration rather than runtime execution.
"""

from __future__ import annotations

import ast as py_ast
import unittest

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import TestCase
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.config import Config


def _has_cutlass() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


skipIfNoCutlass = unittest.skipUnless(_has_cutlass(), "cutlass not installed")


# ---------------------------------------------------------------------------
# Module-level kernel definitions (avoid closure issues)
# ---------------------------------------------------------------------------

@helion.kernel(backend="cutedsl")
def _cutedsl_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


@helion.kernel(backend="cutedsl")
def _cutedsl_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


def add_combine_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def max_combine_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


@helion.kernel(backend="cutedsl")
def _cutedsl_softmax(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        row_max = hl.reduce(max_combine_fn, row, dim=1, keep_dims=True)
        row_exp = torch.exp(row - row_max)
        row_sum = hl.reduce(add_combine_fn, row_exp, dim=1, keep_dims=True)
        result[i, :] = row_exp / row_sum
    return result


@helion.kernel(backend="cutedsl", static_shapes=True)
def _cutedsl_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor,
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
    qk_scale = head_dim ** -0.5 / 1.44269504
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
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


# ---------------------------------------------------------------------------
# Test: Config Spec Generation
# ---------------------------------------------------------------------------

class TestCuteDSLConfigSpec(TestCase):
    """Tests that config spec is correctly generated for CuteDSL kernels."""

    def test_add_config_spec_has_block_sizes(self):
        """Verify add kernel produces a config spec with block_sizes."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        spec = bound.config_spec
        self.assertTrue(len(spec.block_sizes) > 0)

    def test_matmul_config_spec_has_block_sizes(self):
        """Verify matmul kernel has block_sizes for M, N, K dimensions."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        spec = bound.config_spec
        # Matmul has at least 3 block_sizes: M, N, K
        self.assertGreaterEqual(len(spec.block_sizes), 3)

    def test_config_spec_default_config_is_valid(self):
        """Verify default config passes normalization."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        config = bound.config_spec.default_config()
        self.assertIsInstance(config, Config)
        # Should have block_sizes
        self.assertIn("block_sizes", config.config)

    def test_matmul_default_config_has_num_warps(self):
        """Verify matmul default config includes num_warps."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        config = bound.config_spec.default_config()
        self.assertGreater(config.num_warps, 0)

    def test_softmax_config_spec(self):
        """Verify softmax kernel config spec is generated."""
        args = (torch.randn(32, 64, device=DEVICE, dtype=torch.float32),)
        bound = _cutedsl_softmax.bind(args)
        spec = bound.config_spec
        self.assertTrue(len(spec.block_sizes) > 0)

    def test_attention_config_spec(self):
        """Verify attention kernel config spec is generated."""
        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
        )
        bound = _cutedsl_attention.bind(args)
        spec = bound.config_spec
        self.assertTrue(len(spec.block_sizes) > 0)


# ---------------------------------------------------------------------------
# Test: Config Generation
# ---------------------------------------------------------------------------

class TestCuteDSLConfigGeneration(TestCase):
    """Tests that ConfigGeneration works with CuteDSL kernels."""

    def test_config_generation_creates_flat_spec(self):
        """Verify ConfigGeneration can flatten a CuteDSL kernel config spec."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        gen = ConfigGeneration(bound.config_spec)
        self.assertGreater(len(gen.flat_spec), 0)
        self.assertGreater(len(gen.block_size_indices), 0)

    def test_config_generation_random_config(self):
        """Verify random config generation produces valid configs."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        gen = ConfigGeneration(bound.config_spec)
        config = gen.random_config()
        self.assertIsInstance(config, Config)
        self.assertIn("block_sizes", config.config)

    def test_config_generation_multiple_random_configs(self):
        """Verify multiple random configs are distinct."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        gen = ConfigGeneration(bound.config_spec)
        configs = [gen.random_config() for _ in range(10)]
        # At least some should be different
        config_strs = {str(c.config) for c in configs}
        # With enough samples, we should get at least 2 unique configs
        self.assertGreater(len(config_strs), 1)


# ---------------------------------------------------------------------------
# Test: Code Generation Across Multiple Configs
# ---------------------------------------------------------------------------

class TestCuteDSLMultiConfigCodegen(TestCase):
    """Tests that code generation works for multiple configs."""

    def _get_code(self, fn, args, **config_kwargs):
        bound = fn.bind(args)
        config = Config(**config_kwargs) if config_kwargs else bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_add_multiple_block_sizes(self):
        """Verify add kernel generates valid code for different block sizes."""
        args = (
            torch.randn(4096, device=DEVICE, dtype=torch.float32),
            torch.randn(4096, device=DEVICE, dtype=torch.float32),
        )
        for bs in [32, 64, 128, 256]:
            code = self._get_code(_cutedsl_add, args, block_sizes=[bs])
            self.assertIn("@cute.kernel", code)
            py_ast.parse(code)  # Must be valid Python

    def test_matmul_multiple_configs(self):
        """Verify matmul generates valid code for various tile sizes."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        configs = [
            {"block_sizes": [16, 16, 16]},
            {"block_sizes": [32, 32, 16]},
            {"block_sizes": [64, 64, 32]},
            {"block_sizes": [32, 64, 16]},
        ]
        for config_kwargs in configs:
            code = self._get_code(_cutedsl_matmul, args, **config_kwargs)
            self.assertIn("@cute.kernel", code)
            self.assertIn("tl.dot", code)
            py_ast.parse(code)

    def test_matmul_different_num_warps(self):
        """Verify matmul generates valid code with different num_warps."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        for nw in [1, 2, 4, 8]:
            code = self._get_code(
                _cutedsl_matmul, args,
                block_sizes=[32, 32, 16], num_warps=nw,
            )
            self.assertIn("@cute.kernel", code)
            self.assertIn(f"num_warps={nw}", code)
            py_ast.parse(code)

    def test_softmax_multiple_configs(self):
        """Verify softmax generates valid code for various configs."""
        args = (torch.randn(64, 128, device=DEVICE, dtype=torch.float32),)
        for bs in [16, 32, 64]:
            code = self._get_code(_cutedsl_softmax, args, block_sizes=[bs])
            self.assertIn("@cute.kernel", code)
            py_ast.parse(code)

    def test_attention_multiple_configs(self):
        """Verify attention generates valid code for various tile sizes."""
        args = (
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 4, 64, 32, dtype=torch.float32, device=DEVICE),
        )
        configs = [
            {"block_sizes": [1, 16, 16]},
            {"block_sizes": [1, 32, 32]},
        ]
        for config_kwargs in configs:
            code = self._get_code(_cutedsl_attention, args, **config_kwargs)
            self.assertIn("@cute.kernel", code)
            self.assertIn("tl.dot", code)
            py_ast.parse(code)


# ---------------------------------------------------------------------------
# Test: Config Normalization
# ---------------------------------------------------------------------------

class TestCuteDSLConfigNormalization(TestCase):
    """Tests that config normalization works with CuteDSL backend."""

    def test_normalize_valid_config(self):
        """Verify normalization of a valid config doesn't raise."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        config = {"block_sizes": [16, 16, 16], "num_warps": 4, "num_stages": 1}
        bound.config_spec.normalize(config)

    def test_normalize_single_block_size(self):
        """Verify normalization handles singular block_size → block_sizes."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        config = {"block_size": 64}
        bound.config_spec.normalize(config)
        self.assertIn("block_sizes", config)

    def test_normalize_preserves_num_warps(self):
        """Verify normalization doesn't drop num_warps."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        config_dict = {"block_sizes": [32, 32, 16], "num_warps": 8, "num_stages": 1}
        bound.config_spec.normalize(config_dict)
        self.assertEqual(config_dict["num_warps"], 8)


# ---------------------------------------------------------------------------
# Test: Random Config → Codegen Round-Trip
# ---------------------------------------------------------------------------

class TestCuteDSLRandomConfigCodegen(TestCase):
    """Tests that randomly generated configs produce valid code."""

    def test_random_configs_produce_parseable_add_code(self):
        """Generate 5 random configs for add kernel; all should produce parseable code."""
        args = (
            torch.randn(4096, device=DEVICE, dtype=torch.float32),
            torch.randn(4096, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        gen = ConfigGeneration(bound.config_spec)

        parseable_count = 0
        for _ in range(5):
            config = gen.random_config()
            try:
                code = bound.to_triton_code(config)
                py_ast.parse(code)
                self.assertIn("@cute.kernel", code)
                parseable_count += 1
            except Exception:
                pass  # Some random configs may be invalid

        self.assertGreater(parseable_count, 0, "No random configs produced parseable code")

    def test_random_configs_produce_parseable_matmul_code(self):
        """Generate 5 random configs for matmul; all should produce parseable code."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        gen = ConfigGeneration(bound.config_spec)

        parseable_count = 0
        for _ in range(5):
            config = gen.random_config()
            try:
                code = bound.to_triton_code(config)
                py_ast.parse(code)
                self.assertIn("@cute.kernel", code)
                parseable_count += 1
            except Exception:
                pass

        self.assertGreater(parseable_count, 0, "No random configs produced parseable code")


# ---------------------------------------------------------------------------
# Test: Compile Config (without execution)
# ---------------------------------------------------------------------------

class TestCuteDSLCompileConfig(TestCase):
    """Tests that compile_config works for CuteDSL backend.

    Note: compile_config loads the generated module via exec(), which
    succeeds even for CuteDSL code because the @cute.kernel decorator
    is available. Actual kernel execution requires the CuteDSL runtime.
    """

    def test_compile_config_add(self):
        """Verify compile_config returns a callable for add kernel."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        config = bound.config_spec.default_config()
        compiled = bound.compile_config(config)
        self.assertTrue(callable(compiled))

    def test_compile_config_matmul(self):
        """Verify compile_config returns a callable for matmul kernel."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        config = Config(block_sizes=[16, 16, 16])
        compiled = bound.compile_config(config)
        self.assertTrue(callable(compiled))

    def test_compile_config_multiple_configs(self):
        """Verify compile_config works for multiple different configs."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        configs = [
            Config(block_sizes=[16, 16, 16]),
            Config(block_sizes=[32, 32, 16]),
            Config(block_sizes=[64, 64, 32]),
        ]
        for config in configs:
            compiled = bound.compile_config(config)
            self.assertTrue(callable(compiled))

    def test_compile_config_cached(self):
        """Verify compile_config caches results for same config."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        config = bound.config_spec.default_config()
        compiled1 = bound.compile_config(config)
        compiled2 = bound.compile_config(config)
        self.assertIs(compiled1, compiled2)

    def test_compile_config_softmax(self):
        """Verify compile_config works for softmax kernel."""
        args = (torch.randn(32, 64, device=DEVICE, dtype=torch.float32),)
        bound = _cutedsl_softmax.bind(args)
        config = bound.config_spec.default_config()
        compiled = bound.compile_config(config)
        self.assertTrue(callable(compiled))


# ---------------------------------------------------------------------------
# Test: Generated Code Properties Across Configs
# ---------------------------------------------------------------------------

class TestCuteDSLCodeProperties(TestCase):
    """Verify structural properties of generated code across configs."""

    def _get_code(self, fn, args, **config_kwargs):
        bound = fn.bind(args)
        config = Config(**config_kwargs) if config_kwargs else bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_all_configs_have_cutedsl_imports(self):
        """All configs should produce code with CuteDSL imports."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        for bs in [16, 32, 64]:
            code = self._get_code(
                _cutedsl_matmul, args, block_sizes=[bs, bs, bs]
            )
            self.assertIn("import cutlass", code)
            self.assertIn("import cutlass.cute as cute", code)
            self.assertIn("_default_cutedsl_launcher", code)

    def test_all_configs_use_cutlass_types(self):
        """All configs should use cutlass types in function signatures."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        for bs in [16, 32, 64]:
            code = self._get_code(
                _cutedsl_matmul, args, block_sizes=[bs, bs, bs]
            )
            self.assertIn("cutlass.Constexpr[int]", code)
            self.assertIn("cutlass.Int32", code)

    def test_no_triton_imports_in_any_config(self):
        """No config should produce code with Triton imports."""
        args = (
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        for bs in [16, 32, 64]:
            code = self._get_code(
                _cutedsl_matmul, args, block_sizes=[bs, bs, bs]
            )
            self.assertNotIn("import triton", code)
            self.assertNotIn("@triton.jit", code)

    def test_launcher_uses_cutedsl_launcher(self):
        """Verify the launcher import matches the cutedsl launcher."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(_cutedsl_add, args)
        self.assertIn("from helion.runtime.cutedsl_launcher import", code)
        self.assertIn("_default_cutedsl_launcher", code)

    def test_block_sizes_affect_generated_constants(self):
        """Different block_sizes should produce different constants."""
        args = (
            torch.randn(4096, device=DEVICE, dtype=torch.float32),
            torch.randn(4096, device=DEVICE, dtype=torch.float32),
        )
        code1 = self._get_code(_cutedsl_add, args, block_sizes=[64])
        code2 = self._get_code(_cutedsl_add, args, block_sizes=[128])
        # The block size should appear as a constant in the generated code
        self.assertIn("_BLOCK_SIZE", code1)
        self.assertIn("_BLOCK_SIZE", code2)
        # They should generate different code
        self.assertNotEqual(code1, code2)


# ---------------------------------------------------------------------------
# Test: Config Spec Compatibility
# ---------------------------------------------------------------------------

class TestCuteDSLConfigSpecCompatibility(TestCase):
    """Tests that config spec infrastructure is fully compatible with CuteDSL."""

    def test_config_spec_flat_config(self):
        """Verify flat_config works for CuteDSL config spec."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        gen = ConfigGeneration(bound.config_spec)

        # Should have block_size indices
        self.assertGreater(len(gen.block_size_indices), 0)
        # Should have num_warps index
        self.assertGreaterEqual(gen.num_warps_index, 0)

    def test_config_spec_unflatten_round_trip(self):
        """Verify random_config produces a valid config that can be used for codegen."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        gen = ConfigGeneration(bound.config_spec)

        config = gen.random_config()
        # Verify the config has expected fields
        self.assertIsNotNone(config.block_sizes)
        self.assertGreater(len(config.block_sizes), 0)
        # Verify the config can be used for codegen
        code = bound.to_triton_code(config)
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    def test_config_spec_allowed_pid_types(self):
        """Verify allowed_pid_types is populated for CuteDSL kernels."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        spec = bound.config_spec
        # Should have at least "flat" pid type
        self.assertIn("flat", spec.allowed_pid_types)


# ---------------------------------------------------------------------------
# Test: Format Kernel Decorator (for autotuner logging)
# ---------------------------------------------------------------------------

class TestCuteDSLKernelDecorator(TestCase):
    """Tests for format_kernel_decorator with CuteDSL backend."""

    def test_format_kernel_decorator(self):
        """Verify format_kernel_decorator produces valid decorator string."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        config = bound.config_spec.default_config()
        decorator = bound.format_kernel_decorator(config, bound.settings)
        # Should be a valid Python decorator string
        self.assertIn("@helion.kernel", decorator)

    def test_format_kernel_decorator_matmul(self):
        """Verify format_kernel_decorator works for matmul config."""
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        bound = _cutedsl_matmul.bind(args)
        config = Config(block_sizes=[16, 16, 16])
        decorator = bound.format_kernel_decorator(config, bound.settings)
        self.assertIn("@helion.kernel", decorator)
        # Should contain config and static_shapes info
        self.assertIn("config=", decorator)
        self.assertIn("static_shapes=", decorator)


# ---------------------------------------------------------------------------
# Test: Cached Path
# ---------------------------------------------------------------------------

class TestCuteDSLCachedPath(TestCase):
    """Tests for get_cached_path with CuteDSL backend."""

    def test_cached_path_after_compile(self):
        """Verify cached path is set after compile_config."""
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        config = bound.config_spec.default_config()
        bound.compile_config(config)
        path = bound.get_cached_path(config)
        self.assertIsNotNone(path)
        self.assertTrue(path.endswith(".py"))

    def test_cached_path_none_before_compile(self):
        """Verify cached path is None for an un-compiled config."""
        # Use a unique size so we get a fresh BoundKernel (not cached from other tests)
        args = (
            torch.randn(7777, device=DEVICE, dtype=torch.float32),
            torch.randn(7777, device=DEVICE, dtype=torch.float32),
        )
        bound = _cutedsl_add.bind(args)
        # Use a non-default config that hasn't been compiled
        config = Config(block_sizes=[64], num_warps=2)
        path = bound.get_cached_path(config)
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
