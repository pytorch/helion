from __future__ import annotations

from pathlib import Path
import random
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipUnlessAMDCDNA
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.language import loops

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def _get_examples_matmul():
    """Lazy accessor to avoid CUDA init during pytest-xdist collection."""
    return import_path(examples_dir / "matmul.py").matmul


@onlyBackends(["triton"])
class TestAMDCDNA(TestCase):
    @skipUnlessAMDCDNA("Test requires AMD CDNA GPU (MI200/MI300 series)")
    def test_amd_cdna_tunables_in_kernel(self) -> None:
        """Test that AMD CDNA tunables are supported."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                waves_per_eu=2,
                matrix_instr_nonkdim=16,
            ),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] + y[tile]
            return result

        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(add_kernel, (x, y))
        expected = x + y

        torch.testing.assert_close(result, expected)

        # Verify that the tunables are passed to Triton
        self.assertIn("waves_per_eu=2", code)
        self.assertIn("matrix_instr_nonkdim=16", code)

    def test_amd_tunables_error_when_not_supported(self) -> None:
        """Test that specifying AMD tunables on non-AMD hardware raises an error."""
        device = torch.device("cuda")
        settings = helion.Settings(backend="triton")

        with (
            patch(
                "helion._compat.is_hip",
                return_value=False,
            ),
            patch(
                "helion.autotuner.config_spec.supports_amd_cdna_tunables",
                return_value=False,
            ),
            patch(
                "helion._compat.supports_amd_cdna_tunables",
                return_value=False,
            ),
        ):
            env = CompileEnvironment(device, settings)

            config = helion.Config(waves_per_eu=2)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                rf"Unsupported config keys for backend '{env.backend_name}': \['waves_per_eu'\]",
            ):
                env.config_spec.normalize(config)

            config = helion.Config(matrix_instr_nonkdim=16)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                rf"Unsupported config keys for backend '{env.backend_name}': \['matrix_instr_nonkdim'\]",
            ):
                env.config_spec.normalize(config)

    def test_rdna_supports_waves_per_eu_only(self) -> None:
        """Test that RDNA hardware supports waves_per_eu but not matrix_instr_nonkdim."""
        device = torch.device("cuda")
        settings = helion.Settings(backend="triton")

        with (
            patch(
                "helion._compat.is_hip",
                return_value=True,
            ),
            patch(
                "helion.autotuner.config_spec.supports_amd_cdna_tunables",
                return_value=False,
            ),
            patch(
                "helion._compat.supports_amd_cdna_tunables",
                return_value=False,
            ),
        ):
            env = CompileEnvironment(device, settings)

            # waves_per_eu should be accepted on RDNA
            config = helion.Config(waves_per_eu=2)
            env.config_spec.normalize(config)

            # matrix_instr_nonkdim should still raise on RDNA (CDNA-only)
            config = helion.Config(matrix_instr_nonkdim=16)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                rf"Unsupported config keys for backend '{env.backend_name}': \['matrix_instr_nonkdim'\]",
            ):
                env.config_spec.normalize(config)

    def test_rdna_tunable_fragments(self) -> None:
        """Test that RDNA hardware only includes waves_per_eu in tunable fragments."""
        device = torch.device("cuda")
        settings = helion.Settings(backend="triton")

        with (
            patch(
                "helion._compat.is_hip",
                return_value=True,
            ),
            patch(
                "helion.autotuner.config_spec.supports_amd_cdna_tunables",
                return_value=False,
            ),
            patch(
                "helion._compat.supports_amd_cdna_tunables",
                return_value=False,
            ),
        ):
            env = CompileEnvironment(device, settings)
            fragments = env.config_spec.backend_tunable_fragments
            self.assertIn("waves_per_eu", fragments)
            self.assertNotIn("matrix_instr_nonkdim", fragments)


@onlyBackends(["triton"])
@skipUnlessAMDCDNA("requires AMD CDNA autotuning search space")
class TestAMDCDNAAutotuner(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        random.seed(112)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        kernel = _get_examples_matmul()
        kernel.reset()
        self.addCleanup(kernel.reset)
        spec = kernel.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    def test_config_fragment1(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        kernel = basic_kernels.add
        kernel.reset()
        self.addCleanup(kernel.reset)
        spec = kernel.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    def test_config_warp_specialize_unroll(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        kernel = basic_kernels.add
        kernel.reset()
        self.addCleanup(kernel.reset)
        spec = kernel.bind(args).config_spec
        overrides = {"range_unroll_factors": [4], "range_warp_specializes": ([True])}
        configs = ConfigGeneration(spec, overrides=overrides).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))
