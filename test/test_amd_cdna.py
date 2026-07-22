from __future__ import annotations

from unittest.mock import patch

import torch

import helion
from helion._compiler.backend import TritonBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessAMDCDNA
from helion._testing import skipUnlessMultiXCD
from helion.autotuner.config_fragment import EnumFragment
from helion.exc import InvalidConfig
import helion.language as hl


@onlyBackends(["triton"])
class TestAMDCDNA(TestCase):
    def test_amd_tunable_fragment_choices(self) -> None:
        with (
            patch("helion._compat.is_hip", return_value=True),
            patch("helion._compat.supports_amd_cdna_tunables", return_value=True),
        ):
            fragments = TritonBackend().tunable_fragments()

        self.assertEqual(
            fragments["waves_per_eu"],
            EnumFragment(choices=(0, 1, 2, 3, 4)),
        )
        self.assertEqual(fragments["waves_per_eu"].default(), 0)
        self.assertEqual(
            fragments["matrix_instr_nonkdim"],
            EnumFragment(choices=(0, 16, 32)),
        )

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


@helion.kernel()
def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


# Separate kernel object so the single-XCD test does not reuse the compile cache
# of the multi-XCD tests.
@helion.kernel()
def _add_single_xcd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


# Separate kernel object so num_sm is captured fresh under the mock (the shared
# kernel above would reuse a ConfigSpec built with the real num_sm).
@helion.kernel()
def _add_misalign(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel()
def _add_partial_active(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


class TestXCDRemapConfig(TestCase):
    """xcd_remap config-API tests that do not require AMD hardware."""

    def test_config_roundtrip(self) -> None:
        cfg = helion.Config(block_sizes=[64, 64], xcd_remap=True)
        self.assertTrue(cfg.xcd_remap)
        self.assertEqual(dict(cfg).get("xcd_remap"), True)
        # repr round-trips back to an equal config
        self.assertEqual(eval(repr(cfg)), cfg)

    def test_config_default_false(self) -> None:
        self.assertFalse(helion.Config(block_sizes=[64, 64]).xcd_remap)


@onlyBackends(["triton"])
class TestXCDRemapCodegen(TestCase):
    """xcd_remap codegen + correctness tests; require a multi-XCD AMD CDNA GPU."""

    def _args(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16)
        return x, y

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_flat_remap_codegen_and_correctness(self) -> None:
        x, y = self._args()
        code, result = code_and_output(
            _add, (x, y), block_sizes=[64, 64], xcd_remap=True
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("_NUM_XCDS", code)
        self.assertIn("xcd_pid", code)
        self.assertIn("get_num_xcd", code)

    @skipUnlessAMDCDNA("Requires AMD CDNA GPU")
    def test_baseline_has_no_remap(self) -> None:
        x, y = self._args()
        code, result = code_and_output(_add, (x, y), block_sizes=[64, 64])
        torch.testing.assert_close(result, x + y)
        self.assertNotIn("_NUM_XCDS", code)

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_l2_grouping_composition_order(self) -> None:
        x, y = self._args()
        code, result = code_and_output(
            _add, (x, y), block_sizes=[64, 64], l2_groupings=[4], xcd_remap=True
        )
        torch.testing.assert_close(result, x + y)
        # remap must run before the L2 grouping math consumes the pid
        self.assertIn("_NUM_XCDS", code)
        self.assertLess(code.index("xcd_pid ="), code.index("group_id ="))

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_persistent_blocked_remap(self) -> None:
        # blocked: remap the worker->block assignment (program_id), not per-pid.
        x, y = self._args()
        code, result = code_and_output(
            _add,
            (x, y),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            xcd_remap=True,
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("_NUM_XCDS", code)
        self.assertIn("xcd_id = tl.program_id(0)", code)  # worker-level remap

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_persistent_blocked_overprovisioned_remap(self) -> None:
        # Regression: blocked + xcd_remap must not collapse when the worker grid
        # is overprovisioned (num_sm_multiplier>1 makes grid_size exceed the tile
        # count, so block_size==1 and most workers idle). The contiguous-per-XCD
        # remap would otherwise pile every valid block onto a couple of XCDs and
        # idle the rest (~3-4x slowdown). Remapping over active workers keeps
        # slack workers out of the schedule while preserving per-XCD contiguity.
        x, y = self._args()
        code, result = code_and_output(
            _add,
            (x, y),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            num_sm_multiplier=8,
            xcd_remap=True,
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("xcd_active_workers", code)
        self.assertIn("tl.where(tl.program_id(0) < xcd_active_workers", code)

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_persistent_blocked_partial_active_workers(self) -> None:
        # Regression: the overprovisioned case is only the smallest instance of
        # the load-balance issue.  With T=2049 tiles and W=2048 workers,
        # block_size=2, so only ceil(T / block_size)=1025 workers are active.
        # The XCD remap must use that active-worker domain, not W.
        x = torch.randn(2049 * 16, 16, device=DEVICE, dtype=torch.float16)
        y = torch.randn_like(x)
        with patch("helion.runtime.get_num_sm", lambda *a, **k: 256):
            code, result = code_and_output(
                _add_partial_active,
                (x, y),
                block_sizes=[16, 16],
                pid_type="persistent_blocked",
                num_sm_multiplier=8,
                xcd_remap=True,
            )
        torch.testing.assert_close(result, x + y)
        self.assertIn(
            "xcd_active_workers = tl.cdiv(total_pids, xcd_safe_block_size)", code
        )
        self.assertIn("tl.where(tl.program_id(0) < xcd_active_workers", code)

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_persistent_interleaved_remap(self) -> None:
        # interleaved: remap each virtual_pid inside the grid-stride loop.
        x, y = self._args()
        code, result = code_and_output(
            _add,
            (x, y),
            block_sizes=[64, 64],
            pid_type="persistent_interleaved",
            xcd_remap=True,
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("_NUM_XCDS", code)
        self.assertIn("xcd_id = virtual_pid", code)  # per-virtual-pid remap

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_persistent_interleaved_overprovisioned_remap(self) -> None:
        # Unlike blocked, interleaved remaps the PID over the tile count (not the
        # worker over the worker grid), so an overprovisioned grid
        # (num_sm_multiplier>1, W>>tiles) stays a balanced bijection and needs no
        # fallback. Guard that it remains correct and keeps the remap enabled.
        x, y = self._args()
        code, result = code_and_output(
            _add,
            (x, y),
            block_sizes=[64, 64],
            pid_type="persistent_interleaved",
            num_sm_multiplier=8,
            xcd_remap=True,
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("xcd_id = virtual_pid", code)  # remap still active (pid-space)
        self.assertNotIn("xcd_worker", code)  # no blocked-style worker fallback

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_interleaved_misaligned_grid_disables(self) -> None:
        # If the persistent grid stride is not XCD-aligned (e.g. reserved_sms),
        # interleaved xcd_remap is silently disabled (perf no-op, still correct).
        x, y = self._args()
        with patch("helion.runtime.get_num_sm", lambda *a, **k: 7):
            code, result = code_and_output(
                _add_misalign,
                (x, y),
                block_sizes=[64, 64],
                pid_type="persistent_interleaved",
                xcd_remap=True,
            )
            torch.testing.assert_close(result, x + y)
            self.assertNotIn("_NUM_XCDS", code)

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_aot_standalone_inlines_get_num_xcd(self) -> None:
        # AOT standalone output must inline get_num_xcd (no Helion runtime dep).
        import ast
        from pathlib import Path
        import tempfile

        from helion.autotuner.aot_compile import generate_standalone_file

        x, y = self._args()
        code, _ = code_and_output(
            _add, (x, y), block_sizes=[64, 64], xcd_remap=True, pid_type="flat"
        )
        self.assertIn("helion.runtime.get_num_xcd(", code)
        out = generate_standalone_file("add", [code], "", Path(tempfile.mkdtemp()))
        txt = out.read_text()
        self.assertIn("def _get_num_xcd", txt)
        self.assertNotIn("helion.runtime.get_num_xcd(", txt)
        ast.parse(txt)

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_direct_configspec_derives_num_sm(self) -> None:
        # A direct ConfigSpec (e.g. UserConfigSpec) must derive num_sm from the
        # device like num_xcd -- not default to 1 -- so the interleaved alignment
        # check doesn't falsely disable xcd_remap.
        from helion._compat import device_num_sm
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=TritonBackend())
        # num_sm is device-derived (the CU count), not the old default of 1.
        self.assertEqual(spec.num_sm, device_num_sm())

    @skipUnlessMultiXCD("Requires a multi-XCD AMD CDNA GPU")
    def test_rejects_xyz(self) -> None:
        x, y = self._args()
        with self.assertRaises(InvalidConfig):
            code_and_output(
                _add, (x, y), block_sizes=[64, 64], pid_type="xyz", xcd_remap=True
            )

    @skipUnlessAMDCDNA("Requires AMD CDNA GPU")
    def test_noop_when_single_xcd(self) -> None:
        # On a single-XCD device xcd_remap=True is a no-op: production
        # normalization (strict, no _fix_invalid) downgrades it to False rather
        # than raising, and no remap is emitted.
        x, y = self._args()
        # Patch where it is used: config_spec imports get_num_xcd at module load.
        with patch("helion.autotuner.config_spec.get_num_xcd", lambda *a, **k: 1):
            bound = _add_single_xcd.bind((x, y))
            cfg = helion.Config(block_sizes=[64, 64], xcd_remap=True)
            bound.config_spec.normalize(cfg)  # strict production path; must not raise
            self.assertFalse(cfg.xcd_remap)
            code, result = code_and_output(
                _add_single_xcd, (x, y), block_sizes=[64, 64], xcd_remap=True
            )
            torch.testing.assert_close(result, x + y)
            self.assertNotIn("_NUM_XCDS", code)
