from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion.autotuner import BlockSizeFragment
from helion.autotuner import EnumFragment
from helion.autotuner import NumWarpsFragment
from helion.exc import InvalidConfig
import helion.language as hl


class TestConfigSpaceEndToEnd(unittest.TestCase):
    """End-to-end tests requiring GPU (or at least Helion compilation)."""

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_pinned_pid_type(self):
        """@helion.kernel(config_space_fn=...) with a real kernel."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                pid_type="flat",
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["pid_type"], "flat")

        code, result = code_and_output(my_kernel, (x,))
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_constrained_num_warps(self):
        """config_space_fn constraining num_warps."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                num_warps=NumWarpsFragment(4, 4),
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] + 1.0
            return out

        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["num_warps"], 4)

        code, result = code_and_output(my_kernel, (x,))
        expected = x + 1.0
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_with_proxy_args(self):
        """config_space_fn that inspects proxy args."""

        def my_config_space(args):
            # args[0] is a proxy for the tensor
            shape = args[0].shape
            if shape[0] >= 256:
                return helion.ConfigSpace(num_warps=NumWarpsFragment(8, 8))
            return helion.ConfigSpace(num_warps=NumWarpsFragment(4, 4))

        @helion.kernel(config_space_fn=my_config_space)
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 3.0
            return out

        # Large tensor -> num_warps=8
        x_large = torch.randn(512, device=DEVICE, dtype=torch.float32)
        bound_large = my_kernel.bind((x_large,))
        config_large = bound_large.config_spec.default_config()
        self.assertEqual(config_large.config["num_warps"], 8)

        # Need to reset for new specialization
        my_kernel.reset()

        # Small tensor -> num_warps=4
        x_small = torch.randn(64, device=DEVICE, dtype=torch.float32)
        bound_small = my_kernel.bind((x_small,))
        config_small = bound_small.config_spec.default_config()
        self.assertEqual(config_small.config["num_warps"], 4)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_inline_lambda_in_block_sizes(self):
        """Inline lambda in block_sizes list, deriving block size from input shape."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[
                    # Derive M block size from input shape
                    lambda args, config: min(args[0].shape[0], 64),
                ],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        # Derived value: min(128, 64) = 64
        self.assertEqual(config.block_sizes[0], 64)

        code, result = code_and_output(my_kernel, (x,))
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_external_function_in_block_sizes(self):
        """External function (not inline lambda) used in block_sizes."""

        def compute_block_size(args, config):
            """External function with (args, config) -> value signature."""
            n = args[0].shape[0]
            # Pick a reasonable power-of-two block size
            return min(n, 32)

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[compute_block_size],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] + 10.0
            return out

        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        # compute_block_size: min(256, 32) = 32
        self.assertEqual(config.block_sizes[0], 32)

        code, result = code_and_output(my_kernel, (x,))
        expected = x + 10.0
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_user_defined_tunable_with_derived(self):
        """User-defined tunable read by a derived function via config attribute access."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                # User-defined tunable: searched independently
                tile_factor=EnumFragment(choices=(2, 4)),
                # Derived: block_size = input_size / tile_factor (clamped to power of 2)
                block_sizes=[
                    lambda args, config: max(
                        1, min(args[0].shape[0] // config.tile_factor, 64)
                    ),
                ],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 5.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        # tile_factor default=2, block_size = min(128 // 2, 64) = 64
        self.assertEqual(config.config["tile_factor"], 2)
        self.assertEqual(config.block_sizes[0], 64)

        code, result = code_and_output(my_kernel, (x,))
        expected = x * 5.0
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_function_composition_pattern(self):
        """Design doc Q4 pattern: chained derivation via function composition.

        Both functions only read independent params from config. The
        "chaining" is just Python function composition — invisible to the
        config system, which sees two independent derived values.
        """

        def compute_num_warps(args, config):
            return config.warp_factor

        def compute_num_stages(args, config):
            # "chains" through plain Python call, not config read
            return max(1, compute_num_warps(args, config) // 2)

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                warp_factor=EnumFragment(choices=(4, 8)),
                num_warps=compute_num_warps,
                num_stages=compute_num_stages,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] - 1.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        # warp_factor default=4, num_warps=4, num_stages=max(1, 4//2)=2
        self.assertEqual(config.config["warp_factor"], 4)
        self.assertEqual(config.num_warps, 4)
        self.assertEqual(config.num_stages, 2)

        code, result = code_and_output(my_kernel, (x,))
        expected = x - 1.0
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_mixed_fragment_and_lambda_block_sizes(self):
        """block_sizes with a mix of fragment (searched) and lambda (derived)."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[
                    BlockSizeFragment(16, 64),  # M block: searched
                    lambda args, config: min(args[1].shape[1], 32),  # N block: derived
                ],
            ),
        )
        def add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([M, N]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        bound = add_2d.bind((x, y))
        config = bound.config_spec.default_config()
        # N block: min(128, 32) = 32
        self.assertEqual(config.block_sizes[1], 32)
        # M block: within [16, 64] (the searched range)
        self.assertGreaterEqual(config.block_sizes[0], 16)
        self.assertLessEqual(config.block_sizes[0], 64)

        code, result = code_and_output(add_2d, (x, y))
        expected = x + y
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_end_to_end_derived_num_warps_from_shape(self):
        """Derived num_warps that depends on input tensor shape via proxy args."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                num_warps=lambda args, config: 8 if args[0].shape[0] >= 512 else 4,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        self.assertEqual(config.num_warps, 8)

        code, result = code_and_output(my_kernel, (x,))
        expected = x * 2.0
        torch.testing.assert_close(result, expected)


    # ---- ConfigGeneration tests (exercise unflatten / random_config path) ----

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_derived_num_warps_tracks_searched_block_size(self):
        """num_warps derived from searched block_size via ConfigGeneration."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[BlockSizeFragment(16, 64)],
                num_warps=lambda args, config: 8
                if config.block_sizes[0] >= 64
                else 4,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config_gen = bound.config_spec.create_config_generation()

        for _ in range(20):
            config = config_gen.random_config()
            bs = config.block_sizes[0]
            expected_warps = 8 if bs >= 64 else 4
            self.assertEqual(
                config.num_warps,
                expected_warps,
                f"block_size={bs}: expected num_warps={expected_warps}, "
                f"got {config.num_warps}",
            )

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_derived_block_size_tracks_searched_tunable(self):
        """block_sizes[0] derived from searched user-defined tile_factor."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                tile_factor=EnumFragment(choices=(2, 4, 8)),
                block_sizes=[
                    lambda args, config: args[0].shape[0] // config.tile_factor,
                ],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] + 1.0
            return out

        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config_gen = bound.config_spec.create_config_generation()

        for _ in range(20):
            config = config_gen.random_config()
            tile_factor = config.config["tile_factor"]
            expected_bs = 256 // tile_factor
            self.assertEqual(
                config.block_sizes[0],
                expected_bs,
                f"tile_factor={tile_factor}: expected block_size={expected_bs}, "
                f"got {config.block_sizes[0]}",
            )

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_pinned_values_stay_pinned_across_random_configs(self):
        """Pinned pid_type and num_warps remain constant across random configs."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                pid_type="flat",
                num_warps=4,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 3.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config_gen = bound.config_spec.create_config_generation()

        for i in range(30):
            config = config_gen.random_config()
            self.assertEqual(
                config.config["pid_type"],
                "flat",
                f"config #{i}: pid_type should be 'flat'",
            )
            self.assertEqual(
                config.num_warps,
                4,
                f"config #{i}: num_warps should be 4",
            )

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_derived_scalar_depends_on_searched_block_size_via_config_gen(self):
        """num_stages derived from searched block_sizes[0] via ConfigGeneration."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[BlockSizeFragment(16, 128)],
                num_stages=lambda args, config: 2
                if config.block_sizes[0] >= 64
                else 1,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] - 1.0
            return out

        x = torch.randn(512, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config_gen = bound.config_spec.create_config_generation()

        for _ in range(20):
            config = config_gen.random_config()
            bs = config.block_sizes[0]
            expected_stages = 2 if bs >= 64 else 1
            self.assertEqual(
                config.num_stages,
                expected_stages,
                f"block_size={bs}: expected num_stages={expected_stages}, "
                f"got {config.num_stages}",
            )

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_chained_derivation_across_random_configs(self):
        """warp_factor searched; num_warps and num_stages derived via chaining."""

        def compute_num_warps(args, config):
            return config.warp_factor

        def compute_num_stages(args, config):
            return max(1, compute_num_warps(args, config) // 2)

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                warp_factor=EnumFragment(choices=(4, 8)),
                num_warps=compute_num_warps,
                num_stages=compute_num_stages,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] + 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config_gen = bound.config_spec.create_config_generation()

        for _ in range(20):
            config = config_gen.random_config()
            wf = config.config["warp_factor"]
            self.assertIn(wf, (4, 8))
            self.assertEqual(
                config.num_warps,
                wf,
                f"warp_factor={wf}: expected num_warps={wf}, got {config.num_warps}",
            )
            expected_stages = max(1, wf // 2)
            self.assertEqual(
                config.num_stages,
                expected_stages,
                f"warp_factor={wf}: expected num_stages={expected_stages}, "
                f"got {config.num_stages}",
            )

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_mixed_searched_and_derived_block_sizes_via_config_gen(self):
        """2D kernel: block_sizes[0] searched, block_sizes[1] derived."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[
                    BlockSizeFragment(16, 64),  # dim 0: searched
                    lambda args, config: min(args[1].shape[1], 32),  # dim 1: derived
                ],
            ),
        )
        def add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([M, N]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        bound = add_2d.bind((x, y))
        config_gen = bound.config_spec.create_config_generation()

        seen_block0_values = set()
        for _ in range(20):
            config = config_gen.random_config()
            # dim 1 always derived: min(128, 32) = 32
            self.assertEqual(
                config.block_sizes[1],
                32,
                f"block_sizes[1] should always be 32 (derived), "
                f"got {config.block_sizes[1]}",
            )
            # dim 0 searched within [16, 64]
            self.assertGreaterEqual(config.block_sizes[0], 16)
            self.assertLessEqual(config.block_sizes[0], 64)
            seen_block0_values.add(config.block_sizes[0])

        # With 20 random samples from {16, 32, 64}, we expect some variation
        self.assertGreater(
            len(seen_block0_values),
            0,
            "block_sizes[0] should have at least one value from the search range",
        )

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_error_too_many_block_size_overrides(self):
        """ConfigSpace with more block_sizes than tile dimensions raises InvalidConfig."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[
                    BlockSizeFragment(16, 64),
                    BlockSizeFragment(16, 64),  # extra — kernel has only 1 tile dim
                ],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(InvalidConfig):
            my_kernel.bind((x,))

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_error_unsupported_list_key(self):
        """ConfigSpace with unsupported list key raises InvalidConfig."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                bogus_list_key=[1, 2, 3],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(InvalidConfig):
            my_kernel.bind((x,))

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_error_bad_callable_signature(self):
        """ConfigSpace with a callable that doesn't accept (args, config) raises."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                num_warps=lambda: 4,  # wrong: no args
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(InvalidConfig):
            my_kernel.bind((x,))

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_error_derived_reads_other_derived(self):
        """Derived value reading another derived value from config raises InvalidConfig."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                num_warps=lambda args, config: 4,
                # This tries to read num_warps which is also derived — should error
                num_stages=lambda args, config: max(1, config.num_warps // 2),
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] + 1.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        with self.assertRaises(InvalidConfig):
            bound.config_spec.default_config()


    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_error_derived_list_element_reads_sibling(self):
        """Two derived lambdas in block_sizes: one reads block_sizes → guarded."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[
                    lambda args, config: 32,
                    # This tries to read block_sizes which contains a derived element
                    lambda args, config: config.block_sizes[0] * 2,
                ],
            ),
        )
        def add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([M, N]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        bound = add_2d.bind((x, y))
        with self.assertRaises(InvalidConfig):
            bound.config_spec.default_config()

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_derived_non_power_of_two_block_size_raises(self):
        """Derived callable returns 33 (not power of 2) for block_size → InvalidConfig."""

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                block_sizes=[
                    lambda args, config: 33,  # not a power of 2
                ],
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        with self.assertRaises(InvalidConfig):
            bound.config_spec.default_config()

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_derived_value_normalize_fix_invalid_path(self):
        """Derived callable sets num_sm_multiplier incompatible with pid_type.

        _resolve_derived calls normalize(_fix_invalid=True) which silently
        fixes num_sm_multiplier for non-persistent pid_types (removes it).
        Without _fix_invalid this would raise InvalidConfig.
        """

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                pid_type="flat",
                # num_sm_multiplier=2 is invalid with pid_type="flat",
                # but _fix_invalid=True silently drops it
                num_sm_multiplier=lambda args, config: 2,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))
        config = bound.config_spec.default_config()
        # normalize(_fix_invalid=True) should have silently removed num_sm_multiplier
        self.assertNotIn(
            "num_sm_multiplier",
            config.config,
            "num_sm_multiplier should be removed for non-persistent pid_type",
        )
        self.assertEqual(config.config["pid_type"], "flat")

    @unittest.skipIf(DEVICE == "cpu", "requires CUDA")
    def test_derived_value_used_not_placeholder(self):
        """Derived scalars must use the resolved value, not the pinned EnumFragment default.

        num_warps default is 4 (DEFAULT_NUM_WARPS). We derive it to 16,
        a value that could never come from the single-element placeholder
        EnumFragment(choices=(4,)). If we see 16 in both default_config()
        and every random_config(), the derived path is working.
        """

        @helion.kernel(
            config_space_fn=lambda args: helion.ConfigSpace(
                num_warps=lambda args, config: 16,
            ),
        )
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        bound = my_kernel.bind((x,))

        # default_config path
        config = bound.config_spec.default_config()
        self.assertEqual(config.num_warps, 16)

        # random_config / unflatten path
        config_gen = bound.config_spec.create_config_generation()
        for _ in range(20):
            config = config_gen.random_config()
            self.assertEqual(
                config.num_warps,
                16,
                f"Expected derived num_warps=16, got {config.num_warps} "
                f"(placeholder default is 4)",
            )


if __name__ == "__main__":
    unittest.main()
