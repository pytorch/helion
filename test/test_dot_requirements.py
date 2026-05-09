from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.cute.strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from helion._compiler.cute.strategies import Tcgen05LayoutOverrides
from helion._compiler.cute.strategies import Tcgen05LayoutStrategy
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.strategies import Tcgen05WarpSpec
from helion._compiler.cute.strategies import validate_tcgen05_strategy_invariants
from helion._compiler.cute.tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipIfMTIA
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
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


@helion.kernel(backend="cute")
def _cute_strategy_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute", autotune_force_persistent=True)
def _cute_strategy_matmul_force_persistent_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def _bind_cute_strategy_kernel():
    """Shared bind helper for the G2-A strategy data-model tests.

    The G2-A tests all need the same 256x256 cute_tcgen05-enabled
    ``config_spec``; hoisting the bind avoids repeating the inline
    kernel definition in every test. The size is the smallest tile
    family that activates the cute_tcgen05 search path
    (``enforce_dot_requirements`` requires ``static_m % 64 == 0`` and
    ``min_dot_size`` of 64x8x16 on B200), so 256 squared is more
    than enough; using the canonical 4096³ benchmark shape here is
    wasteful for what is purely a metadata round-trip test.

    For tests that exercise codegen (``to_triton_code()``), use
    ``_bind_cute_strategy_kernel_with_patch`` so the
    ``patch_cute_mma_support`` context stays active across the
    codegen call — ``cute_mma.py`` consults
    ``get_cute_mma_support()`` during codegen, and a bare bind
    followed by a codegen call would silently hit the non-tcgen05
    fallback on a host without native tcgen05.
    """
    args = (
        torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
    )
    with patch_cute_mma_support():
        return _cute_strategy_matmul_kernel.bind(args)


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
        # This small-N problem cannot form the validated 256x256 CtaGroup.TWO
        # tile, so the autotuner keeps cluster_m narrowed to 1.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 1)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
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
        # K=8192 can form validated CtaGroup.TWO products at bk >= 32 even
        # though bk=16 is over the K-tile cap. The search exposes cluster_m=2,
        # and normalization drops only the invalid per-bk products.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 1)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        over_cap_config = {
            "block_sizes": [256, 256, 16],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(over_cap_config, _fix_invalid=True)
        self.assertEqual(over_cap_config["tcgen05_cluster_m"], 1)
        self.assertEqual(over_cap_config["pid_type"], "flat")
        valid_config = {
            "block_sizes": [128, 256, 32],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(valid_config, _fix_invalid=True)
        self.assertEqual(valid_config["tcgen05_cluster_m"], 2)
        self.assertEqual(valid_config["pid_type"], "persistent_interleaved")
        self.assertEqual(valid_config["block_sizes"][:3], [256, 256, 32])
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

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_enters_validated_search_space(self) -> None:
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
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        search_fragments = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search_fragments["tcgen05_cluster_m"].choices, (1, 2))

        config = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["tcgen05_cluster_m"], 2)
        self.assertEqual(config["l2_groupings"], [1])

        for override in (
            {"pid_type": "flat"},
            {"block_sizes": [128, 256, 16]},
            {"block_sizes": [256, 128, 16]},
            {"l2_groupings": [16]},
            {"pid_type": "persistent_interleaved"},
        ):
            with self.subTest(override=override):
                config = {
                    "block_sizes": [256, 256, 16],
                    "l2_groupings": [1],
                    "pid_type": "persistent_blocked",
                    "tcgen05_cluster_m": 2,
                    **override,
                }
                spec.normalize(config, _fix_invalid=True)
                expected_l2_groupings = override.get("l2_groupings", [1])
                self.assertEqual(config["tcgen05_cluster_m"], 2)
                self.assertEqual(config["pid_type"], "persistent_interleaved")
                self.assertEqual(config["block_sizes"][:3], [256, 256, 16])
                self.assertEqual(config["l2_groupings"], expected_l2_groupings)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m1_persistent_search_caps_m_tile(self) -> None:
        """Search-only cluster_m=1 persistent configs stay on tcgen05 M tiles."""

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
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec

        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            with self.subTest(pid_type=pid_type):
                config = {
                    "block_sizes": [256, 32, 16],
                    "pid_type": pid_type,
                    "tcgen05_cluster_m": 1,
                }
                spec.normalize(config, _fix_invalid=True)
                self.assertEqual(config["tcgen05_cluster_m"], 1)
                self.assertEqual(config["pid_type"], pid_type)
                self.assertEqual(
                    config["block_sizes"][:3],
                    [TCGEN05_ONE_CTA_MAX_BLOCK_M, 32, 16],
                )

        flat_config = {
            "block_sizes": [256, 32, 16],
            "pid_type": "flat",
            "tcgen05_cluster_m": 1,
        }
        spec.normalize(flat_config, _fix_invalid=True)
        self.assertEqual(flat_config["block_sizes"][:3], [256, 32, 16])

        two_cta_config = {
            "block_sizes": [256, 32, 16],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(two_cta_config, _fix_invalid=True)
        self.assertEqual(two_cta_config["tcgen05_cluster_m"], 2)
        self.assertEqual(two_cta_config["pid_type"], "persistent_interleaved")
        self.assertEqual(two_cta_config["block_sizes"][:3], [256, 256, 16])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_seeded_in_initial_populations(self) -> None:
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
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)

        def assert_seeded(configs: list[helion.Config]) -> None:
            seeded = [
                config.config
                for config in configs
                if config.config["tcgen05_cluster_m"] == 2
            ]
            self.assertEqual(len(seeded), 1)
            seed = seeded[0]
            self.assertEqual(
                seed["block_sizes"][:3],
                [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 128],
            )
            self.assertEqual(
                seed["indexing"],
                ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            )
            self.assertEqual(seed["l2_groupings"], [TCGEN05_TWO_CTA_SEED_L2_GROUPING])
            self.assertEqual(seed["pid_type"], "persistent_interleaved")
            self.assertEqual(seed["tcgen05_num_epi_warps"], 4)

        config_gen = bound.config_spec.create_config_generation()
        zero_flat = config_gen.random_population_flat(0)
        self.assertEqual(len(zero_flat), 1)
        zero_config = config_gen.unflatten(zero_flat[0])
        self.assertEqual(zero_config.config["tcgen05_cluster_m"], 1)
        one_flat = config_gen.random_population_flat(1)
        self.assertEqual(len(one_flat), 1)
        one_config = config_gen.unflatten(one_flat[0])
        self.assertEqual(one_config.config["tcgen05_cluster_m"], 1)
        one_config_population = config_gen.random_population(1)
        self.assertEqual(len(one_config_population), 1)
        self.assertEqual(one_config_population[0].config["tcgen05_cluster_m"], 1)
        assert_seeded(config_gen.random_population(2))

        acf_config_gen = bound.config_spec.create_config_generation(
            advanced_controls_files=["/tmp/helion-test.acf"]
        )
        acf_configs = acf_config_gen.random_population(2)
        self.assertEqual(len(acf_configs), 2)
        self.assertEqual(
            {config.config["advanced_controls_file"] for config in acf_configs},
            {"/tmp/helion-test.acf"},
        )
        assert_seeded(acf_configs)

        with patch.object(
            PatternSearch, "_find_similar_cached_configs", return_value=[]
        ):
            search = PatternSearch(
                bound,
                args,
                initial_population=30,
                initial_population_strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                best_available_pad_random=False,
            )
            configs = [
                search.config_gen.unflatten(flat)
                for flat in search._generate_initial_population_flat()
            ]
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].config["tcgen05_cluster_m"], 1)
        assert_seeded(configs)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_seed_indexing_matches_live_spec(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_epilogue(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma_epilogue.bind(args)
        self.assertGreater(bound.config_spec.indexing.length, 3)

        configs = bound.config_spec.create_config_generation().random_population(2)
        seeded = [
            config.config
            for config in configs
            if config.config["tcgen05_cluster_m"] == 2
        ]
        self.assertEqual(len(seeded), 1)
        self.assertEqual(
            len(seeded[0]["indexing"]),
            bound.config_spec.indexing.length,
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_projection_falls_back_before_mutation(
        self,
    ) -> None:
        """Invalid cluster_m=2 search products fall back without pid churn."""

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
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec

        for block_sizes in (
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 8],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 24],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, True],
        ):
            with self.subTest(block_sizes=block_sizes):
                original_block_sizes = list(block_sizes)
                config = {
                    "block_sizes": block_sizes,
                    "l2_groupings": [1],
                    "pid_type": "flat",
                    "tcgen05_cluster_m": 2,
                }
                spec._fix_tcgen05_cluster_m2_search_config(config)
                self.assertEqual(config["tcgen05_cluster_m"], 1)
                self.assertEqual(config["pid_type"], "flat")
                self.assertEqual(config["block_sizes"], original_block_sizes)
                self.assertEqual(config["l2_groupings"], [1])

        original_allowed_pid_types = spec.allowed_pid_types
        try:
            spec.allowed_pid_types = ("flat",)
            config = {
                "block_sizes": [
                    TCGEN05_TWO_CTA_BLOCK_M,
                    TCGEN05_TWO_CTA_BLOCK_N,
                    16,
                ],
                "l2_groupings": [1],
                "pid_type": "flat",
                "tcgen05_cluster_m": 2,
            }
            spec._fix_tcgen05_cluster_m2_search_config(config)
        finally:
            spec.allowed_pid_types = original_allowed_pid_types
        self.assertEqual(config["tcgen05_cluster_m"], 1)
        self.assertEqual(config["pid_type"], "flat")
        self.assertEqual(
            config["block_sizes"],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 16],
        )
        self.assertEqual(config["l2_groupings"], [1])

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
        stays narrowed to ``(1,)`` when the problem cannot form the validated
        256x256 CtaGroup.TWO tile, and the num_epi_warps search is narrowed
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
        # This N=128 problem cannot form a validated 256x256 CtaGroup.TWO
        # tile, so the autotune search stays narrowed to cluster_m=1.
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
        # The cluster_m search is narrowed to (1,) unless the matmul caller
        # proves it can form validated CtaGroup.TWO search candidates.
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

        spec = stub.bind(args).config_spec
        spec.allowed_pid_types = (
            "flat",
            "xyz",
            "persistent_blocked",
            "persistent_interleaved",
        )
        spec.narrow_tcgen05_autotune_to_validated_configs(
            allow_persistent_pid_types=True,
            allow_cluster_m2_search=True,
            cluster_m2_static_k=4096,
        )
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
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
        # This small-N problem cannot form the validated 256x256
        # CtaGroup.TWO tile, so cluster_m is narrowed to 1.
        self.assertEqual(flat_fields["tcgen05_cluster_m"].choices, (1,))
        self.assertIn("persistent_blocked", flat_fields["pid_type"].choices)
        self.assertIn("persistent_interleaved", flat_fields["pid_type"].choices)
        self.assertNotIn("num_threads", flat_fields)

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

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_data_model_round_trip(self) -> None:
        """G2-A: ``Tcgen05Strategy`` / ``Tcgen05PersistenceModel`` /
        ``Tcgen05LayoutStrategy`` / ``Tcgen05WarpSpec`` /
        ``Tcgen05LayoutOverrides`` are wired through ``ConfigSpec`` so
        that ``helion.Config(...)`` round-trips them and
        ``default_config()`` exposes the documented defaults
        (``ROLE_LOCAL_MONOLITHIC`` strategy with the pinned 6-warp
        spec; ``epi_warps`` lives in the existing
        ``tcgen05_num_epi_warps`` field).
        """

        spec = _bind_cute_strategy_kernel().config_spec

        # Defaults match the documented G2-A pin: ROLE_LOCAL_MONOLITHIC
        # strategy with the existing 6-warp role-local spec. Persistence
        # model is derived from the active pid_type ("flat" -> non-
        # persistent) so serialized configs cannot encode contradictions.
        default_cfg = spec.default_config()
        self.assertEqual(
            default_cfg.config["tcgen05_strategy"], "role_local_monolithic"
        )
        self.assertEqual(default_cfg.config["pid_type"], "flat")
        self.assertEqual(
            default_cfg.config["tcgen05_persistence_model"], "non_persistent"
        )
        self.assertEqual(default_cfg.config["tcgen05_layout_strategy"], "default")
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_ab_load_warps"], 1)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_mma_warps"], 1)
        # ``epi_warps`` is the existing tcgen05_num_epi_warps knob.
        self.assertEqual(default_cfg.config["tcgen05_num_epi_warps"], 4)
        self.assertNotIn("tcgen05_warp_spec_epi_warps", default_cfg.config)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_epi_load_warps"], 0)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_scheduler_warps"], 0)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_register_decrease"], 120)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_register_increase"], 256)
        for key in (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_smem_swizzle_a",
            "tcgen05_layout_overrides_smem_swizzle_b",
            "tcgen05_layout_overrides_d_store_box_n",
        ):
            self.assertIsNone(default_cfg.config[key])

        # JSON round-trip preserves every strategy field exactly.
        replayed = helion.Config.from_json(default_cfg.to_json())
        self.assertEqual(replayed, default_cfg)

        # An explicit user-supplied config round-trips through
        # normalize. Use persistent pid_type so the explicit
        # ``static_persistent`` agrees.
        cfg = helion.Config(
            block_sizes=[256, 256, 16],
            l2_groupings=[1],
            pid_type="persistent_blocked",
            tcgen05_cluster_m=2,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_monolithic",
            tcgen05_persistence_model="static_persistent",
            tcgen05_layout_strategy="default",
            tcgen05_warp_spec_ab_load_warps=1,
            tcgen05_warp_spec_mma_warps=1,
            tcgen05_warp_spec_epi_load_warps=0,
            tcgen05_warp_spec_scheduler_warps=0,
            tcgen05_warp_spec_register_decrease=120,
            tcgen05_warp_spec_register_increase=256,
        )
        spec.normalize(cfg)
        self.assertEqual(cfg.config["tcgen05_strategy"], "role_local_monolithic")
        self.assertEqual(cfg.config["tcgen05_persistence_model"], "static_persistent")
        self.assertEqual(cfg.config["tcgen05_num_epi_warps"], 4)
        self.assertEqual(cfg.config["tcgen05_warp_spec_register_decrease"], 120)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_reject_illegal(self) -> None:
        """G2-A: validation rejects illegal combinations.

        - ``tcgen05_strategy`` and ``tcgen05_layout_strategy`` are
          narrowed at the autotune fragment to the implemented set so
          unimplemented strategies are loudly rejected at the user
          surface (matches ``restrict_tcgen05_num_epi_warps_*``).
        - ``tcgen05_warp_spec_*`` knobs are narrowed similarly until
          G2-B/C reads them.
        - The cross-fragment validator catches strategy-conditional
          violations that span multiple fragments — exercised
          directly in
          ``test_cute_tcgen05_strategy_invariants_helper_unit``
          for the strategies the autotune fragment narrowing makes
          unreachable from the user surface today.
        """

        spec = _bind_cute_strategy_kernel().config_spec

        base = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
        }

        # ``ROLE_LOCAL_WITH_SCHEDULER`` is now an implemented
        # strategy; explicit user configs that select it must
        # *also* set ``scheduler_warps=1`` to satisfy the
        # cross-fragment invariant.
        with self.assertRaises(InvalidConfig):
            # WITH_SCHEDULER + scheduler_warps=0 (the default) is
            # rejected by the cross-fragment validator.
            spec.normalize(
                helion.Config(**base, tcgen05_strategy="role_local_with_scheduler")
            )
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(**base, tcgen05_layout_strategy="explicit_epi_tile")
            )
        with self.assertRaises(InvalidConfig):
            # MONOLITHIC + scheduler_warps=1 is rejected: MONOLITHIC
            # requires scheduler_warps=0.
            spec.normalize(helion.Config(**base, tcgen05_warp_spec_scheduler_warps=1))
        with self.assertRaises(InvalidConfig):
            spec.normalize(helion.Config(**base, tcgen05_warp_spec_ab_load_warps=2))
        with self.assertRaises(InvalidConfig):
            spec.normalize(helion.Config(**base, tcgen05_warp_spec_mma_warps=2))

        # ``WITH_SCHEDULER`` + ``cluster_m=2`` is accepted. Each
        # CTA in the cluster runs its own scheduler that publishes
        # locally and consumers release locally; both CTAs converge
        # on the same cluster-level virtual_pid via the
        # ``// cluster_m`` collapse in the consumer. See
        # ``cute_mma._codegen_cute_mma`` ``consumer_mask_to_leader``
        # comment for the full topology.
        with_scheduler_cluster_m2 = helion.Config(
            **base,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
        )
        spec.normalize(with_scheduler_cluster_m2)
        self.assertEqual(
            with_scheduler_cluster_m2.config["tcgen05_strategy"],
            "role_local_with_scheduler",
        )
        self.assertEqual(with_scheduler_cluster_m2.config["tcgen05_cluster_m"], 2)

        # WITH_SCHEDULER + scheduler_warps=1 + cluster_m=1 is also
        # valid and round-trips cleanly.
        cluster_m1_base = {
            **base,
            "tcgen05_cluster_m": 1,
        }
        with_scheduler_cfg = helion.Config(
            **cluster_m1_base,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
        )
        spec.normalize(with_scheduler_cfg)
        self.assertEqual(
            with_scheduler_cfg.config["tcgen05_strategy"],
            "role_local_with_scheduler",
        )
        self.assertEqual(
            with_scheduler_cfg.config["tcgen05_warp_spec_scheduler_warps"], 1
        )
        self.assertEqual(with_scheduler_cfg.config["tcgen05_cluster_m"], 1)

        # ``DYNAMIC_PERSISTENT`` is not in the persistence-model
        # fragment surface today (no codegen supports it).
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(**base, tcgen05_persistence_model="dynamic_persistent")
            )

        # ``epi_warps != 4`` -> rejected via ``tcgen05_num_epi_warps``
        # validation (single source of truth).
        with self.assertRaises(InvalidConfig):
            spec.normalize(helion.Config(**base, tcgen05_num_epi_warps=2))

        # Persistence model must agree with pid_type. The explicit
        # ``static_persistent`` contradicts ``pid_type=flat``.
        flat_base = {**base, "pid_type": "flat", "tcgen05_cluster_m": 1}
        with self.assertRaises(InvalidConfig) as ctx:
            spec.normalize(
                helion.Config(
                    **flat_base, tcgen05_persistence_model="static_persistent"
                )
            )
        self.assertIn("contradicts pid_type", str(ctx.exception))

        # Layout overrides with a concrete value under DEFAULT layout
        # strategy must be rejected — the override would be silently
        # ignored otherwise.
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(
                    **base,
                    tcgen05_layout_strategy="default",
                    tcgen05_layout_overrides_epi_tile_m=64,
                )
            )

        # The pinned ROLE_LOCAL_MONOLITHIC config still normalizes
        # cleanly so the rejection paths are not over-broad.
        cfg = helion.Config(
            **base,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_monolithic",
        )
        spec.normalize(cfg)
        self.assertEqual(cfg.config["tcgen05_strategy"], "role_local_monolithic")

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_helper_unit(self) -> None:
        """``validate_tcgen05_strategy_invariants`` covers the
        cross-fragment cases the autotune narrowing makes unreachable
        from the user surface today (persistence model not supported
        by the chosen strategy, scheduler_warps mismatching the
        strategy) plus the positive case where ``EXPLICIT_EPI_TILE``
        accepts non-None layout overrides.

        The earlier warpgroup-alignment requirement on
        ``ROLE_LOCAL_WITH_SCHEDULER`` was relaxed once the initial
        7-warp implementation landed (1 ab_load + 1 mma + 4 epi + 1
        scheduler = 7). The eventual 8-warp variant with a C-input
        epi-load warp will re-introduce the alignment requirement
        when register-split tuning becomes warpgroup-uniform.
        """
        # scheduler_warps=0 under WITH_SCHEDULER is rejected (the
        # strategy demands one scheduler warp).
        wrong_scheduler_count = Tcgen05WarpSpec(
            ab_load_warps=1,
            mma_warps=1,
            epi_warps=4,
            epi_load_warps=0,
            scheduler_warps=0,
            register_split=(120, 256),
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=wrong_scheduler_count,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertTrue(any("scheduler_warps=1" in e for e in errors))

        # DYNAMIC_PERSISTENT under a strategy that does not support it.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.DYNAMIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertTrue(any("dynamic_persistent" in e for e in errors))

        # ``ROLE_LOCAL_WITH_SCHEDULER`` runs at cluster_m ∈ {1, 2}.
        # cluster_m=3+ falls outside the supported set; the
        # validator must reject so a user config can't reach an
        # untested cluster shape.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=dataclasses.replace(
                ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
            ),
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=4,
        )
        self.assertTrue(
            any("tcgen05_cluster_m=4" in e for e in errors), msg=str(errors)
        )

        # Positive control: ROLE_LOCAL_WITH_SCHEDULER + cluster_m=2
        # is now accepted (the per-CTA scheduler-warp topology is
        # cluster-correct).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=dataclasses.replace(
                ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
            ),
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # Positive case: EXPLICIT_EPI_TILE + non-None overrides is
        # accepted — the validator must not drift into rejecting all
        # override values regardless of layout strategy.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(epi_tile_m=64, epi_tile_n=32),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [])

        # Negative control: clean ROLE_LOCAL_MONOLITHIC default is
        # always accepted.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_cluster_n(self) -> None:
        """G2 cluster_n=2 validator coverage (cute_plan.md §6.12.7).

        ``cluster_n=2`` only runs under ``ROLE_LOCAL_MONOLITHIC`` with
        ``cluster_m=2`` (the validated 4-CTA cluster envelope). The
        validator rejects:
          - ``cluster_n=2`` with ``cluster_m=1`` (V=1 has no 4-CTA path)
          - ``cluster_n=2`` under ``ROLE_LOCAL_WITH_SCHEDULER`` (the
            scheduler-broadcast topology is not validated for cluster_n>1)
        and accepts ``cluster_n=2`` under ``ROLE_LOCAL_MONOLITHIC`` +
        ``cluster_m=2``.
        """
        # Positive control: cluster_n=2 + ROLE_LOCAL_MONOLITHIC + cluster_m=2.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=2,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # cluster_n=2 with cluster_m=1: rejected (requires the 4-CTA
        # V=2 cluster).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
            cluster_n=2,
        )
        self.assertTrue(
            any("requires tcgen05_cluster_m=2" in e for e in errors),
            msg=str(errors),
        )

        # cluster_n=2 under ROLE_LOCAL_WITH_SCHEDULER: rejected
        # (scheduler-broadcast topology is not validated for cluster_n>1).
        with_sched = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=2,
        )
        self.assertTrue(
            any("tcgen05_cluster_n in [1]" in e for e in errors),
            msg=str(errors),
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_clc_persistent(self) -> None:
        """G2-H (cute_plan.md): ``Tcgen05PersistenceModel.CLC_PERSISTENT``
        is only valid under ``ROLE_LOCAL_WITH_SCHEDULER`` on arch >= 100.

        The validator must reject the model under MONOLITHIC (the
        scheduler-warp role only exists in WITH_SCHEDULER) and on
        arch < 100 (CLC is a Blackwell sm_100+ instruction). The
        positive control: WITH_SCHEDULER + arch_major=10 +
        scheduler_warps=1 + persistent_* pid_type accepts cleanly.
        """
        # Positive control: CLC + WITH_SCHEDULER + sm_100 (arch=10).
        with_sched = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            arch_major=10,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # CLC under MONOLITHIC: rejected (the strategy doesn't
        # support CLC because it has no scheduler warp to issue
        # the query).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
            arch_major=10,
        )
        self.assertTrue(any("clc_persistent" in e for e in errors), msg=str(errors))

        # CLC on arch < 100: rejected.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            arch_major=9,
        )
        self.assertTrue(
            any("requires CUDA compute capability major >= 10" in e for e in errors),
            msg=str(errors),
        )

        # CLC overlays a runtime cancel on the persistent-grid
        # launch, so it must agree with ``pid_type=persistent_*``;
        # CLC paired with ``pid_type=flat`` is rejected with the
        # contradiction error (validator asks user to set both
        # consistently).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="flat",
            cluster_m=1,
            arch_major=10,
        )
        self.assertTrue(
            any("contradicts pid_type" in e for e in errors), msg=str(errors)
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_persistence_model_enum_value_pin(self) -> None:
        """Pin the string literal that ``CuteTcgen05MatmulPlan.is_clc_persistent``
        compares against to the actual enum's ``.value``.

        ``CuteTcgen05MatmulPlan.persistence_model`` is stored as a
        ``str`` (the enum's ``.value``) so the dataclass stays free
        of cute-internal imports. The ``is_clc_persistent`` property
        reads the enum value lazily and compares — this test pins
        that the canonical value is ``"clc_persistent"`` so a rename
        of the enum member would either propagate via the lazy
        import or trip this test loudly. Without it a renamed enum
        could silently degrade ``is_clc_persistent`` to always-False
        because all the comparisons would be against a stale string
        literal in serialized configs.
        """
        self.assertEqual(Tcgen05PersistenceModel.CLC_PERSISTENT.value, "clc_persistent")
        self.assertEqual(
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value, "static_persistent"
        )
        # Round-trip via ``CuteTcgen05MatmulPlan`` to confirm the
        # property tracks the enum value.
        from helion._compiler.device_function import CuteTcgen05MatmulPlan

        plan_clc = CuteTcgen05MatmulPlan(
            bm=256,
            bn=256,
            bk=128,
            k_tile_count=4,
            cluster_m=2,
            is_two_cta=True,
            uses_role_local_persistent_body=True,
            uses_cluster_m2_one_cta_role_local_bridge=False,
            cta_thread_count=256,
            physical_m_threads=32,
            acc_stage_count=2,
            ab_stage_count=2,
            c_stage_count=2,
            epi_warp_count=4,
            ab_load_warp_count=1,
            scheduler_warp_count=1,
            sched_stage_count=1,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )
        self.assertTrue(plan_clc.is_clc_persistent)
        plan_static = dataclasses.replace(
            plan_clc,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        self.assertFalse(plan_static.is_clc_persistent)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_warpgroup_alignment_branch(
        self,
    ) -> None:
        """The warpgroup-alignment branch of
        ``validate_tcgen05_strategy_invariants`` is currently dead
        code (``_STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL`` is
        empty) because today's two strategies tolerate non-aligned
        role-warp totals via ``CuteTcgen05MatmulPlan.launched_warp_count``
        rounding at the launch boundary. Patch the set to include
        an existing strategy enum and pass a misaligned warp_spec
        to confirm the validator's alignment check still fires —
        so a future strategy that opts in catches misconfigured
        warp counts loudly.
        """
        from helion._compiler.cute import strategies as strategies_module

        misaligned = Tcgen05WarpSpec(
            ab_load_warps=1,
            mma_warps=1,
            epi_warps=4,
            epi_load_warps=0,
            scheduler_warps=1,  # 1+1+4+1 = 7, not warpgroup-aligned
            register_split=(120, 256),
        )
        with patch.object(
            strategies_module,
            "_STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL",
            frozenset({Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER}),
        ):
            errors = validate_tcgen05_strategy_invariants(
                strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
                persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
                layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
                warp_spec=misaligned,
                layout_overrides=Tcgen05LayoutOverrides(),
                pid_type="persistent_blocked",
                cluster_m=1,
            )
        self.assertTrue(any("warpgroup-aligned" in e for e in errors), msg=str(errors))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_fix_invalid_resets_to_defaults(self) -> None:
        """G2-A: ``normalize(_fix_invalid=True)`` silently rolls a
        broken strategy record back to the documented defaults rather
        than raising. Mirrors the cluster_m=2 search canonicalization
        path used by ``_fix_tcgen05_cluster_m2_search_config``.
        Layout-override values are silently dropped to ``None``.
        """

        spec = _bind_cute_strategy_kernel().config_spec

        # A user-supplied config that hits the cross-fragment
        # validator (DEFAULT layout + concrete override). Without
        # ``_fix_invalid`` this raises; with it, the strategy fields
        # reset to defaults derived from the active pid_type.
        config = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
            "tcgen05_layout_strategy": "default",
            "tcgen05_layout_overrides_epi_tile_m": 64,
        }
        spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["tcgen05_strategy"], "role_local_monolithic")
        self.assertEqual(config["tcgen05_persistence_model"], "static_persistent")
        self.assertEqual(config["tcgen05_layout_strategy"], "default")
        # Override that triggered the rollback is now None.
        self.assertIsNone(config["tcgen05_layout_overrides_epi_tile_m"])

        # An out-of-range override under DEFAULT also fixes silently.
        config2 = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
            "tcgen05_layout_overrides_epi_tile_n": "not-an-int",
        }
        spec.normalize(config2, _fix_invalid=True)
        self.assertIsNone(config2["tcgen05_layout_overrides_epi_tile_n"])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_normalize_idempotent_after_pid_type_fixup(
        self,
    ) -> None:
        """G2-A regression: the strategy default/invariant pass must
        run *after* ``pid_type`` canonicalization and the
        ``_fix_tcgen05_cluster_m{1,2}_*_search_config`` rewrites,
        otherwise ``tcgen05_persistence_model`` is derived from the
        pre-fixup ``pid_type`` and a re-``normalize()`` over the
        already-normalized config trips the
        ``contradicts pid_type`` invariant.

        The path: a search config with ``pid_type="flat"`` and
        ``tcgen05_cluster_m=2`` lands in ``_fix_tcgen05_cluster_m2_search_config``,
        which rewrites ``pid_type`` to ``persistent_interleaved``. The
        derived persistence model must follow that rewrite.
        """

        spec = _bind_cute_strategy_kernel().config_spec

        config: dict[str, object] = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(config, _fix_invalid=True)
        # The cluster_m2 fixup rewrote pid_type; the persistence-model
        # default agrees with the post-fixup pid_type.
        self.assertEqual(config["pid_type"], "persistent_interleaved")
        self.assertEqual(config["tcgen05_persistence_model"], "static_persistent")

        # Re-normalize on the already-normalized config is idempotent
        # — it does not raise and does not change any field.
        snapshot = dict(config)
        spec.normalize(config, _fix_invalid=False)
        self.assertEqual(config, snapshot)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_flat_round_trip_with_force_persistent(
        self,
    ) -> None:
        """G2-A regression: ``flatten(unflatten(default_flat())) ==
        default_flat()`` even when ``autotune_force_persistent`` has
        narrowed ``allowed_pid_types`` so the default ``pid_type``
        is ``persistent_blocked`` rather than ``flat``.

        ``tcgen05_persistence_model`` is fully derived from
        ``pid_type`` (see ``derive_persistence_model_from_pid_type``)
        so giving it its own slot in ``_flat_fields()`` would mean
        the flat default carries ``non_persistent`` (the
        ``EnumFragment`` default) while the post-normalize value is
        ``static_persistent`` (derived from the persistent
        ``pid_type``). The ``flatten``/``unflatten`` round trip would
        then stabilize on the post-normalize value and the
        autotuner's ``default_flat()`` baseline would diverge from
        every other flat config it generates. Pin the round-trip so
        the field stays out of the autotune surface until a strategy
        decouples it.
        """

        args = (
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_strategy_matmul_force_persistent_kernel.bind(args)
        spec = bound.config_spec
        # autotune_force_persistent removes flat/xyz from the
        # allowed pid_types, so the EnumFragment(pid_type) default
        # is "persistent_blocked".
        self.assertEqual(
            spec.allowed_pid_types,
            ("persistent_blocked", "persistent_interleaved"),
        )
        cg = ConfigGeneration(spec)
        default_flat = cg.default_flat()
        round_tripped = cg.flatten(cg.unflatten(default_flat))
        self.assertEqual(default_flat, round_tripped)
        # Cross-check: the unflattened config's persistence model is
        # the derived value (static_persistent), and the autotune
        # surface (``_flat_fields``) excludes the field so it does
        # not carry a stale flat-config default.
        config = cg.unflatten(default_flat)
        self.assertEqual(
            config.config["tcgen05_persistence_model"], "static_persistent"
        )
        self.assertNotIn("tcgen05_persistence_model", spec._flat_fields())

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_default_lowering_byte_identical(
        self,
    ) -> None:
        """G2-A pins generated code byte-identity: the strategy data
        model is plumbed through ``ConfigSpec`` but no codegen path
        reads it yet. The retained role-local seed produces the same
        kernel source whether the strategy fields are explicitly set
        to their documented defaults or omitted entirely.
        """

        # ``cute_mma.py`` consults ``get_cute_mma_support()`` during
        # codegen, so the patch must remain active across both
        # ``to_triton_code()`` calls — without it, on a host without
        # native tcgen05 support both kernels silently fall through to
        # the non-tcgen05 path and the byte-identity check still
        # passes vacuously. The ``make_trivial_tiled_mma`` assert is a
        # tcgen05-specific marker that catches this regression.
        args = (
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_strategy_matmul_kernel.bind(args)
            baseline_seed = {
                "block_sizes": [256, 256, 16],
                "l2_groupings": [1],
                "pid_type": "persistent_interleaved",
                "tcgen05_cluster_m": 2,
                "tcgen05_ab_stages": 2,
                "tcgen05_acc_stages": 2,
                "tcgen05_c_stages": 2,
                "tcgen05_num_epi_warps": 4,
            }
            baseline = helion.Config(**baseline_seed)
            with_strategy = helion.Config(
                **baseline_seed,
                tcgen05_strategy="role_local_monolithic",
                tcgen05_persistence_model="static_persistent",
                tcgen05_layout_strategy="default",
                tcgen05_warp_spec_ab_load_warps=1,
                tcgen05_warp_spec_mma_warps=1,
                tcgen05_warp_spec_epi_load_warps=0,
                tcgen05_warp_spec_scheduler_warps=0,
                tcgen05_warp_spec_register_decrease=120,
                tcgen05_warp_spec_register_increase=256,
            )

            baseline_code = bound.to_triton_code(baseline)
            with_strategy_code = bound.to_triton_code(with_strategy)
        self.assertIn("make_trivial_tiled_mma", baseline_code)
        self.assertIn("make_trivial_tiled_mma", with_strategy_code)
        self.assertEqual(baseline_code, with_strategy_code)


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
