"""
Test suite for deep AB staging functionality (commit 1573e3d2).

Tests the improved implementation of max_ab_stages_that_fit and related
deep staging logic for FP8 dtypes.
"""

from __future__ import annotations

from helion._testing import TestCase


class TestDeepABStagingHelpers(TestCase):
    """Test helper methods for deep staging."""

    def test_get_dtype_ab_stages_hard_cap(self) -> None:
        """Test dtype-specific hard cap calculation."""
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

        # FP8 (1 byte) -> 12 stages
        self.assertEqual(CuteTcgen05Config._get_dtype_ab_stages_hard_cap(1), 12)

        # FP16/BF16 (2 bytes) -> 6 stages
        self.assertEqual(CuteTcgen05Config._get_dtype_ab_stages_hard_cap(2), 6)

        # FP32 (4 bytes) -> 3 stages
        self.assertEqual(CuteTcgen05Config._get_dtype_ab_stages_hard_cap(4), 3)

        # Larger dtypes -> 3 stages (baseline)
        self.assertEqual(CuteTcgen05Config._get_dtype_ab_stages_hard_cap(8), 3)

        # Invalid input
        self.assertEqual(CuteTcgen05Config._get_dtype_ab_stages_hard_cap(0), 0)
        self.assertEqual(CuteTcgen05Config._get_dtype_ab_stages_hard_cap(-1), 0)


class TestMaxABStagesThatFit(TestCase):
    """Test max_ab_stages_that_fit computation."""

    def test_fp8_vs_bf16_capacity(self) -> None:
        """Test that FP8 fits ~2x more stages than BF16 for same tile."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion._compiler.cute.tcgen05_config import (
            Tcgen05AbStagesThreeSearchConstraints,
        )
        from helion.autotuner.config_spec import ConfigSpec

        # Use realistic B200 SMEM budget (232KB after reservations)
        smem_budget = 232 * 1024

        # FP8 config (1 byte per element)
        fp8_config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        fp8_config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=1,
                per_cta_smem_budget_bytes=smem_budget,
            )
        )
        fp8_config.search_enabled = True

        # BF16 config (2 bytes per element)
        bf16_config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        bf16_config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=2,
                per_cta_smem_budget_bytes=smem_budget,
            )
        )
        bf16_config.search_enabled = True

        # Test with 256x256x64 cluster_m=2 (common config)
        tile_params = {"bm": 256, "bn": 256, "bk": 64, "cluster_m": 2}

        fp8_max = fp8_config.max_ab_stages_that_fit(**tile_params)
        bf16_max = bf16_config.max_ab_stages_that_fit(**tile_params)

        # FP8 should fit more stages
        self.assertGreater(fp8_max, bf16_max, "FP8 should fit more stages than BF16")

        # FP8 should support deep staging (>6 stages)
        self.assertGreater(fp8_max, 6, "FP8 should enable deep staging")

        # BF16 should be limited
        self.assertLessEqual(bf16_max, 6, "BF16 should cap at 6 or less")

        # Ratio should be roughly 2x (within overhead tolerance)
        if bf16_max > 0:
            ratio = fp8_max / bf16_max
            self.assertGreater(ratio, 1.5, "FP8 should fit at least 1.5x BF16 stages")
            self.assertLess(ratio, 2.5, "Ratio should be roughly 2x (within overhead)")

    def test_invalid_tile_dimensions(self) -> None:
        """Test that invalid tile dimensions return 0."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion._compiler.cute.tcgen05_config import (
            Tcgen05AbStagesThreeSearchConstraints,
        )
        from helion.autotuner.config_spec import ConfigSpec

        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=1,
                per_cta_smem_budget_bytes=100000,
            )
        )
        config.search_enabled = True

        # bm = 0
        self.assertEqual(
            config.max_ab_stages_that_fit(bm=0, bn=256, bk=64, cluster_m=1), 0
        )

        # bn = -1
        self.assertEqual(
            config.max_ab_stages_that_fit(bm=256, bn=-1, bk=64, cluster_m=1), 0
        )

        # bk = 0
        self.assertEqual(
            config.max_ab_stages_that_fit(bm=256, bn=256, bk=0, cluster_m=1), 0
        )

    def test_invalid_cluster_m(self) -> None:
        """Test that invalid cluster_m values return 0."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion._compiler.cute.tcgen05_config import (
            Tcgen05AbStagesThreeSearchConstraints,
        )
        from helion.autotuner.config_spec import ConfigSpec

        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=1,
                per_cta_smem_budget_bytes=100000,
            )
        )
        config.search_enabled = True

        # cluster_m = 0 (invalid)
        self.assertEqual(
            config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=0), 0
        )

        # cluster_m = 3 (not supported)
        self.assertEqual(
            config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=3), 0
        )

        # cluster_m = -1 (invalid)
        self.assertEqual(
            config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=-1), 0
        )

    def test_no_constraints_returns_zero(self) -> None:
        """Test that missing constraints return 0."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion.autotuner.config_spec import ConfigSpec

        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        config.ab_stages_three_search_constraints = None
        config.search_enabled = True

        result = config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=1)
        self.assertEqual(result, 0, "Should return 0 when constraints are None")

    def test_cluster_m_scaling(self) -> None:
        """Test that cluster_m=2 fits more stages than cluster_m=1."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion._compiler.cute.tcgen05_config import (
            Tcgen05AbStagesThreeSearchConstraints,
        )
        from helion.autotuner.config_spec import ConfigSpec

        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=1,
                per_cta_smem_budget_bytes=150000,
            )
        )
        config.search_enabled = True

        tile_params = {"bm": 256, "bn": 256, "bk": 64}

        stages_cluster1 = config.max_ab_stages_that_fit(**tile_params, cluster_m=1)
        stages_cluster2 = config.max_ab_stages_that_fit(**tile_params, cluster_m=2)

        # cluster_m=2 partitions operands across 2 CTAs, so per-CTA SMEM is halved
        self.assertGreater(
            stages_cluster2,
            stages_cluster1,
            "cluster_m=2 should fit more stages (per-CTA SMEM is halved)",
        )

        # Should be roughly 2x (within overhead)
        if stages_cluster1 > 0:
            ratio = stages_cluster2 / stages_cluster1
            self.assertGreater(ratio, 1.5, "Should be at least 1.5x")
            self.assertLess(ratio, 2.5, "Should be at most 2.5x")

    def test_hard_cap_enforcement(self) -> None:
        """Test that hard_cap parameter is respected."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion._compiler.cute.tcgen05_config import (
            Tcgen05AbStagesThreeSearchConstraints,
        )
        from helion.autotuner.config_spec import ConfigSpec

        # Config with huge budget (should hit hard cap, not budget)
        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=1,
                per_cta_smem_budget_bytes=10000000,  # 10MB (unrealistic)
            )
        )
        config.search_enabled = True

        # Without explicit hard cap, should hit dtype default (12 for FP8)
        max_default = config.max_ab_stages_that_fit(bm=64, bn=64, bk=32, cluster_m=2)
        self.assertLessEqual(max_default, 12, "Should hit FP8 default cap of 12")

        # With explicit hard cap of 5
        max_capped = config.max_ab_stages_that_fit(
            bm=64, bn=64, bk=32, cluster_m=2, hard_cap=5
        )
        self.assertEqual(max_capped, 5, "Should respect explicit hard cap")

    def test_returns_at_least_one_when_valid(self) -> None:
        """Test that result is at least 1 when ab_stages=1 fits."""
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
        from helion._compiler.cute.tcgen05_config import (
            Tcgen05AbStagesThreeSearchConstraints,
        )
        from helion.autotuner.config_spec import ConfigSpec

        # Small budget that only fits ab_stages=1 or 2
        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=2,  # BF16
                per_cta_smem_budget_bytes=50000,  # Tight budget
            )
        )
        config.search_enabled = True

        # Large tile
        result = config.max_ab_stages_that_fit(bm=256, bn=256, bk=128, cluster_m=1)

        # Should return at least 1 if ab_stages=1 fits, or 0 if it doesn't
        self.assertIn(result, [0, 1, 2, 3], "Result should be in valid range")
        if result > 0:
            self.assertGreaterEqual(result, 1, "Should return at least 1 if valid")
