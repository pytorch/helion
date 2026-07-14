from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import unittest

from helion.autotuner.search_space_logger import ExplorationReport
from helion.autotuner.search_space_logger import FeatureExplorationStats
from helion.autotuner.search_space_logger import FeatureExplorationTracker
from helion.autotuner.search_space_logger import SearchSpaceDimension
from helion.autotuner.search_space_logger import SearchSpaceSummary
from helion.autotuner.search_space_logger import _analyze_dimension_size
from helion.autotuner.search_space_logger import analyze_search_space
from helion.autotuner.search_space_logger import save_exploration_report
from helion.autotuner.search_space_logger import save_search_space_summary


class _FakeConfig:
    """Minimal stand-in for runtime.Config used by the tracker.

    ``FeatureExplorationTracker`` only reads attributes via
    ``_extract_feature_value``, so a simple attribute holder is enough.
    """

    def __init__(
        self,
        *,
        block_sizes: list[int],
        loop_orders: list[int],
        num_warps: int,
        epilogue_subtile: object = None,
    ) -> None:
        self.block_sizes = block_sizes
        self.loop_orders = loop_orders
        self.num_warps = num_warps
        self.epilogue_subtile = epilogue_subtile


def _summary(
    *,
    dimensions: list[SearchSpaceDimension],
    disabled_features: list[str] | None = None,
    shape_constraints: list[str] | None = None,
    backend: str = "triton",
    total: int | None = 138240,
) -> SearchSpaceSummary:
    return SearchSpaceSummary(
        kernel_name="kernel_under_test",
        specialization_key=None,
        backend=backend,
        hardware="NVIDIA H100",
        dimensions=dimensions,
        total_search_space_size=total,
        enabled_features=[],
        disabled_features=disabled_features or [],
        shape_constraints=shape_constraints or [],
    )


class TestFeatureExplorationStats(unittest.TestCase):
    def test_total_options_defaults_to_len_values(self) -> None:
        stats = FeatureExplorationStats(
            feature_name="num_warps",
            all_possible_values=[1, 2, 4, 8, 16, 32],
            tested_values=[1, 2, 4],
            coverage_percent=50.0,
        )
        self.assertEqual(stats.total_options, 6)
        self.assertEqual(stats.to_summary_line(), "num_warps: 3/6 options tested (50.0%)")

    def test_zero_options_renders_not_applicable(self) -> None:
        """A feature with no autotunable choices renders text, not '0/0'."""
        stats = FeatureExplorationStats(
            feature_name="epilogue_subtile",
            all_possible_values=[],
            tested_values=[],
            coverage_percent=0.0,
            total_options=0,
        )
        line = stats.to_summary_line()
        self.assertNotIn("0/0", line)
        self.assertIn("no autotunable choices", line)

    def test_displayed_tested_count_clamped_to_total(self) -> None:
        """The displayed numerator never exceeds the denominator (no '7/4')."""
        stats = FeatureExplorationStats(
            feature_name="l2_groupings",
            all_possible_values=[],
            tested_values=[1, 2, 4, 8, 16, 32, 64],  # 7 distinct
            coverage_percent=100.0,
            total_options=4,  # deliberately-too-small estimate
        )
        self.assertEqual(stats.displayed_tested_count, 4)
        self.assertEqual(
            stats.to_summary_line(), "l2_groupings: 4/4 options tested (100.0%)"
        )


class TestFeatureExplorationTracker(unittest.TestCase):
    def test_coverage_denominator_uses_size_when_values_absent(self) -> None:
        """Regression: block_sizes/loop_orders reported '16/0' (div-by-zero)."""
        dims = [
            # No explicit `values`; only `size` is known.
            SearchSpaceDimension(name="block_sizes", dim_type="discrete", size=4096),
            SearchSpaceDimension(name="loop_orders", dim_type="discrete", size=24),
            SearchSpaceDimension(
                name="num_warps",
                dim_type="discrete",
                size=6,
                values=[1, 2, 4, 8, 16, 32],
            ),
        ]
        tracker = FeatureExplorationTracker(_summary(dimensions=dims))
        for i in range(16):
            tracker.record_config(
                _FakeConfig(
                    block_sizes=[32 + i, 64],
                    loop_orders=[0, 1] if i % 2 else [1, 0],
                    num_warps=[1, 2, 4, 8, 16, 32][i % 6],
                )
            )

        report = tracker.generate_report("LFBOTreeSearch", 502.5)
        by_name = {s.feature_name: s for s in report.feature_stats}

        # block_sizes: 16 unique tuples tested, denominator = size (4096), not 0.
        self.assertEqual(by_name["block_sizes"].total_options, 4096)
        self.assertEqual(len(by_name["block_sizes"].tested_values), 16)
        self.assertAlmostEqual(
            by_name["block_sizes"].coverage_percent, 16 / 4096 * 100
        )

        # loop_orders: 2 unique permutations tested, denominator = size (24).
        self.assertEqual(by_name["loop_orders"].total_options, 24)
        self.assertEqual(len(by_name["loop_orders"].tested_values), 2)

        # num_warps: explicit values, full coverage.
        self.assertEqual(by_name["num_warps"].total_options, 6)
        self.assertEqual(by_name["num_warps"].coverage_percent, 100.0)

    def test_coverage_clamped_to_100_percent(self) -> None:
        """Tested set larger than the estimated size must not exceed 100%."""
        dims = [SearchSpaceDimension(name="loop_orders", dim_type="discrete", size=2)]
        tracker = FeatureExplorationTracker(_summary(dimensions=dims))
        # Record more distinct permutations than the size estimate (2).
        for order in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0]):
            tracker.record_config(
                _FakeConfig(block_sizes=[32], loop_orders=order, num_warps=4)
            )
        report = tracker.generate_report("LFBOTreeSearch", 1.0)
        stats = report.feature_stats[0]
        self.assertLessEqual(stats.coverage_percent, 100.0)
        self.assertEqual(stats.coverage_percent, 100.0)

    def test_empty_dimension_reports_zero_not_crash(self) -> None:
        dims = [SearchSpaceDimension(name="loop_orders", dim_type="discrete", size=0)]
        tracker = FeatureExplorationTracker(_summary(dimensions=dims))
        report = tracker.generate_report("LFBOTreeSearch", 1.0)
        stats = report.feature_stats[0]
        self.assertEqual(stats.total_options, 0)
        self.assertEqual(stats.coverage_percent, 0.0)

    def test_zero_option_feature_excluded_from_aggregates(self) -> None:
        """A feature with no available choices (0/0) must not drag avg/min to 0%."""
        dims = [
            # epilogue_subtile is inapplicable to this kernel (size 0).
            SearchSpaceDimension(
                name="epilogue_subtile", dim_type="categorical", size=0
            ),
            SearchSpaceDimension(
                name="num_warps",
                dim_type="discrete",
                size=6,
                values=[1, 2, 4, 8, 16, 32],
            ),
        ]
        tracker = FeatureExplorationTracker(_summary(dimensions=dims))
        for w in (1, 2, 4, 8, 16, 32):
            tracker.record_config(
                _FakeConfig(block_sizes=[32], loop_orders=[0], num_warps=w)
            )
        report = tracker.generate_report("LFBOTreeSearch", 1.0)
        # Only num_warps (fully covered) contributes to the aggregates.
        self.assertEqual(report.avg_feature_coverage, 100.0)
        self.assertEqual(report.min_feature_coverage, 100.0)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _capture(summary: SearchSpaceSummary) -> list[str]:
    logger = logging.getLogger("test_search_space_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _ListHandler()
    logger.handlers = [handler]
    summary.log_summary(logger)
    return handler.lines


class TestExploredValidityTracking(unittest.TestCase):
    """Explored-config validity breakdown (valid vs. rejected candidates)."""

    def _tracker(self) -> FeatureExplorationTracker:
        dims = [
            SearchSpaceDimension(
                name="num_warps",
                dim_type="discrete",
                size=6,
                values=[1, 2, 4, 8, 16, 32],
            )
        ]
        return FeatureExplorationTracker(_summary(dimensions=dims))

    def test_valid_and_invalid_counts(self) -> None:
        # No invalids recorded: all attempts valid (100%).
        tracker = self._tracker()
        tracker.record_config(
            _FakeConfig(block_sizes=[32], loop_orders=[0], num_warps=4)
        )
        report = tracker.generate_report("RandomSearch", 1.0)
        self.assertEqual(report.explored_valid, 1)
        self.assertEqual(report.explored_invalid, 0)
        self.assertEqual(report.explored_total, 1)
        self.assertEqual(report.explored_valid_percent, 100.0)

        # Mix of valid and invalid, including a batched record_invalid(2).
        tracker = self._tracker()
        for warps in (1, 2, 4):
            tracker.record_config(
                _FakeConfig(block_sizes=[32], loop_orders=[0], num_warps=warps)
            )
        tracker.record_invalid()
        tracker.record_invalid(2)
        report = tracker.generate_report("RandomSearch", 1.0)
        self.assertEqual(report.explored_valid, 3)
        self.assertEqual(report.explored_invalid, 3)
        self.assertEqual(report.explored_total, 6)
        self.assertEqual(report.explored_valid_percent, 50.0)

    def test_record_invalid_ignores_nonpositive(self) -> None:
        tracker = self._tracker()
        tracker.record_invalid(0)
        tracker.record_invalid(-5)
        self.assertEqual(tracker.invalid_config_count, 0)

    def test_zero_total_percent_is_zero(self) -> None:
        report = ExplorationReport(
            kernel_name="k",
            backend="triton",
            search_algorithm="RandomSearch",
            elapsed_seconds=1.0,
            configs_tested=0,
            total_search_space_size=100,
            feature_stats=[],
            avg_feature_coverage=0.0,
            min_feature_coverage=0.0,
        )
        self.assertEqual(report.explored_total, 0)
        self.assertEqual(report.explored_valid_percent, 0.0)

    def _summary_lines(self, report: ExplorationReport, name: str) -> str:
        logger = logging.getLogger(f"test_search_space_logger.{name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _ListHandler()
        logger.handlers = [handler]
        report.log_summary(logger)
        return "\n".join(handler.lines)

    def _report(self, *, explored_valid: int, explored_invalid: int) -> ExplorationReport:
        return ExplorationReport(
            kernel_name="k",
            backend="triton",
            search_algorithm="RandomSearch",
            elapsed_seconds=1.0,
            configs_tested=explored_valid,
            total_search_space_size=100,
            feature_stats=[],
            avg_feature_coverage=0.0,
            min_feature_coverage=0.0,
            explored_valid=explored_valid,
            explored_invalid=explored_invalid,
        )

    def test_log_summary_breakdown(self) -> None:
        # With attempts, the breakdown line is present and formatted.
        report = self._report(explored_valid=8, explored_invalid=2)
        joined = self._summary_lines(report, "validity")
        self.assertIn("Configs attempted: 10", joined)
        self.assertIn("8 valid", joined)
        self.assertIn("2 invalid", joined)
        self.assertIn("80.0% valid", joined)

        # With no attempts, the breakdown line is omitted entirely.
        report = self._report(explored_valid=0, explored_invalid=0)
        joined = self._summary_lines(report, "validity_empty")
        self.assertNotIn("Configs attempted", joined)


class TestDisabledFeatureGrouping(unittest.TestCase):
    def test_generic_backend_features_collapsed(self) -> None:
        disabled = [
            f"cute_flash_x{i}: Not supported by triton backend" for i in range(78)
        ]
        disabled += [
            "epilogue_subtile: Not a matmul-like kernel",
            "pallas_loop_type: Triton backend (no Pallas loops)",
        ]
        lines = _capture(_summary(dimensions=[], disabled_features=disabled))
        joined = "\n".join(lines)

        # Header still reports the true total.
        self.assertIn("Disabled features (80):", joined)
        # Feature-specific reasons are listed individually.
        self.assertIn("- epilogue_subtile: Not a matmul-like kernel", joined)
        self.assertIn("- pallas_loop_type: Triton backend (no Pallas loops)", joined)
        # The 78 generic backend features collapse into a single count line.
        summary_lines = [
            ln for ln in lines if "feature(s) not supported by triton backend" in ln
        ]
        self.assertEqual(len(summary_lines), 1)
        self.assertIn("78 feature(s) not supported by triton backend", summary_lines[0])
        # No individual cute_flash_* line should be emitted.
        self.assertFalse(
            any(
                "- cute_flash_x" in ln and "feature(s)" not in ln for ln in lines
            )
        )

    def test_no_generic_line_when_none_collapsible(self) -> None:
        disabled = ["epilogue_subtile: Not a matmul-like kernel"]
        lines = _capture(_summary(dimensions=[], disabled_features=disabled))
        self.assertFalse(
            any("feature(s) not supported by" in ln for ln in lines)
        )


class TestExplorationReportSummary(unittest.TestCase):
    def test_report_summary_uses_total_options(self) -> None:
        stats = [
            FeatureExplorationStats(
                feature_name="block_sizes",
                all_possible_values=[],
                tested_values=[(32,)] * 1,
                coverage_percent=0.024,
                total_options=4096,
            )
        ]
        report = ExplorationReport(
            kernel_name="k",
            backend="triton",
            search_algorithm="LFBOTreeSearch",
            elapsed_seconds=1.0,
            configs_tested=1,
            total_search_space_size=138240,
            feature_stats=stats,
            avg_feature_coverage=0.024,
            min_feature_coverage=0.024,
        )
        logger = logging.getLogger("test_search_space_logger.report")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _ListHandler()
        logger.handlers = [handler]
        report.log_summary(logger)
        joined = "\n".join(handler.lines)
        # Denominator is 4096 (total_options), never 0.
        self.assertIn("block_sizes: 1/4096 options tested", joined)
        self.assertIn("only 1 of 4096 values tested", joined)


class _FakeSpec:
    """Duck-typed ConfigSpec exposing only what a branch under test reads."""

    def __init__(self, **attrs: object) -> None:
        self.__dict__.update(attrs)


class _FakeLoopSpec:
    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids


class TestAnalyzeDimensionSize(unittest.TestCase):
    def test_loop_orders_uses_factorial_of_block_ids_per_spec(self) -> None:
        """A single spec permuting 3 block_ids => 3! = 6 orderings, not 1."""
        spec = _FakeSpec(loop_orders=[_FakeLoopSpec([0, 1, 2])])
        dim = _analyze_dimension_size(spec, "loop_orders")
        assert dim is not None
        self.assertEqual(dim.size, 6)
        self.assertIn("1 permutable loop", dim.constrained_by or "")

    def test_loop_orders_product_across_specs(self) -> None:
        """Two specs (3! and 2!) => 6 * 2 = 12; single-id specs contribute 1."""
        spec = _FakeSpec(
            loop_orders=[
                _FakeLoopSpec([0, 1, 2]),
                _FakeLoopSpec([3, 4]),
                _FakeLoopSpec([5]),  # single id: only one ordering
            ]
        )
        dim = _analyze_dimension_size(spec, "loop_orders")
        assert dim is not None
        self.assertEqual(dim.size, 12)
        self.assertIn("2 permutable loop", dim.constrained_by or "")

    def test_l2_groupings_seven_values_per_slot(self) -> None:
        """PowerOfTwoFragment(1, 64, 1) => {1,2,4,8,16,32,64} = 7 per slot."""
        spec = _FakeSpec(l2_groupings=[object(), object()])
        dim = _analyze_dimension_size(spec, "l2_groupings")
        assert dim is not None
        self.assertEqual(dim.size, 7**2)

    def test_pid_type_reports_disabled_values(self) -> None:
        spec = _FakeSpec(allowed_pid_types=("flat", "xyz"))
        dim = _analyze_dimension_size(spec, "pid_type")
        assert dim is not None
        self.assertEqual(dim.size, 2)
        self.assertEqual(dim.values, ["flat", "xyz"])
        # The two persistent pid_types are reported as disabled.
        self.assertIn("2 pid_type(s) disabled", dim.constrained_by or "")
        self.assertIn("persistent_blocked", dim.constrained_by or "")
        self.assertIn("persistent_interleaved", dim.constrained_by or "")

    def test_pid_type_no_constraint_when_all_allowed(self) -> None:
        spec = _FakeSpec(
            allowed_pid_types=(
                "flat",
                "xyz",
                "persistent_blocked",
                "persistent_interleaved",
            )
        )
        dim = _analyze_dimension_size(spec, "pid_type")
        assert dim is not None
        self.assertEqual(dim.size, 4)
        self.assertIsNone(dim.constrained_by)


class TestDisallowPidTypeReasons(unittest.TestCase):
    def _spec(self) -> object:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        return ConfigSpec(backend=TritonBackend())

    def test_reason_recorded_on_disallow(self) -> None:
        spec = self._spec()
        spec.disallow_pid_type("xyz", reason="grid too large for y/z")
        self.assertEqual(
            spec.disallowed_pid_type_reasons["xyz"], "grid too large for y/z"
        )
        self.assertNotIn("xyz", spec.allowed_pid_types)

    def test_first_reason_wins(self) -> None:
        """A pid_type disabled twice keeps the first reason (already removed the
        second time)."""
        spec = self._spec()
        spec.disallow_pid_type("xyz", reason="first")
        spec.disallow_pid_type("xyz", reason="second")
        self.assertEqual(spec.disallowed_pid_type_reasons["xyz"], "first")

    def test_no_reason_leaves_map_empty(self) -> None:
        spec = self._spec()
        spec.disallow_pid_type("xyz")
        self.assertNotIn("xyz", spec.disallowed_pid_type_reasons)

    def test_analyze_search_space_surfaces_reason(self) -> None:
        spec = self._spec()
        spec.disallow_pid_type("xyz", reason="grid too large for y/z")
        summary = analyze_search_space(spec, kernel_name="k")
        pid_constraints = [
            c for c in summary.shape_constraints if c.startswith("pid_type restricted")
        ]
        self.assertEqual(len(pid_constraints), 1)
        self.assertIn("xyz (grid too large for y/z)", pid_constraints[0])

    def test_analyze_search_space_disabled_without_reason(self) -> None:
        """A pid_type disabled with no reason still lists the bare name."""
        spec = self._spec()
        spec.disallow_pid_type("xyz")
        summary = analyze_search_space(spec, kernel_name="k")
        pid_constraints = [
            c for c in summary.shape_constraints if c.startswith("pid_type restricted")
        ]
        self.assertEqual(len(pid_constraints), 1)
        self.assertIn("disabled: xyz", pid_constraints[0])
        self.assertNotIn("xyz (", pid_constraints[0])

    def test_stale_reason_ignored_when_pid_type_reallowed(self) -> None:
        """If a pid_type is disallowed-with-reason then later re-added to
        allowed_pid_types directly, the stale reason must not appear."""
        spec = self._spec()
        spec.disallow_pid_type("xyz", reason="temporarily out")
        # Re-widen allowed_pid_types directly (as some backends do), leaving the
        # stale reason behind in disallowed_pid_type_reasons.
        spec.allowed_pid_types = (*spec.allowed_pid_types, "xyz")
        self.assertIn("xyz", spec.disallowed_pid_type_reasons)  # still recorded
        summary = analyze_search_space(spec, kernel_name="k")
        pid_constraints = [
            c for c in summary.shape_constraints if c.startswith("pid_type restricted")
        ]
        # xyz is allowed again, so it is neither listed nor annotated.
        for c in pid_constraints:
            self.assertNotIn("xyz", c)


class TestRestrictionReasons(unittest.TestCase):
    """Non-pid_type restrictions (recorded via ``restriction_reasons``) are
    surfaced in the summary, and verbose mode logs restrictions live."""

    def _spec(self) -> object:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        return ConfigSpec(backend=TritonBackend())

    def test_analyze_search_space_surfaces_restriction(self) -> None:
        spec = self._spec()
        spec.restriction_reasons.append(("tcgen05 cluster_m restricted to [1]", "why"))
        summary = analyze_search_space(spec, kernel_name="k")
        self.assertIn(
            "tcgen05 cluster_m restricted to [1] (why)", summary.shape_constraints
        )

    def test_record_restriction_dedupes_repeat(self) -> None:
        """The same restriction applied repeatedly (e.g. once per matmul on a
        shared ConfigSpec) is recorded once, so the summary has no duplicates."""
        from helion.autotuner import config_spec as cs

        store: list[tuple[str, str]] = []
        cs._record_restriction(store, "tcgen05 narrowed", "matmul cute backend", False)
        cs._record_restriction(store, "tcgen05 narrowed", "matmul cute backend", False)
        self.assertEqual(store, [("tcgen05 narrowed", "matmul cute backend")])

    def test_disallow_pid_type_logs_live_when_verbose(self) -> None:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner import config_spec as cs
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=TritonBackend(), log_restrictions_verbose=True)
        with self.assertLogs(cs.log, level="INFO") as captured:
            spec.disallow_pid_type("xyz", reason="grid too large")
        self.assertTrue(any("xyz" in line for line in captured.output))

    def test_no_live_log_when_flag_off(self) -> None:
        """With the flag off, disallow records the reason but emits no INFO log."""
        from helion._compiler.backend import TritonBackend
        from helion.autotuner import config_spec as cs
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=TritonBackend(), log_restrictions_verbose=False)
        with self.assertNoLogs(cs.log, level="INFO"):
            spec.disallow_pid_type("xyz", reason="grid too large")
        self.assertEqual(spec.disallowed_pid_type_reasons["xyz"], "grid too large")


class TestSaveOutputPathHandling(unittest.TestCase):
    """save_search_space_summary / save_exploration_report must never crash the
    autotuner on awkward output paths (directory, missing parents, etc.) and
    must produce per-kernel/per-hash filenames that don't clobber each other."""

    def _summary_obj(self, kernel_name: str = "kernel_under_test") -> SearchSpaceSummary:
        dims = [
            SearchSpaceDimension(
                name="num_warps",
                dim_type="discrete",
                size=6,
                values=[1, 2, 4, 8, 16, 32],
            )
        ]
        summary = _summary(dimensions=dims)
        summary.kernel_name = kernel_name
        return summary

    def _report_obj(self) -> ExplorationReport:
        return ExplorationReport(
            kernel_name="k",
            backend="triton",
            search_algorithm="Test",
            elapsed_seconds=1.0,
            configs_tested=3,
            total_search_space_size=6,
            feature_stats=[],
            avg_feature_coverage=0.0,
            min_feature_coverage=0.0,
        )

    def test_save_embeds_kernel_and_hash(self) -> None:
        """The saved filename embeds the kernel name and cache hash tokens."""
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "out.json")
            saved = save_search_space_summary(
                self._summary_obj(), 3, "Test", 1.0, target, "deadbeef"
            )
            self.assertEqual(
                saved,
                os.path.join(d, "out.kernel_under_test.deadbeef.json"),
            )
            self.assertTrue(os.path.isfile(saved))
            json.loads(Path(saved).read_text())

    def test_hash_determines_path(self) -> None:
        """Distinct hashes give distinct files; the same hash reuses one file."""
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "out.json")
            a = save_search_space_summary(
                self._summary_obj("kernel_a"), 3, "Test", 1.0, target, "aaaa"
            )
            b = save_search_space_summary(
                self._summary_obj("kernel_b"), 3, "Test", 1.0, target, "bbbb"
            )
            # Different kernel/hash -> distinct files (no clobber).
            self.assertNotEqual(a, b)
            self.assertTrue(os.path.isfile(a))
            self.assertTrue(os.path.isfile(b))
            # Re-tuning the same kernel/shape (same hash) reuses one file.
            a2 = save_search_space_summary(
                self._summary_obj("kernel_a"), 5, "Test", 2.0, target, "aaaa"
            )
            self.assertEqual(a, a2)

    def test_save_without_hash(self) -> None:
        """With no cache hash: kernel-only filename, and existing files get a
        numeric suffix instead of being clobbered."""
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "out.json")
            first = save_search_space_summary(
                self._summary_obj(), 3, "Test", 1.0, target, None
            )
            self.assertEqual(first, os.path.join(d, "out.kernel_under_test.json"))
            self.assertTrue(os.path.isfile(first))
            # A second save with no distinguishing hash must not clobber.
            second = save_search_space_summary(
                self._summary_obj(), 3, "Test", 1.0, target, None
            )
            self.assertNotEqual(first, second)
            self.assertTrue(os.path.isfile(second))

    def test_save_to_existing_directory(self) -> None:
        """A directory path gets a default filename appended, not a crash."""
        with tempfile.TemporaryDirectory() as d:
            saved = save_search_space_summary(
                self._summary_obj(), 3, "Test", 1.0, d, "deadbeef"
            )
            self.assertEqual(
                saved,
                os.path.join(
                    d, "autotune_search_space.kernel_under_test.deadbeef.json"
                ),
            )
            self.assertTrue(os.path.isfile(saved))

    def test_save_to_trailing_separator_directory(self) -> None:
        """A path ending in a separator is treated as a (possibly new) dir."""
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "nested") + os.sep
            saved = save_search_space_summary(
                self._summary_obj(), 3, "Test", 1.0, target, "deadbeef"
            )
            self.assertEqual(
                saved,
                os.path.join(
                    d, "nested", "autotune_search_space.kernel_under_test.deadbeef.json"
                ),
            )
            self.assertTrue(os.path.isfile(saved))

    def test_exploration_report_skipped_when_summary_failed(self) -> None:
        """An empty summary path (save failed) means nothing to write."""
        saved = save_exploration_report(self._report_obj(), "")
        self.assertEqual(saved, "")

    def test_exploration_report_only_swaps_extension(self) -> None:
        """A ".json" earlier in the stem must not be substituted."""
        with tempfile.TemporaryDirectory() as d:
            # Summary stem itself contains ".json" (e.g. a kernel named
            # "foo.json"); only the trailing extension should be swapped.
            summary_path = os.path.join(d, "out.foo.json.deadbeef.json")
            saved = save_exploration_report(self._report_obj(), summary_path)
            self.assertEqual(
                saved,
                os.path.join(d, "out.foo.json.deadbeef_exploration.json"),
            )
            self.assertTrue(os.path.isfile(saved))
            json.loads(Path(saved).read_text())

    def test_save_never_raises_on_bad_path(self) -> None:
        """A write failure returns an empty string instead of raising."""
        with tempfile.TemporaryDirectory() as d:
            # A path whose parent is an existing *file* cannot be created as a
            # directory, so mkdir/write fails; the function must swallow it.
            blocker = os.path.join(d, "blocker")
            Path(blocker).write_text("x")
            target = os.path.join(blocker, "out.json")
            saved = save_search_space_summary(
                self._summary_obj(), 3, "Test", 1.0, target, "deadbeef"
            )
            self.assertEqual(saved, "")

    def test_exploration_report_never_raises_on_bad_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "blocker")
            Path(blocker).write_text("x")
            summary_path = os.path.join(blocker, "out.json")
            saved = save_exploration_report(self._report_obj(), summary_path)
            self.assertEqual(saved, "")


if __name__ == "__main__":
    unittest.main()