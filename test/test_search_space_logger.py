from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest

from helion.autotuner.config_spec import SearchDimensionInfo
from helion.autotuner.search_space_logger import DimensionStats
from helion.autotuner.search_space_logger import SearchSpaceReport
from helion.autotuner.search_space_logger import SearchSpaceTracker
from helion.autotuner.search_space_logger import _dimension_from_info
from helion.autotuner.search_space_logger import analyze_search_space


@dataclass
class _FakeConfig:
    """Minimal stand-in for runtime.Config used by the tracker.

    ``SearchSpaceTracker`` reads feature values via ``_extract_feature_value``
    and a canonical config id via ``.config``.
    """

    block_sizes: list[int] = field(kw_only=True)
    loop_orders: list[int] = field(kw_only=True)
    num_warps: int = field(kw_only=True)
    epilogue_subtile: object = field(default=None, kw_only=True)

    @property
    def config(self) -> dict[str, object]:
        return {
            "block_sizes": list(self.block_sizes),
            "loop_orders": list(self.loop_orders),
            "num_warps": self.num_warps,
            "epilogue_subtile": self.epilogue_subtile,
        }

    def get(self, key: str, default: object = None) -> object:
        return self.config.get(key, default)


def _report(
    *,
    dimensions: list[DimensionStats],
    disabled_features: list[str] | None = None,
    shape_constraints: list[str] | None = None,
    backend: str = "triton",
    total: int | None = 138240,
    kernel_name: str = "kernel_under_test",
) -> SearchSpaceReport:
    return SearchSpaceReport(
        kernel_name=kernel_name,
        specialization_key=None,
        backend=backend,
        hardware="NVIDIA H100",
        dimensions=dimensions,
        total_search_space_size=total,
        disabled_features=disabled_features or [],
        shape_constraints=shape_constraints or [],
    )


def _dim(
    name: str, cardinality: int, values: list[object] | None = None
) -> DimensionStats:
    return DimensionStats(
        name=name,
        dim_type="discrete",
        cardinality=cardinality,
        possible_values=values,
    )


class TestDimensionStats(unittest.TestCase):
    def test_summary_line_and_coverage(self) -> None:
        dim = _dim("num_warps", 6, [1, 2, 4, 8, 16, 32])
        for w in (1, 2, 4):
            dim.observe(w)
        self.assertEqual(dim.observed_count, 3)
        self.assertEqual(dim.coverage_percent, 50.0)
        self.assertEqual(dim.to_summary_line(), "num_warps: 3/6 options tested (50.0%)")

    def test_zero_cardinality_renders_not_applicable(self) -> None:
        dim = _dim("epilogue_subtile", 0)
        self.assertFalse(dim.applicable)
        self.assertIn("no autotunable choices", dim.to_summary_line())

    def test_observed_count_clamped_to_cardinality(self) -> None:
        """A deliberately-too-small cardinality estimate clamps the count."""
        dim = _dim("l2_groupings", 4)
        for v in (1, 2, 4, 8, 16):  # 5 distinct, estimate is 4
            dim.observe(v)
        self.assertEqual(dim.observed_count, 4)
        self.assertEqual(dim.coverage_percent, 100.0)


class TestSearchSpaceTracker(unittest.TestCase):
    def test_coverage_denominator_uses_cardinality_when_values_absent(self) -> None:
        """Regression: block_sizes/loop_orders must not divide by zero."""
        dims = [
            _dim("block_sizes", 4096),
            _dim("loop_orders", 24),
            _dim("num_warps", 6, [1, 2, 4, 8, 16, 32]),
        ]
        tracker = SearchSpaceTracker(_report(dimensions=dims))
        for i in range(16):
            tracker.record_config(
                _FakeConfig(
                    block_sizes=[32 + i, 64],
                    loop_orders=[0, 1] if i % 2 else [1, 0],
                    num_warps=[1, 2, 4, 8, 16, 32][i % 6],
                )
            )
        report = tracker.finish("LFBOTreeSearch", 502.5)
        by_name = {d.name: d for d in report.dimensions}

        self.assertEqual(by_name["block_sizes"].cardinality, 4096)
        self.assertEqual(by_name["block_sizes"].observed_count, 16)
        self.assertAlmostEqual(by_name["block_sizes"].coverage_percent, 16 / 4096 * 100)

        self.assertEqual(by_name["loop_orders"].cardinality, 24)
        self.assertEqual(by_name["loop_orders"].observed_count, 2)

        self.assertEqual(by_name["num_warps"].coverage_percent, 100.0)

        # The rendered summary uses cardinality as the denominator, and flags
        # the under-explored dimension.
        joined = "\n".join(_capture(report, "coverage"))
        self.assertIn("block_sizes: 16/4096 options tested", joined)
        self.assertIn("only 16 of 4096 values tested", joined)

    def test_empty_dimension_reports_zero_not_crash(self) -> None:
        dims = [_dim("loop_orders", 0)]
        tracker = SearchSpaceTracker(_report(dimensions=dims))
        report = tracker.finish("LFBOTreeSearch", 1.0)
        self.assertEqual(report.dimensions[0].cardinality, 0)
        self.assertEqual(report.dimensions[0].coverage_percent, 0.0)

    def test_zero_option_feature_excluded_from_aggregates(self) -> None:
        dims = [_dim("epilogue_subtile", 0), _dim("num_warps", 6, [1, 2, 4, 8, 16, 32])]
        tracker = SearchSpaceTracker(_report(dimensions=dims))
        for w in (1, 2, 4, 8, 16, 32):
            tracker.record_config(
                _FakeConfig(block_sizes=[32], loop_orders=[0], num_warps=w)
            )
        report = tracker.finish("LFBOTreeSearch", 1.0)
        self.assertEqual(report.avg_feature_coverage, 100.0)
        self.assertEqual(report.min_feature_coverage, 100.0)

    def test_distinct_configs_counted_by_canonical_id(self) -> None:
        """Re-attempting an identical config counts once (canonical id set)."""
        tracker = SearchSpaceTracker(_report(dimensions=[_dim("num_warps", 6)]))
        for _ in range(3):
            tracker.record_config(
                _FakeConfig(block_sizes=[32], loop_orders=[0], num_warps=4)
            )
        report = tracker.finish("RandomSearch", 1.0)
        self.assertEqual(report.configs_tested, 1)
        self.assertEqual(report.explored_valid, 3)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _capture(
    report: SearchSpaceReport, name: str = "test_search_space_logger"
) -> list[str]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _ListHandler()
    logger.handlers = [handler]
    report.log_summary(logger)
    return handler.lines


class TestExploredValidityTracking(unittest.TestCase):
    """Explored-config validity breakdown (valid vs. rejected candidates)."""

    def _finished(
        self, *, valid_configs: list[_FakeConfig], invalid: int
    ) -> SearchSpaceReport:
        tracker = SearchSpaceTracker(
            _report(dimensions=[_dim("num_warps", 6)], total=100)
        )
        for cfg in valid_configs:
            tracker.record_config(cfg)
        tracker.record_invalid(invalid)
        return tracker.finish("RandomSearch", 1.0)

    def _cfg(self, w: int) -> _FakeConfig:
        return _FakeConfig(block_sizes=[32], loop_orders=[0], num_warps=w)

    def test_valid_and_invalid_counts(self) -> None:
        report = self._finished(
            valid_configs=[self._cfg(w) for w in (1, 2, 4, 8)], invalid=2
        )
        self.assertEqual(report.explored_valid, 4)
        self.assertEqual(report.explored_invalid, 2)
        self.assertEqual(report.explored_total, 6)
        self.assertAlmostEqual(report.explored_valid_percent, 4 / 6 * 100)

    def test_record_invalid_ignores_nonpositive(self) -> None:
        tracker = SearchSpaceTracker(_report(dimensions=[]))
        tracker.record_invalid(0)
        tracker.record_invalid(-5)
        self.assertEqual(tracker.invalid_config_count, 0)

    def test_zero_attempts_percent(self) -> None:
        report = self._finished(valid_configs=[], invalid=0)
        self.assertEqual(report.explored_total, 0)
        self.assertEqual(report.explored_valid_percent, 0.0)

    def test_log_summary_breakdown(self) -> None:
        report = self._finished(
            valid_configs=[self._cfg(w) for w in (1, 2, 4, 8, 16, 32, 1, 2)], invalid=2
        )
        # explored_valid is a raw attempt count (8), on the same scale as
        # explored_invalid; configs_tested is the distinct count (6).
        self.assertEqual(report.explored_valid, 8)
        self.assertEqual(report.configs_tested, 6)
        joined = "\n".join(_capture(report, "validity"))
        self.assertIn("Configs attempted: 10", joined)
        self.assertIn("8 valid", joined)
        self.assertIn("2 invalid", joined)

        empty = self._finished(valid_configs=[], invalid=0)
        joined = "\n".join(_capture(empty, "validity_empty"))
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
        lines = _capture(_report(dimensions=[], disabled_features=disabled))
        joined = "\n".join(lines)

        self.assertIn("Disabled features (80):", joined)
        self.assertIn("- epilogue_subtile: Not a matmul-like kernel", joined)
        self.assertIn("- pallas_loop_type: Triton backend (no Pallas loops)", joined)
        summary_lines = [
            ln for ln in lines if "feature(s) not supported by triton backend" in ln
        ]
        self.assertEqual(len(summary_lines), 1)
        self.assertIn("78 feature(s) not supported by triton backend", summary_lines[0])
        self.assertFalse(
            any("- cute_flash_x" in ln and "feature(s)" not in ln for ln in lines)
        )

    def test_no_generic_line_when_none_collapsible(self) -> None:
        disabled = ["epilogue_subtile: Not a matmul-like kernel"]
        lines = _capture(_report(dimensions=[], disabled_features=disabled))
        self.assertFalse(any("feature(s) not supported by" in ln for ln in lines))


class _FakeSpec:
    """Duck-typed ConfigSpec exposing only what a branch under test reads."""

    def __init__(self, **attrs: object) -> None:
        self.__dict__.update(attrs)


class TestDimensionFromInfo(unittest.TestCase):
    def test_pid_type_reports_disabled_values(self) -> None:
        spec = _FakeSpec(
            allowed_pid_types=("flat", "xyz"), tensor_numel_constraints=None
        )
        info = SearchDimensionInfo(
            name="pid_type",
            cardinality=2,
            values=["flat", "xyz"],
            is_sequence=False,
            num_items=0,
        )
        dim = _dimension_from_info(info, spec)
        assert dim is not None
        self.assertEqual(dim.cardinality, 2)
        self.assertEqual(dim.possible_values, ["flat", "xyz"])
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
            ),
            tensor_numel_constraints=None,
        )
        info = SearchDimensionInfo(
            name="pid_type",
            cardinality=4,
            values=["flat", "xyz", "persistent_blocked", "persistent_interleaved"],
            is_sequence=False,
            num_items=0,
        )
        dim = _dimension_from_info(info, spec)
        assert dim is not None
        self.assertEqual(dim.cardinality, 4)
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
        report = analyze_search_space(spec, kernel_name="k")
        pid_constraints = [
            c for c in report.shape_constraints if c.startswith("pid_type restricted")
        ]
        self.assertEqual(len(pid_constraints), 1)
        self.assertIn("xyz (grid too large for y/z)", pid_constraints[0])

    def test_analyze_search_space_disabled_without_reason(self) -> None:
        spec = self._spec()
        spec.disallow_pid_type("xyz")
        report = analyze_search_space(spec, kernel_name="k")
        pid_constraints = [
            c for c in report.shape_constraints if c.startswith("pid_type restricted")
        ]
        self.assertEqual(len(pid_constraints), 1)
        self.assertIn("disabled: xyz", pid_constraints[0])
        self.assertNotIn("xyz (", pid_constraints[0])

    def test_stale_reason_ignored_when_pid_type_reallowed(self) -> None:
        spec = self._spec()
        spec.disallow_pid_type("xyz", reason="temporarily out")
        spec.allowed_pid_types = (*spec.allowed_pid_types, "xyz")
        self.assertIn("xyz", spec.disallowed_pid_type_reasons)
        report = analyze_search_space(spec, kernel_name="k")
        pid_constraints = [
            c for c in report.shape_constraints if c.startswith("pid_type restricted")
        ]
        for c in pid_constraints:
            self.assertNotIn("xyz", c)


class TestTotalSearchSpaceSize(unittest.TestCase):
    """The reported total is the exact combinatorial product of per-dimension
    cardinalities (arbitrary precision, never truncated), or None ('unknown')
    when a dimension's cardinality can't be determined."""

    def _spec(self) -> object:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        return ConfigSpec(backend=TritonBackend())

    def test_large_product_not_truncated(self) -> None:
        """A space far above 1e12 reports the exact big int, not None/'unknown'."""
        from helion.autotuner.config_spec import SearchDimensionInfo

        spec = self._spec()
        # 4^4 * 3^7 * ... style: force a >1e12 product via controlled dims.
        dims = [
            SearchDimensionInfo("indexing", 256, None, False, 0),
            SearchDimensionInfo("load_eviction_policies", 2187, None, False, 0),
            SearchDimensionInfo("a", 25, None, False, 0),
            SearchDimensionInfo("b", 25, None, False, 0),
            SearchDimensionInfo("c", 9, None, False, 0),
            SearchDimensionInfo("d", 9, None, False, 0),
            SearchDimensionInfo("e", 8, None, False, 0),
            SearchDimensionInfo("f", 8, None, False, 0),
            SearchDimensionInfo("g", 6, None, False, 0),
        ]
        spec.iter_search_dimensions = lambda value_limit=100: iter(dims)  # type: ignore[method-assign]
        report = analyze_search_space(spec, kernel_name="k")
        expected = 256 * 2187 * 25 * 25 * 9 * 9 * 8 * 8 * 6
        self.assertEqual(report.total_search_space_size, expected)
        self.assertGreater(expected, 10**12)
        # Rendered as the exact integer, never "unknown"/"infinite".
        self.assertEqual(report.to_dict()["total_search_space_size"], str(expected))

    def test_unknown_cardinality_makes_total_unknown(self) -> None:
        """A dimension whose cardinality is unknown (0 sentinel) -> None/'unknown'."""
        from helion.autotuner.config_spec import SearchDimensionInfo

        spec = self._spec()
        dims = [
            SearchDimensionInfo("known", 8, None, False, 0),
            SearchDimensionInfo("custom", 0, None, False, 0),  # unreportable
        ]
        spec.iter_search_dimensions = lambda value_limit=100: iter(dims)  # type: ignore[method-assign]
        report = analyze_search_space(spec, kernel_name="k")
        self.assertIsNone(report.total_search_space_size)
        self.assertEqual(report.to_dict()["total_search_space_size"], "unknown")


class TestRestrictionReasons(unittest.TestCase):
    def _spec(self) -> object:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        return ConfigSpec(backend=TritonBackend())

    def test_analyze_search_space_surfaces_restriction(self) -> None:
        spec = self._spec()
        spec.restriction_reasons.append(("tcgen05 cluster_m restricted to [1]", "why"))
        report = analyze_search_space(spec, kernel_name="k")
        self.assertIn(
            "tcgen05 cluster_m restricted to [1] (why)", report.shape_constraints
        )

    def test_record_restriction_dedupes_repeat(self) -> None:
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
        from helion._compiler.backend import TritonBackend
        from helion.autotuner import config_spec as cs
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=TritonBackend(), log_restrictions_verbose=False)
        with self.assertNoLogs(cs.log, level="INFO"):
            spec.disallow_pid_type("xyz", reason="grid too large")
        self.assertEqual(spec.disallowed_pid_type_reasons["xyz"], "grid too large")


class TestSaveReport(unittest.TestCase):
    """report.save() must never crash on awkward paths and must produce
    per-kernel/per-hash filenames that don't clobber each other."""

    def _report_obj(self, kernel_name: str = "kernel_under_test") -> SearchSpaceReport:
        dim = _dim("num_warps", 6, [1, 2, 4, 8, 16, 32])
        return _report(dimensions=[dim], kernel_name=kernel_name)

    def test_save_embeds_kernel_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "out.json")
            saved = self._report_obj().save(target, "deadbeef")
            self.assertEqual(
                saved, os.path.join(d, "out.kernel_under_test.deadbeef.json")
            )
            self.assertTrue(os.path.isfile(saved))
            json.loads(Path(saved).read_text())

    def test_hash_determines_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "out.json")
            a = self._report_obj("kernel_a").save(target, "aaaa")
            b = self._report_obj("kernel_b").save(target, "bbbb")
            self.assertNotEqual(a, b)
            self.assertTrue(os.path.isfile(a))
            self.assertTrue(os.path.isfile(b))
            a2 = self._report_obj("kernel_a").save(target, "aaaa")
            self.assertEqual(a, a2)

    def test_save_without_hash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "out.json")
            first = self._report_obj().save(target, None)
            self.assertEqual(first, os.path.join(d, "out.kernel_under_test.json"))
            self.assertTrue(os.path.isfile(first))
            second = self._report_obj().save(target, None)
            self.assertNotEqual(first, second)
            self.assertTrue(os.path.isfile(second))

    def test_save_to_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            saved = self._report_obj().save(d, "deadbeef")
            self.assertEqual(
                saved,
                os.path.join(
                    d, "autotune_search_space.kernel_under_test.deadbeef.json"
                ),
            )
            self.assertTrue(os.path.isfile(saved))

    def test_save_to_trailing_separator_directory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "nested") + os.sep
            saved = self._report_obj().save(target, "deadbeef")
            self.assertEqual(
                saved,
                os.path.join(
                    d, "nested", "autotune_search_space.kernel_under_test.deadbeef.json"
                ),
            )
            self.assertTrue(os.path.isfile(saved))

    def test_save_never_raises_on_bad_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "blocker")
            Path(blocker).write_text("x")
            target = os.path.join(blocker, "out.json")
            saved = self._report_obj().save(target, "deadbeef")
            self.assertEqual(saved, "")


if __name__ == "__main__":
    unittest.main()
