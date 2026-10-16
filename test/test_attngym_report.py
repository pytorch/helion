from __future__ import annotations

from contextlib import redirect_stdout
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.cute.attngym_report import FIGURE_GROUPS
from benchmarks.cute.attngym_report import FIGURE_IMPLS
from benchmarks.cute.attngym_report import aggregate
from benchmarks.cute.attngym_report import main
from benchmarks.cute.attngym_report import read_records
from benchmarks.cute.attngym_report import write_csv


def _record(impl, shape_id, latency, *, accuracy="PASS", throughput=100.0, **fields):
    return {
        "suite": "figure",
        "variant": "causal",
        "shape_id": shape_id,
        "shape": {
            "b": 1,
            "hq": 8,
            "hkv": 8,
            "m": 1024,
            "n": 1024,
            "d": 64,
            "dv": 64,
            "dtype": "bfloat16",
        },
        "impl": impl,
        "accuracy": accuracy,
        "median_ms": latency,
        "median_tflops": throughput,
        "gpu": "NVIDIA GB300",
        "physical_gpu": "2",
        "helion_commit": "abc123",
        "input_seed": 0,
        "benchmark_timer": "event",
        "benchmark_statistic": "median",
        "num_runs": 5,
        "warmup_ms": 1000,
        "rep_ms": 500,
        "thermal_warmup_ms": 10000,
        "cooldown_temp": 0.0,
        "skip_correctness": False,
        "accuracy_atol": 0.05,
        "accuracy_rtol": 0.02,
        **fields,
    }


def _variant(report, name="causal"):
    return next(row for row in report["variants"] if row["variant"] == name)


def _terminal_case(shape_id, *, variant="causal", **fields):
    return [
        _record(
            impl,
            shape_id,
            2.0 if impl == "helion-cute" else 4.0 if impl == "flex-triton" else None,
            accuracy="PASS" if impl in {"helion-cute", "flex-triton"} else "SKIP",
            throughput=100.0 if impl in {"helion-cute", "flex-triton"} else None,
            variant=variant,
            **fields,
        )
        for impl in FIGURE_IMPLS
    ]


def _case(shape_id, *, variant="causal", **dimensions):
    shape = _record("helion-cute", shape_id, 2.0)["shape"]
    return {"variant": variant, "shape_id": shape_id, "shape": {**shape, **dimensions}}


def _manifest(*cases, **fields):
    return {"suite": "figure", "cases": list(cases), **fields}


class TestAttentionGymReport(unittest.TestCase):
    def test_fastest_baseline_is_selected_per_shape(self):
        report = aggregate(
            [
                _record("helion-cute", "small", 2.0, throughput=100.0),
                _record("flex-triton", "small", 8.0),
                _record("fa4", "small", 4.0),
                _record("helion-cute", "large", 4.0, throughput=300.0),
                _record("flex-triton", "large", 2.0),
                _record("fa4", "large", 16.0),
            ],
            expected_shapes=2,
        )
        # Ratios are 2 and 0.5, not a ratio of aggregate backend timings.
        self.assertAlmostEqual(report["geomean_speedup_vs_best_other"], 1.0)
        self.assertAlmostEqual(_variant(report)["speedup_vs_best_other"], 1.0)
        self.assertEqual(_variant(report)["impls"]["helion-cute"]["mean_tflops"], 200.0)
        self.assertEqual(
            {row["best_other_impl"] for row in report["shapes"]}, {"fa4", "flex-triton"}
        )
        self.assertTrue(report["comparison_complete"])
        self.assertFalse(report["campaign_complete"])
        self.assertFalse(report["complete"])

    def test_failed_and_unavailable_results_cannot_improve_score(self):
        report = aggregate(
            [
                _record("helion-cute", "small", 2.0),
                _record("flex-triton", "small", 6.0),
                _record("fa4", "small", 0.01, accuracy="FAIL", throughput=99999.0),
                _record("sdpa", "small", None, accuracy="SKIP", throughput=None),
                _record("flex-cute", "small", None, accuracy="ERROR", throughput=None),
                _record(
                    "helion-cute", "large", 0.01, accuracy="FAIL", throughput=99999.0
                ),
                _record("flex-triton", "large", 4.0),
            ],
            expected_shapes=2,
        )
        row = _variant(report)
        self.assertFalse(report["complete"])
        self.assertEqual(report["compared_shapes"], 1)
        self.assertEqual(report["uncompared_observed_shapes"], 1)
        self.assertAlmostEqual(report["geomean_speedup_vs_best_other"], 3.0)
        self.assertIsNone(row["impls"]["fa4"]["mean_tflops"])
        self.assertEqual(row["impls"]["fa4"]["fail_shapes"], 1)
        self.assertEqual(row["impls"]["sdpa"]["skip_shapes"], 1)
        self.assertEqual(row["impls"]["flex-cute"]["error_shapes"], 1)
        self.assertEqual(row["impls"]["helion-cute"]["mean_tflops"], 100.0)
        self.assertEqual(row["impls"]["helion-cute"]["status"], "PARTIAL")
        self.assertIn("1/2", row["impls"]["helion-cute"]["display"])
        self.assertIn("INCOMPLETE", report["coverage"])

    def test_legacy_shapes_are_not_collapsed(self):
        records = []
        for length, latency in ((1024, 2.0), (4096, 8.0)):
            for impl in ("helion-cute", "flex-triton"):
                record = _record(impl, "unused", latency)
                record.pop("shape_id")
                record["shape"]["m"] = length
                records.append(record)
        report = aggregate(records, expected_shapes=2)
        self.assertEqual(report["observed_shapes"], 2)
        self.assertEqual(report["compared_shapes"], 2)
        self.assertEqual({row["shape"]["m"] for row in report["shapes"]}, {1024, 4096})

    def test_duplicate_cells_and_mixed_measurements_are_rejected(self):
        record = _record("helion-cute", "small", 2.0)
        with self.assertRaisesRegex(ValueError, "Duplicate result cell"):
            aggregate([record, {**record, "median_ms": 1.0}])
        for field, value in (
            ("gpu", "NVIDIA B200"),
            ("physical_gpu", "3"),
            ("helion_commit", "def456"),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"Mixed {field}"),
            ):
                aggregate(
                    [record, _record("flex-triton", "small", 2.0, **{field: value})]
                )

    def test_shape_id_cannot_hide_different_dimensions(self):
        helion = _record("helion-cute", "small", 2.0)
        baseline = _record("flex-triton", "small", 2.0)
        baseline["shape"]["m"] = 4096
        with self.assertRaisesRegex(ValueError, "Inconsistent shape metadata"):
            aggregate([helion, baseline])

    def test_different_cases_may_use_different_gpus_of_the_same_model(self):
        report = aggregate(
            [
                _record("helion-cute", "small", 2.0, physical_gpu="0"),
                _record("flex-triton", "small", 4.0, physical_gpu="0"),
                _record("helion-cute", "large", 4.0, physical_gpu="2"),
                _record("flex-triton", "large", 8.0, physical_gpu="2"),
            ],
            expected_shapes=2,
        )
        self.assertTrue(report["comparison_complete"])
        self.assertEqual(report["physical_gpus"], ["0", "2"])
        self.assertEqual(
            {row["shape_id"]: row["physical_gpu"] for row in report["shapes"]},
            {"small": "0", "large": "2"},
        )
        self.assertEqual(report["geomean_speedup_vs_best_other"], 2.0)

    def test_missing_shapes_and_missing_baselines_remain_visible(self):
        report = aggregate([_record("helion-cute", "small", 2.0)], expected_shapes=87)
        self.assertFalse(report["complete"])
        self.assertEqual(report["compared_shapes"], 0)
        self.assertEqual(report["unobserved_expected_shapes"], 86)
        self.assertEqual(report["shapes"][0]["reason"], "No correct baseline")
        self.assertIsNone(report["geomean_speedup_vs_best_other"])
        self.assertEqual(_variant(report)["impls"]["fa4"]["status"], "MISSING")
        self.assertEqual(len(report["variants"]), 29)
        self.assertIn("0/87", report["coverage"])

    def test_no_kernel_wrong_output_and_error_have_distinct_labels(self):
        for status, label in (
            ("SKIP", "no kernel"),
            ("FAIL", "wrong output"),
            ("ERROR", "error"),
        ):
            with self.subTest(status=status):
                report = aggregate(
                    [
                        _record(
                            "helion-cute",
                            "small",
                            None,
                            accuracy=status,
                            throughput=None,
                        )
                    ],
                    expected_shapes=1,
                )
                self.assertEqual(
                    _variant(report)["impls"]["helion-cute"]["display"], label
                )

    def test_pass_requires_valid_latency(self):
        for latency in (None, 0, -1, float("nan"), float("inf")):
            with self.subTest(latency=latency), self.assertRaises(ValueError):
                aggregate([_record("helion-cute", "small", latency)])

    def test_helion_triton_is_not_a_hidden_baseline(self):
        report = aggregate(
            [
                _record("helion-cute", "small", 2.0),
                _record("flex-triton", "small", 4.0),
                _record("helion-triton", "small", 0.01),
            ],
            expected_shapes=1,
        )
        self.assertEqual(report["geomean_speedup_vs_best_other"], 2.0)
        self.assertEqual(report["excluded_impls"], ["helion-triton"])

    def test_jsonl_and_csv_preserve_partial_results(self):
        records = [
            _record("helion-cute", "small", 2.0),
            _record("flex-triton", "small", 4.0),
            _record("helion-cute", "large", None, accuracy="ERROR", throughput=None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            report = aggregate(read_records([source]), expected_shapes=2)
            output = Path(directory) / "nested" / "table.csv"
            write_csv(output, report)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            row = _variant({"variants": rows})
            self.assertEqual(row["helion-cute_status"], "PARTIAL")
            self.assertEqual(row["helion-cute_pass_shapes"], "1")
            self.assertEqual(row["observed_shapes"], "2")
            self.assertEqual(row["expected_shapes"], "2")
            self.assertEqual(row["compared_shapes"], "1")
            # Summary remains valid strict JSON even with unavailable cells.
            json.dumps(report, allow_nan=False)

    def test_mixed_source_and_manifest_fingerprints_are_rejected_across_cases(self):
        for field in ("source_fingerprint", "manifest_sha256"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"Mixed {field}"),
            ):
                aggregate(
                    [
                        _record("helion-cute", "small", 2.0, **{field: "first-run"}),
                        _record("flex-triton", "large", 4.0, **{field: "second-run"}),
                    ],
                    expected_shapes=2,
                )

    def test_input_timing_and_correctness_settings_must_match_within_a_case(self):
        differences = (
            ("input_seed", 1),
            ("benchmark_timer", "cuda_graph"),
            ("benchmark_statistic", "mean"),
            ("num_runs", 3),
            ("warmup_ms", 500),
            ("rep_ms", 100),
            ("thermal_warmup_ms", 0),
            ("cooldown_temp", 55.0),
            ("skip_correctness", True),
            ("accuracy_atol", 0.01),
            ("accuracy_rtol", 0.005),
        )
        for field, value in differences:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"Mixed {field}"),
            ):
                aggregate(
                    [
                        _record("helion-cute", "small", 2.0),
                        _record("flex-triton", "small", 4.0, **{field: value}),
                    ],
                    expected_shapes=1,
                )

    def test_different_cases_may_use_different_consistent_settings(self):
        report = aggregate(
            [
                *_terminal_case("small", input_seed=0, num_runs=5),
                *_terminal_case("large", input_seed=1, num_runs=3),
            ],
            expected_shapes=2,
        )
        self.assertTrue(report["complete"])
        self.assertEqual(
            {row["shape_id"]: row["input_seed"] for row in report["shapes"]},
            {"small": 0, "large": 1},
        )

    def test_87_cases_of_one_variant_cannot_complete_the_default_figure(self):
        records = [
            row for index in range(87) for row in _terminal_case(f"causal.{index}")
        ]
        report = aggregate(records)
        self.assertFalse(report["complete"])
        self.assertFalse(report["campaign_complete"])
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(report["unobserved_expected_shapes"], 84)
        self.assertEqual(len(report["missing_variants"]), 28)
        self.assertIn("INCOMPLETE", report["coverage"])
        self.assertEqual(_variant(report)["impls"]["helion-cute"]["status"], "PARTIAL")

    def test_default_figure_cells_use_three_expected_shapes(self):
        report = aggregate(_terminal_case("causal.small"))
        row = _variant(report)
        self.assertEqual(row["expected_shapes"], 3)
        self.assertEqual(row["observed_shapes"], 1)
        self.assertEqual(row["missing_shapes"], 2)
        self.assertEqual(row["impls"]["helion-cute"]["status"], "PARTIAL")
        self.assertIn("1/3", row["impls"]["helion-cute"]["display"])
        self.assertEqual(row["impls"]["helion-cute"]["missing_shapes"], 2)
        self.assertEqual(report["unobserved_expected_shapes"], 86)
        self.assertFalse(report["comparison_complete"])

    def test_default_figure_is_complete_with_three_cases_for_all_29_rows(self):
        records = [
            row
            for group, names in FIGURE_GROUPS
            for name in names
            for index in range(3)
            for row in _terminal_case(f"{name}.{index}", variant=name)
        ]
        report = aggregate(records)
        self.assertTrue(report["campaign_complete"])
        self.assertTrue(report["comparison_complete"])
        self.assertTrue(report["complete"])
        self.assertEqual(report["compared_shapes"], 87)
        self.assertEqual(report["expected_backend_cells"], 522)
        self.assertEqual(report["missing_backend_cells"], 0)
        self.assertEqual(report["unobserved_expected_shapes"], 0)

    def test_missing_backend_keeps_a_fully_compared_campaign_incomplete(self):
        records = [row for row in _terminal_case("small") if row["impl"] != "fa4"]
        report = aggregate(records, manifest=_manifest(_case("small")))
        self.assertTrue(report["comparison_complete"])
        self.assertFalse(report["campaign_complete"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["expected_backend_cells"], 6)
        self.assertEqual(report["missing_backend_cells"], 1)
        self.assertEqual(report["shapes"][0]["missing_impls"], ["fa4"])

    def test_terminal_skips_allow_campaign_completion(self):
        report = aggregate(_terminal_case("small"), manifest=_manifest(_case("small")))
        self.assertTrue(report["campaign_complete"])
        self.assertTrue(report["comparison_complete"])
        self.assertTrue(report["complete"])
        self.assertEqual(_variant(report)["impls"]["fa4"]["display"], "no kernel")
        self.assertEqual(
            _variant(report, "alibi")["impls"]["fa4"]["display"], "not requested"
        )

    def test_manifest_validates_exact_case_ids_and_dimensions(self):
        manifest = _manifest(_case("small"))
        with self.assertRaisesRegex(ValueError, "Unexpected case"):
            aggregate(_terminal_case("different"), manifest=manifest)
        records = _terminal_case("small")
        for row in records:
            row["shape"]["m"] = 4096
        with self.assertRaisesRegex(ValueError, "Shape dimensions disagree"):
            aggregate(records, manifest=manifest)

    def test_duplicate_manifest_ids_and_cross_variant_result_ids_are_rejected(self):
        for second in (_case("same"), _case("same", variant="alibi")):
            with (
                self.subTest(second=second),
                self.assertRaisesRegex(ValueError, "Duplicate manifest case ID"),
            ):
                aggregate(
                    _terminal_case("same"), manifest=_manifest(_case("same"), second)
                )
        with self.assertRaisesRegex(ValueError, "Duplicate shape_id across variants"):
            aggregate(
                [
                    _record("helion-cute", "same", 2.0),
                    _record("flex-triton", "same", 4.0, variant="alibi"),
                ],
                expected_shapes=2,
            )

    def test_manifest_missing_cases_remain_in_summary_and_cell_denominators(self):
        report = aggregate(
            _terminal_case("small"),
            manifest=_manifest(_case("small"), _case("medium"), _case("large")),
        )
        self.assertEqual(report["expected_shapes"], 3)
        self.assertEqual(report["observed_shapes"], 1)
        self.assertEqual(report["unobserved_expected_shapes"], 2)
        self.assertEqual(set(report["missing_case_ids"]), {"medium", "large"})
        self.assertEqual(report["missing_backend_cells"], 12)
        self.assertEqual(len(report["shapes"]), 3)
        cell = _variant(report)["impls"]["helion-cute"]
        self.assertEqual(cell["status"], "PARTIAL")
        self.assertIn("1/3", cell["display"])

    def test_campaign_manifest_selects_its_cases_and_validates_gpu_and_settings(self):
        manifest = _manifest(
            _case("small"),
            _case("large"),
            campaign={
                "selected_cases": ["small"],
                "case_gpus": {"small": 2},
                "input_seed": 0,
            },
        )
        report = aggregate(_terminal_case("small"), manifest=manifest)
        self.assertEqual(report["expected_shapes"], 1)
        self.assertTrue(report["complete"])
        with self.assertRaisesRegex(ValueError, "Unexpected case"):
            aggregate(_terminal_case("large"), manifest=manifest)
        with self.assertRaisesRegex(ValueError, "GPU assignment disagrees"):
            aggregate(_terminal_case("small", physical_gpu="3"), manifest=manifest)
        with self.assertRaisesRegex(ValueError, "Mixed input_seed"):
            aggregate(_terminal_case("small", input_seed=1), manifest=manifest)

    def test_manifest_fingerprint_must_match_the_supplied_manifest(self):
        manifest = _manifest(_case("small"))
        manifest_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report = aggregate(
            _terminal_case("small", manifest_sha256=manifest_hash), manifest=manifest
        )
        self.assertTrue(report["manifest_validated"])
        with self.assertRaisesRegex(ValueError, "Mixed manifest_sha256"):
            aggregate(
                _terminal_case("small", manifest_sha256="different"), manifest=manifest
            )
        manifest["source_provenance"] = {"source_fingerprint": "expected-source"}
        with self.assertRaisesRegex(ValueError, "Mixed source_fingerprint"):
            aggregate(
                _terminal_case("small", source_fingerprint="different-source"),
                manifest=manifest,
            )

    def test_cli_reads_manifest_and_writes_expected_case_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / "results.jsonl"
            manifest = Path(directory) / "manifest.json"
            summary = Path(directory) / "summary.json"
            inputs.write_text(
                "\n".join(json.dumps(row) for row in _terminal_case("small")) + "\n"
            )
            manifest.write_text(json.dumps(_manifest(_case("small"), _case("large"))))
            argv = [
                "attngym_report.py",
                "--input",
                str(inputs),
                "--manifest",
                str(manifest),
                "--summary-output",
                str(summary),
            ]
            with patch("sys.argv", argv), redirect_stdout(StringIO()):
                main()
            report = json.loads(summary.read_text())
            self.assertEqual(report["expected_shapes"], 2)
            self.assertEqual(report["missing_case_ids"], ["large"])
            self.assertFalse(report["complete"])


if __name__ == "__main__":
    unittest.main()
