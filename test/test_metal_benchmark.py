from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmarks import run_metal
from benchmarks.metal import correctness
from benchmarks.metal import dashboard
from benchmarks.metal import harness
from benchmarks.metal import manifest
from benchmarks.metal import results as results_mod
from benchmarks.metal import timing
from benchmarks.metal.correctness import compare
from benchmarks.metal.inputs import generator
from benchmarks.metal.inputs import input_hash
from benchmarks.metal.manifest import BASELINES
from benchmarks.metal.manifest import PROVIDERS
from benchmarks.metal.manifest import Case
from benchmarks.metal.manifest import Kernel
from benchmarks.metal.providers import mlx_available
import pytest
import torch

from helion import exc

# Metal runs on macOS only: CI's Linux jobs skip the whole file.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests require macOS"
)

_requires_mps = unittest.skipIf(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    "the Metal benchmark needs an Apple GPU",
)
_requires_mlx = unittest.skipIf(not mlx_available(), "mlx is not installed")


def _tmpdir(test: unittest.TestCase) -> Path:
    """A fresh directory removed when *test* ends."""
    path = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


class TestManifest(unittest.TestCase):
    def test_selected_rows_match_the_nightly(self) -> None:
        expected = {
            "rms_norm": [0, 1, 2, 3, 4, 5],
            "rms_norm-bwd": [0, 1, 2, 4, 5],
            "layer_norm": [
                0,
                2,
                3,
                5,
                6,
                8,
                9,
                11,
                12,
                14,
                15,
                17,
                18,
                20,
                21,
                23,
                24,
                26,
                27,
                29,
            ],
            "layer_norm-bwd": [0, 3, 6, 10, 13, 16, 19, 23, 26, 29],
            "welford": [0, 4, 9],
            "softmax": [
                0,
                5,
                10,
                15,
                20,
                26,
                31,
                36,
                41,
                46,
                51,
                56,
                61,
                66,
                71,
                77,
                82,
                87,
                92,
                97,
            ],
            "cross_entropy": [0, 1, 2, 3, 4, 5],
            "jsd": [0, 1, 2, 3, 4, 5],
            "kl_div": [0, 1, 2, 3, 4, 5],
            "rope": [0, 2, 4, 5, 7],
            "gemm": [0, 2, 3, 5, 6, 8, 9, 11],
            "grouped_gemm": [0, 1, 2, 3],
            "int4_gemm": [0, 3, 7, 10, 14, 17, 21, 24, 28, 31],
            "flash_attention": [0, 1, 2, 4, 5, 6],
            "gdn_fwd_h": [0, 1, 2, 3, 4, 5],
            "mamba2_chunk_state": [0, 1, 2, 3, 4, 5],
            "mamba2_chunk_scan": [0, 1, 2, 3, 4, 5],
        }
        got = {k.name: [c.index for c in k.cases] for k in manifest.KERNELS}
        self.assertEqual(got, expected)
        for k in manifest.KERNELS:
            self.assertEqual(len(k.rows), k.available, k.name)
            self.assertIn(k.dtype, k.tolerance, k.name)
        self.assertEqual(
            manifest.kernel("gemm").cases[2].params, {"M": 2048, "N": 4096, "K": 2048}
        )

    def test_guard_set_and_sentinel_exist(self) -> None:
        for name, index in (*manifest.GUARD_SET, manifest.SENTINEL):
            manifest.case(name, index)

    def test_manifest_loads_without_torch_or_triton(self) -> None:
        # The CLI parses arguments before importing torch.
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('torch', 'triton'):\n"
            "            raise ImportError(name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import benchmarks.metal.manifest\n"
        )
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-c", code], cwd=root, check=True)

    def test_only_baselines_get_looser_tolerances(self) -> None:
        for k in manifest.KERNELS:
            self.assertNotIn("helion", k.baseline_tolerance, k.name)
        welford = manifest.kernel("welford")
        strict = manifest.tolerance(welford, "helion", "bfloat16")
        self.assertEqual(strict, welford.tolerance["bfloat16"])
        self.assertGreater(manifest.tolerance(welford, "mlx", "bfloat16")[1], strict[1])

    def test_flash_attention_takes_the_dashboard_tolerance(self) -> None:
        self.assertEqual(
            manifest.kernel("flash_attention").tolerance, {"bfloat16": (2e-2, 1e-2)}
        )

    def test_hash_covers_operator_code(self) -> None:
        before = manifest.manifest_hash()
        real = Path.read_bytes

        def edited(path: Path) -> bytes:
            data = real(path)
            return data + b"#" if path.name == "softmax.py" else data

        with patch.object(Path, "read_bytes", edited):
            self.assertNotEqual(manifest.manifest_hash(), before)


class TestInputs(unittest.TestCase):
    def test_inputs_are_deterministic_and_keep_dtype_and_layout(self) -> None:
        from benchmarks.metal.ops import gemm

        case = Case("gemm", 0, "", "float16", {"M": 64, "N": 32, "K": 16})
        first = gemm.inputs(case, generator(case))
        second = gemm.inputs(case, generator(case))
        self.assertEqual(input_hash(first), input_hash(second))
        self.assertEqual(first["w"].dtype, torch.float16)
        self.assertEqual(first["w"].stride(), (1, 16))
        other = Case("gemm", 1, "", "float16", case.params)
        self.assertNotEqual(
            input_hash(first), input_hash(gemm.inputs(other, generator(other)))
        )

    def test_int4_packings_hold_the_same_weights(self) -> None:
        from benchmarks.metal.ops import int4_gemm

        case = Case("int4_gemm", 0, "", "bfloat16", {"B": 1, "L": 2, "K": 128, "N": 64})
        host = int4_gemm.inputs(case, generator(case))
        self.assertTrue(host["w_packed"].is_contiguous())
        unpacked = int4_gemm._unpack(host["w_packed"])
        shifts = 4 * torch.arange(8, dtype=torch.int64)
        nibbles = (host["w_mlx"].to(torch.int64)[..., None] >> shifts) & 0xF
        self.assertTrue(
            torch.equal(unpacked.float(), (nibbles.reshape(64, 128).T - 8).float())
        )


class TestReference(unittest.TestCase):
    def test_hand_computed_values(self) -> None:
        from benchmarks.metal.ops import cross_entropy
        from benchmarks.metal.ops import jsd
        from benchmarks.metal.ops import softmax

        x = torch.tensor([[0.0, torch.log(torch.tensor(3.0))]], dtype=torch.float16)
        torch.testing.assert_close(
            softmax.reference(x),
            torch.tensor([[0.25, 0.75]], dtype=torch.float64),
            atol=1e-3,
            rtol=0,
        )
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        loss = cross_entropy.reference(logits, torch.tensor([0, 1]))
        self.assertAlmostEqual(
            float(loss), float(torch.log(torch.tensor(2.0))), places=6
        )
        same = torch.tensor([[0.25, 0.75]]).log()
        self.assertAlmostEqual(float(jsd.reference(same, same)), 0.0, places=12)


class TestComparator(unittest.TestCase):
    def test_rejects_each_kind_of_mismatch(self) -> None:
        good = torch.tensor([1.0, 2.0, float("nan")])
        self.assertTrue(compare(good, good.double(), 1e-4, 1e-5).passed)
        bad_value = torch.tensor([1.0, 2.1, float("nan")])
        self.assertFalse(compare(bad_value, good, 1e-4, 1e-5).passed)
        self.assertFalse(
            compare(torch.tensor([1.0, 2.0, 3.0]), good, 1e-4, 1e-5).passed
        )
        self.assertFalse(compare(torch.ones(2), torch.ones(3), 1e-4, 1e-5).passed)
        self.assertFalse(compare((good,), (good, good), 1e-4, 1e-5).passed)
        inf = torch.tensor([float("inf")])
        self.assertFalse(compare(-inf, inf, 1e-4, 1e-5).passed)
        check = harness._compare(good.half(), good, 1e-4, 1e-5, [torch.float32])
        self.assertFalse(check.passed)
        self.assertIn("dtypes", check.reason)

    def test_catches_a_dropped_tail(self) -> None:
        expected = torch.rand(1000) + 0.5
        dropped = expected.clone()
        dropped[-1] = 0
        rtol, atol = manifest.kernel("rms_norm").tolerance["float32"]
        self.assertFalse(compare(dropped, expected, rtol, atol).passed)

    def test_mismatches_at_chunk_boundaries_are_caught(self) -> None:
        expected = torch.rand(20) + 0.5
        with patch.object(correctness, "CHUNK", 7):
            self.assertTrue(compare(expected, expected, 1e-4, 1e-5).passed)
            for index in (0, 6, 7, 13, 14, 19):
                bad = expected.clone()
                bad[index] = float("nan") if index % 2 else 0
                check = compare(bad, expected, 1e-4, 1e-5)
                self.assertFalse(check.passed, index)


class _FakeBound:
    """Counts calls and completions; each call returns a fresh object."""

    def __init__(self) -> None:
        self.calls = 0
        self.completed: list[object] = []
        self.events = None

    def call(self) -> object:
        self.calls += 1
        return object()

    def complete(self, out: object) -> None:
        self.completed.append(out)

    def sync(self) -> None:
        pass


class TestTiming(unittest.TestCase):
    def test_each_sample_completes_exactly_its_own_call(self) -> None:
        bound = _FakeBound()
        samples = timing.latency(bound)
        self.assertEqual(bound.calls, len(bound.completed))
        self.assertEqual(len(set(map(id, bound.completed))), len(bound.completed))
        self.assertGreater(len(samples), 0)

    def test_batched_time_keeps_every_output_until_completion(self) -> None:
        bound = _FakeBound()
        batch = timing.kernel_ms(bound, output_bytes=timing.BATCH_BYTES_CAP // 4)
        self.assertEqual(batch.source, "wall_batched")
        self.assertLessEqual(batch.enqueue_ms, batch.kernel_ms)
        # An untimed batch of the same size warms the outputs' buffers.
        self.assertEqual([len(outs) for outs in bound.completed], [4, 4])
        self.assertEqual(bound.calls, 8)

    def test_paired_speedup_is_median_of_block_ratios(self) -> None:
        def record(medians: list[float]) -> dict:
            return {"latency": [{"median": m} for m in medians]}

        baseline = record([2.0, 4.0, 9.0])
        helion = record([1.0, 4.0, 3.0])
        self.assertEqual(results_mod.paired_speedup(baseline, helion), 2.0)


def _timed(ms: float, kernel_ms: float, enqueue_ms: float, source: str) -> dict:
    return {
        "status": "ok",
        "error": None,
        "latency": [{"median": ms}],
        "kernel_ms": [kernel_ms],
        "enqueue_ms": [enqueue_ms],
        "kernel_ms_source": source,
    }


class TestResults(unittest.TestCase):
    def test_loss_tags(self) -> None:
        def tag(helion: dict, baseline: dict) -> str:
            cell = {"providers": {"helion": helion, "torch_eager": baseline}}
            return results_mod.loss_tag(cell, "torch_eager")

        events = "mps_events"
        self.assertEqual(
            tag(_timed(2, 1.5, 0.1, events), _timed(1, 1.0, 0.1, events)),
            "kernel-bound",
        )
        self.assertEqual(
            tag(_timed(2, 0.5, 0.1, events), _timed(1, 1.0, 0.1, events)),
            "dispatch-bound",
        )
        # The host could not keep the GPU busy with Helion's calls.
        self.assertEqual(
            tag(_timed(2, 1.5, 1.4, events), _timed(1, 1.0, 0.1, events)),
            "dispatch-bound",
        )
        # A dispatch-limited baseline's device time is only an upper bound.
        self.assertEqual(
            tag(_timed(2, 0.5, 0.1, events), _timed(1, 1.0, 0.95, events)), "unknown"
        )
        self.assertEqual(
            tag(_timed(2, 1.5, 0.1, events), _timed(1, 1.0, 0.1, "wall_batched")),
            "unknown",
        )

    def test_backend_gaps_are_unsupported_and_compiler_failures_are_not(self) -> None:
        self.assertEqual(
            harness.classify(exc.BackendImplementationMissing("metal", "x")),
            "unsupported",
        )
        self.assertEqual(
            harness.classify(exc.BackendUnsupported("metal", "x")), "unsupported"
        )
        self.assertEqual(
            harness.classify(exc.InternalError(RuntimeError("x"))), "compile_error"
        )

    def test_summary_and_export(self) -> None:
        out = _tmpdir(self)
        results = results_mod.Results.create(
            out,
            {
                "manifest": {"version": "v", "hash": "0" * 12},
                "environment": {
                    "chip": "chip",
                    "macos": "1",
                    "torch": "t",
                    "mlx": "m",
                    "helion_commit": "c",
                },
                "run": {
                    "providers": list(PROVIDERS),
                    "autotune_effort": "none",
                    "autotune_seed": "1",
                    "blocks": 1,
                    "order_seed": 0,
                    "force": True,
                    "resumes": [
                        {
                            "at": "T",
                            "after": "timeout",
                            "preflight_problems": ["running on battery"],
                        }
                    ],
                },
            },
        )
        helion = _timed(4.0, 3.0, 0.1, "mps_events")
        helion["dashboard"] = {"passed": False}
        results.cells["k/0"] = {
            "kernel": "k",
            "index": 0,
            "label": "(1,)",
            "state": "done",
            "sync_floor_ms": {"torch": 0.1},
            "providers": {
                "torch_eager": _timed(2.0, 1.0, 0.1, "mps_events"),
                "torch_compile": {"status": "wrong_result", "error": "bad values"},
                "helion": helion,
            },
        }
        results.cells["k/1"] = {
            "kernel": "k",
            "index": 1,
            "label": "(2,)",
            "state": "done",
            "error": "process crash during setup in an earlier run",
            "providers": {},
        }
        text = results_mod.summarize(results)
        self.assertIn("torch_compile: **wrong_result** — bad values", text)
        self.assertIn("0.50x vs torch_eager, kernel-bound", text)
        self.assertIn("process crash during setup", text)
        self.assertIn(
            "resumed T after timeout; preflight then: running on battery", text
        )
        records = {
            r["metric"]["name"]: r["metric"]["benchmark_values"]
            for r in results_mod.helionbench_records(results)
        }
        # The dashboard's accuracy is its 1e-2 gate, not the strict verdict.
        self.assertEqual(records["helion_accuracy"], [0.0])
        self.assertEqual(records["helion_speedup"], [0.5])
        results_mod.write_reports(results)
        page = (out / "dashboard.html").read_text()
        # One mark per cell, and the failed cell's tooltip names its cause.
        self.assertEqual(page.count('class="mark"'), 2 * (1 + len(BASELINES)))
        self.assertIn("process crash during setup", page)

    def test_dashboard_escapes_errors_and_shows_only_run_providers(self) -> None:
        data = {
            "manifest": {"version": "v"},
            "environment": {
                "chip": "c",
                "macos": "m",
                "torch": "t",
                "mlx": "x",
                "helion_commit": "h",
            },
            "run": {"providers": ["torch_eager", "helion"], "autotune_effort": "none"},
        }
        cell = {
            "kernel": "k",
            "index": 0,
            "label": "(1,)",
            "providers": {
                "torch_eager": {"status": "ok"},
                "helion": {"status": "runtime_error", "error": "<img src=x>"},
            },
        }
        page = dashboard.render(data, [cell])
        self.assertIn("&lt;img src=x&gt;", page)
        self.assertNotIn("<img", page)
        self.assertEqual(page.count('class="tile-name"'), 2)
        self.assertNotIn(">mlx</text>", page)
        self.assertIn("No finished cells.", dashboard.render(data, []))
        # A lone baseline keeps its label; a run without Helion has no Helion
        # sections.
        data["run"]["providers"] = ["helion", "mlx"]
        self.assertIn(">mlx</text>", dashboard.render(data, [cell]))
        data["run"]["providers"] = ["torch_eager"]
        self.assertNotIn("Why Helion fails", dashboard.render(data, [cell]))

    def test_dashboard_groups_failures_by_generalized_cause(self) -> None:
        unsupported = (
            "BackendUnsupported: Backend 'metal' does not support: persistent "
            "reduction over {n} elements needs {n} threads; give it a chunk"
        )
        missing = (
            "BackendImplementationMissing: Backend 'metal' is missing required "
            "implementation: Aten lowering codegen not registered for "
            "<OpOverload(op='prims.iota', overload='default')>"
        )
        causes = {
            dashboard.cause({"status": "unsupported", "error": unsupported.format(n=n)})
            for n in (2048, 16384)
        }
        self.assertEqual(
            causes, {"persistent reduction over N elements needs N threads"}
        )
        self.assertEqual(
            dashboard.cause({"status": "compile_error", "error": missing}),
            "no lowering for prims.iota",
        )
        self.assertEqual(
            dashboard.cause({"status": "wrong_result", "error": "1/1 values"}),
            "wrong result",
        )

    def test_samples_of_a_rerun_cell_are_replaced(self) -> None:
        out = _tmpdir(self)
        results = results_mod.Results.create(out, {"run": {}})
        results.append_samples({"cell": "a/0", "ms": [1.0]})
        results.append_samples({"cell": "b/0", "ms": [2.0]})
        results.drop_samples("a/0")
        self.assertNotIn('"a/0"', (out / "samples.jsonl").read_text())
        results_mod.Results.create(out, {"run": {}})
        self.assertFalse((out / "samples.jsonl").exists())


def _write_block(out: Path, side: str, block: int, cells: dict[str, dict]) -> None:
    results = results_mod.Results.create(out / side / f"block{block}", {"run": {}})
    for key, record in cells.items():
        results.cells[key] = {"state": "done", "providers": {"helion": record}}
    results.save()


def _helion_block(ms: float, source: str = "cache") -> dict:
    record = _timed(ms, ms, 0.1, "mps_events")
    record["evidence"] = {"config_source": source}
    return record


class TestAB(unittest.TestCase):
    def test_ab_verdicts_respect_the_noise_floor(self) -> None:
        self.assertEqual(harness.verdict(1.10, 0.05), "better")
        self.assertEqual(harness.verdict(1 / 1.10, 0.05), "worse")
        self.assertEqual(harness.verdict(1.03, 0.05), "neutral")
        calibration = {"spreads": {"replay": {"a/0": 0.02, "b/0": 0.04}, "retune": {}}}
        self.assertEqual(harness.noise_threshold(calibration, "replay", "a/0")[0], 0.04)
        self.assertAlmostEqual(
            harness.noise_threshold(calibration, "replay", "c/0")[0], 0.06
        )
        self.assertIsNone(harness.noise_threshold(calibration, "retune", "a/0")[0])

    def test_a_calibration_in_another_format_is_ignored(self) -> None:
        home = _tmpdir(self)
        env = {"chip": "c", "macos": "m", "torch": "t", "mlx": "x"}
        stored = {"identity": env, "blocks": 5, "spreads": {"a/0": 0.1}}
        with patch.dict(os.environ, {"HELION_METAL_BENCH_HOME": str(home)}):
            harness.calibration_path().write_text(json.dumps(stored))
            self.assertIsNone(harness.load_calibration(env))
            stored["format"] = harness.CALIBRATION_FORMAT
            harness.calibration_path().write_text(json.dumps(stored))
            self.assertEqual(harness.load_calibration(env), stored)

    def test_calibrated_spread_is_the_worst_block(self) -> None:
        self.assertAlmostEqual(harness.ratio_spread([1.1, 0.8, 1.0]), 0.25)
        self.assertAlmostEqual(harness.ratio_spread([1.0, 1.0]), 0.0)

    def test_verdicts_cover_flaky_crashed_hung_and_replayed_candidates(
        self,
    ) -> None:
        out = _tmpdir(self)
        wrong = {"status": "wrong_result", "error": "x"}
        hung = {"status": "timeout", "error": "x"}
        for block in range(2):
            _write_block(
                out,
                "base",
                block,
                {
                    "fast/0": _helion_block(2.0),
                    "flaky/0": _helion_block(1.0),
                    "crash/0": _helion_block(1.0),
                    "hung/0": _helion_block(1.0),
                },
            )
            cand = {
                "fast/0": _helion_block(1.0, "tuned" if block else "cache"),
                "flaky/0": wrong if block else _helion_block(1.0),
            }
            if block == 0:
                # Later blocks skip a cell that timed out.
                cand["crash/0"] = _helion_block(1.0)
                cand["hung/0"] = hung
            _write_block(out, "cand", block, cand)
        keys = ["fast/0", "flaky/0", "crash/0", "hung/0"]
        sides = {"base": "A", "cand": "B"}
        spreads = {"fast/0": 0.05}
        calibration = {"spreads": {"retune": spreads, "replay": spreads}}
        retune = harness.ab_verdicts(out, 2, "retune", sides, keys, calibration)
        cells = retune["cells"]
        self.assertEqual(cells["fast/0"]["verdict"], "better")
        self.assertEqual(cells["fast/0"]["speedup"], 2.0)
        self.assertEqual(cells["flaky/0"]["correctness"], "regressed")
        self.assertNotIn("verdict", cells["flaky/0"])
        self.assertEqual(cells["crash/0"]["cand_status"], ["ok", "crash"])
        self.assertEqual(cells["crash/0"]["correctness"], "regressed")
        self.assertEqual(cells["hung/0"]["cand_status"], ["timeout", "timeout"])
        # The candidate re-tuned in block 1, so replay has no verdict for it.
        replay = harness.ab_verdicts(out, 2, "replay", sides, keys, calibration)
        self.assertEqual(replay["cells"]["fast/0"]["verdict"], "replay_miss")
        self.assertEqual(replay["cells"]["fast/0"]["replay_miss"], ["cand"])
        uncalibrated = {"spreads": {"retune": {}, "replay": {}}}
        report = harness.ab_verdicts(out, 2, "retune", sides, keys, uncalibrated)
        self.assertEqual(report["cells"]["fast/0"]["verdict"], "uncalibrated")


class TestSourceState(unittest.TestCase):
    def test_only_source_edits_count(self) -> None:
        repo = _tmpdir(self)

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-C", str(repo), *args], check=True, capture_output=True
            )

        git("init")
        (repo / "helion").mkdir()
        (repo / "helion" / "a.py").write_text("x = 1\n")
        git("add", ".")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
        out = repo / "benchmarks" / "out"
        with patch.object(harness, "REPO_ROOT", repo):
            self.assertIsNone(harness.source_state(()))
            # A run's own output, even inside a source directory, is excluded.
            out.mkdir(parents=True)
            (out / "results.json").write_text("{}")
            (repo / "notes.txt").write_text("elsewhere")
            self.assertIsNone(harness.source_state((out,)))
            new = repo / "benchmarks" / "new.py"
            new.write_text("y = 1\n")
            untracked = harness.source_state((out,))
            self.assertIsNotNone(untracked)
            new.write_text("y = 2\n")
            self.assertNotEqual(harness.source_state((out,)), untracked)
            new.unlink()
            # An untracked nested repo (e.g. a leftover A/B worktree) is not
            # a source file.
            nested = repo / "benchmarks" / "nested"
            nested.mkdir()
            subprocess.run(
                ["git", "init", str(nested)], check=True, capture_output=True
            )
            (nested / "f.py").write_text("z = 1\n")
            self.assertIsNone(harness.source_state((out,)))
            (repo / "helion" / "a.py").write_text("x = 2\n")
            self.assertIsNotNone(harness.source_state((out,)))


class TestSupervisor(unittest.TestCase):
    def test_restarts_mark_the_death_and_resume_once(self) -> None:
        launched = []

        def popen(child: list[str], **kwargs: object) -> object:
            launched.append(list(child))
            return types.SimpleNamespace(pid=0, returncode=0)

        deaths = iter(["crash", "timeout", None])
        with (
            patch.object(run_metal.subprocess, "Popen", popen),
            patch.object(run_metal, "_watch", lambda *a: next(deaths)),
            patch.object(run_metal, "_kill_group", lambda proc: None),
            self.assertRaises(SystemExit),
        ):
            run_metal.supervise(["--output", "x"], Path("x/results.json"), 1.0)
        first, second, third = launched
        self.assertNotIn("--resume", first)
        self.assertEqual(second[-2:], ["--last-death=crash", "--resume"])
        self.assertEqual(third.count("--resume"), 1)
        self.assertEqual(
            [a for a in third if a.startswith("--last-death")],
            ["--last-death=timeout"],
        )

    def test_a_restarted_run_is_not_gated_by_preflight(self) -> None:
        fresh = run_metal.parse_args(["--output", "x"])
        self.assertTrue(run_metal.preflight_gates(fresh))
        restarted = run_metal.parse_args(
            ["--output", "x", "--resume", "--last-death=crash"]
        )
        self.assertFalse(run_metal.preflight_gates(restarted))

    def test_base_tunes_seed_only_a_retune_ab(self) -> None:
        args = ["--output", "x", "--ab", "a", "b"]
        tuned = run_metal.parse_args([*args, "--retune", "--base-tunes", "c"])
        self.assertEqual(tuned.base_tunes, Path("c"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_metal.parse_args([*args, "--replay", "r", "--base-tunes", "c"])

    def _watch(self, code: str) -> str | None:
        proc = subprocess.Popen([sys.executable, "-c", code])
        missing = _tmpdir(self) / "results.json"
        with patch.object(run_metal, "POLL_S", 0.05):
            death = run_metal._watch(proc, missing, stall_s=0.5)
        # ``_watch`` leaves a stalled child running.
        proc.kill()
        proc.wait()
        return death

    def test_watch_reports_exit_crash_and_stall(self) -> None:
        self.assertIsNone(self._watch("pass"))
        self.assertEqual(
            self._watch("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
            "crash",
        )
        self.assertEqual(self._watch("import time; time.sleep(60)"), "timeout")


def _fake_op(helion_error: Exception | None) -> types.ModuleType:
    op = types.ModuleType("fake_op")
    op.MODE = "fwd"
    op.HELION_KERNELS = (
        types.SimpleNamespace(
            name="fake_kernel", reset=lambda: None, settings=types.SimpleNamespace()
        ),
    )
    op.inputs = lambda case, gen: {"x": torch.randn(8, 16, generator=gen)}
    op.reference = lambda x: x.double() * 2
    op.eager = lambda x: x * 2

    def helion(x: torch.Tensor) -> torch.Tensor:
        if helion_error is not None:
            raise helion_error
        return x * 2

    op.helion = helion
    op.mlx = lambda x: x * 2
    return op


_FAKE_KERNEL = Kernel(
    name="fake",
    module="fake",
    source="test",
    available=1,
    num_inputs=1,
    rows=({"M": 8},),
    tolerance={"float32": (1e-4, 1e-5)},
)

# Tiny rows for every kernel, small enough to check all of them in seconds.
_TINY_ROWS = {
    "rms_norm": [{"M": 8, "H": 64}],
    "rms_norm-bwd": [{"M": 8, "H": 64}],
    "layer_norm": [{"M": 8, "N": 64}],
    "layer_norm-bwd": [{"M": 8, "N": 64}],
    "welford": [{"S": 16, "D": 64}],
    "softmax": [{"M": 8, "N": 256}],
    "cross_entropy": [{"N": 16, "V": 64}],
    "jsd": [{"BT": 8, "V": 64}],
    "kl_div": [{"BT": 8, "V": 64}],
    "rope": [{"hidden": 256, "T": 16}],
    "gemm": [{"M": 32, "N": 16, "K": 64}],
    "grouped_gemm": [{"G": 2, "N": 32}],
    "int4_gemm": [{"B": 1, "L": 2, "K": 128, "N": 64}],
    "flash_attention": [{"B": 1, "H": 2, "S": 32, "D": 64}],
    "gdn_fwd_h": [{"B": 1, "H": 2, "T": 128, "chunk": 64, "K": 64, "V": 64}],
    "mamba2_chunk_state": [
        {"B": 1, "H": 2, "G": 1, "L": 128, "chunk": 64, "P": 16, "N": 16}
    ],
    "mamba2_chunk_scan": [
        {"B": 1, "H": 2, "G": 1, "L": 128, "chunk": 64, "P": 16, "N": 16}
    ],
}


@_requires_mps
class TestHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmpdir(self)
        env = patch.dict(os.environ, {"HELION_CACHE_DIR": str(self.tmp / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def _results(self) -> results_mod.Results:
        return results_mod.Results.create(self.tmp / "run", {"run": {}})

    def _options(self, providers: tuple[str, ...]) -> harness.Options:
        return harness.Options(providers, blocks=1, correctness_only=True, order_seed=0)

    def _run_fake(
        self, op: types.ModuleType, providers: tuple[str, ...], **patches: object
    ) -> dict:
        results = self._results()
        with patch.multiple(harness, load_op=lambda kernel: op, **patches):
            harness.run_cell(
                results,
                _FAKE_KERNEL,
                _FAKE_KERNEL.cases[0],
                self._options(providers),
                "crash",
            )
        return results.cells["fake/0"]

    def test_failure_is_recorded_without_fallback(self) -> None:
        cell = self._run_fake(
            _fake_op(exc.BackendUnsupported("metal", "test feature")),
            ("torch_eager", "helion", "mlx"),
            mlx_available=lambda: False,
        )
        self.assertEqual(cell["state"], "done")
        self.assertEqual(cell["providers"]["torch_eager"]["status"], "ok")
        self.assertEqual(cell["providers"]["helion"]["status"], "unsupported")
        self.assertEqual(cell["providers"]["mlx"]["status"], "unavailable")

    def test_helion_must_run_every_kernel_on_metal(self) -> None:
        ok = {"kernel": "fake_kernel", "backend": "metal", "autotune_effort": "none"}
        for evidence in ([], [{**ok, "backend": "triton"}]):
            cell = self._run_fake(
                _fake_op(None), ("helion",), helion_evidence=lambda op, e=evidence: e
            )
            self.assertEqual(cell["providers"]["helion"]["status"], "runtime_error")
        cell = self._run_fake(
            _fake_op(None), ("helion",), helion_evidence=lambda op: [ok]
        )
        helion = cell["providers"]["helion"]
        self.assertEqual(helion["status"], "ok")
        self.assertEqual(helion["evidence"]["config_source"], "default")

    def test_resume_marks_the_step_that_died(self) -> None:
        results = self._results()
        case = _FAKE_KERNEL.cases[0]
        cell = harness._new_cell(_FAKE_KERNEL, case)
        cell["providers"]["torch_eager"] = {"status": "ok", "error": None}
        cell["in_progress"] = "helion:build"
        results.cells["fake/0"] = cell
        self.assertEqual(results_mod.done_cells(results), [])
        with patch.object(harness, "load_op", return_value=_fake_op(None)):
            harness.run_cell(
                results,
                _FAKE_KERNEL,
                case,
                self._options(("torch_eager", "helion")),
                "timeout",
            )
        providers = results.cells["fake/0"]["providers"]
        self.assertEqual(providers["helion"]["status"], "timeout")
        # Eager was checked again, not carried over from the dead run.
        self.assertEqual(providers["torch_eager"]["status"], "ok")
        self.assertIn("strict", providers["torch_eager"])

    def test_a_death_in_a_cell_step_ends_the_cell(self) -> None:
        for step in harness.CELL_STEPS:
            cell = harness._new_cell(_FAKE_KERNEL, _FAKE_KERNEL.cases[0])
            cell["providers"]["torch_eager"] = {"status": "ok", "error": None}
            cell["in_progress"] = step
            harness._resume_cell(cell, "crash")
            self.assertEqual(cell["state"], "done")
            self.assertIn(step, cell["error"])
        # A death during the dashboard's eager run is eager's.
        cell = harness._new_cell(_FAKE_KERNEL, _FAKE_KERNEL.cases[0])
        cell["in_progress"] = "torch_eager:dashboard"
        harness._resume_cell(cell, "crash")
        self.assertEqual(cell["providers"], {"torch_eager": unittest.mock.ANY})
        self.assertEqual(cell["providers"]["torch_eager"]["status"], "crash")

    def test_a_setup_failure_fails_only_the_cell(self) -> None:
        op = _fake_op(None)

        def reference(x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("MPS backend out of memory")

        op.reference = reference
        cell = self._run_fake(op, ("torch_eager",))
        self.assertEqual(cell["state"], "done")
        self.assertIn("setup failed", cell["error"])

    def test_a_death_measuring_the_sentinel_is_recorded(self) -> None:
        results = self._results()
        results.data["run"]["in_progress"] = "sentinel_start_ms"

        def unexpected() -> float:
            raise AssertionError("the sentinel is measured again")

        with patch.object(harness, "sentinel_ms", unexpected):
            harness._measure_sentinel(results, "start", "timeout")
        run = results.data["run"]
        self.assertNotIn("sentinel_start_ms", run)
        self.assertIn("timeout", run["sentinel_start_error"])

    def test_resume_refuses_a_different_run(self) -> None:
        options = self._options(("torch_eager",))
        with patch.dict(os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "1"}):
            header = harness.run_header(
                {"blocks": 1, "resume": False}, options, self.tmp / "run"
            )
        self.assertEqual(harness._same_run(header, header), [])
        resumed = {
            **header,
            "run": {**header["run"], "args": {"blocks": 1, "resume": True}},
        }
        self.assertEqual(harness._same_run(header, resumed), [])
        changed = {**header, "run": {**header["run"], "args": {"blocks": 2}}}
        self.assertTrue(harness._same_run(header, changed))
        other = {**header, "manifest": {"version": "x", "hash": "y"}}
        self.assertTrue(harness._same_run(header, other))

    def test_every_operator_matches_its_reference_at_tiny_shapes(self) -> None:
        """Every provider, Helion at each kernel's default config."""
        providers = ("torch_eager", "helion")
        if mlx_available():
            providers += ("mlx",)
        for name, rows in _TINY_ROWS.items():
            base = manifest.kernel(name)
            kernel = dataclasses.replace(
                base, rows=tuple(rows), available=len(rows), num_inputs=len(rows)
            )
            results = self._results()
            with contextlib.ExitStack() as stack:
                for helion_kernel in harness.load_op(kernel).HELION_KERNELS:
                    stack.enter_context(
                        patch.object(helion_kernel.settings, "autotune_effort", "none")
                    )
                for case in kernel.cases:
                    harness.run_cell(
                        results, kernel, case, self._options(providers), "crash"
                    )
                    cell = results.cells[f"{name}/{case.index}"]
                    for provider in providers:
                        record = cell["providers"][provider]
                        self.assertEqual(
                            record["status"], "ok", f"{name} {case.params} {provider}"
                        )

    def test_helion_runs_metal_code_and_is_timed(self) -> None:
        from benchmarks.metal.ops import softmax

        kernel = Kernel(
            name="softmax",
            module="softmax",
            source="test",
            available=1,
            num_inputs=1,
            rows=({"M": 64, "N": 256},),
            dtype="float16",
            tolerance=manifest.kernel("softmax").tolerance,
        )
        providers = (
            ("torch_eager", "helion", "mlx")
            if mlx_available()
            else ("torch_eager", "helion")
        )
        results = self._results()
        options = harness.Options(
            providers, blocks=2, correctness_only=False, order_seed=0
        )
        with patch.object(softmax.softmax_two_pass.settings, "autotune_effort", "none"):
            harness.run_cell(results, kernel, kernel.cases[0], options, "crash")
        cell = results.cells["softmax/0"]
        helion = cell["providers"]["helion"]
        self.assertEqual(helion["status"], "ok", helion.get("error"))
        self.assertEqual(
            {k["backend"] for k in helion["evidence"]["kernels"]}, {"metal"}
        )
        for provider in providers:
            self.assertEqual(len(cell["providers"][provider]["latency"]), 2)
            self.assertEqual(len(cell["providers"][provider]["enqueue_ms"]), 2)
        self.assertEqual(len(cell["block_orders"]), 2)


if __name__ == "__main__":
    unittest.main()
