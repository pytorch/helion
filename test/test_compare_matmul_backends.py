from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import benchmarks.compare_matmul_backends as compare_matmul_backends
from benchmarks.compare_matmul_backends import _build_subprocess_cmd
from benchmarks.compare_matmul_backends import _build_two_cta_ncu_command
from benchmarks.compare_matmul_backends import _helion_role_local_loop_summaries
from benchmarks.compare_matmul_backends import _make_helion_config_from_args
from benchmarks.compare_matmul_backends import _parse_csv_list
from benchmarks.compare_matmul_backends import _parse_indexing_list
from benchmarks.compare_matmul_backends import _parse_int_list
from benchmarks.compare_matmul_backends import _parse_ncu_csv_metrics
from benchmarks.compare_matmul_backends import _parse_optional_bool_list
from benchmarks.compare_matmul_backends import _quack_codegen_sources
from benchmarks.compare_matmul_backends import _run_helion_two_cta_ncu_report
from benchmarks.compare_matmul_backends import _run_helion_two_cta_ncu_target
from benchmarks.compare_matmul_backends import _source_bundle_summary
from benchmarks.compare_matmul_backends import _source_marker_counts
from benchmarks.compare_matmul_backends import _source_marker_lines
from benchmarks.compare_matmul_backends import _source_symbol_marker_summaries
from benchmarks.compare_matmul_backends import _two_cta_diagnostic_variant_args
from benchmarks.compare_matmul_backends import _two_cta_seed_args
from benchmarks.compare_matmul_backends import _validate_args


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "m": 4096,
        "n": 4096,
        "k": 4096,
        "dtype": "bfloat16",
        "num_runs": 5,
        "warmup_ms": 1000,
        "rep_ms": 500,
        "seed": 0,
        "impl": "all",
        "impls": None,
        "print_codegen": False,
        "json": False,
        "skip_correctness": 0,
        "helion_two_cta_diagnostic_sweep": False,
        "helion_two_cta_codegen_report": False,
        "helion_two_cta_ncu_report": False,
        "helion_two_cta_ncu_target": False,
        "ncu_binary": "ncu",
        "ncu_metrics": [
            "sm__cycles_elapsed.avg",
            "smsp__cycles_elapsed.avg",
        ],
        "ncu_warmup_launches": 1,
        "ncu_profile_launches": 1,
        "ncu_command_only": False,
        "quack_path": None,
        "quack_tile_m": 256,
        "quack_tile_n": 256,
        "quack_cluster_m": 2,
        "quack_cluster_n": 1,
        "quack_pingpong": 0,
        "quack_persistent": 1,
        "quack_dynamic_persistent": 1,
        "quack_max_swizzle_size": 8,
        "helion_backend": "cute",
        "helion_force_config": 1,
        "helion_tile_m": 256,
        "helion_tile_n": 256,
        "helion_tile_k": 128,
        "helion_num_stages": 1,
        "helion_num_warps": 4,
        "helion_maxnreg": None,
        "helion_l2_grouping": 4,
        "helion_indexing": None,
        "helion_pid_type": "persistent_blocked",
        "helion_cluster_m": 2,
        "helion_ab_stages": 2,
        "helion_acc_stages": 2,
        "helion_c_stages": 2,
        "helion_num_epi_warps": 4,
        "helion_require_tcgen05": 1,
        "helion_range_flattens": [None, True],
        "helion_range_multi_buffers": [None, False],
        "helion_range_warp_specializes": [None, True],
        "helion_range_num_stages": [0, 0],
        "helion_range_unroll_factors": [0, 0],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestCompareMatmulBackends(unittest.TestCase):
    @staticmethod
    def _write_quack_codegen_sources(quack_root: Path) -> None:
        sources = {
            Path("quack") / "gemm_sm100.py": (
                "class Gemm:\n"
                "    def producer(self):\n"
                "        cute.arch.griddepcontrol_wait()\n"
            ),
            Path("quack")
            / "gemm_base.py": "def store():\n    PipelineTmaStore.create()\n",
            Path("quack") / "pipeline.py": "PipelineTmaUmma\n",
            Path("quack") / "tile_scheduler.py": "TileSchedulerCls\n",
        }
        for relative_path, source in sources.items():
            path = quack_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")

    def test_helion_fixed_config_keeps_default_range_hints(self) -> None:
        config = _make_helion_config_from_args(_args()).config

        self.assertNotIn("indexing", config)
        self.assertNotIn("maxnreg", config)
        self.assertEqual(config["range_flattens"], [None, True])
        self.assertEqual(config["range_multi_buffers"], [None, False])
        self.assertEqual(config["range_warp_specializes"], [None, True])

    def test_helion_fixed_config_accepts_diagnostic_knobs(self) -> None:
        config = _make_helion_config_from_args(
            _args(
                helion_indexing=[
                    "tensor_descriptor",
                    "tensor_descriptor",
                    "tensor_descriptor",
                ],
                helion_maxnreg=128,
                helion_range_flattens=[None, None],
                helion_range_multi_buffers=[None, None],
                helion_range_warp_specializes=[None, None],
                helion_range_num_stages=[1, 2],
                helion_range_unroll_factors=[2, 4],
            )
        ).config

        self.assertEqual(
            config["indexing"],
            ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
        )
        self.assertEqual(config["maxnreg"], 128)
        self.assertEqual(config["range_flattens"], [None, None])
        self.assertEqual(config["range_multi_buffers"], [None, None])
        self.assertEqual(config["range_warp_specializes"], [None, None])
        self.assertEqual(config["range_num_stages"], [1, 2])
        self.assertEqual(config["range_unroll_factors"], [2, 4])

    def test_subprocess_command_preserves_diagnostic_knobs(self) -> None:
        args = _args(
            helion_indexing=["tensor_descriptor", "pointer", "pointer"],
            helion_maxnreg=64,
            helion_range_flattens=[False, None],
            helion_range_multi_buffers=[None, True],
            helion_range_warp_specializes=[True, None],
        )

        cmd = _build_subprocess_cmd(args, "helion-cute")

        self.assertIn("--helion-indexing", cmd)
        self.assertIn("tensor_descriptor,pointer,pointer", cmd)
        self.assertIn("--helion-maxnreg", cmd)
        self.assertIn("64", cmd)
        self.assertIn("--helion-range-flattens", cmd)
        self.assertIn("false,none", cmd)
        self.assertIn("--helion-range-multi-buffers", cmd)
        self.assertIn("none,true", cmd)
        self.assertIn("--helion-range-warp-specializes", cmd)
        self.assertIn("true,none", cmd)

    def test_cli_list_parsers(self) -> None:
        self.assertEqual(_parse_optional_bool_list("none,true"), [None, True])
        self.assertEqual(_parse_optional_bool_list("false,0"), [False, False])
        self.assertEqual(_parse_int_list("1,2"), [1, 2])
        self.assertEqual(_parse_csv_list("a, b"), ["a", "b"])
        self.assertEqual(
            _parse_indexing_list("tensor_descriptor,pointer"),
            ["tensor_descriptor", "pointer"],
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_optional_bool_list("none,true,false")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_int_list("1")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_csv_list(" , ")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_indexing_list("not_an_indexing")

    def test_two_cta_diagnostic_sweep_pins_validated_family(self) -> None:
        variants = dict(_two_cta_diagnostic_variant_args(_args()))

        self.assertEqual(
            set(variants),
            {"seed", "seed_maxnreg128", "warps16", "quick_selected_family"},
        )
        for variant_args in variants.values():
            config = _make_helion_config_from_args(variant_args).config
            self.assertEqual(config["block_sizes"], [256, 256, 128])
            self.assertEqual(config["pid_type"], "persistent_blocked")
            self.assertEqual(config["tcgen05_cluster_m"], 2)
            self.assertEqual(config["l2_groupings"], [4])
            self.assertEqual(config["tcgen05_num_epi_warps"], 4)
            self.assertEqual(config["range_flattens"], [None, None])
            self.assertEqual(config["range_multi_buffers"], [None, None])
            self.assertEqual(config["range_warp_specializes"], [None, None])

        seed_config = _make_helion_config_from_args(variants["seed"]).config
        self.assertEqual(seed_config["num_warps"], 4)
        self.assertNotIn("maxnreg", seed_config)
        self.assertEqual(
            seed_config["indexing"],
            ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
        )

        maxnreg_config = _make_helion_config_from_args(
            variants["seed_maxnreg128"]
        ).config
        self.assertEqual(maxnreg_config["maxnreg"], 128)
        self.assertEqual(maxnreg_config["num_warps"], 4)

        warps16_config = _make_helion_config_from_args(variants["warps16"]).config
        self.assertEqual(warps16_config["num_warps"], 16)
        self.assertNotIn("maxnreg", warps16_config)

        quick_config = _make_helion_config_from_args(
            variants["quick_selected_family"]
        ).config
        self.assertEqual(quick_config["num_warps"], 16)
        self.assertEqual(quick_config["maxnreg"], 128)
        self.assertEqual(
            quick_config["indexing"],
            ["tensor_descriptor", "pointer", "tensor_descriptor"],
        )

    def test_two_cta_codegen_report_uses_validated_seed_family(self) -> None:
        args = _args()
        variants = dict(_two_cta_diagnostic_variant_args(args))
        seed_args = _two_cta_seed_args(args)
        self.assertEqual(vars(seed_args), vars(variants["seed"]))

        config = _make_helion_config_from_args(seed_args).config

        self.assertEqual(config["block_sizes"], [256, 256, 128])
        self.assertEqual(config["pid_type"], "persistent_blocked")
        self.assertEqual(config["tcgen05_cluster_m"], 2)
        self.assertEqual(config["l2_groupings"], [4])
        self.assertEqual(config["num_warps"], 4)
        self.assertNotIn("maxnreg", config)
        self.assertEqual(
            config["indexing"],
            ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
        )

    def test_source_marker_helpers_count_aliases_and_lines(self) -> None:
        source = (
            "PipelineTmaStore.create(\n"
            "while work_tile.is_valid_tile:\n"
            "tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            "epilogue_barrier.arrive_and_wait()\n"
            "cute.arch.griddepcontrol_wait()\n"
            "cute.arch.griddepcontrol_launch_dependents()"
        )

        counts = _source_marker_counts(source)
        lines = _source_marker_lines(source)

        self.assertEqual(counts["pipeline_tma_store"], 1)
        self.assertEqual(counts["work_tile_loop"], 1)
        self.assertEqual(counts["epilogue_barrier_wait"], 2)
        self.assertEqual(counts["griddepcontrol_wait"], 1)
        self.assertEqual(counts["griddepcontrol_launch_dependents"], 1)
        self.assertEqual(lines["pipeline_tma_store"], [1])
        self.assertEqual(lines["epilogue_barrier_wait"], [3, 4])

    def test_source_bundle_summary_reports_file_locations(self) -> None:
        summary = _source_bundle_summary(
            {
                "generated.py": "PipelineTmaStore.create(\n",
                "quack.py": "cute.arch.griddepcontrol_wait()\n",
            }
        )

        self.assertEqual(summary["line_count"], 2)
        self.assertEqual(summary["markers"]["pipeline_tma_store"], 1)
        self.assertEqual(summary["markers"]["griddepcontrol_wait"], 1)
        self.assertEqual(
            summary["first_marker_locations"]["pipeline_tma_store"],
            ["generated.py:1"],
        )
        self.assertEqual(
            summary["first_marker_locations"]["griddepcontrol_wait"],
            ["quack.py:1"],
        )

    def test_helion_role_local_loop_summaries_group_markers_by_role(self) -> None:
        code = """
def kernel():
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5):
        cute.arch.griddepcontrol_wait()
        sched = cutlass.utils.StaticPersistentTileScheduler.create(params)
        while tcgen05_role_local_0_work_tile.is_valid_tile:
            tcgen05_ab_pipeline.producer_acquire(state)
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(6):
        sched = cutlass.utils.StaticPersistentTileScheduler.create(params)
        while tcgen05_role_local_1_work_tile.is_valid_tile:
            tcgen05_acc_pipeline.producer_acquire(state)
"""

        summaries = _helion_role_local_loop_summaries(code)

        self.assertEqual(
            [summary["role"] for summary in summaries], ["tma_load", "mma_exec"]
        )
        self.assertEqual(summaries[0]["markers"]["griddepcontrol_wait"], 1)
        self.assertEqual(summaries[0]["markers"]["persistent_scheduler"], 1)
        self.assertEqual(summaries[0]["markers"]["work_tile_loop"], 1)
        self.assertEqual(summaries[0]["markers"]["producer_acquire"], 1)
        self.assertEqual(
            [entry["marker"] for entry in summaries[0]["marker_trace"]],
            [
                "griddepcontrol_wait",
                "persistent_scheduler",
                "work_tile_loop",
                "producer_acquire",
            ],
        )

    def test_source_symbol_marker_summaries_groups_quack_markers(self) -> None:
        summaries = _source_symbol_marker_summaries(
            {
                "quack/gemm_sm100.py": """
class Gemm:
    def producer(self):
        cute.arch.griddepcontrol_wait()
        PipelineTmaUmma()

def epilogue():
    PipelineTmaStore.create()
""",
                "quack/pipeline.py": "def no_markers():\n    return None\n",
            }
        )

        self.assertEqual(
            [(summary["source"], summary["symbol"]) for summary in summaries],
            [
                ("quack/gemm_sm100.py", "Gemm.producer"),
                ("quack/gemm_sm100.py", "epilogue"),
            ],
        )
        self.assertEqual(summaries[0]["markers"]["griddepcontrol_wait"], 1)
        self.assertEqual(summaries[0]["markers"]["pipeline_tma_umma"], 1)
        self.assertEqual(summaries[1]["markers"]["pipeline_tma_store"], 1)

    def test_source_symbol_marker_summaries_do_not_truncate(self) -> None:
        source = "\n\n".join(
            f"def marker_{idx}():\n    PipelineTmaStore.create()" for idx in range(20)
        )

        summaries = _source_symbol_marker_summaries({"quack/generated.py": source})

        self.assertEqual(len(summaries), 20)
        self.assertEqual(summaries[-1]["symbol"], "marker_19")

    def test_quack_codegen_sources_use_requested_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            quack_root = Path(tmpdir)
            self._write_quack_codegen_sources(quack_root)

            sources = _quack_codegen_sources(quack_root)

        gemm_source = str(Path(quack_root.name) / "quack" / "gemm_sm100.py")
        self.assertEqual(
            set(sources),
            {
                gemm_source,
                str(Path(quack_root.name) / "quack" / "gemm_base.py"),
                str(Path(quack_root.name) / "quack" / "pipeline.py"),
                str(Path(quack_root.name) / "quack" / "tile_scheduler.py"),
            },
        )
        self.assertIn("griddepcontrol_wait", sources[gemm_source])

    def test_two_cta_codegen_report_reads_requested_quack_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            quack_root = Path(tmpdir)
            self._write_quack_codegen_sources(quack_root)
            prepared = argparse.Namespace(
                active_config="fake-config",
                code="def kernel():\n    PipelineTmaStore.create()\n",
                codegen={"uses_tcgen05": True},
            )
            with mock.patch.object(
                compare_matmul_backends, "_prepare_helion", return_value=prepared
            ):
                payload = compare_matmul_backends._run_helion_two_cta_codegen_report(
                    _args(quack_path=str(quack_root))
                )

        gemm_source = str(Path(quack_root.name) / "quack" / "gemm_sm100.py")
        self.assertIn(gemm_source, payload["quack"]["sources"])
        self.assertEqual(payload["quack"]["markers"]["griddepcontrol_wait"], 1)
        self.assertEqual(
            payload["quack"]["marker_symbols"][0]["symbol"],
            "Gemm.producer",
        )
        self.assertEqual(payload["helion"]["role_local_loops"], [])

    def test_two_cta_codegen_report_fails_cleanly_without_quack_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(
                    compare_matmul_backends, "_prepare_helion"
                ) as prepare_helion,
                self.assertRaisesRegex(SystemExit, "Set --quack-path"),
            ):
                compare_matmul_backends._run_helion_two_cta_codegen_report(
                    _args(quack_path=tmpdir)
                )

            prepare_helion.assert_not_called()

    def test_two_cta_ncu_command_targets_validated_seed_family(self) -> None:
        args = _args(
            m=2048,
            n=1024,
            k=512,
            dtype="float16",
            seed=123,
            ncu_binary="/opt/ncu",
            ncu_metrics=["metric_a", "metric_b"],
            ncu_warmup_launches=2,
            ncu_profile_launches=3,
        )

        cmd = _build_two_cta_ncu_command(args)

        self.assertEqual(cmd[0], "/opt/ncu")
        self.assertIn("--metrics", cmd)
        self.assertIn("metric_a,metric_b", cmd)
        self.assertIn("--profile-from-start", cmd)
        self.assertIn("off", cmd)
        self.assertIn("--launch-skip", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--launch-count", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--helion-two-cta-ncu-target", cmd)
        self.assertIn("--helion-backend", cmd)
        self.assertIn("cute", cmd)
        self.assertIn("--skip-correctness", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--m", cmd)
        self.assertIn("2048", cmd)
        self.assertIn("--dtype", cmd)
        self.assertIn("float16", cmd)

    def test_two_cta_ncu_report_command_only_does_not_run_profiler(self) -> None:
        payload = _run_helion_two_cta_ncu_report(
            _args(
                helion_two_cta_ncu_report=True,
                ncu_binary="/missing/ncu",
                ncu_command_only=True,
            )
        )

        self.assertTrue(payload["command_only"])
        self.assertEqual(payload["command"][0], "/missing/ncu")
        self.assertNotIn("returncode", payload)

    def test_two_cta_ncu_target_brackets_only_target_launches(self) -> None:
        events: list[str] = []
        prepared = argparse.Namespace(
            a=object(),
            b=object(),
            dtype=object(),
            active_config="fake-config",
            codegen={"uses_tcgen05_two_cta": True},
        )

        def bound(_a: object, _b: object) -> object:
            events.append("bound")
            return object()

        prepared.bound = bound

        cudart = mock.Mock()

        def start() -> int:
            events.append("start")
            return 0

        def stop() -> int:
            events.append("stop")
            return 0

        def synchronize() -> None:
            events.append("sync")

        cudart.cudaProfilerStart.side_effect = start
        cudart.cudaProfilerStop.side_effect = stop

        with (
            mock.patch.object(
                compare_matmul_backends, "_prepare_helion", return_value=prepared
            ),
            mock.patch.object(
                compare_matmul_backends.torch.cuda,
                "cudart",
                return_value=cudart,
            ),
            mock.patch.object(
                compare_matmul_backends.torch.cuda,
                "synchronize",
                side_effect=synchronize,
            ),
        ):
            payload = _run_helion_two_cta_ncu_target(
                _args(
                    skip_correctness=1,
                    ncu_warmup_launches=2,
                    ncu_profile_launches=3,
                )
            )

        self.assertEqual(
            events,
            [
                "sync",
                "start",
                "bound",
                "bound",
                "sync",
                "bound",
                "bound",
                "bound",
                "sync",
                "stop",
            ],
        )
        self.assertEqual(payload["warmup_launches"], 2)
        self.assertEqual(payload["profile_launches"], 3)

    def test_two_cta_ncu_report_missing_binary_fails_cleanly(self) -> None:
        with (
            mock.patch.object(
                compare_matmul_backends.subprocess,
                "run",
                side_effect=FileNotFoundError,
            ),
            self.assertRaisesRegex(SystemExit, "requires Nsight Compute"),
        ):
            _run_helion_two_cta_ncu_report(
                _args(
                    helion_two_cta_ncu_report=True,
                    ncu_binary="/missing/ncu",
                )
            )

    def test_parse_ncu_csv_metrics(self) -> None:
        metrics = _parse_ncu_csv_metrics(
            "==PROF== Connected\n"
            '"ID","Kernel Name","Section Name","Metric Name","Metric Unit",'
            '"Metric Value"\n'
            '"0","kernel","Scheduler","sm__cycles_elapsed.avg","cycle","10"\n'
            '"0","kernel","Warp","smsp__warp_issue_stalled_barrier_per_warp_active.pct",'
            '"%","3.5"\n'
        )

        self.assertEqual(metrics["sm__cycles_elapsed.avg"][0]["value"], "10")
        self.assertEqual(metrics["sm__cycles_elapsed.avg"][0]["unit"], "cycle")
        self.assertEqual(
            metrics["smsp__warp_issue_stalled_barrier_per_warp_active.pct"][0][
                "section"
            ],
            "Warp",
        )

    def test_two_cta_diagnostic_sweep_rejects_print_codegen(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-diagnostic-sweep does not support --print-codegen",
        ):
            _validate_args(
                _args(
                    helion_two_cta_diagnostic_sweep=True,
                    print_codegen=True,
                )
            )

    def test_two_cta_diagnostic_sweep_rejects_non_helion_impl(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-diagnostic-sweep only supports",
        ):
            _validate_args(
                _args(
                    helion_two_cta_diagnostic_sweep=True,
                    impl="quack-direct",
                )
            )

    def test_two_cta_codegen_report_rejects_print_codegen(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-codegen-report already prints codegen markers",
        ):
            _validate_args(
                _args(
                    helion_two_cta_codegen_report=True,
                    print_codegen=True,
                )
            )

    def test_two_cta_codegen_report_rejects_non_cute_backend(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-codegen-report requires --helion-backend cute",
        ):
            _validate_args(
                _args(
                    helion_two_cta_codegen_report=True,
                    helion_backend="triton",
                )
            )

    def test_two_cta_codegen_report_rejects_non_helion_impl(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-codegen-report only supports",
        ):
            _validate_args(
                _args(
                    helion_two_cta_codegen_report=True,
                    impl="quack-direct",
                )
            )

    def test_two_cta_ncu_report_rejects_print_codegen(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-ncu-report does not support --print-codegen",
        ):
            _validate_args(
                _args(
                    helion_two_cta_ncu_report=True,
                    print_codegen=True,
                )
            )

    def test_two_cta_ncu_report_rejects_non_cute_backend(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--helion-two-cta-ncu-report requires --helion-backend cute",
        ):
            _validate_args(
                _args(
                    helion_two_cta_ncu_report=True,
                    helion_backend="triton",
                )
            )

    def test_two_cta_ncu_report_rejects_invalid_launch_counts(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--ncu-warmup-launches must be >= 0",
        ):
            _validate_args(
                _args(
                    helion_two_cta_ncu_report=True,
                    ncu_profile_launches=0,
                )
            )

    def test_two_cta_special_modes_are_exclusive(self) -> None:
        with self.assertRaisesRegex(SystemExit, "choose only one"):
            _validate_args(
                _args(
                    helion_two_cta_diagnostic_sweep=True,
                    helion_two_cta_ncu_report=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
