from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import benchmarks.compare_matmul_backends as compare_matmul_backends
from benchmarks.compare_matmul_backends import _build_subprocess_cmd
from benchmarks.compare_matmul_backends import _make_helion_config_from_args
from benchmarks.compare_matmul_backends import _parse_indexing_list
from benchmarks.compare_matmul_backends import _parse_int_list
from benchmarks.compare_matmul_backends import _parse_optional_bool_list
from benchmarks.compare_matmul_backends import _quack_codegen_sources
from benchmarks.compare_matmul_backends import _source_bundle_summary
from benchmarks.compare_matmul_backends import _source_marker_counts
from benchmarks.compare_matmul_backends import _source_marker_lines
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
            Path("quack") / "gemm_sm100.py": "cute.arch.griddepcontrol_wait()\n",
            Path("quack") / "gemm_base.py": "PipelineTmaStore.create(\n",
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
        self.assertEqual(
            _parse_indexing_list("tensor_descriptor,pointer"),
            ["tensor_descriptor", "pointer"],
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_optional_bool_list("none,true,false")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_int_list("1")
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
            "cute.arch.griddepcontrol_wait()"
        )

        counts = _source_marker_counts(source)
        lines = _source_marker_lines(source)

        self.assertEqual(counts["pipeline_tma_store"], 1)
        self.assertEqual(counts["work_tile_loop"], 1)
        self.assertEqual(counts["epilogue_barrier_wait"], 2)
        self.assertEqual(counts["griddepcontrol_wait"], 1)
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
                code="PipelineTmaStore.create(\n",
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

    def test_two_cta_special_modes_are_exclusive(self) -> None:
        with self.assertRaisesRegex(SystemExit, "choose only one"):
            _validate_args(
                _args(
                    helion_two_cta_diagnostic_sweep=True,
                    helion_two_cta_codegen_report=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
