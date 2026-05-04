from __future__ import annotations

import argparse
import unittest

from benchmarks.compare_matmul_backends import _build_subprocess_cmd
from benchmarks.compare_matmul_backends import _make_helion_config_from_args
from benchmarks.compare_matmul_backends import _parse_indexing_list
from benchmarks.compare_matmul_backends import _parse_int_list
from benchmarks.compare_matmul_backends import _parse_optional_bool_list


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
        "skip_correctness": 0,
        "quack_path": None,
        "quack_tile_m": 256,
        "quack_tile_n": 256,
        "quack_cluster_m": 2,
        "quack_cluster_n": 1,
        "quack_pingpong": 0,
        "quack_persistent": 1,
        "quack_dynamic_persistent": 1,
        "quack_max_swizzle_size": 8,
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


if __name__ == "__main__":
    unittest.main()
