from __future__ import annotations

from pathlib import Path
import random
import tempfile
from unittest.mock import patch

from expecttest import TestCase
import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import import_path
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.random_search import RandomSearch
import helion.language as hl

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"
examples_matmul = import_path(examples_dir / "matmul.py").matmul


class TestAutotuner(TestCase):
    maxDiff = 16384

    def setUp(self):
        super().setUp()
        random.seed(112)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = examples_matmul.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedInline(
            "\n".join(map(repr, configs)),
            """\
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[0, 1]], num_warps=4, num_stages=3, indexing='pointer', l2_grouping=1, use_yz_grid=False)
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[1, 0]], num_warps=8, num_stages=3, indexing='block_ptr', l2_grouping=2)
helion.Config(block_sizes=[[64, 128], [16]], loop_orders=[[1, 0]], num_warps=4, num_stages=6, indexing='pointer', l2_grouping=2)
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[1, 0]], num_warps=4, num_stages=1, indexing='pointer', l2_grouping=1, use_yz_grid=False)
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[0, 1]], num_warps=2, num_stages=5, indexing='tensor_descriptor', l2_grouping=16)
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[0, 1]], num_warps=16, num_stages=7, indexing='pointer', l2_grouping=16)
helion.Config(block_sizes=[[256, 64], [32]], loop_orders=[[0, 1]], num_warps=8, num_stages=4, indexing='tensor_descriptor', l2_grouping=32)
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[0, 1]], num_warps=16, num_stages=6, indexing='pointer', l2_grouping=64)
helion.Config(block_sizes=[[16, 32], [32]], loop_orders=[[0, 1]], num_warps=1, num_stages=1, indexing='tensor_descriptor', l2_grouping=32)
helion.Config(block_sizes=[[16, 16], [16]], loop_orders=[[1, 0]], num_warps=4, num_stages=4, indexing='tensor_descriptor', l2_grouping=4)""",
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    def test_config_fragment1(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedInline(
            "\n".join(map(repr, configs)),
            """\
helion.Config(block_sizes=[[8, 16, 16]], loop_orders=[[0, 1, 2]], num_warps=4, num_stages=3, indexing='pointer', use_yz_grid=False)
helion.Config(block_sizes=[256], loop_orders=[[1, 0, 2]], num_warps=2, num_stages=8, indexing='pointer')
helion.Config(block_sizes=[256], loop_orders=[[0, 2, 1]], num_warps=32, num_stages=8, indexing='tensor_descriptor')
helion.Config(block_sizes=[[8, 16, 1]], loop_orders=[[2, 0, 1]], num_warps=8, num_stages=2, indexing='tensor_descriptor', use_yz_grid=False)
helion.Config(block_sizes=[1], loop_orders=[[1, 2, 0]], num_warps=4, num_stages=7, indexing='pointer')
helion.Config(block_sizes=[[1, 2, 128]], loop_orders=[[1, 0, 2]], num_warps=32, num_stages=2, indexing='block_ptr', use_yz_grid=False)
helion.Config(block_sizes=[[1, 16, 2]], loop_orders=[[0, 1, 2]], num_warps=4, num_stages=2, indexing='pointer', use_yz_grid=False)
helion.Config(block_sizes=[4096], loop_orders=[[1, 0, 2]], num_warps=1, num_stages=2, indexing='tensor_descriptor')
helion.Config(block_sizes=[[1, 2, 1]], loop_orders=[[1, 0, 2]], num_warps=4, num_stages=7, indexing='block_ptr', use_yz_grid=False)
helion.Config(block_sizes=[1], loop_orders=[[2, 1, 0]], num_warps=1, num_stages=8, indexing='block_ptr')""",
        )

    def test_save_load_config(self):
        config = helion.Config(
            block_sizes=[[64, 64], [32]],
            loop_orders=[[1, 0]],
            num_warps=2,
            num_stages=1,
            indexing="block_ptr",
            l2_grouping=32,
        )
        with tempfile.NamedTemporaryFile() as f:
            config.save(f.name)
            loaded_config = helion.Config.load(f.name)
            self.assertEqual(config, loaded_config)
        self.assertExpectedInline(
            config.to_json(),
            """\
{
  "block_sizes": [
    [
      64,
      64
    ],
    [
      32
    ]
  ],
  "loop_orders": [
    [
      1,
      0
    ]
  ],
  "num_warps": 2,
  "num_stages": 1,
  "indexing": "block_ptr",
  "l2_grouping": 32
}""",
        )

    def test_run_fixed_config(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1024], loop_orders=[[0, 2, 1]], num_warps=8
            )
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))

    def test_run_finite_search(self):
        @helion.kernel(
            configs=[
                helion.Config(block_sizes=[1024], loop_orders=[[0, 2, 1]], num_warps=8),
                helion.Config(block_sizes=[1024], num_warps=8),
                helion.Config(block_sizes=[1, 64, 64], num_warps=8),
                helion.Config(block_sizes=[1, 1, 512], num_warps=8),
            ],
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))
        torch.testing.assert_close(add(*args), sum(args))

    def test_random_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        best = RandomSearch(bound_kernel, args, 5).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    def test_differential_evolution_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        best = DifferentialEvolutionSearch(
            bound_kernel, args, 5, num_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    def test_use_default_config(self):
        @helion.kernel(use_default_config=True)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        result = add(*args)
        torch.testing.assert_close(result, sum(args))
