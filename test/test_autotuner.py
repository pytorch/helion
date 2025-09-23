from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import importlib
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import skipIfRocm
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner.config_generation import ConfigGeneration
import helion.autotuner.finite_search as finite_search_module
from helion.autotuner.random_search import RandomSearch

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
import helion.language as hl
from helion.language import loops

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"
examples_matmul = import_path(examples_dir / "matmul.py").matmul


def trigger_cuda_unrecoverable_error() -> tuple[
    BoundKernel,
    tuple[object, ...],
    helion.Config,
    helion.Config,
]:
    B, T, V = 8, 512, 4096

    @helion.kernel(
        static_shapes=True,
    )
    def kernel_small_first_dim_block_size(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        log_target: bool = False,
    ) -> torch.Tensor:
        BT, V_local = y_pred.shape
        assert y_true.shape == y_pred.shape, (
            f"Shape mismatch: {y_true.shape} != {y_pred.shape}"
        )

        loss = torch.zeros((BT,), dtype=torch.float32, device=y_pred.device)
        kl_loss = torch.zeros_like(y_pred)

        block_size_n = hl.register_block_size(V_local)
        BT_SIZE = 2
        loss_sum = torch.zeros(
            [BT_SIZE, block_size_n], dtype=torch.float32, device=y_pred.device
        )

        for tile_bt in hl.tile(BT, block_size=BT_SIZE):
            loss_sum[:, :] = hl.zeros(
                [BT_SIZE, block_size_n], dtype=torch.float32
            )
            for tile_v in hl.tile(V_local, block_size=block_size_n):
                y_true_val = y_true[tile_bt, tile_v]

                if log_target:
                    kl_loss[tile_bt, tile_v] = y_true_val * 0.0
                else:
                    kl_loss[tile_bt, tile_v] = y_true_val * 0.0

                hl.atomic_add(loss_sum, [tile_bt, tile_v], kl_loss[tile_bt, tile_v])

            loss[tile_bt] = loss_sum[:, :].sum(dim=-1)

        return torch.sum(loss) / BT

    y_pred = torch.randn(B * T, V, device=DEVICE, dtype=torch.float32)
    y_true = torch.randn(B * T, V, device=DEVICE, dtype=torch.float32)

    args = (y_pred, y_true, False)
    bound_kernel = kernel_small_first_dim_block_size.bind(args)

    bad_config = helion.Config(
        block_sizes=[128],
        num_stages=5,
        num_warps=1,
        pid_type="persistent_interleaved",
        range_flattens=[None, True],
        range_multi_buffers=[None, True],
        range_num_stages=[2, 3],
        range_unroll_factors=[1, 2],
    )

    good_config = bound_kernel.config_spec.default_config()
    return bound_kernel, args, bad_config, good_config


@contextmanager
def without_env_var(name: str):
    sentinel = object()
    previous = os.environ.pop(name, sentinel)
    try:
        yield
    finally:
        if previous is not sentinel:
            os.environ[name] = previous


class RecordingRandomSearch(RandomSearch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[float] = []

    def _autotune(self):
        self.samples.append(random.random())
        return super()._autotune()


class TestAutotuner(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        random.seed(112)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("failure on rocm")
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = examples_matmul.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    def test_config_fragment1(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    def test_save_load_config(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
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
        self.assertExpectedJournal(config.to_json())

    def test_run_fixed_config(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1024, 1, 1],
                flatten_loops=[True],
                loop_orders=[[0, 2, 1]],
                num_warps=8,
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
                helion.Config(
                    block_sizes=[1024, 1, 1],
                    flatten_loops=[True],
                    loop_orders=[[0, 2, 1]],
                    num_warps=8,
                ),
                helion.Config(
                    block_sizes=[1024, 1, 1], flatten_loops=[True], num_warps=8
                ),
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

    @skipIfRocm("too slow on rocm")
    def test_random_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        random.seed(123)
        best = RandomSearch(bound_kernel, args, 20).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skipIfRocm("too slow on rocm")
    def test_differential_evolution_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        random.seed(123)
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

    def test_autotuner_disabled(self):
        @helion.kernel(use_default_config=False)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        with (
            patch.dict(os.environ, {"HELION_DISALLOW_AUTOTUNING": "1"}),
            pytest.raises(
                expected_exception=helion.exc.AutotuningDisallowedInEnvironment,
                match="Autotuning is disabled by HELION_DISALLOW_AUTOTUNING=1, please provide a config to @helion.kernel via the config= argument.",
            ),
        ):
            add(*args)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_persistent_worker_env_preserves_cuda_context(self):
        import helion.autotuner.base_search as base_search_module

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_PERSISTENT_WORKER": "1",
                "HELION_AUTOTUNE_TIMING": "0",
            },
            clear=False,
        ):
            base_search = importlib.reload(base_search_module)
            finite_search = importlib.reload(finite_search_module)

            (
                bound_kernel,
                args,
                bad_config,
                good_config,
            ) = trigger_cuda_unrecoverable_error()

            search = finite_search.FiniteSearch(
                bound_kernel,
                args,
                configs=[bad_config, good_config],
            )
            self.assertTrue(search._use_persistent_worker)

            try:
                best_config = search.autotune()
                self.assertIs(best_config, good_config)

                self.assertEqual(search.counters["benchmark"], 2)

                fn = bound_kernel.compile_config(best_config)
                result = fn(*args)
                self.assertTrue(torch.is_tensor(result))
                self.assertTrue(torch.all(torch.isfinite(result)))
            finally:
                search._close_benchmark_worker()

            torch.cuda.synchronize()
            torch.randn(1, device=DEVICE)

        importlib.reload(base_search_module)
        importlib.reload(finite_search_module)


class TestAutotuneRandomSeed(RefEagerTestDisabled, TestCase):
    def _autotune_and_record(self, **settings: object) -> float:
        search_capture: dict[str, RecordingRandomSearch] = {}

        def autotuner_factory(bound_kernel, args, **kwargs):
            search = RecordingRandomSearch(bound_kernel, args, count=4, **kwargs)
            search_capture["search"] = search
            return search

        kernel_settings = {
            "autotuner_fn": autotuner_factory,
        }
        kernel_settings.update(settings)

        @helion.kernel(**kernel_settings)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound_kernel = add.bind(args)
        bound_kernel.autotune(args)
        torch.testing.assert_close(bound_kernel(*args), sum(args), rtol=1e-2, atol=1e-1)

        search = search_capture["search"]
        assert search.samples, (
            "expected RecordingRandomSearch to record a random sample"
        )
        return search.samples[0]

    @skipIfRocm("accuracy difference")
    def test_autotune_random_seed_from_env_var(self) -> None:
        # same env var value -> same random sample
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "4242"}, clear=False
        ):
            first = self._autotune_and_record()
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "4242"}, clear=False
        ):
            second = self._autotune_and_record()
        self.assertEqual(first, second)

        # different env var values -> different random samples
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "101"}, clear=False
        ):
            first = self._autotune_and_record()
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "102"}, clear=False
        ):
            second = self._autotune_and_record()
        self.assertNotEqual(first, second)

    @skipIfRocm("accuracy difference")
    def test_autotune_random_seed_from_settings(self) -> None:
        # same autotune_random_seed setting -> same random sample
        first = self._autotune_and_record(autotune_random_seed=4242)
        second = self._autotune_and_record(autotune_random_seed=4242)
        self.assertEqual(first, second)

        # different autotune_random_seed settings -> different random samples
        first = self._autotune_and_record(autotune_random_seed=101)
        second = self._autotune_and_record(autotune_random_seed=102)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
