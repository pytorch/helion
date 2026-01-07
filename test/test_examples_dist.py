from __future__ import annotations

import contextlib
from datetime import timedelta
from functools import partial
import os
from typing import ClassVar
from unittest.mock import patch

from examples.distributed.utils import distributed_benchmark
import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import run_tests

import helion
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipIfRocm


@instantiate_parametrized_tests
class TestExamplesDist(TestCase, MultiProcessTestCase):
    _nvshmem_env: ClassVar[dict[str, str]] = {
        # Configure NVSHMEM to use smaller heap and work without NVSwitch
        # Default heap is 128GB which fails cuMemMap on AWS H100 instances
        "NVSHMEM_SYMMETRIC_SIZE": "4G",
        # Disable NVLink Switch features (not available on AWS H100 instances)
        "NVSHMEM_DISABLE_NVLS": "1",
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(patch.dict(os.environ, cls._nvshmem_env))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_stack.close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _cleanup_process(self):
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_all_gather_matmul(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "all_gather_matmul.py")

        M, N, K = 4096, 6656, 16384

        a_shared = symm_mem.empty(
            M // self.world_size, K, dtype=torch.bfloat16, device=self.device
        ).normal_()

        b = (
            torch.randn((K, N), device=self.device, dtype=torch.bfloat16)
            .T.contiguous()
            .T
        )

        symm_mem_group = dist.group.WORLD
        if symm_mem_group is None:
            raise RuntimeError("No symmetric memory group available")
        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=symm_mem_group)
        a_shape = list(a_shared.shape)
        a_shape[0] *= symm_mem_hdl.world_size
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
        progress = torch.zeros(
            symm_mem_hdl.world_size,
            dtype=torch.uint32,
            device=a_shared.device,
        )
        backend_stream = mod.copy_engine_all_gather_w_progress(
            a_out, a_shared, progress, 1
        )

        code, result = code_and_output(
            mod.helion_matmul_w_progress,
            (a_out, a_shared, b, progress, 1, symm_mem_hdl.rank),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        # Synchronize CUDA before running reference
        torch.cuda.synchronize()

        golden_a = a_shared.clone()
        ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_matmul(
            golden_a, [b], gather_dim=0, group_name=symm_mem_group.group_name
        )

        torch.testing.assert_close(result, mm_golden[0], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(a_out, ag_golden)

        torch.cuda.current_stream().wait_stream(backend_stream)
        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_all_reduce(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "all_reduce.py")

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        ).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)
        local_signal_pad = symm_mem_hdl.get_signal_pad(
            symm_mem_hdl.rank, dtype=torch.int32
        ).view(-1, symm_mem_hdl.world_size)
        signal_pad_addrs = mod.dev_array_to_tensor_short(
            symm_mem_hdl.signal_pad_ptrs_dev,
            (symm_mem_hdl.world_size,),
            dtype=torch.uint64,
            device=a_shared.device,
        )

        code, result = code_and_output(
            mod.one_shot_all_reduce_kernel,
            (
                signal_pad_addrs,
                local_signal_pad,
                a_shared,
                symm_mem_hdl.rank,
                group.group_name,
            ),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        # Synchronize CUDA before running reference
        torch.cuda.synchronize()

        a_shared_ref = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        )
        a_shared_ref.copy_(a_shared)
        expected = mod.reference_one_shot_all_reduce(a_shared_ref)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @pytest.mark.timeout(120)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_one_shot_allreduce_bias_rmsnorm_autotune(self):
        """Test distributed autotuning for one_shot_allreduce_bias_rmsnorm kernel."""
        from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
            create_benchmark_inputs,
        )
        from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
            one_shot_allreduce_bias_rmsnorm_kernel,
        )
        from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
            reference_one_shot_allreduce_bias_rmsnorm,
        )

        self._init_process()
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        test_kernel = helion.kernel(
            configs=[
                helion.Config(block_sizes=[8], num_warps=8),
                helion.Config(block_sizes=[16], num_warps=4),
            ],
            autotune_baseline_fn=lambda x,
            buf,
            bias,
            weight,
            sig,
            eps,
            rank,
            ws,
            gn: reference_one_shot_allreduce_bias_rmsnorm(x, bias, weight, eps),
            autotune_benchmark_fn=partial(
                distributed_benchmark, inputs_fn=create_benchmark_inputs
            ),
            static_shapes=True,
            autotune_effort="full",
        )(one_shot_allreduce_bias_rmsnorm_kernel.fn)

        (
            x,
            symm_mem_buffer,
            bias,
            weight,
            signal_pad_ptrs_dev,
            eps,
            rank,
            world_size,
            group_name,
        ) = create_benchmark_inputs()
        x_ref = x.clone()

        result = test_kernel(
            x,
            symm_mem_buffer,
            bias,
            weight,
            signal_pad_ptrs_dev,
            eps,
            rank,
            world_size,
            group_name,
        )

        torch.cuda.synchronize()
        expected = reference_one_shot_allreduce_bias_rmsnorm(x_ref, bias, weight, eps)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @pytest.mark.timeout(120)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_one_shot_allreduce_bias_rmsnorm_autotune_skip_hanging_config(self):
        """Test distributed autotuning skips configs that timeout.

        This test forces a timeout for block_sizes=[1] by using a very short
        timeout (0.5s) which causes the worker process to be killed. This verifies:
        1. Workers are properly killed when they exceed the timeout
        2. The timed-out config is excluded from tracking (timing=inf)
        3. Only successful configs ([8] and [16]) are considered for selection
        """
        from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
            create_benchmark_inputs,
        )
        from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
            one_shot_allreduce_bias_rmsnorm_kernel,
        )
        from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
            reference_one_shot_allreduce_bias_rmsnorm,
        )

        self._init_process()
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        # Track which configs are benchmarked
        benchmarked_block_sizes: list[list[int]] = []

        def tracking_benchmark(
            fns: list,
            *,
            repeat: int = 1,
            desc: str | None = None,
            timeout: float = 30.0,
            inputs_fn=None,
        ):
            """Wrapper around distributed_benchmark that tracks successfully benchmarked configs.

            Uses a very short timeout for block_sizes=[1] to force the worker to be
            killed, testing the timeout skip behavior.
            """
            # Use very short timeout for block_sizes=[1] to force worker to be killed
            # Workers take ~12-15s to start up, so 0.5s timeout will definitely kill them
            effective_timeout = timeout
            for fn in fns:
                if fn.config.block_sizes == [1]:
                    effective_timeout = 0.5  # Force timeout by killing worker
                    break

            results = distributed_benchmark(
                fns, repeat=repeat, desc=desc, timeout=effective_timeout, inputs_fn=inputs_fn
            )

            # Only track actual benchmarks (repeat > 1) that didn't timeout
            # benchmark_function calls the custom benchmark fn twice per config:
            # once with repeat=1 for warmup/accuracy, once with repeat=50 for actual
            if repeat > 1:
                for fn, result in zip(fns, results, strict=True):
                    # Skip configs that timed out (timing == inf)
                    if result.timing != float("inf"):
                        block_sizes = fn.config.block_sizes
                        benchmarked_block_sizes.append(block_sizes)
            return results

        # Re-decorate kernel with test configs
        test_kernel = helion.kernel(
            configs=[
                helion.Config(block_sizes=[8], num_warps=8),
                helion.Config(
                    block_sizes=[1], num_warps=8
                ),  # forced timeout - tracking_benchmark uses 0.5s timeout for this
                helion.Config(block_sizes=[16], num_warps=8),
            ],
            autotune_baseline_fn=lambda x,
            buf,
            bias,
            weight,
            sig,
            eps,
            rank,
            ws,
            gn: reference_one_shot_allreduce_bias_rmsnorm(x, bias, weight, eps),
            autotune_benchmark_fn=partial(
                tracking_benchmark, inputs_fn=create_benchmark_inputs
            ),
            static_shapes=True,
            autotune_effort="full",
        )(one_shot_allreduce_bias_rmsnorm_kernel.fn)

        (
            x,
            symm_mem_buffer,
            bias,
            weight,
            signal_pad_ptrs_dev,
            eps,
            rank,
            world_size,
            group_name,
        ) = create_benchmark_inputs()
        x_ref = x.clone()

        args = (
            x,
            symm_mem_buffer,
            bias,
            weight,
            signal_pad_ptrs_dev,
            eps,
            rank,
            world_size,
            group_name,
        )
        result = test_kernel(*args)

        torch.cuda.synchronize()  # Verify GPU not hung

        # Verify only non-hanging configs were successfully benchmarked
        # Config [1] is simulated as a timeout and should be skipped from tracking
        assert benchmarked_block_sizes == [[8], [16]], (
            f"Expected only [8] and [16] to be benchmarked, got: {benchmarked_block_sizes}"
        )

        # Verify the selected config is one of the two valid configs (not the timed-out one)
        bound_kernel = test_kernel.bind(args)
        selected_config = bound_kernel._config
        assert selected_config is not None, "No config was selected by autotuner"
        assert selected_config.block_sizes in [[8], [16]], (
            f"Selected config block_sizes={selected_config.block_sizes} "
            f"not in expected set [[8], [16]]"
        )

        expected = reference_one_shot_allreduce_bias_rmsnorm(x_ref, bias, weight, eps)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @pytest.mark.timeout(120)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_all_reduce_autotune(self):
        """Test distributed autotuning for all_reduce kernel."""
        from examples.distributed.all_reduce import create_benchmark_inputs
        from examples.distributed.all_reduce import one_shot_all_reduce_kernel
        from examples.distributed.all_reduce import reference_one_shot_all_reduce

        self._init_process()
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        test_kernel = helion.kernel(
            configs=[
                helion.Config(block_sizes=[8192], num_warps=32),
                helion.Config(block_sizes=[4096], num_warps=16),
            ],
            autotune_baseline_fn=lambda sig,
            pad,
            a,
            rank,
            gn: reference_one_shot_all_reduce(a),
            autotune_benchmark_fn=partial(
                distributed_benchmark, inputs_fn=create_benchmark_inputs
            ),
            static_shapes=True,
            autotune_effort="full",
        )(one_shot_all_reduce_kernel.fn)

        (signal_pad_addrs, local_signal_pad, a_shared, rank, group_name) = (
            create_benchmark_inputs()
        )
        a_ref = a_shared.clone()

        result = test_kernel(
            signal_pad_addrs,
            local_signal_pad,
            a_shared,
            rank,
            group_name,
        )

        torch.cuda.synchronize()
        expected = reference_one_shot_all_reduce(a_ref)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @pytest.mark.timeout(120)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_matmul_reduce_scatter_autotune(self):
        """Test distributed autotuning for matmul_reduce_scatter kernel."""
        from examples.distributed.matmul_reduce_scatter import create_benchmark_inputs
        from examples.distributed.matmul_reduce_scatter import (
            matmul_reduce_scatter_kernel,
        )
        from examples.distributed.matmul_reduce_scatter import (
            reference_matmul_reduce_scatter,
        )

        self._init_process()
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        test_kernel = helion.kernel(
            configs=[
                helion.Config(
                    block_sizes=[64, 64, 32],
                    num_warps=8,
                    num_stages=3,
                    indexing="block_ptr",
                ),
                helion.Config(
                    block_sizes=[32, 32, 32],
                    num_warps=4,
                    num_stages=2,
                    indexing="block_ptr",
                ),
            ],
            autotune_baseline_fn=lambda a,
            b,
            buf,
            sig,
            rank,
            ws,
            gn: reference_matmul_reduce_scatter(a, b),
            autotune_benchmark_fn=partial(
                distributed_benchmark, inputs_fn=create_benchmark_inputs
            ),
            static_shapes=True,
            autotune_effort="full",
            dot_precision="ieee",  # Use full FP32 precision to match reference
        )(matmul_reduce_scatter_kernel.fn)

        (a, b, symm_mem_buffer, signal_pad_ptrs_dev, rank, world_size, group_name) = (
            create_benchmark_inputs()
        )
        a_ref, b_ref = a.clone(), b.clone()

        result = test_kernel(
            a,
            b,
            symm_mem_buffer,
            signal_pad_ptrs_dev,
            rank,
            world_size,
            group_name,
        )

        torch.cuda.synchronize()
        expected = reference_matmul_reduce_scatter(a_ref, b_ref)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @pytest.mark.timeout(180)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_all_gather_matmul_autotune(self):
        """Test distributed autotuning for all_gather_matmul kernel."""
        from examples.distributed.all_gather_matmul import create_benchmark_inputs
        from examples.distributed.all_gather_matmul import helion_matmul_w_progress

        self._init_process()
        group = dist.group.WORLD

        test_kernel = helion.kernel(
            configs=[
                helion.Config(
                    block_sizes=[128, 256, 64],
                    num_warps=8,
                    num_stages=3,
                    indexing="block_ptr",
                ),
                helion.Config(
                    block_sizes=[64, 128, 32],
                    num_warps=4,
                    num_stages=2,
                    indexing="block_ptr",
                ),
            ],
            autotune_baseline_fn=lambda a,
            a_shared,
            b,
            progress,
            splits_per_rank,
            rank: torch.ops.symm_mem.fused_all_gather_matmul(
                a_shared.clone(),
                [b],
                gather_dim=0,
                group_name=dist.group.WORLD.group_name,
            )[1][0],
            autotune_benchmark_fn=partial(
                distributed_benchmark, inputs_fn=create_benchmark_inputs
            ),
            static_shapes=True,
            autotune_effort="full",
            dot_precision="ieee",
        )(helion_matmul_w_progress.fn)

        a_out, a_shared, b, progress, splits_per_rank, rank = create_benchmark_inputs()

        result = test_kernel(a_out, a_shared, b, progress, splits_per_rank, rank)

        torch.cuda.synchronize()
        golden_a = a_shared.clone()
        ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_matmul(
            golden_a, [b], gather_dim=0, group_name=group.group_name
        )
        torch.testing.assert_close(result, mm_golden[0], rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_one_shot_allreduce_bias_rmsnorm(self):
        self._init_process()

        mod = import_path(
            EXAMPLES_DIR / "distributed" / "one_shot_allreduce_bias_rmsnorm.py"
        )

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        N, D = 128, 4096
        dtype = torch.float32
        eps = 1e-5

        x = torch.randn(N, D, dtype=dtype, device=self.device)

        torch.manual_seed(42)
        bias = torch.randn(D, dtype=dtype, device=self.device)
        weight = torch.randn(D, dtype=dtype, device=self.device)

        x_ref = x.clone()

        symm_mem_buffer = symm_mem.empty(N, D, dtype=dtype, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        code, result = code_and_output(
            mod.one_shot_allreduce_bias_rmsnorm_kernel,
            (
                x,
                symm_mem_buffer,
                bias,
                weight,
                symm_mem_hdl.signal_pad_ptrs_dev,
                eps,  # EPS constexpr
                symm_mem_hdl.rank,  # RANK constexpr
                symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
                group.group_name,  # GROUP_NAME constexpr
            ),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        torch.cuda.synchronize()

        expected = mod.reference_one_shot_allreduce_bias_rmsnorm(
            x_ref, bias, weight, eps
        )

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()

    @pytest.mark.timeout(120)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_distributed_seed_sync_and_timing_consistency(self):
        """Test that seed synchronization works and all ranks receive same timing.

        This test intercepts the normal autotuning flow to verify:
        1. All ranks use the same random seed (captured from Settings)
        2. All ranks receive the same timing from distributed_benchmark
        """
        from examples.distributed.all_reduce import create_benchmark_inputs
        from examples.distributed.all_reduce import one_shot_all_reduce_kernel
        from examples.distributed.all_reduce import reference_one_shot_all_reduce

        self._init_process()

        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        # Capture seed and timing during the normal autotuning flow
        # Note: We capture the seed from the parent process (not the worker) since
        # Settings reads the same env vars / defaults and the seed is synchronized
        # across ranks before autotuning begins.
        from helion.runtime.settings import Settings
        settings = Settings()
        captured_seed = settings.autotune_random_seed

        captured_timings: list[list[float]] = []

        original_distributed_benchmark = distributed_benchmark

        def intercepting_benchmark(
            fns,
            *,
            repeat=1,
            desc=None,
            timeout=30.0,
            inputs_fn,
        ):
            # Call the original distributed_benchmark
            results = original_distributed_benchmark(
                fns,
                repeat=repeat,
                desc=desc,
                timeout=timeout,
                inputs_fn=inputs_fn,
            )
            # Capture the timings received by this rank
            if results and repeat > 1:  # Only capture actual benchmark runs, not warmup
                timings = [r.timing if hasattr(r, 'timing') else r for r in results]
                captured_timings.append(timings)
            return results

        test_kernel = helion.kernel(
            configs=[
                helion.Config(block_sizes=[8192], num_warps=32),
                helion.Config(block_sizes=[4096], num_warps=16),
            ],
            autotune_baseline_fn=lambda sig,
            pad,
            a,
            rank,
            gn: reference_one_shot_all_reduce(a),
            autotune_benchmark_fn=partial(
                intercepting_benchmark, inputs_fn=create_benchmark_inputs
            ),
            static_shapes=True,
            autotune_effort="full",
        )(one_shot_all_reduce_kernel.fn)

        (signal_pad_addrs, local_signal_pad, a_shared, rank, group_name) = (
            create_benchmark_inputs()
        )

        # Run the kernel - this triggers autotuning with our intercepting functions
        result = test_kernel(
            signal_pad_addrs,
            local_signal_pad,
            a_shared,
            rank,
            group_name,
        )

        torch.cuda.synchronize()

        # Verify result is correct
        a_ref = a_shared.clone()
        expected = reference_one_shot_all_reduce(a_ref)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        # Now verify seed and timing consistency across all ranks
        # Create a Gloo group for CPU-side verification
        gloo_group = dist.new_group(backend="gloo")

        # Verify seed consistency: gather seeds from all ranks
        assert captured_seed is not None, f"Rank {self.rank}: No seed was captured"
        seed_tensor = torch.tensor([captured_seed], dtype=torch.int64, device="cpu")
        gathered_seeds = [
            torch.zeros(1, dtype=torch.int64, device="cpu")
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_seeds, seed_tensor, group=gloo_group)
        all_seeds = [int(s.item()) for s in gathered_seeds]
        assert all(seed == all_seeds[0] for seed in all_seeds), (
            f"Seeds not synchronized across ranks: {all_seeds}"
        )

        # Verify timing consistency: gather timings from all ranks
        assert len(captured_timings) > 0, f"Rank {self.rank}: No timings were captured"
        # Use the last captured timing batch (the actual benchmark, not warmup)
        my_timings = captured_timings[-1]
        timings_tensor = torch.tensor(my_timings, dtype=torch.float64, device="cpu")
        gathered_timings = [
            torch.zeros_like(timings_tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_timings, timings_tensor, group=gloo_group)

        # Verify all ranks received identical timings
        for i in range(1, self.world_size):
            assert torch.allclose(gathered_timings[0], gathered_timings[i]), (
                f"Timing mismatch between rank 0 and rank {i}: "
                f"{gathered_timings[0].tolist()} vs {gathered_timings[i].tolist()}"
            )

        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_matmul_reduce_scatter(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        M, N, K = 512, 768, 1024
        dtype = torch.float32

        # Each rank has the same random seed for reproducibility
        torch.manual_seed(42 + self.rank)
        a = torch.randn(M, K, dtype=dtype, device=self.device)

        # Weight matrix is the same across all ranks
        torch.manual_seed(42)
        b = torch.randn(K, N, dtype=dtype, device=self.device)

        # Clone for reference computation
        a_ref = a.clone()
        b_ref = b.clone()

        # Setup symmetric memory like the wrapper does
        symm_mem_buffer = symm_mem.empty(M, N, dtype=dtype, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        code, result = code_and_output(
            mod.matmul_reduce_scatter_kernel,
            (
                a,
                b,
                symm_mem_buffer,
                symm_mem_hdl.signal_pad_ptrs_dev,
                symm_mem_hdl.rank,  # RANK constexpr
                symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
                group.group_name,  # GROUP_NAME constexpr
            ),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        torch.cuda.synchronize()

        expected = mod.reference_matmul_reduce_scatter(a_ref, b_ref)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self._cleanup_process()


if __name__ == "__main__":
    run_tests()
