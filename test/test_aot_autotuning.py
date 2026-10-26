"""
Tests for AOT Autotuning Framework
==================================

Tests for the collect/measure/evaluate workflow.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib.util
import inspect
import itertools
import json
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch
import weakref

import numpy as np
import pytest
import torch
from torch._inductor.runtime.triton_compat import OutOfResources
from torch._inductor.utils import fresh_cache

import helion
from helion import exc
from helion._compiler.backend import TritonBackend
from helion._hardware import HardwareInfo
from helion._hardware import get_hardware_info
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
import helion.autotuner.aot_cache as aot_cache_module
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import _deserialize_tuple
from helion.autotuner.aot_cache import _deserialize_value
from helion.autotuner.aot_cache import _serialize_tuple
from helion.autotuner.aot_cache import _serialize_value
from helion.autotuner.aot_cache import get_aot_mode
from helion.autotuner.aot_compile import _standalone_call_key
from helion.autotuner.aot_compile import generate_standalone_file
from helion.autotuner.aot_kernel import aot_key
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import ShapeConfigData
from helion.autotuner.heuristic_generator import compute_validity_partitions
from helion.autotuner.heuristic_generator import select_config_subset
import helion.language as hl
from helion.runtime.config import Config
import helion.runtime.triton.launcher as launcher_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from pathlib import Path


def test_future_cuda_arch_does_not_load_sm103_aot_fallback(tmp_path: Path) -> None:
    source_path = tmp_path / "demo.py"
    source_path.touch()
    (tmp_path / "_helion_aot_demo_cuda_sm103.py").touch()
    sm100_heuristic = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    sm100_heuristic.touch()
    future_hardware = HardwareInfo(
        device_kind="cuda",
        hardware_name="NVIDIA future GPU",
        runtime_version="14.0",
        compute_capability="sm120",
    )

    aot_cache_module.clear_heuristic_cache()
    try:
        with patch.object(
            aot_cache_module,
            "get_hardware_info",
            return_value=future_hardware,
        ):
            assert aot_cache_module.find_heuristic_file(source_path) == sm100_heuristic
    finally:
        aot_cache_module.clear_heuristic_cache()


def _make_aot_fit_matmul() -> helion.Kernel:
    @helion.aot_kernel()
    def aot_fit_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(out.dtype)
        return out

    return aot_fit_matmul


def _make_aot_fit_matmul_small_gpu() -> helion.Kernel:
    # The same matmul as aot_fit_matmul, but its own generated code: Triton
    # checks a kernel's shared memory only the first time the kernel loads.
    @helion.aot_kernel()
    def aot_fit_matmul_small_gpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(out.dtype)
        return out

    return aot_fit_matmul_small_gpu


def _make_aot_fit_scaled_matmul() -> helion.Kernel:
    @helion.aot_kernel()
    def aot_fit_scaled_matmul(
        x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            if scale.shape[0] == m:
                scale_blk = scale[tile_m, :]
            else:
                scale_blk = scale[0, 0].expand(tile_m.block_size, 1)
            out[tile_m, tile_n] = (scale_blk * acc).to(out.dtype)
        return out

    return aot_fit_scaled_matmul


def _make_aot_fit_count_calls() -> helion.Kernel:
    @helion.aot_kernel(ignore_warnings=[exc.TensorOperationInWrapper])
    def aot_fit_count_calls(
        x: torch.Tensor, y: torch.Tensor, calls: torch.Tensor
    ) -> torch.Tensor:
        calls.add_(1)
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(out.dtype)
        return out

    return aot_fit_count_calls


@contextlib.contextmanager
def _heuristic_for_older_gpu(
    tmp_path: Path, configs: dict[str, dict[str, object]]
) -> Iterator[None]:
    """Make AOT selection pick *configs* from a heuristic for an older GPU."""
    hardware = get_hardware_info()
    computes = hardware.get_compatible_compute_ids()
    if len(computes) < 2:
        pytest.skip("no older compute capability to fall back to")
    source_stem = os.path.basename(__file__).removesuffix(".py")
    heuristic = tmp_path / (
        f"_helion_aot_{source_stem}_{hardware.device_kind}_{computes[1]}.py"
    )
    heuristic.write_text(
        "".join(
            f"def autotune_{name}(*args):\n    return {config!r}\n"
            for name, config in configs.items()
        )
    )
    with (
        patch.dict(os.environ, {"HELION_HEURISTIC_DIR": str(tmp_path)}),
        patch.object(TritonBackend, "_fitted_kernels", weakref.WeakSet()),
    ):
        AOTAutotuneCache.clear_caches()
        try:
            yield
        finally:
            AOTAutotuneCache.clear_caches()


@contextlib.contextmanager
def _shared_memory_limit(limit: int) -> Iterator[None]:
    """Make Triton treat this GPU as one with *limit* bytes of shared memory."""
    import triton.compiler.compiler as triton_compiler

    if limit > triton_compiler.max_shared_mem(torch.cuda.current_device()):
        pytest.skip(f"this GPU has less than {limit} bytes of shared memory")
    with patch.object(triton_compiler, "max_shared_mem", lambda device: limit):
        yield


class _FakeTritonKernel:
    """Stands in for the Triton kernel generated for *config*.

    Launched with some ``num_warps`` and ``num_stages``, it needs ``usage`` of a
    resource for *config* with those launch parameters.
    """

    def __init__(
        self, name: str, config: Config, bound_kernel: _FakeBoundKernel
    ) -> None:
        self.__name__ = name
        self.config = config
        self.bound_kernel = bound_kernel

    def run(
        self,
        *args: object,
        grid: object,
        warmup: bool,
        num_warps: int,
        num_stages: int,
        **kwargs: object,
    ) -> object:
        bound_kernel = self.bound_kernel
        launched = Config.from_dict(
            {**self.config.config, "num_warps": num_warps, "num_stages": num_stages}
        )
        usage = bound_kernel.usage(launched)
        error = OutOfResources(usage, bound_kernel.limit, bound_kernel.resource)
        if warmup:

            def load() -> None:
                if usage > bound_kernel.limit:
                    raise error

            return SimpleNamespace(_init_handles=load)
        bound_kernel.attempts.append((self.__name__, launched))
        if (launch_error := bound_kernel.launch_error(launched)) is not None:
            raise launch_error
        if usage > bound_kernel.limit:
            raise error
        bound_kernel.launched.append((self.__name__, launched))
        bound_kernel.launched_kernels.append(self.config)
        return SimpleNamespace()


class _FakeKernel:
    name = "demo"


_FAKE_BOUND_KERNEL_IDS = itertools.count()


class _FakeBoundKernel:
    """A bound kernel whose configs' kernels fail to launch when they need more
    of a resource than ``limit``.

    Its generated host code calls ``before_launch``, launches each kernel in
    ``kernel_names`` through the launcher it is given, then calls
    ``after_launch``.
    """

    def __init__(
        self,
        tmp_path: Path,
        limit: int,
        usage: Callable[[Config], int],
        resource: str = "shared memory",
        kernel_names: Sequence[str] = ("_helion_demo",),
        launch_kwargs: str = "",
        grid: Callable[[Config], str] = lambda config: "(1,)",
    ) -> None:
        self.tmp_path = tmp_path / f"bound_kernel_{next(_FAKE_BOUND_KERNEL_IDS)}"
        self.tmp_path.mkdir()
        self.limit = limit
        self.usage = usage
        self.resource = resource
        self.kernel_names = kernel_names
        self.launch_kwargs_source = launch_kwargs
        self.grid = grid
        self.kernel = _FakeKernel()
        self.env = SimpleNamespace(device=torch.device("meta"), process_group_name=None)
        self.settings = SimpleNamespace(print_output_code=False)
        self.launch_error: Callable[[Config], Exception | None] = lambda config: None
        self.before_launch: Callable[..., object] = lambda *args: None
        self.after_launch: Callable[..., object] = lambda *args: None
        self._compile_cache: dict[Config, Callable[..., object]] = {}
        self._cache_path_map: dict[Config, str] = {}
        self._config_from_other_device = False
        self._run: Callable[..., object] | None = None
        self._config: Config | None = None
        self.compiled: list[Config] = []
        self.printed: list[Config] = []
        self.attempts: list[tuple[str, Config]] = []
        self.launched: list[tuple[str, Config]] = []
        self.launched_kernels: list[Config] = []
        self.default_launches: list[object] = []

    def _normalized_config_copy(self, config: Config) -> Config:
        return config

    def compile_config(
        self, config: Config, *, allow_print: bool = True
    ) -> Callable[..., object]:
        if (compiled := self._compile_cache.get(config)) is not None:
            return compiled
        backend = TritonBackend()
        launch = (
            f"num_warps={backend.effective_num_warps(config)}, "
            f"num_stages={config.num_stages}{self.launch_kwargs_source}"
        )
        source = "def demo(x, *, _launcher=_default_launcher):\n"
        source += (
            "    before_launch(x)\n    if not x.numel():\n        return x\n"
            + "".join(
                f"    _launcher({name}, {self.grid(config)}, x, {launch})\n"
                for name in self.kernel_names
            )
        )
        source += "    after_launch(x)\n    return x\n"
        path = self.tmp_path / f"fake_host_{len(self.compiled)}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        for name in self.kernel_names:
            setattr(module, name, _FakeTritonKernel(name, config, self))
        module.torch = torch
        module.before_launch = lambda *args: self.before_launch(*args)
        module.after_launch = lambda *args: self.after_launch(*args)
        module._default_launcher = lambda *args, **kwargs: self.default_launches.append(
            args[0]
        )
        spec.loader.exec_module(module)
        self.compiled.append(config)
        if allow_print:
            self.printed.append(config)
        self._compile_cache[config] = module.demo
        self._cache_path_map[config] = str(path)
        return module.demo

    def set_config(self, config: Config) -> None:
        self._run = self.compile_config(config)
        self._config = config

    def select(self, config: Config) -> Callable[..., object]:
        """Install *config* as autotune() does for a selected config."""
        self._run = TritonBackend().fit_on_first_launch(
            cast("Any", self), config, self.compile_config(config)
        )
        self._config = config
        return self._run


def _buffers(config: Config) -> int:
    return (config.num_stages - 1) * 64 * 1024


def _stages(launches: Sequence[tuple[str, Config]]) -> list[int]:
    return [config.num_stages for _, config in launches]


@skipUnlessBackends(["triton"])
class TestFitOnFirstLaunch:
    def test_lowers_num_stages_until_the_kernel_fits(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        x = torch.zeros(4)
        assert bound_kernel.select(Config(block_sizes=[64], num_stages=8))(x) is x
        # 7 * 64 KiB of buffers does not fit; 4 * 64 KiB does, 5 * 64 KiB does not.
        assert _stages(bound_kernel.attempts) == [8, 5]
        assert _stages(bound_kernel.launched) == [5]
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=5)
        # Later calls run the fitted config directly, with the default launcher.
        assert bound_kernel._run is bound_kernel._compile_cache[bound_kernel._config]
        bound_kernel._run(x)
        assert [
            kernel.config.num_stages for kernel in bound_kernel.default_launches
        ] == [5]

    def test_steps_down_when_the_estimate_is_too_big(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 300_000, lambda config: 150_000 + _buffers(config)
        )
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))(torch.zeros(4))
        assert _stages(bound_kernel.attempts) == [8, 4, 3]
        assert bound_kernel._config is not None
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=3)

    @pytest.mark.parametrize(
        ("num_stages", "range_num_stages", "expected"),
        [
            (8, [0, 8], (3, [0, 3])),
            # A per-loop count above num_stages sizes that loop's buffers.
            (1, [0, 4], (1, [0, 3])),
            (2, [0, 8], (2, [0, 3])),
        ],
    )
    def test_lowers_per_loop_stages_with_num_stages(
        self,
        tmp_path: Path,
        num_stages: int,
        range_num_stages: list[int],
        expected: tuple[int, list[int]],
    ) -> None:
        def usage(config: Config) -> int:
            per_loop = cast("list[int]", config.config["range_num_stages"])
            return (max([config.num_stages, *per_loop]) - 1) * 64 * 1024

        bound_kernel = _FakeBoundKernel(tmp_path, 150_000, usage)
        bound_kernel.select(
            Config(
                block_sizes=[64],
                num_stages=num_stages,
                range_num_stages=range_num_stages,
            )
        )(torch.zeros(4))
        config = bound_kernel._config
        assert config is not None
        assert (config.num_stages, config.config["range_num_stages"]) == expected
        assert config.block_sizes == [64]

    def test_threads_halve_num_warps(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 128, lambda config: config.num_warps * 32, resource="threads"
        )
        bound_kernel.select(Config(block_sizes=[64], num_warps=16))(torch.zeros(4))
        assert [config.num_warps for _, config in bound_kernel.attempts] == [16, 8, 4]
        assert bound_kernel.launched_kernels == [Config(block_sizes=[64], num_warps=4)]

    def test_warp_specialized_loops_keep_four_warps(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 64, lambda config: config.num_warps * 32, resource="threads"
        )
        run = bound_kernel.select(
            Config(block_sizes=[64], num_warps=4, range_warp_specializes=[True])
        )
        with pytest.raises(OutOfResources):
            run(torch.zeros(4))
        assert bound_kernel.compiled == [bound_kernel._config]

    @pytest.mark.parametrize("resource", ["shared memory", "tensor memory"])
    def test_raises_the_original_error_when_nothing_smaller_fits(
        self, tmp_path: Path, resource: str
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 1000, lambda config: 5000 * config.num_stages, resource=resource
        )
        config = Config(
            block_sizes=[64], num_stages=1 if resource == "shared memory" else 4
        )
        with pytest.raises(OutOfResources) as error:
            bound_kernel.select(config)(torch.zeros(4))
        assert error.value.required == 5000 * config.num_stages
        assert bound_kernel.compiled == [config]
        assert bound_kernel._config == config

    def test_gives_up_after_the_attempt_limit(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 0, lambda config: 1, resource="threads"
        )
        with pytest.raises(OutOfResources):
            bound_kernel.select(Config(block_sizes=[64], num_warps=2**20))(
                torch.zeros(4)
            )
        assert len(bound_kernel.attempts) == 17

    @pytest.mark.parametrize(
        "failure",
        [
            "stages",
            "warps",
            "attempt_limit",
            "host_code",
            "codegen",
            "launch",
        ],
    )
    def test_failed_fit_restores_original_config_on_repeated_calls(
        self, tmp_path: Path, failure: str
    ) -> None:
        threads = failure in ("warps", "attempt_limit", "host_code")
        config = Config(
            block_sizes=[64],
            num_warps=2**20 if failure == "attempt_limit" else 16,
            num_stages=8,
        )
        usage = (
            (lambda config: config.num_warps * 32)
            if threads
            else (lambda config: config.num_stages * 1000)
        )
        bound_kernel = _FakeBoundKernel(
            tmp_path,
            192
            if failure == "host_code"
            else 3500
            if failure in ("codegen", "launch")
            else 0,
            usage,
            resource="threads" if threads else "shared memory",
            grid=lambda candidate: (
                "(2,)" if failure == "host_code" and candidate.num_warps < 8 else "(1,)"
            ),
        )
        run = bound_kernel.select(config)
        compile_config = bound_kernel.compile_config

        def compile_candidate(
            candidate: Config, *, allow_print: bool = True
        ) -> Callable[..., object]:
            if failure == "codegen" and candidate.num_stages == 3:
                raise RuntimeError("later codegen failed")
            return compile_config(candidate, allow_print=allow_print)

        bound_kernel.compile_config = compile_candidate  # type: ignore[method-assign]
        if failure == "launch":
            bound_kernel.launch_error = lambda candidate: (
                RuntimeError("later launch failed")
                if candidate.num_stages == 3
                else None
            )
        for call in range(3):
            with pytest.raises(OutOfResources) as error:
                run(torch.zeros(4))
            assert error.value.required == usage(config)
            assert bound_kernel._config == config
            assert run.current[0] == config
            assert run.current[1] is run.compiled
            assert bound_kernel.launched == []
            if call == 0:
                assert len(bound_kernel.attempts) > 1
                compiled = list(bound_kernel.compiled)
                attempts = len(bound_kernel.attempts)
            else:
                assert bound_kernel.compiled == compiled
                assert len(bound_kernel.attempts) == attempts + call
                assert bound_kernel.attempts[-1][1] == config

    def test_interrupted_fit_restores_original_config_after_partial_progress(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 3500, lambda config: config.num_stages * 1000
        )
        config = Config(block_sizes=[64], num_stages=8)
        run = bound_kernel.select(config)
        compile_config = bound_kernel.compile_config
        before_launch, after_launch = Mock(), Mock()
        bound_kernel.before_launch = before_launch
        bound_kernel.after_launch = after_launch

        def interrupt(
            candidate: Config, *, allow_print: bool = True
        ) -> Callable[..., object]:
            if candidate.num_stages == 3:
                raise KeyboardInterrupt
            return compile_config(candidate, allow_print=allow_print)

        bound_kernel.compile_config = interrupt  # type: ignore[method-assign]
        with pytest.raises(KeyboardInterrupt):
            run(torch.zeros(4))
        bound_kernel.compile_config = compile_config  # type: ignore[method-assign]
        run(torch.zeros(4))
        assert _stages(bound_kernel.attempts) == [8, 4, 8, 4, 3]
        assert _stages(bound_kernel.launched) == [3]
        assert (before_launch.call_count, after_launch.call_count) == (2, 1)

    def test_concurrent_failed_fit_preserves_original_errors(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path,
            3500,
            lambda candidate: (
                8000
                if candidate.num_stages == 8
                else candidate.num_stages * 1000 + 3000
            ),
        )
        config = Config(block_sizes=[64], num_stages=8)
        run = bound_kernel.select(config)
        candidate_started = threading.Event()
        finish_candidate = threading.Event()
        first_thread: int | None = None

        def block_first_candidate(candidate: Config) -> Exception | None:
            if candidate.num_stages == 4 and threading.get_ident() == first_thread:
                candidate_started.set()
                assert finish_candidate.wait(timeout=10)
            return None

        bound_kernel.launch_error = block_first_candidate

        def call(first: bool) -> int:
            nonlocal first_thread
            if first:
                first_thread = threading.get_ident()
            with pytest.raises(OutOfResources) as error:
                run(torch.zeros(4))
            return error.value.required

        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(call, True)
            try:
                assert candidate_started.wait(timeout=10)
                second = pool.submit(call, False)
                assert second.result(timeout=10) == 8000
            finally:
                finish_candidate.set()
            assert first.result(timeout=10) == 8000
        with pytest.raises(OutOfResources) as error:
            run(torch.zeros(4))
        assert error.value.required == 8000
        assert bound_kernel._config == config
        assert bound_kernel.launched == []

    def test_other_errors_raise_the_original_error(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))

        def fail(config: Config, *, allow_print: bool = True) -> Callable[..., object]:
            raise RuntimeError("codegen failed")

        bound_kernel.compile_config = fail  # type: ignore[method-assign]
        with pytest.raises(OutOfResources) as error:
            run(torch.zeros(4))
        assert error.value.required == 7 * 64 * 1024

    def test_a_smaller_config_that_fails_otherwise_raises_the_original_error(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        bound_kernel.launch_error = lambda config: (
            RuntimeError("compile failed") if config.num_stages == 5 else None
        )
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        for _ in range(2):
            with pytest.raises(OutOfResources) as error:
                run(torch.zeros(4))
            assert error.value.required == 7 * 64 * 1024
        # The second call does not try the failing config again.
        assert _stages(bound_kernel.attempts) == [8, 5, 8]

    def test_distributed_kernels_are_not_fitted(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        bound_kernel.env.process_group_name = "pg"
        config = Config(block_sizes=[64], num_stages=8)
        assert bound_kernel.select(config) is bound_kernel._compile_cache[config]

    def test_host_code_runs_once_per_call(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        before_launch, after_launch = Mock(), Mock()
        bound_kernel.before_launch = before_launch
        bound_kernel.after_launch = after_launch
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        run(torch.zeros(4))
        assert _stages(bound_kernel.launched) == [5]
        assert (before_launch.call_count, after_launch.call_count) == (1, 1)
        assert bound_kernel._run is not None
        bound_kernel._run(torch.zeros(4))
        assert (before_launch.call_count, after_launch.call_count) == (2, 2)

    def test_launches_its_own_kernel_when_the_host_code_matches(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))(torch.zeros(4))
        assert bound_kernel.launched_kernels == [Config(block_sizes=[64], num_stages=5)]

    def test_launches_the_running_kernel_with_fewer_stages_when_the_host_code_differs(
        self, tmp_path: Path
    ) -> None:
        # Code generated later can compute other arguments, for example after it
        # guarded on an input size, so the running kernel is launched instead.
        bound_kernel = _FakeBoundKernel(
            tmp_path, 300_000, _buffers, grid=lambda config: f"({config.num_stages},)"
        )
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))(torch.zeros(4))
        assert bound_kernel.launched_kernels == [Config(block_sizes=[64], num_stages=8)]
        assert _stages(bound_kernel.launched) == [5]
        # Later calls run the fitted config's own code.
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=5)

    @pytest.mark.parametrize(
        ("resource", "config"),
        [
            ("threads", Config(block_sizes=[64], num_warps=16)),
            (
                "shared memory",
                Config(block_sizes=[64], num_stages=1, range_num_stages=[0, 8]),
            ),
        ],
        ids=["num_warps", "per_loop_stages"],
    )
    def test_differing_host_code_allows_only_fewer_stages(
        self, tmp_path: Path, resource: str, config: Config
    ) -> None:
        def usage(config: Config) -> int:
            per_loop = cast("list[int]", config.config.get("range_num_stages", []))
            return config.num_warps * 32 + max([config.num_stages, *per_loop]) * 1000

        bound_kernel = _FakeBoundKernel(
            tmp_path,
            usage(config) - 1,
            usage,
            resource=resource,
            grid=lambda config: (
                f"({config.num_warps}, {config.config.get('range_num_stages')})"
            ),
        )
        with pytest.raises(OutOfResources):
            bound_kernel.select(config)(torch.zeros(4))
        assert bound_kernel.launched == []

    @pytest.mark.parametrize(
        ("launch_kwargs", "fitted"),
        [
            (", launch_cooperative_grid=True", False),
            (", _minimum_resident_programs=2", False),
            (", _persistent_state_specs=((x, 1, torch.float32, False),)", False),
            (", _remote_copy_scratch_specs=((x, 1),)", False),
            (
                (
                    ", _remote_copy_signal_dst=x,"
                    " _remote_copy_signal_slots_per_program=1,"
                    " _remote_copy_process_group_name='pg'"
                ),
                False,
            ),
            (
                (
                    ", _remote_barrier_signal_slots_per_program=1,"
                    " _remote_barrier_process_group_name='pg'"
                ),
                False,
            ),
            # Arguments at their defaults set nothing up.
            (", launch_cooperative_grid=False, _minimum_resident_programs=0", True),
        ],
        ids=[
            "cooperative",
            "residency",
            "persistent_state",
            "scratch",
            "remote_copy",
            "remote_barrier",
            "defaults",
        ],
    )
    def test_launches_that_set_up_launcher_state_are_not_fitted(
        self, tmp_path: Path, launch_kwargs: str, fitted: bool
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 300_000, _buffers, launch_kwargs=launch_kwargs
        )
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        state = Mock(return_value=torch.zeros(1))
        with (
            patch.object(launcher_module, "_validate_resident_program_capacity"),
            patch.object(launcher_module, "_get_persistent_state", state),
            patch.object(launcher_module, "_get_remote_copy_scratch", state),
            patch.object(launcher_module, "_get_remote_copy_signal", state),
            patch.object(launcher_module, "_get_remote_barrier_signal", state),
        ):
            if fitted:
                run(torch.zeros(4))
                assert _stages(bound_kernel.launched) == [5]
            else:
                with pytest.raises(OutOfResources):
                    run(torch.zeros(4))
                assert bound_kernel.compiled == [bound_kernel._config]

    def test_keeps_a_config_that_fits(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 10**9, _buffers)
        config = Config(block_sizes=[64], num_stages=8)
        with caplog.at_level(logging.WARNING, logger="helion"):
            bound_kernel.select(config)(torch.zeros(4))
        assert bound_kernel._config is config
        assert bound_kernel._run is bound_kernel._compile_cache[config]
        assert bound_kernel.compiled == [config]
        assert caplog.records == []

    def test_later_kernels_of_the_call_use_the_fitted_config(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path,
            300_000,
            _buffers,
            kernel_names=("_helion_demo", "_helion_demo_1"),
        )
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))(torch.zeros(4))
        assert [
            (name, config.num_stages) for name, config in bound_kernel.launched
        ] == [("_helion_demo", 5), ("_helion_demo_1", 5)]

    def test_empty_call_keeps_first_launch_fitting_armed(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        run(torch.zeros(0))
        assert bound_kernel._run is run
        assert bound_kernel.attempts == []
        run(torch.zeros(4))
        assert _stages(bound_kernel.launched) == [5]
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=5)

    def test_empty_call_does_not_publish_a_pending_candidate(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 300_000, lambda config: 150_000 + _buffers(config)
        )
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        started, finish = threading.Event(), threading.Event()

        def pause_candidate(config: Config) -> None:
            if config.num_stages == 4:
                started.set()
                assert finish.wait(10)

        bound_kernel.launch_error = pause_candidate
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(run, torch.zeros(4))
            try:
                assert started.wait(10)
                run(torch.zeros(0))
                assert bound_kernel._run is run
            finally:
                finish.set()
            first.result(timeout=10)
        assert _stages(bound_kernel.launched) == [3]
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=3)

    def test_interrupted_caller_does_not_publish_another_calls_reset_state(
        self, tmp_path: Path
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            tmp_path, 3500, lambda config: config.num_stages * 1000
        )
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        started, finish = threading.Event(), threading.Event()
        first_thread: int | None = None
        interrupted = False
        before, after = Mock(), Mock()
        bound_kernel.before_launch, bound_kernel.after_launch = before, after

        def pause_or_interrupt(config: Config) -> None:
            nonlocal interrupted
            if config.num_stages != 3:
                return
            if threading.get_ident() == first_thread:
                started.set()
                assert finish.wait(10)
            elif not interrupted:
                interrupted = True
                raise KeyboardInterrupt

        def first_call() -> None:
            nonlocal first_thread
            first_thread = threading.get_ident()
            run(torch.zeros(4))

        bound_kernel.launch_error = pause_or_interrupt
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(first_call)
            try:
                assert started.wait(10)
                with pytest.raises(KeyboardInterrupt):
                    run(torch.zeros(4))
            finally:
                finish.set()
            first.result(timeout=10)
        assert bound_kernel._run is run
        bound_kernel._run(torch.zeros(4))
        assert _stages(bound_kernel.launched) == [3, 3]
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=3)
        assert bound_kernel.default_launches == []
        assert (before.call_count, after.call_count) == (3, 2)

    def test_concurrent_first_calls_fit_once(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        barrier = threading.Barrier(8)

        def call() -> None:
            barrier.wait()
            run(torch.zeros(4))

        with ThreadPoolExecutor(8) as pool:
            for future in [pool.submit(call) for _ in range(8)]:
                future.result()
        assert [config.num_stages for config in bound_kernel.compiled] == [8, 5]
        assert _stages(bound_kernel.launched) == [5] * 8

    def test_an_interrupted_fit_can_be_retried(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        compile_config = bound_kernel.compile_config

        def interrupt(
            config: Config, *, allow_print: bool = True
        ) -> Callable[..., object]:
            raise KeyboardInterrupt

        bound_kernel.compile_config = interrupt  # type: ignore[method-assign]
        with pytest.raises(KeyboardInterrupt):
            run(torch.zeros(4))
        bound_kernel.compile_config = compile_config  # type: ignore[method-assign]
        run(torch.zeros(4))
        assert _stages(bound_kernel.launched) == [5]

    def test_warns_once_per_kernel_with_the_fields_it_changed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        kernel = _FakeKernel()
        with (
            patch.object(TritonBackend, "_fitted_kernels", weakref.WeakSet()),
            caplog.at_level(logging.WARNING, logger="helion"),
        ):
            # Two shapes of one kernel, then another kernel with the same name.
            for each_kernel in (kernel, kernel, _FakeKernel()):
                bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
                bound_kernel.kernel = each_kernel
                bound_kernel.select(Config(block_sizes=[64], num_stages=8))(
                    torch.zeros(4)
                )
        assert len(caplog.records) == 2
        assert (
            "demo: the selected config exceeds this device's shared memory limit "
            "(458752 > 300000), so it runs with num_stages=5 instead."
        ) in caplog.text

    @pytest.mark.parametrize("print_output_code", [True, False])
    def test_prints_the_fitted_config_with_the_output_code(
        self, tmp_path: Path, print_output_code: bool
    ) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        bound_kernel.settings.print_output_code = print_output_code
        config = Config(block_sizes=[64], num_stages=8)
        bound_kernel.select(config)(torch.zeros(4))
        # The fitted config compiles silently during the fit; when output code is
        # printed, set_config compiles it again to print it.
        fitted = Config(block_sizes=[64], num_stages=5)
        expected = [config, fitted] if print_output_code else [config]
        assert bound_kernel.printed == expected

    def test_tracing_calls_the_compiled_host_function(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 10**9, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            run(torch.zeros(4))
        # The host code used its own launcher, and the config is still unchecked.
        assert len(bound_kernel.default_launches) == 1
        assert bound_kernel.launched == []
        assert bound_kernel._run is run


@skipUnlessBackends(["triton"])
class TestFitConfigForRendering:
    def test_fits_flagged_configs_with_the_example_args(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        bound_kernel._config_from_other_device = True
        TritonBackend().fit_config_for_rendering(
            cast("Any", bound_kernel), (torch.zeros(4),)
        )
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=5)
        # Nothing was launched, and the fitted config is no longer wrapped.
        assert bound_kernel.launched == []
        assert bound_kernel._run is bound_kernel._compile_cache[bound_kernel._config]

    def test_checks_a_flagged_config_that_fits_once(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 10**9, _buffers)
        before_launch = Mock()
        bound_kernel.before_launch = before_launch
        config = Config(block_sizes=[64], num_stages=8)
        run = bound_kernel.select(config)
        bound_kernel._config_from_other_device = True
        for _ in range(2):
            TritonBackend().fit_config_for_rendering(
                cast("Any", bound_kernel), (torch.zeros(4),)
            )
        assert before_launch.call_count == 1
        assert (bound_kernel._config, bound_kernel._run) == (config, run)

    def test_leaves_unflagged_configs_alone(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        TritonBackend().fit_config_for_rendering(
            cast("Any", bound_kernel), (torch.zeros(4),)
        )
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=8)
        assert bound_kernel._run is run

    def test_keeps_the_config_when_nothing_smaller_fits(self, tmp_path: Path) -> None:
        bound_kernel = _FakeBoundKernel(tmp_path, 1000, lambda config: 5000)
        config = Config(block_sizes=[64], num_stages=8)
        bound_kernel.select(config)
        bound_kernel._config_from_other_device = True
        TritonBackend().fit_config_for_rendering(
            cast("Any", bound_kernel), (torch.zeros(4),)
        )
        assert bound_kernel._config is config

    @pytest.mark.parametrize(
        ("file_name", "expected"),
        [
            ("_helion_aot_demo_cuda_sm90.py", True),
            ("_helion_aot_demo_cuda_sm90__policy_abc.py", True),
            ("_helion_aot_demo_cuda_sm121.py", False),
            ("_helion_aot_demo_cuda_sm121__policy_abc.py", False),
            ("heuristic_demo.py", False),
            (None, False),
        ],
    )
    def test_heuristic_is_for_other_device(
        self, tmp_path: Path, file_name: str | None, expected: bool
    ) -> None:
        cache = object.__new__(AOTAutotuneCache)
        cache._find_heuristic_file = lambda: (
            None if file_name is None else tmp_path / file_name
        )
        gb10 = HardwareInfo(
            device_kind="cuda",
            hardware_name="NVIDIA GB10",
            runtime_version="13.0",
            compute_capability="sm121",
        )
        with patch.object(aot_cache_module, "get_hardware_info", return_value=gb10):
            assert cache._heuristic_is_for_other_device() is expected

    @pytest.mark.parametrize(
        ("mode", "other_device", "expected"),
        [
            ("evaluate", True, True),
            ("evaluate", False, False),
            ("compile", True, False),
        ],
    )
    def test_get_flags_configs_from_another_device(
        self, mode: str, other_device: bool, expected: bool
    ) -> None:
        config = Config(block_sizes=[64])
        cache = object.__new__(AOTAutotuneCache)
        cache.mode = mode
        cache.args = ()
        cache.kernel = SimpleNamespace(_config_from_other_device=False)
        cache._maybe_run_compile = lambda: None
        cache._get_heuristic_config = lambda args: config
        cache._heuristic_is_for_other_device = lambda: other_device
        assert cache.get() is config
        assert cache.kernel._config_from_other_device is expected


# Each stage of [64, 128, 128] bf16 tiles buffers 48 KiB, so 8 stages need far
# more shared memory per block than any current GPU has.
_OVERSIZED_MATMUL_CONFIG: dict[str, object] = {
    "block_sizes": [64, 128, 128],
    "num_stages": 8,
    "num_warps": 4,
}


@skipUnlessBackends(["triton"])
class TestFitOnFirstLaunchGPU:
    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16)
        return x, y

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT heuristic config selection")
    @skipIfTileIR("checks Triton's num_stages and shared memory accounting")
    def test_older_gpu_heuristic_config_is_fitted_at_first_launch(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with (
            _heuristic_for_older_gpu(
                tmp_path, {"aot_fit_matmul": _OVERSIZED_MATMUL_CONFIG}
            ),
            patch.dict(os.environ, {"HELION_PRINT_OUTPUT_CODE": "1"}),
            caplog.at_level(logging.WARNING, logger="helion"),
        ):
            kernel = _make_aot_fit_matmul()
            x, y = self._inputs()
            out = kernel(x, y)
            bound_kernel = kernel.bind((x, y))
            config = bound_kernel._config
            assert config is not None
            # Later calls run the fitted config directly.
            assert bound_kernel._run is bound_kernel._compile_cache[config]
            torch.testing.assert_close(kernel(x, y), x @ y, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(out, x @ y, atol=1e-1, rtol=1e-2)
        # Fewer stages are enough on every GPU: the tiles must stay as tuned.
        assert config.block_sizes == [64, 128, 128]
        assert 2 <= config.num_stages < 8
        assert "exceeds this device's shared memory limit" in caplog.text
        # The code that runs is printed, as for any other config.
        assert f"num_stages={config.num_stages})" in capsys.readouterr().err

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT heuristic config selection")
    @skipIfTileIR("checks Triton's num_stages and shared memory accounting")
    def test_host_code_runs_once_per_call(self, tmp_path: Path) -> None:
        with _heuristic_for_older_gpu(
            tmp_path, {"aot_fit_count_calls": _OVERSIZED_MATMUL_CONFIG}
        ):
            kernel = _make_aot_fit_count_calls()
            x, y = self._inputs()
            calls = torch.zeros(1, device=DEVICE)
            out = kernel(x, y, calls)
            assert calls.item() == 1
            kernel(x, y, calls)
            assert calls.item() == 2
        torch.testing.assert_close(out, x @ y, atol=1e-1, rtol=1e-2)

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT heuristic config selection")
    @skipIfTileIR("checks Triton's num_stages and shared memory accounting")
    @pytest.mark.parametrize("autotune_first", [False, True])
    def test_a_given_config_is_not_fitted(self, autotune_first: bool) -> None:
        kernel = helion.kernel(
            config=helion.Config(**_OVERSIZED_MATMUL_CONFIG),  # pyrefly: ignore[bad-argument-type]
        )(_make_aot_fit_matmul().fn)
        args = self._inputs()
        if autotune_first:
            kernel.autotune(args, force=False)
        with pytest.raises(OutOfResources):
            kernel(*args)

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT heuristic config selection")
    @skipIfTileIR("checks Triton's num_stages and shared memory accounting")
    def test_fits_to_a_gpu_with_less_shared_memory(self, tmp_path: Path) -> None:
        # Three stages fit every GPU in CI, but not one with 64 KiB per block.
        config = {**_OVERSIZED_MATMUL_CONFIG, "num_stages": 3}
        with (
            _heuristic_for_older_gpu(tmp_path, {"aot_fit_matmul_small_gpu": config}),
            _shared_memory_limit(64 * 1024),
        ):
            kernel = _make_aot_fit_matmul_small_gpu()
            x, y = self._inputs()
            out = kernel(x, y)
            fitted = kernel.bind((x, y))._config
        torch.testing.assert_close(out, x @ y, atol=1e-1, rtol=1e-2)
        assert fitted is not None
        assert fitted.num_stages < 3

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT heuristic config selection")
    @skipIfTileIR("checks Triton's num_stages and shared memory accounting")
    def test_fits_when_the_code_generated_later_differs(self, tmp_path: Path) -> None:
        # Generating this kernel's code the first time guards on scale.size(0),
        # so code generated afterwards, for a smaller config, takes other
        # arguments; the first call then runs its kernel with fewer stages.
        config = {**_OVERSIZED_MATMUL_CONFIG, "num_stages": 3}
        with (
            _heuristic_for_older_gpu(tmp_path, {"aot_fit_scaled_matmul": config}),
            _shared_memory_limit(64 * 1024),
        ):
            kernel = _make_aot_fit_scaled_matmul()
            x, y = self._inputs()
            for scale in (
                torch.full((1, 1), 2.0, device=DEVICE),
                torch.rand(x.size(0), 1, device=DEVICE) + 0.5,
            ):
                for _ in range(2):
                    torch.testing.assert_close(
                        kernel(x, y, scale),
                        (scale * (x.float() @ y.float())).to(x.dtype),
                        atol=2e-1,
                        rtol=2e-2,
                    )

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode does not autotune")
    @skipIfTileIR("checks Triton's shared memory accounting")
    def test_autotuner_skips_configs_too_big_for_a_smaller_gpu(self) -> None:
        too_big = helion.Config(block_sizes=[64, 128, 128], num_stages=4, num_warps=4)
        fits = helion.Config(block_sizes=[32, 64, 64], num_stages=2, num_warps=4)
        with _shared_memory_limit(48 * 1024):
            kernel = helion.kernel(configs=[too_big, fits])(
                _make_aot_fit_matmul_small_gpu().fn
            )
            x, y = self._inputs()
            out = kernel(x, y)
            config = kernel.bind((x, y))._config
        torch.testing.assert_close(out, x @ y, atol=1e-1, rtol=1e-2)
        assert config == fits

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT heuristic config selection")
    @skipIfTileIR("checks Triton's num_stages and shared memory accounting")
    def test_torch_compile_renders_the_fitted_config(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        torch._dynamo.reset()
        with (
            _heuristic_for_older_gpu(
                tmp_path, {"aot_fit_matmul": _OVERSIZED_MATMUL_CONFIG}
            ),
            fresh_cache(),
            caplog.at_level(logging.WARNING, logger="helion"),
        ):
            kernel = _make_aot_fit_matmul()
            x, y = self._inputs()

            @torch.compile(fullgraph=True)
            def double_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return kernel(a, b) * 2

            out = double_matmul(x, y)
        torch.testing.assert_close(out, 2 * (x @ y), atol=2e-1, rtol=1e-2)
        assert "exceeds this device's shared memory limit" in caplog.text


@onlyBackends(["triton", "cute"])
class TestShapeKey:
    """Tests for ShapeKey class."""

    def test_to_dict_and_back(self) -> None:
        hardware = HardwareInfo(
            device_kind="cuda",
            hardware_name="RTX4090",
            runtime_version="12.4",
            compute_capability="sm89",
        )
        key = ShapeKey(
            kernel_name="test_kernel",
            specialization_key=(1024, 2048, "float32"),
            hardware_id=hardware.hardware_id,
        )
        d = key.to_dict()
        restored = ShapeKey.from_dict(d)
        assert restored.kernel_name == key.kernel_name
        assert restored.hardware_id == key.hardware_id

    def test_stable_hash(self) -> None:
        key1 = ShapeKey("k", (1, 2, 3), "hw")
        key2 = ShapeKey("k", (1, 2, 3), "hw")
        assert key1.stable_hash() == key2.stable_hash()

        key3 = ShapeKey("k", (1, 2, 4), "hw")
        assert key1.stable_hash() != key3.stable_hash()

    def test_search_policy_is_structural_and_legacy_serialization_is_unchanged(
        self,
    ) -> None:
        legacy = ShapeKey("k", (1, 2, 3), "hw")
        policy = ShapeKey("k", (1, 2, 3), "hw", search_policy_hash="full-v1")

        assert "search_policy_hash" not in legacy.to_dict()
        assert legacy.stable_hash() == "7736dd40a5cb84bd"
        assert policy.to_dict()["search_policy_hash"] == "full-v1"
        assert legacy.stable_hash() != policy.stable_hash()
        assert ShapeKey.from_dict(legacy.to_dict()).search_policy_hash == ""

    def test_aot_shape_key_includes_cute_flash_search_policy(self) -> None:
        cache = object.__new__(AOTAutotuneCache)
        cache.autotuner = SimpleNamespace()
        cache.args = ()
        cache.hardware_id = "hw"
        specialization_key = Mock(return_value=("shape",))
        cache.kernel = SimpleNamespace(
            config_spec=SimpleNamespace(cute_flash_search_enabled=True),
            kernel=SimpleNamespace(
                name="kernel",
                specialization_key=specialization_key,
            ),
        )

        with patch(
            "helion.autotuner.local_cache._cute_flash_search_policy_hash",
            return_value="full-v1",
        ) as policy_hash:
            key = cache._create_shape_key()

        assert key.search_policy_hash == "full-v1"
        specialization_key.assert_called_once_with(())
        policy_hash.assert_called_once_with(
            cache.autotuner,
            cute_flash_search_enabled=True,
        )

    def test_aot_shape_key_preserves_uncacheable_instance_identity(self) -> None:
        cache = object.__new__(AOTAutotuneCache)
        first_autotuner = SimpleNamespace(_search_policy_cacheable=True)
        cache.autotuner = first_autotuner
        cache.args = ()
        cache.hardware_id = "hw"
        cache.kernel = SimpleNamespace(
            config_spec=SimpleNamespace(cute_flash_search_enabled=True),
            kernel=SimpleNamespace(
                name="kernel",
                specialization_key=Mock(return_value=("shape",)),
            ),
        )
        nonces = iter(("random-nonce-1", "random-nonce-2", "random-nonce-3"))

        def uncacheable_policy(autotuner, *, cute_flash_search_enabled):
            assert cute_flash_search_enabled
            autotuner._search_policy_cacheable = False
            return next(nonces)

        with patch(
            "helion.autotuner.local_cache._cute_flash_search_policy_hash",
            side_effect=uncacheable_policy,
        ):
            first = cache._create_shape_key()
        cache.autotuner = SimpleNamespace(_search_policy_cacheable=True)
        with patch(
            "helion.autotuner.local_cache._cute_flash_search_policy_hash",
            side_effect=uncacheable_policy,
        ):
            second = cache._create_shape_key()

        assert first.search_policy_hash == "random-nonce-1"
        assert second.search_policy_hash == "random-nonce-2"
        assert first != second

    def test_uncacheable_policy_still_uses_aot_evaluate_selection(self) -> None:
        selected = Config(block_sizes=[8])
        cache = object.__new__(AOTAutotuneCache)
        cache.mode = "evaluate"
        cache._verbose = False
        cache.autotuner = SimpleNamespace(
            _search_policy_cacheable=False,
            log=Mock(),
        )
        cache.args = ()
        cache.get = Mock(return_value=selected)  # type: ignore[method-assign]
        cache._maybe_run_input_fn_workflows = Mock()  # type: ignore[method-assign]
        cache._run_autotune_trials = Mock()  # type: ignore[method-assign]

        result = cache.autotune()

        assert result == selected
        cache.get.assert_called_once_with()
        cache._run_autotune_trials.assert_not_called()

    def test_uncacheable_policy_still_persists_aot_collection(self) -> None:
        selected = Config(block_sizes=[8])
        cache = object.__new__(AOTAutotuneCache)
        cache.mode = "collect"
        cache.autotuner = SimpleNamespace(
            _search_policy_cacheable=False,
            log=Mock(),
        )
        cache.args = ()
        cache.get = Mock()  # type: ignore[method-assign]
        cache.put = Mock()  # type: ignore[method-assign]
        cache._maybe_run_input_fn_workflows = Mock()  # type: ignore[method-assign]
        cache._run_autotune_trials = Mock(  # type: ignore[method-assign]
            return_value=selected
        )

        result = cache.autotune()

        assert result == selected
        cache.get.assert_not_called()
        cache.put.assert_called_once_with(selected)


@onlyBackends(["triton", "cute"])
class TestCodeSerialization:
    """Tests for specialization keys containing function code objects (e.g. callable kernel args)."""

    def test_code_round_trips_through_serialize_value(self) -> None:
        def fn(v):
            return v * 2

        serialized = _serialize_value(fn.__code__)
        deserialized = _deserialize_value(serialized)
        assert deserialized == (
            fn.__code__.co_code,
            fn.__code__.co_consts,
            fn.__code__.co_names,
        )

    def test_stable_hash_same_for_rename(self) -> None:
        def double(v):
            return v * 2

        def double_renamed(v):
            return v * 2

        h1 = ShapeKey("k", (double.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (double_renamed.__code__,), "hw").stable_hash()
        assert h1 == h2

    def test_stable_hash_differs_for_behavior_change(self) -> None:
        def double(v):
            return v * 2

        def triple(v):
            return v * 3

        h1 = ShapeKey("k", (double.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (triple.__code__,), "hw").stable_hash()
        assert h1 != h2

    def test_stable_hash_differs_for_nested_lambda_behavior_change(self) -> None:
        def with_nested_lambda_2():
            return lambda v: v * 2

        def with_nested_lambda_3():
            return lambda v: v * 3

        h1 = ShapeKey("k", (with_nested_lambda_2.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (with_nested_lambda_3.__code__,), "hw").stable_hash()
        assert h1 != h2

    def test_stable_hash_for_conames(self) -> None:
        def with_sin(v):
            return v.sin()

        def with_cos(v):
            return v.cos()

        h1 = ShapeKey("k", (with_sin.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (with_cos.__code__,), "hw").stable_hash()
        assert h1 != h2

    def test_code_serializes_complex_and_ellipsis_consts(self) -> None:
        def with_complex(v):
            return v * 1j

        def with_ellipsis(v):  # noqa: FURB118
            return v[...]

        for fn in (with_complex, with_ellipsis):
            serialized = _serialize_value(fn.__code__)
            json.dumps(serialized)  # must not raise
            assert _deserialize_value(serialized) == (
                fn.__code__.co_code,
                fn.__code__.co_consts,
                fn.__code__.co_names,
            )

    def test_stable_hash_survives_save_load_round_trip(self) -> None:
        def fn(v):
            return v * 2

        key = ShapeKey("k", (fn.__code__,), "hw")
        original_hash = key.stable_hash()

        # Simulate a JSON save/load cycle
        reloaded = ShapeKey.from_dict(json.loads(json.dumps(key.to_dict())))
        assert reloaded.stable_hash() == original_hash
        reloaded_again = ShapeKey.from_dict(json.loads(json.dumps(reloaded.to_dict())))
        assert reloaded_again.stable_hash() == original_hash


@onlyBackends(["triton", "cute"])
class TestSerializeTuple:
    """Tests for tuple serialization."""

    def test_simple_tuple(self) -> None:
        t = (1, 2, 3)
        serialized = _serialize_tuple(t)
        deserialized = _deserialize_tuple(serialized)
        assert deserialized == t

    def test_nested_tuple(self) -> None:
        t = (1, (2, 3), 4)
        serialized = _serialize_tuple(t)
        deserialized = _deserialize_tuple(serialized)
        assert deserialized == t


@onlyBackends(["triton", "cute"])
class TestConfigSubsetSelection:
    """Tests for config subset selection algorithm."""

    def test_single_config_optimal(self) -> None:
        # Create data where one config is optimal for all shapes
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            timings=np.array(
                [
                    [1.0, 2.0],  # Config 0 is best for shape 0
                    [1.0, 2.0],  # Config 0 is best for shape 1
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(goal_type="max_slowdown", threshold=1.1)
        selected, stats = select_config_subset(data, target)

        assert len(selected) == 1
        assert selected[0] == 0  # Config 0 should be selected

    def test_multiple_configs_needed(self) -> None:
        # Create data where different configs are optimal for different shapes
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            timings=np.array(
                [
                    [1.0, 10.0],  # Config 0 is best for shape 0
                    [10.0, 1.0],  # Config 1 is best for shape 1
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(goal_type="max_slowdown", threshold=1.1)
        selected, stats = select_config_subset(data, target)

        # Both configs needed to meet performance goal
        assert len(selected) == 2


@onlyBackends(["triton", "cute"])
class TestGetAOTMode:
    """Tests for get_aot_mode."""

    def test_default_mode(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            if "HELION_AOT_MODE" in os.environ:
                del os.environ["HELION_AOT_MODE"]
            # Default mode is "evaluate" to enable heuristic-based config selection
            assert get_aot_mode() == "evaluate"

    def test_collect_mode(self) -> None:
        with patch.dict(os.environ, {"HELION_AOT_MODE": "collect"}):
            assert get_aot_mode() == "collect"

    def test_invalid_mode(self) -> None:
        with (
            patch.dict(os.environ, {"HELION_AOT_MODE": "invalid"}),
            pytest.raises(ValueError),
        ):
            get_aot_mode()


@onlyBackends(["triton", "cute"])
class TestBatchedParameter:
    """Tests for the batched parameter in aot_kernel."""

    def test_extract_features_without_batched(self) -> None:
        """Test that extract_shape_features includes all dimensions without batched."""
        x = torch.randn(32, 128)
        features = extract_shape_features([x])

        assert "arg0_ndim" in features
        assert "arg0_dim0" in features
        assert "arg0_dim1" in features
        assert "arg0_numel" in features
        assert features["arg0_dim0"] == 32
        assert features["arg0_dim1"] == 128

    def test_extract_features_with_batched(self) -> None:
        """Test that extract_shape_features excludes batched dimensions."""
        x = torch.randn(32, 128)
        # First dimension is batched
        features = extract_shape_features([x], batched=[[0, None]])

        assert "arg0_ndim" in features
        assert "arg0_dim0" not in features  # Batched dim excluded
        assert "arg0_dim1" in features  # Non-batched dim included
        assert "arg0_numel" not in features  # numel excluded when has batched dims
        assert features["arg0_dim1"] == 128

    def test_extract_features_multiple_args(self) -> None:
        """Test batched with multiple arguments (like rms_norm)."""
        weight = torch.randn(128)
        input_tensor = torch.randn(32, 128)
        eps = 1e-5

        # weight: not batched, input: first dim batched, eps: scalar
        batched = [[None], [0, None], None]
        features = extract_shape_features([weight, input_tensor, eps], batched=batched)

        # Weight features (not batched)
        assert "arg0_dim0" in features
        assert "arg0_numel" in features
        assert features["arg0_dim0"] == 128

        # Input features (first dim batched)
        assert "arg1_dim0" not in features  # Batched
        assert "arg1_dim1" in features  # Not batched
        assert "arg1_numel" not in features  # Excluded due to batched dim
        assert features["arg1_dim1"] == 128

        # Scalar feature
        assert "arg2_scalar" in features
        assert features["arg2_scalar"] == eps

    def test_aot_key_same_for_different_batch_sizes(self) -> None:
        """Test that different batch sizes produce the same key when batched is specified."""
        x1 = torch.randn(32, 128)
        x2 = torch.randn(64, 128)  # Different batch size, same hidden dim

        key1 = aot_key(x1, batched=[[0, None]])
        key2 = aot_key(x2, batched=[[0, None]])

        assert key1 == key2

    def test_aot_key_different_for_different_non_batch_dims(self) -> None:
        """Test that different non-batch dimensions produce different keys."""
        x1 = torch.randn(32, 128)
        x2 = torch.randn(32, 256)  # Same batch size, different hidden dim

        key1 = aot_key(x1, batched=[[0, None]])
        key2 = aot_key(x2, batched=[[0, None]])

        assert key1 != key2

    def test_aot_key_rms_norm_scenario(self) -> None:
        """Test the rms_norm scenario with weight, input, eps."""
        weight = torch.randn(128)
        input1 = torch.randn(32, 128)
        input2 = torch.randn(64, 128)  # Different batch size
        eps = 1e-5

        batched = [[None], [0, None], None]

        key1 = aot_key(weight, input1, eps, batched=batched)
        key2 = aot_key(weight, input2, eps, batched=batched)

        # Keys should be the same despite different batch sizes
        assert key1 == key2

    def test_batched_with_no_batched_dims(self) -> None:
        """Test that specifying all None in batched is equivalent to no batched."""
        x = torch.randn(32, 128)

        # All dimensions marked as not batched
        features_with_batched = extract_shape_features([x], batched=[[None, None]])
        features_without_batched = extract_shape_features([x])

        assert features_with_batched == features_without_batched


@onlyBackends(["triton", "cute"])
class TestConfigValidityPartitioning:
    """Tests for config validity partitioning in select_config_subset."""

    def test_single_config_with_validity_partitioning(self) -> None:
        """With max_configs=1, partitioning selects one config per partition."""
        # Two independent groups of shapes with disjoint valid configs
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": i} for i in range(4)],
            timings=np.array(
                [
                    [1.0, 2.0, np.inf, np.inf],  # Partition 1
                    [2.0, 1.0, np.inf, np.inf],  # Partition 1
                    [np.inf, np.inf, 1.0, 2.0],  # Partition 2
                    [np.inf, np.inf, 2.0, 1.0],  # Partition 2
                ]
            ),
            configs=[Config(block_sizes=[i]) for i in [64, 128, 256, 512]],
            shape_hashes=["s0", "s1", "s2", "s3"],
            config_hashes=["c0", "c1", "c2", "c3"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, max_configs=1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        # Should select configs from both partitions despite max_configs=1
        assert len(selected) >= 2
        # All shapes should have a valid selected config
        for i in range(4):
            assert any(np.isfinite(data.timings[i, j]) for j in selected)
        assert stats["num_partitions"] == 2

    def test_partitioning_independent_optimization(self) -> None:
        """Each partition selects its own optimal config independently."""
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": i} for i in range(4)],
            timings=np.array(
                [
                    [1.0, 5.0, np.inf, np.inf],  # Partition 1: config 0 best
                    [1.5, 5.0, np.inf, np.inf],
                    [np.inf, np.inf, 1.0, 5.0],  # Partition 2: config 2 best
                    [np.inf, np.inf, 1.5, 5.0],
                ]
            ),
            configs=[Config(block_sizes=[i]) for i in [64, 128, 256, 512]],
            shape_hashes=["s0", "s1", "s2", "s3"],
            config_hashes=["c0", "c1", "c2", "c3"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, max_configs=1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        # Config 0 for partition 1, config 2 for partition 2
        assert 0 in selected  # Best for partition 1
        assert 2 in selected  # Best for partition 2
        assert stats["num_partitions"] == 2

    def test_all_configs_valid_no_partitioning(self) -> None:
        """Single partition when all configs are valid for all shapes."""
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            timings=np.array(
                [
                    [1.0, 2.0],
                    [1.0, 2.0],
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        assert stats["num_partitions"] == 1
        assert len(selected) == 1
        assert selected[0] == 0  # Config 0 is best

    def test_uncoverable_shapes_skipped(self) -> None:
        """Shapes with no valid config are handled gracefully."""
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": i} for i in range(3)],
            timings=np.array(
                [
                    [1.0, 2.0],
                    [2.0, 1.0],
                    [np.inf, np.inf],  # No valid config
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s0", "s1", "s2"],
            config_hashes=["c0", "c1"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        # Stats should not be inf or nan
        assert np.isfinite(stats["max_slowdown"])
        assert np.isfinite(stats["geomean_slowdown"])
        assert np.isfinite(stats["avg_slowdown"])
        # Coverable shapes should be covered
        assert len(selected) >= 1

    def test_compute_validity_partitions_basic(self) -> None:
        """Test union-find partitioning directly."""
        timings = np.array(
            [
                [1.0, np.inf],
                [2.0, np.inf],
                [np.inf, 1.0],
                [np.inf, 2.0],
            ]
        )
        partitions, uncoverable = compute_validity_partitions(timings)

        assert len(partitions) == 2
        assert len(uncoverable) == 0

        # Check that shapes are correctly grouped
        partition_sets = [set(p) for p in partitions]
        assert {0, 1} in partition_sets
        assert {2, 3} in partition_sets

    def test_compute_validity_partitions_shared_config(self) -> None:
        """Shapes sharing a valid config are in the same partition."""
        timings = np.array(
            [
                [1.0, np.inf, 2.0],  # Configs 0,2 valid
                [np.inf, 1.0, 3.0],  # Configs 1,2 valid — connected via config 2
                [np.inf, 2.0, np.inf],  # Config 1 valid — connected via config 1
            ]
        )
        partitions, uncoverable = compute_validity_partitions(timings)

        # All connected through shared configs → single partition
        assert len(partitions) == 1
        assert set(partitions[0]) == {0, 1, 2}
        assert len(uncoverable) == 0

    def test_compute_validity_partitions_uncoverable(self) -> None:
        """Shapes with all-inf timings are uncoverable."""
        timings = np.array(
            [
                [1.0, 2.0],
                [np.inf, np.inf],  # Uncoverable
            ]
        )
        partitions, uncoverable = compute_validity_partitions(timings)

        assert len(partitions) == 1
        assert partitions[0] == [0]
        assert uncoverable == [1]

    def test_mixed_dimensionality_end_to_end(self) -> None:
        """Partitioning + decision tree correctly routes 2D vs 3D inputs."""
        from helion.autotuner.decision_tree_backend import DecisionTreeBackend

        # 2D shapes: no arg0_dim2 in feature dict
        # 3D shapes: arg0_dim2 present
        shape_features = [
            {"arg0_ndim": 2, "arg0_dim0": 1024, "arg0_dim1": 512},
            {"arg0_ndim": 2, "arg0_dim0": 2048, "arg0_dim1": 256},
            {"arg0_ndim": 3, "arg0_dim0": 32, "arg0_dim1": 1024, "arg0_dim2": 512},
            {"arg0_ndim": 3, "arg0_dim0": 64, "arg0_dim1": 512, "arg0_dim2": 256},
        ]

        # Configs 0,1 only valid for 2D; configs 2,3 only valid for 3D
        timings = np.array(
            [
                [1.0, 5.0, np.inf, np.inf],
                [1.5, 5.0, np.inf, np.inf],
                [np.inf, np.inf, 1.0, 5.0],
                [np.inf, np.inf, 1.5, 5.0],
            ]
        )

        data = ShapeConfigData(
            kernel_name="test_mixed",
            shape_features=shape_features,
            timings=timings,
            configs=[
                Config(block_sizes=[64]),
                Config(block_sizes=[128]),
                Config(block_sizes=[256]),
                Config(block_sizes=[512]),
            ],
            shape_hashes=["s0", "s1", "s2", "s3"],
            config_hashes=["c0", "c1", "c2", "c3"],
        )

        # Step 1: Config selection — partitioning should pick one config per partition
        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.5, max_configs=1, verbose=False
        )
        selected_indices, stats = select_config_subset(data, target)

        assert stats["num_partitions"] == 2
        assert 0 in selected_indices  # Best for 2D partition
        assert 2 in selected_indices  # Best for 3D partition

        # Step 2: Train decision tree on the partitioned selection
        data.selected_config_indices = selected_indices
        selected_configs = [data.configs[i] for i in selected_indices]

        # Gather all feature names across shapes (2D shapes lack arg0_dim2)
        feature_names = sorted(
            {
                k
                for f in shape_features
                for k, v in f.items()
                if isinstance(v, (int, float))
            }
        )

        backend = DecisionTreeBackend()
        result = backend.generate_heuristic(
            kernel_name="test_mixed",
            data=data,
            selected_configs=selected_configs,
            feature_names=feature_names,
        )

        # Tree should perfectly separate 2D from 3D shapes
        assert result.model_accuracy == 1.0

        # Step 3: Execute generated code and verify runtime predictions
        exec_globals: dict[str, object] = {"torch": torch}
        exec(result.generated_code, exec_globals)

        key_fn = exec_globals["key_test_mixed"]
        autotune_fn = exec_globals["autotune_test_mixed"]

        # 2D tensor → config index 0 (the 2D partition's config)
        assert key_fn(torch.randn(100, 200)) == 0
        # 3D tensor → config index 1 (the 3D partition's config)
        assert key_fn(torch.randn(10, 100, 200)) == 1

        # autotune returns the actual config dicts
        assert autotune_fn(torch.randn(100, 200)) == dict(selected_configs[0])
        assert autotune_fn(torch.randn(10, 100, 200)) == dict(selected_configs[1])


def _load_generated(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_static_standalone_call_key_covers_runtime_specialization() -> None:
    class Point(NamedTuple):
        x: int
        y: int

    tensor = torch.empty_strided((2, 3), (4, 1))
    base = _standalone_call_key((tensor, (1, 2), torch.float16, torch.device("cpu")))

    assert base != _standalone_call_key(
        (
            torch.empty_strided((3, 2), (2, 1)),
            (1, 2),
            torch.float16,
            torch.device("cpu"),
        )
    )
    assert base != _standalone_call_key(
        (tensor.as_strided((2, 3), (3, 1)), (1, 2), torch.float16, torch.device("cpu"))
    )
    assert base != _standalone_call_key(
        (tensor.to(torch.float64), (1, 2), torch.float16, torch.device("cpu"))
    )
    assert base != _standalone_call_key(
        (tensor, (1, 3), torch.float16, torch.device("cpu"))
    )
    assert _standalone_call_key(({"a": 1, "b": 2},)) == _standalone_call_key(
        ({"b": 2, "a": 1},)
    )
    with pytest.raises(TypeError, match="does not support object"):
        _standalone_call_key((object(),))
    with pytest.raises(TypeError, match="Point"):
        _standalone_call_key((Point(1, 2),))


def test_static_standalone_call_key_covers_descriptor_predicates() -> None:
    backing = torch.empty(32, dtype=torch.float32)
    aligned = backing.as_strided((2, 3), (4, 1), storage_offset=0)
    unaligned = backing.as_strided((2, 3), (4, 1), storage_offset=1)
    aligned_offset = backing.as_strided((2, 3), (4, 1), storage_offset=4)
    other_aligned = torch.empty_strided((2, 3), (4, 1))

    assert aligned.data_ptr() % 16 == 0
    assert unaligned.data_ptr() % 16 != 0
    assert aligned_offset.data_ptr() % 16 == 0
    assert aligned_offset.storage_offset() != 0
    assert _standalone_call_key((aligned,)) == _standalone_call_key((unaligned,))

    def descriptor_key(tensor: torch.Tensor) -> tuple[object, ...]:
        return _standalone_call_key((tensor,), tensor_descriptor_guards=True)

    assert descriptor_key(aligned) == descriptor_key(other_aligned)
    assert descriptor_key(aligned) != descriptor_key(unaligned)
    assert descriptor_key(aligned) != descriptor_key(aligned_offset)


def test_aot_cache_canonicalizes_defaults_for_compile_get(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.empty(2)
    signature = inspect.signature(lambda x, metadata=(1, 2): x)

    def normalize_args(*args: object) -> tuple[object, ...]:
        bound = signature.bind(*args)
        bound.apply_defaults()
        return tuple(bound.args)

    kernel_api = SimpleNamespace(
        name="demo",
        normalize_args=normalize_args,
        specialization_key=lambda args: tuple(args),
    )
    bound_kernel = SimpleNamespace(
        kernel=kernel_api,
        config_spec=SimpleNamespace(cute_flash_search_enabled=False),
        is_cacheable=lambda: True,
    )
    autotuner = SimpleNamespace(kernel=bound_kernel, args=(tensor,))
    monkeypatch.setenv("HELION_AOT_MODE", "compile")
    monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda: SimpleNamespace(hardware_id="test-hardware"),
    )

    cache = AOTAutotuneCache(autotuner)
    assert cache.args == (tensor, (1, 2))
    compiled: list[bool] = []
    selected_args: list[tuple[object, ...]] = []
    config = Config(block_sizes=[16])
    cache._maybe_run_compile = lambda: compiled.append(True)

    def get_config(args: tuple[object, ...]) -> Config:
        selected_args.append(args)
        return config

    cache._get_heuristic_config = get_config
    assert cache.get() is config
    assert compiled == [True]
    assert selected_args == [(tensor, (1, 2))]


def test_standalone_preserves_cute_launcher_import(tmp_path: Path) -> None:
    output = generate_standalone_file(
        "demo",
        [
            (
                "from __future__ import annotations\n"
                "from helion.runtime import default_cute_launcher as "
                "_default_cute_launcher\n\n"
                "def demo(x, *, _launcher=_default_cute_launcher):\n"
                "    return x\n"
            )
        ],
        "",
        tmp_path,
    )

    source = output.read_text()
    assert (
        "from helion.runtime import default_cute_launcher as _default_cute_launcher"
        in source
    )


@pytest.mark.parametrize(
    "descriptor_config",
    [
        {"block_sizes": [16], "indexing": "tensor_descriptor"},
        {"block_sizes": [16], "atomic_indexing": ["tensor_descriptor"]},
    ],
)
def test_dynamic_standalone_rejects_tensor_descriptors_before_codegen(
    tmp_path: Path,
    descriptor_config: dict[str, object],
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text(f"CONFIGS = [{descriptor_config!r}]\n")
    normalized: list[Config] = []
    codegen_calls: list[Config] = []

    def normalize(config: Config) -> Config:
        normalized.append(config)
        return config

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache.kernel = SimpleNamespace(
        settings=SimpleNamespace(static_shapes=False),
        kernel=SimpleNamespace(name="demo", __code__=kernel_function.__code__),
        _normalized_config_copy=normalize,
        to_triton_code=lambda config: codegen_calls.append(config),
    )
    cache._find_heuristic_file = lambda: heuristic_path

    AOTAutotuneCache.clear_caches()
    with pytest.raises(exc.InvalidAPIUsage, match="dynamic standalone AOT"):
        cache._maybe_run_compile()
    assert len(normalized) == 1
    assert codegen_calls == []
    assert "demo" not in AOTAutotuneCache._compiled_kernels
    assert not (tmp_path / "source_demo_standalone.py").exists()
    AOTAutotuneCache.clear_caches()


def test_dynamic_standalone_still_accepts_pointer_config(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text(
        "CONFIGS = [{'block_sizes': [16], 'indexing': 'pointer'}]\n"
    )
    codegen_calls: list[Config] = []

    def to_triton_code(config: Config) -> str:
        codegen_calls.append(config)
        return "from __future__ import annotations\n\ndef demo(x):\n    return x\n"

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache.kernel = SimpleNamespace(
        settings=SimpleNamespace(static_shapes=False),
        kernel=SimpleNamespace(name="demo", __code__=kernel_function.__code__),
        _normalized_config_copy=lambda config: config,
        to_triton_code=to_triton_code,
    )
    cache._find_heuristic_file = lambda: heuristic_path

    AOTAutotuneCache.clear_caches()
    cache._maybe_run_compile()
    assert len(codegen_calls) == 1
    assert (tmp_path / "source_demo_standalone.py").exists()
    AOTAutotuneCache.clear_caches()


def test_static_aot_compile_accumulates_observed_shapes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x, metadata=(1, 2)):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    signature = inspect.signature(kernel_function)

    def normalize_args(*args: object) -> tuple[object, ...]:
        bound = signature.bind(*args)
        bound.apply_defaults()
        return tuple(bound.args)

    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    output_path = tmp_path / "source_demo_standalone.py"
    config = Config(block_sizes=[16])

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache.args = (torch.empty(2),)

    def get_heuristic_config(args: tuple[object, ...]) -> Config:
        assert args[1] == (1, 2)
        return config

    cache._get_heuristic_config = get_heuristic_config

    def to_triton_code(_config: Config) -> str:
        size = cache.args[0].size(0)
        return (
            "from __future__ import annotations\n\n"
            "def demo(x, metadata=(1, 2)):\n"
            f"    return {size}\n"
        )

    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(
            __code__=kernel_function.__code__,
            name="demo",
            normalize_args=normalize_args,
        ),
        to_triton_code=to_triton_code,
    )

    def compile_shapes(sizes: tuple[int, ...]) -> str:
        AOTAutotuneCache.clear_caches()
        for size in sizes:
            cache.args = (torch.empty(size),)
            cache._compile_current_static_shape(heuristic_path, "demo")
            assert output_path.exists()
        return output_path.read_text()

    forward_source = compile_shapes((2, 3))
    reverse_source = compile_shapes((3, 2))
    assert forward_source == reverse_source
    module = _load_generated(output_path, "test_static_aot_compile")
    assert module.demo(torch.empty(2)) == 2
    assert module.demo(x=torch.empty(2)) == 2
    assert module.demo(torch.empty(3)) == 3
    with pytest.raises(ValueError, match="No standalone variant"):
        module.demo(torch.empty(4))

    cache.kernel.to_triton_code = lambda _config: (
        "from __future__ import annotations\n\ndef demo(x):\n    return 99\n"
    )
    with pytest.raises(RuntimeError, match="value-derived compile-time metadata"):
        cache._compile_current_static_shape(heuristic_path, "demo")

    prior_source = output_path.read_text()
    cache.args = (torch.empty(4),)

    def fail_compile(_config: Config) -> str:
        raise ValueError("compile failed")

    cache.kernel.to_triton_code = fail_compile
    with pytest.raises(RuntimeError, match="variant failed to compile"):
        cache._compile_current_static_shape(heuristic_path, "demo")
    assert output_path.read_text() == prior_source
    AOTAutotuneCache.clear_caches()


def test_static_aot_compile_dispatches_descriptor_alignment(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    config = Config(block_sizes=[16])

    backing = torch.empty(32, dtype=torch.float32)
    aligned = backing.as_strided((2, 3), (4, 1), storage_offset=0)
    unaligned = backing.as_strided((2, 3), (4, 1), storage_offset=1)
    aligned_offset = backing.as_strided((2, 3), (4, 1), storage_offset=4)

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache._get_heuristic_config = lambda _args: config
    result = 0

    def to_triton_code(_config: Config) -> str:
        return (
            f"from __future__ import annotations\n\ndef demo(x):\n    return {result}\n"
        )

    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(
            __code__=kernel_function.__code__,
            name="demo",
            normalize_args=lambda *args: tuple(args),
        ),
        to_triton_code=to_triton_code,
    )

    AOTAutotuneCache.clear_caches()
    for tensor, expected in ((aligned, 1), (unaligned, 2)):
        cache.args = (tensor,)
        result = expected
        cache._compile_current_static_shape(
            heuristic_path,
            "demo",
            tensor_descriptor_guards=True,
        )

    output_path = tmp_path / "source_demo_standalone.py"
    module = _load_generated(output_path, "test_static_aot_descriptor_alignment")
    assert module.demo(torch.empty_strided((2, 3), (4, 1))) == 1
    assert module.demo(unaligned) == 2
    with pytest.raises(ValueError, match="No standalone variant"):
        module.demo(aligned_offset)
    AOTAutotuneCache.clear_caches()


def test_static_aot_compile_supports_non_file_kernel_source(tmp_path: Path) -> None:
    namespace: dict[str, object] = {}
    exec(compile("def demo(x):\n    return x\n", "<stdin>", "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    config = Config(block_sizes=[16])

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache.args = (torch.empty(2),)
    cache._get_heuristic_config = lambda _args: config
    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(
            __code__=kernel_function.__code__,
            name="demo",
            normalize_args=lambda *args: tuple(args),
        ),
        to_triton_code=lambda _config: (
            "from __future__ import annotations\n\ndef demo(x):\n    return 2\n"
        ),
    )

    AOTAutotuneCache.clear_caches()
    cache._compile_current_static_shape(heuristic_path, "demo")
    output_path = tmp_path / "demo_standalone.py"
    assert output_path.exists()
    module = _load_generated(output_path, "test_static_aot_non_file")
    assert module.demo(torch.empty(2)) == 2

    other_dir = tmp_path / "other"
    cache.data_dir = other_dir
    cache._compile_current_static_shape(heuristic_path, "demo")
    other_output = other_dir / "demo_standalone.py"
    assert other_output.exists()
    AOTAutotuneCache.clear_caches()


def test_static_aot_compile_serializes_concurrent_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    config = Config(block_sizes=[16])
    codegen_barrier = threading.Barrier(2)

    def make_cache(size: int) -> AOTAutotuneCache:
        cache = object.__new__(AOTAutotuneCache)
        cache.data_dir = tmp_path
        cache.hardware_id = "test-hardware"
        cache.args = (torch.empty(size),)
        cache._get_heuristic_config = lambda _args: config

        def to_triton_code(_config: Config) -> str:
            codegen_barrier.wait(timeout=5)
            return (
                "from __future__ import annotations\n\n"
                f"def demo(x):\n    return {size}\n"
            )

        cache.kernel = SimpleNamespace(
            kernel=SimpleNamespace(
                __code__=kernel_function.__code__,
                name="demo",
                normalize_args=lambda *args: tuple(args),
            ),
            to_triton_code=to_triton_code,
        )
        return cache

    import helion.autotuner.aot_compile as aot_compile

    original_generate = aot_compile.generate_standalone_file
    active_writers = 0
    max_active_writers = 0
    writers_lock = threading.Lock()

    def tracked_generate(
        kernel_name: str,
        triton_codes: list[str],
        heuristic_code: str,
        output_dir: Path,
        kernel_source_file: str | None = None,
        dispatch_keys: list[tuple[object, ...]] | None = None,
        tensor_descriptor_guards: bool = False,
    ) -> Path:
        nonlocal active_writers, max_active_writers
        with writers_lock:
            active_writers += 1
            max_active_writers = max(max_active_writers, active_writers)
        try:
            time.sleep(0.05)
            return original_generate(
                kernel_name,
                triton_codes,
                heuristic_code,
                output_dir,
                kernel_source_file,
                dispatch_keys,
                tensor_descriptor_guards,
            )
        finally:
            with writers_lock:
                active_writers -= 1

    monkeypatch.setattr(aot_compile, "generate_standalone_file", tracked_generate)
    AOTAutotuneCache.clear_caches()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                make_cache(size)._compile_current_static_shape,
                heuristic_path,
                "demo",
            )
            for size in (2, 3)
        ]
        for future in futures:
            future.result()

    assert max_active_writers == 1
    module = _load_generated(
        tmp_path / "source_demo_standalone.py",
        "test_static_aot_concurrent",
    )
    assert module.demo(torch.empty(2)) == 2
    assert module.demo(torch.empty(3)) == 3
    AOTAutotuneCache.clear_caches()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
