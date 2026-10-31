"""
Resource fitting for AOT configs borrowed from another device.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
import logging
import os
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch
import weakref

import pytest
import torch
from torch._inductor.runtime.triton_compat import OutOfResources
from torch._inductor.utils import fresh_cache

import helion
from helion import exc
from helion._compiler.backend import TritonBackend
from helion._compiler.triton.config_fitting import annotate_launch_contract
from helion._hardware import HardwareInfo
from helion._hardware import get_hardware_info
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
import helion.autotuner.aot_cache as aot_cache_module
from helion.autotuner.aot_cache import AOTAutotuneCache
import helion.language as hl
from helion.runtime.config import Config
import helion.runtime.triton.launcher as launcher_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from pathlib import Path


class TestLaunchContract:
    @staticmethod
    def _key(source: str) -> str:
        root = ast.parse(source)
        original = ast.dump(root)
        annotate_launch_contract(root)
        assignment = root.body.pop()
        assert ast.dump(root) == original
        assert isinstance(assignment, ast.Assign)
        assert isinstance(assignment.value, ast.Constant)
        return assignment.value.value

    @pytest.mark.parametrize(
        ("before", "after", "compatible"),
        [
            ("num_warps=8, num_stages=8", "num_warps=4, num_stages=2", True),
            ("return x + 1", "return x + 2", True),  # Device code can change.
            ("WIDTH = 1", "WIDTH = 2", False),
            ("(WIDTH,)", "(WIDTH + 1,)", False),
            ("device, (WIDTH,), x", "device, (WIDTH,), x + 1", False),
            ("helper(num_stages=8)", "helper(num_stages=2)", False),
            ("num_stages=8)", "num_stages=stages)", False),
            ('"num_stages=8"', '"num_stages=2"', False),
        ],
    )
    def test_compatibility_uses_host_and_global_semantics(
        self, before: str, after: str, compatible: bool
    ) -> None:
        source = """
import triton
WIDTH = 1
@triton.jit
def device(x):
    return x + 1
def demo(x, *, _launcher=default_launcher):
    helper(num_stages=8)
    _launcher(device, (WIDTH,), x, num_warps=8, num_stages=8)
    return "num_stages=8"
"""
        assert before in source
        assert (
            self._key(source) == self._key(source.replace(before, after))
        ) is compatible

    def test_formatting_and_source_locations_do_not_affect_the_contract(self) -> None:
        source = "def demo(x):\n    return _launcher(kernel, (1,), x, num_stages=8)\n"
        assert self._key(source) == self._key(
            "\n# comment\n" + source.replace("=8", " = 8")
        )


class TestFitEligibility:
    def test_same_device_selection_is_not_wrapped(self) -> None:
        bound = _FakeBoundKernel(300_000, _buffers)
        config = Config(block_sizes=[64], num_stages=8)
        compiled = bound.compile_config(config)
        assert (
            TritonBackend().fit_on_first_launch(cast("Any", bound), config, compiled)
            is compiled
        )

    def test_missing_contract_does_not_replace_the_device_kernel(self) -> None:
        bound = _FakeBoundKernel(
            128, lambda config: config.num_warps * 32, resource="threads"
        )
        compile_config = bound.compile_config

        def without_contract(
            config: Config, *, allow_print: bool = True
        ) -> Callable[..., object]:
            compiled = compile_config(config, allow_print=allow_print)
            compiled.__globals__.pop("_helion_launch_contract")
            return compiled

        bound.compile_config = without_contract  # type: ignore[method-assign]
        with pytest.raises(OutOfResources, match="threads"):
            bound.select(Config(block_sizes=[64], num_warps=8))(torch.zeros(4))
        assert bound.launched == []

    def test_module_bindings_must_match_before_reusing_a_launch(self) -> None:
        bound = _FakeBoundKernel(
            128,
            lambda config: config.num_warps * 32,
            resource="threads",
            grid=lambda config: "(GRID,)",
            bindings=lambda config: f"GRID = {config.num_warps}\n",
        )
        run = bound.select(Config(block_sizes=[64], num_warps=8))
        with pytest.raises(OutOfResources, match="threads") as error:
            run(torch.zeros(4))
        assert error.value.required == 256
        assert bound.launched == []

    @pytest.mark.parametrize("probe", ["empty", "interrupt", "empty_candidate"])
    def test_unvalidated_render_remains_eligible(self, probe: str) -> None:
        bound = _FakeBoundKernel(300_000, _buffers)
        config = Config(block_sizes=[64], num_stages=8)
        run = bound.select(config)
        backend = TritonBackend()
        with (
            patch.object(
                bound,
                "compile_config",
                side_effect=lambda *args, **kwargs: lambda *args, **kwargs: None,
            )
            if probe == "empty_candidate"
            else contextlib.nullcontext()
        ):
            if probe == "interrupt":
                with (
                    patch.object(bound, "before_launch", side_effect=KeyboardInterrupt),
                    pytest.raises(KeyboardInterrupt),
                ):
                    backend.fit_config_for_rendering(
                        cast("Any", bound), (torch.zeros(4),)
                    )
            else:
                backend.fit_config_for_rendering(
                    cast("Any", bound), (torch.zeros(0 if probe == "empty" else 4),)
                )
        assert bound._run is run
        assert bound._config == config
        assert bound._config_from_other_device
        assert bound.launched == []
        backend.fit_config_for_rendering(cast("Any", bound), (torch.zeros(4),))
        assert bound._config == Config(block_sizes=[64], num_stages=5)
        assert not bound._config_from_other_device
        assert bound.launched == []

    def test_render_can_validate_the_last_permitted_candidate(self) -> None:
        bound = _FakeBoundKernel(
            16 * 32, lambda config: config.num_warps * 32, resource="threads"
        )
        bound.select(Config(block_sizes=[64], num_warps=2**20))
        TritonBackend().fit_config_for_rendering(cast("Any", bound), (torch.zeros(4),))
        assert bound._config is not None and bound._config.num_warps == 16
        assert len(bound.compiled) == 17
        assert bound.launched == []

    def test_render_validates_later_native_kernels(self) -> None:
        bound = _FakeBoundKernel(300_000, _buffers, kernel_names=("first", "second"))
        bound.select(Config(block_sizes=[64], num_stages=8))
        native_run = _FakeTritonKernel.run

        def run(kernel: _FakeTritonKernel, *args: object, **kwargs: object) -> object:
            with patch.object(
                bound,
                "usage",
                lambda config: 0 if kernel.__name__ == "first" else _buffers(config),
            ):
                return native_run(kernel, *args, **kwargs)  # pyrefly: ignore [bad-argument-type]

        with patch.object(_FakeTritonKernel, "run", run):
            TritonBackend().fit_config_for_rendering(
                cast("Any", bound), (torch.zeros(4),)
            )
        assert bound._config is not None and bound._config.num_stages == 5
        assert not bound._config_from_other_device
        assert bound.launched == []


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
    tmp_path: Path, configs: dict[str, dict[str, object]], *, other_device: bool = True
) -> Iterator[None]:
    """Make AOT selection pick *configs* from a heuristic for an older GPU."""
    hardware = get_hardware_info()
    computes = hardware.get_compatible_compute_ids()
    if other_device and len(computes) < 2:
        pytest.skip("no older compute capability to fall back to")
    source_stem = os.path.basename(__file__).removesuffix(".py")
    heuristic = tmp_path / (
        f"_helion_aot_{source_stem}_{hardware.device_kind}_{computes[int(other_device)]}.py"
    )
    heuristic.write_text(
        "".join(
            f"def autotune_{name}(*args):\n    return {config!r}\n"
            for name, config in configs.items()
        )
    )
    with (
        patch.dict(os.environ, {"HELION_HEURISTIC_DIR": str(tmp_path)}),
        patch.object(TritonBackend, "_fitted_kernels", weakref.WeakSet(), create=True),
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


class _FakeBoundKernel:
    """A bound kernel whose configs' kernels fail to launch when they need more
    of a resource than ``limit``.

    Its generated host code calls ``before_launch``, launches each kernel in
    ``kernel_names`` through the launcher it is given, then calls
    ``after_launch``.
    """

    def __init__(
        self,
        limit: int,
        usage: Callable[[Config], int],
        resource: str = "shared memory",
        kernel_names: Sequence[str] = ("_helion_demo",),
        launch_kwargs: str = "",
        grid: Callable[[Config], str] = lambda config: "(1,)",
        bindings: Callable[[Config], str] = lambda config: "",
    ) -> None:
        self.limit = limit
        self.usage = usage
        self.resource = resource
        self.kernel_names = kernel_names
        self.launch_kwargs_source = launch_kwargs
        self.grid = grid
        self.bindings = bindings
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
        source = (
            self.bindings(config) + "def demo(x, *, _launcher=_default_launcher):\n"
        )
        source += (
            "    before_launch(x)\n    if not x.numel():\n        return x\n"
            + "".join(
                f"    _launcher({name}, {self.grid(config)}, x, {launch})\n"
                for name in self.kernel_names
            )
        )
        source += "    after_launch(x)\n    return x\n"
        root = ast.parse(source)
        annotate_launch_contract(root)
        namespace: dict[str, Any] = {
            "torch": torch,
            "before_launch": lambda *args: self.before_launch(*args),
            "after_launch": lambda *args: self.after_launch(*args),
            "_default_launcher": lambda *args, **kwargs: self.default_launches.append(
                args[0]
            ),
            **{
                name: _FakeTritonKernel(name, config, self)
                for name in self.kernel_names
            },
        }
        exec(
            compile(ast.fix_missing_locations(root), "<fake_compiler>", "exec"),
            namespace,
        )
        compiled = namespace["demo"]
        self.compiled.append(config)
        if allow_print:
            self.printed.append(config)
        self._compile_cache[config] = compiled
        self._cache_path_map[config] = "<fake_compiler>"
        return compiled

    def set_config(self, config: Config) -> None:
        self._run = self.compile_config(config)
        self._config = config

    def select(self, config: Config) -> Callable[..., object]:
        """Install *config* as AOT selection does for a borrowed config."""
        self._config_from_other_device = True
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
    def test_lowers_num_stages_until_the_kernel_fits(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
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

    def test_steps_down_when_the_estimate_is_too_big(self) -> None:
        bound_kernel = _FakeBoundKernel(
            300_000, lambda config: 150_000 + _buffers(config)
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
        num_stages: int,
        range_num_stages: list[int],
        expected: tuple[int, list[int]],
    ) -> None:
        def usage(config: Config) -> int:
            per_loop = cast("list[int]", config.config["range_num_stages"])
            return (max([config.num_stages, *per_loop]) - 1) * 64 * 1024

        bound_kernel = _FakeBoundKernel(150_000, usage)
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

    def test_threads_halve_num_warps(self) -> None:
        bound_kernel = _FakeBoundKernel(
            128, lambda config: config.num_warps * 32, resource="threads"
        )
        bound_kernel.select(Config(block_sizes=[64], num_warps=16))(torch.zeros(4))
        assert [config.num_warps for _, config in bound_kernel.attempts] == [16, 8, 4]
        assert bound_kernel.launched_kernels == [Config(block_sizes=[64], num_warps=4)]

    def test_warp_specialized_loops_keep_four_warps(self) -> None:
        bound_kernel = _FakeBoundKernel(
            64, lambda config: config.num_warps * 32, resource="threads"
        )
        run = bound_kernel.select(
            Config(block_sizes=[64], num_warps=4, range_warp_specializes=[True])
        )
        with pytest.raises(OutOfResources):
            run(torch.zeros(4))
        assert bound_kernel.compiled == [bound_kernel._config]

    @pytest.mark.parametrize("resource", ["shared memory", "tensor memory"])
    def test_raises_the_original_error_when_nothing_smaller_fits(
        self, resource: str
    ) -> None:
        bound_kernel = _FakeBoundKernel(
            1000, lambda config: 5000 * config.num_stages, resource=resource
        )
        config = Config(
            block_sizes=[64], num_stages=1 if resource == "shared memory" else 4
        )
        with pytest.raises(OutOfResources) as error:
            bound_kernel.select(config)(torch.zeros(4))
        assert error.value.required == 5000 * config.num_stages
        assert bound_kernel.compiled == [config]
        assert bound_kernel._config == config

    def test_gives_up_after_the_attempt_limit(self) -> None:
        bound_kernel = _FakeBoundKernel(0, lambda config: 1, resource="threads")
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
        self, failure: str
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
        self,
    ) -> None:
        bound_kernel = _FakeBoundKernel(3500, lambda config: config.num_stages * 1000)
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

    def test_concurrent_failed_fit_preserves_original_errors(self) -> None:
        bound_kernel = _FakeBoundKernel(
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

    def test_other_errors_raise_the_original_error(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))

        def fail(config: Config, *, allow_print: bool = True) -> Callable[..., object]:
            raise RuntimeError("codegen failed")

        bound_kernel.compile_config = fail  # type: ignore[method-assign]
        with pytest.raises(OutOfResources) as error:
            run(torch.zeros(4))
        assert error.value.required == 7 * 64 * 1024

    def test_a_smaller_config_that_fails_otherwise_raises_the_original_error(
        self,
    ) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
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

    def test_distributed_kernels_are_not_fitted(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        bound_kernel.env.process_group_name = "pg"
        config = Config(block_sizes=[64], num_stages=8)
        assert bound_kernel.select(config) is bound_kernel._compile_cache[config]

    def test_host_code_runs_once_per_call(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
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

    def test_launches_its_own_kernel_when_the_host_code_matches(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))(torch.zeros(4))
        assert bound_kernel.launched_kernels == [Config(block_sizes=[64], num_stages=5)]

    def test_launches_the_running_kernel_with_fewer_stages_when_the_host_code_differs(
        self,
    ) -> None:
        # Code generated later can compute other arguments, for example after it
        # guarded on an input size, so the running kernel is launched instead.
        bound_kernel = _FakeBoundKernel(
            300_000, _buffers, grid=lambda config: f"({config.num_stages},)"
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
        self, resource: str, config: Config
    ) -> None:
        def usage(config: Config) -> int:
            per_loop = cast("list[int]", config.config.get("range_num_stages", []))
            return config.num_warps * 32 + max([config.num_stages, *per_loop]) * 1000

        bound_kernel = _FakeBoundKernel(
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
        self, launch_kwargs: str, fitted: bool
    ) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers, launch_kwargs=launch_kwargs)
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

    def test_keeps_a_config_that_fits(self, caplog: pytest.LogCaptureFixture) -> None:
        bound_kernel = _FakeBoundKernel(10**9, _buffers)
        config = Config(block_sizes=[64], num_stages=8)
        with caplog.at_level(logging.WARNING, logger="helion"):
            bound_kernel.select(config)(torch.zeros(4))
        assert bound_kernel._config is config
        assert bound_kernel._run is bound_kernel._compile_cache[config]
        assert bound_kernel.compiled == [config]
        assert caplog.records == []

    def test_later_kernels_of_the_call_use_the_fitted_config(self) -> None:
        bound_kernel = _FakeBoundKernel(
            300_000,
            _buffers,
            kernel_names=("_helion_demo", "_helion_demo_1"),
        )
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))(torch.zeros(4))
        assert [
            (name, config.num_stages) for name, config in bound_kernel.launched
        ] == [("_helion_demo", 5), ("_helion_demo_1", 5)]

    def test_empty_call_keeps_first_launch_fitting_armed(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        run(torch.zeros(0))
        assert bound_kernel._run is run
        assert bound_kernel.attempts == []
        run(torch.zeros(4))
        assert _stages(bound_kernel.launched) == [5]
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=5)

    def test_empty_call_does_not_publish_a_pending_candidate(self) -> None:
        bound_kernel = _FakeBoundKernel(
            300_000, lambda config: 150_000 + _buffers(config)
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
        self,
    ) -> None:
        bound_kernel = _FakeBoundKernel(3500, lambda config: config.num_stages * 1000)
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

    def test_concurrent_first_calls_fit_once(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
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

    def test_an_interrupted_fit_can_be_retried(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
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
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kernel = _FakeKernel()
        with (
            patch.object(TritonBackend, "_fitted_kernels", weakref.WeakSet()),
            caplog.at_level(logging.WARNING, logger="helion"),
        ):
            # Two shapes of one kernel, then another kernel with the same name.
            for each_kernel in (kernel, kernel, _FakeKernel()):
                bound_kernel = _FakeBoundKernel(300_000, _buffers)
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
        self, print_output_code: bool
    ) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        bound_kernel.settings.print_output_code = print_output_code
        config = Config(block_sizes=[64], num_stages=8)
        bound_kernel.select(config)(torch.zeros(4))
        # The fitted config compiles silently during the fit; when output code is
        # printed, set_config compiles it again to print it.
        fitted = Config(block_sizes=[64], num_stages=5)
        expected = [config, fitted] if print_output_code else [config]
        assert bound_kernel.printed == expected

    def test_tracing_calls_the_compiled_host_function(self) -> None:
        bound_kernel = _FakeBoundKernel(10**9, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            run(torch.zeros(4))
        # The host code used its own launcher, and the config is still unchecked.
        assert len(bound_kernel.default_launches) == 1
        assert bound_kernel.launched == []
        assert bound_kernel._run is run


@skipUnlessBackends(["triton"])
class TestFitConfigForRendering:
    def test_fits_flagged_configs_with_the_example_args(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        bound_kernel._config_from_other_device = True
        TritonBackend().fit_config_for_rendering(
            cast("Any", bound_kernel), (torch.zeros(4),)
        )
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=5)
        # Nothing was launched, and the fitted config is no longer wrapped.
        assert bound_kernel.launched == []
        assert bound_kernel._run is bound_kernel._compile_cache[bound_kernel._config]

    def test_checks_a_flagged_config_that_fits_once(self) -> None:
        bound_kernel = _FakeBoundKernel(10**9, _buffers)
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

    def test_leaves_unflagged_configs_alone(self) -> None:
        bound_kernel = _FakeBoundKernel(300_000, _buffers)
        run = bound_kernel.select(Config(block_sizes=[64], num_stages=8))
        bound_kernel._config_from_other_device = False
        TritonBackend().fit_config_for_rendering(
            cast("Any", bound_kernel), (torch.zeros(4),)
        )
        assert bound_kernel._config == Config(block_sizes=[64], num_stages=8)
        assert bound_kernel._run is run

    def test_keeps_the_config_when_nothing_smaller_fits(self) -> None:
        bound_kernel = _FakeBoundKernel(1000, lambda config: 5000)
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

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT selection")
    @skipIfTileIR("checks Triton launch fitting")
    def test_same_device_aot_config_is_not_wrapped(self, tmp_path: Path) -> None:
        config = {**_OVERSIZED_MATMUL_CONFIG, "num_stages": 2}
        with _heuristic_for_older_gpu(
            tmp_path, {"aot_fit_matmul": config}, other_device=False
        ):
            kernel = _make_aot_fit_matmul()
            x, y = self._inputs()
            bound = kernel.bind((x, y))
            selected = bound.autotune((x, y), force=False)
            assert not bound._config_from_other_device
            assert (
                bound._run
                is bound._compile_cache[bound._normalized_config_copy(selected)]
            )
            torch.testing.assert_close(kernel(x, y), x @ y, atol=1e-1, rtol=1e-2)

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses local search")
    @skipIfTileIR("checks Triton launch fitting")
    def test_local_search_and_cache_are_not_wrapped(self) -> None:
        from helion.autotuner import FiniteSearch

        configs = [
            Config(block_sizes=[32, 64, 64], num_stages=2),
            Config(block_sizes=[64, 64, 64], num_stages=2),
        ]
        kernel = helion.kernel(autotune_precompile=None)(_make_aot_fit_matmul().fn)
        args = self._inputs()
        bound = kernel.bind(args)
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "FiniteSearch"}):
            selected = bound.autotune(args, configs=configs)
            assert not bound._config_from_other_device
            assert (
                bound._run
                is bound._compile_cache[bound._normalized_config_copy(selected)]
            )
            assert "_helion_launch_contract" not in bound._run.__globals__
            with patch.object(
                FiniteSearch, "_autotune", side_effect=AssertionError("cache miss")
            ):
                assert bound.autotune(args, configs=configs, force=False) == selected
            assert (
                bound._run
                is bound._compile_cache[bound._normalized_config_copy(selected)]
            )
        torch.testing.assert_close(
            kernel(*args), args[0] @ args[1], atol=1e-1, rtol=1e-2
        )

    @skipIfNotCUDA()
    @skipIfRefEager("ref-eager mode bypasses AOT selection")
    @skipIfTileIR("checks compiler launch contracts")
    def test_config_compiled_before_aot_selection_gets_a_contract(
        self, tmp_path: Path
    ) -> None:
        config = Config(**_OVERSIZED_MATMUL_CONFIG)  # pyrefly: ignore [bad-argument-type]
        with _heuristic_for_older_gpu(tmp_path, {"aot_fit_matmul": config.config}):
            kernel = _make_aot_fit_matmul()
            args = self._inputs()
            bound = kernel.bind(args)
            first = bound.compile_config(config)
            assert "_helion_launch_contract" not in first.__globals__
            bound.autotune(args, force=False)
            assert (
                "_helion_launch_contract"
                in bound._compile_cache[
                    bound._normalized_config_copy(config)
                ].__globals__
            )
            torch.testing.assert_close(
                kernel(*args), args[0] @ args[1], atol=1e-1, rtol=1e-2
            )
