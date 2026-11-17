"""A backend written only against helion.backend_api compiles and runs a kernel
(a smoke test: it reuses Triton codegen)."""

from __future__ import annotations

from typing import Callable
from typing import ClassVar
import unittest

import pytest
import torch

import helion
from helion import backend_api
from helion._compiler import backend_registry
from helion._compiler.backend_registry import unregister_compiler_backend
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner import benchmarking
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl


class _ApiOnlyBackend(backend_api.TritonBackend):
    # A backend written only against helion.backend_api, reusing Triton
    # codegen under a new name (the TileIRBackend pattern).
    @property
    def name(self) -> str:
        return "_test_api_only"

    @property
    def codegen_name(self) -> str:
        return "triton"


@onlyBackends(["triton"])
@skipIfRefEager("compiles a kernel; ref mode runs it eagerly")
class TestBackendApiCompile(TestCase):
    def test_api_only_backend_compiles_and_runs(self) -> None:
        repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
        backend_api.register_compiler_backend(_ApiOnlyBackend)
        try:

            @helion.kernel(backend="_test_api_only")
            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            y = torch.randn(64, 64, device=DEVICE)
            bound = add.bind((x, y))
            out = bound.compile_config(bound.config_spec.default_config())(x, y)
        finally:
            unregister_compiler_backend("_test_api_only")
            backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before
        torch.testing.assert_close(out, x + y)
        self.assertIsInstance(bound.env.backend, _ApiOnlyBackend)


# What docs/api/backend_api.md says the autotuner passes a plugin's timers.
_SINGLE_TIMER_KEYWORDS = {
    "warmup",
    "rep",
    "return_mode",
    "process_group_name",
    "l2_flush_mode",
}
_INTERLEAVED_TIMER_KEYWORDS = {"repeat", "desc", "max_total_ms"}


class _TimedBackend(_ApiOnlyBackend):
    # Records the keywords the autotuner passes to this backend's timers.
    single: ClassVar[list[frozenset[str]]] = []
    paired: ClassVar[list[frozenset[str]]] = []

    @property
    def name(self) -> str:
        return "_test_timed"

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        def timer(fn: Callable[[], object], **kwargs: object) -> float:
            type(self).single.append(frozenset(kwargs))
            return benchmarking.do_bench(fn, **kwargs)  # pyrefly: ignore[bad-argument-type]

        return timer

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        def timer(fns: list[Callable[[], object]], **kwargs: object) -> list[float]:
            type(self).paired.append(frozenset(kwargs))
            return benchmarking.interleaved_bench(fns, **kwargs)  # pyrefly: ignore[bad-argument-type]

        return timer


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@onlyBackends(["triton"])
@skipIfRefEager("autotuning compiles candidates; ref mode runs the kernel eagerly")
class TestBackendTimers(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _TimedBackend.single = []
        _TimedBackend.paired = []
        repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
        self.addCleanup(
            setattr, backend_registry, "_REPAIRED_CODEGEN_NAMES", repaired_before
        )
        backend_api.register_compiler_backend(_TimedBackend)
        self.addCleanup(unregister_compiler_backend, "_test_timed")
        self.x = torch.randn(256, 256, device=DEVICE)
        kernel = helion.kernel(
            backend="_test_timed", autotune_log_level=0, autotune_precompile=None
        )(_add)
        self.bound = kernel.bind((self.x, self.x))

    def test_single_config_timer_gets_the_documented_keywords(self) -> None:
        configs = [
            self.bound.config_spec.default_config(),
            helion.Config(block_sizes=[32, 32], num_warps=4),
        ]
        backend_api.FiniteSearch(
            self.bound, (self.x, self.x), configs=configs
        ).autotune()
        self.assertTrue(_TimedBackend.single)
        for keywords in _TimedBackend.single:
            self.assertLessEqual(keywords, _SINGLE_TIMER_KEYWORDS)
            self.assertLessEqual({"warmup", "rep", "return_mode"}, keywords)

    @pytest.mark.slow  # runs a short pattern search
    def test_interleaved_timer_gets_the_documented_keywords(self) -> None:
        PatternSearch(
            self.bound,
            (self.x, self.x),
            initial_population=4,
            copies=1,
            max_generations=1,
        ).autotune()
        self.assertTrue(_TimedBackend.paired)
        for keywords in _TimedBackend.paired:
            self.assertLessEqual(keywords, _INTERLEAVED_TIMER_KEYWORDS)
            self.assertIn("repeat", keywords)


if __name__ == "__main__":
    unittest.main()
