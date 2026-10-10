from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import TestCase
from helion._utils import counters
from helion.autotuner.base_search import BaseAutotuner
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache
from helion.runtime.ref_mode import is_ref_mode_enabled


@helion.kernel(backend="triton")
def _cached_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    scale = hl.specialize(scale)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


_SHIFT = 1


class _OpaqueState:
    enabled = True


_OPAQUE = _OpaqueState()


@helion.kernel(backend="triton")
def _cached_opaque(x: torch.Tensor) -> torch.Tensor:
    if _OPAQUE.enabled:
        return x
    return x


@helion.kernel(backend="triton")
def _cached_shift(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + _SHIFT
    return out


def _bound(x: torch.Tensor, scale: int = 2, **settings: object):
    settings.setdefault("autotune_effort", "none")
    kernel = helion.kernel(backend="triton", generated_code_cache=True, **settings)(
        _cached_scale.fn
    )
    return kernel.bind((x, scale))


def _source_loader(root: Path, sources: list[str]):
    def load(source: str, *, extra: str = ""):
        sources.append(source)
        path = root / f"module_{len(sources)}.py"
        path.write_text(source)
        # Importing/launching the generated module belongs to the CUDA test.
        return SimpleNamespace(
            __file__=str(path),
            _cached_scale=lambda *args: None,
            _runtime_scale=operator.mul,
        )

    return load


@unittest.skipIf(
    is_ref_mode_enabled(helion.Settings()),
    "generated source caching requires frontend compilation",
)
class _GeneratedCodeCacheTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self.cache_root = Path(root)
        self._test_stack.enter_context(
            patch.dict(os.environ, {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"})
        )


@instantiate_parametrized_tests
class TestGeneratedCodeCache(_GeneratedCodeCacheTestCase):
    def setUp(self) -> None:
        super().setUp()
        # CPU tests exercise the real frontend/codegen, but never import Triton.
        for name, value in (
            ("helion_key", "helion-v1"),
            ("torch_key_wrapper", "torch-v1"),
            ("triton_key_wrapper", "triton-v1"),
        ):
            self._test_stack.enter_context(
                patch.object(cache, name, return_value=value)
            )
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )

    def test_fresh_binding_skips_codegen(self) -> None:
        config = helion.Config(block_sizes=[32])
        x = torch.ones(16)
        first = _bound(x)
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            first.set_config(config)
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                second = _bound(x)
            self.assertEqual(compile_fn.call_count, 0)
            with patch.object(
                second,
                "to_triton_code",
                side_effect=AssertionError("codegen cache miss"),
            ):
                second.set_config(config)
        self.assertEqual(sources[0], sources[1])
        self.assertEqual(
            counters["generated_code_cache"], {"miss": 1, "hit": 1, "frontend_hit": 1}
        )

    def test_dependency_discovery_order_does_not_change_binding(self) -> None:
        x = torch.ones(16)
        config = helion.Config(block_sizes=[32])
        sources: list[str] = []
        getclosurevars = inspect.getclosurevars

        def reverse_dependencies(fn):
            closure = getclosurevars(fn)
            return closure._replace(
                globals=dict(reversed(closure.globals.items())),
                nonlocals=dict(reversed(closure.nonlocals.items())),
            )

        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(x).set_config(config)
            with (
                patch.object(cache.inspect, "getclosurevars", reverse_dependencies),
                patch.object(
                    KernelCompiler,
                    "compile",
                    side_effect=AssertionError("frontend ran"),
                ),
            ):
                _bound(x).set_config(config)
        self.assertEqual(sources[0], sources[1])

    def test_only_selected_config_is_saved(self) -> None:
        bound = _bound(torch.ones(16))
        winner = helion.Config(block_sizes=[32])
        other = helion.Config(block_sizes=[64])
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            bound.compile_config(winner)
            bound.compile_config(other)
            self.assertFalse((self.cache_root / "generated_code").exists())
            with patch.object(
                bound,
                "to_triton_code",
                side_effect=AssertionError("winner regenerated"),
            ):
                bound.set_config(winner)
            fresh = _bound(torch.ones(16))
            with patch.object(
                fresh, "to_triton_code", side_effect=AssertionError("winner cache miss")
            ):
                fresh.set_config(winner)
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[0], sources[2])
        entries = list((self.cache_root / "generated_code").glob("*.json"))
        self.assertEqual(len(entries), 1)
        self.assertEqual(json.loads(entries[0].read_text())["source"], sources[0])

    def test_search_seed_does_not_change_source_key(self) -> None:
        first = _bound(torch.ones(16), autotune_random_seed=1)
        second = _bound(torch.ones(16), autotune_random_seed=2)
        config = helion.Config(block_sizes=[32])
        key = cache.generated_code_cache_key(
            first, first._normalized_config_copy(config)
        )
        self.assertIsNotNone(key)
        self.assertEqual(
            key,
            cache.generated_code_cache_key(
                second, second._normalized_config_copy(config)
            ),
        )

    def test_dynamic_shapes_skip_frontend(self) -> None:
        config = helion.Config(block_sizes=[32])
        x = torch.ones(16)
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(x, static_shapes=False).set_config(config)
            with patch.object(
                KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
            ):
                fresh = _bound(x, static_shapes=False)
                fresh.set_config(config)
        self.assertEqual(sources[0], sources[1])

    def test_other_config_materializes_frontend(self) -> None:
        x = torch.ones(16)
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(x).set_config(helion.Config(block_sizes=[32]))
            fresh = _bound(x)
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                fresh.set_config(helion.Config(block_sizes=[64]))
            self.assertEqual(compile_fn.call_count, 1)
        self.assertNotEqual(sources[0], sources[1])

    def test_frontend_introspection_after_inputs_expire(self) -> None:
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(torch.ones(16)).set_config(helion.Config(block_sizes=[32]))
        fresh = _bound(torch.ones(16))
        self.assertIn("tl.store", fresh.to_code(helion.Config(block_sizes=[64])))

    def test_preset_selector_still_runs(self) -> None:
        calls = []
        config = helion.Config(block_sizes=[32])

        class PresetSearch(BaseAutotuner):
            def autotune(self, *, skip_cache=False):
                calls.append(skip_cache)
                return config

        def selector(bound, args):
            return PresetSearch()

        x = torch.ones(16)
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(
                x, static_shapes=False, autotuner_fn=selector, autotune_effort="quick"
            ).autotune((x, 2), force=False)
            with patch.object(
                KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
            ):
                _bound(
                    x,
                    static_shapes=False,
                    autotuner_fn=selector,
                    autotune_effort="quick",
                ).autotune((x, 2), force=False)
        self.assertEqual(calls, [False, False])

    def test_changed_decorator_config_takes_precedence(self) -> None:
        x = torch.ones(16)
        first = helion.Config(block_sizes=[32])
        second = helion.Config(block_sizes=[64])
        selector = Mock(side_effect=AssertionError("explicit config ignored"))
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(x, config=first, autotuner_fn=selector).autotune((x, 2), force=False)
            fresh = _bound(x, config=second, autotuner_fn=selector)
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                self.assertEqual(fresh.autotune((x, 2), force=False), second)
            self.assertEqual(compile_fn.call_count, 1)
        self.assertNotEqual(sources[0], sources[1])

    def test_cached_kernel_call_and_reset(self) -> None:
        x = torch.ones(16)
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(x).set_config(helion.Config(block_sizes=[32]))
            with patch.object(
                KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
            ):
                fresh = _bound(x)
                self.assertIsNone(fresh.kernel(x, 2))
                with self.assertRaisesRegex(TypeError, "missing a required argument"):
                    fresh(x)
                fresh.kernel.reset()
                self.assertIsNone(fresh(x, 2))
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                rebound = fresh.kernel.bind((x, 3))
            self.assertEqual(compile_fn.call_count, 1)
            self.assertIsNot(rebound, fresh)

    @parametrize("damaged", ["binding", "source"])
    def test_broken_binding_artifact_compiles_normally(self, damaged: str) -> None:
        x = torch.ones(16)
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            _bound(x).set_config(helion.Config(block_sizes=[32]))
            directory = self.cache_root / "generated_code"
            path = (
                next((directory / "bindings").glob("*.json"))
                if damaged == "binding"
                else next(directory.glob("*.json"))
            )
            path.write_text("{")
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                fresh = _bound(x)
            self.assertEqual(compile_fn.call_count, 1)
            fresh.set_config(helion.Config(block_sizes=[32]))

    def test_skip_cache_does_not_publish_binding(self) -> None:
        x = torch.ones(16)
        bound = _bound(x)
        sources: list[str] = []
        os.environ["HELION_SKIP_CACHE"] = "1"
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            bound.set_config(helion.Config(block_sizes=[32]))
        self.assertFalse((self.cache_root / "generated_code").exists())

    def test_opaque_dependency_keeps_frontend(self) -> None:
        kernel = helion.kernel(_cached_opaque.fn, generated_code_cache=True)
        self.assertIsNone(cache.exact_input_key(kernel, (torch.ones(16),)))

    def test_global_values_invalidate_frontend_identity(self) -> None:
        def bind():
            kernel = helion.kernel(
                backend="triton", generated_code_cache=True, autotune_effort="none"
            )(_cached_shift.fn)
            return kernel.bind((torch.ones(16),))

        first = bind()
        config = helion.Config(block_sizes=[32])
        key = cache.generated_code_cache_key(
            first, first._normalized_config_copy(config)
        )
        self._test_stack.enter_context(
            patch.dict(_cached_shift.fn.__globals__, {"_SHIFT": 2})
        )
        second = bind()
        other = cache.generated_code_cache_key(
            second, second._normalized_config_copy(config)
        )
        # Even runtime globals conservatively invalidate a pre-frontend artifact.
        self.assertTrue(key is not None and key != other)
        self.assertEqual(first.to_code(config), second.to_code(config))

    @parametrize(
        "change",
        [
            "shape",
            "stride",
            "dtype",
            "scale",
            "config",
            "settings",
            "version",
            "hardware",
        ],
    )
    def test_key_invalidation(self, change: str) -> None:
        config = helion.Config(block_sizes=[32])
        x = torch.ones(16)
        first = _bound(x)
        key = cache.generated_code_cache_key(
            first, first._normalized_config_copy(config)
        )
        settings: dict[str, object] = {}
        scale = 2
        if change == "shape":
            x = torch.ones(32)
        elif change == "stride":
            x = torch.ones(32)[::2]
        elif change == "dtype":
            x = x.to(torch.float64)
        elif change == "scale":
            scale = 3
        elif change == "config":
            config = helion.Config(block_sizes=[64])
        elif change == "settings":
            settings["fast_math"] = True
        elif change == "version":
            self._test_stack.enter_context(
                patch.object(cache, "triton_key_wrapper", lambda: "triton-v2")
            )
        else:
            self._test_stack.enter_context(
                patch.object(cache, "get_device_name", lambda device: "different GPU")
            )
        second = _bound(x, scale, **settings)
        other = cache.generated_code_cache_key(
            second, second._normalized_config_copy(config)
        )
        self.assertTrue(key is not None and other is not None and (key != other))

    @parametrize("skip", ["disabled", "descriptor", "skip_cache"])
    def test_ineligible_configs_use_codegen(self, skip: str) -> None:
        bound = _bound(torch.ones(16))
        if skip == "disabled":
            bound.settings.generated_code_cache = False
        if skip == "skip_cache":
            os.environ["HELION_SKIP_CACHE"] = "1"
        config = bound._normalized_config_copy(helion.Config(block_sizes=[32]))
        if skip == "descriptor":
            config = helion.Config(block_sizes=[32], indexing="tensor_descriptor")
        self.assertIsNone(cache.generated_code_cache_key(bound, config))
        self.assertFalse((self.cache_root / "generated_code").exists())

    @parametrize(
        "contents",
        ["{", "null", "{}", '{"source": 1}', '{"source": "pass", "sha256": "bad"}'],
    )
    def test_corrupt_entry_is_a_miss(self, contents: str) -> None:
        directory = self.cache_root / "generated_code"
        directory.mkdir()
        (directory / "test.json").write_text(contents)
        self.assertIsNone(cache.load_generated_code("test"))
        cache.save_generated_code("test", "def add(x):\n    return x + 1\n")
        self.assertEqual(
            cache.load_generated_code("test"), "def add(x):\n    return x + 1\n"
        )

    def test_atomic_writes_and_failed_publish(self) -> None:
        sources = [f"def f():\n    return {i}\n" for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda source: cache.save_generated_code("shared", source), sources
                )
            )
        self.assertIn(cache.load_generated_code("shared"), sources)
        directory = self.cache_root / "generated_code"
        self.assertEqual(
            sorted(path.name for path in directory.iterdir() if path.name != ".lock"),
            ["shared.json"],
        )
        with patch.object(cache.os, "replace", side_effect=OSError("read-only cache")):
            cache.save_generated_code("shared", "replacement")
        self.assertIn(cache.load_generated_code("shared"), sources)
        self.assertEqual(
            sorted(path.name for path in directory.iterdir() if path.name != ".lock"),
            ["shared.json"],
        )

    @parametrize("hash_seed", ["0", "1"])
    def test_cpu_reload_in_fresh_process(self, hash_seed: str) -> None:
        bound = _bound(torch.ones(16))
        sources: list[str] = []
        with patch.object(
            PyCodeCache, "load", side_effect=_source_loader(self.cache_root, sources)
        ):
            bound.set_config(helion.Config(block_sizes=[32]))
        code = """
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
    setattr(module.cache, name, lambda name=name: {"helion_key": "helion-v1", "torch_key_wrapper": "torch-v1", "triton_key_wrapper": "triton-v1"}[name])
module.cache.supports_torch_compile_fusion = lambda: True
sources = []
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend cache miss")), patch.object(module.PyCodeCache, "load", side_effect=module._source_loader(Path(sys.argv[3]), sources)):
    bound = module._bound(torch.ones(16))
    with patch.object(bound, "to_triton_code", side_effect=AssertionError("codegen cache miss")):
        bound.set_config(module.helion.Config(block_sizes=[32]))
assert module.counters["generated_code_cache"]["hit"] == 1
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                __file__,
                _cached_scale.fn.__module__,
                str(self.cache_root),
            ],
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            timeout=30,
        )


@instantiate_parametrized_tests
class TestGeneratedCodeCacheCUDA(_GeneratedCodeCacheTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), reason="requires CUDA and Triton")
    @parametrize("static_shapes", [True, False])
    def test_cuda_reload_in_fresh_process(self, static_shapes: bool) -> None:
        os.environ["HELION_CACHE_DIR"] = str(self.cache_root)
        os.environ.pop("HELION_SKIP_CACHE", None)
        x = torch.randn(16, device=torch.device("cuda"))
        bound = _bound(x, static_shapes=static_shapes)
        bound.set_config(helion.Config(block_sizes=[32]))
        torch.testing.assert_close(bound(x, 2), x * 2)
        code = """
import importlib.util
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
x = torch.randn(16, device=torch.device("cuda"))
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend cache miss")):
    bound = module._bound(x, static_shapes=sys.argv[3] == "True")
    with patch.object(bound, "to_triton_code", side_effect=AssertionError("codegen cache miss")):
        bound.set_config(module.helion.Config(block_sizes=[32]))
        torch.testing.assert_close(bound(x, 2), x * 2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = bound(x, 2)
        x.fill_(3)
        graph.replay()
        torch.testing.assert_close(out, x * 2)
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                __file__,
                _cached_scale.fn.__module__,
                str(static_shapes),
            ],
            check=True,
            timeout=30,
        )


if __name__ == "__main__":
    unittest.main()
