from __future__ import annotations

import operator
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test.test_generated_code_cache import _GeneratedCodeCacheTestCase

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._utils import counters
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _guard_fact
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import PyCodeCache


@helion.kernel(backend="triton")
def _dynamic_source_add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_m, tile_n]
    return out


def _dynamic_source_kernel():
    return helion.kernel(
        backend="triton",
        cache_level="code",
        static_shapes=False,
        autotune_effort="quick",
    )(_dynamic_source_add_bias.fn)


def _dynamic_source_args(rows: int, *, device=None, broadcast: bool = False):
    return (
        torch.randn(rows, 32, device=device),
        torch.randn(1 if broadcast else rows, 32, device=device),
    )


class TestDynamicGeneratedSource(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        super().setUp()
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(root)
        self._test_stack.enter_context(
            patch.dict(
                "os.environ", {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"}
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            self._test_stack.enter_context(patch.object(cache, name, return_value=name))
        self._test_stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._load)
        )
        self.sources: list[str] = []
        counters["generated_code_cache"].clear()

    def _load(self, source: str, *, extra: str = "") -> SimpleNamespace:
        self.sources.append(source)
        path = self.root / f"module_{len(self.sources)}.py"
        path.write_text(source)
        return SimpleNamespace(
            __file__=str(path), _dynamic_source_add_bias=operator.add
        )

    def _kernel(self):
        # The default adaptive autotuner keeps the frontend authoritative;
        # source discovery must not turn its saved config into a winner.
        return _dynamic_source_kernel()

    def test_adaptive_dynamic_source_hit_restores_codegen_guards(self) -> None:
        config = helion.Config(block_sizes=[8, 32])
        # These unseen row counts share the frontend's capped descriptor
        # extent class. Different frontend guard classes must remain misses.
        cold = self._kernel()
        bound = cold.bind(_dynamic_source_args(64))
        self.assertIsNotNone(bound._generated_code)
        self.assertIsNone(bound._generated_code.artifact_key)
        bound.set_config(config)
        facts = bound._guard_fact_results()
        self.assertTrue(facts, "the example must record a codegen size guard")
        counters["generated_code_cache"].clear()

        warm = self._kernel()
        args = _dynamic_source_args(96)
        loaded = warm.bind(args)
        self.assertIsInstance(loaded, BoundKernel)
        with patch.object(
            loaded, "to_triton_code", side_effect=AssertionError("source cache miss")
        ):
            loaded.set_config(config)
        self.assertEqual(self.sources, [self.sources[0], self.sources[0]])
        self.assertEqual(counters["generated_code_cache"]["hit"], 1)
        self.assertEqual(counters["generated_code_cache"]["frontend_hit"], 0)
        self.assertEqual(loaded._guard_fact_results(), facts)
        torch.testing.assert_close(loaded(*args), args[0] + args[1])

        # Singleton bias violates the saved broadcast facts and must receive
        # a separate binding before any source can be reused.
        singleton = _dynamic_source_args(96, broadcast=True)
        other = warm._dispatch_bind(singleton)
        self.assertIsNot(other, loaded)
        with patch.object(
            other, "to_triton_code", wraps=other.to_triton_code
        ) as render:
            other.set_config(config)
        render.assert_called_once()
        torch.testing.assert_close(other(*singleton), singleton[0] + singleton[1])

    def test_source_discovery_does_not_choose_the_config(self) -> None:
        self._kernel().bind(_dynamic_source_args(64)).set_config(
            helion.Config(block_sizes=[8, 32])
        )
        fresh = self._kernel().bind(_dynamic_source_args(64))
        changed = helion.Config(block_sizes=[16, 32])
        with patch.object(
            fresh, "to_triton_code", wraps=fresh.to_triton_code
        ) as render:
            fresh.set_config(changed)
        render.assert_called_once()
        self.assertEqual(fresh._config, changed)
        self.assertNotEqual(self.sources[0], self.sources[1])

    def _cold_source_link(self):
        config = helion.Config(block_sizes=[8, 32])
        bound = self._kernel().bind(_dynamic_source_args(64))
        bound.set_config(config)
        normalized = bound._normalized_config_copy(config)
        lookup_key = cache._generated_code_cache_key(
            bound, normalized, include_guard_facts=False
        )
        self.assertIsNotNone(lookup_key)
        schema = cache.load_binding_schema(lookup_key)
        self.assertIsNotNone(schema)
        params = tuple(bound.kernel.signature.parameters)
        results = tuple(
            bound._guard_fact_value(_guard_fact(guard, params))
            for guard in schema["guards"]
        )
        variant_key = cache.binding_variant_key(lookup_key, schema, results, ())
        artifact = cache._load_compiled_kernel_entry(variant_key)
        self.assertIsNotNone(artifact)
        return config, normalized, lookup_key, schema, variant_key, artifact[2]

    def _assert_source_fallback(self, config: helion.Config) -> None:
        args = _dynamic_source_args(64)
        fresh = self._kernel().bind(args)
        with patch.object(
            fresh, "to_triton_code", wraps=fresh.to_triton_code
        ) as render:
            fresh.set_config(config)
        render.assert_called_once()
        torch.testing.assert_close(fresh(*args), args[0] + args[1])

    def test_out_of_range_discovery_guard_falls_back(self) -> None:
        config, _, lookup_key, _, _, _ = self._cold_source_link()
        cache.save_binding_schema(
            lookup_key,
            {
                "guards": [["fact", ["size", ["arg", 99], 0], 1, False, False]],
                "compiler_seed_facts": [],
            },
        )
        self._assert_source_fallback(config)

    def test_discovery_config_mismatch_falls_back(self) -> None:
        config, _, _, _, variant_key, source_key = self._cold_source_link()
        changed = helion.Config(block_sizes=[16, 32])
        cache.save_compiled_kernel(variant_key, changed, changed, source_key)
        self._assert_source_fallback(config)

    def test_discovery_source_identity_mismatch_falls_back(self) -> None:
        config, normalized, _, _, variant_key, _ = self._cold_source_link()
        other_key = "0" * 64
        cache.save_generated_code(other_key, "# unrelated cached source")
        cache.save_compiled_kernel(variant_key, normalized, normalized, other_key)
        self._assert_source_fallback(config)
        self.assertNotIn("# unrelated cached source", self.sources)

    def test_partially_evicted_discovery_entry_falls_back(self) -> None:
        config, _, _, _, _, source_key = self._cold_source_link()
        cache._cache_path(source_key).unlink()
        self._assert_source_fallback(config)

    def test_guard_serialization_requires_one_source_per_extractor(self) -> None:
        args = _dynamic_source_args(64)
        bound = self._kernel().bind(args)
        with (
            patch.object(bound, "_specialize_extra", return_value=[lambda args: 1]),
            self.assertRaisesRegex(AssertionError, "guards must match"),
        ):
            cache.GeneratedCodeState.create(bound, args, None, "input-key")


class TestDynamicGeneratedSourceCUDA(_GeneratedCodeCacheTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), reason="requires CUDA and Triton")
    def test_source_discovery_runs_real_triton_and_separates_singletons(self) -> None:
        config = helion.Config(block_sizes=[8, 32])
        cold = _dynamic_source_kernel()
        args = _dynamic_source_args(64, device=DEVICE)
        bound = cold.bind(args)
        bound.set_config(config)
        torch.testing.assert_close(bound(*args), args[0] + args[1])
        self.assertTrue(bound._guard_fact_results())

        warm = _dynamic_source_kernel()
        unseen = _dynamic_source_args(96, device=DEVICE)
        loaded = warm.bind(unseen)
        self.assertIsInstance(loaded, BoundKernel)
        self.assertIsNotNone(loaded._generated_code)
        self.assertIsNone(loaded._generated_code.artifact_key)
        with patch.object(
            loaded, "to_triton_code", side_effect=AssertionError("source cache miss")
        ):
            loaded.set_config(config)
            torch.testing.assert_close(loaded(*unseen), unseen[0] + unseen[1])

        singleton = _dynamic_source_args(96, device=DEVICE, broadcast=True)
        other = warm._dispatch_bind(singleton)
        self.assertIsNot(other, loaded)
        with patch.object(
            other, "to_triton_code", wraps=other.to_triton_code
        ) as render:
            other.set_config(config)
        render.assert_called_once()
        torch.testing.assert_close(other(*singleton), singleton[0] + singleton[1])
        # The singleton's new guard schema also keeps the loaded large-input
        # binding valid when eager dispatch returns to that bucket.
        torch.testing.assert_close(warm(*unseen), unseen[0] + unseen[1])
