from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._utils import counters
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _guard_fact
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import PyCodeCache


@helion.kernel(backend="triton")
def _dynamic_source_increment(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


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
            __file__=str(path), _dynamic_source_increment=lambda x: x + 1
        )

    def _kernel(self):
        # The default adaptive autotuner keeps the frontend authoritative;
        # source discovery must not turn its saved config into a winner.
        return helion.kernel(
            backend="triton",
            cache_level="code",
            static_shapes=False,
            autotune_effort="quick",
        )(_dynamic_source_increment.fn)

    def test_adaptive_dynamic_source_hit_restores_codegen_guards(self) -> None:
        config = helion.Config(block_sizes=[32])
        cold = self._kernel()
        bound = cold.bind((torch.ones(16),))
        self.assertIsNotNone(bound._generated_code)
        self.assertIsNone(bound._generated_code.artifact_key)
        bound.set_config(config)
        facts = bound._guard_fact_results()
        self.assertTrue(facts, "the example must record a codegen size guard")
        counters["generated_code_cache"].clear()

        warm = self._kernel()
        value = torch.ones(48)
        loaded = warm.bind((value,))
        self.assertIsInstance(loaded, BoundKernel)
        with patch.object(
            loaded, "to_triton_code", side_effect=AssertionError("source cache miss")
        ):
            loaded.set_config(config)
        self.assertEqual(self.sources, [self.sources[0], self.sources[0]])
        self.assertEqual(counters["generated_code_cache"]["hit"], 1)
        self.assertEqual(counters["generated_code_cache"]["frontend_hit"], 0)
        self.assertEqual(loaded._guard_fact_results(), facts)
        torch.testing.assert_close(loaded(value), value + 1)

        # A singleton no longer satisfies the saved != 1 fact. Restoring the
        # codegen facts must split dispatch before that source can be reused.
        singleton = torch.ones(1)
        other = warm._dispatch_bind((singleton,))
        self.assertIsNot(other, loaded)
        with patch.object(
            other, "to_triton_code", wraps=other.to_triton_code
        ) as render:
            other.set_config(config)
        render.assert_called_once()
        torch.testing.assert_close(other(singleton), singleton + 1)

    def test_source_discovery_does_not_choose_the_config(self) -> None:
        self._kernel().bind((torch.ones(16),)).set_config(
            helion.Config(block_sizes=[32])
        )
        fresh = self._kernel().bind((torch.ones(16),))
        changed = helion.Config(block_sizes=[64])
        with patch.object(
            fresh, "to_triton_code", wraps=fresh.to_triton_code
        ) as render:
            fresh.set_config(changed)
        render.assert_called_once()
        self.assertEqual(fresh._config, changed)
        self.assertNotEqual(self.sources[0], self.sources[1])

    def _cold_source_link(self):
        config = helion.Config(block_sizes=[32])
        bound = self._kernel().bind((torch.ones(16),))
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
        fresh = self._kernel().bind((torch.ones(16),))
        with patch.object(
            fresh, "to_triton_code", wraps=fresh.to_triton_code
        ) as render:
            fresh.set_config(config)
        render.assert_called_once()
        torch.testing.assert_close(fresh(torch.ones(16)), torch.full((16,), 2.0))

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
        changed = helion.Config(block_sizes=[64])
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
        args = (torch.ones(16),)
        bound = self._kernel().bind(args)
        with (
            patch.object(bound, "_specialize_extra", return_value=[lambda args: 1]),
            self.assertRaisesRegex(AssertionError, "guards must match"),
        ):
            cache.GeneratedCodeState.create(bound, args, None, "input-key")
