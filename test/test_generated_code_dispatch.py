from __future__ import annotations

import hashlib
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from test.test_generated_code_binding import _runtime_scale
from test.test_generated_code_cache import _runner

import helion
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner.base_search import BaseAutotuner
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _CachedBoundKernel
from helion.runtime.cached_kernel import _guard_extractor
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache
from helion.runtime.kernel import _GuardedInputFact
from helion.runtime.kernel import _PreparedMetadataSpecializationExtractor

_CONFIG = helion.Config(block_sizes=[32])


def _specialized_rows(x: torch.Tensor, scale: int) -> torch.Tensor:
    hl.specialize(x.size(1))
    scale = hl.specialize(scale)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :] * scale
    return out


def _add_one_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + 1
    return out


class _PresetSearch(BaseAutotuner):
    def autotune(self, *, skip_cache: bool = False) -> helion.Config:
        return helion.Config(block_sizes=[16, 32], indexing="pointer")


def _select(bound: object, args: object) -> _PresetSearch:
    return _PresetSearch()


def _dynamic_2d_kernel(**settings: object) -> Kernel:
    return helion.kernel(
        backend="triton",
        static_shapes=False,
        cache_level="code",
        autotuner_fn=_select,
        **settings,
    )(_add_one_2d)


def _scaled_mm(
    a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor
) -> torch.Tensor:
    # The layout of vLLM's scaled_mm: per-row scales [M, 1], per-column [N, 1].
    m, k = a.shape
    n = b.size(1)
    hl.specialize(k)
    hl.specialize(n)
    out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc = a_scales[tile_m, :] * acc
        acc = b_scales[tile_n, :].T * acc
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def _scaled_mm_kernel() -> Kernel:
    return helion.kernel(
        backend="triton",
        static_shapes=False,
        cache_level="code",
        config=helion.Config(block_sizes=[16, 32, 32]),
    )(_scaled_mm)


def _scaled_mm_check(kernel: Kernel, m: int) -> None:
    device = torch.device("cuda")
    a = torch.randn(m, 64, device=device, dtype=torch.float16)
    b = torch.randn(128, 64, device=device, dtype=torch.float16).t()
    a_scales = torch.rand(m, 1, device=device) + 0.5
    b_scales = torch.rand(128, 1, device=device) + 0.5
    expected = (a.float() @ b.float()) * a_scales * b_scales.T
    torch.testing.assert_close(
        kernel(a, b, a_scales, b_scales).float(), expected, atol=0.1, rtol=0.02
    )


def _source(name: str, prop: TensorProperty, idx: int) -> TensorPropertySource:
    return TensorPropertySource(LocalSource(name, is_input=True), prop, idx)


def _rewrite_schema(key: str, schema: dict[str, list[object]]) -> None:
    cache._save_entry(
        cache._schema_path(key),
        {
            "payload": schema,
            "sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )


@skipIfRefEager("generated source caching requires frontend compilation")
class TestGeneratedCodeDispatch(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = Path(self._test_stack.enter_context(tempfile.TemporaryDirectory()))
        self._test_stack.enter_context(
            patch.dict(
                "os.environ",
                {"HELION_CACHE_DIR": str(self.root), "HELION_SKIP_CACHE": "0"},
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            self._test_stack.enter_context(patch.object(cache, name, return_value=name))
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self.loads = 0
        self._test_stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._load)
        )

    def _load(self, source: str, *, extra: str = "") -> SimpleNamespace:
        self.loads += 1
        path = self.root / f"module_{self.loads}.py"
        path.write_text(source)
        # The CPU tests cover dispatch; the CUDA test launches real source.
        return SimpleNamespace(
            __file__=str(path),
            _runtime_scale=operator.mul,
            _specialized_rows=operator.mul,
        )

    def _kernel(self, fn: object = _runtime_scale, **settings: object) -> Kernel:
        settings.setdefault("config", _CONFIG)
        settings.setdefault("cache_level", "code")
        return helion.kernel(backend="triton", static_shapes=False, **settings)(fn)

    def _frontend_count(self):
        return patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        )

    def test_cache_hit_publishes_fast_dispatch(self) -> None:
        x = torch.ones(16)
        self._kernel()(x, 2.0)
        kernel = self._kernel()
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            torch.testing.assert_close(kernel(x, 2.0), x * 2)
        runner = _runner(kernel, (x, 2.0))
        self.assertIsInstance(runner, _CachedBoundKernel)
        prepared = kernel._prepared_call
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertIs(prepared.bound, runner)
        # Steady-state launches use the prepared call, as native bindings do:
        # no binding, dispatch-key construction or runner facade per launch.
        with (
            patch.object(Kernel, "_bind", side_effect=AssertionError("bind")),
            patch.object(
                Kernel,
                "_fast_dispatch_key_and_guards",
                side_effect=AssertionError("dispatch key"),
            ),
            patch.object(
                _CachedBoundKernel, "__call__", side_effect=AssertionError("facade")
            ),
        ):
            for scale in (2.0, 3.0, 4.0):
                torch.testing.assert_close(kernel(x, scale), x * scale)
            torch.testing.assert_close(prepared.bound._run(x, 5.0), x * 5)
        # Direct calls may validate once, then reuse their own guard.
        torch.testing.assert_close(runner(x, 6.0), x * 6)
        with patch.object(Kernel, "_bind", side_effect=AssertionError("bind")):
            torch.testing.assert_close(runner(x, 7.0), x * 7)

    def test_specialized_runner_uses_prepared_guards(self) -> None:
        x = torch.ones(8, 32)
        self._kernel(_specialized_rows)(x, 2)
        kernel = self._kernel(_specialized_rows)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            torch.testing.assert_close(kernel(x, 2), x * 2)
            torch.testing.assert_close(kernel(x, 2), x * 2)
        runner = _runner(kernel, (x, 2))
        self.assertIsInstance(runner, _CachedBoundKernel)
        with (
            patch.object(Kernel, "_bind", side_effect=AssertionError("bind")),
            patch.object(
                _CachedBoundKernel, "__call__", side_effect=AssertionError("facade")
            ),
        ):
            torch.testing.assert_close(kernel(x, 2), x * 2)
        # Direct calls revalidate value guards once, then use their own guard.
        torch.testing.assert_close(runner(x, 2), x * 2)
        with patch.object(Kernel, "_bind", side_effect=AssertionError("bind")):
            torch.testing.assert_close(runner(x, 2), x * 2)

    def test_warm_dynamic_binding_serves_unseen_shapes(self) -> None:
        self._kernel()(torch.ones(16), 2.0)
        kernel = self._kernel()
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for size in (16, 64, 1000, 3):
                y = torch.ones(size)
                torch.testing.assert_close(kernel(y, 2.0), y * 2)
        self.assertEqual(
            len({id(bound) for bound in kernel._bound_kernels.values()}), 1
        )

    def test_saved_guards_select_bindings_before_frontend(self) -> None:
        self._kernel(_specialized_rows)(torch.ones(8, 32), 2)
        kernel = self._kernel(_specialized_rows)
        with self._frontend_count() as compile_fn:
            # A different dynamic row count matches the saved guards.
            y = torch.ones(100, 32)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 0)
            # New specialized values miss the saved guards and compile once.
            y = torch.ones(8, 64)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            torch.testing.assert_close(
                kernel(torch.ones(4, 64), 2), torch.ones(4, 64) * 2
            )
            self.assertEqual(compile_fn.call_count, 1)
            torch.testing.assert_close(kernel(y, 3), y * 3)
            self.assertEqual(compile_fn.call_count, 2)
        # Each specialization saved its own binding for later processes.
        warm = self._kernel(_specialized_rows)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for shape, scale in (((7, 32), 2), ((9, 64), 2), ((5, 64), 3)):
                y = torch.ones(shape)
                torch.testing.assert_close(warm(y, scale), y * scale)

    def test_runner_guards_match_frontend_extractors(self) -> None:
        x = torch.ones(8, 32)
        self._kernel(_specialized_rows)(x, 2)
        runner = _runner(self._kernel(_specialized_rows), (x, 2))
        self.assertIsInstance(runner, _CachedBoundKernel)
        native = self._kernel(_specialized_rows, cache_level="config").bind((x, 2))
        runner_extractors = runner._specialize_extra()
        native_extractors = native._specialize_extra()
        self.assertEqual(
            [type(extractor) for extractor in runner_extractors],
            [type(extractor) for extractor in native_extractors],
        )
        for args in ((x, 2), (torch.ones(3, 64), 5), (torch.ones(8, 32)[:, ::2], 2)):
            self.assertEqual(
                [extractor(args) for extractor in runner_extractors],
                [extractor(args) for extractor in native_extractors],
            )

    def test_public_bind_returns_binding_while_calls_skip_frontend(self) -> None:
        x = torch.ones(16)
        self._kernel()(x, 2.0)
        kernel = self._kernel()
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for scale in (2.0, 3.0):
                torch.testing.assert_close(kernel(x, scale), x * scale)
            self.assertIsInstance(_runner(kernel, (x, 2.0)), _CachedBoundKernel)
        with self._frontend_count() as compile_fn:
            bound = kernel.bind((x, 2.0))
        # bind() hands out a binding with a frontend, compiled once.
        self.assertIs(type(bound), BoundKernel)
        self.assertEqual(compile_fn.call_count, 1)
        self.assertIsNotNone(bound.host_function)
        with self._frontend_count() as compile_fn:
            self.assertIs(kernel.bind((x, 3.0)), bound)
            self.assertIs(_runner(kernel, (x, 3.0)), bound)
            torch.testing.assert_close(bound(x, 3.0), x * 3)
            torch.testing.assert_close(kernel(x, 4.0), x * 4)
        self.assertEqual(compile_fn.call_count, 0)

    def test_published_schema_guards_must_match_saved_guards(self) -> None:
        # A saved schema whose guards differ from the ones a native binding
        # published for the signature is a miss even when both guard lists
        # evaluate to the same results on the lookup inputs.
        x = torch.ones(32, 32)
        self._kernel(_specialized_rows)(x, 2)
        kernel = self._kernel(_specialized_rows)
        signature = kernel._base_specialization_key((x, 2))
        with self._frontend_count() as compile_fn:
            # A native binding publishes the frontend's guards first.
            y = torch.ones(64, 64)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 1)
            self.assertEqual(
                kernel._published_guards[signature],
                (["size", ["arg", 0], 1], ["arg", 1]),
            )
            # Save x's binding again under guards on x.size(0) instead of
            # x.size(1): the same results for x, a different schema.
            key = cache.compiled_kernel_cache_key(kernel, (x, 2), signature)
            assert key is not None
            schema = cache.load_binding_schema(key)
            assert schema is not None
            schema["guards"] = [["size", ["arg", 0], 0], ["arg", 1]]
            cache._save_entry(
                cache._schema_path(key),
                {
                    "payload": schema,
                    "sha256": hashlib.sha256(
                        json.dumps(schema, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                },
            )
            (native_key,) = kernel._bound_kernels
            cache.save_compiled_kernel(
                cache.binding_variant_key(
                    key, schema, (32, 2), native_key.compiler_seed_results
                ),
                _CONFIG,
                _CONFIG,
                json.loads(
                    next(
                        (self.root / "generated_code" / "bindings").glob("*.json")
                    ).read_text()
                )["payload"]["source_key"],
            )
            # Equal results under different guards do not revive the runner.
            torch.testing.assert_close(kernel(x, 2), x * 2)
            self.assertEqual(compile_fn.call_count, 2)
            self.assertFalse(
                any(bound._is_disk_runner for bound in kernel._bound_kernels.values())
            )
            z = torch.ones(4, 32)
            torch.testing.assert_close(kernel(z, 2), z * 2)
            self.assertEqual(compile_fn.call_count, 2)

    def _guard_facts(self, kernel: Kernel, signature: object) -> list[object]:
        return [
            extractor.extractor
            for extractor in kernel._specialize_extra[signature]
            if isinstance(getattr(extractor, "extractor", None), _GuardedInputFact)
        ]

    def test_size_one_guard_facts_rekey_runners(self) -> None:
        # A static_shapes=False compile bound at a size-1 dim bakes ``size == 1``
        # into its code and publishes it as a guard fact. Disk runners of the
        # signature are re-keyed on their own inputs, as native bindings are on
        # their hints, and keep serving without the frontend.
        x = torch.ones(8, 32)
        one = torch.ones(1, 64)
        self._kernel(_specialized_rows)(x, 2)
        kernel = self._kernel(_specialized_rows)
        signature = kernel._base_specialization_key((x, 2))
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            torch.testing.assert_close(kernel(x, 2), x * 2)
        runner = _runner(kernel, (x, 2))
        self.assertIsInstance(runner, _CachedBoundKernel)
        self.assertEqual(kernel._published_guards[signature], runner._serialized_guards)
        with self._frontend_count() as compile_fn:
            torch.testing.assert_close(kernel(one, 2), one * 2)
            self.assertEqual(compile_fn.call_count, 1)
            self.assertTrue(self._guard_facts(kernel, signature))
            # The serialized facts follow the runner's schema.
            published = kernel._published_guards[signature]
            saved = len(runner._serialized_guards)
            self.assertEqual(published[:saved], runner._serialized_guards)
            self.assertEqual({guard[0] for guard in published[saved:]}, {"fact"})
            self.assertIn(runner, kernel._bound_kernels.values())
            self.assertIsNone(runner._delegate)
            for y in (x, torch.ones(3, 32)):
                torch.testing.assert_close(kernel(y, 2), y * 2)
                self.assertIs(_runner(kernel, (y, 2)), runner)
            self.assertEqual(compile_fn.call_count, 1)
        self.assertIsNone(runner._delegate)
        # The size-1 binding was saved with its fact. x's binding, saved before
        # the fact existed, stays keyed by the schema without it.
        with self._frontend_count() as compile_fn:
            fresh = self._kernel(_specialized_rows)
            for y in (x, one, torch.ones(1, 64), torch.ones(3, 32)):
                torch.testing.assert_close(fresh(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 0)

    def test_runner_guard_facts_match_native_bindings(self) -> None:
        # A runner evaluates guard facts on its captured input metadata; the
        # values equal a native binding's at its bind-time hints for the same
        # inputs, and the facts themselves on those inputs.
        self._kernel(_specialized_rows)(torch.ones(8, 32), 2)
        publisher = self._kernel(_specialized_rows)
        one = torch.ones(1, 64)
        publisher(one, 2)
        facts = self._guard_facts(
            publisher, publisher._base_specialization_key((one, 2))
        )
        self.assertTrue(facts)
        for x in (
            torch.ones(8, 32),
            torch.ones(1, 32),
            torch.ones(5, 64)[:, ::2],
            torch.ones(32, 7).t(),
        ):
            runner = _runner(self._kernel(_specialized_rows), (x, 2))
            self.assertIsInstance(runner, _CachedBoundKernel)
            native = self._kernel(_specialized_rows, cache_level="config").bind((x, 2))
            runner_values = [runner._guard_fact_value(fact) for fact in facts]
            self.assertNotIn(None, runner_values)
            self.assertEqual(
                runner_values, [fact.bound_value(native.env) for fact in facts]
            )
            self.assertEqual(runner_values, [fact((x, 2)) for fact in facts])

    def test_bindings_with_guard_facts_are_saved(self) -> None:
        # The size-1 binding publishes guard facts first; it and a later
        # binding of the same signature are saved, keyed by the facts' values.
        one = torch.ones(1, 64)
        y = torch.ones(8, 64)
        kernel = self._kernel(_specialized_rows)
        signature = kernel._base_specialization_key((one, 2))
        with self._frontend_count() as compile_fn:
            torch.testing.assert_close(kernel(one, 2), one * 2)
            self.assertTrue(self._guard_facts(kernel, signature))
            # y has the size-1 binding's saved guard values but not its fact.
            torch.testing.assert_close(kernel(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 2)
        self.assertEqual(
            len(list((self.root / "generated_code" / "bindings").glob("*.json"))), 2
        )
        # A later process serves each input from its own binding.
        warm = self._kernel(_specialized_rows)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for z in (torch.ones(5, 64), one, y):
                torch.testing.assert_close(warm(z, 2), z * 2)
        # Once a size-1 compile publishes facts, a new runner loads keyed by
        # the facts' values on its inputs.
        warm = self._kernel(_specialized_rows)
        with self._frontend_count() as compile_fn:
            row = torch.ones(1, 32)
            torch.testing.assert_close(warm(row, 2), row * 2)
            self.assertEqual(compile_fn.call_count, 1)
            torch.testing.assert_close(warm(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 1)
            self.assertIsInstance(_runner(warm, (y, 2)), _CachedBoundKernel)
            torch.testing.assert_close(warm(one, 2), one * 2)
            self.assertIsInstance(_runner(warm, (one, 2)), _CachedBoundKernel)
            self.assertEqual(compile_fn.call_count, 1)

    def test_codegen_guard_facts_key_saved_source(self) -> None:
        # Code generation can publish guard facts the frontend did not. The
        # source is saved under a key that includes them, and the binding is
        # saved with them, so later processes only serve inputs that agree.
        x = torch.ones(8, 32)
        kernel = self._kernel(_specialized_rows)
        original = BoundKernel._guarded_input_facts
        rows = sympy.Symbol("rows")
        rows_fact = _PreparedMetadataSpecializationExtractor(
            _GuardedInputFact.create(
                rows,
                sympy.Integer(1),
                {
                    rows: TensorPropertySource(
                        LocalSource("x", is_input=True), TensorProperty.SIZE, 0
                    )
                },
                block=False,
                equal=False,
                param_names=("x", "scale"),
            )
        )
        calls = 0
        keys: list[str | None] = []
        source_key = cache.generated_code_cache_key

        def codegen_fact(bound: BoundKernel) -> list[object]:
            nonlocal calls
            calls += 1
            facts = original(bound)
            # Binding publishes first; every later call is code generation.
            return facts if calls == 1 else [*facts, rows_fact]

        def record_key(bound: BoundKernel, config: helion.Config) -> str | None:
            keys.append(source_key(bound, config))
            return keys[-1]

        with (
            patch.object(BoundKernel, "_guarded_input_facts", codegen_fact),
            patch.object(cache, "generated_code_cache_key", record_key),
        ):
            torch.testing.assert_close(kernel(x, 2), x * 2)
        self.assertGreater(calls, 1)
        # Keyed before code generation, then again with its fact.
        before, after = keys
        self.assertIsNotNone(after)
        self.assertNotEqual(before, after)
        sources = {path.stem for path in (self.root / "generated_code").glob("*.json")}
        self.assertEqual(sources, {after})
        signature = kernel._base_specialization_key((x, 2))
        self.assertEqual(
            kernel._published_guards[signature][-1],
            ["fact", ["size", ["arg", 0], 0], 1, False, False],
        )
        warm = self._kernel(_specialized_rows)
        with self._frontend_count() as compile_fn:
            for y in (x, torch.ones(3, 32)):
                torch.testing.assert_close(warm(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 0)
            # A size-1 input does not have the fact's value the code assumed.
            one = torch.ones(1, 32)
            torch.testing.assert_close(warm(one, 2), one * 2)
            self.assertEqual(compile_fn.call_count, 1)

    def test_guard_fact_serialization_round_trip(self) -> None:
        # Facts on an input size or stride and a constant or another one
        # serialize exactly and rebuild equal to the frontend's, so a runner's
        # facts deduplicate with a native binding's.
        size0, size1, stride0 = sympy.symbols("s0 s1 s2", integer=True)
        names = ("x", "scale")
        facts = [
            _GuardedInputFact.create(
                size1,
                sympy.Integer(1),
                {size1: _source("x", TensorProperty.SIZE, 1)},
                block=False,
                equal=True,
                param_names=names,
            ),
            _GuardedInputFact.create(
                size0,
                sympy.Integer(1),
                {size0: _source("x", TensorProperty.SIZE, 0)},
                block=False,
                equal=False,
                param_names=names,
            ),
            _GuardedInputFact.create(
                size0,
                stride0,
                {
                    size0: _source("x", TensorProperty.SIZE, 1),
                    stride0: _source("x", TensorProperty.STRIDE, 0),
                },
                block=False,
                equal=True,
                param_names=names,
            ),
            _GuardedInputFact.create(
                size1,
                sympy.Integer(32),
                {size1: _source("x", TensorProperty.SIZE, 1)},
                block=True,
                equal=True,
                param_names=names,
            ),
        ]
        arguments = {"x": 0, "scale": 1}
        inputs = (
            torch.ones(8, 1),
            torch.ones(1, 32),
            torch.ones(4, 20),
            torch.ones(3, 3),
        )
        for fact in facts:
            guard = json.loads(json.dumps(cache.serialized_fact(fact, arguments)))
            self.assertEqual(guard[0], "fact")
            self.assertTrue(cache._valid_guard(guard))
            rebuilt = _guard_extractor(guard, names)
            self.assertEqual(rebuilt, _PreparedMetadataSpecializationExtractor(fact))
            for x in inputs:
                self.assertEqual(rebuilt((x, 2)), fact((x, 2)))
        # An expression over sizes has no serialized form.
        self.assertIsNone(
            cache.serialized_fact(
                _GuardedInputFact.create(
                    size0 + size1,
                    sympy.Integer(8),
                    {
                        size0: _source("x", TensorProperty.SIZE, 0),
                        size1: _source("x", TensorProperty.SIZE, 1),
                    },
                    block=False,
                    equal=True,
                    param_names=names,
                ),
                arguments,
            )
        )
        for guard in (
            ["fact", ["size", ["arg", 0], 1], True, True, False],
            ["fact", ["size", ["arg", 0], 1], "1", True, False],
            ["fact", ["size", ["arg", 0], 1], 1, True],
            ["fact", ["size", ["arg", 0], 1], 1, 1, False],
            ["fact", ["item", ["arg", 0], 0], 1, True, False],
            ["fact", ["size", ["item", ["arg", 0], 0], 1], 1, True, False],
            ["fact", 1, ["size", ["arg", 0], 1], True, False],
            ["fact_v2", ["size", ["arg", 0], 1], 1, True, False],
        ):
            self.assertFalse(cache._valid_guard(guard), guard)

    def test_published_schema_facts_must_match_saved_facts(self) -> None:
        # A saved schema whose facts differ from the ones published for the
        # signature is a miss, even when they give the same results.
        one = torch.ones(1, 32)
        x = torch.ones(8, 32)
        cold = self._kernel(_specialized_rows)
        cold(one, 2)
        cold(x, 2)
        kernel = self._kernel(_specialized_rows)
        signature = kernel._base_specialization_key((x, 2))
        key = cache.compiled_kernel_cache_key(kernel, (x, 2), signature)
        assert key is not None
        with self._frontend_count() as compile_fn:
            # A native binding publishes the schema and its facts first.
            y = torch.ones(1, 64)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 1)
            schema = cache.load_binding_schema(key)
            assert schema is not None
            self.assertEqual(
                kernel._published_guards[signature], tuple(schema["guards"])
            )
            fact = schema["guards"][-1]
            self.assertEqual(fact, ["fact", ["size", ["arg", 0], 0], 1, True, False])
            # Save x's binding again under ``x.size(0) == 2``: the same result
            # for x, a different schema.
            schema["guards"][-1] = ["fact", ["size", ["arg", 0], 0], 2, True, False]
            _rewrite_schema(key, schema)
            (native_key,) = kernel._bound_kernels
            cache.save_compiled_kernel(
                cache.binding_variant_key(
                    key, schema, (32, 2, False), native_key.compiler_seed_results
                ),
                _CONFIG,
                _CONFIG,
                json.loads(
                    next(
                        (self.root / "generated_code" / "bindings").glob("*.json")
                    ).read_text()
                )["payload"]["source_key"],
            )
            torch.testing.assert_close(kernel(x, 2), x * 2)
            self.assertEqual(compile_fn.call_count, 2)
            self.assertFalse(
                any(bound._is_disk_runner for bound in kernel._bound_kernels.values())
            )

    def test_unknown_guard_fact_is_a_miss(self) -> None:
        one = torch.ones(1, 32)
        self._kernel(_specialized_rows)(one, 2)
        kernel = self._kernel(_specialized_rows)
        key = cache.compiled_kernel_cache_key(
            kernel, (one, 2), kernel._base_specialization_key((one, 2))
        )
        assert key is not None
        schema = cache.load_binding_schema(key)
        assert schema is not None
        self.assertEqual(schema["guards"][-1][0], "fact")
        schema["guards"][-1] = ["fact", ["size", ["arg", 0], 0], "1", True, False]
        _rewrite_schema(key, schema)
        with (
            self._frontend_count() as compile_fn,
            self.assertLogs(cache.log, level="WARNING") as logs,
        ):
            torch.testing.assert_close(kernel(one, 2), one * 2)
        self.assertEqual(compile_fn.call_count, 1)
        self.assertIn("invalid binding guard", "\n".join(logs.output))

    def test_public_bind_keeps_sibling_runners(self) -> None:
        x = torch.ones(8, 32)
        y = torch.ones(8, 64)
        self._kernel(_specialized_rows)(x, 2)
        self._kernel(_specialized_rows)(y, 2)
        kernel = self._kernel(_specialized_rows)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for z in (x, y):
                torch.testing.assert_close(kernel(z, 2), z * 2)
        sibling = _runner(kernel, (y, 2))
        self.assertIsInstance(sibling, _CachedBoundKernel)
        with self._frontend_count() as compile_fn:
            bound = kernel.bind((x, 2))
            self.assertIs(type(bound), BoundKernel)
            self.assertEqual(compile_fn.call_count, 1)
            # The frontend keyed the guards the runners published, so the
            # sibling runner keeps serving its inputs.
            self.assertIn(sibling, kernel._bound_kernels.values())
            self.assertIs(_runner(kernel, (y, 2)), sibling)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            torch.testing.assert_close(kernel(x, 2), x * 2)
            self.assertIs(_runner(kernel, (x, 2)), bound)
            self.assertEqual(compile_fn.call_count, 1)

    def test_unknown_attributes_do_not_compile(self) -> None:
        x = torch.ones(16)
        self._kernel()(x, 2.0)
        runner = _runner(self._kernel(), (x, 2.0))
        with self._frontend_count() as compile_fn:
            with self.assertRaises(AttributeError):
                runner.not_a_binding_attribute  # noqa: B018
            self.assertFalse(hasattr(runner, "_matches_inputs"))
            self.assertEqual(compile_fn.call_count, 0)
            with self.assertLogs("helion.runtime.cached_kernel", level="DEBUG") as logs:
                self.assertIsNotNone(runner.config_spec)
            self.assertEqual(compile_fn.call_count, 1)
        self.assertIn("config_spec requires the Helion frontend", logs.output[0])

    def test_source_key_ignores_shapes_within_dynamic_bucket(self) -> None:
        # Compiler seed heuristics (``triton_pointwise`` fires wherever the
        # device reports an SM count, including Pallas interpret mode) size
        # their seeds from the first binding's layout and size hints, and seeds
        # enter the config-spec fingerprint as they do for the autotune cache.
        # Disable them so the key reflects only the frontend's specialization.
        def key(x: torch.Tensor) -> str | None:
            bound = self._kernel(disable_autotuner_heuristics=True).bind((x, 2.0))
            return cache.generated_code_cache_key(
                bound, bound._normalized_config_copy(_CONFIG)
            )

        self.assertIsNotNone(key(torch.ones(16)))
        self.assertEqual(key(torch.ones(16)), key(torch.ones(64)[::2]))
        self.assertNotEqual(
            key(torch.ones(16)), key(torch.ones(16, dtype=torch.float64))
        )


@skipIfRefEager("generated source caching requires frontend compilation")
class TestGeneratedCodeDispatchCUDA(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), reason="requires CUDA and Triton")
    def test_warm_process_serves_unseen_shapes_without_frontend(self) -> None:
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self._test_stack.enter_context(
            patch.dict(os.environ, {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"})
        )
        # A custom selector keeps every descriptor guard active, so where the
        # hardware emits them each power-of-two row class is its own binding,
        # as without the cache; elsewhere one dynamic binding serves every m.
        kernel = _dynamic_2d_kernel()
        for m in (4, 16):
            x = torch.randn(m, 64, device=torch.device("cuda"))
            torch.testing.assert_close(kernel(x), x + 1)
        bindings = len(kernel._bound_kernels)
        self.assertIn(bindings, (1, 2))
        code = """
import importlib.util
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
kernel = module._dynamic_2d_kernel()
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend ran")):
    # Unseen row counts within the row classes the cold process saved.
    for m in (5, 7, 17, 31):
        x = torch.randn(m, 64, device=torch.device("cuda"))
        torch.testing.assert_close(kernel(x), x + 1)
        torch.testing.assert_close(kernel(x), x + 1)
        assert kernel._prepared_call is not None
        assert isinstance(kernel._prepared_call.bound, module._CachedBoundKernel)
# The saved bindings cover the same row classes as the cold process.
assert len(kernel._bound_kernels) == int(sys.argv[3]), kernel._bound_kernels
# The public binding is an ordinary one; it compiles the frontend once here.
bound = kernel.bind((x,))
assert type(bound) is module.BoundKernel, type(bound)
torch.testing.assert_close(bound(x), x + 1)
torch.testing.assert_close(kernel(x), x + 1)
with patch.object(module.Kernel, "_bind", side_effect=AssertionError("bind")):
    torch.testing.assert_close(kernel(x), x + 1)
    torch.testing.assert_close(bound(x), x + 1)
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                __file__,
                _add_one_2d.__module__,
                str(bindings),
            ],
            check=True,
            timeout=60,
        )

    @unittest.skipIf(not torch.cuda.is_available(), reason="requires CUDA and Triton")
    def test_scaled_mm_guard_facts_warm_without_frontend(self) -> None:
        # Per-row and per-column scales of shape [M, 1] and [N, 1] fire size
        # guard facts on every compile, and M == 1 more. The bindings are saved
        # with them, and a fresh process serves every M without the frontend.
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self._test_stack.enter_context(
            patch.dict(os.environ, {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"})
        )
        kernel = _scaled_mm_kernel()
        for m in (32, 1):
            _scaled_mm_check(kernel, m)
        published = [guards for guards in kernel._published_guards.values() if guards]
        self.assertTrue(published)
        self.assertTrue(
            any(guard[0] == "fact" for guards in published for guard in guards)
        )
        self.assertTrue(list(Path(root, "generated_code", "bindings").glob("*.json")))
        code = """
import importlib.util
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
kernel = module._scaled_mm_kernel()
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend ran")):
    for m in (1, 2, 5, 32, 33, 64, 1):
        module._scaled_mm_check(kernel, m)
"""
        subprocess.run(
            [sys.executable, "-c", code, __file__, _scaled_mm.__module__],
            check=True,
            timeout=120,
        )

    @unittest.skipIf(not torch.cuda.is_available(), reason="requires CUDA and Triton")
    def test_size_one_compile_keeps_warm_runners(self) -> None:
        # vLLM decode and CUDA graph capture call with M == 1 in a process
        # whose other row counts load from disk. Where descriptor guards give
        # M == 1 its own key, it compiles once and publishes a size-1 guard
        # fact; the loaded runners are re-keyed and keep serving.
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self._test_stack.enter_context(
            patch.dict(os.environ, {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"})
        )
        device = torch.device("cuda")
        cold = _dynamic_2d_kernel()
        for m in (4, 16):
            x = torch.randn(m, 64, device=device)
            torch.testing.assert_close(cold(x), x + 1)
        kernel = _dynamic_2d_kernel()
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            for m in (4, 16):
                x = torch.randn(m, 64, device=device)
                torch.testing.assert_close(kernel(x), x + 1)
            self.assertEqual(compile_fn.call_count, 0)
            runners = [
                bound
                for bound in kernel._bound_kernels.values()
                if bound._is_disk_runner
            ]
            self.assertTrue(runners)
            for _ in range(2):
                one = torch.randn(1, 64, device=device)
                torch.testing.assert_close(kernel(one), one + 1)
            self.assertLessEqual(compile_fn.call_count, 1)
            compiled = compile_fn.call_count
            for runner in runners:
                self.assertIn(runner, kernel._bound_kernels.values())
            for m in (4, 16, 5, 17, 1):
                for _ in range(2):
                    x = torch.randn(m, 64, device=device)
                    torch.testing.assert_close(kernel(x), x + 1)
            self.assertEqual(compile_fn.call_count, compiled)


if __name__ == "__main__":
    unittest.main()
