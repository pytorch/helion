from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.llm.prompting import build_author_seed_section
from helion.autotuner.llm.prompting import build_compiler_analysis_section
from helion.autotuner.llm.workload import compute_workload_hints
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.settings import Settings

examples_dir = Path(__file__).parent.parent / "examples"


def _get_examples_matmul():
    """Lazy accessor to avoid CUDA init during pytest-xdist collection."""
    return import_path(examples_dir / "matmul.py").matmul


class TestLLMGuidedSearch(TestCase):
    """Tests for LLMGuidedSearch config parsing and utility methods."""

    @classmethod
    def _make_mock_search(cls, **overrides):
        """Create a minimal LLMGuidedSearch-like object for testing internal methods."""
        from helion.autotuner.llm_search import LLMGuidedSearch

        search = LLMGuidedSearch.__new__(LLMGuidedSearch)
        search.settings = Settings()
        search.log = AutotuningLogger(search.settings)
        search._messages = []
        search._all_benchmark_results = []
        search._latest_results_by_config_key = {}
        search.configs_per_round = 15
        search._llm_call_times = []
        search._benchmark_times = []
        search.model = "gpt-5.5"
        search.api_base = None
        search.api_key = None
        search.request_timeout_s = 120.0
        search.effort_level = None
        search.fast_mode = False

        # Mock config_spec with a normalize that accepts anything.
        search.config_spec = SimpleNamespace(
            normalize=lambda raw, _fix_invalid=False: None,
            default_config=lambda: helion.Config(block_sizes=[64]),
            _flat_fields=dict,
        )
        search._default_config_dict = dict(search.config_spec.default_config())
        for name, value in overrides.items():
            setattr(search, name, value)
        return search

    def test_parse_configs_accepts_common_llm_outputs(self):
        """LLM config parsing accepts the response shapes we expect to see in practice."""
        import json

        from helion.autotuner.llm_search import LLMGuidedSearch

        cases = [
            (
                "structured",
                json.dumps(
                    {"configs": [{"block_sizes": [64]}, {"block_sizes": [128]}]}
                ),
                2,
            ),
            (
                "python-literals",
                '{"configs": [{"block_sizes": [64], "maxnreg": None, "flag": True}]}',
                1,
            ),
            (
                "deduplicates",
                json.dumps({"configs": [{"block_sizes": [64]}, {"block_sizes": [64]}]}),
                1,
            ),
            ("malformed", "not json at all", 0),
        ]

        for name, response, expected in cases:
            with self.subTest(name=name):
                search = self._make_mock_search()
                configs = LLMGuidedSearch._parse_configs(search, response)
                self.assertEqual(len(configs), expected)

    def test_parse_configs_rejects_shape_guesses(self):
        """LLM config parsing rejects guessed scalar/list shapes instead of repairing them."""
        from helion.autotuner.llm_search import LLMGuidedSearch

        search = self._make_mock_search(
            config_spec=SimpleNamespace(
                normalize=lambda raw, _fix_invalid=False: None,
                default_config=lambda: helion.Config(block_sizes=[64, 64], num_warps=4),
                _flat_fields=lambda: {
                    "block_sizes": ListOf(IntegerFragment(1, 256, 64), length=2),
                    "num_stages": IntegerFragment(1, 8, 2),
                    "num_warps": PowerOfTwoFragment(1, 32, 4),
                },
            ),
        )
        search._default_config_dict = dict(search.config_spec.default_config())

        cases = [
            ("non-power-of-two num_warps", '{"configs": [{"num_warps": 6}]}', 0),
            ("scalar block_sizes", '{"configs": [{"block_sizes": 64}]}', 0),
            ("list scalar field", '{"configs": [{"num_stages": [2]}]}', 0),
            (
                "well-shaped config",
                '{"configs": [{"block_sizes": [64, 128], "num_warps": 8, "num_stages": 2}]}',
                1,
            ),
        ]

        for name, response, expected in cases:
            with self.subTest(name=name):
                configs = LLMGuidedSearch._parse_configs(search, response)
                self.assertEqual(len(configs), expected)

    def test_context_window_keeps_prefix_and_recent_history(self):
        """Prompt context always keeps the fixed prefix and trims only old round history."""
        from helion.autotuner.llm_search import _MAX_CONTEXT_ROUNDS
        from helion.autotuner.llm_search import LLMGuidedSearch

        for short_history in (False, True):
            with self.subTest(short_history=short_history):
                search = self._make_mock_search()
                search._messages = [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "initial"},
                ]
                if short_history:
                    search._messages.append(
                        {"role": "assistant", "content": "response"}
                    )
                    expected_len = 3
                else:
                    for i in range(10):
                        search._messages.append(
                            {"role": "user", "content": f"round {i}"}
                        )
                        search._messages.append(
                            {"role": "assistant", "content": f"resp {i}"}
                        )
                    expected_len = 2 + _MAX_CONTEXT_ROUNDS * 2

                context = LLMGuidedSearch._get_context_messages(search)
                self.assertEqual(context[0]["content"], "system")
                self.assertEqual(context[1]["content"], "initial")
                self.assertEqual(len(context), expected_len)

    @onlyBackends(["triton", "cute"])
    def test_autotune_runs_full_llm_guided_loop_with_mocked_provider(self):
        """LLM-guided search runs the public round loop with mocked LLM and benchmark backends."""
        import concurrent.futures
        import json

        from helion.autotuner.base_search import BenchmarkResult
        from helion.autotuner.llm_search import LLMGuidedSearch

        class FakeBenchmarkProvider:
            def __init__(
                self,
                *,
                kernel,
                settings,
                config_spec,
                args,
                log,
                autotune_metrics,
            ) -> None:
                del kernel, settings, config_spec, args, log, autotune_metrics
                self.mutated_arg_indices: list[int] = []
                self.setup_called = False
                self.cleanup_called = False

            def setup(self) -> None:
                self.setup_called = True

            def cleanup(self) -> None:
                self.cleanup_called = True

            def set_budget_exceeded_fn(self, fn) -> None:
                pass

        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float16),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float16),
        )
        bound = _get_examples_matmul().bind(args)
        default_config = bound.config_spec.default_config()
        search = LLMGuidedSearch(
            bound,
            args,
            configs_per_round=2,
            max_rounds=3,
            initial_random_configs=0,
        )
        search._benchmark_provider_cls = FakeBenchmarkProvider
        search._default_config_dict = dict(default_config)

        def sparse_config(cfg: helion.Config) -> dict[str, object]:
            return {
                key: value
                for key, value in dict(cfg).items()
                if key not in default_config or value != default_config[key]
            }

        def collect_candidate_configs(count: int) -> list[helion.Config]:
            candidates: list[helion.Config] = []
            seen = {repr(default_config)}
            raw_candidates = [
                {"num_warps": 8},
                {"num_stages": 2},
                {"pid_type": "xyz"},
                {"num_warps": 8, "num_stages": 2},
                {"num_warps": 8, "pid_type": "xyz"},
            ]
            for raw in raw_candidates:
                parsed = search._parse_configs(json.dumps({"configs": [raw]}))
                if not parsed:
                    continue
                candidate = parsed[0]
                key = repr(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) == count:
                    return candidates
            raise AssertionError(
                f"Could not find {count} valid non-default configs for matmul"
            )

        round0_cfg, round1_cfg = collect_candidate_configs(2)
        default_key = repr(default_config)
        round0_key = repr(round0_cfg)
        round1_key = repr(round1_cfg)
        round0_sparse = sparse_config(round0_cfg)
        round1_sparse = sparse_config(round1_cfg)

        llm_requests: list[list[dict[str, str]]] = []
        benchmark_batches: list[tuple[str, list[dict[str, object]]]] = []
        rebenchmark_descs: list[str] = []
        async_request_count = 0
        sync_request_count = 0
        llm_responses = iter(
            [
                json.dumps({"configs": [{}, round0_sparse]}),
                json.dumps({"configs": [round0_sparse, round1_sparse]}),
                json.dumps({"configs": [round1_sparse]}),
            ]
        )
        perf_by_key = {
            default_key: 10.0,
            round0_key: 5.0,
            round1_key: 3.0,
        }
        rebench_perf_by_key = {
            default_key: 9.5,
            round0_key: 4.5,
            round1_key: 2.5,
        }

        def fake_call_llm_async(
            self, messages: list[dict[str, str]]
        ) -> concurrent.futures.Future[str]:
            del self
            nonlocal async_request_count
            async_request_count += 1
            llm_requests.append([dict(message) for message in messages])
            future: concurrent.futures.Future[str] = concurrent.futures.Future()
            future.set_result(next(llm_responses))
            return future

        def fake_call_llm(self, messages: list[dict[str, str]]) -> str:
            del self
            nonlocal sync_request_count
            sync_request_count += 1
            llm_requests.append([dict(message) for message in messages])
            return next(llm_responses)

        def fake_benchmark_batch(
            self, configs: list[helion.Config], *, desc: str = "Benchmarking"
        ) -> list[BenchmarkResult]:
            batch_keys = [repr(config) for config in configs]
            benchmark_batches.append(
                (desc, [sparse_config(config) for config in configs])
            )

            results: list[BenchmarkResult] = []
            for config, key in zip(configs, batch_keys, strict=True):
                perf = perf_by_key[key]
                self.best_perf_so_far = min(self.best_perf_so_far, perf)
                self._autotune_metrics.num_configs_tested += 1
                results.append(
                    BenchmarkResult(
                        config=config,
                        fn=lambda *args, **kwargs: None,
                        perf=perf,
                        status="ok",
                        compile_time=0.01,
                    )
                )
            return results

        def fake_rebenchmark_population(
            self,
            members=None,
            *,
            desc="Rebenchmarking",
            target_ms=200.0,
            use_isolated=True,
            confirm_suspicious=True,
            use_interleaved=True,
        ):
            rebenchmark_descs.append(desc)
            for member in self.population:
                member.perfs.append(rebench_perf_by_key[repr(member.config)])

        def fake_rebenchmark(
            self,
            members,
            *,
            desc="Rebenchmarking",
            target_ms=200.0,
            use_isolated=True,
            confirm_suspicious=True,
            use_interleaved=True,
        ):
            rebenchmark_descs.append(desc)
            for member in members:
                member.perfs.append(rebench_perf_by_key[repr(member.config)])

        with (
            patch.object(
                LLMGuidedSearch,
                "_call_llm_async",
                autospec=True,
                side_effect=fake_call_llm_async,
            ),
            patch.object(
                LLMGuidedSearch,
                "_call_llm",
                autospec=True,
                side_effect=fake_call_llm,
            ),
            patch.object(
                LLMGuidedSearch,
                "benchmark_batch",
                autospec=True,
                side_effect=fake_benchmark_batch,
            ),
            patch.object(
                LLMGuidedSearch,
                "rebenchmark_population",
                autospec=True,
                side_effect=fake_rebenchmark_population,
            ),
            patch.object(
                LLMGuidedSearch,
                "rebenchmark",
                autospec=True,
                side_effect=fake_rebenchmark,
            ),
        ):
            best = search.autotune(skip_cache=True)

        self.assertEqual(dict(best), dict(round1_cfg))
        self.assertTrue(search.benchmark_provider.setup_called)
        self.assertTrue(search.benchmark_provider.cleanup_called)
        self.assertEqual(len(llm_requests), 3)
        self.assertEqual(async_request_count, 1)
        self.assertEqual(sync_request_count, 2)
        self.assertEqual([len(messages) for messages in llm_requests], [2, 4, 6])
        self.assertIn("4.5000 ms", llm_requests[1][-1]["content"])
        self.assertIn("2.5000 ms", llm_requests[2][-1]["content"])
        self.assertEqual(
            [desc for desc, _ in benchmark_batches],
            ["Round 0 seed", "Round 0 LLM", "Round 1"],
        )
        self.assertEqual(benchmark_batches[0][1], [sparse_config(default_config)])
        self.assertEqual(benchmark_batches[1][1], [round0_sparse])
        self.assertEqual(benchmark_batches[2][1], [round1_sparse])
        self.assertEqual(
            rebenchmark_descs,
            [
                "Round 0: verifying top configs",
                "Round 1: verifying top configs",
                "Final verification top 3 configs",
            ],
        )
        self.assertEqual(search._autotune_metrics.num_configs_tested, 3)
        self.assertEqual(len(search._all_benchmark_results), 3)
        self.assertEqual(len(search.population), 3)
        self.assertEqual(len(search._benchmarked_members), 3)
        self.assertEqual(search.best.perf, 2.5)
        self.assertEqual(dict(best), dict(round1_cfg))
        self.assertEqual(dict(search.best.config), dict(round1_cfg))

    def test_initial_llm_response_propagates_failure(self):
        """Round 0 LLM future failures are not silently swallowed."""
        import concurrent.futures

        search = self._make_mock_search()
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        future.set_exception(RuntimeError("simulated provider 401"))
        with self.assertRaisesRegex(RuntimeError, "simulated provider 401"):
            search._wait_for_initial_llm_response(future)


class TestAuthorSeedConfigPrompt(TestCase):
    """Author-provided autotune_seed_configs surface in the initial prompt."""

    def test_renders_all_seed_config_values(self):
        kernel = SimpleNamespace(
            settings=Settings(
                autotune_seed_configs=[
                    helion.Config(block_sizes=[1, 1]),
                    helion.Config(block_sizes=[16, 16], num_warps=8),
                ]
            )
        )
        section = build_author_seed_section(kernel)
        self.assertIn("Author Seed Configs", section)
        self.assertIn('"block_sizes":[1,1]', section)
        self.assertIn('"block_sizes":[16,16]', section)
        self.assertIn('"num_warps":8', section)

    def test_empty_when_no_author_seeds(self):
        kernel = SimpleNamespace(settings=Settings())
        self.assertEqual(build_author_seed_section(kernel), "")


class TestCompilerAnalysisPrompt(TestCase):
    """Compiler-fired heuristics + analytical seed configs surface in the prompt."""

    def test_renders_heuristic_name_purpose_and_seed(self):
        config_spec = SimpleNamespace(
            autotuner_heuristics=["triton_skinny_gemm"],
            compiler_seed_configs=[helion.Config(block_sizes=[64, 64, 256])],
        )
        section = build_compiler_analysis_section(config_spec)
        self.assertIn("Compiler Analysis", section)
        self.assertIn("triton_skinny_gemm", section)
        self.assertIn("skinny GEMM", section)
        self.assertIn('"block_sizes":[64,64,256]', section)
        self.assertIn("structural prior", section.lower())

    def test_empty_when_nothing_fired(self):
        config_spec = SimpleNamespace(autotuner_heuristics=[], compiler_seed_configs=[])
        self.assertEqual(build_compiler_analysis_section(config_spec), "")


def _meta_tensor(shape: list[int]) -> torch.Tensor:
    """Shape/dtype-only tensor for workload-hint tests (no real device needed)."""
    return torch.randn(shape, device="meta", dtype=torch.float16)  # @ignore-device-lint


class TestReductionWorkloadHints(TestCase):
    """Pure (non-attention, non-matmul) reductions get cache-eviction guidance."""

    def test_reduction_gets_eviction_hint_family_general(self):
        args = (_meta_tensor([1024, 1024]),)
        hints = compute_workload_hints(args, workload_traits=frozenset({"reduction"}))
        self.assertIn("load_eviction_policies", hints)
        self.assertIn("last", hints.lower())
        # Family-general: no specific kernel name or shape.
        self.assertNotIn("softmax", hints.lower())
        self.assertNotIn("1024", hints)

    def test_not_emitted_for_matmul_attention_or_non_reduction(self):
        sentence = "memory-bound reduction/normalization that streams"
        for traits in (
            frozenset({"matmul", "reduction"}),
            frozenset({"matmul", "reduction", "attention_reduction"}),
            frozenset(),
        ):
            with self.subTest(traits=sorted(traits)):
                args = (
                    _meta_tensor([1024, 1024]),
                    _meta_tensor([1024, 1024]),
                )
                hints = compute_workload_hints(args, workload_traits=traits)
                self.assertNotIn(sentence, hints)


class TestLargeMatmulWorkloadHints(TestCase):
    """Large standalone 2D GEMMs get coherent flat + small-K guidance.

    The default matmul guidance is counterproductive for a large GEMM (it pushes
    persistent and caps K-tile at 64); the large branch suppresses it and
    recommends flat pid + K-tile 32. Double-gated to a rank-2 pure-matmul
    workload so it cannot leak to fused or batched (SSM-style) kernels.
    """

    _PERSISTENT_PUSH = "try pid_type='persistent_blocked' with l2_groupings"
    _DEFAULT_START_TILE = "as starting point"

    @staticmethod
    def _hints(shapes, traits=frozenset({"matmul"})):
        args = tuple(_meta_tensor(s) for s in shapes)
        return compute_workload_hints(args, workload_traits=frozenset(traits))

    def test_large_2d_gemm_gets_flat_k32_and_suppresses_default(self):
        hints = self._hints([[4096, 4096], [4096, 4096]])
        self.assertIn("large compute-bound GEMM", hints)
        self.assertIn("K-tile 32", hints)
        self.assertIn("pid_type='flat'", hints)
        self.assertNotIn(self._PERSISTENT_PUSH, hints)
        self.assertNotIn(self._DEFAULT_START_TILE, hints)

    def test_medium_gemm_keeps_default_guidance(self):
        hints = self._hints([[2048, 2048], [2048, 2048]])
        self.assertNotIn("large compute-bound GEMM", hints)
        self.assertIn(self._DEFAULT_START_TILE, hints)

    def test_does_not_leak_to_fused_batched_or_non_matmul(self):
        # Fused matmul+reduction, batched (rank-3) operands, and non-matmul
        # workloads must never get the standalone large-GEMM hint.
        cases = (
            ([[4096, 4096], [4096, 4096]], frozenset({"matmul", "reduction"})),
            ([[8, 4096, 4096], [8, 4096, 4096]], frozenset({"matmul"})),
            ([[4096, 4096]], frozenset()),
        )
        for shapes, traits in cases:
            with self.subTest(traits=sorted(traits), rank=len(shapes[0])):
                self.assertNotIn(
                    "large compute-bound GEMM", self._hints(shapes, traits)
                )


class TestLLMTransport(TestCase):
    """Tests for provider selection and HTTP payload translation."""

    @staticmethod
    def _response_payload(
        provider: str, text: str = '{"configs": []}'
    ) -> dict[str, object]:
        if provider == "openai_responses":
            return {"output": [{"content": [{"type": "text", "text": text}]}]}
        return {"content": [{"type": "text", "text": text}]}

    def test_vertex_provider_request_shape(self):
        """The vertex provider posts a model-less Anthropic body to the Vertex
        publisher rawPredict URL, and extra headers parse from JSON / lines."""
        from helion.autotuner.llm.transport import _extra_headers
        from helion.autotuner.llm.transport import call_provider
        from helion.autotuner.llm.transport import infer_provider
        from helion.autotuner.llm.transport import normalize_provider

        self.assertEqual(normalize_provider("vertex"), "vertex")
        self.assertEqual(infer_provider("vertex/claude-sonnet-4-5"), "vertex")

        captured = {}

        def fake_post_json(url, payload, headers, *, request_timeout_s):
            del request_timeout_s
            captured.update(url=url, payload=payload, headers=headers)
            return self._response_payload("anthropic")

        env = {
            "HELION_LLM_VERTEX_PROJECT": "proj-123",
            "HELION_LLM_VERTEX_LOCATION": "us-east5",
            "HELION_LLM_EXTRA_HEADERS": '{"X-Gateway": "abc"}',
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "helion.autotuner.llm.transport._post_json",
                side_effect=fake_post_json,
            ):
                text = call_provider(
                    "vertex",
                    model="vertex/claude-sonnet-4-5",
                    api_base="https://gw.example/v1",
                    api_key="tok",
                    messages=[{"role": "user", "content": "suggest configs"}],
                    max_output_tokens=256,
                    request_timeout_s=60.0,
                )
            self.assertEqual(text, '{"configs": []}')
            self.assertEqual(
                captured["url"],
                "https://gw.example/v1/projects/proj-123/locations/us-east5"
                "/publishers/anthropic/models/claude-sonnet-4-5:rawPredict",
            )
            self.assertNotIn("model", captured["payload"])
            self.assertEqual(
                captured["payload"]["anthropic_version"], "vertex-2023-10-16"
            )
            self.assertEqual(captured["headers"]["Authorization"], "Bearer tok")
            self.assertEqual(_extra_headers(), {"X-Gateway": "abc"})

        with patch.dict(
            os.environ,
            {"HELION_LLM_EXTRA_HEADERS": "X-One: 1\nX-Two: two"},
            clear=False,
        ):
            self.assertEqual(_extra_headers(), {"X-One": "1", "X-Two": "two"})

    def test_vertex_provider_standard_env_fallback(self):
        """With the helion-specific knobs unset, the vertex provider resolves the
        endpoint/project/location from the standard Anthropic-on-Vertex SDK env
        vars (so a pre-configured environment needs no extra setup)."""
        from helion.autotuner.llm.transport import call_provider

        captured = {}

        def fake_post_json(url, payload, headers, *, request_timeout_s):
            captured.update(url=url, payload=payload, headers=headers)
            return self._response_payload("anthropic")

        env = {
            "ANTHROPIC_VERTEX_BASE_URL": "https://std.example/v1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "std-proj",
            "CLOUD_ML_REGION": "global",
        }
        with patch.dict(os.environ, env, clear=False):
            for stale in (
                "HELION_LLM_API_BASE",
                "HELION_LLM_VERTEX_PROJECT",
                "HELION_LLM_VERTEX_LOCATION",
            ):
                os.environ.pop(stale, None)
            with patch(
                "helion.autotuner.llm.transport._post_json",
                side_effect=fake_post_json,
            ):
                call_provider(
                    "vertex",
                    model="vertex/claude-sonnet-4-5",
                    api_base=None,
                    api_key=None,
                    messages=[{"role": "user", "content": "suggest configs"}],
                    max_output_tokens=256,
                    request_timeout_s=60.0,
                )
            self.assertEqual(
                captured["url"],
                "https://std.example/v1/projects/std-proj/locations/global"
                "/publishers/anthropic/models/claude-sonnet-4-5:rawPredict",
            )

    def test_transport_helpers_and_http_request_shapes(self):
        """Provider helpers build the expected request/response shapes for each backend."""
        from helion.autotuner.llm.transport import _resolve_api_base
        from helion.autotuner.llm.transport import _resolve_api_key
        from helion.autotuner.llm.transport import _resolve_v1_endpoint
        from helion.autotuner.llm.transport import call_provider
        from helion.autotuner.llm.transport import extract_anthropic_text
        from helion.autotuner.llm.transport import extract_openai_response_text
        from helion.autotuner.llm.transport import infer_provider
        from helion.autotuner.llm.transport import normalize_effort_level
        from helion.autotuner.llm.transport import responses_input_from_messages
        from helion.autotuner.llm.transport import split_system_messages

        self.assertEqual(infer_provider("claude-opus-4-7"), "anthropic")
        self.assertEqual(infer_provider("gpt-5.5"), "openai_responses")
        self.assertEqual(infer_provider("custom/model"), "unsupported")
        self.assertEqual(infer_provider("custom/model", "anthropic"), "anthropic")
        self.assertEqual(infer_provider("custom/model", "openai"), "openai_responses")
        self.assertEqual(normalize_effort_level("MAX"), "max")
        self.assertEqual(normalize_effort_level("off"), "none")
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            infer_provider("gpt-5.5", "bogus")
        with self.assertRaisesRegex(ValueError, "Unsupported LLM effort level"):
            normalize_effort_level("bogus")

        self.assertEqual(
            _resolve_v1_endpoint("https://api.openai.com", "responses"),
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(
            _resolve_v1_endpoint("https://api.openai.com/v1", "responses"),
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(
            _resolve_v1_endpoint("https://proxy.example/v1/messages", "messages"),
            "https://proxy.example/v1/messages",
        )
        self.assertEqual(
            _resolve_api_base("openai_responses", "https://explicit.example/v1"),
            "https://explicit.example/v1",
        )
        self.assertEqual(
            _resolve_api_key("openai_responses", "explicit-key"), "explicit-key"
        )

        system, history = split_system_messages(
            [
                {"role": "system", "content": "tune kernels"},
                {"role": "user", "content": "suggest configs"},
                {"role": "assistant", "content": '{"configs": []}'},
            ]
        )
        self.assertEqual(system, "tune kernels")
        self.assertEqual(
            [message["role"] for message in history], ["user", "assistant"]
        )

        payload = responses_input_from_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        )
        self.assertEqual(
            payload[0]["content"], [{"type": "input_text", "text": "hello"}]
        )
        self.assertEqual(
            payload[1]["content"], [{"type": "output_text", "text": "world"}]
        )
        self.assertEqual(
            extract_openai_response_text(
                {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": '{"configs": []}'}],
                        }
                    ]
                }
            ),
            '{"configs": []}',
        )
        self.assertEqual(
            extract_anthropic_text(
                {
                    "content": [
                        {"type": "text", "text": '{"configs": [{"num_warps": 4}]}'}
                    ]
                }
            ),
            '{"configs": [{"num_warps": 4}]}',
        )

        captured = {}
        messages = [
            {"role": "system", "content": "tune kernels"},
            {"role": "user", "content": "suggest configs"},
            {"role": "assistant", "content": '{"configs": []}'},
        ]

        def fake_post_json(url, payload, headers, *, request_timeout_s):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            captured["request_timeout_s"] = request_timeout_s

        cases = [
            {
                "name": "openai",
                "provider": "openai_responses",
                "response_payload": self._response_payload("openai_responses"),
                "expected_text": '{"configs": []}',
                "model": "gpt-5.5",
                "api_base": "https://api.openai.com",
                "expected_url": "https://api.openai.com/v1/responses",
                "api_key": "openai-test-key",
                "request_assertions": lambda captured: (
                    self.assertEqual(
                        captured["headers"]["Authorization"],
                        "Bearer openai-test-key",
                    ),
                    self.assertEqual(
                        captured["payload"]["instructions"], "tune kernels"
                    ),
                    self.assertEqual(captured["payload"]["input"][0]["role"], "user"),
                ),
            },
            {
                "name": "anthropic",
                "provider": "anthropic",
                "response_payload": self._response_payload(
                    "anthropic", '{"configs": [{"num_warps": 4}]}'
                ),
                "expected_text": '{"configs": [{"num_warps": 4}]}',
                "model": "claude-opus-4-7",
                "api_base": "https://api.anthropic.com",
                "expected_url": "https://api.anthropic.com/v1/messages",
                "api_key": "anthropic-test-key",
                "request_assertions": lambda captured: (
                    self.assertEqual(
                        captured["headers"]["x-api-key"],
                        "anthropic-test-key",
                    ),
                    self.assertEqual(captured["payload"]["system"], "tune kernels"),
                    self.assertEqual(
                        captured["payload"]["messages"][0]["role"], "user"
                    ),
                ),
            },
        ]

        for case in cases:
            with self.subTest(name=case["name"]):
                captured.clear()

                def fake_post_json_with_response(
                    url,
                    payload,
                    headers,
                    *,
                    request_timeout_s,
                    response_payload=case["response_payload"],
                ):
                    fake_post_json(
                        url,
                        payload,
                        headers,
                        request_timeout_s=request_timeout_s,
                    )
                    return response_payload

                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch(
                        "helion.autotuner.llm.transport._post_json",
                        side_effect=fake_post_json_with_response,
                    ),
                ):
                    response = call_provider(
                        case["provider"],
                        model=case["model"],
                        api_base=case["api_base"],
                        api_key=case["api_key"],
                        messages=messages,
                        max_output_tokens=512,
                        request_timeout_s=120.0,
                    )
                self.assertEqual(response, case["expected_text"])
                self.assertEqual(captured["url"], case["expected_url"])
                self.assertEqual(captured["request_timeout_s"], 120.0)
                case["request_assertions"](captured)

    def test_supports_anthropic_adaptive_version_matrix(self):
        """Adaptive-thinking detection follows per-family minimum versions."""
        from helion.autotuner.llm.transport import _supports_anthropic_adaptive

        cases = {
            # Opus 4.5+ supports adaptive; 4.4 / 4.1 / 4 (no minor) do not.
            "claude-opus-4-5": True,
            "claude-opus-4-6": True,
            "claude-opus-4-7": True,
            "claude-opus-4-4": False,
            "claude-opus-4-1": False,
            "claude-opus-4": False,
            # Sonnet only crosses the threshold at 4.6.
            "claude-sonnet-4-6": True,
            "claude-sonnet-4-5": False,
            # Variants and date suffixes still match the family/version prefix.
            "claude-opus-4-7[1m]": True,
            "claude-opus-4-7-20260201": True,
            # A bare date suffix (no minor) must not be parsed as minor=20250514.
            "claude-opus-4-20250514": False,
            # Future major / minor releases auto-recognized.
            "claude-opus-5-0": True,
            "claude-opus-4-99": True,
            "claude-sonnet-5-0": True,
            # Families without an adaptive entry stay on the legacy path.
            "claude-haiku-4-5": False,
            # Old-naming models (version before family) never match.
            "claude-3-5-sonnet-20241022": False,
            "claude-3-7-sonnet-20250219": False,
            # Non-Anthropic / unknown model strings fall through.
            "gpt-5.5": False,
            "custom-model": False,
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_supports_anthropic_adaptive(model), expected)

    def test_transport_effort_level_payloads(self):
        """Reasoning effort maps to provider-specific request payload fields."""
        from helion.autotuner.llm.transport import call_provider

        captured = {}
        messages = [{"role": "user", "content": "suggest configs"}]

        def fake_post_json(url, payload, headers, *, request_timeout_s):
            del url, headers, request_timeout_s
            captured["payload"] = payload
            if "input" in payload:
                return self._response_payload("openai_responses")
            return self._response_payload("anthropic")

        # gpt-5.5 supports the xhigh effort level, so the provider-neutral "max"
        # maps directly to xhigh on the OpenAI side.
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "openai_responses",
                model="gpt-5.5",
                api_base="https://api.openai.com",
                api_key="openai-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                effort_level="max",
            )
        self.assertEqual(captured["payload"]["reasoning"], {"effort": "xhigh"})

        # Older OpenAI models without xhigh support fall back to "high" for the
        # provider-neutral "max" knob.
        captured.clear()
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "openai_responses",
                model="gpt-5-2",
                api_base="https://api.openai.com",
                api_key="openai-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                effort_level="max",
            )
        self.assertEqual(captured["payload"]["reasoning"], {"effort": "high"})

        # Modern Claude models (Opus 4.5+, Sonnet 4.6+) use adaptive thinking;
        # the model self-picks its budget so no budget_tokens is sent. Required on
        # Opus 4.7 where manual budget_tokens returns HTTP 400.
        captured.clear()
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "anthropic",
                model="claude-opus-4-7",
                api_base="https://api.anthropic.com",
                api_key="anthropic-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                effort_level="max",
            )
        self.assertEqual(captured["payload"]["thinking"], {"type": "adaptive"})
        self.assertEqual(captured["payload"]["output_config"], {"effort": "max"})
        # max_tokens reserves the adaptive thinking budget (24000) + visible output (512).
        self.assertEqual(captured["payload"]["max_tokens"], 24512)
        self.assertNotIn("temperature", captured["payload"])

        # Legacy Claude models (Opus 4 / Sonnet 4.5 and earlier) still go through
        # the manual budget_tokens path with the same low/medium/high/max table.
        captured.clear()
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "anthropic",
                model="claude-opus-4",
                api_base="https://api.anthropic.com",
                api_key="anthropic-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                effort_level="high",
            )
        self.assertEqual(
            captured["payload"]["thinking"],
            {"type": "enabled", "budget_tokens": 8192},
        )
        self.assertEqual(captured["payload"]["max_tokens"], 8704)
        self.assertNotIn("output_config", captured["payload"])

    def test_transport_fast_mode_payload_and_headers(self):
        """Fast mode sets the Anthropic beta header + speed body field, and is
        orthogonal to effort_level — both can be sent on the same request."""
        from helion.autotuner.llm.transport import call_provider

        captured = {}
        messages = [{"role": "user", "content": "suggest configs"}]

        def fake_post_json(url, payload, headers, *, request_timeout_s):
            del url, request_timeout_s
            captured["payload"] = payload
            captured["headers"] = headers
            if "input" in payload:
                return self._response_payload("openai_responses")
            return self._response_payload("anthropic")

        # fast_mode alone on Opus 4.7: only `speed: "fast"` + beta header, no thinking.
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "anthropic",
                model="claude-opus-4-7",
                api_base="https://api.anthropic.com",
                api_key="anthropic-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                fast_mode=True,
            )
        self.assertEqual(captured["headers"]["anthropic-beta"], "fast-mode-2026-02-01")
        self.assertEqual(captured["payload"]["speed"], "fast")
        self.assertNotIn("thinking", captured["payload"])
        self.assertNotIn("temperature", captured["payload"])
        self.assertEqual(captured["payload"]["max_tokens"], 512)

        # fast_mode + effort_level=max on Opus 4.7: both speed and adaptive thinking
        # are forwarded (the Anthropic API accepts the combination).
        captured.clear()
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "anthropic",
                model="claude-opus-4-7",
                api_base="https://api.anthropic.com",
                api_key="anthropic-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                effort_level="max",
                fast_mode=True,
            )
        self.assertEqual(captured["headers"]["anthropic-beta"], "fast-mode-2026-02-01")
        self.assertEqual(captured["payload"]["speed"], "fast")
        self.assertEqual(captured["payload"]["thinking"], {"type": "adaptive"})
        self.assertEqual(captured["payload"]["output_config"], {"effort": "max"})
        self.assertNotIn("temperature", captured["payload"])

        # OpenAI: fast_mode is accepted but is a no-op (no beta header, no speed field).
        captured.clear()
        with patch(
            "helion.autotuner.llm.transport._post_json",
            side_effect=fake_post_json,
        ):
            call_provider(
                "openai_responses",
                model="gpt-5.5",
                api_base="https://api.openai.com",
                api_key="openai-test-key",
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
                fast_mode=True,
            )
        self.assertNotIn("anthropic-beta", captured["headers"])
        self.assertNotIn("speed", captured["payload"])


class TestBedrockTransport(TestCase):
    """Tests for the AWS Bedrock provider (boto3/SigV4, no API key)."""

    def test_infer_provider_requires_explicit_bedrock_prefix(self):
        """Bedrock must be opted into with an explicit `bedrock/` prefix (or the
        HELION_LLM_PROVIDER override). Bare region-prefixed Bedrock model IDs are
        NOT auto-routed, since the same string can be served by the direct API."""
        from helion.autotuner.llm.transport import infer_provider

        cases = {
            # Explicit bedrock/ prefix -> bedrock.
            "bedrock/us.anthropic.claude-sonnet-4-6": "bedrock",
            "bedrock/us.anthropic.claude-opus-4-8": "bedrock",
            # Bare region-prefixed Bedrock IDs are NOT auto-detected.
            "us.anthropic.claude-sonnet-4-6": "unsupported",
            "apac.anthropic.claude-3-5-haiku-20241022-v1:0": "unsupported",
            # Bare Anthropic / OpenAI IDs keep their existing providers.
            "claude-opus-4-7": "anthropic",
            "anthropic/claude-3-5-sonnet": "anthropic",
            "gpt-5.5": "openai_responses",
            "custom/model": "unsupported",
            # Explicit provider override still routes a bare id to bedrock.
            ("us.anthropic.claude-sonnet-4-6", "bedrock"): "bedrock",
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                if isinstance(model, tuple):
                    self.assertEqual(infer_provider(model[0], model[1]), expected)
                else:
                    self.assertEqual(infer_provider(model), expected)

    def test_normalize_provider_aliases_and_error(self):
        """Bedrock provider aliases canonicalize; bedrock joins the error message."""
        from helion.autotuner.llm.transport import normalize_provider

        for alias in ("bedrock", "aws_bedrock", "aws-bedrock"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_provider(alias), "bedrock")
        with self.assertRaisesRegex(ValueError, "bedrock"):
            normalize_provider("bogus")

    def test_strip_provider_prefix_drops_bedrock(self):
        """The bedrock/ prefix is stripped like anthropic/ and openai/."""
        from helion.autotuner.llm.transport import strip_provider_prefix

        self.assertEqual(
            strip_provider_prefix("bedrock/us.anthropic.claude-opus-4-8"),
            "us.anthropic.claude-opus-4-8",
        )
        self.assertEqual(
            strip_provider_prefix("us.anthropic.claude-opus-4-8"),
            "us.anthropic.claude-opus-4-8",
        )

    def test_temperature_deprecated_version_matrix(self):
        """`temperature` deprecation follows per-family minimum versions and
        tolerates Bedrock-prefixed IDs."""
        from helion.autotuner.llm.transport import _temperature_deprecated

        cases = {
            # Opus 4.7+ rejects temperature (both bare and Bedrock-prefixed forms).
            "claude-opus-4-7": True,
            "claude-opus-4-8": True,
            "us.anthropic.claude-opus-4-7": True,
            "us.anthropic.claude-opus-4-8": True,
            "us.anthropic.claude-opus-5-0": True,
            # Opus 4.6 and earlier still accept temperature.
            "claude-opus-4-6": False,
            "us.anthropic.claude-opus-4-6-v1": False,
            "us.anthropic.claude-opus-4-5-20251101-v1:0": False,
            # Sonnet has no temperature-deprecation entry -> always accepts it.
            "claude-sonnet-4-6": False,
            "us.anthropic.claude-sonnet-4-6": False,
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": False,
            # Non-Anthropic / unknown strings fall through.
            "gpt-5.5": False,
            "custom-model": False,
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_temperature_deprecated(model), expected)

    def test_resolve_bedrock_region_precedence(self):
        """Region comes from api_base override first, then AWS_REGION,
        then AWS_DEFAULT_REGION."""
        from helion.autotuner.llm.transport import _resolve_bedrock_region

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_bedrock_region("us-west-2"), "us-west-2")
            self.assertIsNone(_resolve_bedrock_region(None))
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True):
            self.assertEqual(_resolve_bedrock_region(None), "us-east-1")
            # Explicit api_base still wins over the env var.
            self.assertEqual(_resolve_bedrock_region("eu-west-1"), "eu-west-1")
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "ap-south-1"}, clear=True):
            self.assertEqual(_resolve_bedrock_region(None), "ap-south-1")

    def test_call_provider_bedrock_invokes_boto3_and_extracts_text(self):
        """call_provider('bedrock', ...) builds the Anthropic body, moves the
        model to modelId, adds anthropic_version, and parses the response."""
        import json

        from helion.autotuner.llm.transport import call_provider

        captured = {}

        class FakeBody:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

        class FakeBedrockClient:
            def invoke_model(self, *, modelId, body):
                captured["modelId"] = modelId
                captured["body"] = json.loads(body)
                resp = {"content": [{"type": "text", "text": '{"configs": []}'}]}
                return {"body": FakeBody(json.dumps(resp).encode("utf-8"))}

        def fake_boto3_client(service, *, region_name=None, config=None):
            captured["service"] = service
            captured["region_name"] = region_name
            return FakeBedrockClient()

        fake_boto3 = SimpleNamespace(client=fake_boto3_client)
        fake_botocore_config = SimpleNamespace(
            config=SimpleNamespace(Config=lambda **kw: ("config", kw))
        )
        messages = [
            {"role": "system", "content": "tune kernels"},
            {"role": "user", "content": "suggest configs"},
        ]
        with (
            patch.dict(os.environ, {"AWS_REGION": "us-east-2"}, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "boto3": fake_boto3,
                    "botocore": SimpleNamespace(config=fake_botocore_config.config),
                    "botocore.config": fake_botocore_config.config,
                },
            ),
        ):
            response = call_provider(
                "bedrock",
                model="us.anthropic.claude-sonnet-4-6",
                api_base=None,
                api_key=None,
                messages=messages,
                max_output_tokens=512,
                request_timeout_s=120.0,
            )

        self.assertEqual(response, '{"configs": []}')
        self.assertEqual(captured["service"], "bedrock-runtime")
        self.assertEqual(captured["region_name"], "us-east-2")
        # The model is named via modelId, not in the request body.
        self.assertEqual(captured["modelId"], "us.anthropic.claude-sonnet-4-6")
        self.assertNotIn("model", captured["body"])
        # Bedrock requires the version string and the standard Anthropic fields.
        self.assertEqual(captured["body"]["anthropic_version"], "bedrock-2023-05-31")
        self.assertEqual(captured["body"]["system"], "tune kernels")
        self.assertEqual(captured["body"]["messages"][0]["role"], "user")

    def test_call_provider_bedrock_missing_boto3_raises_clear_error(self):
        """A helpful error is raised when boto3 is not installed."""
        import builtins

        from helion.autotuner.llm.transport import call_provider

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "boto3" or name.startswith("botocore"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=fake_import),
            self.assertRaisesRegex(RuntimeError, "boto3 is not installed"),
        ):
            call_provider(
                "bedrock",
                model="us.anthropic.claude-sonnet-4-6",
                api_base=None,
                api_key=None,
                messages=[{"role": "user", "content": "hi"}],
                max_output_tokens=64,
                request_timeout_s=120.0,
            )


class TestLLMSeededLFBOTreeSearch(TestCase):
    """Tests for the two-stage LLM-seeded hybrid autotuner."""

    def test_profile_kwargs_and_env_overrides(self):
        """Hybrid profile wiring forwards shared LLM settings and hybrid env overrides."""
        from helion.autotuner import LLMSeededLFBOTreeSearch
        from helion.autotuner import LLMSeededSearch

        kwargs = LLMSeededLFBOTreeSearch.get_kwargs_from_profile(
            get_effort_profile("full"), Settings()
        )
        self.assertEqual(kwargs["llm_model"], "gpt-5-2")
        self.assertEqual(kwargs["llm_configs_per_round"], 15)
        self.assertEqual(kwargs["llm_max_rounds"], 1)
        self.assertEqual(kwargs["llm_initial_random_configs"], 10)
        self.assertEqual(kwargs["llm_compile_timeout_s"], 15)
        self.assertFalse(kwargs["best_available_pad_random"])

        with patch.dict(
            os.environ,
            {"HELION_HYBRID_SECOND_STAGE_ALGORITHM": "PatternSearch"},
            clear=False,
        ):
            generic_kwargs = LLMSeededSearch.get_kwargs_from_profile(
                get_effort_profile("full"), Settings()
            )
        self.assertEqual(generic_kwargs["second_stage_algorithm"], "PatternSearch")
        self.assertIn("max_generations", generic_kwargs["second_stage_kwargs"])

        kernel = SimpleNamespace(
            settings=Settings(),
            config_spec=SimpleNamespace(),
        )
        with patch.dict(
            os.environ,
            {
                "HELION_HYBRID_LLM_MAX_ROUNDS": "2",
                "HELION_LLM_PROVIDER": "openai",
                "HELION_LLM_EFFORT_LEVEL": "max",
                "HELION_LLM_FAST_MODE": "1",
            },
            clear=False,
        ):
            kwargs = LLMSeededLFBOTreeSearch.get_kwargs_from_profile(
                get_effort_profile("full"), Settings()
            )
        self.assertEqual(kwargs["llm_max_rounds"], 2)
        self.assertEqual(kwargs["llm_provider"], "openai")
        self.assertEqual(kwargs["llm_effort_level"], "max")
        self.assertTrue(kwargs["llm_fast_mode"])

        search = LLMSeededLFBOTreeSearch(kernel, (), **kwargs)
        self.assertEqual(search.llm_provider, "openai")
        self.assertEqual(search.llm_effort_level, "max")
        self.assertTrue(search.llm_fast_mode)

    def test_selected_by_env(self):
        """HELION_AUTOTUNER selects the hybrid autotuner and applies profile defaults."""
        from helion.autotuner import LLMSeededLFBOTreeSearch

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        with patch.dict(os.environ, {"HELION_AUTOTUNER": "LLMSeededLFBOTreeSearch"}):

            @helion.kernel(autotune_effort="full")
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add.bind(args)
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertIsInstance(autotuner.autotuner, LLMSeededLFBOTreeSearch)
            self.assertEqual(autotuner.autotuner.llm_max_rounds, 1)
            self.assertFalse(autotuner.autotuner.best_available_pad_random)

    def test_handoff_runs_llm_then_lfbo(self):
        """The hybrid flow runs LLM seeding first, then injects that seed into LFBO."""
        from helion.autotuner import InitialPopulationStrategy
        from helion.autotuner import LLMSeededLFBOTreeSearch
        from helion.runtime.config import Config

        llm_instances = []
        lfbo_instances = []

        class FakeBenchmarkProvider:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def set_budget_exceeded_fn(self, fn) -> None:
                pass

        class FakeLLMSearch:
            def __init__(self, kernel, args, **kwargs) -> None:
                self.kernel = kernel
                self.args = args
                self.kwargs = kwargs
                self.best_perf_so_far = 0.9
                self._autotune_metrics = AutotuneMetrics(
                    num_configs_tested=7,
                    num_compile_failures=1,
                    num_accuracy_failures=2,
                    num_generations=3,
                )
                llm_instances.append(self)

            def autotune(self, *, skip_cache=False):
                self.skip_cache = skip_cache
                return Config(num_warps=4)

        class FakeLFBOSearch:
            def __init__(
                self,
                kernel,
                args,
                *,
                initial_population_strategy=None,
                best_available_pad_random=True,
                **kwargs,
            ) -> None:
                self.kernel = kernel
                self.args = args
                self.kwargs = {
                    **kwargs,
                    "initial_population_strategy": initial_population_strategy,
                    "best_available_pad_random": best_available_pad_random,
                }
                self.best_perf_so_far = 0.5
                self._autotune_metrics = AutotuneMetrics(
                    num_configs_tested=11,
                    num_compile_failures=3,
                    num_accuracy_failures=5,
                    num_generations=6,
                )
                self.seed_configs = None
                lfbo_instances.append(self)

            def set_best_available_seed_configs(self, configs):
                self.seed_configs = list(configs)

            def autotune(self):
                return Config(num_warps=8)

        kernel = SimpleNamespace(
            settings=Settings(),
            config_spec=SimpleNamespace(),
            env=SimpleNamespace(device=DEVICE, process_group_name=None),
        )
        args = (torch.randn([8], device=DEVICE),)
        with (
            patch("helion.autotuner.llm_seeded_lfbo.LLMGuidedSearch", FakeLLMSearch),
            patch(
                "helion.autotuner.llm_seeded_lfbo._resolve_second_stage_algorithm",
                return_value=FakeLFBOSearch,
            ),
            patch(
                "helion.autotuner.llm_seeded_lfbo._supports_best_available_handoff",
                return_value=True,
            ),
        ):
            search = LLMSeededLFBOTreeSearch(
                kernel,
                args,
                llm_max_rounds=2,
                best_available_pad_random=True,
            )
            search._benchmark_provider_cls = FakeBenchmarkProvider
            search._prepare()
            self.assertIsInstance(search.benchmark_provider, FakeBenchmarkProvider)
            best = search._autotune()

        self.assertEqual(best["num_warps"], 8)
        self.assertEqual(llm_instances[0].kwargs["max_rounds"], 2)
        self.assertTrue(llm_instances[0].skip_cache)
        self.assertEqual(
            lfbo_instances[0].kwargs["initial_population_strategy"],
            InitialPopulationStrategy.FROM_BEST_AVAILABLE,
        )
        self.assertTrue(lfbo_instances[0].kwargs["best_available_pad_random"])
        self.assertEqual(lfbo_instances[0].seed_configs, [Config(num_warps=4)])
        self.assertEqual(search._autotune_metrics.num_configs_tested, 18)
        self.assertEqual(search._autotune_metrics.num_compile_failures, 4)
        self.assertEqual(search._autotune_metrics.num_accuracy_failures, 7)
        self.assertEqual(search._autotune_metrics.num_generations, 9)
        self.assertEqual(search.hybrid_stage_breakdown["llm_seed_configs_tested"], 7)
        self.assertEqual(
            search.hybrid_stage_breakdown["second_stage_configs_tested"], 11
        )

    @onlyBackends(["triton", "cute"])
    def test_autotune_runs_full_hybrid_loop_with_mocked_stages(self):
        """The public hybrid autotune entrypoint runs both stages and returns stage-2 best."""
        from helion.autotuner import InitialPopulationStrategy
        from helion.autotuner import LLMSeededLFBOTreeSearch
        from helion.runtime.config import Config

        llm_instances = []
        lfbo_instances = []
        stage_order: list[str] = []

        class FakeBenchmarkProvider:
            def __init__(
                self,
                *,
                kernel,
                settings,
                config_spec,
                args,
                log,
                autotune_metrics,
            ) -> None:
                del kernel, settings, config_spec, args, log, autotune_metrics
                self.mutated_arg_indices: list[int] = []
                self.setup_called = False
                self.cleanup_called = False

            def setup(self) -> None:
                self.setup_called = True

            def cleanup(self) -> None:
                self.cleanup_called = True

            def set_budget_exceeded_fn(self, fn) -> None:
                pass

        class FakeLLMSearch:
            def __init__(self, kernel, args, **kwargs) -> None:
                self.kernel = kernel
                self.args = args
                self.kwargs = kwargs
                self.best_perf_so_far = 0.9
                self._autotune_metrics = AutotuneMetrics(
                    num_configs_tested=4,
                    num_compile_failures=1,
                    num_accuracy_failures=0,
                    num_generations=2,
                )
                llm_instances.append(self)

            def autotune(self, *, skip_cache=False):
                self.skip_cache = skip_cache
                stage_order.append("llm")
                return Config(num_warps=4)

        class FakeLFBOSearch:
            def __init__(
                self,
                kernel,
                args,
                *,
                initial_population_strategy=None,
                best_available_pad_random=True,
                **kwargs,
            ) -> None:
                self.kernel = kernel
                self.args = args
                self.kwargs = {
                    **kwargs,
                    "initial_population_strategy": initial_population_strategy,
                    "best_available_pad_random": best_available_pad_random,
                }
                self.best_perf_so_far = 0.5
                self._autotune_metrics = AutotuneMetrics(
                    num_configs_tested=6,
                    num_compile_failures=2,
                    num_accuracy_failures=1,
                    num_generations=5,
                )
                self.seed_configs = None
                lfbo_instances.append(self)

            def set_best_available_seed_configs(self, configs):
                self.seed_configs = list(configs)

            def autotune(self):
                stage_order.append("lfbo")
                return Config(num_warps=8)

        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float16),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float16),
        )
        bound = _get_examples_matmul().bind(args)
        with (
            patch("helion.autotuner.llm_seeded_lfbo.LLMGuidedSearch", FakeLLMSearch),
            patch(
                "helion.autotuner.llm_seeded_lfbo._resolve_second_stage_algorithm",
                return_value=FakeLFBOSearch,
            ),
            patch(
                "helion.autotuner.llm_seeded_lfbo._supports_best_available_handoff",
                return_value=True,
            ),
        ):
            search = LLMSeededLFBOTreeSearch(bound, args, llm_max_rounds=2)
            search._benchmark_provider_cls = FakeBenchmarkProvider
            best = search.autotune(skip_cache=True)

        self.assertEqual(stage_order, ["llm", "lfbo"])
        self.assertEqual(dict(best), {"num_warps": 8})
        self.assertTrue(search.benchmark_provider.setup_called)
        self.assertTrue(search.benchmark_provider.cleanup_called)
        self.assertEqual(llm_instances[0].kwargs["max_rounds"], 2)
        self.assertTrue(llm_instances[0].skip_cache)
        self.assertEqual(
            lfbo_instances[0].kwargs["initial_population_strategy"],
            InitialPopulationStrategy.FROM_BEST_AVAILABLE,
        )
        self.assertEqual(lfbo_instances[0].seed_configs, [Config(num_warps=4)])
        self.assertTrue(search.hybrid_stage_breakdown["used_llm_seed"])
        self.assertEqual(search.hybrid_stage_breakdown["llm_seed_configs_tested"], 4)
        self.assertEqual(
            search.hybrid_stage_breakdown["second_stage_configs_tested"], 6
        )
        self.assertEqual(search._autotune_metrics.num_configs_tested, 10)
        self.assertEqual(search._autotune_metrics.num_compile_failures, 3)
        self.assertEqual(search._autotune_metrics.num_accuracy_failures, 1)
        self.assertEqual(search._autotune_metrics.num_generations, 7)

    def test_zero_llm_rounds_falls_back_to_lfbo_strategy(self):
        """Disabling LLM rounds skips stage 1 and leaves the second-stage strategy unchanged."""
        from helion.autotuner import InitialPopulationStrategy
        from helion.autotuner import LLMSeededLFBOTreeSearch
        from helion.runtime.config import Config

        lfbo_instances = []

        class FakeBenchmarkProvider:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def set_budget_exceeded_fn(self, fn) -> None:
                pass

        class FailIfLLMConstructed:
            def __init__(self, *args, **kwargs) -> None:
                raise AssertionError("LLM seed stage should be skipped")

        class FakeLFBOSearch:
            def __init__(self, kernel, args, **kwargs) -> None:
                self.kwargs = kwargs
                self.best_perf_so_far = 0.4
                self._autotune_metrics = AutotuneMetrics(num_configs_tested=3)
                lfbo_instances.append(self)

            def autotune(self):
                return Config(num_warps=16)

        kernel = SimpleNamespace(
            settings=Settings(),
            config_spec=SimpleNamespace(),
            env=SimpleNamespace(device=DEVICE, process_group_name=None),
        )
        args = (torch.randn([8], device=DEVICE),)
        with (
            patch(
                "helion.autotuner.llm_seeded_lfbo.LLMGuidedSearch",
                FailIfLLMConstructed,
            ),
            patch(
                "helion.autotuner.llm_seeded_lfbo._resolve_second_stage_algorithm",
                return_value=FakeLFBOSearch,
            ),
        ):
            search = LLMSeededLFBOTreeSearch(
                kernel,
                args,
                llm_max_rounds=0,
                second_stage_kwargs={
                    "initial_population_strategy": InitialPopulationStrategy.FROM_RANDOM
                },
            )
            search._benchmark_provider_cls = FakeBenchmarkProvider
            search._prepare()
            best = search._autotune()

        self.assertEqual(best["num_warps"], 16)
        self.assertEqual(
            lfbo_instances[0].kwargs["initial_population_strategy"],
            InitialPopulationStrategy.FROM_RANDOM,
        )
        self.assertFalse(search.hybrid_stage_breakdown["used_llm_seed"])
