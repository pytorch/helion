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
        search.model = "gpt-5-2"
        search.api_base = None
        search.api_key = None
        search.request_timeout_s = 120.0

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

    @onlyBackends(["triton"])
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
                        fn=lambda: None,
                        perf=perf,
                        status="ok",
                        compile_time=0.01,
                    )
                )
            return results

        def fake_rebenchmark_population(self, members=None, *, desc="Rebenchmarking"):
            del members
            rebenchmark_descs.append(desc)
            for member in self.population:
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
            ["Round 0: verifying top configs", "Round 1: verifying top configs"],
        )
        self.assertEqual(search._autotune_metrics.num_configs_tested, 3)
        self.assertEqual(len(search._all_benchmark_results), 3)
        self.assertEqual(len(search.population), 3)
        self.assertEqual(search.best.perf, 2.5)
        self.assertEqual(dict(best), dict(round1_cfg))
        self.assertEqual(dict(search.best.config), dict(round1_cfg))


class TestLLMTransport(TestCase):
    """Tests for provider selection and HTTP payload translation."""

    @staticmethod
    def _response_payload(
        provider: str, text: str = '{"configs": []}'
    ) -> dict[str, object]:
        if provider == "openai_responses":
            return {"output": [{"content": [{"type": "text", "text": text}]}]}
        return {"content": [{"type": "text", "text": text}]}

    def test_transport_helpers_and_http_request_shapes(self):
        """Provider helpers build the expected request/response shapes for each backend."""
        from helion.autotuner.llm.transport import _resolve_api_base
        from helion.autotuner.llm.transport import _resolve_api_key
        from helion.autotuner.llm.transport import _resolve_v1_endpoint
        from helion.autotuner.llm.transport import call_provider
        from helion.autotuner.llm.transport import extract_anthropic_text
        from helion.autotuner.llm.transport import extract_openai_response_text
        from helion.autotuner.llm.transport import infer_provider
        from helion.autotuner.llm.transport import responses_input_from_messages
        from helion.autotuner.llm.transport import split_system_messages

        self.assertEqual(infer_provider("claude-haiku-4.5"), "anthropic")
        self.assertEqual(infer_provider("gpt-5-2"), "openai_responses")
        self.assertEqual(infer_provider("custom/model"), "unsupported")
        self.assertEqual(infer_provider("custom/model", "anthropic"), "anthropic")
        self.assertEqual(infer_provider("custom/model", "openai"), "openai_responses")
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            infer_provider("gpt-5-2", "bogus")

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
                "model": "gpt-5-2",
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
                "model": "claude-3-5-haiku-latest",
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
            },
            clear=False,
        ):
            kwargs = LLMSeededLFBOTreeSearch.get_kwargs_from_profile(
                get_effort_profile("full"), Settings()
            )
        self.assertEqual(kwargs["llm_max_rounds"], 2)
        self.assertEqual(kwargs["llm_provider"], "openai")

        search = LLMSeededLFBOTreeSearch(kernel, (), **kwargs)
        self.assertEqual(search.llm_provider, "openai")

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

    @onlyBackends(["triton"])
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
