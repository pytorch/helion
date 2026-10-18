from __future__ import annotations

import copy
from dataclasses import replace
import inspect
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import torch

from test._autotuner_test_utils import AutotunerTestCase
from test._autotuner_test_utils import _assert_phase_config_manifest
from test._autotuner_test_utils import _cute_flash_test_config_spec
from test._autotuner_test_utils import _get_examples_matmul
from test._autotuner_test_utils import _stub_flash_conditional_proposals

import helion
from helion import exc
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import CAUSAL_MASK_KIND
from helion._compiler.cute.attention_plan import ROW_KV_BEGIN_MASK_KIND
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.cute_flash import _flash_clc_heads_per_batch_candidates
from helion._compiler.cute.cute_flash import (
    _flash_clc_heads_per_batch_coverage_candidates,
)
from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner import DESurrogateHybrid
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import LLMGuidedSearch
from helion.autotuner import LLMSeededLFBOTreeSearch
from helion.autotuner import LLMSeededSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.benchmarking import MirroredBenchmarkTrace
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_generation import CoordinateNeighborProjection
from helion.autotuner.config_generation import _flash_log_maximin_refinements
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.local_cache import _cute_flash_search_policy_hash
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.search_space_logger import canonical_config_id
from helion.autotuner.surrogate_pattern_search import (
    flash_terminal_measurement_is_valid,
)
from helion.autotuner.surrogate_pattern_search import (
    flash_terminal_refinement_result_is_valid,
)
import helion.language as hl
from helion.runtime.settings import Settings
from helion.runtime.settings import default_autotuner_fn

_CACHE_POLICY_BASELINE_SCALE = 1
_CACHE_POLICY_DYNAMIC_HELPER: object = None


@onlyBackends(["triton", "cute"])
class TestAutotunerCuteFlash(AutotunerTestCase):
    def test_lfbo_flash_terminal_coordinate_refinement_runs_two_rounds(self):
        def make_fn(source_id: int):
            def fn():
                return None

            fn.source_hash = f"{source_id:064x}"
            return fn

        def make_member(value: int, perf: float | None) -> PopulationMember:
            return PopulationMember(
                fn=make_fn(value),
                perfs=[] if perf is None else [perf],
                flat_values=[value],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_wait_hint=value,
                ),
                status="unknown" if perf is None else "ok",
            )

        def projections(value: int) -> list[CoordinateNeighborProjection]:
            targets = (1, 2) if value == 0 else (0, 2)
            result = []
            for target in targets:
                projected = make_member(target, None).config
                projected.config["cute_vector_widths"] = [1, 1, 1]
                result.append(
                    CoordinateNeighborProjection(
                        flat_index=0,
                        key=cute_flash.FLASH_WAIT_HINT_KEY,
                        sequence_index=None,
                        from_value=value,
                        to_value=target,
                        outcome="candidate",
                        flat_values=[target],
                        config=projected,
                    )
                )
            return result

        def canonicalize_projections(projections, *, base_config):
            return [
                replace(
                    projection,
                    config=make_member(int(projection.to_value), None).config,
                )
                for projection in projections
            ]

        best = make_member(0, 10.0)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.best_perf_so_far = best.perf
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
            ),
        )
        search.config_gen = SimpleNamespace(
            flatten=lambda config: [config.config[cute_flash.FLASH_WAIT_HINT_KEY]],
            canonicalize_coordinate_projections=canonicalize_projections,
        )
        leaf_generation = SimpleNamespace(
            flatten=search.config_gen.flatten,
            coordinate_neighbor_projections=lambda flat, *, radius: projections(
                int(flat[0])
            ),
        )
        search._flash_leaf_config_generation = Mock(return_value=leaf_generation)
        search.flash_structural_search = replace(
            get_effort_profile("full").flash_structural_search,
            terminal_coordinate_rounds=2,
            terminal_coordinate_beam_width=2,
        )
        search._cute_flash_lane_policy_enabled = True
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}
        search._autotune_metrics = SimpleNamespace(
            num_generations=20,
            num_configs_tested=100,
            search_phase_metrics={},
        )
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.radius = 2
        search.min_improvement_delta = 0.001
        search.log = Mock()
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.make_unbenchmarked = Mock(
            side_effect=lambda flat: make_member(int(flat[0]), None)
        )

        initial_perfs = {1: 9.9, 2: 10.2}

        def benchmark_population(members, *, desc):
            self.assertIn("Terminal coordinate refinement", desc)
            self.assertEqual(
                search._autotune_metrics.search_phase_metrics[
                    "terminal_coordinate_refinement"
                ]["preterminal_num_configs_tested"],
                100,
            )
            search._autotune_metrics.num_configs_tested += len(members)
            for member in members:
                value = member.config.config[cute_flash.FLASH_WAIT_HINT_KEY]
                member.perfs.append(initial_perfs[value])
                member.status = "ok"
                search._record_benchmarked_member(member)
            return members

        search.benchmark_population = Mock(side_effect=benchmark_population)
        timing_rounds = (
            {0: 10.0, 1: 9.8, 2: 10.3},
            {0: 10.0, 1: 10.4, 2: 9.5},
            {0: 10.2, 1: 9.9, 2: 9.6},
        )

        def mirrored_rebenchmark(members, *, desc, target_ms):
            values = timing_rounds[mirrored_rebenchmark.calls]
            mirrored_rebenchmark.calls += 1
            timings = [
                values[member.config.config[cute_flash.FLASH_WAIT_HINT_KEY]]
                for member in members
            ]
            for member, timing in zip(members, timings, strict=True):
                member.perfs.append(timing)
            order = list(range(len(members)))
            return MirroredBenchmarkTrace(
                orders=[order, list(reversed(order))],
                elapsed_ms=[timings, list(reversed(timings))],
                medians_ms=timings,
                sweep_count=2,
                calls_per_sample=1,
                total_calls=2,
            )

        mirrored_rebenchmark.calls = 0
        search.mirrored_rebenchmark = Mock(side_effect=mirrored_rebenchmark)

        selected = search.run_terminal_refinement(best)

        self.assertEqual(
            selected.config.config[cute_flash.FLASH_WAIT_HINT_KEY],
            2,
        )
        self.assertEqual(search._autotune_metrics.num_generations, 20)
        self.assertEqual(search.mirrored_rebenchmark.call_count, 3)
        self.assertEqual(
            [
                invocation.kwargs["target_ms"]
                for invocation in search.mirrored_rebenchmark.call_args_list
            ],
            [200.0, 200.0, 5000.0],
        )
        transcript = search._autotune_metrics.search_phase_metrics[
            "terminal_coordinate_refinement"
        ]
        self.assertTrue(
            all(
                "cute_vector_widths" not in entry["config"]
                for entry in transcript["config_manifest"].values()
            )
        )
        self.assertTrue(transcript["completed"])
        self.assertEqual(transcript["termination_reason"], "round_limit")
        self.assertEqual(transcript["rounds_started"], 2)
        self.assertEqual(transcript["rounds_completed"], 2)
        self.assertEqual(
            [round_metric["accepted"] for round_metric in transcript["rounds"]],
            [True, True],
        )
        self.assertEqual(transcript["new_candidate_count"], 2)
        self.assertEqual(transcript["preterminal_num_configs_tested"], 100)
        self.assertEqual(search._autotune_metrics.num_configs_tested, 102)
        self.assertEqual(transcript["preterminal_registry_config_count"], 1)
        self.assertRegex(
            transcript["preterminal_registry_config_ids_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(transcript["beam_width"], 2)
        self.assertEqual(transcript["projection_parent_count"], 3)
        self.assertEqual(transcript["maximum_projection_parent_count"], 3)
        self.assertTrue(
            all(
                len(round_metric["parent_config_ids"]) <= transcript["beam_width"]
                for round_metric in transcript["rounds"]
            )
        )
        self.assertEqual(transcript["reused_candidate_count"], 0)
        self.assertEqual(transcript["intra_terminal_reused_candidate_count"], 1)
        self.assertEqual(
            transcript["accepted_config_ids"],
            [canonical_config_id(make_member(value, 1.0).config) for value in (1, 2)],
        )
        self.assertNotIn(
            canonical_config_id(make_member(1, 1.0).config),
            transcript["rounds"][1]["beam_config_ids"],
        )
        self.assertEqual(
            transcript["final_config_id"], canonical_config_id(selected.config)
        )
        self.assertEqual(
            transcript["confirmation"]["candidate_config_ids"],
            [
                canonical_config_id(make_member(value, 1.0).config)
                for value in (0, 1, 2)
            ],
        )
        self.assertEqual(
            transcript["rounds"][0]["measurement"]["elapsed_ms"][0],
            [10.0, 9.8, 10.3],
        )

    def test_lfbo_flash_terminal_coordinate_refinement_skips_multi_shape(self):
        best = Mock(spec=PopulationMember)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.benchmark_provider = object.__new__(MultiShapeBenchmarkProvider)

        self.assertIs(search.run_terminal_refinement(best), best)
        self.assertEqual(search._autotune_metrics.search_phase_metrics, {})

    def test_lfbo_flash_terminal_coordinate_refinement_finalizes_missing_leaf(self):
        best = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[1],
            config=helion.Config(block_sizes=[1, 128, 128]),
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._terminal_refinement_members = {}
        search._autotune_metrics = SimpleNamespace(
            num_generations=20,
            num_configs_tested=100,
            search_phase_metrics={},
        )
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.radius = 2
        search.min_improvement_delta = 0.001
        search.log = Mock()

        self.assertIs(search.run_terminal_refinement(best), best)
        transcript = search._autotune_metrics.search_phase_metrics[
            "terminal_coordinate_refinement"
        ]
        self.assertTrue(transcript["completed"])
        self.assertEqual(transcript["termination_reason"], "no_candidates")
        self.assertEqual(
            transcript["confirmation"]["skipped_reason"],
            "missing_structural_leaf",
        )
        self.assertRegex(transcript["config_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_terminal_refinement_registry_preserves_successful_duplicate(self):
        config = helion.Config(block_sizes=[1, 128, 128])
        successful = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        failed = PopulationMember(
            fn=lambda: None,
            perfs=[math.inf],
            flat_values=[1],
            config=copy.deepcopy(config),
            status="error",
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}

        search._record_benchmarked_member(successful)
        search._record_benchmarked_member(failed)

        stored = search._terminal_refinement_members[config]
        self.assertEqual(stored.status, "ok")
        self.assertEqual(stored.perf, 10.0)

    def test_terminal_refinement_registry_drops_quarantined_config(self):
        config = helion.Config(block_sizes=[1, 128, 128])
        successful = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        quarantined = PopulationMember(
            fn=lambda: None,
            perfs=[math.inf],
            flat_values=[1],
            config=copy.deepcopy(config),
            status="error",
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True, backend_name="cute"
        )
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}

        search._record_benchmarked_member(successful)
        self.assertIn(config, search._terminal_refinement_members)

        search._refresh_benchmarked_members_after_rebenchmark([quarantined])

        self.assertNotIn(config, search._terminal_refinement_members)

    def test_lfbo_flash_terminal_refinement_skips_budgeted_search(self):
        best = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[0],
            config=helion.Config(block_sizes=[1]),
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=1.0,
        )
        search.benchmark_population = Mock()
        search.mirrored_rebenchmark = Mock()

        self.assertIs(search.run_terminal_refinement(best), best)
        self.assertEqual(search._autotune_metrics.search_phase_metrics, {})
        search.benchmark_population.assert_not_called()
        search.mirrored_rebenchmark.assert_not_called()

    def test_lfbo_flash_terminal_coordinate_refinement_crosses_beam_valley(self):
        def member(
            e2e_offset: int,
            rescale_threshold: int,
            perf: float | None,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[e2e_offset, rescale_threshold],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_e2e_offset=e2e_offset,
                    cute_flash_rescale_threshold=float(rescale_threshold),
                ),
                status="unknown" if perf is None else "ok",
            )

        best = member(7, 12, 10.0)
        intermediate_config = member(5, 12, None).config
        endpoint_config = member(5, 4, None).config

        def projections(flat) -> list[CoordinateNeighborProjection]:
            if flat == [7, 12.0]:
                return [
                    CoordinateNeighborProjection(
                        0,
                        cute_flash.FLASH_E2E_OFFSET_KEY,
                        None,
                        7,
                        5,
                        "candidate",
                        [5, 12.0],
                        intermediate_config,
                    )
                ]
            if flat == [5, 12.0]:
                return [
                    CoordinateNeighborProjection(
                        1,
                        cute_flash.FLASH_RESCALE_THRESHOLD_KEY,
                        None,
                        12.0,
                        4.0,
                        "candidate",
                        [5, 4.0],
                        endpoint_config,
                    )
                ]
            return []

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.best_perf_so_far = best.perf
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            backend=SimpleNamespace(generated_source_hash=lambda _fn: None),
        )
        search.config_gen = SimpleNamespace(
            flatten=lambda config: [
                config.config[cute_flash.FLASH_E2E_OFFSET_KEY],
                config.config[cute_flash.FLASH_RESCALE_THRESHOLD_KEY],
            ],
            canonicalize_coordinate_projections=lambda projections, *, base_config: (
                projections
            ),
        )
        search._flash_leaf_config_generation = Mock(
            return_value=SimpleNamespace(
                flatten=search.config_gen.flatten,
                coordinate_neighbor_projections=lambda flat, *, radius: projections(
                    flat
                ),
            )
        )
        search.flash_structural_search = replace(
            get_effort_profile("full").flash_structural_search,
            terminal_coordinate_rounds=2,
        )
        search._cute_flash_lane_policy_enabled = True
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}
        search._autotune_metrics = SimpleNamespace(
            num_generations=20,
            search_phase_metrics={},
        )
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.radius = 2
        search.min_improvement_delta = 0.001
        search.log = Mock()
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.make_unbenchmarked = Mock(
            side_effect=lambda flat: member(int(flat[0]), int(flat[1]), None)
        )

        def benchmark_population(members, **_kwargs):
            attempt_perfs = {(5, 12): 10.1, (5, 4): 9.9}
            for candidate in members:
                values = tuple(int(value) for value in candidate.flat_values)
                candidate.perfs.append(attempt_perfs[values])
                candidate.status = "ok"
                search._record_benchmarked_member(candidate)
            return members

        search.benchmark_population = Mock(side_effect=benchmark_population)

        timing_rounds = (
            {(7, 12): 10.0, (5, 12): 10.005},
            {(7, 12): 10.0, (5, 12): 10.02, (5, 4): 9.8},
            {(7, 12): 10.0, (5, 12): 10.02, (5, 4): 9.75},
        )

        def mirrored_rebenchmark(members, **_kwargs):
            values = timing_rounds[mirrored_rebenchmark.calls]
            mirrored_rebenchmark.calls += 1
            timings = [
                values[tuple(int(value) for value in candidate.flat_values)]
                for candidate in members
            ]
            for candidate, timing in zip(members, timings, strict=True):
                candidate.perfs.append(timing)
            order = list(range(len(members)))
            return MirroredBenchmarkTrace(
                orders=[order, list(reversed(order))],
                elapsed_ms=[timings, list(reversed(timings))],
                medians_ms=timings,
                sweep_count=2,
                calls_per_sample=1,
                total_calls=2,
            )

        mirrored_rebenchmark.calls = 0
        search.mirrored_rebenchmark = Mock(side_effect=mirrored_rebenchmark)

        selected = search.run_terminal_refinement(best)

        self.assertEqual(selected.config, endpoint_config)
        self.assertEqual(search.mirrored_rebenchmark.call_count, 3)
        transcript = search._autotune_metrics.search_phase_metrics[
            "terminal_coordinate_refinement"
        ]
        self.assertEqual(transcript["rounds_completed"], 2)
        self.assertEqual(transcript["termination_reason"], "round_limit")
        self.assertEqual(
            [round_metric["accepted"] for round_metric in transcript["rounds"]],
            [False, True],
        )
        self.assertIn(
            canonical_config_id(intermediate_config),
            transcript["rounds"][1]["parent_config_ids"],
        )
        self.assertEqual(
            transcript["rounds"][0]["candidate_results"][0]["selection_perf"],
            10.005,
        )
        self.assertTrue(transcript["confirmation"]["accepted"])
        self.assertEqual(
            transcript["final_config_id"], canonical_config_id(endpoint_config)
        )

    def test_lfbo_flash_starting_points_retain_all_live_families(self):
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        degree2 = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        degree1 = member(0.6, "fa4_2cta", "deg1_16x8", 1)
        two_cta_plain = member(0.7, "fa4_2cta", "1x1", 2)
        fa4 = member(0.8, "fa4", "1x1", 3)
        ws = member(0.9, "ws_overlap", "1x1", 4)
        clc = member(1.0, "fa4_clc", "1x1", 5)
        tma_4d = member(5.0, "fa4_tma_4d", "1x1", 6)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 8
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=None)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [
            degree2,
            degree1,
            two_cta_plain,
            fa4,
            ws,
            clc,
            tma_4d,
        ]
        paths = search._select_starting_paths()
        self.assertEqual(len(paths), search.copies)
        self.assertEqual(len({item.config for item, _constraints in paths}), 7)
        self.assertEqual(
            {
                item.config.config["cute_flash_pipeline_family"]
                for item, _constraints in paths
            },
            {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc", "fa4_tma_4d"},
        )
        self.assertEqual(
            sum(
                item.config.config["cute_flash_pipeline_family"] == "fa4_2cta"
                for item, _constraints in paths
            ),
            4,
        )
        self.assertEqual(
            paths[-1][1],
            (),
        )
        self.assertIs(paths[-1][0], degree2)
        self.assertEqual(
            dict(next(c for item, c in paths if item is two_cta_plain))[
                "cute_flash_pipeline_family"
            ],
            "fa4_2cta",
        )
        self.assertIn(degree1.config, {item.config for item, _ in paths})
        self.assertEqual(
            {
                item["family"]
                for item in search._autotune_metrics.search_phase_metrics[
                    "retained_families"
                ]
            },
            {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc", "fa4_tma_4d"},
        )
        retained_paths = [
            path
            for family in search._autotune_metrics.search_phase_metrics[
                "retained_families"
            ]
            for path in family["starting_paths"]
        ]
        self.assertEqual(
            [path["config_id"] for path in retained_paths if path["unrestricted"]],
            [canonical_config_id(degree2.config)],
        )
        self.assertTrue(
            all(
                family["score_compound_packet"] is None
                for family in search._autotune_metrics.search_phase_metrics[
                    "retained_families"
                ]
            )
        )

    def test_lfbo_flash_family_probe_covers_every_family_and_compound_leaf(self):
        def member(
            perf: float, family: str, packet: str, softmax_disc: bool, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        fa4 = member(1.0, "fa4", "1x1", True, 0)
        fa4_better_protocol = member(0.9, "fa4", "1x1", False, 1)
        ws = member(1.1, "ws_overlap", "1x1", True, 2)
        degree2 = member(0.7, "fa4_2cta", "deg2_16x6", False, 3)
        degree1 = member(0.8, "fa4_2cta", "deg1_16x8", False, 4)
        unqualified = member(0.1, "fa4_2cta", "deg2_16x6", False, 5)
        population = [
            fa4,
            fa4_better_protocol,
            ws,
            degree2,
            degree1,
            unqualified,
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search._flash_qualified_compound_config_ids = {
            flash_structural_leaf_from_config(degree2.config.config): {
                canonical_config_id(degree2.config)
            },
            flash_structural_leaf_from_config(degree1.config.config): {
                canonical_config_id(degree1.config)
            },
        }

        paths = search._flash_family_probe_paths(population)

        self.assertEqual(len(paths), 5)
        self.assertEqual(paths[-1], (degree2, None, (), True))
        self.assertEqual(
            {path[0].config for path in paths[:-1]},
            {
                fa4_better_protocol.config,
                ws.config,
                degree2.config,
                degree1.config,
            },
        )
        self.assertTrue(all(path[2] for path in paths[:-1]))
        self.assertNotIn(unqualified.config, {path[0].config for path in paths})

    def test_lfbo_flash_post_probe_scores_control_parent_promotion(self):
        def member(perf: float, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initially_slow = member(1.5, "fa4_2cta", 0)
        probe_winner = member(0.8, "fa4_2cta", 1)
        qualification_leader = member(1.0, "fa4", 2)
        unrestricted_family_base = member(2.0, "ws_overlap", 3)
        unrestricted_winner = member(0.5, "ws_overlap", 4)
        population = [
            initially_slow,
            probe_winner,
            qualification_leader,
            unrestricted_family_base,
            unrestricted_winner,
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=1)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search._flash_parent_score_config_ids = {
            canonical_config_id(member.config)
            for member in (
                initially_slow,
                probe_winner,
                qualification_leader,
                unrestricted_family_base,
            )
        }
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (unrestricted_winner, ()))
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertTrue(
            next(
                family["parent_promoted"]
                for family in retained
                if family["family"] == "fa4_2cta"
            )
        )
        self.assertFalse(
            next(
                family["parent_promoted"]
                for family in retained
                if family["family"] == "ws_overlap"
            )
        )

    def test_lfbo_flash_starting_points_keep_distinct_softmax_protocols(self):
        def member(perf: float, softmax_disc: bool) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=softmax_disc,
                ),
                status="ok",
            )

        whole_row = member(1.0, True)
        chunked = member(1.1, False)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 3
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [whole_row, chunked]

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (whole_row, ()))
        self.assertEqual(paths[0][0], whole_row)
        self.assertEqual(paths[1][0], chunked)
        self.assertEqual(
            dict(paths[1][1]),
            {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_softmax_disc": False,
            },
        )
        family = search._autotune_metrics.search_phase_metrics["retained_families"][0]
        self.assertTrue(family["score_softmax_disc"])
        self.assertEqual(
            {path["softmax_disc"] for path in family["starting_paths"]},
            {False, True},
        )

    def test_lfbo_flash_constructor_reserves_structural_starting_paths(self):
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None

        def initialize_pattern_search(
            search, *, kernel, args, copies: int, **_kwargs
        ) -> None:
            search.config_spec = kernel.config_spec
            search.settings = kernel.settings
            search.benchmark_provider = SimpleNamespace()
            search._terminal_refinement_members = None
            search.args = args
            search.copies = copies
            search.config_gen = SimpleNamespace(
                flash_structural_starting_path_limit=lambda **_kwargs: 17,
                flash_structural_family_probe_path_limit=lambda _cap, _generations: 18,
            )

        flash_kernel = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            settings=Settings(autotune_budget_seconds=None),
        )
        budgeted_flash_kernel = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            settings=Settings(autotune_budget_seconds=1.0),
        )
        multi_shape_args = _MultiShapeAutotuneArgs(
            cases=((Mock(), (object(),)),),
            aggregation="geomean",
            relative_to=None,
            cache_tag=None,
            workload_key=("test",),
        )
        non_flash_kernel = SimpleNamespace(
            config_spec=SimpleNamespace(cute_flash_search_enabled=False),
            settings=Settings(autotune_budget_seconds=None),
        )
        with (
            patch("helion.autotuner.surrogate_pattern_search.HAS_ML_DEPS", True),
            patch.object(PatternSearch, "__init__", initialize_pattern_search),
        ):
            flash_search = LFBOPatternSearch(
                flash_kernel,
                (),
                copies=2,
                flash_structural_search=policy,
            )
            budgeted_flash_search = LFBOPatternSearch(
                budgeted_flash_kernel,
                (),
                copies=2,
                flash_structural_search=policy,
            )
            multi_shape_search = LFBOPatternSearch(
                flash_kernel,
                multi_shape_args,
                copies=2,
                flash_structural_search=policy,
            )
            non_flash_search = LFBOPatternSearch(
                non_flash_kernel,
                (),
                copies=2,
                flash_structural_search=policy,
            )

        self.assertEqual(flash_search._flash_promoted_path_limit, 17)
        self.assertEqual(flash_search._flash_family_probe_path_limit, 18)
        self.assertEqual(flash_search.copies, 18)
        self.assertEqual(flash_search._terminal_refinement_members, {})
        self.assertIsNone(budgeted_flash_search._terminal_refinement_members)
        self.assertIsNone(multi_shape_search._terminal_refinement_members)
        self.assertEqual(non_flash_search.copies, 2)

    def test_lfbo_flash_starting_points_do_not_prune_close_sibling_leaf(self):
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        best_family = member(1.0, "fa4", "1x1", 0)
        best_family_second = member(1.001, "fa4", "1x1", 1)
        degree2 = member(1.01, "fa4_2cta", "deg2_16x6", 2)
        degree1 = member(1.011, "fa4_2cta", "deg1_16x8", 3)
        two_cta_plain = member(1.015, "fa4_2cta", "1x1", 4)
        clc = member(3.0, "fa4_clc", "1x1", 5)
        tma = member(3.1, "fa4_tma_4d", "1x1", 6)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 6
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [
            best_family,
            best_family_second,
            degree2,
            degree1,
            two_cta_plain,
            clc,
            tma,
        ]

        paths = search._select_starting_paths()

        selected = {item.config for item, _constraints in paths}
        self.assertEqual(paths[-1], (best_family, ()))
        self.assertEqual(
            selected,
            {
                best_family.config,
                degree2.config,
                degree1.config,
                two_cta_plain.config,
                best_family_second.config,
            },
        )
        self.assertNotIn(clc.config, selected)
        self.assertNotIn(tma.config, selected)
        self.assertEqual(paths[-1], (best_family, ()))
        self.assertEqual(
            dict(next(c for item, c in paths if item is degree2))[
                "cute_flash_exp2_packet"
            ],
            "deg2_16x6",
        )
        self.assertEqual(
            dict(next(c for item, c in paths if item is degree1))[
                "cute_flash_exp2_packet"
            ],
            "deg1_16x8",
        )

    def test_lfbo_flash_family_score_ignores_compound_leaf_count(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        many_leaf_family = [
            member(1.5, "fa4_2cta", "1x1", 0),
            member(0.9, "fa4_2cta", "deg2_16x6", 1),
            member(0.95, "fa4_2cta", "deg1_16x8", 2),
        ]
        ordinary_families = [
            member(perf, family, "1x1", index + 3)
            for index, (perf, family) in enumerate(
                (
                    (0.8, "fa4"),
                    (1.0, "ws_overlap"),
                    (1.1, "fa4_clc"),
                    (1.2, "fa4_tma_4d"),
                )
            )
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 5
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [*many_leaf_family, *ordinary_families]

        search._select_starting_paths()

        retained = {
            item["family"]
            for item in search._autotune_metrics.search_phase_metrics[
                "retained_families"
            ]
        }
        self.assertEqual(retained, {"fa4", "ws_overlap", "fa4_clc", "fa4_tma_4d"})
        self.assertNotIn("fa4_2cta", retained)

    def test_lfbo_flash_global_compound_winner_is_unrestricted(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        population = [
            winner,
            member(1.5, "fa4_2cta", "1x1", 1),
            member(1.0, "fa4", "1x1", 2),
            member(1.1, "ws_overlap", "1x1", 3),
            member(1.2, "fa4_clc", "1x1", 4),
            member(1.3, "fa4_tma_4d", "1x1", 5),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [
                path["config_id"]
                for family in retained
                for path in family["starting_paths"]
                if path["unrestricted"]
            ],
            [canonical_config_id(winner.config)],
        )
        self.assertEqual(
            {family["family"] for family in retained},
            {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc"},
        )
        self.assertFalse(
            next(
                family["parent_promoted"]
                for family in retained
                if family["family"] == "fa4_2cta"
            )
        )

    def test_lfbo_flash_dominated_compound_winner_gets_only_global_path(
        self,
    ) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        dominated_parent = member(3.0, "fa4_2cta", "1x1", 1)
        population = [
            winner,
            dominated_parent,
            member(1.0, "fa4", "1x1", 2),
            member(1.1, "ws_overlap", "1x1", 3),
            member(1.2, "fa4_clc", "1x1", 4),
            member(1.3, "fa4_tma_4d", "1x1", 5),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        self.assertNotIn(dominated_parent.config, {item.config for item, _ in paths})
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [family["family"] for family in retained],
            ["fa4_2cta", "fa4", "ws_overlap", "fa4_clc"],
        )
        self.assertFalse(retained[0]["parent_promoted"])
        self.assertEqual(len(retained[0]["starting_paths"]), 1)
        self.assertTrue(retained[0]["starting_paths"][0]["unrestricted"])
        self.assertTrue(all(family["parent_promoted"] for family in retained[1:]))

    def test_lfbo_flash_compound_only_family_gets_only_global_path(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        failed_parent = member(math.inf, "fa4_2cta", "1x1", 1)
        failed_parent.status = "error"
        population = [
            winner,
            failed_parent,
            member(1.0, "fa4", "1x1", 2),
            member(1.1, "ws_overlap", "1x1", 3),
            member(1.2, "fa4_clc", "1x1", 4),
            member(1.3, "fa4_tma_4d", "1x1", 5),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        self.assertNotIn(failed_parent.config, {item.config for item, _ in paths})
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [family["family"] for family in retained],
            ["fa4_2cta", "fa4", "ws_overlap", "fa4_clc"],
        )
        self.assertFalse(retained[0]["parent_promoted"])
        self.assertEqual(len(retained[0]["starting_paths"]), 1)
        self.assertEqual(sum(not constraints for _member, constraints in paths), 1)

    def test_lfbo_flash_sparse_successes_retain_each_compound_leaf(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        ordinary = member(1.0, "fa4", "1x1", 1)
        compound_only_filler = member(1.1, "fa4_2cta", "deg1_16x8", 2)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 5
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [winner, ordinary, compound_only_filler]

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        self.assertEqual(len(paths), 4)
        self.assertEqual(sum(not constraints for _member, constraints in paths), 1)
        self.assertIn(
            compound_only_filler.config, {member.config for member, _ in paths}
        )

    def test_lfbo_flash_dynamic_capacity_covers_asymmetric_promoted_families(
        self,
    ) -> None:
        def member(
            perf: float,
            family: str,
            packet: str,
            softmax_disc: bool,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        compound = [
            member(0.5, "fa4_2cta", "deg2_16x6", False, 0),
            member(0.6, "fa4_2cta", "deg1_16x8", False, 1),
        ]
        wide_family = [
            member(1.0, "fa4", "1x1", True, 2),
            member(1.2, "fa4", "1x1", True, 3),
            member(1.01, "fa4", "1x1", False, 4),
            member(1.21, "fa4", "1x1", False, 5),
        ]
        narrow_family = [
            member(1.1, "ws_overlap", "1x1", True, 6),
            member(1.3, "ws_overlap", "1x1", True, 7),
        ]
        unpromoted = member(5.0, "fa4_clc", "1x1", True, 8)
        population = [*compound, *wide_family, *narrow_family, unpromoted]
        catalog = list(
            dict.fromkeys(
                flash_structural_leaf_from_config(item.config.config)
                for item in population
            )
        )
        self.assertNotIn(None, catalog)

        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        policy = replace(
            policy,
            retained_families=2,
            retained_candidates_per_leaf=2,
        )
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        with patch.object(
            generation, "flash_structural_leaf_catalog", return_value=catalog
        ):
            capacity = generation.flash_structural_starting_path_limit(
                minimum=1,
                retained_families=policy.retained_families,
                retained_candidates_per_leaf=policy.retained_candidates_per_leaf,
            )

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = capacity
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = policy
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        selected = {item.config for item, _constraints in paths}
        self.assertEqual(capacity, 8)
        self.assertEqual(len(paths), capacity)
        self.assertEqual(
            selected,
            {
                item.config
                for item in population
                if item not in (wide_family[3], unpromoted)
            },
        )
        self.assertEqual(paths[-1], (compound[0], ()))
        self.assertTrue({item.config for item in compound} <= selected)
        self.assertEqual(
            {
                (
                    item.config.config["cute_flash_pipeline_family"],
                    item.config.config["cute_flash_softmax_disc"],
                )
                for item, _constraints in paths
                if item.config.config["cute_flash_exp2_packet"] == "1x1"
            },
            {("fa4", True), ("fa4", False), ("ws_overlap", True)},
        )

    def test_lfbo_flash_starting_points_replace_dominated_family(self):
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        best = member(1.0, "fa4", "1x1", 0)
        best_sibling = member(1.1, "fa4", "1x1", 1)
        best_third = member(1.15, "fa4", "1x1", 5)
        close_leaf = member(1.2, "fa4", "deg2_16x6", 2)
        competitive = member(1.3, "fa4_2cta", "1x1", 3)
        dominated = member(2.01, "ws_overlap", "1x1", 4)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 5
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [
            best,
            best_sibling,
            best_third,
            close_leaf,
            competitive,
            dominated,
        ]

        paths = search._select_starting_paths()

        selected = {item.config for item, _constraints in paths}
        self.assertEqual(
            selected,
            {
                best.config,
                best_sibling.config,
                close_leaf.config,
                competitive.config,
            },
        )
        self.assertEqual(len(paths), 5)
        self.assertEqual(paths[-1], (best, ()))
        self.assertNotIn(best_third.config, selected)
        self.assertNotIn(dominated.config, selected)
        self.assertEqual(
            {
                item["family"]
                for item in search._autotune_metrics.search_phase_metrics[
                    "retained_families"
                ]
            },
            {"fa4", "fa4_2cta"},
        )

    def test_lfbo_flash_starting_points_break_performance_ties_by_config_id(self):
        def member(wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        population = [member(wait_hint) for wait_hint in (3, 1, 2)]
        expected = sorted(
            population, key=lambda item: canonical_config_id(item.config)
        )[:2]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 3
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(
            [member for member, _constraints in paths],
            [*expected, expected[0]],
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [path["config_id"] for path in retained[0]["starting_paths"]],
            [
                canonical_config_id(expected[0].config),
                canonical_config_id(expected[1].config),
                canonical_config_id(expected[0].config),
            ],
        )
        self.assertEqual(
            [path["unrestricted"] for path in retained[0]["starting_paths"]],
            [False, False, True],
        )

    def test_lfbo_flash_starting_points_retain_alternate_pipeline_depth(self):
        stage_key = "cute_flash_kv_stage"

        def member(perf: float, stage: int, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        population = [
            member(1.0, 2, 0),
            member(1.01, 2, 1),
            member(1.1, 3, 2),
            member(1.2, 4, 3),
            member(1.3, 5, 4),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population
        lanes = tuple((stage_key, stage) for stage in (2, 3, 4, 5))
        leaf = flash_structural_leaf_from_config(population[0].config.config)
        assert leaf is not None
        search._flash_qualified_pipeline_lanes = {leaf: lanes}
        search._flash_pipeline_lanes = Mock(
            side_effect=AssertionError("must reuse qualified lane catalog")
        )

        paths = search._select_starting_paths()

        self.assertEqual(
            [item.config.config[stage_key] for item, _ in paths], [2, 3, 2]
        )
        self.assertEqual(paths[-1], (population[0], ()))
        self.assertEqual(
            [
                dict(constraints)[stage_key]
                for _item, constraints in paths
                if stage_key in dict(constraints)
            ],
            [3],
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"][0]
        self.assertEqual(
            [path["pipeline_lane"] for path in retained["starting_paths"]],
            [
                None,
                {"key": stage_key, "value": 3},
                None,
            ],
        )
        search._flash_pipeline_lanes.assert_not_called()

    def test_lfbo_flash_global_lane_alternate_is_not_crowded_out(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float,
            family: str,
            packet: str,
            softmax_disc: bool,
            stage: int,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        global_winner = member(1.0, "fa4_2cta", "1x1", True, 2, 0)
        lane_alternate = member(1.2, "fa4_2cta", "1x1", True, 3, 1)
        compound_siblings = [
            member(1.001, "fa4_2cta", "deg2_16x6", True, 2, 2),
            member(1.002, "fa4_2cta", "deg1_16x8", True, 2, 3),
        ]
        family_perfs = (
            ("fa4_2cta", 1.01),
            ("fa4", 1.05),
            ("ws_overlap", 1.06),
            ("fa4_clc", 1.07),
            ("fa4_tma_4d", 1.08),
        )
        ordinary_members: dict[tuple[str, bool], PopulationMember] = {
            ("fa4_2cta", True): global_winner
        }
        wait_hint = 4
        for family, perf in family_perfs:
            for softmax_disc in (True, False):
                if (family, softmax_disc) in ordinary_members:
                    continue
                ordinary_members[(family, softmax_disc)] = member(
                    perf + (0.001 if not softmax_disc else 0.0),
                    family,
                    "1x1",
                    softmax_disc,
                    2,
                    wait_hint,
                )
                wait_hint += 1
        population = [
            global_winner,
            lane_alternate,
            *compound_siblings,
            *(
                ordinary
                for key, ordinary in ordinary_members.items()
                if key != ("fa4_2cta", True)
            ),
        ]
        global_leaf = flash_structural_leaf_from_config(global_winner.config.config)
        assert global_leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        assert search.flash_structural_search is not None
        search.copies = search.flash_structural_search.starting_paths
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population
        search._flash_qualified_pipeline_lanes = {
            global_leaf: ((stage_key, 2), (stage_key, 3))
        }

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (global_winner, ()))
        self.assertEqual(len(paths), 12)
        lane_path = next(path for path in paths if path[0] is lane_alternate)
        self.assertEqual(dict(lane_path[1])[stage_key], 3)
        selected = {item.config for item, _constraints in paths}
        self.assertIn(compound_siblings[0].config, selected)
        self.assertIn(compound_siblings[1].config, selected)
        retained_parent_families = {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc"}
        self.assertEqual(
            {
                (
                    item.config.config["cute_flash_pipeline_family"],
                    item.config.config.get("cute_flash_softmax_disc", True),
                )
                for item, _constraints in paths
                if item.config.config["cute_flash_exp2_packet"] == "1x1"
            },
            {
                (family, softmax_disc)
                for family in retained_parent_families
                for softmax_disc in (True, False)
            },
        )
        self.assertFalse(
            {
                ordinary_members[("fa4_tma_4d", True)].config,
                ordinary_members[("fa4_tma_4d", False)].config,
            }
            & selected
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(len(retained), 4)
        global_paths = retained[0]["starting_paths"]
        lane_metric = next(
            path
            for path in global_paths
            if path["config_id"] == canonical_config_id(lane_alternate.config)
        )
        self.assertEqual(
            lane_metric["pipeline_lane"],
            {"key": stage_key, "value": 3},
        )

    def test_lfbo_flash_unpromoted_global_family_preserves_compound_slots(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float,
            family: str,
            packet: str,
            softmax_disc: bool,
            stage: int,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        compound_global = member(0.5, "fa4_2cta", "deg2_16x6", True, 2, 0)
        ordinary_kv2 = member(3.0, "fa4_2cta", "1x1", True, 2, 1)
        ordinary_kv3 = member(3.1, "fa4_2cta", "1x1", True, 3, 2)
        compound_filler = member(0.6, "fa4", "deg2_16x6", True, 2, 3)
        promoted_family_perfs = (
            ("fa4", 1.0),
            ("ws_overlap", 1.1),
            ("fa4_clc", 1.2),
            ("fa4_tma_4d", 1.3),
        )
        promoted_ordinary: dict[tuple[str, bool], PopulationMember] = {}
        wait_hint = 4
        for family, perf in promoted_family_perfs:
            for softmax_disc in (True, False):
                promoted_ordinary[(family, softmax_disc)] = member(
                    perf + (0.01 if not softmax_disc else 0.0),
                    family,
                    "1x1",
                    softmax_disc,
                    2,
                    wait_hint,
                )
                wait_hint += 1
        promoted_secondaries = {
            family: member(
                perf + 0.2,
                family,
                "1x1",
                True,
                3,
                wait_hint + index,
            )
            for index, (family, perf) in enumerate(promoted_family_perfs)
        }
        population = [
            compound_global,
            ordinary_kv2,
            ordinary_kv3,
            compound_filler,
            *promoted_ordinary.values(),
            *promoted_secondaries.values(),
        ]
        ordinary_leaf = flash_structural_leaf_from_config(ordinary_kv2.config.config)
        assert ordinary_leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        assert search.flash_structural_search is not None
        search.copies = search.flash_structural_search.starting_paths + 1
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population
        search._flash_qualified_pipeline_lanes = {
            ordinary_leaf: ((stage_key, 2), (stage_key, 3))
        }

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (compound_global, ()))
        self.assertEqual(len(paths), 15)
        selected = {item.config for item, _constraints in paths}
        self.assertNotIn(ordinary_kv3.config, selected)
        self.assertNotIn(ordinary_kv2.config, selected)
        self.assertIn(compound_filler.config, selected)
        self.assertEqual(
            selected,
            {
                compound_global.config,
                compound_filler.config,
                *(item.config for item in promoted_ordinary.values()),
                *(item.config for item in promoted_secondaries.values()),
            },
        )
        self.assertEqual(
            {
                (
                    item.config.config["cute_flash_pipeline_family"],
                    item.config.config["cute_flash_softmax_disc"],
                )
                for item, _constraints in paths
                if item.config.config["cute_flash_pipeline_family"]
                in dict(promoted_family_perfs)
            },
            set(promoted_ordinary),
        )
        self.assertEqual(
            [
                item
                for item, _constraints in paths
                if item.config.config["cute_flash_exp2_packet"] != "1x1"
            ],
            [compound_global, compound_filler, compound_global],
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        global_paths = retained[0]["starting_paths"]
        self.assertFalse(retained[0]["parent_promoted"])
        self.assertEqual(
            {family["family"] for family in retained if family["parent_promoted"]},
            set(dict(promoted_family_perfs)),
        )
        self.assertFalse(
            any(
                path["config_id"] == canonical_config_id(ordinary_kv3.config)
                for path in global_paths
            )
        )
        compound_metric = next(
            path
            for family in retained
            for path in family["starting_paths"]
            if path["config_id"] == canonical_config_id(compound_filler.config)
        )
        self.assertIsNone(compound_metric["pipeline_lane"])

    def test_lfbo_flash_pipeline_lane_neighbor_limits_preserve_budget(self):
        quotas = [
            (("cute_flash_kv_stage", 2), 1),
            (("cute_flash_kv_stage", 3), 2),
            (("cute_flash_kv_stage", 4), 1),
        ]
        limits = LFBOPatternSearch._flash_lane_neighbor_limits(quotas, 300)
        self.assertEqual(sum(limits), 300)
        self.assertEqual(limits, [75, 150, 75])

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        self.assertEqual(search._flash_qualification_neighbor_limit(), 200)
        search.num_neighbors_cap = 3
        self.assertEqual(search._flash_qualification_neighbor_limit(), 3)

    def test_lfbo_flash_pipeline_lane_passes_measure_then_condition_every_value(
        self,
    ):
        lanes = tuple(("cute_flash_kv_stage", value) for value in range(2, 11))

        passes = LFBOPatternSearch._flash_lane_qualification_passes(
            lanes,
            candidate_limit=4,
            conditional_candidates_per_lane=1,
            minimum_passes=2,
        )

        self.assertEqual(len(passes), 6)
        self.assertTrue(
            all(kind == "witness" for batch in passes[:3] for kind, _ in batch)
        )
        self.assertTrue(
            all(kind == "conditional" for batch in passes[3:] for kind, _ in batch)
        )
        for kind in ("witness", "conditional"):
            self.assertEqual(
                [lane for batch in passes for job, lane in batch if job == kind],
                list(lanes),
            )

    def test_lfbo_flash_unmeasured_schedule_anchor_records_provenance(self):
        def member(
            perf: float | None, softmax_disc: bool, flat_value: bool
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[flat_value],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=softmax_disc,
                ),
                status="unknown" if perf is None else "ok",
            )

        initial = member(1.0, True, True)
        anchor = member(None, False, False)
        leaves = [
            flash_structural_leaf_from_config(item.config.config)
            for item in (initial, anchor)
        ]
        assert all(leaf is not None for leaf in leaves)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flatten=lambda config: [config.config["cute_flash_softmax_disc"]],
            flash_structural_leaf_catalog=lambda: leaves,
            flash_low_confound_schedule_anchor_configs=lambda: [
                anchor.config,
                initial.config,
            ],
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [
                initial.config,
                anchor.config,
            ],
        )
        search.copies = 2
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = replace(policy, qualification_rounds=0)
        search.population = [initial]
        search.initial_population = 2
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.make_unbenchmarked = Mock(return_value=anchor)
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("anchor-only test must not generate neighbors")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            self.assertEqual(members, [anchor])
            self.assertEqual(desc, "Low-confound schedule anchors:")
            anchor.perfs = [0.75]
            anchor.status = "ok"

        search.benchmark_population = Mock(side_effect=benchmark)
        visited = {initial.config}

        self.assertEqual(search._run_flash_structural_qualification(visited), 0)

        self.assertIn(anchor.config, visited)
        search.make_unbenchmarked.assert_called_once_with([False])
        search.set_generation.assert_called_once_with(0)
        search.benchmark_population.assert_called_once()
        search._pruned_pattern_search_from.assert_not_called()
        self.assertEqual(search.train_x, [[False]])
        self.assertEqual(search.train_y, [0.75])
        self.assertEqual(search.train_configs, [anchor.config])

        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertEqual(
            metrics["schedule_anchor_design_source"],
            "live family x ordinary packet x softmax protocol from fragment defaults",
        )
        self.assertTrue(metrics["schedule_anchor_pass_planned"])
        self.assertTrue(metrics["schedule_anchor_pass_started"])
        self.assertTrue(metrics["schedule_anchor_complete"])
        self.assertEqual(metrics["schedule_anchor_count"], 2)
        self.assertEqual(metrics["qualification_passes_planned"], 1)
        self.assertEqual(metrics["qualification_passes_started"], 1)
        self.assertEqual(metrics["qualification_passes_completed"], 1)
        self.assertTrue(metrics["completed"])
        self.assertEqual(
            [result["config_id"] for result in metrics["schedule_anchor_results"]],
            [
                canonical_config_id(anchor.config),
                canonical_config_id(initial.config),
            ],
        )
        result = metrics["schedule_anchor_results"][0]
        self.assertEqual(result["config_id"], canonical_config_id(anchor.config))
        self.assertFalse(result["softmax_disc"])
        self.assertEqual(result["attempt_perf"], 0.75)
        self.assertEqual(result["selection_perf"], 0.75)
        self.assertEqual(result["measurement_pass_index"], 1)
        self.assertEqual(
            metrics["measurement_timeline"][1],
            {
                "pass_index": 1,
                "updates": [
                    {
                        "config_id": canonical_config_id(anchor.config),
                        "attempt_perf": 0.75,
                        "selection_perf": 0.75,
                        "status": "ok",
                        "source_hash": None,
                    }
                ],
            },
        )

    def test_lfbo_flash_phase_records_initial_generation_order(self):
        def member(wait_hint: int, perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        generated = [member(0, 2.0), member(1, 1.0)]
        leaf = flash_structural_leaf_from_config(generated[0].config.config)
        assert leaf is not None
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flatten=lambda config: [config.config["cute_flash_wait_hint"]],
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=lambda: [
                item.config for item in generated
            ],
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in generated
            ],
        )
        search.copies = 2
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = replace(policy, qualification_rounds=0)
        search.population = list(reversed(generated))
        search.initial_population = len(generated)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search.train_source_hashes = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.make_unbenchmarked = Mock(
            side_effect=AssertionError("all exact-space configs are already measured")
        )
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted exact space needs no neighbors")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search.benchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in generated},
                initial_population=generated,
            ),
            0,
        )

        expected_ids = [canonical_config_id(item.config) for item in generated]
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["initial_config_ids"], expected_ids)
        self.assertEqual(
            [result["config_id"] for result in metrics["initial_results"]],
            expected_ids,
        )
        self.assertEqual(metrics["leaf_results"][0]["initial_config_ids"], expected_ids)
        self.assertEqual(search.population, list(reversed(generated)))

    def test_lfbo_flash_pipeline_lane_conditional_count_is_linear(self):
        stage_key = "cute_flash_kv_stage"

        def member(perf: float | None, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(1.0, 0)
        candidate_groups = [
            (member(None, 1), member(None, 2)),
            (member(None, 3), member(None, 4)),
        ]
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            conditional_candidates_per_pipeline_lane=2,
        )
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        selected_limits: list[int] = []

        def qualification_path(
            _index, current, _visited, _constraints, *, selected_limit, **_kwargs
        ):
            selected_limits.append(selected_limit)
            candidates = candidate_groups[len(selected_limits) - 1]
            return iter(((current, *candidates[: selected_limit - 1]),))

        def conditional_candidates(members, _leaf, constraints, known, limit):
            parent = members[0]
            added = next(
                qualification_path(0, parent, known, constraints, selected_limit=2)
            )
            return parent, list(added[1:]), {}

        search._flash_conditional_parent_candidates = conditional_candidates

        def benchmark(members, *, desc):
            self.assertEqual(desc, "Structural qualification 2:")
            self.assertCountEqual(
                members,
                [candidate_groups[0][0], candidate_groups[1][0]],
            )
            for candidate in members:
                candidate.perfs = [2.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )
        self.assertEqual(selected_limits, [2, 2])
        lane = search._autotune_metrics.search_phase_metrics["leaf_results"][0][
            "pipeline_lanes"
        ][0]
        self.assertEqual(len(lane["conditional_candidate_ids"]), 2)
        self.assertEqual(
            search._autotune_metrics.search_phase_metrics["candidate_count"], 2
        )
        self.assertTrue(lane["complete"])

    def test_lfbo_flash_exact_lane_exhaustion_skips_conditional_child(self):
        stage_key = "cute_flash_kv_stage"

        def member(wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0 + wait_hint],
                flat_values=[2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = [member(0), member(1)]
        leaf = flash_structural_leaf_from_config(initial[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in initial
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = initial
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted lane must not generate a child")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial}
            ),
            2,
        )
        search._pruned_pattern_search_from.assert_not_called()
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertTrue(metrics["exact_space_enumerated"])
        self.assertTrue(metrics["exact_space_exhausted"])
        self.assertEqual(
            metrics["exact_space_config_ids"],
            [canonical_config_id(item.config) for item in initial],
        )
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertTrue(lane["space_exhausted"])
        self.assertEqual(lane["space_config_count"], 2)
        self.assertFalse(lane["conditional_required"])
        self.assertEqual(lane["conditional_candidate_ids"], [])
        self.assertTrue(lane["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_conditional_ids_never_relabel_initial_configs(self):
        stage_key = "cute_flash_kv_stage"
        initial = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[2, 0],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=0,
            ),
            status="ok",
        )
        duplicate = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[2, 0],
            config=copy.deepcopy(initial.config),
            status="ok",
        )
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_conditional_parent_candidates = Mock(
            return_value=(initial, [duplicate], {})
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertTrue(lane["conditional_required"])
        self.assertEqual(lane["conditional_candidate_ids"], [])
        self.assertFalse(lane["complete"])
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["candidate_count"], 0)

    def test_lfbo_flash_alias_collapsed_exact_space_uses_raw_initial_budget(self):
        initial = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[0],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_wait_hint=0,
            ),
            status="ok",
        )
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation.flat_spec = [EnumFragment(tuple(range(7)))]
        generation.compiler_coverage_enabled = True
        generation._override_values = {}
        # The one-fragment stub carries no pipeline-family axis, so the exact
        # enumeration runs as one global product.
        generation._key_to_flat_indices = {}
        generation.unflatten = lambda _flat: initial.config
        self.assertIsNone(generation.flash_exact_effective_search_space_configs(1))
        self.assertEqual(
            generation.flash_exact_effective_search_space_configs(7),
            [initial.config],
        )
        exact_space = Mock(wraps=generation.flash_exact_effective_search_space_configs)
        generation.flash_exact_effective_search_space_configs = exact_space
        generation.encode_config = lambda flat: flat
        generation.flash_structural_leaf_catalog = lambda: [leaf]
        generation.flash_low_confound_schedule_anchor_configs = list
        generation.flash_pipeline_lane_catalog = lambda: {leaf: ()}
        generation.flash_clc_lane_catalog = dict
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = generation
        search.copies = 5
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = 7
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted leaf must not generate a child")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 0
        )
        search._pruned_pattern_search_from.assert_not_called()
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertTrue(metrics["exact_space_exhausted"])
        self.assertEqual(metrics["exact_space_raw_budget"], 7)
        exact_space.assert_called_once_with(7)
        self.assertEqual(metrics["qualification_passes_planned"], 0)
        leaf_metric = metrics["leaf_results"][0]
        self.assertTrue(leaf_metric["space_exhausted"])
        self.assertEqual(leaf_metric["space_config_count"], 1)
        self.assertFalse(leaf_metric["ordinary_search_required"])
        self.assertEqual(leaf_metric["rounds"], [])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_no_lane_exhaustion_is_leaf_local(self):
        def member(perf: float | None, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        exhausted = member(1.0, "fa4", 0)
        searchable = member(5.0, "ws_overlap", 0)
        children = [member(None, "ws_overlap", wait_hint) for wait_hint in (1, 2)]
        exhausted_leaf = flash_structural_leaf_from_config(exhausted.config.config)
        searchable_leaf = flash_structural_leaf_from_config(searchable.config.config)
        assert exhausted_leaf is not None and searchable_leaf is not None
        leaves = [exhausted_leaf, searchable_leaf]
        exact_space = [
            exhausted.config,
            searchable.config,
            *(child.config for child in children),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: leaves,
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: exact_space,
        )
        search.copies = 5
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [exhausted, searchable]
        search.initial_population = len(exact_space)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        generated: list[PopulationMember] = []

        def qualification_path(_index, current, *_args, **_kwargs):
            self.assertEqual(
                current.config.config["cute_flash_pipeline_family"], "ws_overlap"
            )
            child = children[len(generated)]
            generated.append(child)
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, **_kwargs):
            self.assertEqual(members, [children[len(generated) - 1]])
            members[0].perfs = [2.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {exhausted.config, searchable.config}
            ),
            2,
        )
        self.assertEqual(generated, children)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["exact_space_exhausted"])
        leaf_results = {result["family"]: result for result in metrics["leaf_results"]}
        exhausted_metric = leaf_results["fa4"]
        self.assertTrue(exhausted_metric["space_exhausted"])
        self.assertFalse(exhausted_metric["ordinary_search_required"])
        self.assertEqual(
            [item["neighbor_generation_limit"] for item in exhausted_metric["rounds"]],
            [0, 0],
        )
        searchable_metric = leaf_results["ws_overlap"]
        self.assertFalse(searchable_metric["space_exhausted"])
        self.assertTrue(searchable_metric["ordinary_search_required"])
        self.assertEqual(
            [item["neighbor_generation_limit"] for item in searchable_metric["rounds"]],
            [200, 200],
        )
        self.assertTrue(metrics["completed"])

    def test_flash_clc_refinements_cover_every_legal_divisor(self):
        for num_bh in (96, 120, 360, 720):
            with self.subTest(num_bh=num_bh):
                legal = _flash_clc_heads_per_batch_candidates(num_bh)
                anchors = _flash_clc_heads_per_batch_coverage_candidates(num_bh)
                refinements = _flash_log_maximin_refinements(legal, anchors)

                self.assertEqual({*anchors, *refinements}, set(legal))
                self.assertFalse(set(anchors) & set(refinements))
                self.assertEqual(len(refinements), len(legal) - len(anchors))

    def test_flash_clc_refinements_keep_deterministic_log_maximin_order(self):
        legal = _flash_clc_heads_per_batch_candidates(720)
        anchors = _flash_clc_heads_per_batch_coverage_candidates(720)
        self.assertEqual(
            _flash_log_maximin_refinements(legal, anchors),
            (
                2,
                16,
                90,
                240,
                3,
                36,
                8,
                4,
                45,
                180,
                20,
                72,
                120,
                6,
                12,
                30,
                9,
                18,
                40,
                80,
                48,
                15,
            ),
        )
        self.assertEqual(_flash_log_maximin_refinements((1, 2, 4), ()), (1, 2, 4))

    def test_flash_clc_generation_catalog_witnesses_and_fixed_overrides(self):
        for num_bh in (96, 120):
            with self.subTest(num_bh=num_bh):
                with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
                    spec = ConfigSpec(
                        backend=CuteBackend(),
                        target_device_capability=(10, 0),
                        num_sm=148,
                    )
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=64,
                    num_kv=512,
                    num_bh=num_bh,
                    dtype=torch.float16,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    standard_dense_output=True,
                )
                fragments = cute_flash.flash_autotune_fragments(
                    64,
                    512,
                    num_bh=num_bh,
                    dtype=torch.float16,
                    standard_dense_output=True,
                    pipeline_family_override="fa4_clc",
                )
                overrides = {
                    key: fragment.default()
                    for key, fragment in fragments.items()
                    if key != cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY
                }
                overrides[cute_flash.FLASH_PIPELINE_FAMILY_KEY] = "fa4_clc"
                generation = spec.create_config_generation(overrides=overrides)

                catalogs = generation.flash_clc_lane_catalog()
                self.assertEqual(len(catalogs), 1)
                leaf, catalog = next(iter(catalogs.items()))
                legal = _flash_clc_heads_per_batch_candidates(num_bh)
                anchors = _flash_clc_heads_per_batch_coverage_candidates(num_bh)
                self.assertEqual(catalog["legal_values"], legal)
                self.assertEqual(catalog["anchor_values"], anchors)
                self.assertEqual(
                    catalog["refinement_values"],
                    _flash_log_maximin_refinements(legal, anchors),
                )
                self.assertEqual(set(catalog["attempted_values"]), set(legal))

                witnesses = generation.flash_clc_lane_witnesses()
                self.assertEqual(
                    {
                        value
                        for witness_leaf, value in witnesses
                        if witness_leaf == leaf
                    },
                    set(legal),
                )
                for value in legal:
                    witness = witnesses[(leaf, value)]
                    _flat, canonical = generation.canonicalize_flat(
                        generation.flatten(witness)
                    )
                    self.assertEqual(
                        flash_structural_leaf_from_config(canonical.config), leaf
                    )
                    self.assertEqual(
                        canonical.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
                        value,
                    )

                refinement = catalog["refinement_values"][0]
                fixed_generation = spec.create_config_generation(
                    overrides={
                        **overrides,
                        cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY: refinement,
                    }
                )
                _flat, fixed = fixed_generation.canonicalize_flat(
                    fixed_generation.default_flat()
                )
                self.assertEqual(
                    fixed.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
                    refinement,
                )
                self.assertEqual(fixed_generation.flash_clc_lane_catalog(), {})

    def test_lfbo_flash_structural_transfers_preserve_pipeline_depths(self):
        with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
            spec = ConfigSpec(
                backend=CuteBackend(),
                target_device_capability=(10, 0),
                num_sm=148,
            )
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=512,
            num_bh=96,
            dtype=torch.float16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            standard_dense_output=True,
            # The compound variants below exist on the fast_math surface only.
            fast_math=True,
        )

        def make_search(generation):
            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.config_gen = generation

            def make_unbenchmarked(flat):
                canonical_flat, config = generation.canonicalize_flat(flat)
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[],
                    flat_values=canonical_flat,
                    config=config,
                    status="ok",
                )

            search.make_unbenchmarked = make_unbenchmarked
            return search

        def make_member(generation, config):
            flat, canonical = generation.canonicalize_flat(generation.flatten(config))
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0],
                flat_values=flat,
                config=canonical,
                status="ok",
            )

        clc_generation = spec.create_config_generation(
            overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc"}
        )
        clc_search = make_search(clc_generation)
        divisor_member = make_member(
            clc_generation,
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4_clc",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_s_stage=2,
                cute_flash_clc_heads_per_batch=2,
                cute_flash_epi_tma=True,
                cute_flash_epi_stg=False,
            ),
        )
        for depth in (11, 12):
            with self.subTest(depth=depth):
                depth_member = make_member(
                    clc_generation,
                    helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4_clc",
                        cute_flash_exp2_packet="1x1",
                        cute_flash_kv_stage=depth,
                        cute_flash_s_stage=2,
                        cute_flash_clc_heads_per_batch=1,
                        cute_flash_epi_tma=False,
                        cute_flash_epi_stg=False,
                    ),
                )
                leaf = flash_structural_leaf_from_config(depth_member.config.config)
                assert leaf is not None

                old_orientation = clc_search._flash_config_variant(
                    divisor_member,
                    {
                        cute_flash.FLASH_KV_STAGE_KEY: depth,
                        cute_flash.FLASH_S_STAGE_KEY: 2,
                    },
                    expected_leaf=leaf,
                )
                self.assertIsNotNone(old_orientation)
                assert old_orientation is not None
                self.assertEqual(
                    old_orientation.config.config[cute_flash.FLASH_KV_STAGE_KEY],
                    10,
                )
                with patch.object(
                    clc_search,
                    "_flash_config_variant",
                    return_value=old_orientation,
                ):
                    self.assertIsNone(
                        clc_search._flash_clc_depth_variant(
                            depth_member,
                            2,
                            expected_leaf=leaf,
                        )
                    )

                combined = clc_search._flash_clc_depth_variant(
                    depth_member,
                    2,
                    expected_leaf=leaf,
                )
                self.assertIsNotNone(combined)
                assert combined is not None
                self.assertEqual(
                    combined.config.config[cute_flash.FLASH_KV_STAGE_KEY], depth
                )
                self.assertEqual(
                    combined.config.config[cute_flash.FLASH_S_STAGE_KEY], 2
                )
                self.assertEqual(
                    combined.config.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
                    2,
                )
                self.assertFalse(combined.config.config[cute_flash.FLASH_EPI_TMA_KEY])

        with patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}):
            compound_generation = spec.create_config_generation(
                overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"}
            )
            compound_search = make_search(compound_generation)
            source = make_member(
                compound_generation,
                helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_2cta",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=6,
                    cute_flash_s_stage=2,
                    cute_flash_softmax_disc=True,
                    cute_flash_stat_transport="ring2",
                ),
            )
            raw_compound = copy.deepcopy(source.config)
            raw_compound.config[cute_flash.FLASH_EXP2_PACKET_KEY] = "deg2_16x6"
            _flat, compound_config = compound_generation.canonicalize_flat(
                compound_generation.flatten(raw_compound)
            )
            compound_leaf = flash_structural_leaf_from_config(compound_config.config)
            assert compound_leaf is not None
            self.assertEqual(compound_config.config[cute_flash.FLASH_KV_STAGE_KEY], 2)
            self.assertIsNone(
                compound_search._flash_compound_variant(
                    source,
                    "deg2_16x6",
                    expected_leaf=compound_leaf,
                )
            )

        offset_generation = spec.create_config_generation(
            overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"}
        )
        offset_search = make_search(offset_generation)
        offset_source = make_member(
            offset_generation,
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4_2cta",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_s_stage=2,
                cute_flash_softmax_disc=False,
                cute_flash_e2e_schedule="16/6",
                cute_flash_e2e_offset=12,
            ),
        )
        raw_offset_target = copy.deepcopy(offset_source.config)
        raw_offset_target.config[cute_flash.FLASH_EXP2_PACKET_KEY] = "deg1_8x2_corr10"
        _flat, offset_target_config = offset_generation.canonicalize_flat(
            offset_generation.flatten(raw_offset_target)
        )
        offset_target_leaf = flash_structural_leaf_from_config(
            offset_target_config.config
        )
        assert offset_target_leaf is not None
        offset_target = offset_search._flash_compound_variant(
            offset_source,
            "deg1_8x2_corr10",
            expected_leaf=offset_target_leaf,
        )
        self.assertIsNotNone(offset_target)
        assert offset_target is not None
        self.assertEqual(
            offset_source.config.config[cute_flash.FLASH_E2E_OFFSET_KEY], 12
        )
        self.assertEqual(
            offset_target.config.config[cute_flash.FLASH_E2E_OFFSET_KEY], 4
        )

    def test_flash_initial_prefix_second_witnesses_only_ordinary_leaves(self):
        # The compound (polynomial exp2 packet) leaves this test tells apart
        # from the ordinary ones exist on the fast_math surface only.
        with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
            spec = ConfigSpec(
                backend=CuteBackend(),
                target_device_capability=(10, 0),
                num_sm=148,
            )
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=512,
            num_bh=64,
            dtype=torch.float16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            standard_dense_output=True,
            fast_math=True,
        )
        fragments = cute_flash.flash_autotune_fragments(
            64,
            512,
            num_bh=64,
            dtype=torch.float16,
            standard_dense_output=True,
            fast_math=True,
        )
        ws_values = cute_flash.flash_effective_config_values(
            cute_flash.resolve_flash_config(
                64,
                512,
                {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
                dtype=torch.float16,
                num_bh=64,
                standard_dense_output=True,
                fast_math=True,
            )
        )
        live_prefix_keys = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY,
            cute_flash.FLASH_EXP2_PACKET_KEY,
            cute_flash.FLASH_WAIT_HINT_KEY,
            *(
                key
                for key, fragment in fragments.items()
                if ws_values.get(key) != fragment.default()
            ),
        }
        generation = spec.create_config_generation(
            overrides={
                key: fragment.default()
                for key, fragment in fragments.items()
                if key not in live_prefix_keys
            }
        )

        rows = generation.flash_deterministic_population_configs()
        leaves = generation.flash_structural_leaf_catalog()
        parent_count = generation.flash_structural_parent_coverage_prefix_count()
        qualification_count = generation.flash_structural_qualification_prefix_count()
        ordinary_leaves = [leaf for leaf in leaves if leaf.compound_exp2_packet is None]
        compound_leaves = [
            leaf for leaf in leaves if leaf.compound_exp2_packet is not None
        ]

        def prefix_leaf_counts(limit: int) -> dict[object, int]:
            counts = dict.fromkeys(leaves, 0)
            for config in rows[:limit]:
                leaf = flash_structural_leaf_from_config(config.config)
                if leaf in counts:
                    counts[leaf] += 1
            return counts

        parent_counts = prefix_leaf_counts(parent_count)
        qualification_counts = prefix_leaf_counts(qualification_count)
        self.assertTrue(compound_leaves)
        self.assertTrue(all(parent_counts[leaf] == 1 for leaf in leaves))
        # A leaf with a single reachable normalized config under these
        # overrides is complete after its one parent row.
        singleton_leaves = set(generation.flash_structural_singleton_leaves())
        self.assertEqual(singleton_leaves, set())
        self.assertTrue(
            all(qualification_counts[leaf] == 2 for leaf in ordinary_leaves)
        )
        self.assertTrue(
            all(qualification_counts[leaf] == 1 for leaf in compound_leaves)
        )
        self.assertEqual(
            generation.flash_structural_coverage_underqualified_values(), []
        )
        self.assertEqual(
            qualification_count,
            parent_count + len(ordinary_leaves),
        )
        self.assertEqual(
            generation.flash_structural_population_budget(parent_count),
            parent_count,
        )
        self.assertEqual(
            generation.flash_structural_population_budget(2 * qualification_count),
            qualification_count,
        )

    def test_flash_initial_prefix_singleton_leaf_takes_one_witness(self):
        # A small grid admits the row_mma family, whose only searchable
        # knobs are its two row knobs. Overriding them leaves the leaf a
        # single reachable normalized config: every other live key is dead
        # for the row programs, so no second distinct witness exists and
        # qualification reserves one row for it, two for every other
        # ordinary leaf.
        with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
            spec = ConfigSpec(
                backend=CuteBackend(),
                target_device_capability=(10, 0),
                num_sm=148,
            )
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        surface = {
            "num_bh": 1,
            "dtype": torch.float16,
            "standard_dense_output": True,
        }
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=4,
            block_size_targets={0: 1, 1: 128, 2: 128},
            **surface,
        )
        fragments = cute_flash.flash_autotune_fragments(64, 4, **surface)
        self.assertIn(
            "row_mma", fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices
        )
        ws_values = cute_flash.flash_effective_config_values(
            cute_flash.resolve_flash_config(
                64, 4, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}, **surface
            )
        )
        live_prefix_keys = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY,
            cute_flash.FLASH_EXP2_PACKET_KEY,
            cute_flash.FLASH_WAIT_HINT_KEY,
            *(
                key
                for key, fragment in fragments.items()
                if ws_values.get(key) != fragment.default()
            ),
        }
        generation = spec.create_config_generation(
            overrides={
                key: fragment.default()
                for key, fragment in fragments.items()
                if key not in live_prefix_keys
            }
        )
        rows = generation.flash_deterministic_population_configs()
        leaves = generation.flash_structural_leaf_catalog()
        singleton_leaves = generation.flash_structural_singleton_leaves()
        self.assertEqual(
            [leaf.pipeline_family for leaf in singleton_leaves], ["row_mma"]
        )
        ordinary_leaves = [
            leaf
            for leaf in leaves
            if leaf.compound_exp2_packet is None and leaf not in singleton_leaves
        ]
        self.assertTrue(ordinary_leaves)
        parent_count = generation.flash_structural_parent_coverage_prefix_count()
        qualification_count = generation.flash_structural_qualification_prefix_count()

        def prefix_leaf_counts(limit: int) -> dict[object, int]:
            counts = dict.fromkeys(leaves, 0)
            for config in rows[:limit]:
                leaf = flash_structural_leaf_from_config(config.config)
                if leaf in counts:
                    counts[leaf] += 1
            return counts

        parent_counts = prefix_leaf_counts(parent_count)
        qualification_counts = prefix_leaf_counts(qualification_count)
        self.assertTrue(all(parent_counts[leaf] == 1 for leaf in leaves))
        self.assertTrue(
            all(qualification_counts[leaf] == 1 for leaf in singleton_leaves)
        )
        self.assertTrue(
            all(qualification_counts[leaf] == 2 for leaf in ordinary_leaves)
        )
        self.assertEqual(qualification_count, parent_count + len(ordinary_leaves))
        self.assertEqual(
            generation.flash_structural_coverage_underqualified_leaves(), []
        )
        # The family value keeps reporting its single witness: the report
        # stays honest about the collapsed surface while the leaf catalog
        # carries the singleton exemption.
        self.assertEqual(
            generation.flash_structural_coverage_underqualified_values(),
            [(cute_flash.FLASH_PIPELINE_FAMILY_KEY, "row_mma", 1)],
        )
        # The row_mma leaf is a singleton only because its knobs are
        # overridden: the live surface gives it two witnesses.
        live = spec.create_config_generation()
        self.assertEqual(live.flash_structural_singleton_leaves(), [])
        self.assertGreaterEqual(
            sum(
                config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "row_mma"
                for config in live.flash_deterministic_population_configs()[
                    : live.flash_structural_qualification_prefix_count()
                ]
            ),
            2,
        )

    def test_lfbo_flash_pipeline_lane_witness_uses_generation_catalog(self):
        stage_key = "cute_flash_kv_stage"
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_exp2_packet="1x1",
            cute_flash_kv_stage=3,
        )
        leaf = flash_structural_leaf_from_config(config.config)
        assert leaf is not None
        flat = [3]
        expected = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=flat,
            config=config,
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_gen = SimpleNamespace(
            flash_pipeline_lane_witnesses=lambda: {(leaf, stage_key, 3): config},
            flatten=Mock(return_value=flat),
            canonicalize_flat=Mock(return_value=(flat, config)),
        )
        search.make_unbenchmarked = Mock(return_value=expected)

        self.assertIs(
            search._flash_pipeline_lane_witness(leaf, (stage_key, 3)),
            expected,
        )
        search.config_gen.flatten.assert_called_once_with(config)
        search.config_gen.canonicalize_flat.assert_called_once_with(flat)
        search.make_unbenchmarked.assert_called_once_with(flat)

    def test_lfbo_flash_starting_points_fall_back_to_finite_rebenchmarks(self):
        def member(wait_hint: int, perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf"), perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="timeout",
            )

        population = [member(0, 1.0), member(1, 2.0)]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 2
        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths, [(population[0], ())])
        self.assertEqual(
            search._autotune_metrics.search_phase_metrics["retained_families"], []
        )

    def test_lfbo_flash_qualification_runs_equal_rounds_per_leaf(self):
        def member(perf: float | None, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(5.0, "fa4", 0)
        first_child = member(None, "fa4", 1)
        winning_child = member(None, "fa4", 2)
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        assert search.flash_structural_search is not None
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.set_generation = Mock()
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)

        def rebenchmark(members, *, desc):
            if winning_child in members and winning_child.perfs:
                winning_child.perfs.append(0.5)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def qualification_path(
            _index,
            current,
            _visited,
            constraints,
            *,
            selected_limit=None,
            neighbor_limit=None,
            required_leaf=None,
            conditional_surface=False,
            disable_early_stopping=False,
        ):
            self.assertEqual(selected_limit, 5)
            self.assertEqual(neighbor_limit, 300)
            self.assertEqual(required_leaf, leaf)
            self.assertTrue(conditional_surface)
            self.assertTrue(disable_early_stopping)
            child = first_child if current is initial else winning_child
            return iter(((current, child),))

        def benchmark(members, *, desc):
            if members == [first_child]:
                self.assertEqual(desc, "Structural qualification 1:")
                first_child.perfs = [4.0]
            else:
                self.assertEqual(members, [winning_child])
                self.assertEqual(desc, "Structural qualification 2:")
                winning_child.perfs = [0.75]

        search._pruned_pattern_search_from = qualification_path
        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )
        self.assertEqual(search.population[0], winning_child)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["phase"], "cute_flash_structural_qualification_v24")
        self.assertEqual(metrics["cute_flash_lane_policy_version"], 16)
        self.assertTrue(metrics["completed"])
        self.assertEqual(metrics["qualification_rounds"], 2)
        self.assertEqual(metrics["qualification_rounds_started"], 2)
        self.assertEqual(metrics["qualification_rounds_completed"], 2)
        self.assertEqual(metrics["retained_family_cap"], 4)
        self.assertEqual(metrics["retained_family_limit"], 1)
        self.assertEqual(metrics["retained_family_slowdown_limit"], 2.0)
        self.assertFalse(metrics["budget_exhausted"])
        self.assertFalse(metrics["family_probe_required"])
        self.assertTrue(metrics["family_probe_complete"])
        self.assertEqual(metrics["family_probe_paths"], [])
        self.assertEqual(metrics["starting_path_limit"], 5)
        self.assertTrue(metrics["unrestricted_path_exhausts_generation_budget"])
        self.assertEqual(metrics["candidate_count"], 2)
        self.assertEqual(len(metrics["leaf_results"][0]["rounds"]), 2)
        self.assertEqual(
            [
                round_metric["neighbor_generation_limit"]
                for round_metric in metrics["leaf_results"][0]["rounds"]
            ],
            [300, 300],
        )
        self.assertEqual(
            [
                round_metric["ordinary_neighbor_generation_limit"]
                for round_metric in metrics["leaf_results"][0]["rounds"]
            ],
            [300, 300],
        )
        self.assertEqual(
            metrics["leaf_results"][0]["retained_config_ids"][0],
            canonical_config_id(winning_child.config),
        )
        qualified = {
            result["config_id"]: result
            for result in metrics["leaf_results"][0]["qualified_results"]
        }
        self.assertEqual(
            qualified[canonical_config_id(winning_child.config)]["attempt_perf"],
            0.75,
        )
        self.assertEqual(
            qualified[canonical_config_id(winning_child.config)]["selection_perf"],
            0.5,
        )
        self.assertEqual(search.set_generation.call_args_list, [call(1), call(2)])
        self.assertEqual(search._fit_surrogate.call_count, 2)

    def test_lfbo_flash_qualification_probes_every_family_before_promotion(self):
        families = ("fa4", "fa4_2cta", "ws_overlap", "fa4_clc", "fa4_tma_4d")

        def member(perf: float | None, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial_perfs = dict(zip(families, (1.0, 1.1, 1.2, 4.0, 5.0), strict=True))
        initial = {
            family: member(initial_perfs[family], family, index)
            for index, family in enumerate(families)
        }
        qualification_children = {
            family: (
                member(None, family, 10 + 2 * index),
                member(None, family, 11 + 2 * index),
            )
            for index, family in enumerate(families)
        }
        constrained_probe_children = {
            family: member(None, family, 30 + index)
            for index, family in enumerate(families)
        }
        global_probe_child = member(None, "fa4_clc", 40)
        leaves = {
            family: flash_structural_leaf_from_config(item.config.config)
            for family, item in initial.items()
        }
        assert all(leaf is not None for leaf in leaves.values())

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 14
        search._flash_promoted_path_limit = 14
        search._flash_family_probe_path_limit = 6
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        assert search.flash_structural_search is not None
        search.population = list(initial.values())
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        typed_leaves = []
        for family in families:
            leaf = leaves[family]
            assert leaf is not None
            typed_leaves.append(leaf)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: typed_leaves,
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(typed_leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.set_generation = Mock()
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def search_path(
            _index,
            current,
            _visited,
            _constraints,
            *,
            selected_limit=None,
            neighbor_limit=None,
            required_leaf=None,
            conditional_surface=False,
            disable_early_stopping=False,
        ):
            self.assertTrue(disable_early_stopping)
            if selected_limit == 5:
                self.assertIsNotNone(required_leaf)
                self.assertTrue(conditional_surface)
                self.assertIsNotNone(neighbor_limit)
                family = required_leaf.pipeline_family
                first, second = qualification_children[family]
                return iter(((current, first), (first, second)))
            self.assertEqual(selected_limit, 20)
            self.assertIsNone(neighbor_limit)
            if required_leaf is None:
                self.assertFalse(conditional_surface)
                return iter(((current, global_probe_child),))
            self.assertTrue(conditional_surface)
            return iter(
                ((current, constrained_probe_children[required_leaf.pipeline_family]),)
            )

        def benchmark(members, *, desc):
            if desc == "Structural family probe 1:":
                for item in members:
                    item.perfs = [
                        0.1
                        if item is global_probe_child
                        else 0.8
                        if item is constrained_probe_children["fa4_tma_4d"]
                        else initial_perfs[
                            item.config.config["cute_flash_pipeline_family"]
                        ]
                    ]
                return
            for item in members:
                family = item.config.config["cute_flash_pipeline_family"]
                assert isinstance(family, str)
                item.perfs = [initial_perfs[family]]

        search._pruned_pattern_search_from = search_path
        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial.values()}
            ),
            3,
        )
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertTrue(metrics["family_probe_required"])
        self.assertTrue(metrics["family_probe_complete"])
        self.assertEqual(metrics["family_probe_path_limit"], 6)
        self.assertEqual(len(metrics["family_probe_paths"]), 6)
        self.assertEqual(metrics["qualification_passes_completed"], 3)

        search._select_starting_paths()
        retained = {
            family["family"]: family
            for family in search._autotune_metrics.search_phase_metrics[
                "retained_families"
            ]
        }
        self.assertTrue(retained["fa4_tma_4d"]["parent_promoted"])
        self.assertFalse(retained["fa4_clc"]["parent_promoted"])
        unrestricted = [
            path
            for family in retained.values()
            for path in family["starting_paths"]
            if path["unrestricted"]
        ]
        self.assertEqual(len(unrestricted), 1)
        self.assertEqual(unrestricted[0]["family"], "fa4_clc")

    def test_lfbo_flash_qualification_skips_family_without_a_measured_member(self):
        """A family whose every member fails has no family probe start; the
        remaining families and the unrestricted path still probe."""
        families = ("fa4", "fa4_2cta", "ws_overlap", "fa4_clc", "fa4_tma_4d")
        failing = "fa4_tma_4d"

        def member(family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        perfs = dict(zip(families, (1.0, 1.1, 1.2, 4.0, 5.0), strict=True))
        initial = [member(family, index) for index, family in enumerate(families)]
        leaves = [flash_structural_leaf_from_config(m.config.config) for m in initial]
        assert all(leaf is not None for leaf in leaves)
        next_hint = iter(range(100, 1000))

        def measure(item: PopulationMember) -> None:
            family = item.config.config["cute_flash_pipeline_family"]
            if family == failing:
                item.status = "error"
                item.perfs = [math.inf]
            else:
                item.perfs = [perfs[family]]

        for item in initial:
            measure(item)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 14
        search._flash_promoted_path_limit = 14
        search._flash_family_probe_path_limit = 6
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = list(initial)
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: leaves,
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.set_generation = Mock()
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def search_path(_index, current, _visited, _constraints, **kwargs):
            leaf = kwargs.get("required_leaf")
            family = (
                current.config.config["cute_flash_pipeline_family"]
                if leaf is None
                else leaf.pipeline_family
            )
            return iter(((current, member(family, next(next_hint))),))

        def benchmark(members, *, desc):
            for item in members:
                measure(item)

        search._pruned_pattern_search_from = search_path
        search.benchmark_population = benchmark

        search._run_flash_structural_qualification({item.config for item in initial})
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertTrue(metrics["family_probe_required"])
        self.assertEqual(metrics["family_probe_path_limit"], 6)
        probed = [path["family"] for path in metrics["family_probe_paths"]]
        self.assertEqual(len(probed), 5)
        self.assertNotIn(failing, probed)

    def test_lfbo_flash_qualification_finishes_round_before_budget_stops_next(self):
        def member(perf: float | None, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(2.0, 0)
        child = member(None, 1)
        declined_child = member(None, 2)
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search._autotune_budget_exceeded_across_ranks = Mock(side_effect=(False, True))
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        generated: list[PopulationMember] = []

        def qualification_path(_index, current, visited, *_args, **_kwargs):
            candidate = child if not generated else declined_child
            generated.append(candidate)
            visited.add(candidate.config)
            return iter(((current, candidate),))

        search._pruned_pattern_search_from = qualification_path

        def benchmark(members, *, desc):
            self.assertEqual(members, [child])
            self.assertEqual(desc, "Structural qualification 1:")
            child.perfs = [1.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 1
        )
        search.rebenchmark_population.assert_called_once()
        rebenchmark_call = search.rebenchmark_population.call_args
        self.assertCountEqual(rebenchmark_call.args[0], [initial, child])
        self.assertEqual(
            rebenchmark_call.kwargs["desc"],
            "Structural qualification 1: verifying",
        )
        search._fit_surrogate.assert_called_once_with()
        self.assertEqual(
            search._autotune_budget_exceeded_across_ranks.call_args_list,
            [call(), call()],
        )
        search.set_generation.assert_called_once_with(1)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["qualification_rounds_started"], 1)
        self.assertEqual(metrics["qualification_rounds_completed"], 1)
        self.assertTrue(metrics["budget_exhausted"])
        self.assertEqual(generated, [child])
        self.assertNotIn(declined_child, search.population)
        self.assertFalse(declined_child.perfs)

    def test_lfbo_flash_qualification_reserves_low_generation_budget(self):
        def member(perf: float | None, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(2.0, 0)
        first_child = member(None, 1)
        declined_child = member(None, 2)
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 2
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        visited = {initial.config}
        generated: list[PopulationMember] = []

        def qualification_path(_index, current, path_visited, *_args, **_kwargs):
            candidate = first_child if not generated else declined_child
            generated.append(candidate)
            path_visited.add(candidate.config)
            return iter(((current, candidate),))

        search._pruned_pattern_search_from = qualification_path

        def benchmark(members, *, desc):
            self.assertEqual(members, [first_child])
            self.assertEqual(desc, "Structural qualification 1:")
            first_child.perfs = [1.0]

        search.benchmark_population = benchmark

        self.assertEqual(search._run_flash_structural_qualification(visited), 1)
        self.assertEqual(generated, [first_child])
        self.assertNotIn(declined_child.config, visited)
        self.assertNotIn(declined_child, search.population)
        self.assertFalse(declined_child.perfs)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_passes_started"], 1)
        self.assertEqual(metrics["qualification_passes_completed"], 1)
        self.assertTrue(metrics["budget_exhausted"])

    def test_lfbo_flash_qualification_synthesizes_missing_pipeline_lane(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float | None, stage: int, wait_hint: int = 0
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[stage],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(2.0, 2)
        synthesized = member(None, 3)
        conditional_children = {stage: member(None, stage, 1) for stage in (2, 3)}
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        witness = Mock(return_value=synthesized)
        search._flash_pipeline_lane_witness = witness
        search._flash_conditional_parent_candidates = lambda members, *_args: (
            members[0],
            [conditional_children[members[0].config.config[stage_key]]],
            {},
        )
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()

        def rebenchmark(_members, *, desc):
            if desc == "Structural qualification 2: verifying":
                initial.perfs.append(2.2)
                synthesized.perfs.append(1.2)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if members == [synthesized]:
                self.assertEqual(desc, "Structural qualification 1:")
                synthesized.perfs = [1.0]
                return
            self.assertCountEqual(members, conditional_children.values())
            self.assertEqual(desc, "Structural qualification 2:")
            for child in members:
                child.perfs = [3.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )

        witness.assert_called_once()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["candidate_count"], 3)
        lanes = metrics["leaf_results"][0]["pipeline_lanes"]
        self.assertTrue(all(lane["complete"] for lane in lanes))
        self.assertEqual(
            lanes[1]["rounds"][0]["candidate_config_ids"],
            [canonical_config_id(synthesized.config)],
        )
        self.assertEqual(
            metrics["leaf_results"][0]["retained_config_ids"],
            [
                canonical_config_id(synthesized.config),
                canonical_config_id(initial.config),
            ],
        )
        conditional_decisions = metrics["leaf_results"][0]["rounds"][1][
            "parent_decisions"
        ]
        decision_by_lane = {
            decision["pipeline_lane"]["value"]: decision
            for decision in conditional_decisions
        }
        self.assertEqual(
            decision_by_lane[2]["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(initial.config),
                    "attempt_perf": 2.0,
                    "selection_perf": 2.0,
                    "status": "ok",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                }
            ],
        )
        self.assertEqual(
            decision_by_lane[3]["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(synthesized.config),
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                }
            ],
        )
        qualified = {
            result["config_id"]: result
            for result in metrics["leaf_results"][0]["qualified_results"]
        }
        self.assertEqual(
            qualified[canonical_config_id(initial.config)]["selection_perf"],
            2.2,
        )
        self.assertEqual(
            qualified[canonical_config_id(synthesized.config)]["selection_perf"],
            1.2,
        )

    def test_lfbo_flash_failed_existing_lane_uses_catalog_witness(self):
        stage_key = "cute_flash_kv_stage"
        failed = PopulationMember(
            fn=lambda: None,
            perfs=[float("inf")],
            flat_values=[2, 0],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=0,
            ),
            status="error",
        )
        catalog_witness = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[2, 1],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=1,
            ),
            status="ok",
        )
        child = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[2, 2],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=2,
            ),
            status="ok",
        )
        leaf = flash_structural_leaf_from_config(failed.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [failed]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_pipeline_lane_witness = Mock(return_value=catalog_witness)

        def qualification_path(_index, current, *_args, **_kwargs):
            self.assertIs(current, catalog_witness)
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        _stub_flash_conditional_proposals(search)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if desc == "Structural qualification 1:":
                self.assertEqual(members, [catalog_witness])
                catalog_witness.perfs = [1.0]
                return
            self.assertEqual(desc, "Structural qualification 2:")
            self.assertEqual(members, [child])
            child.perfs = [2.0]

        search.benchmark_population = benchmark

        self.assertEqual(search._run_flash_structural_qualification({failed.config}), 2)
        search._flash_pipeline_lane_witness.assert_called_once_with(
            leaf, (stage_key, 2)
        )
        lane = search._autotune_metrics.search_phase_metrics["leaf_results"][0][
            "pipeline_lanes"
        ][0]
        self.assertTrue(lane["witness_succeeded"])
        self.assertFalse(lane["terminal_failure_exhausted"])
        self.assertEqual(
            lane["witness_config_id"], canonical_config_id(catalog_witness.config)
        )
        self.assertEqual(
            lane["conditional_candidate_ids"],
            [canonical_config_id(child.config)],
        )
        self.assertEqual(
            lane["successful_conditional_candidate_ids"],
            [canonical_config_id(child.config)],
        )
        rounds = search._autotune_metrics.search_phase_metrics["leaf_results"][0][
            "rounds"
        ]
        witness_decision = rounds[0]["parent_decisions"][0]
        self.assertEqual(witness_decision["selection_kind"], "catalog_witness")
        self.assertEqual(
            witness_decision["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(catalog_witness.config),
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "unknown",
                    "source_hash": None,
                    "measurement_pass_index": None,
                }
            ],
        )
        conditional_decision = rounds[1]["parent_decisions"][0]
        self.assertEqual(
            conditional_decision["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(catalog_witness.config),
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                },
                {
                    "config_id": canonical_config_id(failed.config),
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "error",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                },
            ],
        )
        self.assertEqual(
            conditional_decision["generated_config_ids"],
            [canonical_config_id(child.config)],
        )
        self.assertTrue(lane["complete"])
        self.assertTrue(search._autotune_metrics.search_phase_metrics["completed"])

    def test_lfbo_flash_failed_lane_gets_one_bounded_repair(self):
        stage_key = "cute_flash_kv_stage"

        def member(wait_hint: int, *, failed: bool = False) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf")] if failed else [],
                flat_values=[2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="error" if failed else "ok",
            )

        failed_witness = member(0, failed=True)
        failed_conditional = member(1)
        successful_repair = member(2)
        leaf = flash_structural_leaf_from_config(failed_witness.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [failed_witness]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_pipeline_lane_witness = Mock(return_value=failed_witness)

        generated = iter((failed_conditional, successful_repair))
        search_parents: list[PopulationMember] = []

        def qualification_path(_index, current, *_args, **_kwargs):
            search_parents.append(current)
            child = next(generated)
            return iter(((current, child),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        _stub_flash_conditional_proposals(search)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        benchmark_descs: list[str] = []

        def benchmark(members, *, desc):
            benchmark_descs.append(desc)
            if desc == "Structural qualification 2:":
                self.assertEqual(members, [failed_conditional])
                failed_conditional.perfs = [float("inf")]
                failed_conditional.status = "error"
                return
            self.assertEqual(desc, "Structural qualification failure repairs 1:")
            self.assertEqual(members, [successful_repair])
            successful_repair.perfs = [1.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({failed_witness.config}), 3
        )
        search._flash_pipeline_lane_witness.assert_called_once_with(
            leaf, (stage_key, 2)
        )
        self.assertEqual(search._pruned_pattern_search_from.call_count, 2)
        repair_parent = min(
            (failed_witness, failed_conditional),
            key=search._flash_member_rank_key,
        )
        self.assertEqual(search_parents, [failed_witness, repair_parent])
        self.assertEqual(
            benchmark_descs,
            [
                "Structural qualification 2:",
                "Structural qualification failure repairs 1:",
            ],
        )
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_failure_retries"], 1)
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertFalse(lane["witness_succeeded"])
        self.assertEqual(
            lane["conditional_candidate_ids"],
            [canonical_config_id(failed_conditional.config)],
        )
        self.assertEqual(lane["successful_conditional_candidate_ids"], [])
        self.assertEqual(
            lane["repair_candidate_ids"],
            [canonical_config_id(successful_repair.config)],
        )
        self.assertEqual(
            lane["successful_repair_candidate_ids"],
            [canonical_config_id(successful_repair.config)],
        )
        self.assertEqual(len(lane["repair_parent_decisions"]), 1)
        self.assertEqual(
            lane["repair_parent_decisions"][0]["selected_config_id"],
            canonical_config_id(repair_parent.config),
        )
        self.assertTrue(lane["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_terminal_failures_exhaust_lane_not_leaf(self):
        stage_key = "cute_flash_kv_stage"

        def run(*, leaf_has_success: bool):
            def member(
                stage: int,
                wait_hint: int,
                perf: float | None,
                *,
                failed: bool = False,
            ) -> PopulationMember:
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[] if perf is None else [perf],
                    flat_values=[stage, wait_hint],
                    config=helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4",
                        cute_flash_exp2_packet="1x1",
                        cute_flash_kv_stage=stage,
                        cute_flash_wait_hint=wait_hint,
                    ),
                    status="error" if failed else "ok",
                )

            failed_witness = member(2, 0, float("inf"), failed=True)
            catalog_shadow = member(2, 0, None)
            failed_conditional = member(2, 1, None)
            failed_repair = member(2, 2, None)
            successful_other = member(3, 0, 1.0)
            leaf = flash_structural_leaf_from_config(failed_witness.config.config)
            assert leaf is not None

            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.config_spec = _cute_flash_test_config_spec()
            search.config_gen = SimpleNamespace(
                encode_config=lambda flat: flat,
                flash_structural_leaf_catalog=lambda: [leaf],
                flash_low_confound_schedule_anchor_configs=list,
                flash_clc_lane_catalog=dict,
                flash_exact_effective_search_space_configs=lambda _limit: None,
            )
            search.copies = 5
            search.num_neighbors = 300
            search.num_neighbors_cap = -1
            search.max_generations = 20
            search.flash_structural_search = get_effort_profile(
                "full"
            ).flash_structural_search
            search.population = [failed_witness]
            if leaf_has_success:
                search.population.append(successful_other)
            search.initial_population = len(search.population)
            search.train_x = []
            search.train_y = []
            search._train_members = []
            search.polish_rounds = 0
            search.train_configs = []
            search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
            search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
            search._flash_pipeline_lane_witness = Mock(return_value=catalog_shadow)
            generated = iter((failed_conditional, failed_repair))

            def qualification_path(_index, current, *_args, **_kwargs):
                self.assertIn(current, (failed_witness, failed_conditional))
                child = next(generated)
                return iter(((current, child),))

            search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
            _stub_flash_conditional_proposals(search)
            search._budgeted_range = lambda *args: range(*args)
            search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
            search.set_generation = Mock()
            search.rebenchmark_population = Mock()
            search._fit_surrogate = Mock()

            def benchmark(members, *, desc):
                only = members[0]
                self.assertIn(only, (failed_conditional, failed_repair))
                self.assertIn(
                    desc,
                    (
                        "Structural qualification 2:",
                        "Structural qualification failure repairs 1:",
                    ),
                )
                only.perfs = [float("inf")]
                only.status = "error"

            search.benchmark_population = benchmark

            self.assertEqual(
                search._run_flash_structural_qualification(
                    {member.config for member in search.population}
                ),
                3,
            )
            search._flash_pipeline_lane_witness.assert_called_once_with(
                leaf, (stage_key, 2)
            )
            metrics = search._autotune_metrics.search_phase_metrics
            assert isinstance(metrics, dict)
            lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
            self.assertEqual(
                lane["witness_config_id"], canonical_config_id(failed_witness.config)
            )
            witness_decision = metrics["leaf_results"][0]["rounds"][0][
                "parent_decisions"
            ][0]
            self.assertEqual(witness_decision["selection_kind"], "catalog_witness")
            self.assertEqual(
                witness_decision["candidate_results"][0]["status"], "error"
            )
            self.assertTrue(lane["terminal_failure_exhausted"])
            self.assertTrue(lane["complete"])
            return metrics

        self.assertTrue(run(leaf_has_success=True)["completed"])
        self.assertFalse(run(leaf_has_success=False)["completed"])

    def test_lfbo_flash_later_repair_batch_rechecks_source_repairs(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            stage: int,
            wait_hint: int,
            *,
            failed: bool = False,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf")] if failed else [],
                flat_values=[stage, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="error" if failed else "ok",
            )

        failed_witnesses = {stage: member(stage, 0, failed=True) for stage in (2, 3)}
        failed_conditionals = {stage: member(stage, 1) for stage in (2, 3)}
        successful_repair = member(2, 2)
        source_hashes: dict[int, str] = {}
        shared_source_hash = "a" * 64
        source_hashes[id(failed_conditionals[3].fn)] = shared_source_hash
        source_hashes[id(successful_repair.fn)] = shared_source_hash
        leaf = flash_structural_leaf_from_config(failed_witnesses[2].config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: source_hashes.get(id(fn))
            ),
        )
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            pipeline_candidates_per_leaf_per_round=1,
        )
        search.population = list(failed_witnesses.values())
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )

        def catalog_witness(_leaf, lane):
            stage = lane[1]
            assert isinstance(stage, int)
            return failed_witnesses[stage]

        search._flash_pipeline_lane_witness = Mock(side_effect=catalog_witness)
        generated = {
            (2, 0): failed_conditionals[2],
            (3, 0): failed_conditionals[3],
            (2, 1): successful_repair,
        }
        calls_by_stage = {2: 0, 3: 0}

        def qualification_path(_index, current, *_args, **_kwargs):
            stage = current.config.config[stage_key]
            call_index = calls_by_stage[stage]
            calls_by_stage[stage] += 1
            return iter(((current, generated[(stage, call_index)]),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        _stub_flash_conditional_proposals(search)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        benchmark_descs: list[str] = []

        def benchmark(members, *, desc):
            benchmark_descs.append(desc)
            only = members[0]
            if only in failed_conditionals.values():
                only.perfs = [float("inf")]
                only.status = "error"
                return
            self.assertIs(only, successful_repair)
            successful_repair.perfs = [1.0]
            failed_conditionals[3].perfs = [1.5]
            failed_conditionals[3].status = "deduplicated"

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {member.config for member in failed_witnesses.values()}
            ),
            5,
        )
        self.assertEqual(
            benchmark_descs,
            [
                "Structural qualification 3:",
                "Structural qualification 4:",
                "Structural qualification failure repairs 1:",
            ],
        )
        self.assertEqual(search._pruned_pattern_search_from.call_count, 3)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_passes_planned"], 5)
        self.assertEqual(metrics["qualification_passes_completed"], 5)
        self.assertEqual(len(metrics["measurement_timeline"]), 6)
        lanes = {
            lane["value"]: lane for lane in metrics["leaf_results"][0]["pipeline_lanes"]
        }
        self.assertEqual(len(lanes[2]["repair_parent_decisions"]), 1)
        self.assertEqual(lanes[3]["repair_parent_decisions"], [])
        self.assertEqual(lanes[3]["successful_repair_candidate_ids"], [])
        self.assertIn(
            canonical_config_id(failed_conditionals[3].config),
            lanes[3]["successful_conditional_candidate_ids"],
        )
        final_updates = {
            update["config_id"]: update
            for update in metrics["measurement_timeline"][-1]["updates"]
        }
        self.assertEqual(
            final_updates[canonical_config_id(failed_conditionals[3].config)][
                "source_hash"
            ],
            shared_source_hash,
        )
        self.assertEqual(
            final_updates[canonical_config_id(successful_repair.config)]["source_hash"],
            shared_source_hash,
        )
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_witness_and_conditional_failures_are_repaired(self):
        stage_key = "cute_flash_kv_stage"
        clc_key = "cute_flash_clc_heads_per_batch"

        def member(wait_hint: int, perf: float | None) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[2, 2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        base = member(0, 10.0)
        pipeline_child = member(1, None)
        failed_witness = member(2, None)
        witness_repair = member(3, None)
        failed_conditional = member(4, None)
        conditional_repair = member(5, None)
        leaf = flash_structural_leaf_from_config(base.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (2,),
                    "search_values": (2,),
                    "anchor_values": (2,),
                    "refinement_values": (),
                    "attempted_values": (2,),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            retained_candidates_per_leaf=1,
        )
        search.population = [base]
        search.initial_population = 1
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_pipeline_lane_witness = Mock(
            side_effect=AssertionError("the measured pipeline witness must be reused")
        )
        search._flash_clc_lane_witness = lambda _leaf, _value: failed_witness
        generated = {
            id(base): pipeline_child,
            id(failed_witness): witness_repair,
            id(witness_repair): failed_conditional,
            id(failed_conditional): conditional_repair,
        }
        constraints_by_parent: dict[int, dict[str, object]] = {}

        def qualification_path(_index, current, _visited, constraints, **_kwargs):
            constraints_by_parent[id(current)] = dict(constraints)
            return iter(((current, generated[id(current)]),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        search._flash_conditional_parent_candidates = lambda members, *_args: (
            members[0],
            [generated[id(members[0])]],
            {},
        )
        search._flash_clc_depth_variant = Mock(return_value=conditional_repair)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        benchmark_descs: list[str] = []

        def benchmark(members, *, desc):
            benchmark_descs.append(desc)
            only = members[0]
            if only is failed_witness or only is failed_conditional:
                only.perfs = [float("inf")]
                only.status = "error"
            else:
                only.perfs = [
                    {
                        id(pipeline_child): 1.0,
                        id(witness_repair): 2.0,
                        id(conditional_repair): 0.5,
                    }[id(only)]
                ]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({base.config}),
            7,
        )
        self.assertEqual(
            benchmark_descs,
            [
                "Structural qualification 2:",
                "CLC divisor witnesses 1:",
                "CLC divisor witness failure repairs 1:",
                "CLC divisor conditional children:",
                "CLC divisor conditional failure repairs 1:",
            ],
        )
        for repaired_parent in (failed_witness, failed_conditional):
            self.assertEqual(constraints_by_parent[id(repaired_parent)][clc_key], 2)
            self.assertEqual(
                constraints_by_parent[id(repaired_parent)][
                    "cute_flash_pipeline_family"
                ],
                "fa4_clc",
            )
        metrics = search._autotune_metrics.search_phase_metrics
        clc = metrics["clc_families"][0]
        self.assertEqual(
            clc["witness_repair_candidate_ids"],
            {"2": [canonical_config_id(witness_repair.config)]},
        )
        self.assertEqual(
            clc["conditional_repair_candidate_ids"],
            {"2": [canonical_config_id(conditional_repair.config)]},
        )
        self.assertEqual(
            clc["selected_config_ids"],
            [canonical_config_id(witness_repair.config)],
        )
        self.assertEqual(
            clc["retained_config_ids"],
            [canonical_config_id(conditional_repair.config)],
        )
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_empty_conditional_generation_is_retried(self):
        clc_key = "cute_flash_clc_heads_per_batch"

        def run(*, retry_generates_candidate: bool, repair_witness: bool = False):
            def member(wait_hint: int, perf: float | None) -> PopulationMember:
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[] if perf is None else [perf],
                    flat_values=[2, wait_hint],
                    config=helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4_clc",
                        cute_flash_exp2_packet="1x1",
                        cute_flash_kv_stage=2,
                        cute_flash_clc_heads_per_batch=2,
                        cute_flash_wait_hint=wait_hint,
                    ),
                    status="ok",
                )

            base = member(-1, 1.0) if repair_witness else member(0, 1.0)
            witness = member(0, None) if repair_witness else base
            witness_repair = member(1, None) if repair_witness else witness
            retry = member(2, None)
            leaf = flash_structural_leaf_from_config(witness.config.config)
            assert leaf is not None

            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.config_spec = _cute_flash_test_config_spec()
            search.config_gen = SimpleNamespace(
                encode_config=lambda flat: flat,
                flash_structural_leaf_catalog=lambda: [leaf],
                flash_low_confound_schedule_anchor_configs=list,
                flash_clc_lane_catalog=lambda: {
                    leaf: {
                        "legal_values": (2,),
                        "search_values": (2,),
                        "anchor_values": (2,),
                        "refinement_values": (),
                        "attempted_values": (2,),
                    }
                },
                flash_exact_effective_search_space_configs=lambda _limit: None,
            )
            search.copies = 5
            search.num_neighbors = 300
            search.num_neighbors_cap = -1
            search.max_generations = 20
            policy = get_effort_profile("full").flash_structural_search
            assert policy is not None
            search.flash_structural_search = replace(
                policy,
                qualification_rounds=0,
                retained_candidates_per_leaf=1,
            )
            search.population = [base]
            search.initial_population = 1
            search.train_x = []
            search.train_y = []
            search._train_members = []
            search.polish_rounds = 0
            search.train_configs = []
            search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
            search._flash_pipeline_lanes = lambda _leaf: ()
            search._flash_clc_lane_witness = lambda _leaf, _value: witness
            search._budgeted_range = lambda *args: range(*args)
            search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
            search.set_generation = Mock()
            search.rebenchmark_population = Mock()
            search._fit_surrogate = Mock()
            search._flash_clc_depth_variant = Mock(
                return_value=(retry if retry_generates_candidate else witness_repair)
            )

            conditional_generation_calls = 0

            def qualification_path(_index, current, _visited, constraints, **_kwargs):
                self.assertEqual(dict(constraints)[clc_key], 2)
                if repair_witness and current is witness:
                    return iter(((witness, witness_repair),))
                nonlocal conditional_generation_calls
                conditional_generation_calls += 1
                self.assertIs(current, witness_repair)
                if conditional_generation_calls == 1 or not retry_generates_candidate:
                    return iter(((witness_repair,),))
                return iter(((witness_repair, retry),))

            search._pruned_pattern_search_from = Mock(side_effect=qualification_path)

            def benchmark(members, *, desc):
                only = members[0]
                if only is witness:
                    self.assertTrue(repair_witness)
                    self.assertEqual(desc, "CLC divisor witnesses 1:")
                    witness.perfs = [float("inf")]
                    witness.status = "error"
                elif only is witness_repair:
                    self.assertTrue(repair_witness)
                    self.assertEqual(desc, "CLC divisor witness failure repairs 1:")
                    witness_repair.perfs = [0.75]
                else:
                    self.assertTrue(retry_generates_candidate)
                    self.assertIs(only, retry)
                    self.assertEqual(desc, "CLC divisor conditional failure repairs 1:")
                    retry.perfs = [0.5]

            search.benchmark_population = Mock(side_effect=benchmark)

            self.assertEqual(
                search._run_flash_structural_qualification({base.config}),
                5 if repair_witness else 4,
            )
            self.assertEqual(conditional_generation_calls, 2)
            metrics = search._autotune_metrics.search_phase_metrics
            clc = metrics["clc_families"][0]
            self.assertEqual(clc["conditional_candidate_ids"], {"2": []})
            self.assertEqual(len(clc["conditional_repair_parent_decisions"]), 1)
            decision = clc["conditional_repair_parent_decisions"][0]
            self.assertEqual(
                decision["selected_config_id"],
                canonical_config_id(witness_repair.config),
            )
            self.assertEqual(
                [result["config_id"] for result in decision["candidate_results"]],
                [canonical_config_id(witness_repair.config)],
            )
            return search, metrics, clc, retry, witness_repair

        search, metrics, clc, retry, _witness = run(retry_generates_candidate=True)
        retry_id = canonical_config_id(retry.config)
        self.assertEqual(clc["conditional_repair_candidate_ids"], {"2": [retry_id]})
        self.assertEqual(
            clc["conditional_repair_parent_decisions"][0]["generated_config_ids"],
            [retry_id],
        )
        search.benchmark_population.assert_called_once()
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

        search, metrics, clc, _retry, _witness = run(retry_generates_candidate=False)
        self.assertEqual(clc["conditional_repair_candidate_ids"], {"2": []})
        self.assertEqual(
            clc["conditional_repair_parent_decisions"][0]["generated_config_ids"],
            [],
        )
        search.benchmark_population.assert_not_called()
        self.assertFalse(clc["complete"])
        self.assertFalse(metrics["completed"])

        search, metrics, clc, retry, witness_repair = run(
            retry_generates_candidate=True,
            repair_witness=True,
        )
        self.assertEqual(
            clc["selected_config_ids"],
            [canonical_config_id(witness_repair.config)],
        )
        self.assertEqual(
            clc["conditional_repair_candidate_ids"],
            {"2": [canonical_config_id(retry.config)]},
        )
        self.assertEqual(search.benchmark_population.call_count, 3)
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_repair_batches_compact_after_source_repair(self):
        clc_key = "cute_flash_clc_heads_per_batch"
        values = tuple(range(1, 10))

        def member(value: int, wait_hint: int, *, failed: bool) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf")] if failed else [],
                flat_values=[value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="error" if failed else "ok",
            )

        failed_witnesses = {value: member(value, 0, failed=True) for value in values}
        repairs = {value: member(value, 1, failed=False) for value in values}
        leaf = flash_structural_leaf_from_config(
            failed_witnesses[values[0]].config.config
        )
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": values,
                    "search_values": values,
                    "anchor_values": values,
                    "refinement_values": (),
                    "attempted_values": values,
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: [
                witness.config for witness in failed_witnesses.values()
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            pipeline_candidates_per_leaf_per_round=4,
        )
        search.population = list(failed_witnesses.values())
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ()
        search._flash_clc_lane_witness = lambda _leaf, value: failed_witnesses[value]

        def qualification_path(_index, current, *_args, **_kwargs):
            value = current.config.config[clc_key]
            return iter(((current, repairs[value]),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        search._flash_clc_depth_variant = Mock(
            side_effect=AssertionError("an exhausted CLC leaf needs no combinations")
        )
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        repaired_batches: list[list[int]] = []

        def benchmark(members, *, desc):
            self.assertTrue(desc.startswith("CLC divisor witness failure repairs "))
            repaired_values = [member.config.config[clc_key] for member in members]
            repaired_batches.append(repaired_values)
            for repaired in members:
                value = repaired.config.config[clc_key]
                repaired.perfs = [1.0 + value / 100]
            if len(repaired_batches) == 1:
                failed_witnesses[5].perfs = [1.05]
                failed_witnesses[5].status = "deduplicated"

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {witness.config for witness in failed_witnesses.values()}
            ),
            3,
        )
        self.assertEqual(repaired_batches, [[1, 2, 3, 4], [6, 7, 8, 9]])
        self.assertEqual(search._pruned_pattern_search_from.call_count, 8)
        search._flash_clc_depth_variant.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_passes_planned"], 3)
        self.assertEqual(metrics["qualification_passes_completed"], 3)
        self.assertEqual(len(metrics["measurement_timeline"]), 4)
        clc = metrics["clc_families"][0]
        self.assertEqual(
            [decision["value"] for decision in clc["witness_repair_parent_decisions"]],
            [1, 2, 3, 4, 6, 7, 8, 9],
        )
        self.assertEqual(
            [
                decision["candidate_results"][0]["measurement_pass_index"]
                for decision in clc["witness_repair_parent_decisions"]
            ],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )
        self.assertEqual(
            [
                decision["neighbor_generation_limit"]
                for decision in clc["witness_repair_parent_decisions"]
            ],
            [75] * 8,
        )
        self.assertEqual(clc["witness_repair_candidate_ids"].get("5"), None)
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_axis_coverage_rejects_policy_failures(self):
        allowed = ("ok", "deduplicated", "error", "timeout", "peer_compilation_fail")
        for status in allowed:
            with self.subTest(status=status):
                succeeded = status in {"ok", "deduplicated"}
                self.assertTrue(
                    LFBOPatternSearch._flash_clc_combination_statuses_allowed(
                        [
                            {
                                "config_id": "candidate",
                                "projected_config_id": "candidate",
                                "attempt_perf": 1.0 if succeeded else None,
                                "selection_perf": 1.0 if succeeded else None,
                                "status": status,
                            },
                            {
                                "config_id": None,
                                "projected_config_id": None,
                                "attempt_perf": None,
                                "selection_perf": None,
                                "status": "projection_rejected",
                            },
                        ]
                    )
                )
        for status in ("accuracy_error", "source_rejected", "filtered", "unknown"):
            with self.subTest(status=status):
                self.assertFalse(
                    LFBOPatternSearch._flash_clc_combination_statuses_allowed(
                        [
                            {
                                "config_id": "candidate",
                                "projected_config_id": "candidate",
                                "attempt_perf": 1.0,
                                "selection_perf": 1.0,
                                "status": "ok",
                            },
                            {
                                "config_id": "failed",
                                "projected_config_id": "failed",
                                "attempt_perf": None,
                                "selection_perf": None,
                                "status": status,
                            },
                        ]
                    )
                )
        self.assertFalse(
            LFBOPatternSearch._flash_clc_combination_statuses_allowed(
                [
                    {
                        "config_id": "candidate",
                        "projected_config_id": "candidate",
                        "attempt_perf": 1.0,
                        "selection_perf": None,
                        "status": "ok",
                    }
                ]
            )
        )

    def test_lfbo_flash_terminal_refinement_allows_recorded_policy_failures(self):
        for status in (
            "ok",
            "deduplicated",
            "error",
            "timeout",
            "peer_compilation_fail",
            "accuracy_error",
            "source_rejected",
        ):
            with self.subTest(status=status):
                succeeded = status in {"ok", "deduplicated"}
                record = {
                    "config_id": "candidate",
                    "attempt_perf": 1.0 if succeeded else None,
                    "selection_perf": 1.0 if succeeded else None,
                    "status": status,
                    "source_hash": "a" * 64,
                }
                self.assertTrue(flash_terminal_refinement_result_is_valid(record))
                if status in {"accuracy_error", "source_rejected"}:
                    self.assertFalse(flash_terminal_measurement_is_valid(record))

        for status in ("filtered", "unknown", "projection_rejected"):
            with self.subTest(status=status):
                self.assertFalse(
                    flash_terminal_refinement_result_is_valid(
                        {
                            "config_id": "candidate",
                            "attempt_perf": None,
                            "selection_perf": None,
                            "status": status,
                            "source_hash": None,
                        }
                    )
                )

    def test_lfbo_flash_qualification_reserves_lanes_before_surrogate_pruning(
        self,
    ):
        family_key = "cute_flash_pipeline_family"
        softmax_key = "cute_flash_softmax_disc"
        stage_key = "cute_flash_kv_stage"

        def member(perf: float | None, stage: int, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=["fa4", stage, wait_hint, True],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = [member(1.0, 2, 0), member(10.0, 3, 0)]
        children = {
            (stage, wait_hint): member(None, stage, wait_hint)
            for stage in (2, 3)
            for wait_hint in range(1, 5)
        }
        leaf = flash_structural_leaf_from_config(initial[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            _key_to_flat_indices={
                family_key: ([0], False),
                stage_key: ([1], False),
                softmax_key: ([3], False),
            },
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 200
        search.num_neighbors_cap = 3
        search.max_generations = 20
        search.patience = 0
        search.frac_selected = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.log = Mock()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [*initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        neighbor_limits: list[int] = []
        next_wait_hint = {2: 1, 3: 1}

        def generate(current, _leaf, constraints, neighbor_limit):
            stage = current.config.config[stage_key]
            self.assertEqual(dict(constraints)[stage_key], stage)
            neighbor_limits.append(neighbor_limit)
            start = next_wait_hint[stage]
            next_wait_hint[stage] += neighbor_limit
            return [
                children[(stage, wait_hint)].flat_values
                for wait_hint in range(start, start + neighbor_limit)
            ]

        by_flat = {
            tuple(item.flat_values): item for item in [*initial, *children.values()]
        }
        search.radius = 2
        search.config_gen.flatten = lambda config: next(
            item.flat_values for item in by_flat.values() if item.config == config
        )
        search.config_gen.canonicalize_flat = lambda flat: (
            flat,
            by_flat[tuple(flat)].config,
        )

        def leaf_generation(_leaf, constraints):
            def projections(flat, *, radius, limit, frozen_indices):
                self.assertEqual(frozen_indices, [])
                parent = by_flat[tuple(flat)]
                return [
                    CoordinateNeighborProjection(
                        2,
                        "cute_flash_wait_hint",
                        None,
                        flat[2],
                        child_flat[2],
                        "candidate",
                        child_flat,
                        by_flat[tuple(child_flat)].config,
                    )
                    for child_flat in generate(parent, _leaf, constraints, limit)
                ]

            return SimpleNamespace(
                flatten=search.config_gen.flatten,
                canonicalize_flat=search.config_gen.canonicalize_flat,
                coordinate_neighbor_projections=projections,
                flash_owned_coordinate_indices=lambda config: [],
            )

        search._flash_leaf_config_generation = leaf_generation
        surrogate_stages: list[int] = []

        def hostile_surrogate(candidates, count):
            stages = {item.config.config[stage_key] for item in candidates}
            self.assertEqual(len(stages), 1)
            surrogate_stages.extend(stages)
            return sorted(
                candidates,
                key=lambda item: (
                    item.config.config[stage_key],
                    item.config.config["cute_flash_wait_hint"],
                ),
            )[:count]

        search._surrogate_select = hostile_surrogate

        def benchmark(members, *, desc):
            self.assertLessEqual(len(members), 4)
            for child in members:
                child.perfs = [20.0 + child.config.config["cute_flash_wait_hint"]]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial}
            ),
            2,
        )

        self.assertEqual(surrogate_stages, [2, 3])
        self.assertEqual(neighbor_limits, [1, 2])
        metrics = search._autotune_metrics.search_phase_metrics
        rounds = metrics["leaf_results"][0]["rounds"]
        self.assertEqual([len(item["candidate_config_ids"]) for item in rounds], [0, 2])
        self.assertEqual([item["neighbor_generation_limit"] for item in rounds], [0, 3])
        self.assertEqual(metrics["neighbor_generation_limit_per_leaf_per_round"], 3)
        self.assertTrue(
            all(
                lane["complete"]
                for lane in metrics["leaf_results"][0]["pipeline_lanes"]
            )
        )

    def test_lfbo_flash_orphan_compound_leaf_fails_catalog_completion(self):
        initial = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[True],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_softmax_disc=True,
            ),
            status="ok",
        )
        orphan_config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_exp2_packet="deg2_16x6",
            cute_flash_softmax_disc=False,
        )
        ordinary_leaf = flash_structural_leaf_from_config(initial.config.config)
        orphan_leaf = flash_structural_leaf_from_config(orphan_config.config)
        assert ordinary_leaf is not None and orphan_leaf is not None
        self.assertIsNone(ordinary_leaf.compound_exp2_packet)
        self.assertIsNotNone(orphan_leaf.compound_exp2_packet)
        self.assertNotEqual(ordinary_leaf.softmax_disc, orphan_leaf.softmax_disc)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [ordinary_leaf, orphan_leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [initial.config],
            flash_pipeline_lane_catalog=lambda: {
                ordinary_leaf: (),
                orphan_leaf: (),
            },
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = 1
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock(
            side_effect=AssertionError("orphan compound leaf has no candidates")
        )
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 1
        )

        search.benchmark_population.assert_not_called()
        search.rebenchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["completed"])
        self.assertFalse(metrics["compound_catalog_complete"])
        expected_error = {
            "family": orphan_leaf.pipeline_family,
            "compound_packet": orphan_leaf.compound_exp2_packet,
            "softmax_disc": orphan_leaf.softmax_disc,
            "error": "missing_ordinary_protocol_leaf",
            "required_parent": {
                "family": orphan_leaf.pipeline_family,
                "compound_packet": None,
                "softmax_disc": orphan_leaf.softmax_disc,
            },
        }
        self.assertEqual(metrics["compound_catalog_errors"], [expected_error])
        self.assertEqual(metrics["compound_leaf_count"], 1)
        self.assertEqual(len(metrics["compound_transfers"]), 1)
        transfer = metrics["compound_transfers"][0]
        self.assertEqual(transfer["catalog_error"], expected_error["error"])
        self.assertEqual(transfer["transfer_target_count"], 0)
        self.assertEqual(transfer["transfers"], [])
        self.assertFalse(transfer["failure_statuses_allowed"])
        self.assertFalse(transfer["complete"])
        self.assertEqual(
            search._flash_qualified_compound_config_ids, {orphan_leaf: set()}
        )

    def test_lfbo_flash_compound_failure_backfills_from_next_ordinary_source(
        self,
    ):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float | None,
            *,
            packet: str,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[packet, 2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_2cta",
                    cute_flash_exp2_packet=packet,
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                    cute_flash_e2e_schedule=("16/6" if packet == "1x1" else "8/2"),
                    cute_flash_e2e_offset=12 if packet == "1x1" else 4,
                ),
                status="ok",
            )

        initial = member(1.0, packet="1x1", wait_hint=0)
        conditional = member(None, packet="1x1", wait_hint=1)
        collapsed_source = member(2.5, packet="1x1", wait_hint=2)
        backfill_source = member(3.0, packet="1x1", wait_hint=3)
        final_source = member(4.0, packet="1x1", wait_hint=4)
        preexisting_failure = member(
            float("inf"), packet="deg1_8x2_corr10", wait_hint=5
        )
        preexisting_failure.status = "error"
        transfers = {
            wait_hint: member(None, packet="deg1_8x2_corr10", wait_hint=wait_hint)
            for wait_hint in (0, 1, 2)
        }
        ordinary_leaf = flash_structural_leaf_from_config(initial.config.config)
        compound_leaf = flash_structural_leaf_from_config(transfers[0].config.config)
        assert ordinary_leaf is not None and compound_leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [ordinary_leaf, compound_leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
            flash_pipeline_lane_catalog=lambda: {
                ordinary_leaf: ((stage_key, 2),),
                compound_leaf: (),
            },
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [
            initial,
            collapsed_source,
            backfill_source,
            final_source,
            preexisting_failure,
        ]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = Mock(
            side_effect=lambda leaf: (
                ((stage_key, 2),)
                if leaf == ordinary_leaf
                else self.fail("compound leaf requested pipeline qualification")
            )
        )
        search._pruned_pattern_search_from = lambda _index, current, *_args, **_kwargs: (
            iter(((current, conditional),))
        )
        _stub_flash_conditional_proposals(search)

        def transfer_variant(source, _overrides, *, expected_leaf):
            self.assertEqual(expected_leaf, compound_leaf)
            wait_hint = source.config.config["cute_flash_wait_hint"]
            if wait_hint == 0:
                return preexisting_failure
            if wait_hint in (1, 2):
                return transfers[0]
            if wait_hint == 3:
                return transfers[1]
            return transfers[2]

        search._flash_config_variant = Mock(side_effect=transfer_variant)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()

        def rebenchmark(_members, *, desc):
            if desc == "Compound packet transfers: verifying":
                initial.perfs.append(100.0)
                conditional.perfs.append(200.0)
                collapsed_source.perfs.append(0.1)
                backfill_source.perfs.append(0.2)
                final_source.perfs.append(0.3)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if members == [conditional]:
                self.assertEqual(desc, "Structural qualification 2:")
                conditional.perfs = [2.0]
                return
            if desc == "Compound packet transfers:":
                self.assertCountEqual(members, [transfers[0], transfers[1]])
                transfers[0].perfs = [float("inf")]
                transfers[0].status = "error"
                transfers[1].perfs = [3.0]
                return
            self.assertEqual(desc, "Compound packet failure backfills 1:")
            self.assertEqual(members, [transfers[2]])
            transfers[2].perfs = [3.0]
            # The backfill discovers the failed primary has the same effective
            # source, so source repair promotes it as well as the new candidate.
            transfers[0].perfs = [3.1]
            transfers[0].status = "deduplicated"

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {
                    initial.config,
                    collapsed_source.config,
                    backfill_source.config,
                    final_source.config,
                    preexisting_failure.config,
                }
            ),
            4,
        )
        search._flash_pipeline_lanes.assert_called_once_with(ordinary_leaf)
        self.assertEqual(search._flash_config_variant.call_count, 5)
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertTrue(metrics["completed"])
        self.assertEqual(metrics["ordinary_leaf_count"], 1)
        self.assertEqual(metrics["compound_leaf_count"], 1)
        self.assertTrue(metrics["compound_catalog_complete"])
        self.assertEqual(metrics["compound_catalog_errors"], [])
        self.assertEqual(metrics["leaves_with_candidates"], 2)
        self.assertEqual(len(metrics["leaf_results"]), 1)
        compound = metrics["compound_transfers"][0]
        self.assertEqual(compound["transfer_target_count"], 2)
        self.assertEqual(compound["transfer_count"], 3)
        self.assertEqual(
            compound["primary_transfer_config_ids"],
            [
                canonical_config_id(transfers[0].config),
                canonical_config_id(transfers[1].config),
            ],
        )
        self.assertEqual(
            compound["successful_transfer_config_ids"],
            [
                canonical_config_id(transfers[0].config),
                canonical_config_id(transfers[1].config),
                canonical_config_id(transfers[2].config),
            ],
        )
        self.assertEqual(
            compound["qualified_transfer_config_ids"],
            [
                canonical_config_id(transfers[0].config),
                canonical_config_id(transfers[1].config),
            ],
        )
        self.assertEqual(
            compound["backfill_rounds"],
            [
                {
                    "repair_index": 0,
                    "required_successes": 1,
                    "failed_transfer_config_ids": [
                        canonical_config_id(transfers[0].config)
                    ],
                    "attempted_source_config_ids": [
                        canonical_config_id(final_source.config)
                    ],
                    "generated_config_ids": [canonical_config_id(transfers[2].config)],
                }
            ],
        )
        self.assertTrue(compound["complete"])
        self.assertEqual(
            [
                transfer["source_config_id"]
                for transfer in metrics["compound_transfers"][0]["transfers"]
            ],
            [
                canonical_config_id(conditional.config),
                canonical_config_id(backfill_source.config),
                canonical_config_id(final_source.config),
            ],
        )
        source_selection = metrics["compound_transfers"][0]["source_selection"]
        self.assertEqual(
            [
                (result["config_id"], result["selection_perf"])
                for result in source_selection["candidate_results"]
            ],
            [
                (canonical_config_id(initial.config), 1.0),
                (canonical_config_id(conditional.config), 2.0),
                (canonical_config_id(collapsed_source.config), 2.5),
                (canonical_config_id(backfill_source.config), 3.0),
                (canonical_config_id(final_source.config), 4.0),
            ],
        )
        self.assertEqual(
            source_selection["attempted_config_ids"],
            [
                canonical_config_id(initial.config),
                canonical_config_id(conditional.config),
                canonical_config_id(collapsed_source.config),
                canonical_config_id(backfill_source.config),
                canonical_config_id(final_source.config),
            ],
        )
        self.assertEqual(
            source_selection["selected_config_ids"],
            [
                canonical_config_id(conditional.config),
                canonical_config_id(backfill_source.config),
                canonical_config_id(final_source.config),
            ],
        )
        self.assertNotIn(
            canonical_config_id(preexisting_failure.config),
            {
                transfer["transferred_config_id"]
                for transfer in metrics["compound_transfers"][0]["transfers"]
            },
        )
        source_configs = {
            canonical_config_id(source.config): source.config.config
            for source in (conditional, backfill_source, final_source)
        }
        for transfer in metrics["compound_transfers"][0]["transfers"]:
            is_repaired = transfer["transferred_config_id"] == canonical_config_id(
                transfers[0].config
            )
            self.assertEqual(transfer["attempt_perf"], 3.1 if is_repaired else 3.0)
            self.assertEqual(transfer["selection_perf"], 3.1 if is_repaired else 3.0)
            self.assertEqual(
                transfer["status"], "deduplicated" if is_repaired else "ok"
            )
            self.assertEqual(
                transfer["projected_config_id"],
                transfer["transferred_config_id"],
            )
            self.assertEqual(
                transfer["source_config"],
                source_configs[transfer["source_config_id"]],
            )
            source_snapshot = transfer["source_config"]
            projected_snapshot = transfer["projected_config"]
            self.assertIsInstance(source_snapshot, dict)
            self.assertIsInstance(projected_snapshot, dict)
            assert isinstance(source_snapshot, dict)
            assert isinstance(projected_snapshot, dict)
            self.assertEqual(
                canonical_config_id(helion.Config.from_dict(source_snapshot)),
                transfer["source_config_id"],
            )
            projected_snapshot_id = canonical_config_id(
                helion.Config.from_dict(projected_snapshot)
            )
            self.assertEqual(
                projected_snapshot_id,
                transfer["projected_config_id"],
            )
            self.assertEqual(
                projected_snapshot_id,
                transfer["transferred_config_id"],
            )
            self.assertEqual(
                transfer["projection_overrides"],
                {cute_flash.FLASH_EXP2_PACKET_KEY: "deg1_8x2_corr10"},
            )
            self.assertEqual(source_snapshot[cute_flash.FLASH_E2E_OFFSET_KEY], 12)
            self.assertEqual(projected_snapshot[cute_flash.FLASH_E2E_OFFSET_KEY], 4)
            self.assertEqual(
                transfer["preserved_pipeline_values"],
                {stage_key: 2},
            )

        untransferred = member(0.1, packet="deg1_8x2_corr10", wait_hint=99)
        search.population.append(untransferred)
        search._flash_pipeline_lanes = lambda leaf: (
            search.config_gen.flash_pipeline_lane_catalog()[leaf]
        )
        selected = {member.config for member, _ in search._select_starting_paths()}
        self.assertNotIn(untransferred.config, selected)
        self.assertNotIn(transfers[2].config, selected)
        self.assertLessEqual({transfers[0].config, transfers[1].config}, selected)

    def test_lfbo_flash_exact_clc_space_skips_conditional_and_combination_work(
        self,
    ):
        stage_key = "cute_flash_kv_stage"

        def member(clc_value: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float(clc_value)],
                flat_values=[2, clc_value],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=clc_value,
                ),
                status="ok",
            )

        initial = [member(1), member(2)]
        leaf = flash_structural_leaf_from_config(initial[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (1, 2),
                    "search_values": (1, 2),
                    "anchor_values": (1, 2),
                    "refinement_values": (),
                    "attempted_values": (1, 2),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in initial
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = initial
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_clc_lane_witness = lambda _leaf, value: initial[value - 1]
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted CLC space must not search")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial}
            ),
            3,
        )
        search._pruned_pattern_search_from.assert_not_called()
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertTrue(metrics["exact_space_exhausted"])
        self.assertEqual(metrics["candidate_count"], 0)
        clc = metrics["clc_families"][0]
        self.assertTrue(clc["space_exhausted"])
        self.assertEqual(clc["value_space_exhausted"], {"1": True, "2": True})
        self.assertEqual(clc["selected_values"], [1, 2])
        self.assertEqual(clc["conditional_values"], [])
        self.assertEqual(clc["conditional_candidate_ids"], {})
        self.assertFalse(clc["combination_required"])
        self.assertEqual(clc["combination_candidate_ids"], [])
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_exact_anchors_do_not_suppress_clc_refinement(self):
        stage_key = "cute_flash_kv_stage"
        clc_key = "cute_flash_clc_heads_per_batch"

        def member(
            perf: float | None, *, clc_value: int, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[2, clc_value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=clc_value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        anchors = [
            member(1.0, clc_value=1, wait_hint=0),
            member(2.0, clc_value=2, wait_hint=0),
        ]
        refinement_witnesses = [
            member(None, clc_value=value, wait_hint=0) for value in (4, 8, 16, 32)
        ]
        pipeline_child = member(None, clc_value=1, wait_hint=1)
        refinement_children = {
            value: member(None, clc_value=value, wait_hint=2)
            for value in (4, 8, 16, 32)
        }
        combinations = {
            value: member(None, clc_value=value, wait_hint=3)
            for value in (1, 2, 4, 8, 16, 32)
        }
        leaf = flash_structural_leaf_from_config(anchors[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (1, 2, 4, 8, 16, 32),
                    "search_values": (1, 2, 4, 8, 16, 32),
                    "anchor_values": (1, 2),
                    "refinement_values": (4, 8, 16, 32),
                    "attempted_values": (1, 2, 4, 8, 16, 32),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in anchors
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            retained_candidates_per_leaf=1,
        )
        search.population = anchors
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_clc_lane_witness = lambda _leaf, value: {
            1: anchors[0],
            2: anchors[1],
            **dict(zip((4, 8, 16, 32), refinement_witnesses, strict=True)),
        }[value]

        def qualification_path(_index, current, _visited, constraints, **_kwargs):
            constraint_values = dict(constraints)
            child = (
                refinement_children[constraint_values[clc_key]]
                if clc_key in constraint_values
                else pipeline_child
            )
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        _stub_flash_conditional_proposals(search)
        composition_sources: list[tuple[PopulationMember, int]] = []

        def clc_depth_variant(source, value, **_kwargs):
            composition_sources.append((source, value))
            return combinations[value]

        search._flash_clc_depth_variant = clc_depth_variant
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if desc == "Structural qualification 2:":
                self.assertEqual(members, [pipeline_child])
                pipeline_child.perfs = [0.8]
            elif desc == "CLC divisor witnesses 1:":
                self.assertEqual(members, refinement_witnesses)
                for witness, perf in zip(
                    refinement_witnesses, (0.5, 3.0, 4.0, 5.0), strict=True
                ):
                    witness.perfs = [perf]
            elif desc == "CLC divisor conditional children:":
                self.assertCountEqual(members, refinement_children.values())
                for offset, child in enumerate(refinement_children.values()):
                    child.perfs = [0.4 + offset * 0.01]
            else:
                self.assertEqual(desc, "CLC depth/divisor combinations:")
                self.assertCountEqual(members, combinations.values())
                for combination in members:
                    combination.perfs = [0.3]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in anchors}
            ),
            5,
        )
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["exact_space_exhausted"])
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertFalse(lane["space_exhausted"])
        self.assertTrue(lane["conditional_required"])
        clc = metrics["clc_families"][0]
        self.assertEqual(
            clc["value_space_exhausted"],
            {
                "1": True,
                "2": True,
                "4": False,
                "8": False,
                "16": False,
                "32": False,
            },
        )
        self.assertEqual(clc["attempted_values"], [1, 2, 4, 8, 16, 32])
        self.assertEqual(clc["selected_values"], [4, 1, 2, 8, 16, 32])
        self.assertEqual(clc["conditional_values"], [4, 8, 16, 32])
        self.assertTrue(clc["combination_required"])
        self.assertEqual(
            composition_sources,
            [(refinement_children[4], value) for value in (4, 8, 16, 32, 1, 2)],
        )
        self.assertEqual(
            set(clc["combination_candidate_ids"]),
            {
                canonical_config_id(combination.config)
                for combination in combinations.values()
            },
        )
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_refines_ranks_and_combines_bounded_values(self):
        stage_key = "cute_flash_kv_stage"
        clc_key = "cute_flash_clc_heads_per_batch"

        def member(
            perf: float | None, *, stage: int = 2, clc_value: int, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[stage, clc_value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_clc_heads_per_batch=clc_value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(10.0, clc_value=1, wait_hint=0)
        pipeline_witness = member(None, stage=3, clc_value=1, wait_hint=0)
        pipeline_children = {
            stage: member(None, stage=stage, clc_value=1, wait_hint=1)
            for stage in (2, 3)
        }
        clc_witnesses = {
            value: initial if value == 1 else member(None, clc_value=value, wait_hint=0)
            for value in (1, 8, 2, 4)
        }
        clc_children = {
            value: member(None, clc_value=value, wait_hint=2) for value in (1, 8, 2, 4)
        }
        # The stage-2/value-4 projection is already measured by the CLC
        # conditional pass, matching a normal canonical-combination collision.
        clc_children[4] = member(None, clc_value=4, wait_hint=1)
        combined = {
            (stage, value): member(
                None,
                stage=stage,
                clc_value=value,
                wait_hint=3,
            )
            for stage in (2, 3)
            for value in (1, 8, 2, 4)
        }
        combined[(2, 4)] = clc_children[4]
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (1, 2, 4, 8),
                    "search_values": (1, 2, 4, 8),
                    "anchor_values": (1, 8),
                    "refinement_values": (2, 4),
                    "attempted_values": (1, 8, 2, 4),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        search._flash_pipeline_lane_witness = lambda _leaf, lane: (
            pipeline_witness if lane == (stage_key, 3) else None
        )
        search._flash_clc_lane_witness = lambda _leaf, value: clc_witnesses[value]

        def qualification_path(_index, current, _visited, constraints, **_kwargs):
            constraint_values = dict(constraints)
            if clc_key in constraint_values:
                child = clc_children[constraint_values[clc_key]]
            else:
                child = pipeline_children[current.config.config[stage_key]]
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        _stub_flash_conditional_proposals(search)
        composition_sources: list[tuple[PopulationMember, int]] = []

        def clc_depth_variant(source, value, **_kwargs):
            composition_sources.append((source, value))
            return combined[(source.config.config[stage_key], value)]

        search._flash_clc_depth_variant = clc_depth_variant
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()

        def rebenchmark(_members, *, desc):
            if desc == "CLC divisor conditional children: verifying":
                clc_witnesses[2].perfs.append(50.0)
                clc_witnesses[4].perfs.append(60.0)
            elif desc == "CLC depth/divisor combinations: verifying":
                pipeline_children[2].perfs.append(9.0)
                pipeline_children[3].perfs.append(8.0)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if members == [pipeline_witness]:
                self.assertEqual(desc, "Structural qualification 1:")
                pipeline_witness.perfs = [1.1]
                return
            if desc == "Structural qualification 2:":
                self.assertCountEqual(members, pipeline_children.values())
                self.assertEqual(desc, "Structural qualification 2:")
                pipeline_children[2].perfs = [0.1]
                pipeline_children[3].perfs = [0.2]
                return
            if desc == "CLC divisor witnesses 1:":
                self.assertCountEqual(
                    members,
                    [clc_witnesses[8], clc_witnesses[2], clc_witnesses[4]],
                )
                self.assertEqual(desc, "CLC divisor witnesses 1:")
                for witness in members:
                    witness.perfs = [float(witness.config.config[clc_key])]
                return
            if desc == "CLC depth/divisor combinations:":
                self.assertCountEqual(
                    members,
                    [
                        combination
                        for key, combination in combined.items()
                        if key != (2, 4)
                    ],
                )
                for combination in members:
                    combination.perfs = [3.0]
                return
            self.assertCountEqual(members, clc_children.values())
            self.assertEqual(desc, "CLC divisor conditional children:")
            clc_children[2].perfs = [float("inf")]
            clc_children[2].status = "error"
            clc_children[4].perfs = [0.5]
            clc_children[1].perfs = [1.5]
            clc_children[8].perfs = [7.5]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 6
        )
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["candidate_count"], 17)
        clc = metrics["clc_families"][0]
        self.assertEqual(clc["legal_values"], [1, 2, 4, 8])
        self.assertEqual(clc["anchor_values"], [1, 8])
        self.assertEqual(clc["refinement_values"], [2, 4])
        self.assertEqual(clc["attempted_values"], [1, 8, 2, 4])
        self.assertEqual(
            clc["witness_config_ids"],
            {
                str(value): canonical_config_id(witness.config)
                for value, witness in clc_witnesses.items()
            },
        )
        self.assertEqual(clc["conditional_values"], [2, 4, 8, 1])
        self.assertEqual(
            [
                (result["value"], result["selection_perf"])
                for result in clc["witness_candidate_results"]
            ],
            [(1, 10.0), (8, 8.0), (2, 2.0), (4, 4.0)],
        )
        self.assertEqual(
            [
                (result["value"], result["selection_perf"])
                for result in clc["witness_selection_results"]
            ],
            [(2, 2.0), (4, 4.0), (8, 8.0), (1, 10.0)],
        )
        self.assertEqual(
            clc["selected_config_ids"],
            [
                canonical_config_id(clc_witnesses[2].config),
                canonical_config_id(clc_witnesses[4].config),
                canonical_config_id(clc_witnesses[8].config),
                canonical_config_id(clc_witnesses[1].config),
            ],
        )
        self.assertEqual(
            [
                (
                    decision["value"],
                    decision["candidate_results"][0]["selection_perf"],
                )
                for decision in clc["conditional_parent_decisions"]
            ],
            [(2, 2.0), (4, 4.0), (8, 8.0), (1, 10.0)],
        )
        self.assertEqual(
            clc["conditional_candidate_ids"],
            {
                str(value): [canonical_config_id(child.config)]
                for value, child in clc_children.items()
            },
        )
        self.assertEqual(clc["retained_values"], [4, 1, 8, 2])
        self.assertEqual(
            [
                (result["value"], result["selection_perf"])
                for result in clc["retained_ranking_results"]
            ],
            [(4, 0.5), (1, 1.5), (8, 7.5), (2, 50.0)],
        )
        self.assertEqual(
            clc["retained_config_ids"],
            [
                canonical_config_id(clc_children[4].config),
                canonical_config_id(clc_children[1].config),
                canonical_config_id(clc_children[8].config),
                canonical_config_id(clc_witnesses[2].config),
            ],
        )
        retained_value_decisions = {
            decision["value"]: decision for decision in clc["retained_value_decisions"]
        }
        self.assertEqual(
            retained_value_decisions[2]["candidate_results"][-1]["status"],
            "error",
        )
        self.assertIsNone(
            retained_value_decisions[2]["candidate_results"][-1]["selection_perf"]
        )
        self.assertEqual(
            [
                (
                    representative["config_id"],
                    (
                        None
                        if representative["assigned_pipeline_lane"] is None
                        else representative["assigned_pipeline_lane"]["value"]
                    ),
                )
                for representative in clc["depth_selection"]["selected_representatives"]
            ],
            [
                (canonical_config_id(pipeline_children[2].config), None),
                (canonical_config_id(pipeline_children[3].config), 3),
            ],
        )
        depth_results = {
            result["config_id"]: result
            for result in clc["depth_selection"]["candidate_results"]
        }
        reused_combination_id = canonical_config_id(clc_children[4].config)
        self.assertIn(reused_combination_id, depth_results)
        self.assertIn(reused_combination_id, clc["combination_candidate_ids"])
        self.assertEqual(
            depth_results[canonical_config_id(pipeline_children[2].config)][
                "selection_perf"
            ],
            0.1,
        )
        self.assertEqual(
            depth_results[canonical_config_id(pipeline_children[3].config)][
                "selection_perf"
            ],
            0.2,
        )
        self.assertEqual(
            set(clc["combination_candidate_ids"]),
            {
                canonical_config_id(combination.config)
                for combination in combined.values()
            },
        )
        self.assertEqual(len(clc["combination_candidate_ids"]), 8)
        self.assertCountEqual(
            composition_sources,
            [
                (pipeline_children[stage], value)
                for stage in (2, 3)
                for value in (4, 1, 8, 2)
            ],
        )
        self.assertFalse(clc["complete"])
        self.assertEqual(
            clc["conditional_repair_candidate_ids"],
            {"2": []},
        )
        self.assertEqual(
            clc["conditional_repair_parent_decisions"][0]["selected_config_id"],
            canonical_config_id(clc_children[2].config),
        )

    def test_lfbo_flash_clc_combination_failure_uses_axis_coverage(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            stage: int,
            clc_value: int,
            wait_hint: int,
            perf: float | None,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[stage, clc_value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_clc_heads_per_batch=clc_value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        depth_2 = member(2, 2, 0, 1.0)
        depth_3 = member(3, 4, 0, 2.0)
        failed_cell = member(2, 4, 1, None)
        successful_cell = member(3, 2, 1, None)
        combined = {
            (2, 2): depth_2,
            (2, 4): failed_cell,
            (3, 2): successful_cell,
            (3, 4): depth_3,
        }
        unmeasured = member(2, 8, 0, None).config
        leaf = flash_structural_leaf_from_config(depth_2.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (2, 4),
                    "search_values": (2, 4),
                    "anchor_values": (2, 4),
                    "refinement_values": (),
                    "attempted_values": (2, 4),
                }
            },
            # Value-specific spaces are measured, avoiding unrelated CLC
            # conditional searches. The extra config keeps the full leaf
            # non-exhaustive so the depth/divisor projection is still required.
            flash_exact_effective_search_space_configs=lambda _limit: [
                depth_2.config,
                depth_3.config,
                unmeasured,
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            qualification_rounds=1,
            conditional_candidates_per_pipeline_lane=0,
        )
        search.population = [depth_2, depth_3]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        search._flash_pipeline_lane_witness = Mock(
            side_effect=AssertionError("measured lane members must be reused")
        )
        search._flash_clc_lane_witness = lambda _leaf, value: {
            2: depth_2,
            4: depth_3,
        }[value]
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("no conditional search is required")
        )
        search._flash_clc_depth_variant = lambda source, value, **_kwargs: combined[
            (source.config.config[stage_key], value)
        ]
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            self.assertEqual(desc, "CLC depth/divisor combinations:")
            self.assertCountEqual(members, [failed_cell, successful_cell])
            failed_cell.perfs = [float("inf")]
            failed_cell.status = "error"
            successful_cell.perfs = [0.5]

        search.benchmark_population = Mock(side_effect=benchmark)

        self.assertEqual(
            search._run_flash_structural_qualification(
                {depth_2.config, depth_3.config}
            ),
            3,
        )
        search.benchmark_population.assert_called_once()
        search._flash_pipeline_lane_witness.assert_not_called()
        search._pruned_pattern_search_from.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        clc = metrics["clc_families"][0]
        self.assertEqual(len(clc["combination_cells"]), 4)
        self.assertCountEqual(
            clc["combination_depth_config_ids"],
            [
                canonical_config_id(depth_2.config),
                canonical_config_id(depth_3.config),
            ],
        )
        self.assertEqual(clc["combination_divisor_values"], [2, 4])
        self.assertCountEqual(
            clc["successful_combination_depth_config_ids"],
            [
                canonical_config_id(depth_2.config),
                canonical_config_id(depth_3.config),
            ],
        )
        self.assertEqual(clc["successful_combination_divisor_values"], [2, 4])
        self.assertTrue(clc["combination_row_coverage_complete"])
        self.assertTrue(clc["combination_column_coverage_complete"])
        cell_by_axis = {
            (cell["depth_config_id"], cell["divisor_value"]): cell
            for cell in clc["combination_cells"]
        }
        failed_result = cell_by_axis[(canonical_config_id(depth_2.config), 4)]
        self.assertEqual(
            failed_result["projected_config_id"],
            canonical_config_id(failed_cell.config),
        )
        self.assertEqual(
            failed_result["config_id"], canonical_config_id(failed_cell.config)
        )
        self.assertIsNone(failed_result["attempt_perf"])
        self.assertIsNone(failed_result["selection_perf"])
        self.assertEqual(failed_result["status"], "error")
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_only_full_flash_unrestricted_path_exhausts_budget(self):
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search

        self.assertTrue(search._path_exhausts_generation_budget(()))
        self.assertFalse(
            search._path_exhausts_generation_budget(
                (("cute_flash_pipeline_family", "fa4"),)
            )
        )

        search.flash_structural_search = get_effort_profile(
            "quick"
        ).flash_structural_search
        self.assertFalse(search._path_exhausts_generation_budget(()))
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        self.assertFalse(search._path_exhausts_generation_budget(()))

    def test_lfbo_pipeline_qualification_is_flash_policy_gated(self):
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.max_generations = 20
        search.config_gen = Mock()

        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        self.assertEqual(search._run_flash_structural_qualification(set()), 0)

        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = None
        self.assertEqual(search._run_flash_structural_qualification(set()), 0)
        self.assertFalse(search.config_gen.flash_structural_leaf_catalog.called)

    def test_lfbo_non_flash_does_not_require_json_serializable_configs(self):
        config = helion.Config(
            block_sizes=[1],
            custom_non_json_value=Path("custom-value"),
        )
        member = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.best_available_pad_random = True
        search.initial_population = 1
        search.log = Mock()
        search.copies = 1
        search.max_generations = 0
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [[1]]
        search.make_unbenchmarked = lambda _flat: member
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name=None),
        )
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda _flat: [0.0])
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = None
        search.train_source_hashes = None
        search._fit_surrogate = Mock()
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        search._cute_flash_lane_policy_enabled = False
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._finalize = lambda: config

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertIs(search._autotune(), config)

    def test_lfbo_flash_qualification_preserves_initial_generation_order(self):
        def member(wait_hint: int, perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        generated = [member(0, 2.0), member(1, 1.0)]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.best_available_pad_random = True
        search.initial_population = 1
        search.log = Mock()
        search.copies = 1
        search.max_generations = 0
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [
            item.flat_values for item in generated
        ]
        search.make_unbenchmarked = lambda flat: generated[flat[0]]
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda flat: flat)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search.train_source_hashes = []
        search._fit_surrogate = Mock()
        search.config_spec = _cute_flash_test_config_spec()
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)

        def qualify(_visited, *, initial_population):
            self.assertEqual(initial_population, generated)
            self.assertEqual(search.population, list(reversed(generated)))
            return 0

        search._run_flash_structural_qualification = Mock(side_effect=qualify)
        search._select_starting_paths = Mock(return_value=[(generated[1], ())])
        search._finalize = lambda: generated[1].config

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertIs(search._autotune(), generated[1].config)

        search._run_flash_structural_qualification.assert_called_once()

    def test_lfbo_retained_flash_paths_use_conditional_surfaces(self):
        def member(family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        unrestricted = member("fa4", 0)
        constrained = member("fa4_2cta", 1)
        constrained_leaf = flash_structural_leaf_from_config(constrained.config.config)
        assert constrained_leaf is not None
        constraints = LFBOPatternSearch._flash_leaf_constraints(constrained_leaf)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.best_available_pad_random = True
        search.initial_population = 1
        search.log = Mock()
        search.copies = 2
        search.max_generations = 0
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [unrestricted.flat_values]
        search.make_unbenchmarked = lambda _flat: unrestricted
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name=None),
        )
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda flat: flat)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._fit_surrogate = Mock()
        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._run_flash_structural_qualification = Mock(return_value=0)
        search._select_starting_paths = Mock(
            return_value=[(unrestricted, ()), (constrained, constraints)]
        )
        search._pruned_pattern_search_from = Mock(return_value=iter(()))
        search._finalize = lambda: unrestricted.config

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertEqual(search._autotune(), unrestricted.config)

        calls = search._pruned_pattern_search_from.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0].kwargs["required_leaf"])
        self.assertFalse(calls[0].kwargs["conditional_surface"])
        self.assertEqual(calls[1].kwargs["required_leaf"], constrained_leaf)
        self.assertTrue(calls[1].kwargs["conditional_surface"])

    def test_lfbo_flash_wall_budget_returns_best_before_required_probe(self):
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_exp2_packet="1x1",
        )
        member = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.best_available_pad_random = True
        search.initial_population = 1
        search.log = Mock()
        search.copies = 1
        search.max_generations = 20
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [[1]]
        search.make_unbenchmarked = lambda _flat: member
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name=None),
        )
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda flat: flat)
        search.train_x = []
        search.train_y = []
        search._train_members = []
        search.polish_rounds = 0
        search.train_configs = []
        search._fit_surrogate = Mock()
        search.config_spec = _cute_flash_test_config_spec()
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)

        def incomplete_qualification(_visited, *, initial_population):
            self.assertEqual(initial_population, [member])
            search._autotune_metrics.search_phase_metrics = {
                "family_probe_required": True,
                "family_probe_complete": False,
            }
            return 1

        search._run_flash_structural_qualification = incomplete_qualification
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=True)
        search._select_starting_paths = Mock()
        search._finalize = Mock(return_value=config)

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertEqual(search._autotune(), config)

        search._select_starting_paths.assert_not_called()
        search._finalize.assert_called_once_with()

    def test_cute_flash_best_available_partial_and_zero_population(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}

        def canonicalize_flat(flat):
            value = flat[0]
            if value < 0:
                raise exc.InvalidConfig("invalid")
            canonical_value = value % len(configs)
            return [canonical_value], configs[canonical_value]

        config_gen = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            flash_deterministic_population_configs=lambda: [configs[1], configs[2]],
            flash_exact_effective_search_space_configs=lambda _limit: None,
            flash_structural_parent_coverage_prefix_count=lambda: 2,
            flash_structural_population_budget=lambda _target: 2,
            flash_structural_qualification_prefix_count=lambda: 2,
            flatten=lambda config: [config.config["block_sizes"][0] - 1],
            canonicalize_flat=canonicalize_flat,
            random_flat=Mock(side_effect=([-1], [4])),
        )
        search = PatternSearch.__new__(PatternSearch)
        search.config_spec = config_gen.config_spec
        search.config_gen = config_gen
        search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        search.best_available_pad_random = True
        search.log = Mock()
        search._pinned_finalist_configs = set()
        search._generate_best_available_population_flat = lambda: [[5], [0], [3]]

        search.initial_population = 1
        partial = search._generate_initial_population_flat()
        self.assertEqual(partial, [[1]])
        self.assertEqual(len(partial), len({tuple(flat) for flat in partial}))

        search.initial_population = 5
        padded = search._generate_initial_population_flat()
        self.assertEqual(padded, [[1], [2], [0], [3], [4]])
        self.assertEqual(config_gen.random_flat.call_count, 2)

        config_gen.flash_structural_parent_coverage_prefix_count = lambda: 0
        search.initial_population = 0
        zero = search._generate_initial_population_flat()
        self.assertEqual(zero, [[0], [3]])

    def test_cute_flash_quick_population_merges_before_limiting(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}

        def canonicalize_flat(flat):
            value = flat[0]
            return [value], configs[value]

        config_gen = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            flash_deterministic_population_configs=lambda: [configs[1], configs[2]],
            flash_exact_effective_search_space_configs=lambda _limit: None,
            flash_structural_population_budget=lambda _target: 2,
            flash_structural_qualification_prefix_count=lambda: 2,
            flatten=lambda config: [config.config["block_sizes"][0] - 1],
            canonicalize_flat=canonicalize_flat,
            random_flat=Mock(side_effect=AssertionError("quick search must not pad")),
        )
        search = PatternSearch.__new__(PatternSearch)
        search.config_gen = config_gen
        search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        search.best_available_pad_random = False
        search.initial_population = 3
        # Config 1 is both a structural row and a pinned compiler seed. Config 3
        # is another pinned seed; configs 0 and 4 model optional cache rows.
        search._pinned_finalist_configs = {configs[1], configs[3]}
        search._generate_best_available_population_flat = lambda: [
            [1],
            [3],
            [0],
            [4],
        ]

        population = search._generate_initial_population_flat()

        self.assertEqual(population, [[1], [2], [3]])
        self.assertEqual(len(population), len({tuple(flat) for flat in population}))
        config_gen.random_flat.assert_not_called()

    def test_cute_flash_best_available_matches_random_exact_space(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}
        exact_configs = [configs[0], configs[1], configs[2]]

        def unflatten(flat):
            return configs[flat[0]]

        exact_space = Mock(return_value=exact_configs)
        config_gen = ConfigGeneration.__new__(ConfigGeneration)
        config_gen.config_spec = _cute_flash_test_config_spec()
        config_gen._config_value_priors = {}
        config_gen.default_flat = lambda: [0]
        config_gen.validate_flash_structural_coverage = Mock()
        config_gen._flash_deterministic_coverage_flats = lambda: [[0]]
        config_gen.flash_deterministic_population_configs = lambda: [configs[0]]
        config_gen.flash_structural_qualification_prefix_count = lambda: 1
        config_gen.flash_structural_population_budget = lambda _target: 1
        config_gen.flash_exact_effective_search_space_configs = exact_space
        config_gen.unflatten = unflatten
        config_gen.flatten = lambda config: [config.config["block_sizes"][0] - 1]
        config_gen.canonicalize_flat = lambda flat: (
            config_gen.flatten(unflatten(flat)),
            unflatten(flat),
        )
        config_gen.user_seed_flat_config_pairs = lambda _configs, _log=None: []
        config_gen.seed_flat_config_pairs = lambda _log=None: []
        config_gen.random_flat = Mock(
            side_effect=AssertionError("an enumerated space must not draw randomly")
        )

        random_search = PatternSearch.__new__(PatternSearch)
        random_search.config_gen = config_gen
        random_search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_RANDOM
        )
        random_search.initial_population = 5
        random_search._autotune_seed_configs = list
        random_search.log = Mock()
        # FROM_RANDOM now pins seed/default configs into final verification.
        random_search._pinned_finalist_configs = set()
        random_population = random_search._generate_initial_population_flat()
        random_unique = [
            list(flat)
            for flat in dict.fromkeys(tuple(flat) for flat in random_population)
        ]

        best_search = PatternSearch.__new__(PatternSearch)
        best_search.config_gen = config_gen
        best_search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        best_search.best_available_pad_random = True
        best_search.initial_population = 5
        best_search.log = Mock()
        best_search._pinned_finalist_configs = {configs[0]}
        # Config 2 aliases an exact row; config 3 models a legal cache row that
        # is outside the active search values used by the exact enumeration.
        best_search._generate_best_available_population_flat = lambda: [
            [0],
            [2],
            [3],
        ]

        best_population = best_search._generate_initial_population_flat()

        self.assertEqual(random_population, [[0], [1], [2], [0], [1]])
        self.assertEqual(random_unique, [[0], [1], [2]])
        self.assertEqual(best_population, [[0], [1], [2], [3]])
        self.assertEqual(best_population[: len(random_unique)], random_unique)
        self.assertLessEqual(len(best_population), best_search.initial_population)
        self.assertEqual(len(best_population), len({tuple(x) for x in best_population}))
        self.assertEqual(exact_space.call_args_list, [call(5), call(5)])
        config_gen.random_flat.assert_not_called()

    def test_cute_flash_best_available_exact_space_is_mandatory(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}

        def canonicalize_flat(flat):
            value = flat[0] % len(configs)
            return [value], configs[value]

        config_gen = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            flash_deterministic_population_configs=lambda: [configs[1]],
            flash_exact_effective_search_space_configs=Mock(
                return_value=[configs[1], configs[2]]
            ),
            flash_structural_population_budget=lambda _target: 1,
            flash_structural_qualification_prefix_count=lambda: 1,
            flatten=lambda config: [config.config["block_sizes"][0] - 1],
            canonicalize_flat=canonicalize_flat,
            random_flat=Mock(side_effect=AssertionError("population is complete")),
        )
        search = PatternSearch.__new__(PatternSearch)
        search.config_gen = config_gen
        search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        search.best_available_pad_random = False
        search.initial_population = 2
        search.log = Mock()
        search._pinned_finalist_configs = {configs[3]}
        # Configs 1 and 2 are optional cache aliases of required rows. Config 4
        # must not displace the exact space after structural and pinned rows.
        search._generate_best_available_population_flat = lambda: [
            [3],
            [1],
            [2],
            [4],
        ]

        population = search._generate_initial_population_flat()

        self.assertEqual(population, [[1], [3], [2]])
        self.assertEqual(len(population), len({tuple(flat) for flat in population}))
        config_gen.flash_exact_effective_search_space_configs.assert_called_once_with(2)
        config_gen.random_flat.assert_not_called()

    def test_cute_flash_population_bounds_compiler_seeds(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(6)}
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation._config_value_priors = {}
        generation.default_flat = lambda: [0]
        generation.validate_flash_structural_coverage = Mock()
        generation._flash_deterministic_coverage_flats = lambda: [[0], [1], [2]]
        generation.flash_structural_qualification_prefix_count = lambda: 2
        generation.flash_structural_population_budget = lambda _target: 3
        generation.flash_exact_effective_search_space_configs = lambda _limit: None
        generation.unflatten = lambda flat: configs[flat[0]]
        generation.flatten = lambda config: [config.config["block_sizes"][0] - 1]
        generation.user_seed_flat_config_pairs = lambda _configs, _log: [
            ([3], configs[3]),
            ([1], configs[1]),
        ]
        generation.seed_flat_config_pairs = lambda _log: [([4], configs[4])]
        generation.biased_random_flat = Mock(
            side_effect=AssertionError("no priors are active")
        )
        generation.random_flat = Mock(return_value=[5])

        population = generation.random_population_flat(
            6, user_seed_configs=[configs[3], configs[1]]
        )
        normalized = [generation.unflatten(flat) for flat in population]
        self.assertEqual(population, [[0], [1], [3], [4], [2], [5]])
        self.assertEqual(len(normalized), len(set(normalized)))
        self.assertEqual(len(normalized), 6)
        self.assertLessEqual({configs[0], configs[1], configs[2]}, set(normalized))
        self.assertLessEqual({configs[3], configs[4]}, set(normalized))
        generation.validate_flash_structural_coverage.assert_called_once_with()

        # A target smaller than the mandatory set may grow for explicit user
        # seeds, but compiler hints stay within the requested population size.
        generation.random_flat.reset_mock()
        small = generation.random_population_flat(
            2, user_seed_configs=[configs[3], configs[1]]
        )
        self.assertEqual(small, [[0], [1], [3]])
        generation.random_flat.assert_not_called()
        self.assertEqual(generation.validate_flash_structural_coverage.call_count, 2)

    def test_cute_flash_exact_space_survives_oversized_mandatory_population(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(4)}
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation._config_value_priors = {}
        generation.default_flat = lambda: [0]
        generation.validate_flash_structural_coverage = Mock()
        generation._flash_deterministic_coverage_flats = lambda: [[0]]
        generation.flash_structural_qualification_prefix_count = lambda: 1
        generation.flash_structural_population_budget = lambda _target: 1
        generation.flash_exact_effective_search_space_configs = lambda _limit: [
            configs[0],
            configs[1],
        ]
        generation.unflatten = lambda flat: configs[flat[0]]
        generation.flatten = lambda config: [config.config["block_sizes"][0] - 1]
        generation.user_seed_flat_config_pairs = lambda _configs, _log: [
            ([3], configs[3])
        ]
        generation.seed_flat_config_pairs = lambda _log: []

        population = generation.random_population_flat(
            2, user_seed_configs=[configs[3]]
        )

        self.assertEqual(population, [[0], [3], [1]])
        self.assertEqual(
            {generation.unflatten(flat) for flat in population},
            {configs[0], configs[1], configs[3]},
        )

    def test_incomplete_cute_flash_structural_design_is_rejected(self):
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation._override_values = {}
        generation._flash_deterministic_coverage_flats = Mock(return_value=[])
        generation._flash_coverage_uncovered_cache = [("family", "missing")]
        generation._flash_coverage_underqualified_cache = [("packet", "thin", 1)]
        generation._flash_coverage_uncovered_interactions_cache = [
            (("family", "packet"), ("missing", "thin"))
        ]

        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design.*uncovered values.*"
            "uncovered interactions",
        ):
            generation.flash_deterministic_population_configs()

        # An unrelated override must not hide a broken flash covering design.
        generation._override_values = {"num_warps": 4}
        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design",
        ):
            generation.flash_deterministic_population_configs()
        generation._override_values = {"cute_flash_wait_hint": 0}
        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design",
        ):
            generation.flash_deterministic_population_configs()
        generation._override_values = {"cute_flash_exp2_packet": "1x1"}
        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design",
        ):
            generation.flash_deterministic_population_configs()

        # A family may legitimately normalize to one effective config. Keep the
        # two-witness shortfall as telemetry for the strict harness, not a fatal
        # invariant for ordinary tuning.
        generation._flash_coverage_uncovered_cache = []
        generation._flash_coverage_uncovered_interactions_cache = []
        self.assertEqual(generation.flash_deterministic_population_configs(), [])

        # Explicit overrides intentionally narrow the advertised search surface;
        # its unpruned diagnostics must not reject that constrained search.
        generation._override_values = {"cute_flash_pipeline_family": "fa4"}
        generation._flash_coverage_uncovered_cache = [("family", "missing")]
        self.assertEqual(generation.flash_deterministic_population_configs(), [])

    def test_lfbo_flash_path_constraint_filters_before_visited(self):
        packet_key = "cute_flash_exp2_packet"

        def member(flat: int, packet: str) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float(flat)],
                flat_values=[flat, packet],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=flat,
                ),
                status="ok",
            )

        current = member(1, "deg2_16x6")
        rejected = member(2, "1x1")
        retained = member(3, "deg2_16x6")
        unselected = member(4, "deg2_16x6")
        by_flat = {2: rejected, 3: retained, 4: unselected}
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.max_generations = 1
        search.patience = 1
        search.frac_selected = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.log = lambda *_args, **_kwargs: None
        search.config_gen = SimpleNamespace(
            _key_to_flat_indices={packet_key: ([1], False)}
        )
        search._generate_neighbors = lambda *_args, **_kwargs: [
            rejected.flat_values,
            retained.flat_values,
            unselected.flat_values,
        ]
        search.make_unbenchmarked = lambda flat: by_flat[flat[0]]
        search._surrogate_select = lambda candidates, _count: candidates[:2]

        visited: set[helion.Config] = set()
        path = search._pruned_pattern_search_from(
            0,
            current,
            visited,
            ((packet_key, "deg2_16x6"),),
        )

        self.assertEqual(next(path), [current, retained])
        self.assertNotIn(rejected.config, visited)
        self.assertIn(retained.config, visited)
        self.assertNotIn(unselected.config, visited)

        selected_counts = []
        search.frac_selected = 0.1
        search._surrogate_select = lambda candidates, count: (
            selected_counts.append(count) or candidates[:count]
        )
        small_path = search._pruned_pattern_search_from(
            0,
            current,
            set(),
            ((packet_key, "deg2_16x6"),),
        )
        self.assertEqual(next(small_path), [current, retained])
        self.assertEqual(selected_counts, [2])

        search._surrogate_select = lambda candidates, count: list(
            reversed(candidates[-count:])
        )
        reordered_visited: set[helion.Config] = set()
        reordered_path = search._pruned_pattern_search_from(
            0,
            current,
            reordered_visited,
            ((packet_key, "deg2_16x6"),),
            selected_limit=2,
        )
        self.assertEqual(next(reordered_path), [current, unselected])
        self.assertIn(current.config, reordered_visited)
        self.assertIn(unselected.config, reordered_visited)
        self.assertNotIn(retained.config, reordered_visited)

        # Unconstrained/non-flash paths also record only selected candidates
        # as visited, so proposals the surrogate discards can be re-proposed
        # once it has learned more.
        search.frac_selected = 0.5
        search._surrogate_select = lambda candidates, count: candidates[:count]
        search._generate_neighbors = lambda _base: [
            rejected.flat_values,
            retained.flat_values,
            unselected.flat_values,
        ]
        unconstrained_visited: set[helion.Config] = set()
        unconstrained = search._pruned_pattern_search_from(
            0, current, unconstrained_visited
        )
        self.assertEqual(next(unconstrained), [current, rejected])
        self.assertEqual(
            unconstrained_visited,
            {current.config, rejected.config},
        )

        # CuTe's unrestricted continuation runs after constrained paths in each
        # generation. Only selected candidates become visited so an unmeasured
        # neighbor remains available to constrained paths and later generations.
        search.max_generations = 2
        search.frac_selected = 0.67
        search._surrogate_select = lambda candidates, count: candidates[:count]
        selected_only_visited: set[helion.Config] = set()
        selected_only = search._pruned_pattern_search_from(
            0,
            current,
            selected_only_visited,
            disable_early_stopping=True,
        )
        self.assertEqual(next(selected_only), [current, rejected])
        self.assertEqual(selected_only_visited, {current.config, rejected.config})
        self.assertEqual(next(selected_only), [current, retained])
        self.assertEqual(
            selected_only_visited,
            {current.config, rejected.config, retained.config},
        )
        self.assertNotIn(unselected.config, selected_only_visited)

    def test_lfbo_ordinary_flash_leaf_rejects_compound_packet(self):
        family_key = "cute_flash_pipeline_family"

        def member(flat: int, packet: str) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float(flat)],
                flat_values=[flat, "fa4", packet],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=flat,
                ),
                status="ok",
            )

        current = member(1, "1x1")
        compound = member(2, "hybrid_deg1_16x8")
        ordinary = member(3, "4x1")
        required_leaf = LFBOPatternSearch._flash_structural_leaf(current)
        assert required_leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.max_generations = 1
        search.patience = 1
        search.frac_selected = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.log = lambda *_args, **_kwargs: None
        search.config_gen = SimpleNamespace(
            _key_to_flat_indices={family_key: ([1], False)}
        )
        search._generate_neighbors = lambda *_args, **_kwargs: [
            compound.flat_values,
            ordinary.flat_values,
        ]
        by_flat = {2: compound, 3: ordinary}
        search.make_unbenchmarked = lambda flat: by_flat[flat[0]]
        search._surrogate_select = lambda candidates, count: candidates[:count]

        visited: set[helion.Config] = set()
        path = search._pruned_pattern_search_from(
            0,
            current,
            visited,
            ((family_key, "fa4"),),
            required_leaf=required_leaf,
        )

        self.assertEqual(next(path), [current, ordinary])
        self.assertNotIn(compound.config, visited)
        self.assertIn(ordinary.config, visited)


class TestCuteFlashSearchPolicyCacheKey(unittest.TestCase):
    @staticmethod
    def _search(
        *,
        effort: str = "full",
        budget_seconds: int | None = None,
        max_generations: int = 20,
        initial_population_strategy: InitialPopulationStrategy = (
            InitialPopulationStrategy.FROM_RANDOM
        ),
        random_seed: int = 0,
        seed_configs: object = None,
        best_available_max_configs: int = 20,
        config_filter: object = None,
        baseline_fn: object = None,
        baseline_atol: float | None = None,
        baseline_rtol: float | None = None,
        best_of_k: int = 1,
        precompile_jobs: int | None = None,
        with_torch_compile_fusion: bool = False,
        search_acf: list[str] | None = None,
        compiler_seed_timeout_retry_repetitions: int | None = None,
    ) -> LFBOTreeSearch:
        settings = helion.Settings(
            backend="cute",
            autotune_effort=effort,
            autotune_budget_seconds=budget_seconds,
            autotune_random_seed=random_seed,
            autotune_seed_configs=seed_configs,
            autotune_best_available_max_configs=best_available_max_configs,
            autotune_config_filter=config_filter,  # type: ignore[arg-type]
            autotune_baseline_fn=baseline_fn,  # type: ignore[arg-type]
            autotune_baseline_atol=baseline_atol,
            autotune_baseline_rtol=baseline_rtol,
            autotune_best_of_k=best_of_k,
            autotune_precompile_jobs=precompile_jobs,
            autotune_with_torch_compile_fusion=with_torch_compile_fusion,
            autotune_search_acf=search_acf or [],
        )
        profile = get_effort_profile(effort)  # type: ignore[arg-type]
        pattern = profile.lfbo_pattern_search
        assert pattern is not None
        search = object.__new__(LFBOTreeSearch)
        search.settings = settings
        search.config_spec = SimpleNamespace(  # type: ignore[assignment]
            compiler_seed_configs=[],
            compiler_coverage_groups=(),
            compiler_seed_timeout_retry_repetitions=(
                compiler_seed_timeout_retry_repetitions
            ),
            backend=SimpleNamespace(config_value_priors_version=1),
            backend_name="cute",
            cute_flash_search_enabled=True,
        )
        search.initial_population = pattern.initial_population
        search.copies = pattern.copies
        search.max_generations = max_generations
        search.initial_population_strategy = initial_population_strategy
        search.best_available_pad_random = pattern.best_available_pad_random
        search.num_neighbors_cap = -1
        search.finishing_rounds = profile.finishing_rounds
        search.min_improvement_delta = 0.001
        search.num_neighbors = 300
        search.radius = 2
        search.frac_selected = 0.1
        search.quantile = 0.1
        search.patience = 1
        search.similarity_penalty = 1.0
        search.polish_rounds = 10
        search.compile_timeout_lower_bound = pattern.compile_timeout_lower_bound
        search.compile_timeout_quantile = pattern.compile_timeout_quantile
        search.flash_structural_search = profile.flash_structural_search
        search._cute_flash_lane_policy_enabled = (
            search.config_spec.cute_flash_search_enabled
            and search.flash_structural_search is not None
        )
        search._flash_promoted_path_limit = search.copies
        search._flash_family_probe_path_limit = 0
        if search._cute_flash_lane_policy_enabled:
            search._flash_promoted_path_limit = 17
            search._flash_family_probe_path_limit = 18
            search.copies = 18
        return search

    def _hash(self, search: BaseSearch, *, enabled: bool = True) -> str:
        return _cute_flash_search_policy_hash(
            search,
            cute_flash_search_enabled=enabled,
        )

    def test_non_flash_preserves_legacy_empty_policy(self) -> None:
        full = self._search()
        quick = self._search(effort="quick", max_generations=5)
        self.assertEqual(self._hash(full, enabled=False), "")
        self.assertEqual(self._hash(quick, enabled=False), "")

    def test_config_generation_policy_version_covers_flash_searches(self) -> None:
        def random_search(*, cute_flash: bool) -> RandomSearch:
            template = self._search(effort="quick", max_generations=5)
            search = object.__new__(RandomSearch)
            search.settings = template.settings
            search.config_spec = copy.deepcopy(template.config_spec)
            search.config_spec.cute_flash_search_enabled = cute_flash
            search.count = 100
            search._benchmark_provider_cls = LocalBenchmarkProvider
            return search

        full = self._search()
        quick = self._search(effort="quick", max_generations=5)
        random_flash = random_search(cute_flash=True)
        for search in (full, quick, random_flash):
            with self.subTest(search=type(search).__name__):
                policy = search.cache_policy()
                assert policy is not None
                algorithm = policy["algorithm"]
                assert isinstance(algorithm, dict)
                self.assertEqual(
                    algorithm["cute_flash_config_generation_policy_version"], 8
                )

        full_policy = full.cache_policy()
        quick_policy = quick.cache_policy()
        random_policy = random_flash.cache_policy()
        assert full_policy is not None
        assert quick_policy is not None
        assert random_policy is not None
        full_algorithm = full_policy["algorithm"]
        quick_algorithm = quick_policy["algorithm"]
        random_algorithm = random_policy["algorithm"]
        assert isinstance(full_algorithm, dict)
        assert isinstance(quick_algorithm, dict)
        assert isinstance(random_algorithm, dict)
        self.assertEqual(
            full_algorithm["cute_flash_lane_policy_version"],
            16,
        )
        self.assertEqual(
            full_algorithm["cute_flash_terminal_coordinate_refinement"],
            {
                "schema_version": 2,
                "policy_version": 2,
                "coordinate_policy": (
                    "same_leaf_full_surface_normalized_coordinate_v2"
                ),
                "rounds": 2,
                "beam_width": 4,
                "radius": 2,
                "minimum_improvement_fraction": 0.001,
                "measurement_policy": "mirrored_rotating_batched_wall_v2",
                "round_target_ms": 200.0,
                "confirmation_target_ms": 5000.0,
            },
        )
        self.assertNotIn("cute_flash_lane_policy_version", quick_algorithm)
        self.assertNotIn("cute_flash_lane_policy_version", random_algorithm)

        non_flash = self._search(effort="quick", max_generations=5)
        non_flash.config_spec.cute_flash_search_enabled = False
        non_flash._cute_flash_lane_policy_enabled = False
        random_non_flash = random_search(cute_flash=False)
        for search in (non_flash, random_non_flash):
            with self.subTest(non_flash_search=type(search).__name__):
                policy = search.cache_policy()
                assert policy is not None
                algorithm = policy["algorithm"]
                assert isinstance(algorithm, dict)
                self.assertNotIn(
                    "cute_flash_config_generation_policy_version", algorithm
                )
                self.assertEqual(algorithm, search._algorithm_cache_policy())

    def test_lane_policy_version_is_cute_flash_only(self) -> None:
        search = self._search()
        search.config_spec.cute_flash_search_enabled = False
        search._cute_flash_lane_policy_enabled = False
        non_flash = search._algorithm_cache_policy()
        self.assertEqual(non_flash["lfbo_version"], 3)
        self.assertNotIn("cute_flash_lane_policy_version", non_flash)
        self.assertIsNone(non_flash["flash_structural_search"])

        search.config_spec.cute_flash_search_enabled = True
        search._cute_flash_lane_policy_enabled = True
        cute_flash_policy = search._algorithm_cache_policy()
        self.assertEqual(cute_flash_policy["lfbo_version"], 3)
        self.assertEqual(cute_flash_policy["cute_flash_lane_policy_version"], 16)
        self.assertEqual(cute_flash_policy["cute_flash_starting_path_limit"], 17)
        self.assertEqual(cute_flash_policy["cute_flash_family_probe_path_limit"], 18)
        self.assertEqual(cute_flash_policy["cute_flash_maximum_path_capacity"], 18)
        self.assertEqual(
            cute_flash_policy["flash_structural_search"],
            search.flash_structural_search,
        )
        self.assertEqual(
            {
                k: v
                for k, v in cute_flash_policy.items()
                if k
                not in {
                    "cute_flash_lane_policy_version",
                    "cute_flash_starting_path_limit",
                    "cute_flash_family_probe_path_limit",
                    "cute_flash_maximum_path_capacity",
                    "cute_flash_terminal_coordinate_refinement",
                    "flash_structural_search",
                }
            },
            {k: v for k, v in non_flash.items() if k != "flash_structural_search"},
        )

        search.flash_structural_search = None
        search._cute_flash_lane_policy_enabled = False
        quick_policy = search._algorithm_cache_policy()
        self.assertNotIn("cute_flash_lane_policy_version", quick_policy)
        self.assertIsNone(quick_policy["flash_structural_search"])
        self.assertEqual(quick_policy["lfbo_version"], 3)

    def test_unlimited_and_finite_family_retention_have_distinct_cache_keys(
        self,
    ) -> None:
        finite = self._search()
        unlimited = self._search()
        assert unlimited.flash_structural_search is not None
        unlimited.flash_structural_search = replace(
            unlimited.flash_structural_search, retained_families=None
        )

        self.assertNotEqual(self._hash(unlimited), self._hash(finite))

    def test_effective_search_breadth_changes_policy(self) -> None:
        baseline = self._hash(self._search())
        variants = (
            self._search(effort="quick", max_generations=5),
            self._search(budget_seconds=60),
            self._search(max_generations=10),
            self._search(
                initial_population_strategy=(
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE
                )
            ),
            self._search(best_of_k=2),
            self._search(precompile_jobs=2),
            self._search(with_torch_compile_fusion=True),
        )
        self.assertTrue(all(self._hash(variant) != baseline for variant in variants))

    def test_random_seed_does_not_change_policy(self) -> None:
        self.assertEqual(
            self._hash(self._search(random_seed=1)),
            self._hash(self._search(random_seed=2)),
        )

    def test_seed_configs_and_best_available_limits_change_policy(self) -> None:
        baseline = self._hash(self._search())
        self.assertNotEqual(
            baseline,
            self._hash(
                self._search(seed_configs=helion.Config(block_sizes=[128, 128]))
            ),
        )
        self.assertNotEqual(
            baseline,
            self._hash(self._search(best_available_max_configs=7)),
        )

    def test_compiler_seed_timeout_retry_changes_policy(self) -> None:
        self.assertNotEqual(
            self._hash(self._search()),
            self._hash(self._search(compiler_seed_timeout_retry_repetitions=3)),
        )

    def test_custom_filter_and_unknown_search_disable_cache_reuse(self) -> None:
        filtered = self._search(config_filter=lambda config: config)
        other_filtered = self._search(config_filter=lambda config: config)
        self.assertEqual(self._hash(filtered), self._hash(filtered))
        self.assertNotEqual(self._hash(filtered), self._hash(other_filtered))
        self.assertIs(filtered._search_policy_cacheable, False)

        class CustomLFBOTreeSearch(LFBOTreeSearch):
            pass

        custom = object.__new__(CustomLFBOTreeSearch)
        custom.__dict__.update(self._search().__dict__)
        other_custom = object.__new__(CustomLFBOTreeSearch)
        other_custom.__dict__.update(self._search().__dict__)
        self.assertEqual(self._hash(custom), self._hash(custom))
        self.assertNotEqual(self._hash(custom), self._hash(other_custom))

    def test_baseline_function_and_tolerances_change_policy(self) -> None:
        def make_baseline(offset):
            def baseline(value):
                return value + offset

            return baseline

        def baseline_increment(value):
            return value + 1

        baseline_identity = make_baseline(0)
        identity = self._search(baseline_fn=baseline_identity)
        other_closure = self._search(baseline_fn=make_baseline(1))
        increment = self._search(baseline_fn=baseline_increment)
        atol = self._search(
            baseline_fn=baseline_identity,
            baseline_atol=1e-2,
        )
        rtol = self._search(
            baseline_fn=baseline_identity,
            baseline_rtol=1e-2,
        )

        self.assertNotEqual(self._hash(identity), self._hash(other_closure))
        self.assertNotEqual(self._hash(identity), self._hash(increment))
        self.assertNotEqual(self._hash(identity), self._hash(atol))
        self.assertNotEqual(self._hash(identity), self._hash(rtol))

    def test_baseline_function_runtime_globals_change_policy(self) -> None:
        global _CACHE_POLICY_BASELINE_SCALE

        def baseline(value):
            return value * _CACHE_POLICY_BASELINE_SCALE

        original = _CACHE_POLICY_BASELINE_SCALE
        try:
            _CACHE_POLICY_BASELINE_SCALE = 1
            first = self._hash(self._search(baseline_fn=baseline))
            _CACHE_POLICY_BASELINE_SCALE = 2
            second = self._hash(self._search(baseline_fn=baseline))
        finally:
            _CACHE_POLICY_BASELINE_SCALE = original

        self.assertNotEqual(first, second)

    def test_unfingerprintable_baseline_helper_disables_cache(self) -> None:
        global _CACHE_POLICY_DYNAMIC_HELPER

        def baseline(value):
            return _CACHE_POLICY_DYNAMIC_HELPER(value)  # type: ignore[operator]

        namespace: dict[str, object] = {}
        exec(
            compile(
                "def dynamic_helper(value):\n    return value\n",
                "<dynamic-helper>",
                "exec",
            ),
            namespace,
        )
        original = _CACHE_POLICY_DYNAMIC_HELPER
        try:
            _CACHE_POLICY_DYNAMIC_HELPER = namespace["dynamic_helper"]
            search = self._search(baseline_fn=baseline)
            self._hash(search)
        finally:
            _CACHE_POLICY_DYNAMIC_HELPER = original

        self.assertIs(search._search_policy_cacheable, False)

    def test_recursive_baseline_closure_disables_cache(self) -> None:
        recursive_value: list[object] = []
        recursive_value.append(recursive_value)

        def baseline(value):
            if recursive_value:
                return value
            raise AssertionError("unreachable")

        search = self._search(baseline_fn=baseline)
        first = self._hash(search)
        second = self._hash(search)

        self.assertEqual(first, second)
        self.assertIs(search._search_policy_cacheable, False)

    def test_advanced_controls_file_contents_change_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            acf = Path(tmp) / "controls.acf"
            acf.write_text("first")
            first = self._hash(self._search(search_acf=[str(acf)]))
            acf.write_text("second")
            second = self._hash(self._search(search_acf=[str(acf)]))

        self.assertNotEqual(first, second)

    def test_unsupported_baseline_disables_cache_reads_and_writes(self) -> None:
        class CallableBaseline:
            def __call__(self, value):
                return value

        search = self._search(baseline_fn=CallableBaseline())
        self._hash(search)
        self.assertIs(search._search_policy_cacheable, False)
        search.log = Mock()

        cache = object.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache.get = Mock(return_value=helion.Config(block_sizes=[1]))  # type: ignore[method-assign]
        cache.put = Mock()  # type: ignore[method-assign]
        cache._run_autotune_trials = Mock(  # type: ignore[method-assign]
            return_value=helion.Config(block_sizes=[2])
        )

        with patch("helion.autotuner.base_cache.should_skip_cache", return_value=False):
            result = cache.autotune()

        self.assertEqual(result, helion.Config(block_sizes=[2]))
        cache.get.assert_not_called()
        cache.put.assert_not_called()
        cache._run_autotune_trials.assert_called_once_with(skip_cache=True)

    def test_llm_seeded_policy_does_not_construct_child_searches(self) -> None:
        search = object.__new__(LLMSeededLFBOTreeSearch)
        search._make_llm_search = Mock(
            side_effect=AssertionError("cache policy constructed the LLM stage")
        )
        search._make_second_stage_search = Mock(
            side_effect=AssertionError("cache policy constructed the second stage")
        )

        self.assertIsNone(search._algorithm_cache_policy())
        search._make_llm_search.assert_not_called()
        search._make_second_stage_search.assert_not_called()

    def test_random_search_policy_rejects_nonlocal_benchmark_provider(self) -> None:
        search = object.__new__(RandomSearch)
        search.count = 100
        search._benchmark_provider_cls = object
        self.assertIsNone(search._algorithm_cache_policy())

        search._benchmark_provider_cls = LocalBenchmarkProvider
        self.assertEqual(
            search._algorithm_cache_policy(), {"random_version": 1, "count": 100}
        )

    def test_registered_de_policy_covers_direct_search_knobs(self) -> None:
        search = object.__new__(DifferentialEvolutionSearch)
        search.population_size = 40
        search.max_generations = 20
        search.crossover_rate = 0.8
        search.immediate_update = True
        search.min_improvement_delta = None
        search.patience = None
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.best_available_pad_random = True
        search.finishing_rounds = 1
        search.compile_timeout_lower_bound = 30.0
        search.compile_timeout_quantile = 0.9
        policy = search._algorithm_cache_policy()
        self.assertEqual(policy["population_size"], 40)
        self.assertEqual(policy["crossover_rate"], 0.8)
        self.assertIs(policy["immediate_update"], True)

    def test_registered_search_constructor_knobs_are_classified(self) -> None:
        classes = (
            DESurrogateHybrid,
            LFBOPatternSearch,
            LFBOTreeSearch,
            LLMGuidedSearch,
            LLMSeededSearch,
            LLMSeededLFBOTreeSearch,
            DifferentialEvolutionSearch,
            FiniteSearch,
            PatternSearch,
            RandomSearch,
        )
        aliases = {"benchmark_provider_cls": "benchmark_provider"}
        secrets = {
            LLMGuidedSearch: {"api_key"},
        }
        inherited_fields = {
            "_cute_flash_lane_policy_enabled",
            "_flash_family_probe_path_limit",
            "_flash_promoted_path_limit",
            "immediate_update",
            "population_size",
            "provider",
            "second_stage_algorithm",
        }

        for search_cls in classes:
            with self.subTest(search_cls=search_cls.__name__):
                parameters = {
                    name
                    for name in inspect.signature(search_cls.__init__).parameters
                    if name not in {"self", "kernel", "args"}
                }
                search = object.__new__(search_cls)
                search.config_spec = _cute_flash_test_config_spec()
                for name in parameters | inherited_fields:
                    setattr(search, name, 1)
                if hasattr(search, "flash_structural_search"):
                    search.flash_structural_search = get_effort_profile(
                        "full"
                    ).flash_structural_search
                search._benchmark_provider_cls = LocalBenchmarkProvider
                search.configs = []
                search.provider = "provider"
                search.model = "model"
                policy = search._algorithm_cache_policy()
                if isinstance(search, LLMSeededSearch):
                    self.assertIsNone(policy)
                    continue
                assert policy is not None
                policy_keys = set(policy)
                classified = {
                    aliases.get(name, name)
                    for name in parameters - secrets.get(search_cls, set())
                }
                self.assertLessEqual(classified, policy_keys)

    def test_strict_key_includes_policy_and_preserves_legacy_default(self) -> None:
        from helion.autotuner.base_cache import StrictAutotuneCacheKey

        common = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "13.0",
            "backend": "cute",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
            "helion_key": "H",
            "torch_key": "T",
            "triton_key": "R",
        }
        legacy = StrictAutotuneCacheKey(**common)
        best_of_five = StrictAutotuneCacheKey(**common, best_of_k=5)
        first = StrictAutotuneCacheKey(**common, search_policy_hash="first")
        second = StrictAutotuneCacheKey(**common, search_policy_hash="second")
        self.assertEqual(
            repr(legacy),
            "StrictAutotuneCacheKey(specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', runtime_name='13.0', "
            "backend='cute', config_spec_hash='h1', extra_cache_key='', "
            "helion_key='H', torch_key='T', triton_key='R')",
        )
        self.assertEqual(
            legacy.stable_hash(),
            "b92201d3cff5d3b9c92a40d17ac8e98022d86261bfdd0b638d3296a0c0908a4d",
        )
        self.assertIn("best_of_k=5", repr(best_of_five))
        self.assertNotEqual(legacy.stable_hash(), best_of_five.stable_hash())
        self.assertNotEqual(first.stable_hash(), second.stable_hash())

    def test_policy_hash_is_structural_and_legacy_default_is_unchanged(self) -> None:
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        common = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "13.0",
            "backend": "cute",
            "config_spec_hash": "h1",
        }
        legacy = LooseAutotuneCacheKey(**common, extra_cache_key="")
        policy = LooseAutotuneCacheKey(
            **common,
            extra_cache_key="",
            search_policy_hash="policy-v1",
        )
        suffix_lookalike = LooseAutotuneCacheKey(
            **common,
            extra_cache_key="search_policy_hash='policy-v1'",
        )

        self.assertNotIn("search_policy_hash", repr(legacy))
        self.assertIn("search_policy_hash='policy-v1'", repr(policy))
        self.assertNotEqual(legacy.stable_hash(), policy.stable_hash())
        self.assertNotEqual(suffix_lookalike.stable_hash(), policy.stable_hash())


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@skipUnlessCuteAvailable("CUTLASS CuTe Python DSL is not available")
@onlyBackends(["cute"])
class TestCuteAutotuner(TestCase):
    def test_implicit_call_uses_autotuner_fn(self) -> None:
        calls: list[bool] = []

        def autotuner_fn(bound_kernel, args, **kwargs):
            class RecordingAutotuner:
                def autotune(self, *, skip_cache: bool = False):
                    calls.append(skip_cache)
                    return bound_kernel.config_spec.default_config()

            return RecordingAutotuner()

        @helion.kernel(
            backend="cute",
            autotuner_fn=autotuner_fn,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))
        self.assertEqual(calls, [False])

    def test_cute_config_generation_repairs_num_threads(self) -> None:
        @helion.kernel(backend="cute", autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([16, 64], device=DEVICE),
            torch.randn([16, 64], device=DEVICE),
        )
        bound = add.bind(args)
        gen = ConfigGeneration(bound.config_spec)
        flat_keys = {
            key for key, _count, _is_sequence in gen.config_spec.flat_key_layout()
        }
        # ``loop_orders`` is exposed for the CuTe non-tcgen05 search
        # surface. ``cute_vector_widths`` is the per-axis vec width slot
        # registered for non-reduction tile blocks.
        # ``load_eviction_policies`` carries per-load-site L1 eviction
        # hints (lowered on the vectorized load forms). ``flatten_loops``
        # selects the flattened multi-dim tile form (flat base-pointer
        # vectorization). ``cute_proven_bounds`` selects proof-based mask
        # elimination. The set still excludes Triton-style knobs that
        # the CuTe path does not consume.
        # Resident packet controls are explicit CuTe config fields; unseeded
        # configs retain their false-only search domains.
        self.assertEqual(
            flat_keys,
            {
                "block_sizes",
                "num_threads",
                "flatten_loops",
                "loop_orders",
                "cute_vector_widths",
                "cute_lane_layouts",
                "cute_cluster_n",
                "cute_min_blocks_per_mp",
                "cute_proven_bounds",
                "cute_packet_prefetch",
                "cute_independent_reduction",
                "cute_cluster_warp_online",
                "cute_register_row",
                "cute_register_row_tensor_copy",
                "cute_scalar_epilogue",
                "cute_scalar_epilogue_keep_inputs",
                "cute_scalar_epilogue_owner_fold",
                "cute_warp_fp32_redux",
                "cute_replicated_reduction",
                "cute_vector_packet_unroll",
                "cute_vloop_sink",
                "cute_lane_unroll",
                "cute_pdl",
                "cute_store_policy",
                "load_eviction_policies",
            },
        )

        repaired = gen.unflatten(
            gen.flatten(helion.Config(block_sizes=[16, 64], num_threads=[128, 128]))
        )
        self.assertEqual(repaired.block_sizes, [16, 64])
        self.assertEqual(repaired.num_threads, [16, 64])

        configs = [gen.random_config() for _ in range(20)]
        self.assertTrue(any(config.num_threads for config in configs))
        for config in configs:
            self.assertLessEqual(
                set(config.config),
                {
                    "block_sizes",
                    "num_threads",
                    "flatten_loops",
                    "loop_orders",
                    "cute_vector_widths",
                    "cute_lane_layouts",
                    "cute_cluster_n",
                    "cute_min_blocks_per_mp",
                    "cute_proven_bounds",
                    "cute_packet_prefetch",
                    "cute_independent_reduction",
                    "cute_cluster_warp_online",
                    "cute_register_row",
                    "cute_register_row_tensor_copy",
                    "cute_scalar_epilogue",
                    "cute_scalar_epilogue_keep_inputs",
                    "cute_scalar_epilogue_owner_fold",
                    "cute_warp_fp32_redux",
                    "cute_replicated_reduction",
                    "cute_vector_packet_unroll",
                    "cute_vloop_sink",
                    "cute_lane_unroll",
                    "cute_pdl",
                    "cute_store_policy",
                    "load_eviction_policies",
                },
            )
            self.assertNotIn("persistent", config.pid_type)
            explicit_threads = [nt for nt in config.num_threads if nt > 0]
            if explicit_threads:
                self.assertLessEqual(math.prod(explicit_threads), 1024)
            for block_size, num_threads in zip(
                config.block_sizes,
                config.num_threads,
                strict=False,
            ):
                if num_threads > 0:
                    self.assertLessEqual(num_threads, block_size)
                    self.assertEqual(block_size % num_threads, 0)
        # Deterministic round-trip pins the widened surface: a config
        # explicitly built with ``loop_orders=[[1, 0]]`` must survive
        # flatten/unflatten unchanged (otherwise the autotuner cannot
        # actually explore the alternate order).
        round_tripped = gen.unflatten(
            gen.flatten(
                helion.Config(
                    block_sizes=[16, 64],
                    num_threads=[16, 64],
                    loop_orders=[[1, 0]],
                    cute_proven_bounds=True,
                )
            )
        )
        self.assertEqual(round_tripped.loop_orders, [[1, 0]])
        self.assertTrue(round_tripped.config["cute_proven_bounds"])

    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="tcgen05 requires CUDA capability >= 10.0"
    )
    def test_cute_tcgen05_search_surface_includes_loop_orders(self) -> None:
        """tcgen05 autotuning can explore alternate output-tile orders."""
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
        )
        bound = _get_examples_matmul().bind(args)
        # Confirm we are on the tcgen05 search branch.
        self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
        flat_keys = {
            key for key, _count, _is_sequence in bound.config_spec.flat_key_layout()
        }
        self.assertIn("loop_orders", flat_keys)

        gen = ConfigGeneration(bound.config_spec)
        config = helion.Config(
            block_sizes=[128, 128, 64],
            loop_orders=[[1, 0]],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
        )
        round_tripped = gen.unflatten(gen.flatten(config))
        self.assertEqual(round_tripped.loop_orders, [[1, 0]])
        code = bound.to_triton_code(round_tripped)
        self.assertIn("cute.gemm(", code)
        self.assertIn("StaticPersistentTileScheduler", code)

        actual = bound.compile_config(round_tripped)(*args)
        torch.testing.assert_close(actual, args[0] @ args[1], atol=0.125, rtol=0.02)

    def test_cute_flash_search_surface(self) -> None:
        """Flash attention exposes a general, normalized CuTe search surface.

        The standard surface must be broad enough to tune rather than replay a
        sequence-length winner. Structural legality can still constrain the
        available families. Legacy and manual fixed configs remain accepted,
        while unrelated CuTe kernels never receive flash-attention knobs.
        """
        from helion._compiler.cute import cute_flash

        @helion.kernel(backend="cute", static_shapes=True)
        def flash_attention(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        @helion.kernel(backend="cute", static_shapes=True)
        def flash_attention_with_aux(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            aux = torch.empty(
                [q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32
            )
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                aux[tile_b, tile_m] = torch.zeros_like(l_i)
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size()), aux.view(q_in.size()[:-1])

        @helion.kernel(backend="cute", static_shapes=True)
        def sparse_flash_attention(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    delta = tile_m.index[None, :, None] - tile_n.index[None, None, :]
                    qk = torch.where((delta >= 0) & (delta <= 64), qk, float("-inf"))
                    m_ij_keepdim = torch.maximum(
                        m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
                    )
                    qk = qk - m_ij_keepdim
                    m_ij = m_ij_keepdim.squeeze(-1)
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        def active_choices(fragment):
            return (
                fragment.choices
                if fragment.search_choices is None
                else fragment.search_choices
            )

        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        attn_bound = flash_attention.bind((q, k, v))
        self.assertTrue(attn_bound.config_spec.cute_flash_search_enabled)
        cache = default_autotuner_fn(attn_bound, (q, k, v))
        self.assertIsInstance(cache, LocalAutotuneCache)
        self.assertTrue(cache.key.search_policy_hash)  # type: ignore[attr-defined]
        self.assertIs(cache.autotuner._search_policy_cacheable, True)
        strict_cache = StrictLocalAutotuneCache(cache.autotuner)
        self.assertEqual(
            strict_cache.key.search_policy_hash,
            cache.key.search_policy_hash,  # type: ignore[attr-defined]
        )
        attn_keys = {
            key
            for key, _count, _is_sequence in attn_bound.config_spec.flat_key_layout()
        }
        self.assertLessEqual(
            set(cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS),
            attn_keys,
        )
        self.assertTrue(set(cute_flash.FLASH_LEGACY_CONFIG_KEYS).isdisjoint(attn_keys))
        self.assertNotIn(cute_flash.FLASH_Q_TILE_COUNT_KEY, attn_keys)
        for generic_key in ("num_threads", "loop_orders", "cute_vector_widths"):
            self.assertNotIn(generic_key, attn_keys)

        bound_fragments = attn_bound.config_spec._flat_fields()
        self.assertLessEqual(
            {"fa4", "ws_overlap"},
            set(active_choices(bound_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
        )
        for key in (
            cute_flash.FLASH_KV_STAGE_KEY,
            cute_flash.FLASH_E2E_SCHEDULE_KEY,
            cute_flash.FLASH_EXP2_PACKET_KEY,
            cute_flash.FLASH_STAT_TRANSPORT_KEY,
            cute_flash.FLASH_PIPELINE_FAMILY_KEY,
            cute_flash.FLASH_SOFTMAX_DISC_KEY,
            cute_flash.FLASH_EPI_TMA_KEY,
            cute_flash.FLASH_PACKED_REDUCE_KEY,
            cute_flash.FLASH_PERSISTENT_LOOP_KEY,
            cute_flash.FLASH_SP_ROW_SUM_KEY,
            cute_flash.FLASH_SOFTMAX_SETUP_KEY,
            cute_flash.FLASH_EPI_TMA_SETUP_KEY,
        ):
            self.assertGreater(
                len(active_choices(bound_fragments[key])),
                1,
                key,
            )

        # A single contiguous-interval mask (sliding window here) prunes KV
        # tiles but runs on both families: the fa4 pair body takes the union
        # of its two query tiles' intervals, so the surface searches fa4 next
        # to ws_overlap and defaults to fa4 at an even tile count. Only plans
        # whose active tiles are not one interval (two composed masks) keep
        # the ws_overlap lock; the per-row KV lower bound evaluates its range
        # prelude per resident query tile, so it runs on the pair body too.
        sparse_bound = sparse_flash_attention.bind((q, k, v))
        sparse_spec = sparse_bound.config_spec
        self.assertTrue(sparse_spec._cute_flash_has_kv_tile_pruning)
        self.assertFalse(sparse_spec._cute_flash_requires_ws_overlap)
        sparse_bound_fragments = sparse_spec._flat_fields()
        self.assertEqual(
            set(
                active_choices(
                    sparse_bound_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
                )
            ),
            {"fa4", "ws_overlap"},
        )
        sparse_default = sparse_spec.default_config().config
        self.assertEqual(sparse_default[cute_flash.FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertTrue(sparse_default[cute_flash.FLASH_PACKED_REDUCE_KEY])
        stale_sparse = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="ws_overlap",
        )
        sparse_spec.normalize(stale_sparse)
        self.assertEqual(
            stale_sparse.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "ws_overlap",
        )
        self.assertNotIn(cute_flash.FLASH_TOPOLOGY_KEY, stale_sparse.config)
        window = AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=64)
        causal = AttentionScoreModifier(CAUSAL_MASK_KIND)
        row_kv_begin = AttentionScoreModifier(
            ROW_KV_BEGIN_MASK_KIND, tensor_name="lb", index_mode="identity"
        )
        for modifiers, locked in (
            ((window,), False),
            ((causal, window), True),
            ((row_kv_begin,), False),
        ):
            plan = AttentionScorePlan(
                head_dim=64, qk_scale_log2=1.0, modifiers=modifiers
            )
            self.assertTrue(plan.has_kv_tile_pruning)
            self.assertIs(plan.requires_ws_overlap, locked, modifiers)

        # A ws_overlap-locked plan (two composed masks here: the causal mask
        # applied by its own ``torch.where`` ahead of the window, which detects
        # as two mask modifiers whose active tiles are not one interval)
        # carries a single-family surface: a config that asks for the fa4
        # family (a cached winner of another plan, say) normalizes back to
        # ws_overlap.
        @helion.kernel(backend="cute", static_shapes=True)
        def chained_sparse_flash_attention(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
                    qk = torch.where(causal, qk, float("-inf"))
                    delta = tile_m.index[None, :, None] - tile_n.index[None, None, :]
                    qk = torch.where((delta >= 0) & (delta <= 64), qk, float("-inf"))
                    m_ij_keepdim = torch.maximum(
                        m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
                    )
                    qk = qk - m_ij_keepdim
                    m_ij = m_ij_keepdim.squeeze(-1)
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        locked_spec = chained_sparse_flash_attention.bind((q, k, v)).config_spec
        self.assertTrue(locked_spec.cute_flash_search_enabled)
        self.assertTrue(locked_spec._cute_flash_requires_ws_overlap)
        self.assertEqual(
            set(
                active_choices(
                    locked_spec._flat_fields()[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
                )
            ),
            {"ws_overlap"},
        )
        wants_fa4 = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
        )
        locked_spec.normalize(wants_fa4)
        self.assertEqual(
            wants_fa4.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY], "ws_overlap"
        )
        self.assertEqual(
            locked_spec.default_config().config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "ws_overlap",
        )

        # Aligned lengths share the same general search space. Only structural
        # legality, such as the parity needed by two-CTA families, may narrow it.
        def direct_fragments(num_kv, *, is_causal=False):
            return cute_flash.flash_autotune_fragments(
                64,
                num_kv,
                num_bh=16,
                dtype=torch.float16,
                is_causal=is_causal,
                standard_dense_output=not is_causal,
                standard_causal_output=is_causal,
            )

        dense_short = direct_fragments(64)
        dense_long = direct_fragments(512)
        self.assertEqual(
            {
                key: set(active_choices(fragment))
                for key, fragment in dense_short.items()
            },
            {
                key: set(active_choices(fragment))
                for key, fragment in dense_long.items()
            },
        )
        dense_families = set(
            active_choices(dense_short[cute_flash.FLASH_PIPELINE_FAMILY_KEY])
        )
        self.assertLessEqual(
            {"fa4", "ws_overlap", "fa4_2cta", "fa4_clc"},
            dense_families,
        )

        paired_only = direct_fragments(66)
        self.assertNotIn(
            "fa4_2cta",
            active_choices(paired_only[cute_flash.FLASH_PIPELINE_FAMILY_KEY]),
        )
        odd = direct_fragments(65)
        self.assertEqual(
            set(active_choices(odd[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"ws_overlap"},
        )

        causal_short = direct_fragments(64, is_causal=True)
        causal_long = direct_fragments(512, is_causal=True)
        self.assertEqual(
            {
                key: set(active_choices(fragment))
                for key, fragment in causal_short.items()
            },
            {
                key: set(active_choices(fragment))
                for key, fragment in causal_long.items()
            },
        )
        self.assertEqual(
            set(active_choices(causal_short[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"fa4", "fa4_2cta_causal", "ws_overlap"},
        )
        for key in (
            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY,
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY,
        ):
            self.assertGreater(len(active_choices(causal_short[key])), 1, key)
        # FA4's L2-sized group (0, resolved per length), the serial head order
        # and the whole 16-head grid.
        self.assertEqual(
            set(active_choices(causal_short[cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY])),
            {0, 1, 16},
        )

        sparse = cute_flash.flash_autotune_fragments(
            64,
            64,
            num_bh=16,
            dtype=torch.float16,
            has_kv_tile_pruning=True,
            requires_ws_overlap=True,
        )
        self.assertEqual(
            set(active_choices(sparse[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"ws_overlap"},
        )

        regular_small = cute_flash.flash_autotune_fragments(
            64,
            1,
            small_biased_candidate=False,
        )
        biased_small = cute_flash.flash_autotune_fragments(
            64,
            1,
            small_biased_candidate=True,
        )
        self.assertEqual(
            set(active_choices(regular_small[cute_flash.FLASH_SMALL_BIASED_KEY])),
            {True},
        )
        self.assertEqual(
            set(active_choices(biased_small[cute_flash.FLASH_SMALL_BIASED_KEY])),
            {False, True},
        )

        # Environment defaults may bias ordering but cannot collapse a standard
        # tuning dimension into a fixed value.
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_KV_STAGE": "6",
                "HELION_CUTE_FLASH_E2E_SCHEDULE": "xu",
            },
        ):
            env_fragments = direct_fragments(64)
        self.assertIn(
            6,
            active_choices(env_fragments[cute_flash.FLASH_KV_STAGE_KEY]),
        )
        self.assertGreater(
            len(active_choices(env_fragments[cute_flash.FLASH_KV_STAGE_KEY])),
            1,
        )
        self.assertIn(
            "xu",
            active_choices(env_fragments[cute_flash.FLASH_E2E_SCHEDULE_KEY]),
        )
        self.assertGreater(
            len(active_choices(env_fragments[cute_flash.FLASH_E2E_SCHEDULE_KEY])),
            1,
        )

        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_TOPOLOGY": "invalid-topology"},
        ):
            invalid_topology_fragments = direct_fragments(64)
        invalid_family_fragment = invalid_topology_fragments[
            cute_flash.FLASH_PIPELINE_FAMILY_KEY
        ]
        self.assertEqual(invalid_family_fragment.default(), "ws_overlap")
        self.assertGreater(len(active_choices(invalid_family_fragment)), 1)

        for manual_family, causal in (("fa4_deep_1cta", True),):
            with (
                self.subTest(manual_family=manual_family),
                patch.dict(
                    os.environ,
                    {"HELION_CUTE_FLASH_PIPELINE_FAMILY": manual_family},
                ),
            ):
                manual_fragments = direct_fragments(512, is_causal=causal)
            manual_fragment = manual_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
            self.assertIn(manual_family, manual_fragment.choices)
            self.assertNotEqual(manual_fragment.default(), manual_family)
            self.assertNotIn(manual_family, active_choices(manual_fragment))

        long_q, long_k, long_v = (
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        long_bound = flash_attention.bind((long_q, long_k, long_v))
        long_gen = ConfigGeneration(long_bound.config_spec)

        legacy_compound = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_clc=True,
            cute_flash_local_tma_partition=True,
            cute_flash_tensor_4d_tma=True,
            cute_flash_clc_heads_per_batch=1,
            cute_flash_clc_pdl=True,
            cute_flash_clc_stages=3,
            cute_flash_q_tile_count=-1,
            cute_flash_mma_interleave=True,
        )
        family_compound = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_clc_local_tma_4d",
            cute_flash_clc_heads_per_batch=1,
            cute_flash_clc_pdl=True,
            cute_flash_clc_stages=3,
        )
        self.assertEqual(
            long_gen.flatten(legacy_compound),
            long_gen.flatten(family_compound),
        )
        long_bound.config_spec.normalize(legacy_compound)
        long_bound.config_spec.normalize(family_compound)
        self.assertEqual(legacy_compound, family_compound)
        self.assertEqual(
            legacy_compound.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "fa4_clc_local_tma_4d",
        )
        self.assertEqual(
            legacy_compound.config[cute_flash.FLASH_Q_TILE_COUNT_KEY],
            2,
        )
        for legacy_key in cute_flash.FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            self.assertNotIn(legacy_key, legacy_compound)

        legacy_two_cta = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_use_2cta=True,
        )
        family_two_cta = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_2cta",
        )
        long_bound.config_spec.normalize(legacy_two_cta)
        long_bound.config_spec.normalize(family_two_cta)
        self.assertEqual(legacy_two_cta, family_two_cta)
        self.assertEqual(
            family_two_cta.config[cute_flash.FLASH_Q_TILE_COUNT_KEY],
            2,
        )

        family_authoritative = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_topology="ws_overlap",
            cute_flash_use_2cta=True,
            cute_flash_cga2_local=True,
            cute_flash_clc=True,
            cute_flash_local_tma_partition=True,
            cute_flash_tensor_4d_tma=True,
        )
        long_bound.config_spec.normalize(family_authoritative, _fix_invalid=True)
        self.assertEqual(
            family_authoritative.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "fa4",
        )
        for legacy_key in cute_flash.FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            self.assertNotIn(legacy_key, family_authoritative)

        inactive_children = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_persistent=False,
            cute_flash_persistent_ctas_per_sm=4,
            cute_flash_recompute_tile_coords=True,
            cute_flash_persistent_loop="counted",
            cute_flash_softmax_disc=True,
            cute_flash_sp_row_sum="whole",
            cute_flash_epi_tma=False,
            cute_flash_epi_tma_setup="role_local",
            cute_flash_clc_heads_per_batch=64,
            cute_flash_clc_pdl=True,
            cute_flash_clc_stages=3,
        )
        long_bound.config_spec.normalize(inactive_children, _fix_invalid=True)
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_PERSISTENT_CTAS_PER_SM_KEY],
            1,
        )
        self.assertFalse(
            inactive_children.config[cute_flash.FLASH_RECOMPUTE_TILE_COORDS_KEY]
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_PERSISTENT_LOOP_KEY],
            "while",
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_SP_ROW_SUM_KEY],
            "fragment",
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_EPI_TMA_SETUP_KEY],
            "shared",
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
            0,
        )

        source_schedule = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_persistent=True,
            cute_flash_persistent_loop="counted",
            cute_flash_softmax_disc=False,
            cute_flash_sp_row_sum="whole",
            cute_flash_softmax_setup="stage_local",
            cute_flash_epi_tma=True,
            cute_flash_epi_tma_setup="role_local",
        )
        long_bound.config_spec.normalize(source_schedule)
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_PERSISTENT_LOOP_KEY],
            "counted",
        )
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_SP_ROW_SUM_KEY],
            "whole",
        )
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_SOFTMAX_SETUP_KEY],
            "stage_local",
        )
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_EPI_TMA_SETUP_KEY],
            "role_local",
        )

        manual_values = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_rescale_threshold=12.0,
            cute_flash_rescale_chunk_cols=64,
            cute_flash_corr_regs=72,
            cute_flash_other_regs=40,
        )
        long_bound.config_spec.normalize(manual_values)
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_RESCALE_THRESHOLD_KEY],
            12.0,
        )
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY],
            64,
        )
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_CORR_REGS_KEY],
            72,
        )
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_OTHER_REGS_KEY],
            40,
        )

        cga2_paired_epilogue = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_cga2_local_tma_4d",
            cute_flash_persistent=True,
            cute_flash_epi_stg=True,
            cute_flash_epi_stg_gmem="pair",
        )
        long_bound.config_spec.normalize(cga2_paired_epilogue)
        self.assertEqual(
            cga2_paired_epilogue.config[cute_flash.FLASH_EPI_STG_GMEM_KEY],
            "stage",
        )

        clc_wide_rescale = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_clc_local_tma_4d",
            cute_flash_rescale_chunk_cols=64,
            cute_flash_clc_heads_per_batch=1,
        )
        long_bound.config_spec.normalize(clc_wide_rescale)
        self.assertEqual(
            clc_wide_rescale.config[cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY],
            32,
        )

        manual_packet = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_exp2_packet="deg2_16x6",
        )
        # The degree-2 polynomial packet is an approximation: without the
        # fast_math setting it is not a legal value; with it the manual packet
        # keeps its measured cadence.
        with self.assertRaisesRegex(exc.InvalidConfig, "deg2_16x6"):
            long_bound.config_spec.normalize(copy.deepcopy(manual_packet))
        fast_math_bound = helion.kernel(
            flash_attention.fn, backend="cute", static_shapes=True, fast_math=True
        ).bind((long_q, long_k, long_v))
        fast_math_bound.config_spec.normalize(manual_packet)
        self.assertEqual(
            manual_packet.config[cute_flash.FLASH_EXP2_PACKET_KEY],
            "deg2_16x6",
        )
        self.assertEqual(
            manual_packet.config[cute_flash.FLASH_E2E_SCHEDULE_KEY],
            "16/6",
        )

        legacy_split = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_e2e_schedule="xu",
            cute_flash_exp2_impl="split",
            cute_flash_e2e_freq=16,
            cute_flash_e2e_res=4,
            cute_flash_e2e_offset=2,
        )
        long_bound.config_spec.normalize(legacy_split)
        self.assertEqual(
            legacy_split.config[cute_flash.FLASH_E2E_SCHEDULE_KEY],
            "16/4",
        )
        self.assertEqual(
            legacy_split.config[cute_flash.FLASH_E2E_OFFSET_KEY],
            2,
        )

        invalid_manual = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_corr_regs=96,
        )
        with self.assertRaises(exc.InvalidConfig):
            long_bound.config_spec.normalize(invalid_manual)

        fast_math_gen = ConfigGeneration(fast_math_bound.config_spec)
        for generation, config in (
            (long_gen, family_compound),
            (long_gen, family_two_cta),
            (long_gen, source_schedule),
            (long_gen, manual_values),
            # The degree-2 packet is legal on the fast_math surface only.
            (fast_math_gen, manual_packet),
        ):
            flat = generation.flatten(config)
            generation.encode_config(flat)
            expected = dict(config.config)
            expected.pop("cute_vector_widths", None)
            expected.pop("cute_lane_layouts", None)
            self.assertEqual(generation.unflatten(flat).config, expected)

        random_configs = [long_gen.random_config() for _ in range(24)]
        self.assertGreater(
            len({tuple(long_gen.flatten(config)) for config in random_configs}),
            1,
        )
        for random_config in random_configs:
            values = random_config.config
            effective = cute_flash.flash_effective_config_values(
                cute_flash.flash_config_from_config(
                    values,
                    64,
                    64,
                    num_bh=1,
                    standard_dense_output=True,
                    target_device_capability=long_bound.config_spec.target_device_capability,
                    tmem_rowmax_compatible=long_bound.config_spec._cute_flash_tmem_rowmax_compatible,
                )
            )
            for key, value in effective.items():
                self.assertEqual(values[key], value, key)
            for legacy_key in cute_flash.FLASH_LEGACY_CONFIG_KEYS:
                self.assertNotIn(legacy_key, values)

        q_odd, k_odd, v_odd = (
            torch.randn(2, 8, 384, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        odd_bound = flash_attention.bind((q_odd, k_odd, v_odd))
        odd_fragments = odd_bound.config_spec._flat_fields()
        self.assertEqual(
            set(active_choices(odd_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"ws_overlap"},
        )
        stale_fa4 = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
        )
        odd_bound.config_spec.normalize(stale_fa4)
        self.assertEqual(
            stale_fa4.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "ws_overlap",
        )
        self.assertNotIn(cute_flash.FLASH_TOPOLOGY_KEY, stale_fa4)

        aux_bound = flash_attention_with_aux.bind((q, k, v))
        self.assertFalse(aux_bound.config_spec.cute_flash_search_enabled)
        aux_keys = {
            key for key, _count, _is_sequence in aux_bound.config_spec.flat_key_layout()
        }
        self.assertIn("num_threads", aux_keys)
        self.assertTrue(set(cute_flash.FLASH_CONFIG_KEYS).isdisjoint(aux_keys))

        mm_bound = _get_examples_matmul().bind(
            (
                torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
                torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            )
        )
        self.assertFalse(mm_bound.config_spec.cute_flash_search_enabled)
        mm_keys = {
            key for key, _count, _is_sequence in mm_bound.config_spec.flat_key_layout()
        }
        self.assertTrue(set(cute_flash.FLASH_CONFIG_KEYS).isdisjoint(mm_keys))

    def test_cute_flash_two_cta_uses_general_softmax_register_search(self) -> None:
        from helion._compiler.cute.cute_flash import _FLASH_SOFTMAX_REGS_VALUES
        from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
        from helion._compiler.cute.cute_flash import flash_autotune_fragments

        general = set(_FLASH_SOFTMAX_REGS_VALUES)
        self.assertLessEqual({176, 184, 192, 200}, general)
        for num_kv in (512, 1536, 2048):
            with self.subTest(num_kv=num_kv):
                fragment = flash_autotune_fragments(
                    64,
                    num_kv,
                    dtype=torch.float16,
                    is_causal=False,
                    standard_dense_output=True,
                    pipeline_family_override="fa4_2cta",
                )[FLASH_SOFTMAX_REGS_KEY]
                self.assertEqual(set(fragment.search_choices or ()), general)
                self.assertLessEqual(general, set(fragment.choices))


if __name__ == "__main__":
    unittest.main()
