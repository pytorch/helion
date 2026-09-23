from __future__ import annotations

import ast
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test.cute_population_contracts import checked_initial_population
from test.cute_population_contracts import with_flat_min_blocks_default
from test.test_cute_flat_grouped_ir import _host
from test.test_cute_flat_grouped_ir import cuda_binding
from test.test_cute_flat_grouped_public import _native_source
from test.test_cute_flat_grouped_public import _runtime_arguments
from test.test_cute_flat_grouped_public import _source
from test.test_cute_tma_rn_two_stage import _scalar_prefix_functions
from test.test_cute_tma_rn_two_stage import low_config
from test.test_cute_tma_rn_two_stage import low_schedule

import helion
from helion._compiler.autotuner_heuristics.cute_grouped_rna import (
    CuteGroupedRnaHeuristic,
)
from helion._compiler.cute import tcgen05_flattened_prefix as prefix_module
from helion._compiler.cute.tcgen05_flat_grouped_config import K_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import PREFIX_SCAN_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import RESIDENT_CTAS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import STAGES_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import WARPS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import (
    normalize_grouped_rna_config,
)
from helion._compiler.cute.tcgen05_flattened_prefix import FlatGroupedBlockPrefixPlan
from helion._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from helion._compiler.cute.tcgen05_storage import NativeSharedAllocation
from helion._compiler.cute.tcgen05_storage import NativeSharedPlacement
from helion._compiler.cute.tcgen05_storage import NativeSharedStoragePlan
from helion._compiler.cute.tcgen05_tma_rn import CONVERSION_KEY
from helion._compiler.cute.tcgen05_tma_rn import TMA_RN
from helion._compiler.cute.tcgen05_tma_rn import WARP_RAW
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch


def _block_config(bound, ctas=2, wrapped=True):
    return helion.Config.from_dict(
        low_config(bound, ctas, wrapped).config | {PREFIX_SCAN_KEY: "block"}
    )


def test_suffix_preserves_offsets_base_and_all_live_bytes():
    allocations = (
        NativeSharedAllocation("large", 129, 128),
        NativeSharedAllocation("small", 4, 4),
    )
    original = NativeSharedStoragePlan.pack(allocations)
    assert original.cache_identity == (("large", 129, 128, 0), ("small", 4, 4, 132))
    appended = original.append(NativeSharedAllocation("summary", 64, 16))
    assert appended.placements[:-1] == original.placements
    assert appended.cache_identity[-1] == ("summary", 64, 16, 144)
    assert appended.alignment == 128
    memory = bytearray([255] * appended.bytes)
    for index, placement in enumerate(appended.placements):
        for offset in range(placement.offset, placement.end):
            assert memory[offset] == 255
            memory[offset] = index
    for index, placement in enumerate(appended.placements):
        assert (
            memory[placement.offset : placement.end]
            == bytes([index]) * placement.allocation.bytes
        )
    for allocation in (
        NativeSharedAllocation("summary", 1, 16),
        NativeSharedAllocation("stronger", 1, 256),
        NativeSharedAllocation("overflow", 1 << 31, 16),
    ):
        with pytest.raises(ValueError):
            appended.append(allocation)
    with pytest.raises(ValueError):
        NativeSharedStoragePlan(
            (
                NativeSharedPlacement(NativeSharedAllocation("base", 8, 8), 0),
                NativeSharedPlacement(NativeSharedAllocation("bad", 16, 16), 16),
            )
        )
    assert NativeSharedStoragePlan.pack(allocations) == original


@pytest.mark.parametrize(
    "groups,warps", [(0, 8), (257, 8), (1, 0), (1, 33), (True, 1), (1, True)]
)
def test_block_plan_rejects_incomplete_or_unbounded_participation(groups, warps):
    with pytest.raises(ValueError):
        FlatGroupedBlockPrefixPlan(groups, warps)


def test_original_resource_suffix_and_exact_two_resident_budget():
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        ordinary = low_schedule(bound, 2)
        selected = low_schedule(bound, 2, prefix_scan="block")
        assert ordinary is not None and selected is not None
        assert selected.block_prefix == FlatGroupedBlockPrefixPlan(256, 8)
        assert selected.thread_block == (32, 8, 1)
        assert selected.storage.placements[:-1] == ordinary.storage.placements
        assert selected.storage.cache_identity[-1] == (
            "tcgen05_prefix_chunk_summaries_ptr",
            64,
            16,
            99856,
        )
        assert selected.storage.bytes == 99920
        assert selected.storage.bytes - ordinary.storage.bytes == 68
        assert (
            low_schedule(bound, 2, prefix_scan="block", shared_capacity=201888)
            is not None
        )
        assert (
            low_schedule(bound, 2, prefix_scan="block", shared_capacity=201887) is None
        )
        assert low_schedule(bound, 2, shared_capacity=201887) is not None
        with pytest.raises(ValueError):
            replace(selected, block_prefix=FlatGroupedBlockPrefixPlan(256, 16))
        assert (
            low_schedule(bound, tma_rn=False, converter_warps=4, prefix_scan="block")
            is None
        )
    with cuda_binding(static=True, groups=257, rows=400, k=128, n=128) as bound:
        assert low_schedule(bound) is not None
        assert low_schedule(bound, prefix_scan="block") is None
        assert not bound.config_spec.cute_grouped_block_prefix_recipes


@pytest.mark.parametrize("value", [None, True, 1, "auto", "BLOCK", []])
def test_invalid_scan_choice_rejects_without_coercion(value):
    config = {CONVERSION_KEY: TMA_RN, K_KEY: 32, STAGES_KEY: 2, PREFIX_SCAN_KEY: value}
    kwargs = {
        "available_k": (32, 64),
        "rn_two_stage_ctas": (1, 2),
        "block_prefix_recipes": ((32, 2, 1),),
    }
    with pytest.raises(helion.exc.InvalidConfig, match=PREFIX_SCAN_KEY):
        normalize_grouped_rna_config(config.copy(), fix_invalid=False, **kwargs)
    normalize_grouped_rna_config(config, fix_invalid=True, **kwargs)
    assert config == {CONVERSION_KEY: TMA_RN, K_KEY: 32, STAGES_KEY: 2}


@pytest.mark.parametrize("family", [{}, {WARPS_KEY: 4}, {CONVERSION_KEY: WARP_RAW}])
def test_block_choice_cannot_silently_switch_or_disappear_in_other_families(family):
    kwargs = {"available_k": (32, 64), "warp_raw_available": True}
    old = family.copy()
    normalize_grouped_rna_config(old, fix_invalid=False, **kwargs)
    config = family | {PREFIX_SCAN_KEY: "block"}
    with pytest.raises(helion.exc.InvalidConfig, match=PREFIX_SCAN_KEY):
        normalize_grouped_rna_config(config.copy(), fix_invalid=False, **kwargs)
    normalize_grouped_rna_config(config, fix_invalid=True, **kwargs)
    assert config == old


def test_exact_effective_recipe_controls_admission_and_repair():
    kwargs = {
        "available_k": (32, 64),
        "rn_two_stage_ctas": (1, 2),
        "block_prefix_recipes": ((32, 6, 1), (32, 2, 1)),
    }
    for stages, ctas, accepted in ((0, 1, True), (2, 1, True), (2, 2, False)):
        config = {
            CONVERSION_KEY: TMA_RN,
            K_KEY: 32,
            STAGES_KEY: stages,
            RESIDENT_CTAS_KEY: ctas,
            PREFIX_SCAN_KEY: "block",
        }
        if accepted:
            normalize_grouped_rna_config(config, fix_invalid=False, **kwargs)
            assert config[PREFIX_SCAN_KEY] == "block"
        else:
            with pytest.raises(helion.exc.InvalidConfig, match=PREFIX_SCAN_KEY):
                normalize_grouped_rna_config(config.copy(), fix_invalid=False, **kwargs)
            normalize_grouped_rna_config(config, fix_invalid=True, **kwargs)
            assert config == {
                CONVERSION_KEY: TMA_RN,
                K_KEY: 32,
                STAGES_KEY: 2,
                RESIDENT_CTAS_KEY: 2,
            }


@pytest.mark.parametrize("bits,wrapped", [(32, False), (64, True)])
def test_nonoriginal_partial_groups_and_output_tail_emit_typed_block_scan(
    bits, wrapped
):
    with cuda_binding(
        static=True, bits=bits, groups=65, rows=901, k=64, n=132
    ) as bound:
        selected = low_schedule(bound, prefix_scan="block")
        assert selected is not None
        assert selected.provider.offset_bits == bits
        assert selected.block_prefix == FlatGroupedBlockPrefixPlan(65, 8)
        source = _source(bound, _block_config(bound, 1, wrapped))
        native = _native_source(source)
        assert (
            "build_flat_tile_prefix(offsets, tcgen05_prefix, tcgen05_prefix_chunk_summaries, 8, 65, 57664, 64, 64, 118932, 132, 132, 128, 128)"
            in native
        )
        assert "_helion_native_grid_z" in source


@pytest.mark.parametrize(
    "changes",
    [
        {"static": False},
        {"dot_precision": "ieee"},
        {"dtype": torch.float16},
        {"mode": "extra_math"},
        {"offsets_stride": 2},
        {"misalign": 1},
    ],
)
def test_block_request_rejects_unsupported_typed_binding(changes):
    with cuda_binding(
        **({"static": True, "groups": 65, "k": 128, "n": 128} | changes)
    ) as bound:
        assert not bound.config_spec.cute_grouped_block_prefix_recipes
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalized_config(_block_config(bound))


@pytest.mark.parametrize(
    "original,ctas,wrapped", [(True, 2, True), (False, 1, False), (False, 2, True)]
)
def test_ordinary_emission_preserves_caller_and_post_prefix_kernel(
    original, ctas, wrapped
):
    with cuda_binding(
        static=True, original=original, groups=256, rows=30856, k=128, n=128
    ) as bound:
        old_config = low_config(bound, ctas, wrapped)
        block_config = _block_config(bound, ctas, wrapped)
        old_public = _source(bound, old_config)
        assert (
            _source(
                bound,
                helion.Config.from_dict(old_config.config | {PREFIX_SCAN_KEY: "warp"}),
            )
            == old_public
        )
        public = _source(bound, block_config)
        old, new = _native_source(old_public), _native_source(public)
        fence = "    cute.arch.mbarrier_init_fence()\n    cute.arch.sync_threads()\n"
        old_kernel = next(
            n
            for n in ast.parse(old).body
            if isinstance(n, ast.FunctionDef) and n.name == "_helion_native_grouped_rna"
        )
        new_kernel = next(
            n
            for n in ast.parse(new).body
            if isinstance(n, ast.FunctionDef) and n.name == old_kernel.name
        )
        old_text, new_text = ast.unparse(old_kernel), ast.unparse(new_kernel)
        assert old_text.split(fence)[1] == new_text.split(fence)[1]
        assert "build_flat_tile_prefix_block as build_flat_tile_prefix" in new
        calls = [
            n
            for n in new_kernel.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "build_flat_tile_prefix"
        ]
        assert len(calls) == 1
        assert ast.literal_eval(calls[0].value.args[3]) == 8
        assert ast.literal_eval(calls[0].value.args[4]) == 256
        assert (
            new_text.index("wait_for_alloc()")
            < new_text.index("build_flat_tile_prefix(")
            < new_text.index(fence.strip())
        )
        assert "('tcgen05_prefix_chunk_summaries_ptr', 64, 16, 99856)" in new
        assert "'grouped_prefix_scan': ('block_checked_prefix_v1', 256, 8)" in new
        # The composed public entry allocates/reshapes and calls its native body
        # exactly as before; all changed source/plan identities live separately.
        name = "jagged_dense_bmm" if original else "flat_contract"
        functions = [
            next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            for source in (old_public, public)
        ]
        assert ast.dump(functions[0]) == ast.dump(functions[1])
        spec = bound.config_spec
        normalized = spec.normalized_config(block_config)
        generator = ConfigGeneration(spec)
        restored = generator.unflatten(generator.flatten(normalized))
        assert spec.normalized_config(restored) == with_flat_min_blocks_default(
            spec, normalized
        )
        assert (
            spec.normalized_config(
                helion.Config.from_dict(json.loads(json.dumps(normalized.config)))
            )
            == normalized
        )


@pytest.mark.parametrize("search_class", [PatternSearch, LFBOPatternSearch])
def test_old_seed_prefix_and_new_complete_choices_reach_initial_population(
    search_class,
):
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        spec = bound.config_spec
        seeds = CuteGroupedRnaHeuristic.get_seed_configs(
            bound.env, _host(bound).device_ir
        )
        assert len(seeds) == 33
        assert all(PREFIX_SCAN_KEY not in seed.config for seed in seeds[:25])
        assert seeds[24].get(CONVERSION_KEY) == WARP_RAW
        old_flag = spec.cute_grouped_block_prefix_seed_enabled
        spec.cute_grouped_block_prefix_seed_enabled = False
        try:
            assert (
                CuteGroupedRnaHeuristic.get_seed_configs(
                    bound.env, _host(bound).device_ir
                )
                == seeds[:25]
            )
        finally:
            spec.cute_grouped_block_prefix_seed_enabled = old_flag
        block = seeds[25:]
        assert {
            (
                s[K_KEY],
                s.get(STAGES_KEY, 0),
                s.get(RESIDENT_CTAS_KEY, 1),
                s.get(DESCRIPTOR_KEY, "dynamic"),
            )
            for s in block
        } == {
            (k, stages, ctas, descriptor)
            for k, stages, ctas in ((32, 0, 1), (64, 0, 1), (32, 2, 1), (32, 2, 2))
            for descriptor in ("dynamic", WRAPPED)
        }
        for seed in block:
            assert spec.normalized_config(seed)[PREFIX_SCAN_KEY] == "block"
        profile = get_effort_profile("full").lfbo_pattern_search
        assert profile is not None
        bound.settings.autotune_random_seed = 202609201
        search = search_class(
            bound,
            _runtime_arguments(bound),
            initial_population=profile.initial_population,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        population = [
            search.config_gen.unflatten(row)
            for row in checked_initial_population(search)
        ]
        assert all(
            with_flat_min_blocks_default(spec, spec.normalized_config(seed))
            in population
            for seed in block
        )


class _Collectives(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        kinds = {
            "cute.arch.shuffle_sync_up": "up",
            "cute.arch.warp_reduction_sum": "sum",
            "cute.arch.sync_threads": "barrier",
        }
        name = ast.unparse(node.func)
        if name not in kinds:
            return node
        if kinds[name] == "sum":
            assert (
                len(node.keywords) == 1 and node.keywords[0].arg == "threads_in_group"
            )
            assert ast.literal_eval(node.keywords[0].value) == 32
        return ast.copy_location(
            ast.Yield(
                ast.Tuple(elts=[ast.Constant(kinds[name]), *node.args], ctx=ast.Load())
            ),
            node,
        )


def _block_code(mutate=None):
    tree = ast.parse(Path(prefix_module.__file__).read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_flat_tile_prefix_block"
    )
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    if mutate is not None:
        fn = mutate(fn)
    fn = _Collectives().visit(fn)
    return compile(
        ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
        "production-block-prefix",
        "exec",
    )


class _Memory:
    def __init__(self, length, state, summaries):
        self.values = [None] * length
        self.state = state
        self.summaries = summaries

    def __getitem__(self, index):
        assert self.summaries and self.state.phase == 1
        assert 0 <= index < len(self.values) and self.values[index] is not None
        return self.values[index]

    def __setitem__(self, index, value):
        assert 0 <= index < len(self.values) and self.values[index] is None
        if self.summaries:
            assert self.state.phase == 0 and self.state.thread % 32 == 31
            assert index in (
                2 * (self.state.thread // 32),
                2 * (self.state.thread // 32) + 1,
            )
        else:
            assert self.state.phase == 1 and index == self.state.thread
        self.values[index] = value


def _execute_block(code, reload, offsets, groups, warps, sizes):
    state = SimpleNamespace(thread=0, phase=0)
    prefix, summaries = _Memory(groups, state, False), _Memory(2 * warps, state, True)
    ns = {
        "cutlass": SimpleNamespace(Int32=int, range_constexpr=range),
        "reload_flat_interval": reload,
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                lane_idx=lambda: state.thread % 32,
                warp_idx=lambda: state.thread // 32,
                make_warp_uniform=lambda x: x,
            )
        ),
    }
    exec(code, ns)
    tasks = [
        ns["build_flat_tile_prefix_block"](
            offsets, prefix, summaries, warps, groups, *sizes, 128, 128
        )
        for _ in range(warps * 32)
    ]
    replies = [None] * len(tasks)
    events = []
    while True:
        current, done = [], []
        for thread, (task, reply) in enumerate(zip(tasks, replies, strict=True)):
            state.thread = thread
            try:
                current.append(task.send(reply))
            except StopIteration:
                done.append(thread)
        if done:
            assert done == list(range(len(tasks))) and not current
            break
        assert len(current) == len(tasks) and len({row[0] for row in current}) == 1
        kind = current[0][0]
        events.append(kind)
        if kind == "barrier":
            assert state.phase == 0 and all(
                value is not None for value in summaries.values
            )
            state.phase = 1
            replies = [None] * len(tasks)
        elif kind == "up":
            assert state.phase == 0 and len({row[2] for row in current}) == 1
            delta = current[0][2]
            replies = [
                current[t - delta if t % 32 >= delta else t][1]
                for t in range(len(tasks))
            ]
        else:
            assert kind == "sum" and state.phase == 1
            replies = [
                sum(row[1] for row in current[(t // 32) * 32 : (t // 32 + 1) * 32])
                for t in range(len(tasks))
            ]
    assert events == ["up"] * 10 + ["barrier", "sum", "sum"]
    assert all(value is not None for value in prefix.values)
    return prefix.values


@pytest.mark.parametrize(
    "warps,groups",
    [(1, 1), (1, 32), (2, 33), (3, 65), (8, 127), (8, 256), (16, 511), (32, 1024)],
)
def test_actual_helper_all_lanes_partial_chunks_and_invalid_suffix(warps, groups):
    code = _block_code()
    values = [(i * 17 + 3) % 242 for i in range(groups)]
    for first_invalid in sorted(
        {0, 1, 31, 32, groups - 1, groups} & set(range(groups + 1))
    ):

        def reload(offsets, group, *args, first_invalid=first_invalid):
            valid = group != first_invalid
            value = values[group] if valid else 0
            return 0, 0, 0 if value == 0 else 1 + 128 * (value - 1), valid

        got = _execute_block(
            code, reload, None, groups, warps, (4000000, 128, 128, 4000000, 128, 128)
        )
        total, expected = 0, []
        for i, value in enumerate(values):
            total += value
            expected.append(total if i < first_invalid else -1)
        assert got == expected


@pytest.mark.parametrize("bits", [32, 64])
def test_actual_typed_intervals_match_scalar_for_overflow_empty_and_every_invalid_position(
    bits,
):
    scalar, Offsets = _scalar_prefix_functions(bits)
    code = _block_code()
    sizes = (3949568, 128, 128, 3949568, 128, 128)
    cases = [[i * 120 for i in range(257)], [0] * 257, list(range(257))]
    edge = [
        -(1 << (bits - 1)),
        -1,
        0,
        1,
        127,
        128,
        30856,
        (1 << (bits - 7)),
        (1 << (bits - 1)) - 1,
    ]
    cases.extend([start, end, *([end] * 255)] for start, end in product(edge, repeat=2))
    cases.extend([*([0] * first), -1, *([1] * (256 - first))] for first in range(256))
    for values in cases:
        offsets = Offsets(values)
        expected = [None] * 256
        scalar["build_flat_tile_prefix"](offsets, expected, 256, *sizes, 128, 128)
        assert (
            _execute_block(code, scalar["reload_flat_interval"], offsets, 256, 8, sizes)
            == expected
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_barrier", "include_own_chunk", "omit_invalid_carry", "narrow_reduction"],
)
def test_cooperative_witness_rejects_broken_publication_and_carries(mutation):
    class Mutate(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                mutation == "missing_barrier"
                and ast.unparse(node) == "cute.arch.sync_threads()"
            ):
                return None
            return self.generic_visit(node)

        def visit_Compare(self, node):
            if mutation == "include_own_chunk" and ast.unparse(node) == "lane < warp":
                node.ops = [ast.LtE()]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if (
                mutation == "narrow_reduction"
                and ast.unparse(node.func) == "cute.arch.warp_reduction_sum"
            ):
                node.keywords[0].value = ast.Constant(8)
            return self.generic_visit(node)

        def visit_BinOp(self, node):
            if (
                mutation == "omit_invalid_carry"
                and ast.unparse(node) == "invalid + prior_invalid"
            ):
                return ast.Name(id="invalid", ctx=ast.Load())
            return self.generic_visit(node)

    with pytest.raises(AssertionError):
        code = _block_code(Mutate().visit)
        for first_invalid in (256, 0):

            def reload(offsets, group, *args, first_invalid=first_invalid):
                return 0, 0, int(group != first_invalid), group != first_invalid

            got = _execute_block(
                code, reload, None, 256, 8, (4000000, 128, 128, 4000000, 128, 128)
            )
            expected = list(range(1, 257)) if first_invalid == 256 else [-1] * 256
            assert got == expected
