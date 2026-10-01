"""Seed policy on declared CPU metadata; no binding or generated-code fixture."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import patch

import sympy
import torch

from test._cute_binding import _mock_cuda_unavailable

from helion._compiler.autotuner_heuristics.register_chain import (
    CuteRegisterChainHeuristic,
)
from helion._compiler.backend import CuteBackend
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteLaneLayoutSpec
from helion.autotuner.config_spec import CuteVectorWidthSpec
from helion.autotuner.config_spec import DotAxes
from helion.autotuner.config_spec import DotAxisKind
from helion.autotuner.config_spec import DotSite
from helion.autotuner.config_spec import KernelGridFact
from helion.autotuner.config_spec import KernelMatmulFact
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ResolvedMatmulFact
from helion.autotuner.config_spec import RootGridFact

if TYPE_CHECKING:
    import helion
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR

_MODE = True


def _environment(
    *,
    capability: tuple[int, int] | None = (9, 0),
    dtype: torch.dtype = torch.bfloat16,
    domains: tuple[sympy.Expr, sympy.Expr] = (sympy.Integer(256), sympy.Integer(128)),
    fixed: tuple[bool, bool] = (False, False),
    shared_root: bool = True,
) -> tuple[CompileEnvironment, DeviceIR]:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=capability,
        device=torch.device("cpu"),
        num_sm=1,
    )
    pairs = ((0, 1), (0, 1) if shared_root else (4, 5))
    sizes = {0: 96, 1: 80, 2: 256, 3: 128}
    if not shared_root:
        sizes.update({4: 96, 5: 80})
    for k, domain in zip((2, 3), domains, strict=True):
        if domain.is_Integer:
            sizes[k] = max(1, int(domain))
    for block_id in reversed(sizes):
        if block_id not in (2, 3) or not fixed[block_id - 2]:
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=sizes[block_id])
            )
    for block_id in sizes:
        spec.num_threads.append(
            NumThreadsSpec(block_id=block_id, size_hint=sizes[block_id])
        )
    for block_id in reversed(sizes):
        spec.cute_vector_widths.append(
            CuteVectorWidthSpec(block_id=block_id, size_hint=sizes[block_id])
        )
        spec.cute_lane_layouts.append(CuteLaneLayoutSpec(block_id=block_id))
    resolved = []
    for index, ((m, n), domain) in enumerate(zip(pairs, domains, strict=True)):
        extent = int(domain) if domain.is_Integer else None
        fact = MatmulFact(2, 2, m, n, index + 2, 96, 80, extent, dtype, dtype)
        spec.matmul_facts.append(fact)
        resolved.append(
            ResolvedMatmulFact(
                fact,
                DotAxes(
                    DotAxisKind.TUNABLE_TILED,
                    DotAxisKind.TUNABLE_TILED,
                    DotAxisKind.FIXED_FULL_EXTENT
                    if fixed[index]
                    else DotAxisKind.TUNABLE_TILED,
                    96,
                    80,
                    extent,
                ),
                DotSite(index + 10, updates_carry=index == 1),
            )
        )
    roots = (RootGridFact(0, pairs[0]),)
    if not shared_root:
        roots += (RootGridFact(1, pairs[1]),)
    spec.kernel_grid_fact = KernelGridFact(
        roots, ((10, 0), (11, 0 if shared_root else 1))
    )
    spec.kernel_matmul_fact = KernelMatmulFact(
        matmuls=tuple(resolved),
        knob_users=(),
        sequential_loop_trips=1,
        live_dot_outputs=(),
        live_promoted_lhs=(),
        live_tile_steps=(),
        pipelined_regions=(),
        resident_regions=(),
        attribution_complete=True,
    )
    spec.cute_tile_loop_paths = tuple(
        (pair, (k,)) for pair, k in zip(pairs, (2, 3), strict=True)
    )
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes={
            block_id: SimpleNamespace(
                numel=domains[block_id - 2]
                if block_id in (2, 3)
                else sympy.Integer(size),
                reduction=block_id in (2, 3),
            )
            for block_id, size in sizes.items()
        },
    )
    device_ir = SimpleNamespace(
        graphs=(), grid_block_ids=[list(root.block_ids) for root in roots]
    )
    return cast("CompileEnvironment", env), cast("DeviceIR", device_ir)


def _chain_seeds(env: CompileEnvironment, device_ir: DeviceIR) -> list[helion.Config]:
    return [
        seed
        for seed in CuteRegisterChainHeuristic.get_seed_configs(env, device_ir)
        if seed.get("cute_register_chain") == _MODE
    ]


class TestCuteRegisterChainSeeds(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(_mock_cuda_unavailable())
        for name in (
            "_lazy_init",
            "current_device",
            "get_device_capability",
            "get_device_properties",
        ):
            stack.enter_context(
                patch.object(
                    torch.cuda, name, side_effect=AssertionError("GPU forbidden")
                )
            )
        # The old FX seed predicates are accepted dependencies. Only the new
        # policy consumes these declared hints; no FX graph or tensor is built.
        self.seed_hint = stack.enter_context(
            patch(
                "helion._compiler.cute.collective_matmul.has_collective_native_seed_candidate",
                return_value=True,
            )
        )

    def test_chain_has_no_default_promotion(self) -> None:
        env, device_ir = _environment()
        seeds = CuteRegisterChainHeuristic.get_seed_configs(env, device_ir)
        self.assertTrue(seeds)
        self.assertEqual(
            CuteRegisterChainHeuristic.get_seed_config(env, device_ir), seeds[0]
        )
        self.assertFalse(CuteRegisterChainHeuristic.should_promote(env))

    def test_bounded_geometry_order_and_scalar_layout(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                env, device_ir = _environment(dtype=dtype)
                spec = env.config_spec
                seeds = _chain_seeds(env, device_ir)
                self.assertEqual(len(seeds), 24)
                self.assertEqual(len(set(seeds)), 24)
                geometries = []
                for seed in seeds:
                    bm, bn, bk = (
                        spec.block_sizes.config_get(seed.block_sizes, block_id)
                        for block_id in (0, 1, 2)
                    )
                    geometries.append((bm, bn, bk))
                    self.assertEqual(
                        spec.block_sizes.config_get(seed.block_sizes, 3), bk
                    )
                    self.assertEqual(
                        spec.num_threads.config_get(seed.num_threads, 0), 64 // bn
                    )
                    self.assertEqual(
                        spec.num_threads.config_get(seed.num_threads, 1), bn
                    )
                    self.assertEqual(
                        [
                            spec.num_threads.config_get(seed.num_threads, k)
                            for k in (2, 3)
                        ],
                        [1, 1],
                    )
                    self.assertEqual(seed["cute_vector_widths"], [1] * 4)
                    self.assertEqual(seed["cute_lane_layouts"], ["blocked"] * 4)
                    self.assertTrue(seed["cute_register_chain"])
                    self.assertFalse(
                        any(
                            key in seed.config
                            for key in (
                                "cute_collective_stages",
                                "cute_collective_recipe",
                                "cute_collective_epilogue",
                                "cute_collective_tmem_a",
                                "cute_collective_native_seeded",
                            )
                        )
                    )
                self.assertEqual(
                    geometries,
                    [
                        (m, n, k)
                        for m in (16, 32, 64)
                        for n in (16, 32)
                        for k in (64, 32, 16, 128)
                    ],
                )

    def test_arch_dtype_and_zero_seed_hint_do_not_add_chain(self) -> None:
        for options in (
            {"capability": None},
            {"capability": (7, 5)},
            {"dtype": torch.float32},
        ):
            with self.subTest(options=options):
                env, device_ir = _environment(**options)
                self.assertEqual(_chain_seeds(env, device_ir), [])
        self.seed_hint.return_value = False
        env, device_ir = _environment()
        self.assertEqual(_chain_seeds(env, device_ir), [])

    def test_distinct_roots_or_non_grid_m_do_not_add_chain(self) -> None:
        env, device_ir = _environment(shared_root=False)
        self.assertEqual(_chain_seeds(env, device_ir), [])
        env, device_ir = _environment()
        env.config_spec.kernel_grid_fact = KernelGridFact(
            (RootGridFact(0, (1,)),), ((10, 0), (11, 0))
        )
        device_ir.grid_block_ids = [[1]]
        self.assertEqual(_chain_seeds(env, device_ir), [])

    def test_single_contraction_and_packed_axis_hints_decline(self) -> None:
        env, device_ir = _environment()
        axes = CuteRegisterChainHeuristic._axes(env, device_ir)
        first = env.config_spec.matmul_facts[0]
        with patch(
            "helion._compiler.autotuner_heuristics.register_chain.cute_matmul_root_placements",
            return_value=((first, (0, 1)),),
        ):
            self.assertEqual(
                CuteRegisterChainHeuristic._register_chain_seeds(
                    env, device_ir, axes, True
                ),
                [],
            )
        with patch.object(
            CuteRegisterChainHeuristic,
            "_axes",
            return_value=tuple((*axis[:3], 2) for axis in axes),
        ):
            self.assertEqual(_chain_seeds(env, device_ir), [])

    def test_static_k_requires_complete_packets(self) -> None:
        env, device_ir = _environment(domains=(sympy.Integer(160), sympy.Integer(96)))
        seeds = _chain_seeds(env, device_ir)
        self.assertEqual(len(seeds), 12)
        self.assertEqual(
            {
                env.config_spec.block_sizes.config_get(seed.block_sizes, 2)
                for seed in seeds
            },
            {16, 32},
        )

    def test_symbolic_k_uses_fragment_hint_without_claiming_packet_count(self) -> None:
        env, device_ir = _environment(
            domains=(
                sympy.Integer(32),
                sympy.Symbol("domain", positive=True, integer=True),
            ),
            fixed=(True, False),
        )
        k_spec = next(
            item for item in env.config_spec.block_sizes if item.block_id == 3
        )
        k_spec.update_max(64)
        seeds = _chain_seeds(env, device_ir)
        self.assertEqual(len(seeds), 18)
        self.assertEqual(
            env.config_spec.block_sizes.config_get(seeds[0].block_sizes, 0), 16
        )
        self.assertEqual(
            env.config_spec.block_sizes.config_get(seeds[0].block_sizes, 1), 16
        )
        self.assertEqual(
            env.config_spec.block_sizes.config_get(seeds[0].block_sizes, 3), 64
        )
        self.assertEqual(len(seeds[0].block_sizes), 3)

    def test_fixed_k_has_no_config_slot_or_duplicate_schedule(self) -> None:
        env, device_ir = _environment(
            domains=(sympy.Integer(16), sympy.Integer(128)), fixed=(True, True)
        )
        seeds = _chain_seeds(env, device_ir)
        self.assertEqual(len(seeds), 6)
        self.assertEqual(len(set(seeds)), 6)
        self.assertTrue(all(len(seed.block_sizes) == 2 for seed in seeds))
        for extent in (8, 48):
            with self.subTest(extent=extent):
                env, device_ir = _environment(
                    domains=(sympy.Integer(extent), sympy.Integer(128)),
                    fixed=(True, True),
                )
                self.assertEqual(_chain_seeds(env, device_ir), [])

    def test_block_bounds_and_proven_bounds_are_respected(self) -> None:
        env, device_ir = _environment()
        spec = env.config_spec
        for item in spec.block_sizes:
            if item.block_id in (0, 1):
                item.update_max(16)
        spec.enable_cute_proven_bounds()
        seeds = _chain_seeds(env, device_ir)
        self.assertEqual(len(seeds), 4)
        self.assertTrue(all("cute_proven_bounds" not in seed.config for seed in seeds))
        for item in spec.block_sizes:
            if item.block_id == 1:
                item.update_max(8)
        self.assertEqual(_chain_seeds(env, device_ir), [])

    def test_seed_lists_are_independent_and_transfer_to_search(self) -> None:
        env, device_ir = _environment()
        seeds = CuteRegisterChainHeuristic.get_seed_configs(env, device_ir)
        original = deepcopy([seed.config for seed in seeds])
        index = next(
            i
            for i, seed in enumerate(seeds)
            if seed.get("cute_register_chain") == _MODE
        )
        for field in (
            "block_sizes",
            "num_threads",
            "cute_vector_widths",
            "cute_lane_layouts",
        ):
            values = seeds[index].config[field]
            first = values[0]
            values[0] = "mutation witness"
            self.assertTrue(
                all(
                    seed.config == original[i]
                    for i, seed in enumerate(seeds)
                    if i != index
                )
            )
            values[0] = first
        spec = env.config_spec
        default = spec.default_config()
        spec.compiler_seed_configs = seeds
        generation = ConfigGeneration(spec)
        restored = generation.unflatten(generation.flatten(seeds[index]))
        self.assertEqual(restored["cute_register_chain"], _MODE)
        self.assertEqual(restored.block_sizes, seeds[index].block_sizes)
        self.assertEqual(restored.num_threads, seeds[index].num_threads)
        self.assertEqual(
            restored["cute_vector_widths"], seeds[index]["cute_vector_widths"]
        )
        self.assertEqual(spec.default_config(), default)
        self.assertEqual([seed.config for seed in seeds], original)


if __name__ == "__main__":
    unittest.main()
