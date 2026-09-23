"""Compiler-owned coverage seeds for an unpacked same-cell register chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...runtime.config import Config
from ..cute.matmul_utils import cute_matmul_root_placements
from .common import dedupe_configs
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteRegisterChainHeuristic(AutotunerHeuristic):
    name = "cute_register_chain"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        # Stride/layout proof is required even for a manually selected config.
        return (
            cls.CACHE_SPECIALIZATION_FACTS if cls._axes(env, device_ir) else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls.get_seed_configs(env, device_ir))

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        from ..cute.collective_matmul import has_collective_native_seed_candidate

        spec = env.config_spec
        if spec.cute_tcgen05_search_enabled or spec.cute_flash_search_enabled:
            return []
        axes = cls._axes(env, device_ir)
        if not axes:
            return []
        axis_ids = {block_id for row in axes for block_id in row[:3]}
        if any(
            item.max_size != 1
            for item in spec.block_sizes
            if len(item.block_ids) == 1 and item.block_id not in axis_ids
        ):
            return []
        return cls._register_chain_seeds(
            env, device_ir, axes, has_collective_native_seed_candidate(device_ir.graphs)
        )

    @staticmethod
    def _axes(
        env: CompileEnvironment, device_ir: DeviceIR
    ) -> tuple[tuple[int, int, int, int], ...]:
        placements = cute_matmul_root_placements(env, device_ir)
        # The late planner requires at least two distinct contractions. A
        # single matmul cannot use this path, including a manually set knob,
        # and must retain its ordinary dynamic-shape binding cache.
        if len(placements) < 2:
            return ()
        axes = []
        occupied: dict[int, tuple[int, ...]] = {}
        roots: dict[tuple[int, ...], tuple[int, int]] = {}
        factors: dict[int, int] = {}
        block_specs = {
            block_id: item
            for item in env.config_spec.block_sizes
            for block_id in item.block_ids
        }
        for fact, root in placements:
            m, n, k = fact.m_block_id, fact.n_block_id, fact.k_block_id
            factor = 1
            if (
                m is None
                or n is None
                or k is None
                or len({m, n, k}) != 3
                or fact.lhs_ndim not in (2, 3)
                or fact.rhs_ndim != fact.lhs_ndim
                or fact.lhs_dtype not in (torch.float16, torch.bfloat16)
                or fact.rhs_dtype != fact.lhs_dtype
                or fact.lhs_ndim == 3
                and not all(
                    (item := block_specs.get(block_id)) is not None
                    and item.max_size == 1
                    for block_id in root
                    if block_id not in (m, n)
                )
                or n not in root
                or k in root
                or root in roots
                and roots[root] != (m, n)
                or any(
                    block_id in occupied and occupied[block_id] != root
                    for block_id in (m, n, k)
                )
                or k in factors
                and factors[k] != factor
            ):
                return ()
            if (m, n, k, factor) not in axes:
                axes.append((m, n, k, factor))
            roots[root] = (m, n)
            occupied.update(dict.fromkeys((m, n, k), root))
            factors[k] = factor
        return tuple(axes)

    @staticmethod
    def _register_chain_seeds(
        env: CompileEnvironment,
        device_ir: DeviceIR,
        axes: tuple[tuple[int, int, int, int], ...],
        seeded: bool,
    ) -> list[Config]:
        """Seed a shared-root half/BF16 chain after the existing schedules.

        Symbolic K domains use the existing fragment bounds only as search
        hints. The late planner must prove a constant positive packet count,
        the complete original masked load domain, and adjacent seed frontiers.
        """
        spec = env.config_spec
        if (
            not seeded
            or spec.target_device_capability is None
            or spec.target_device_capability[0] < 8
            or any(factor != 1 for _m, _n, _k, factor in axes)
        ):
            return []
        placements = cute_matmul_root_placements(env, device_ir)
        if len(placements) < 2:
            return []
        m, n, _k, _factor = axes[0]
        root = placements[0][1]
        if (
            m not in root
            or n not in root
            or any(
                placed_root != root
                or fact.m_block_id != m
                or fact.n_block_id != n
                or fact.lhs_dtype not in (torch.float16, torch.bfloat16)
                for fact, placed_root in placements
            )
        ):
            return []
        blocks = {
            item.block_id: item for item in spec.block_sizes if len(item.block_ids) == 1
        }
        axis_ids = {block_id for row in axes for block_id in row[:3]}
        singleton_axes = set(blocks) - axis_ids
        fragments = {
            block_id: item._fragment(spec) for block_id, item in blocks.items()
        }
        k_ids = {k for _m, _n, k, _factor in axes}
        domains = {k: env.block_sizes[k].numel for k in k_ids}
        static_domains = {
            k: int(domain) for k, domain in domains.items() if domain.is_Integer
        }
        if any(extent <= 0 for extent in static_domains.values()):
            return []
        fixed_k = k_ids - set(blocks)
        if any(static_domains.get(k) not in (16, 32, 64, 128) for k in fixed_k):
            return []
        result = []
        for bm in (16, 32, 64):
            for bn in (16, 32):
                for bk in (64, 32, 16, 128):
                    if any(
                        extent % bk
                        for k, extent in static_domains.items()
                        if k not in fixed_k
                    ):
                        continue
                    values = {m: bm, n: bn}
                    values.update(dict.fromkeys(k_ids - fixed_k, bk))
                    values.update(dict.fromkeys(singleton_axes, 1))
                    if not all(
                        fragments[block_id].low <= value <= fragments[block_id].high
                        for block_id, value in values.items()
                    ):
                        continue
                    threads = {m: 64 // bn, n: bn}
                    threads.update(dict.fromkeys(k_ids, 1))
                    threads.update(dict.fromkeys(singleton_axes, 0))
                    seed = Config.from_dict(
                        {
                            "block_sizes": _seq_config_list(spec.block_sizes, values),
                            "num_threads": _seq_config_list(spec.num_threads, threads),
                            "cute_vector_widths": _seq_config_list(
                                spec.cute_vector_widths,
                                dict.fromkeys(axis_ids | singleton_axes, 1),
                            ),
                            "cute_lane_layouts": _seq_config_list(
                                spec.cute_lane_layouts,
                                dict.fromkeys(axis_ids | singleton_axes, "blocked"),
                            ),
                            "cute_register_chain": True,
                        }
                    )
                    result.append(seed)
        return dedupe_configs(result)
