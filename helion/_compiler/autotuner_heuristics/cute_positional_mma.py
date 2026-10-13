"""Coarse positional-MMA search facts; the emitter proves resolved geometry."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.config_spec import CUTE_POSITIONAL_MMA_KEY
from ...autotuner.config_spec import CUTE_POSITIONAL_MMA_WARPS_M_KEY
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


def _has_candidate(device_ir: DeviceIR) -> bool:
    from ..cute.mma_support import cute_fp32_dot_uses_tf32
    from ..cute.positional_domains import contraction_of

    for graph in device_ir.graphs:
        for node in graph.graph.nodes:
            contraction = contraction_of(node)
            if contraction is None:
                continue
            lhs, rhs, result = (
                value.meta["val"] for value in (contraction.lhs, contraction.rhs, node)
            )
            if (
                lhs.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or rhs.dtype != lhs.dtype
                or result.dtype != torch.float32
                or lhs.dtype == torch.float32
                and not cute_fp32_dot_uses_tf32()
            ):
                continue
            # Symbolic leading tile dimensions can resolve to one. Static
            # non-singleton batch dimensions cannot use this warp-MMA schedule.
            if all(
                value.ndim >= 2
                and all(
                    not isinstance(size, int) or size == 1 for size in value.shape[:-2]
                )
                for value in (lhs, rhs, result)
            ):
                return True
    return False


class CutePositionalMmaHeuristic(AutotunerHeuristic):
    name = "cute_positional_mma"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.positional_root import has_positional_contraction_candidate

        spec = env.config_spec
        host = device_ir.host_function
        capability = spec.target_device_capability
        if (
            host is not None
            and capability is not None
            and capability[0] >= 8
            and not spec.cute_chained_matmul_search_enabled
            and not spec.cute_tcgen05_search_enabled
            and not spec.cute_flash_search_enabled
            and not spec.cute_flash_bwd_search_enabled
            and spec.cute_chunk_prepare_schedule is None
            and "flat" in spec.allowed_pid_types
            and not env.settings.autotune_force_persistent
        ):
            env_context = nullcontext() if env.has_current() else env
            with env_context, host:
                spec.cute_positional_mma_search_enabled = bool(
                    has_positional_contraction_candidate(device_ir.graphs)
                    and _has_candidate(device_ir)
                )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        # Other whole-root families can register after positional facts.
        # Resolve exclusivity again when the registry requests their seeds.
        return (
            spec.cute_positional_mma_search_enabled
            and not spec.cute_chained_matmul_search_enabled
            and not spec.cute_tcgen05_search_enabled
            and not spec.cute_flash_search_enabled
            and not spec.cute_flash_bwd_search_enabled
            and spec.cute_chunk_prepare_schedule is None
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not cls.is_eligible(env, device_ir):
            return None
        config = env.config_spec.default_config()
        config.config.update(pid_type="flat", **{CUTE_POSITIONAL_MMA_KEY: True})
        return config

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        default = cls.get_seed_config(env, device_ir)
        if default is None:
            return None
        row_warp = deepcopy(default)
        row_warp.config[CUTE_POSITIONAL_MMA_WARPS_M_KEY] = 1
        return [default, row_warp]
