"""``cute_register_subblocks``: register-resident static sub-block programs.

``register_facts`` runs the config-free matcher (``match_register_subblocks``)
on the device IR; where it matches, the family switch and its knobs enter
the search surface with the facts (tile choices, independent groups, the
register estimate of every knob corner) ``normalize`` validates against.
The seeds are the measured winners of the backward anchored gram (B200,
2048 x 64 x 128: 16-channel tiles with one item of prefetch on one warp,
168 us against the fragment route's 500 and Triton's 338; the 32-channel
two-warp Triton-like structure; the two-warp 16-channel variant) where the
facts admit them, then the legal corners by register estimate.  The family
becomes the autotune-off default when the root's flattened dots total at
least ``SEED_MACS`` multiply-adds, on the GPU generations it was validated on.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ..cute.cute_register_subblocks import COLSIDE_KEY
from ..cute.cute_register_subblocks import KEY
from ..cute.cute_register_subblocks import MASK_KEY
from ..cute.cute_register_subblocks import MODE_ON
from ..cute.cute_register_subblocks import PREFETCH_KEY
from ..cute.cute_register_subblocks import ROLL_KEY
from ..cute.cute_register_subblocks import WARPS_KEY
from ..cute.cute_register_subblocks import match_register_subblocks
from ..cute.cute_register_subblocks import register_subblock_facts
from ..cute.cute_register_subblocks import register_subblocks_seed_knobs
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..cute.cute_register_subblocks import RegisterSubblockFacts
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteRegisterSubblocksHeuristic(AutotunerHeuristic):
    """Admit and seed the register sub-block family."""

    name = KEY
    backend = "cute"
    promote_seed_to_default = True
    PROMOTE_TARGETS = (("cuda", "sm100"), ("cuda", "sm103"))
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @classmethod
    def _facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> RegisterSubblockFacts | None:
        host_function = device_ir.host_function
        if host_function is None or not isinstance(env, CompileEnvironment):
            # The matcher reads shapes, strides and symbol origins from the
            # environment; a stand-in environment (the heuristic registry's
            # unit tests) has nothing to match against.
            return None
        with contextlib.ExitStack() as stack:
            if not CompileEnvironment.has_current():
                stack.enter_context(env)
            stack.enter_context(host_function)
            program, _reason = match_register_subblocks(device_ir.graphs)
            if program is None:
                return None
            return register_subblock_facts(program)

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        facts = cls._facts(env, device_ir)
        if facts is None or all(
            reason is not None for reason in facts.failures.values()
        ):
            # (a matched program no knob corner renders, e.g. one reading a
            # stored window back in another layout, stays off the surface)
            return frozenset()
        env.config_spec.enable_cute_register_subblocks_search(facts)
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_register_subblocks is not None

    @classmethod
    def should_promote(cls, env: CompileEnvironment) -> bool:
        facts = env.config_spec.cute_register_subblocks_facts
        return facts is not None and facts.seed_worthy and super().should_promote(env)

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        spec = env.config_spec
        facts = spec.cute_register_subblocks_facts
        if facts is None or spec.cute_register_subblocks is None:
            return None
        base = spec.default_config().config
        seeds: list[Config] = []
        for knobs in register_subblocks_seed_knobs(facts):
            config = dict(base)
            config[KEY] = MODE_ON
            for key in (WARPS_KEY, PREFETCH_KEY, MASK_KEY, COLSIDE_KEY, ROLL_KEY):
                config[key] = knobs[key]
            if facts.tile_block_id is not None:
                block_sizes = list(config["block_sizes"])  # pyrefly: ignore[bad-argument-type]
                block_sizes[spec.block_sizes.block_id_to_index(facts.tile_block_id)] = (
                    knobs["bk"]
                )
                config["block_sizes"] = block_sizes
            seeds.append(Config.from_dict(config))
        return seeds or None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None
