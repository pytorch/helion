"""An additive witness for exact coarse-rank selection."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def register_topk_coarse_keys_coverage(env: CompileEnvironment) -> None:
    spec = env.config_spec
    if not spec.cute_topk_coarse_keys_available:
        return
    previous = spec.create_config_generation()
    try:
        _, default = previous.canonicalize_flat(previous.default_flat())
        carrier = next(
            (
                seed
                for seed in spec.compiler_seed_configs
                if seed.config.get("cute_topk_selection_layout") == "distributed"
            ),
            Config.from_dict(
                deepcopy(default.config) | {"cute_topk_selection_layout": "distributed"}
            ),
        )
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_topk_coarse_keys_search_enabled = True
    spec.create_config_generation().strict_config_pair(
        Config.from_dict(deepcopy(carrier.config) | {"cute_topk_coarse_keys": True})
    )
    # Deferred registration leaves all previous populations and RNG draws intact.
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.topk_coarse_keys",
            version=1,
            key="cute_topk_coarse_keys",
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )


def register_topk_key_recovery_coverage(env: CompileEnvironment) -> None:
    """Append packed recovery after every preexisting population and witness."""
    spec = env.config_spec
    group = next(
        (g for g in spec.compiler_coverage_groups if g.key == "cute_topk_coarse_keys"),
        None,
    )
    if group is None:
        return
    carrier = group.witnesses[0].carrier
    carrier.config["cute_topk_coarse_keys"] = True
    spec.cute_topk_key_recovery_search_enabled = True
    spec.create_config_generation().strict_config_pair(
        Config.from_dict(
            deepcopy(carrier.config) | {"cute_topk_key_recovery": "packed"}
        )
    )
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.topk_key_recovery",
            version=1,
            key="cute_topk_key_recovery",
            domain=("direct", "packed"),
            legacy="direct",
            witnesses=(CoverageWitness(carrier, "packed"),),
            dependencies=(CoverageDependency(group.mechanism, group.key, True),),
            deferred=True,
        )
    )
