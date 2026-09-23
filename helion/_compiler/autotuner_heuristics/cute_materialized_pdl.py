"""Bounded serial/PDL coverage on ordinary materialized pipeline carriers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import cast

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ..cute.tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from .cute_materialized_operand import CuteMaterializedOperandHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def register_materialized_pdl_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    spec = env.config_spec
    state = spec._cute_tcgen05_config
    if state.materialized_operand_pdl_roots is None:
        return
    indices = state._matmul_block_indices()
    assert indices is not None
    mi, ni, ki = indices
    search_fields = spec._flat_fields()
    carriers: dict[int, list[Config]] = {2: [], 1: []}
    # Compose the existing native geometry and pointwise-vector layout policy,
    # independently of the automatic-heuristics toggle.
    for seed in CuteMaterializedOperandHeuristic.get_seed_configs(env, device_ir):
        if not state.materialized_operand_pdl_supported(seed.config):
            continue
        try:
            carrier = deepcopy(seed)
            spec.normalize(carrier)
        except InvalidConfig:
            continue
        # Ordinary paired seeds explicitly disable the separate direct-entry
        # path. Omit only those neutral payloads when its search fields are
        # absent, so strict coverage transfer does not lose supplied settings.
        for key in (
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
        ):
            if key not in search_fields and carrier.config.get(key) is False:
                carrier.config.pop(key)
        carriers[cast("int", carrier.config["tcgen05_acc_stages"])].append(carrier)
    witnesses = []
    for choices in carriers.values():
        if not choices:
            continue
        # Smallest output tile, widest K packet, then the capacity-priced deep
        # ring. These are compiler geometry preferences, not workload identities.
        carrier = min(
            choices,
            key=lambda c: (
                c.block_sizes[mi],
                c.block_sizes[ni],
                -c.block_sizes[ki],
                -cast("int", c.config["tcgen05_ab_stages"]),
            ),
        )
        witnesses.extend(CoverageWitness(carrier, value) for value in (False, True))
    if witnesses:
        spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.materialized_operand_pdl",
                version=1,
                key="tcgen05_materialized_pdl",
                domain=(False, True),
                legacy=False,
                witnesses=tuple(witnesses),
            )
        )
