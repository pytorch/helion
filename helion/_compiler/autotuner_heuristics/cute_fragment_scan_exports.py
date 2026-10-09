"""Default-off, append-only coverage for completed short-scan register exports."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import cast

import sympy

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_warp_scan import active_warp_scan_roots
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    import torch

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact
KEY = "cute_fragment_scan_exports"


def validate_scan_exports(
    env: CompileEnvironment, device_ir: DeviceIR, config: Config
) -> None:
    from ..cute.completed_scan import completed_scan_plans
    from ..cute.fragment_storage import configured_fragment_expr

    def shape(sizes: object) -> tuple[int, ...]:
        result = []
        for size in cast("tuple[int | torch.SymInt, ...]", sizes):
            value = env.specialize_expr(
                cast(
                    "sympy.Expr",
                    configured_fragment_expr(
                        env,
                        sympy.sympify(size),
                        lambda bid: env.block_sizes[bid].from_config(config),
                    ),
                )
            )
            if not value.is_number:
                raise InvalidConfig(
                    "completed scan export extent is not statically configured"
                )
            result.append(int(value))
        return tuple(result)

    for root, graphs in fragment_root_regions(device_ir):
        if root in active_warp_scan_roots(env, config) and completed_scan_plans(
            next(g.graph for g in graphs if g.graph_id == root), shape
        ):
            return
    raise InvalidConfig(
        "completed scan exports require a short one-row warp scan and closed terminal integer extrema"
    )


class CuteFragmentScanExportsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.completed_scan import completed_scan_plans

        env.config_spec.cute_fragment_scan_exports_root_ids = frozenset(
            root
            for root, graphs in fragment_root_regions(device_ir)
            if completed_scan_plans(next(g.graph for g in graphs if g.graph_id == root))
        )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_scan_exports_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_scan_exports_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_scan_exports_root_ids:
        return
    scan_group = next(
        (
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_warp_scan"
        ),
        None,
    )
    if scan_group is None:
        return
    carrier = Config.from_dict(
        deepcopy(scan_group.witnesses[0].carrier.config) | {scan_group.key: True}
    )
    from ..cute.completed_scan import completed_scan_plans

    # The mechanism needs one complete row per CTA. Derive only the leading
    # tile coordinates of a proved scan; do not choose a problem name/shape or
    # silently repair a submitted configuration.
    carriers = [carrier]
    host = device_ir.host_function
    assert host is not None
    with host:
        for root, graphs in fragment_root_regions(device_ir):
            for plan in completed_scan_plans(
                next(g.graph for g in graphs if g.graph_id == root)
            ):
                candidate = Config.from_dict(deepcopy(carrier.config))
                value = cast("torch.Tensor", plan.scan.meta["val"])
                for extent in value.shape[:-1]:
                    bid = env.resolve_block_id(extent)
                    if bid is not None and bid in spec.block_sizes.valid_block_ids():
                        candidate.block_sizes[
                            spec.block_sizes.block_id_to_index(bid)
                        ] = 1
                if candidate not in carriers:
                    carriers.append(candidate)
    previous = spec.cute_fragment_scan_exports_search_enabled
    spec.cute_fragment_scan_exports_search_enabled = True
    for carrier in carriers:
        try:
            _, requested = spec.create_config_generation().strict_config_pair(
                Config.from_dict(carrier.config | {KEY: True})
            )
            host = device_ir.host_function
            assert host is not None
            with host:
                validate_scan_exports(env, device_ir, requested)
            break
        except InvalidConfig:
            continue
    else:
        spec.cute_fragment_scan_exports_search_enabled = previous
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_scan_exports",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            dependencies=(
                *scan_group.dependencies,
                CoverageDependency(scan_group.mechanism, scan_group.key, True),
            ),
            deferred=True,
        )
    )
