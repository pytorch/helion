"""Default-off coverage for one-row warp producer prefixes."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import cast

import sympy

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import scan_ops
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    import torch

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_warp_producer_regions"


def validate_warp_producer_regions(
    env: CompileEnvironment, device_ir: DeviceIR, config: Config
) -> None:
    from ..cute.fragment_storage import configured_fragment_expr
    from ..cute.warp_producer_regions import producer_prefix

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
                raise InvalidConfig("warp producer extent is not statically configured")
            result.append(int(value))
        return tuple(result)

    for root, graphs in fragment_root_regions(device_ir):
        if (
            producer_prefix(
                next(g.graph for g in graphs if g.graph_id == root), env, graphs, shape
            )
            is not None
        ):
            return
    raise InvalidConfig("warp producers require a one-row scalar-frontier prefix")


class CuteFragmentWarpProducerRegionsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        env.config_spec.cute_fragment_warp_producer_regions_root_ids = frozenset(
            root
            for root, graphs in fragment_root_regions(device_ir)
            if any(
                n.target is scan_ops._associative_scan
                for g in graphs
                if g.graph_id == root
                for n in g.graph.nodes
            )
        )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_warp_producer_regions_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_warp_producer_regions_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_warp_producer_regions_root_ids:
        return
    scan = next(
        (
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_warp_scan"
        ),
        None,
    )
    reduction = next(
        (
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_reduction"
        ),
        None,
    )
    if scan is None or reduction is None:
        return
    carrier = Config.from_dict(
        deepcopy(scan.witnesses[0].carrier.config)
        | {scan.key: True, reduction.key: "warp"}
    )
    carriers = [carrier]
    host = device_ir.host_function
    assert host is not None
    with host:
        for root, graphs in fragment_root_regions(device_ir):
            for graph in graphs:
                if graph.graph_id != root:
                    continue
                for node in graph.graph.nodes:
                    if node.target is not scan_ops._associative_scan:
                        continue
                    candidate = Config.from_dict(deepcopy(carrier.config))
                    value = cast("torch.Tensor", node.meta["val"])
                    for extent in value.shape[:-1]:
                        bid = env.resolve_block_id(extent)
                        if (
                            bid is not None
                            and bid in spec.block_sizes.valid_block_ids()
                        ):
                            candidate.block_sizes[
                                spec.block_sizes.block_id_to_index(bid)
                            ] = 1
                    if candidate not in carriers:
                        carriers.append(candidate)
    old = spec.cute_fragment_warp_producer_regions_search_enabled
    spec.cute_fragment_warp_producer_regions_search_enabled = True
    for carrier in carriers:
        try:
            _, requested = spec.create_config_generation().strict_config_pair(
                Config.from_dict(carrier.config | {KEY: True})
            )
            with host:
                validate_warp_producer_regions(env, device_ir, requested)
            break
        except InvalidConfig:
            continue
    else:
        spec.cute_fragment_warp_producer_regions_search_enabled = old
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_warp_producer_regions",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            dependencies=(
                *scan.dependencies,
                CoverageDependency(scan.mechanism, scan.key, True),
                CoverageDependency(reduction.mechanism, reduction.key, "warp"),
            ),
            deferred=True,
        )
    )
