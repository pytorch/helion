"""Additive, default-off coverage for bounded complete warp-prefix scans."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import sympy
import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import scan_ops
from ...runtime.config import Config
from .cute_fragment_bounded_gather import bounded_gather_roots
from .cute_fragment_common import FragmentRootRequirement
from .cute_fragment_common import active_fragment_roots
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_warp_scan"
WARP_SCAN_DTYPES = (torch.float32, torch.float64, torch.int32, torch.int64)
MAX_WARP_SCAN_SIZE = 32 * 32


def active_warp_scan_roots(env: CompileEnvironment, config: Config) -> frozenset[int]:
    spec = env.config_spec
    return active_fragment_roots(
        spec.cute_fragment_warp_scan_root_ids,
        spec.cute_fragment_warp_scan_requirements,
        config.config,
    )


def warp_scan_supported(dtype: torch.dtype, capacity: int | None) -> bool:
    """Eligibility of one scan, using its configured physical axis capacity."""
    return (
        dtype in WARP_SCAN_DTYPES
        and capacity is not None
        and capacity > 0
        and (capacity <= MAX_WARP_SCAN_SIZE or dtype in (torch.int32, torch.int64))
    )


def validate_warp_scan_request(
    env: CompileEnvironment, device_ir: DeviceIR, config: Config
) -> None:
    """Reject a vacuous request once, before splitting explicit grid phases.

    Each phase may contain long or unsupported-dtype scans which keep its
    selected serial/cooperative schedule. At least one operation in a proved
    root must have a bounded configured extent, using the same resolver as
    fragment allocation. No tracing hint is used without its metadata guard.
    """
    from ..cute.fragment_storage import configured_fragment_expr

    for root, graphs in fragment_root_regions(device_ir):
        if root not in active_warp_scan_roots(env, config):
            continue
        for info in graphs:
            for node in info.graph.nodes:
                if node.target is not scan_ops._associative_scan:
                    continue
                value = node.meta["val"]
                assert isinstance(value, torch.Tensor)
                dim = node.args[2]
                assert isinstance(dim, int)
                extent = value.shape[dim]
                expression = (
                    extent._sympy_() if isinstance(extent, torch.SymInt) else extent
                )
                resolved = env.specialize_expr(
                    sympy.sympify(
                        configured_fragment_expr(
                            env,
                            sympy.sympify(expression),
                            lambda bid: env.block_sizes[bid].from_config(config),
                        )
                    )
                )
                capacity = int(resolved) if resolved.is_number else None
                if warp_scan_supported(value.dtype, capacity):
                    return
    raise InvalidConfig(
        "warp-prefix scan requires at least one supported operation with positive "
        "axis capacity; floating-point capacity must be <= 1024"
    )


class CuteFragmentWarpScanHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:

        host = device_ir.host_function
        assert host is not None
        roots = set()
        dependent = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                scans = [
                    node
                    for info in graphs
                    for node in info.graph.nodes
                    if node.target is scan_ops._associative_scan
                ]
                if scans and any(
                    isinstance(node.meta.get("val"), torch.Tensor)
                    and node.meta["val"].dtype in WARP_SCAN_DTYPES
                    for node in scans
                ):
                    if computed_fragment_discovery_supported(env, graphs):
                        roots.add(root)
                    else:
                        dependent.add(root)
        if dependent:
            dependent.intersection_update(
                bounded_gather_roots(env, device_ir, allow_unbound=True)
            )
        env.config_spec.cute_fragment_warp_scan_root_ids = frozenset(roots)
        env.config_spec.cute_fragment_warp_scan_requirements = tuple(
            FragmentRootRequirement(root, frozenset({"cute_fragment_bounded_gather"}))
            for root in sorted(dependent)
        )
        # Configured capacities and runtime logical bounds are checked again
        # during emission. Exact input metadata must remain bound on rebinding.
        return (
            frozenset({"input_tensor_metadata"}) if roots or dependent else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_warp_scan_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_warp_scan_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    spec.cute_fragment_warp_scan_search_enabled = True
    register_fragment_boolean_coverage(spec, KEY, carrier)


def register_gather_warp_scan_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Append a dependent-only witness after its existing gather carrier.

    An ordinary scan group keeps its exact declaration and position. Mixed roots
    can use an explicit combined config, but gain no extra initial witness here.
    """
    spec = env.config_spec
    if (
        spec.cute_fragment_warp_scan_root_ids
        or not spec.cute_fragment_warp_scan_requirements
    ):
        return
    if any(group.key == KEY for group in spec.compiler_coverage_groups):
        return
    group = next(
        (
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_bounded_gather"
        ),
        None,
    )
    if group is None:
        return
    carrier = Config.from_dict(group.witnesses[0].carrier.config | {group.key: True})
    try:
        spec.create_config_generation().strict_config_pair(carrier)
    except InvalidConfig:
        return
    previous = spec.cute_fragment_warp_scan_search_enabled
    spec.cute_fragment_warp_scan_search_enabled = True
    try:
        _, requested = spec.create_config_generation().strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
        host = device_ir.host_function
        assert host is not None
        with host:
            validate_warp_scan_request(env, device_ir, requested)
    except InvalidConfig:
        spec.cute_fragment_warp_scan_search_enabled = previous
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_warp_scan",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            dependencies=(CoverageDependency(group.mechanism, group.key, True),),
            deferred=True,
        )
    )
