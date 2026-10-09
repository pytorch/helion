"""Conservative storage preferences for supplemental fragment coverage.

These estimates choose seeds only. They never establish compiler admission or
remove ordinary configurations, defaults, seeds, or coverage witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from ... import exc
from ...language import _tracing_ops
from ...language import matmul_ops
from ...language import scan_ops
from ..cute.fragment_storage import aligned_shared_bytes
from ..cute.fragment_storage import configured_fragment_expr
from ..cute.tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


@dataclass(frozen=True)
class FragmentStorageTerm:
    shape: tuple[int | torch.SymInt, ...]
    dtype: torch.dtype
    copies: int = 1

    def shared_bytes(self, env: CompileEnvironment, config: Config) -> int:
        extents = []
        for size in self.shape:
            expression = (
                size._sympy_()
                if isinstance(size, torch.SymInt)
                else sympy.Integer(size)
            )
            value = env.specialize_expr(
                sympy.sympify(
                    configured_fragment_expr(
                        env,
                        expression,
                        lambda bid: env.block_sizes[bid].from_config(config),
                    )
                )
            )
            if not value.is_number or int(value) < 1:
                raise exc.InvalidConfig("unproved fragment resource extent")
            extents.append(int(value))
        return self.copies * aligned_shared_bytes(math.prod(extents), self.dtype)


@dataclass(frozen=True)
class FragmentResourceCatalog:
    """A fixed logical graph's no-reuse allocation upper bound.

    Every straight-line emitter allocates at most its tensor output, except scan
    scratch/source materialization and dot operand materialization. Counting even
    lazy tensor outputs gives an upper bound without duplicating allocator
    liveness. Nested graphs need a separate lifetime proof and are not estimated.
    """

    roots: tuple[tuple[FragmentStorageTerm, ...], ...]

    @classmethod
    def create(
        cls, env: CompileEnvironment, ir: DeviceIR, config: Config
    ) -> FragmentResourceCatalog | None:
        from ..cute.materialized_fission_codegen import _region_graph_ids
        from ..device_ir import HelperFunctionGraphInfo
        from ..device_ir import RootGraphInfo

        roots = (
            env.config_spec.cute_fragment_scan_root_ids
            | env.config_spec.cute_fragment_reduction_root_ids
        )
        if not roots:
            return None
        # Do not let configuration-specific simplification hide control flow
        # whose other geometry could require a different allocation catalog.
        if any(
            _tracing_ops.is_for_loop_target(node.target)
            or node.target in (_tracing_ops._if, _tracing_ops._while_loop)
            for root in roots
            for node in ir.graphs[root].graph.nodes
        ):
            return None
        graphs = ir.build_codegen_graphs(config, roll_reductions=False)
        terms_by_root = []
        for root in sorted(roots):
            if not isinstance(graphs[root], RootGraphInfo):
                return None
            nodes = list(graphs[root].graph.nodes)
            if any(
                _tracing_ops.is_for_loop_target(node.target)
                or node.target in (_tracing_ops._if, _tracing_ops._while_loop)
                for node in nodes
            ):
                return None
            # The already-proved additive scan helper emits scalar arithmetic,
            # not a nested fragment allocation. Other nested graphs decline.
            scan_helpers = {
                node.args[0]
                for node in nodes
                if node.target is scan_ops._associative_scan
            }
            reachable = _region_graph_ids(graphs, root)
            if reachable != {root, *scan_helpers} or any(
                not isinstance(graphs[helper], HelperFunctionGraphInfo)
                for helper in scan_helpers
            ):
                return None
            terms = []
            for node in graphs[root].graph.nodes:
                if (
                    node.op != "call_function"
                    or node.target is _tracing_ops._host_tensor
                ):
                    continue
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    terms.append(FragmentStorageTerm(tuple(value.shape), value.dtype))
                if node.target is scan_ops._associative_scan:
                    source = node.args[1].meta["val"]
                    terms.append(
                        FragmentStorageTerm(tuple(source.shape), source.dtype, 2)
                    )
                elif node.target is matmul_ops.dot:
                    for argument in node.args[:2]:
                        source = argument.meta["val"]
                        terms.append(
                            FragmentStorageTerm(tuple(source.shape), source.dtype)
                        )
            terms_by_root.append(tuple(terms))
        return cls(tuple(terms_by_root))

    def shared_bytes(self, env: CompileEnvironment, config: Config) -> int:
        return max(
            sum(term.shared_bytes(env, config) for term in terms)
            for terms in self.roots
        )


def fragment_resource_carrier(env: CompileEnvironment, ir: DeviceIR) -> Config | None:
    """Descend only legal geometry coordinates until a storage bound fits.

    Strict decrease over finite block-size domains terminates without random
    draws. A failed proof/local minimum leaves all existing coverage untouched.
    The catalog is built once; prospective neighbors never rebuild graphs or
    invoke code generation.
    """
    spec = env.config_spec
    if not (spec.cute_fragment_scan_root_ids or spec.cute_fragment_reduction_root_ids):
        return None
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
    if not capacity:
        return None
    generation = spec.create_config_generation()
    host = ir.host_function
    assert host is not None
    with host:
        try:
            _, initial = generation.canonicalize_flat(generation.default_flat())
            catalog = FragmentResourceCatalog.create(env, ir, initial)
            if catalog is None:
                return None
            current = initial
            bound = catalog.shared_bytes(env, current)
            if bound <= capacity:
                return None
            while bound > capacity:
                candidates = []
                for item in generation.coordinate_neighbor_projections(
                    generation.flatten(current)
                ):
                    if item.key != "block_sizes" or item.outcome != "candidate":
                        continue
                    candidate = item.config
                    assert candidate is not None
                    if any(
                        candidate.get(key) != current.get(key)
                        for key in set(candidate) | set(current)
                        if key != "block_sizes"
                    ):
                        continue
                    estimate = catalog.shared_bytes(env, candidate)
                    if estimate < bound:
                        candidates.append((estimate, item.flat_index, candidate))
                if not candidates:
                    return None
                bound, _, current = min(candidates, key=operator.itemgetter(slice(2)))
            generation.strict_config_pair(current)
            return current
        except exc.InvalidConfig:
            return None
