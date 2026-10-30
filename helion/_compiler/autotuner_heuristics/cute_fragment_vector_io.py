"""``cute_fragment_vector_io``: 16-byte row packets for proved host windows.

The fragment lowering moves every host window one scalar at a time: a 2- or
4-byte load (or store) per element behind its own address and predicate
chain, which makes the staging phases of a tile-staging kernel issue-bound
(the anchored chunk gram issues 2.7x Triton's instructions).  With the knob
on, a window whose innermost coordinate is proved at compile time to walk
whole, aligned, in-bounds packets of a unit-stride dimension (a full slice,
a packet-multiple tile, or an ``hl.arange`` window at a packet-multiple
offset) moves as 16-byte packets: one address and predicate per four fp32 /
eight 16-bit elements, a vector global access and a vector shared store.
Windows the proof declines keep the scalar walk, so the knob is only offered
on roots with at least one candidate window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_vector_io"


def _candidate_window(env: CompileEnvironment, node: Node) -> bool:
    """A host load/store whose innermost index could form row packets: the
    emitter proves the rest (alignment, static extents, index form)."""
    from ..cute.fragment_vector_io import packet_lanes
    from ..cute.fragment_vector_io import static_strides

    source = node.args[0]
    if not (isinstance(source, Node) and source.target is _tracing_ops._host_tensor):
        return False
    fake = source.meta.get("val")
    if not isinstance(fake, torch.Tensor) or not fake.ndim:
        return False
    lanes = packet_lanes(fake.dtype)
    strides = static_strides(fake)
    if lanes is None or strides is None or strides[-1] != 1:
        return False
    if any(stride % lanes for stride in strides[:-1]):
        return False
    indices = [index for index in node.args[1] if index is not None]  # type: ignore[union-attr]
    if len(indices) != fake.ndim:
        return False
    last = indices[-1]
    if isinstance(last, slice):
        return last == slice(None)
    if isinstance(last, Node):
        proxy = last.meta.get("val")
        if isinstance(proxy, torch.SymInt):
            return env.resolve_block_id(proxy) is not None
        return last.target is torch.ops.prims.iota.default
    return False


def _root_has_candidate(env: CompileEnvironment, graphs: list[GraphInfo]) -> bool:
    return any(
        node.target in (memory_ops.load, memory_ops.store)
        and _candidate_window(env, node)
        for info in graphs
        for node in info.graph.nodes
    )


class CuteFragmentVectorIoHeuristic(AutotunerHeuristic):
    """Register every complete fragment root with a packetizable window."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_vector_io_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_discovery_supported(env, graphs)
                and _root_has_candidate(env, graphs)
            )
        # The packet proof reads static strides, extents and the base
        # alignment residue: all of them must survive rebinding.
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_vector_io_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_vector_io_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_vector_io_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous_enabled = spec.cute_fragment_vector_io_search_enabled
    spec.cute_fragment_vector_io_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_vector_io_search_enabled = previous_enabled
