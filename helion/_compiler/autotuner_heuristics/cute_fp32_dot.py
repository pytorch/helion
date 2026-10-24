"""``cute_fp32_dot``: tf32 ``mma.sync`` for fp32 dots in computed fragments.

The CTA-local fragment lowering (``computed_fragment.py``) contracts an fp32
``hl.dot`` as one serial FMA chain per output element with both operands read
from shared memory (two smem reads per FMA).  Triton's ``tl.dot`` runs the
same dot on tensor cores at tf32 precision.  This knob lets the autotuner
choose: ``"simt"`` keeps the exact fp32 chain, ``"tf32_mma"`` lowers every
qualifying dot of the root through ``mma.sync.aligned.m16n8k8`` tf32 with the
operands staged in the fragment buffers (the ``input_precision='tf32'``
numerics Helion's default ``dot_precision`` maps to; ``"ieee"`` never offers
the mode).  Seeds offer ``tf32_mma`` once the root's qualifying dots total
at least 64x64x64 multiply-adds (one such dot, or many 16-row dots as the
anchored chunk grams issue); smaller roots keep it an ordinary search choice.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import matmul_ops
from ...runtime.config import Config
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fp32_dot"
MODES = ("simt", "tf32_mma")
# A dot this large on every axis makes tf32_mma a seeded witness.
SEED_DOT_EXTENT = 64
# ``mma.sync`` with tf32 operands exists from sm_80.
MIN_CAPABILITY = 8


def fragment_tf32_dot_supported(node: Node, env: CompileEnvironment) -> bool:
    """Whether *node* is an fp32 ``hl.dot`` the tf32 fragment MMA may lower.

    Operands and the accumulator must be fp32 (the tensor core truncates the
    staged fp32 words to tf32; nothing is rounded to half), the kernel's
    ``dot_precision`` must map to tf32 and the target must have ``mma.sync``
    tf32.  Tile divisibility (16 x 8 x 8) is a codegen matter: partial tiles
    are predicated, so a qualifying dot never falls back silently.
    """
    from ..cute.mma_support import cute_fp32_dot_uses_tf32

    if node.op != "call_function" or node.target is not matmul_ops.dot:
        return False
    if len(node.args) < 2:
        return False
    operands = [node.args[0], node.args[1]]
    if len(node.args) > 2 and node.args[2] is not None:
        operands.append(node.args[2])
    for operand in operands:
        value = operand.meta.get("val") if isinstance(operand, torch.fx.Node) else None
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            return False
    out = node.meta.get("val")
    if not isinstance(out, torch.Tensor) or out.dtype != torch.float32:
        return False
    if not cute_fp32_dot_uses_tf32():
        return False
    capability = env.config_spec.target_device_capability
    return capability is not None and capability[0] >= MIN_CAPABILITY


def _extent_hint(env: CompileEnvironment, size: int | torch.SymInt) -> int:
    """A tiled axis counts with its full extent: the block size is the search's."""
    block_id = env.resolve_block_id(size) if isinstance(size, torch.SymInt) else None
    if block_id is not None:
        return env.block_sizes[block_id].size_hint()
    return env.size_hint(size)


def _dot_extent_hints(node: Node, env: CompileEnvironment) -> tuple[int, int, int]:
    lhs, rhs = node.args[:2]
    assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
    left, right = lhs.meta["val"], rhs.meta["val"]
    return (
        _extent_hint(env, left.shape[-2]),
        _extent_hint(env, right.shape[-1]),
        _extent_hint(env, left.shape[-1]),
    )


def fragment_fp32_dot_roots(env: CompileEnvironment, ir: DeviceIR) -> frozenset[int]:
    return frozenset(
        root
        for root, graphs in fragment_root_regions(ir)
        if any(
            fragment_tf32_dot_supported(node, env)
            for info in graphs
            for node in info.graph.nodes
        )
        and computed_fragment_discovery_supported(env, graphs)
    )


def fragment_fp32_dot_seed_worthy(env: CompileEnvironment, ir: DeviceIR) -> bool:
    """An admitted root's qualifying dots total at least ``SEED_DOT_EXTENT``
    cubed multiply-adds -- one 64x64x64 dot, or eight 16-row dots over a
    128-long contraction, as the anchored chunk grams issue per tile."""
    roots = env.config_spec.cute_fp32_dot_root_ids
    total = 0
    for root, graphs in fragment_root_regions(ir):
        if root not in roots:
            continue
        for info in graphs:
            for node in info.graph.nodes:
                if fragment_tf32_dot_supported(node, env):
                    total += math.prod(_dot_extent_hints(node, env))
    return total >= SEED_DOT_EXTENT**3


class CuteFp32DotHeuristic(AutotunerHeuristic):
    """Register the fp32-dot roots; coverage offers the mode, nothing promotes it."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        if not any(
            fragment_tf32_dot_supported(node, env)
            for info in device_ir.graphs
            for node in info.graph.nodes
        ):
            return frozenset()
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fp32_dot_root_ids = fragment_fp32_dot_roots(
                env, device_ir
            )
            env.config_spec.cute_fp32_dot_seed_worthy = fragment_fp32_dot_seed_worthy(
                env, device_ir
            )
        # Shape/stride assumptions of an admitted root must survive rebinding.
        return frozenset({"input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fp32_dot_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    supplemental_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fp32_dot_root_ids:
        return
    carrier = fragment_coverage_carrier(spec)
    if carrier is None:
        return
    spec.cute_fp32_dot_search_enabled = True
    generation = spec.create_config_generation()
    try:
        for mode in MODES:
            requested = Config.from_dict(deepcopy(carrier.config) | {KEY: mode})
            _, effective = generation.strict_config_pair(requested)
            if effective.config.get(KEY, "simt") != mode:
                spec.cute_fp32_dot_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_fp32_dot_search_enabled = False
        return
    # The group owns the field from here on.  Only a root with a dot of at
    # least 64 on every axis seeds the mode; smaller dots keep it an ordinary
    # enum choice of the search without a declared witness.
    seeded = spec.cute_fp32_dot_seed_worthy
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.computed_fragment_fp32_dot",
            version=1,
            key=KEY,
            domain=MODES,
            legacy="simt",
            witnesses=(CoverageWitness(carrier, "tf32_mma"),) if seeded else (),
            supplemental_witnesses=(CoverageWitness(supplemental_carrier, "tf32_mma"),)
            if seeded and supplemental_carrier is not None
            else (),
        )
    )
