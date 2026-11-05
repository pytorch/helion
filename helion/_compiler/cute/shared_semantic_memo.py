"""Optimization-only typed value keys after the complete shared proof.

Original DAG nodes/provenance and strict dag_keys remain authoritative. These
keys are never an admission proof or a substitute for final bound alias facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import TYPE_CHECKING

import sympy
import torch

from . import cute_epilogue as epi

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SharedSemanticMemo:
    scope: torch.fx.Node
    strict_keys: tuple[tuple[object, ...], ...]
    value_keys: tuple[tuple[object, ...], ...]
    raw_keys: tuple[tuple[object, ...], ...]
    load_signatures: tuple[tuple[object, ...], ...]
    origins_by_value: tuple[tuple[torch.fx.Node, ...], ...]


def _dimension_key(value: object) -> object:
    if type(value) is int:
        return (int, value)
    if isinstance(value, torch.SymInt) and isinstance(value.node.expr, sympy.Expr):
        return (torch.SymInt, value.node.expr)
    raise ValueError("semantic CSE needs proved integer layout metadata")


def _tensor_signature(value: object) -> object:
    if not isinstance(value, torch.Tensor) or value.layout is not torch.strided:
        raise ValueError("semantic CSE needs a strided typed tensor")
    return (
        value.dtype,
        value.device,
        tuple(_dimension_key(v) for v in value.shape),
        tuple(_dimension_key(v) for v in value.stride()),
        _dimension_key(value.storage_offset()),
    )


def _access_signature(leaf: epi._AuxiliaryTensorLoadExpr) -> object:
    from ...language import memory_ops

    load = leaf.load_node
    if (
        load.op != "call_function"
        or load.target is not memory_ops.load
        or load.kwargs
        or not 2 <= len(load.args) <= 4
        or not isinstance(load.args[0], torch.fx.Node)
        or load.args[0].graph is not load.graph
        or not isinstance(load.args[1], (list, tuple))
        or any(v is not None for v in load.args[2:])
    ):
        raise ValueError("semantic CSE needs an unchanged unmasked load")
    indices: list[object] = []
    for index in load.args[1]:
        if isinstance(index, torch.fx.Node) and index.graph is load.graph:
            indices.append((torch.fx.Node, index))
        elif type(index) is int:
            indices.append((int, index))
        else:
            raise ValueError("semantic CSE needs original owned load coordinates")
    return (
        load.graph,
        load.args[0],  # Original host input identity, never a sampled pointer/name.
        tuple(indices),
        _tensor_signature(load.args[0].meta.get("val")),
        _tensor_signature(load.meta.get("val")),
        leaf.broadcast_axis,
    )


def _supported_load(leaf: epi._AuxiliaryTensorLoadExpr) -> bool:
    # Mapped/modular transports remain opaque strict-origin values for now.
    return type(leaf) in (
        epi._AuxiliaryTensorLoadExpr,
        epi._ReadonlyBoolAuxLoadExpr,
        epi._ReadonlyBoolPredicateLoadExpr,
        epi._MVectorBroadcastAuxLoadExpr,
        epi._MVectorScalarPrefixLoadExpr,
    )


def load_signature(leaf: epi._AuxiliaryTensorLoadExpr) -> object:
    from .shared_epilogue_dag import raw_aux_key

    if not _supported_load(leaf):
        return ("opaque", raw_aux_key(leaf))
    return (type(leaf), _access_signature(leaf), leaf.template)


class _ValuePool:
    def __init__(self, scope: torch.fx.Node, anchor: torch.fx.Node) -> None:
        self.scope = scope
        self.anchor = anchor
        self.values: dict[object, int] = {}
        self.origins: list[list[torch.fx.Node]] = []

    def intern(self, key: object, origin: torch.fx.Node) -> object:
        if key not in self.values:
            # Original admission/node budgets ran before this optimization.
            # This is a separate conservative work bound, never a smaller
            # post-CSE budget used to admit a previously unproved graph.
            if len(self.values) >= 256:
                raise ValueError("bounded semantic CSE value pool exceeded")
            self.values[key] = len(self.values)
            self.origins.append([])
        index = self.values[key]
        if origin not in self.origins[index]:
            self.origins[index].append(origin)
        return ("shared-semantic-value", self.scope, index)

    def dag(self, dag: epi._PointwiseDagExpr) -> tuple[object, ...]:
        from .shared_epilogue_dag import dag_keys

        strict = dag_keys(dag)  # Full original provenance/topology, unchanged.
        if len(dag.nodes) > 64:
            raise ValueError("unproved semantic CSE DAG budget")
        names: dict[int, object] = {}
        result = []
        for node, (origin, dtype), old in zip(
            dag.nodes, dag.provenance, strict, strict=True
        ):
            if isinstance(node, epi._CurrentTensorExpr):
                if origin is not self.anchor or dtype is not torch.float32:
                    raise ValueError("semantic current escaped the proved anchor")
                atom = ("current", self.anchor, dtype)
            elif origin.graph is not self.scope.graph:
                raise ValueError("semantic value escaped the proved store graph")
            elif isinstance(node, epi._MVectorScalarPrefixLoadExpr):
                prefix = self.dag(node.prefix)
                atom = (type(node), dtype, load_signature(node), prefix[-1])
            elif isinstance(node, epi._AuxiliaryTensorLoadExpr):
                atom = (
                    (type(node), dtype, load_signature(node))
                    if _supported_load(node)
                    else ("opaque", old)
                )
            elif (
                isinstance(node, epi._ScalarConstantExpr)
                and node.dtype is torch.float32
            ):
                value = float(node.literal)
                if not math.isfinite(value):
                    raise ValueError("unproved nonfinite semantic literal")
                bits = struct.pack("<f", value)
                # Retain spelling as well as exact storage bits: no additional
                # weak-scalar/double-rounding equivalence is inferred here.
                atom = ("constant", dtype, node.dtype, bits, node.literal)
            elif isinstance(node, epi._UnaryTensorExpr):
                atom = (
                    "unary",
                    origin.target,
                    dtype,
                    node.step,
                    names[id(node.operand)],
                )
            elif isinstance(node, epi._BinaryTensorExpr):
                atom = (
                    "binary",
                    origin.target,
                    dtype,
                    node.op_name,
                    node.op_template,
                    names[id(node.lhs)],
                    names[id(node.rhs)],
                )
            elif isinstance(node, epi._WhereTensorExpr):
                atom = (
                    "where",
                    origin.target,
                    dtype,
                    node.op_template,
                    names[id(node.predicate)],
                    names[id(node.lhs)],
                    names[id(node.rhs)],
                )
            else:
                # Runtime scalars and non-FP32 constants retain origin identity.
                atom = ("opaque", old)
            key = self.intern(atom, origin)
            names[id(node)] = key
            result.append(key)
        return tuple(result)


def build_shared_semantic_memo(
    chains: Sequence[epi.Tcgen05UnaryEpilogueChain],
    scope: torch.fx.Node,
    anchor: torch.fx.Node,
) -> SharedSemanticMemo | None:
    """Called only after the complete shared store/type/readonly/alias proof."""
    from .shared_epilogue_dag import dag_keys
    from .shared_epilogue_dag import raw_aux_key

    pool = _ValuePool(scope, anchor)
    strict_sets = []
    value_sets = []
    raw_sets = []
    signatures = []
    representatives: dict[object, object] = {}
    extra_equivalence = False
    try:
        for chain in chains:
            if len(chain.steps) != 1 or not isinstance(
                chain.steps[0].expr, epi._PointwiseDagExpr
            ):
                raise ValueError("semantic CSE needs an already proved shared DAG")
            dag = chain.steps[0].expr
            strict = dag_keys(dag)
            values = pool.dag(dag)
            for old, new in zip(strict, values, strict=True):
                if new in representatives and representatives[new] != old:
                    extra_equivalence = True
                representatives.setdefault(new, old)
            strict_sets.append(strict)
            value_sets.append(values)
            loads = []
            checked = []
            for leaf in chain.auxiliary_tensor_loads:
                signature = load_signature(leaf)
                # The raw fragment may be reused across distinct computed
                # prefixes; their VALUE keys still include every typed prefix
                # operation/child and cannot collapse different computations.
                key = ("shared-semantic-load", scope, signature)
                old = raw_aux_key(leaf)
                if key in representatives and representatives[key] != old:
                    extra_equivalence = True
                representatives.setdefault(key, old)
                loads.append(key)
                checked.append(signature)
            raw_sets.append(tuple(loads))
            signatures.append(tuple(checked))
    except (ValueError, KeyError, OverflowError):
        return None  # Keep the old proved rendering, not wider admission.
    if not extra_equivalence:
        return None
    return SharedSemanticMemo(
        scope,
        tuple(strict_sets),
        tuple(value_sets),
        tuple(raw_sets),
        tuple(signatures),
        tuple(tuple(origins) for origins in pool.origins),
    )
