"""Lower explicit grid synchronization to ordered CuTe kernel launches."""

from __future__ import annotations

import ast
import copy
import dataclasses
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _tracing_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..host_function import HostFunction
from ..type_info import BarrierResultType
from ..type_info import BlockSizeType
from ..type_info import TensorType
from .materialized_fission import MaterializedFissionPlan
from .materialized_fission import _shape_only
from .materialized_fission_codegen import generate_materialized_fission
from .promote_output_axis import _fresh_tensors

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...runtime.config import Config


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise exc.BackendUnsupported("cute", f"ordered grid phases: {detail}")


def generate_ordered_phases(
    host: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    *,
    store_transform: Callable[..., ast.AST] | None = None,
    load_transform: Callable[..., ast.AST] | None = None,
    extra_params: list[str] | None = None,
) -> ast.Module:
    """Bundle one complete root per explicit barrier phase.

    Separate launches implement the existing grid-wide ordering contract.
    The ordinary compiler has already checked each phase's memory dependencies;
    this path does not infer additional synchronization for unmarked loops.
    """
    ir = host.device_ir
    _require(not ir.implicit_dependency_starts, "implicit dependency phases")
    _require(
        all(
            node.target is not _tracing_ops._if
            for graph in ir.graphs
            for node in graph.graph.nodes
        ),
        "runtime conditionals inside grid phases",
    )
    _require(
        all(
            node.target is not _tracing_ops._while_loop
            for graph in ir.graphs
            for node in graph.graph.nodes
        ),
        "runtime while loops inside grid phases",
    )
    _require(
        all(len(phase.roots) == 1 for phase in ir.phases),
        "each phase must contain one top-level grid loop",
    )
    roots = [phase.root_nodes[0] for phase in ir.phases]
    _require(
        len(roots) == len(ir.root_ids)
        and all(
            isinstance(root, ast.For)
            and isinstance(root, ExtendedAST)
            and root._loop_type is LoopType.GRID
            for root in roots
        ),
        "phase/root ownership mismatch",
    )
    first = host.body.index(roots[0])
    last = host.body.index(roots[-1])
    for statement in host.body[first : last + 1]:
        if statement in roots:
            continue
        _require(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ExtendedAST)
            and isinstance(statement.value._type_info, BarrierResultType),
            "host effects between grid phases",
        )
    tail = host.body[last + 1 :]
    _require(
        len(tail) == 1 and isinstance(tail[0], ast.Return), "post-grid host effects"
    )

    # Remove only the barriers validated above before proving that captured
    # allocations are fresh. The general freshness proof deliberately rejects
    # unrecognized host calls, including barriers.
    definition = dataclasses.replace(
        host.definition,
        body=[*host.body[:first], *roots, *tail],
    )
    bundle = HostFunction(definition, host.location)
    bundle.compiler_state = host.compiler_state
    bundle.local_types = host.local_types
    bundle.device_ir = copy.copy(ir)
    bundle.device_ir.host_function = bundle

    tensor_names = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    bound_names = set(host.params.arguments)
    fresh = _fresh_tensors(bundle)
    captures = []
    for statement in host.body[:first]:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue
        _require(isinstance(statement, ast.Assign), "unsupported host prelude")
        assert isinstance(statement, ast.Assign)
        targets = [
            node.id
            for target in statement.targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ]
        _require(
            bool(targets)
            and len(set(targets)) == len(targets)
            and not bound_names.intersection(targets),
            "rebound host metadata or allocation",
        )
        bound_names.update(targets)
        assert isinstance(statement.value, ExtendedAST)
        value_type = statement.value._type_info
        if isinstance(value_type, TensorType):
            _require(
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and value_type.proxy() in fresh,
                "host tensor prelude must allocate a fresh named buffer",
            )
            captures.extend(targets)
            tensor_names.update(targets)
        else:
            _require(
                isinstance(value_type, BlockSizeType)
                or _shape_only(statement.value, tensor_names),
                "host metadata expression is not shape-only",
            )

    plan = MaterializedFissionPlan(
        root_index=first,
        source_root_key=ast.dump(roots[0]),
        materialized_names=tuple(captures),
        region_count=len(roots),
    )
    return generate_materialized_fission(
        bundle,
        config,
        emit_repro_caller,
        plan,
        store_transform=store_transform,
        load_transform=load_transform,
        extra_params=extra_params,
    )
