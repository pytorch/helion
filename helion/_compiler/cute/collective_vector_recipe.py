"""Group a proven scalar operand recipe into contiguous register vectors.

The collective lowering has already proved pure single-assignment recipes,
read-only inputs, control placement and disjoint output storage. This module
changes only the within-tile traversal: invariant values are reused, proven
global loads are vectorized, and every remaining expression runs per lane.
It retains the caller's arithmetic and masks, including protected FP32 products.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .contiguous_copy import _expr
from .contiguous_copy import _replace
from .contiguous_copy import plan_contiguous_copy
from .scalar_recipe import _clone
from .scalar_recipe import _read_names

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .collective_operand_packet import SharedOperandPacket
    from .contiguous_copy import CopyTensorFacts


def emit_vector_recipe(
    statements: Sequence[ast.Assign],
    value: ast.expr,
    *,
    coordinate: str,
    width: int,
    tensors: Mapping[str, CopyTensorFacts],
    aligned_names: Mapping[str, int],
    destination: str,
    destination_indices: tuple[ast.expr, ...] | None,
    fresh_name: Callable[[str], str],
    destination_offset: ast.expr | None = None,
    shared_packet: SharedOperandPacket | None = None,
) -> list[ast.stmt]:
    """Emit one full destination vector from an already admitted recipe.

    The caller bounds the vector to its shared tile and uses a positive power
    of two width. Source tails keep their exact scalar predicates. Only loads
    with the existing contiguous-copy proof use a 128-bit global transfer;
    unaligned, gathered and differently sized loads stay scalar per lane.
    """
    assert width > 0 and width & (width - 1) == 0
    targets: list[str] = []
    for statement in statements:
        assert len(statement.targets) == 1
        target = statement.targets[0]
        assert isinstance(target, ast.Name)
        targets.append(target.id)
    assert len(set(targets)) == len(targets) and coordinate not in targets
    lane = fresh_name("recipe_lane")
    varying = {coordinate}
    common: list[ast.stmt] = []
    scalar: list[ast.stmt] = []
    replacements = {coordinate: _expr(f"{coordinate} + {lane}")}
    for index, statement in enumerate(statements):
        target = targets[index]
        assert not (_read_names(statement.value) & set(targets[index:]))
        if not (_read_names(statement.value) & varying):
            common.append(_clone(statement))
            continue
        varying.add(target)
        # Look only at actual load definitions. Re-vectorizing their aliases
        # would duplicate memory traffic and undo the register reuse below.
        has_load = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            for node in ast.walk(statement.value)
        )
        plan = (
            plan_contiguous_copy(
                statements[: index + 1],
                _expr(target),
                coordinate=coordinate,
                tensors=tensors,
                aligned_names=aligned_names,
            )
            if has_load
            else None
        )
        if plan is not None and plan.width == width:
            load, registers = plan.emit_to_registers(fresh_name)
            common.extend(load)
            scalar.extend(ast.parse(f"{target} = {registers}[{lane}]").body)
        else:
            copied = _replace(statement, replacements)
            assert isinstance(copied, ast.stmt)
            scalar.append(copied)
    stored = _replace(value, replacements)
    assert destination_offset is None or destination_indices is None
    indices = (
        _replace(
            ast.Tuple(elts=list(destination_indices), ctx=ast.Load()), replacements
        )
        if destination_indices is not None
        else ast.Name(id=lane, ctx=ast.Load())
    )
    assert isinstance(stored, ast.expr) and isinstance(indices, ast.expr)
    if destination_offset is not None:
        indices = ast.BinOp(_clone(destination_offset), ast.Add(), indices)
    store_destination = destination
    store_indices = indices
    if shared_packet is not None:
        assert width == shared_packet.width
        assert destination_indices is not None and destination_offset is None
        store_destination = fresh_name("operand_packet")
        common.append(shared_packet.allocate(store_destination))
        store_indices = ast.Name(id=lane, ctx=ast.Load())
    scalar.append(
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=_expr(store_destination), slice=store_indices, ctx=ast.Store()
                )
            ],
            value=stored,
        )
    )
    common.append(
        ast.For(
            target=ast.Name(id=lane, ctx=ast.Store()),
            iter=_expr(f"cutlass.range_constexpr({width})"),
            body=scalar,
            orelse=[],
        )
    )
    if shared_packet is not None:
        assert destination_indices is not None
        common.extend(
            shared_packet.store(
                store_destination, destination, destination_indices, fresh_name
            )
        )
    return [ast.fix_missing_locations(node) for node in common]
