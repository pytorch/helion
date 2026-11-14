"""Fold a V-loop's maximum or minimum of bf16 / f16 lanes as packed half2
words, bit for bit.

An ``amax`` / ``amin`` of a half-precision row the backend computes in fp32
reaches the kernel as a fold over the lanes of a loaded 16-bit packet::

    for v in cutlass.range_constexpr(V):
        load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
        ...
        acc = cute.arch.fmax(acc, cutlass.Float32(load), nan=True)

which costs a widening and an fp32 ``max`` per lane.  The packet already
holds the lanes two to a word, and sm_80 compares two halves per
instruction (``max.NaN.bf16x2``), so this pass moves the fold after the loop
as ``acc = cute.arch.fmax(acc, _cute_half_packet_extremum_f32(packet, V,
cutlass.BFloat16, 'max'), nan=True)``: ``V / 2 - 1`` packed instructions
(ptxas merges pairs of them into three-input ``VHMNMX``) and two widenings.

The result is the scalar fold's, bit for bit: widening is exact and
monotonic, the NaN-propagating ``max`` / ``min`` is associative and
commutative in both precisions (-0.0 orders below +0.0 in either operand
order, any NaN gives the canonical NaN), and the closing fp32 op turns a
packed NaN half into the canonical fp32 NaN the scalar fold returns.

A fold qualifies when its running value is read and written by that
statement alone in the loop body, its lane operand is the widened bitcast
of ``packet[v]`` (directly or through one fp32 local) for the loop's own
index ``v``, which the body never rebinds, each assigned earlier in the
same trip by the last statement before its use that writes it, the loop
body reads ``packet`` only through such subscripts, and every value the
kernel gives ``packet`` is a loaded packet, directly or through plain
aliases (``loaded_packets``): an SSA vector, whose lanes no store under
another name can change before the moved fold reads them.  The lane
statements stay in the loop for their other readers.
The pass runs under ``cute_pair_lane_math``, with the packed fp32 pairs.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..ast_extension import expr_from_string
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written

if TYPE_CHECKING:
    from ..device_ir import DeviceIR

HELPER = "_cute_half_packet_extremum_f32"
_FOLDS = {"cute.arch.fmax": "max", "cute.arch.fmin": "min"}
_HALVES = ("cutlass.BFloat16", "cutlass.Float16")
_FP32 = "cutlass.Float32"
_EXTREMUM_REDUCTIONS = (
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
    torch.ops.aten.max.dim,
    torch.ops.aten.min.dim,
)
_CONVERSIONS = (
    torch.ops.prims.convert_element_type.default,
    torch.ops.aten._to_copy.default,
)


def has_half_extremum_reduction(device_ir: DeviceIR) -> bool:
    """Whether the kernel traces a maximum or minimum reduction of bf16 or
    f16 values, widened or not: with the pairs of ``pair_lane_math``, what
    offers ``cute_pair_lane_math`` to the autotuner.  Structure only."""
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or node.target not in _EXTREMUM_REDUCTIONS:
                continue
            source = node.args[0]
            while isinstance(source, torch.fx.Node) and source.target in _CONVERSIONS:
                source = source.args[0]
            if (
                isinstance(source, torch.fx.Node)
                and isinstance(value := source.meta.get("val"), torch.Tensor)
                and value.dtype in (torch.bfloat16, torch.float16)
            ):
                return True
    return False


def pack_half_extremum_folds(
    body: list[ast.stmt], *, target_device_capability: tuple[int, int] | None
) -> list[ast.stmt]:
    """``body`` with the qualifying half-precision extremum folds packed."""
    if target_device_capability is None or target_device_capability < (8, 0):
        return body
    return _rewrite_block(body, loaded_packets(body))


def loads_packet(value: ast.expr) -> bool:
    """Whether ``value`` loads a packet, a vector of values read from
    memory."""
    if not isinstance(value, ast.Call):
        return False
    path = call_path_str(value.func)
    return path is not None and (
        path in ("cute.arch.load", "_cute_resident_load_vector")
        or path.startswith("_cute_load")
    )


def loaded_packets(body: list[ast.stmt]) -> frozenset[str]:
    """The names every binding of which in ``body`` assigns a loaded packet
    (``loads_packet``), a plain alias of another such name, or an element of
    a list that holds only such names (the loads-first lane unroll stashes
    its packets in one): SSA values, whose lanes no store can change.  Such
    a list is bound only to list displays and otherwise only appended to or
    indexed by element; any other use (a call it is passed to, an alias, a
    slice) may put anything into it."""
    parents = {
        id(child): parent
        for statement in body
        for parent in ast.walk(statement)
        for child in ast.iter_child_nodes(parent)
    }
    values: dict[str, list[ast.expr | None]] = {}
    elements: dict[str, list[ast.expr]] = {}
    other_uses: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name):
                continue
            parent = parents.get(id(node))
            if isinstance(node.ctx, ast.Store):
                if isinstance(parent, ast.Assign) and parent.targets == [node]:
                    values.setdefault(node.id, []).append(parent.value)
                    if isinstance(parent.value, ast.List):
                        elements.setdefault(node.id, []).extend(parent.value.elts)
                else:
                    # A loop variable, or an augmented or unpacked target.
                    values.setdefault(node.id, []).append(None)
            elif (
                isinstance(parent, ast.Subscript)
                and parent.value is node
                and not isinstance(parent.slice, ast.Slice)
            ):
                if isinstance(parent.ctx, ast.Store):
                    assign = parents.get(id(parent))
                    if isinstance(assign, ast.Assign) and assign.targets == [parent]:
                        elements.setdefault(node.id, []).append(assign.value)
                    else:
                        other_uses.add(node.id)
            elif (
                isinstance(parent, ast.Attribute)
                and parent.attr == "append"
                and isinstance(call := parents.get(id(parent)), ast.Call)
                and call.func is parent
                and len(call.args) == 1
            ):
                elements.setdefault(node.id, []).append(call.args[0])
            else:
                other_uses.add(node.id)
    # A resident load gives a register fragment, which a store can change:
    # it counts only while the body does nothing with it but read elements.
    loaded = {
        name
        for name, items in values.items()
        if None not in items
        and not (
            any(
                isinstance(item, ast.Call)
                and call_path_str(item.func) == "_cute_resident_load_vector"
                for item in items
            )
            and (name in other_uses or name in elements)
        )
    }
    lists = {
        name
        for name, items in values.items()
        if name not in other_uses and all(isinstance(item, ast.List) for item in items)
    }

    def packet_valued(value: ast.expr | None) -> bool:
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id in loaded
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            return value.value.id in lists
        return loads_packet(value)

    # The greatest fixed point: a cycle (a packet appended to a list and
    # read back from it) passes values around and adds none.
    changed = True
    while changed:
        changed = False
        for name in list(lists):
            if not all(map(packet_valued, elements.get(name, ()))):
                lists.discard(name)
                changed = True
        for name in list(loaded):
            if not all(map(packet_valued, values[name])):
                loaded.discard(name)
                changed = True
    return frozenset(loaded)


def _rewrite_block(body: list[ast.stmt], loaded: frozenset[str]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement in body:
        for field in ("body", "orelse", "finalbody"):
            block = getattr(statement, field, None)
            if isinstance(block, list) and block:
                setattr(statement, field, _rewrite_block(block, loaded))
        if isinstance(statement, ast.For):
            result.extend(_pack_loop(statement, loaded))
        else:
            result.append(statement)
    return result


def _call(node: ast.AST, path: str, arity: int) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) == path
        and len(node.args) == arity
        and not node.keywords
    ):
        return node
    return None


def _half_lane(value: ast.expr, lane: str) -> tuple[str, str] | None:
    """``(packet, half type)`` when ``value`` is
    ``cutlass.Uint16(packet[lane]).bitcast(half)``."""
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "bitcast"
        and len(value.args) == 1
        and not value.keywords
        and call_path_str(value.args[0]) in _HALVES
    ):
        return None
    load = _call(value.func.value, "cutlass.Uint16", 1)
    if load is None:
        return None
    subscript = load.args[0]
    if not (
        isinstance(subscript, ast.Subscript)
        and isinstance(subscript.value, ast.Name)
        and isinstance(subscript.slice, ast.Name)
        and subscript.slice.id == lane
    ):
        return None
    half = call_path_str(value.args[0])
    assert half is not None
    return subscript.value.id, half


def _packet_only_subscripted(loop: ast.For, packet: str) -> bool:
    """Every mention of ``packet`` in the loop body is ``packet[...]`` read."""
    subscripted = {
        id(node.value)
        for statement in loop.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Name)
    }
    return all(
        id(node) in subscripted
        for statement in loop.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and node.id == packet
    )


def _pack_loop(loop: ast.For, loaded: frozenset[str]) -> list[ast.stmt]:
    """``loop`` without its qualifying folds, followed by their packed forms;
    ``loaded`` holds the names of loaded packets (``loaded_packets``)."""
    extent = loop_extent(loop, kinds=("constexpr",), minimum=2)
    if (
        extent is None
        or extent % 2
        or any(
            isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise))
            for statement in loop.body
            for node in ast.walk(statement)
        )
    ):
        # A lane that leaves the body early skips its fold.
        return [loop]
    assert isinstance(loop.target, ast.Name)
    lane = loop.target.id
    written = [names_written(statement) for statement in loop.body]
    read = [names_read(statement) for statement in loop.body]
    if any(lane in names for names in written):
        return [loop]
    packed: list[ast.stmt] = []
    kept: list[ast.stmt] = []
    for index, statement in enumerate(loop.body):
        fold = _match_fold(loop.body, index, lane, written)
        if fold is not None:
            accumulator, lane_operand, packet, half = fold
            others = [i for i in range(len(loop.body)) if i != index]
            if (
                not any(accumulator in written[i] | read[i] for i in others)
                and packet in loaded
                and not any(packet in names for names in written)
                and _packet_only_subscripted(loop, packet)
            ):
                packed.append(
                    _packed_statement(statement, lane_operand, packet, extent, half)
                )
                continue
        kept.append(statement)
    if not packed:
        return [loop]
    # The lane definitions the folds read stay: the body is never empty.
    loop.body = kept
    return [loop, *packed]


def _definition(
    body: list[ast.stmt], index: int, name: str, written: list[set[str]]
) -> tuple[int, ast.expr] | None:
    """The index and value of the assignment whose value ``name`` holds at
    ``body[index]``: the last statement before it that writes ``name``,
    when that is a plain ``name = value`` (not a write in a nested block).
    None when another statement writes it last, or none before it in the
    body does (it holds an earlier trip's value, or one from before the
    loop)."""
    for earlier in reversed(range(index)):
        if name in written[earlier]:
            statement = body[earlier]
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name
            ):
                return earlier, statement.value
            return None
    return None


def _match_fold(
    body: list[ast.stmt], index: int, lane: str, written: list[set[str]]
) -> tuple[str, int, str, str] | None:
    """``(accumulator, lane operand index, packet, half type)`` of the
    statement ``body[index]`` when it is ``acc =
    cute.arch.fmax|fmin(acc, cutlass.Float32(x), nan=True)`` (either operand
    order) with ``x`` holding a half lane of ``packet`` there, or an fp32
    local of one (``_definition``); ``written`` holds the names each
    statement of ``body`` writes."""
    statement = body[index]
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
        and call_path_str(statement.value.func) in _FOLDS
        and len(statement.value.args) == 2
        and len(statement.value.keywords) == 1
        and statement.value.keywords[0].arg == "nan"
        and isinstance(statement.value.keywords[0].value, ast.Constant)
        and statement.value.keywords[0].value.value is True
    ):
        return None
    accumulator = statement.targets[0].id
    accumulators = [
        isinstance(argument, ast.Name) and argument.id == accumulator
        for argument in statement.value.args
    ]
    if accumulators.count(True) != 1:
        return None
    lane_operand = accumulators.index(False)
    value: ast.expr = statement.value.args[lane_operand]
    # ``cutlass.Float32(x)``; ``x`` the half lane, or ``cutlass.Float32`` of it.
    for _ in range(2):
        widened = _call(value, _FP32, 1)
        if widened is None or not isinstance(widened.args[0], ast.Name):
            return None
        definition = _definition(body, index, widened.args[0].id, written)
        if definition is None:
            return None
        index, value = definition
        match = _half_lane(value, lane)
        if match is not None:
            return accumulator, lane_operand, *match
    return None


def _packed_statement(
    statement: ast.stmt, lane_operand: int, packet: str, extent: int, half: str
) -> ast.stmt:
    """The fold ``statement`` (at its source location) with its lane operand
    replaced by the packed fold of ``packet``'s first ``extent`` lanes."""
    new = deep_clone(statement)
    assert isinstance(new, ast.Assign) and isinstance(new.value, ast.Call)
    op = _FOLDS[str(call_path_str(new.value.func))]
    packed = expr_from_string(f"{HELPER}({packet}, {extent}, {half}, {op!r})")
    assert isinstance(packed, ast.expr)
    new.value.args[lane_operand] = packed
    return new
