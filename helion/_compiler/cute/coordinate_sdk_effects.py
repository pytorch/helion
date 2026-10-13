"""Closed read-only uses of scalar coordinates by the supported CuTe SDK.

This is an effect contract, not a claim that arbitrary SDK calls are pure. A
thread slice retains its input Numeric object. Only its closed partition/retile
uses below are allowed to retain a coordinate used by the sharing pass.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from types import BuiltinFunctionType
from types import CodeType
from types import FunctionType
from types import MethodDescriptorType
from types import WrapperDescriptorType
from typing import TYPE_CHECKING

from .simplify_proven_bounds import _call_path

if TYPE_CHECKING:
    from collections.abc import Sequence


# This bounded contract deliberately declines an unreviewed SDK implementation.
# It is independent of any kernel, shape, schedule or generated variable name.
_AUDITED_SDK = "d5eae113cbd429b5ec3780a51cbced83324d9d30e8f96d689db3a1fce13319a8"


def _code_facts(code: CodeType) -> object:
    return (
        code.co_code,
        tuple(_code_facts(x) if type(x) is CodeType else x for x in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        getattr(code, "co_exceptiontable", b""),
    )


def _function_facts(function: object) -> object:
    if function is None or type(function) in (bool, int, float, str, bytes):
        return (type(function).__name__, function)
    if type(function) is tuple:
        return ("tuple", tuple(map(_function_facts, function)))
    if type(function) is property:
        return (
            "property",
            *map(_function_facts, (function.fget, function.fset, function.fdel)),
        )
    if type(function) in (
        BuiltinFunctionType,
        MethodDescriptorType,
        WrapperDescriptorType,
    ):
        assert isinstance(
            function, (BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType)
        )
        return (
            type(function).__name__,
            function.__module__ if type(function) is BuiltinFunctionType else None,
            function.__qualname__,
        )
    if isinstance(function, type):
        return (
            "class",
            function.__module__,
            function.__qualname__,
            tuple((base.__module__, base.__qualname__) for base in function.__mro__),
            tuple(sorted(function.__dict__)),
            tuple(
                (key, value)
                for key, value in sorted(function.__dict__.items())
                if value is None or type(value) in (bool, int, float, str, bytes)
            ),
        )
    if type(function) is not FunctionType:
        return (type(function).__module__, type(function).__qualname__)
    wrapped = function.__dict__.get("__wrapped__")
    return (
        function.__module__,
        function.__qualname__,
        _code_facts(function.__code__),
        repr(function.__defaults__),
        repr(function.__kwdefaults__),
        tuple(
            _function_facts(cell.cell_contents) for cell in (function.__closure__ or ())
        ),
        _function_facts(wrapped) if wrapped is not None else None,
    )


def _bindings() -> tuple[object, ...]:
    import cutlass
    from cutlass import cute
    from cutlass.base_dsl import typing as numeric
    from cutlass.cute import atom
    from cutlass.cute import core
    from cutlass.cute import tuple as tuples

    return (
        cutlass.Int32,
        numeric.Int32,
        atom.Int32,
        cute.make_tiled_mma,
        atom.make_tiled_mma,
        cute.make_tiled_copy_A,
        atom.make_tiled_copy_A,
        cute.make_tiled_copy_B,
        atom.make_tiled_copy_B,
        atom._make_tiled_copy,
        atom.TiledMma,
        atom.ThrMma,
        atom.TiledCopy,
        atom.ThrCopy,
        atom.TiledMma.get_slice,
        atom.ThrMma.__init__,
        atom.ThrMma.__dict__["thr_idx"],
        atom.ThrMma.partition_A,
        atom.ThrMma.partition_B,
        atom.ThrMma.partition_C,
        atom.TiledCopy.get_slice,
        atom.ThrCopy.__init__,
        atom.ThrCopy.__dict__["thr_idx"],
        atom.ThrCopy.partition_S,
        atom.ThrCopy.partition_D,
        atom.ThrCopy.retile,
        atom.TiledCopy.retile,
        atom._pack_coord,
        core._pack_coord,
        core._pack_x,
        core._get_typed_value,
        tuples.transform_leaf,
        numeric._binary_op,
        numeric.Numeric.__init__,
        numeric.Numeric.to,
        numeric.Numeric.__dict__["ir_value"],
        numeric.Integer.__init__,
        *(
            getattr(numeric.Int32, name)
            for name in (
                "__add__",
                "__radd__",
                "__mul__",
                "__rmul__",
                "__floordiv__",
                "__rfloordiv__",
                "__mod__",
                "__rmod__",
                "__eq__",
                "__ne__",
                "__lt__",
                "__le__",
                "__gt__",
                "__ge__",
            )
        ),
        atom.Atom.__init__,
        atom.MmaAtom.__init__,
        atom.CopyAtom.__init__,
        atom.TiledMma.__init__,
        atom.TiledCopy.__init__,
        numeric.Numeric,
        numeric.Integer,
        type(numeric.Int32),
        type(numeric.Int32).__call__,
        cutlass.range_constexpr,
        cute.arch.thread_idx,
        cute.arch.block_idx,
        core.transform_leaf,
        core.Numeric,
        numeric._binary_op_type_promote,
        atom.Atom.__dict__["op"],
        core._check_coord,
        core.flatten_to_tuple,
        tuples.flatten_to_tuple,
        tuples.wrap,
        core.is_integer,
        core.Integer,
        core.IntValue,
        numeric.as_numeric,
        numeric.Numeric._from_python_value,
        numeric.Numeric.__dict__["dtype"],
        numeric.NumericMeta,
        *(
            getattr(numeric.operator, name)
            for name in (
                "add",
                "mul",
                "floordiv",
                "mod",
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
            )
        ),
    )


def _sdk_facts() -> tuple[str, tuple[object, ...]]:
    from cutlass.base_dsl import typing as numeric
    from cutlass.cute import atom
    from cutlass.cute import core
    from cutlass.cute import tuple as tuples

    files = tuple(
        hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
        for module in (numeric, atom, core, tuples)
    )
    bindings = _bindings()
    # Exported aliases must continue to be the implementations inspected here.
    aliases = tuple(
        bindings[a] is bindings[b]
        for a, b in (
            (0, 1),
            (0, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (25, 26),
            (27, 28),
        )
    )
    digest = hashlib.sha256(
        repr(
            (
                files,
                aliases,
                tuple(map(_function_facts, bindings)),
                (
                    numeric.Int32.width,
                    numeric.Int32.signed,
                    repr(numeric.Int32.numpy_dtype),
                ),
            )
        ).encode()
    ).hexdigest()
    return digest, bindings


@dataclass(frozen=True)
class CoordinateSDKEffects:
    fingerprint: str
    bindings: tuple[object, ...]

    def check(self) -> None:
        try:
            fingerprint, bindings = _sdk_facts()
        except (ImportError, AttributeError) as error:
            raise ValueError("coordinate SDK effect contract unavailable") from error
        if (
            fingerprint != _AUDITED_SDK
            or fingerprint != self.fingerprint
            or any(a is not b for a, b in zip(self.bindings, bindings, strict=True))
        ):
            raise ValueError("coordinate SDK effect contract changed")


def coordinate_sdk_effects() -> CoordinateSDKEffects | None:
    try:
        fingerprint, bindings = _sdk_facts()
    except (ImportError, AttributeError):
        return None
    if fingerprint != _AUDITED_SDK:
        return None
    return CoordinateSDKEffects(fingerprint, bindings)


def readonly_slice_uses(body: Sequence[ast.AST]) -> frozenset[int]:
    """Return exact Name uses whose retained wrappers have closed readers.

    The original factory and its first use as get_slice are in one statement
    list. All uses of the resulting slice are original partition/retile calls;
    passing, returning, aliasing or exposing the receiver declines the exemption.
    The caller must separately bind the SDK contract and the complete body.
    """
    nodes: list[ast.AST] = []
    pending = list(reversed(body))
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
    parents = {
        id(child): node for node in nodes for child in ast.iter_child_nodes(node)
    }
    writes = Counter(
        node.id
        for node in nodes
        if type(node) is ast.Name and type(node.ctx) in (ast.Store, ast.Del)
    )
    for node in nodes:
        if type(node) is ast.ImportFrom:
            for alias in node.names:
                writes[alias.asname or alias.name] += 1
    loads: dict[str, list[ast.Name]] = {}
    for node in nodes:
        if type(node) is ast.Name and type(node.ctx) is ast.Load:
            loads.setdefault(node.id, []).append(node)
    result: set[int] = set()

    def visit(statements: Sequence[ast.AST]) -> None:
        factories: dict[str, tuple[str, ast.Name | None]] = {}
        for statement in statements:
            if type(statement) is ast.Assign and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
                if type(target) is ast.Name and writes[target.id] == 1:
                    if type(value) is ast.Call:
                        path = _call_path(value.func)
                        if path is not None and path in (
                            ("cute", "make_tiled_mma"),
                            ("cute", "make_tiled_copy_A"),
                            ("cute", "make_tiled_copy_B"),
                        ):
                            uses = loads.get(target.id, [])
                            factories[target.id] = (path[-1], uses[0] if uses else None)
                        elif (
                            type(value.func) is ast.Attribute
                            and value.func.attr == "get_slice"
                            and type(value.func.value) is ast.Name
                            and len(value.args) == 1
                            and type(value.args[0]) is ast.Name
                            and not value.keywords
                        ):
                            owner = value.func.value
                            factory = factories.get(owner.id)
                            if factory is not None and factory[1] is owner:
                                methods = (
                                    {"partition_A", "partition_B", "partition_C"}
                                    if factory[0] == "make_tiled_mma"
                                    else {"partition_S", "partition_D", "retile"}
                                )
                                readers = loads.get(target.id, [])
                                valid = bool(readers)
                                for reader in readers:
                                    attr = parents[id(reader)]
                                    call = parents.get(id(attr))
                                    if not (
                                        type(attr) is ast.Attribute
                                        and attr.value is reader
                                        and attr.attr in methods
                                        and type(call) is ast.Call
                                        and call.func is attr
                                        and len(call.args) == 1
                                        and not call.keywords
                                    ):
                                        valid = False
                                        break
                                if valid:
                                    result.add(id(value.args[0]))
            for field in ("body", "orelse"):
                nested = getattr(statement, field, None)
                if type(nested) is list:
                    visit(nested)

    visit(body)
    return frozenset(result)
