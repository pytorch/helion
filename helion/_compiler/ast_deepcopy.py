"""Copy ordinary Python AST graphs without recursing through AST children."""

from __future__ import annotations

import _ast
import ast
import copy
import copyreg
from types import BuiltinFunctionType
from types import MethodDescriptorType
from types import WrapperDescriptorType
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    from typing import Any

_T = TypeVar("_T")
_AST_TYPES = frozenset(
    value
    for value in vars(_ast).values()
    if isinstance(value, type) and issubclass(value, ast.AST)
)
_ATOMIC_TYPES = frozenset(
    (type(None), bool, int, float, complex, str, bytes, type(Ellipsis))
)
_OBJECT_PROTOCOLS = ("__hash__", "__eq__", "__ne__", "__reduce_ex__", "__getstate__")
_AST_PROTOCOLS = {
    "__init__": WrapperDescriptorType,
    "__reduce__": MethodDescriptorType,
    "__getattribute__": WrapperDescriptorType,
    "__setattr__": WrapperDescriptorType,
    "__delattr__": WrapperDescriptorType,
}
_ABSENT_PROTOCOLS = ("__copy__", "__deepcopy__", "__setstate__", "__getattr__")
_PROTOCOL_NAMES = frozenset(
    (*_OBJECT_PROTOCOLS, *_AST_PROTOCOLS, *_ABSENT_PROTOCOLS, "__new__")
)


def _ordinary_ast_protocols(cls: type[ast.AST]) -> bool:
    # Check native descriptor provenance, not a snapshot that might already
    # contain customized constructors or access/hash hooks.
    # Inspect class dictionaries without invoking a custom descriptor during
    # admission. Even an eventual fallback must not add user-visible lookups.
    protocols = {}
    for base in reversed(cls.__mro__):
        protocols.update(vars(base))
    if any(protocols.get(name) is not None for name in _ABSENT_PROTOCOLS):
        return False
    if any(
        protocols.get(name) is not vars(object).get(name) for name in _OBJECT_PROTOCOLS
    ):
        return False
    for name, descriptor_type in _AST_PROTOCOLS.items():
        protocol = protocols.get(name)
        if (
            type(protocol) is not descriptor_type
            or protocol.__objclass__ is not ast.AST
            or protocol.__name__ != name
        ):
            return False
    constructor = protocols.get("__new__")
    return (
        type(constructor) is BuiltinFunctionType
        and constructor.__self__ is ast.AST
        and constructor.__name__ == "__new__"
    )


def _ordinary_deepcopy_dispatch(cls: type) -> bool:
    if cls in _ATOMIC_TYPES:
        expected = vars(copy)["_deepcopy_atomic"]
    elif cls is list:
        expected = vars(copy)["_deepcopy_list"]
    elif cls is dict:
        expected = vars(copy)["_deepcopy_dict"]
    elif cls is tuple:
        expected = vars(copy)["_deepcopy_tuple"]
    elif cls in _AST_TYPES or cls in (set, frozenset):
        expected = None
    else:
        return False
    return (
        vars(copy)["_deepcopy_dispatch"].get(cls) is expected
        and cls not in vars(copy)["dispatch_table"]
        and cls not in copyreg.dispatch_table
    )


def _ordinary_registry_key(key: object) -> bool:
    meta = type(key)
    if type(meta) is not type or not issubclass(meta, type):
        return False
    # Extension classes can have a metaclass while retaining identity lookup.
    # Inspect its slots without invoking key or metaclass attribute hooks.
    protocols = {}
    for base in reversed(type.__getattribute__(meta, "__mro__")):
        protocols.update(type.__getattribute__(base, "__dict__"))
    return all(
        protocols.get(name) is vars(object)[name]
        for name in ("__hash__", "__eq__", "__ne__")
    )


def deepcopy_ast(value: _T) -> _T:
    """Preserve deepcopy's attributes, aliases and cycles for plain AST graphs.

    Native AST reconstruction adds several Python frames per AST level. Reserve
    every AST copy in one memo first, so copying its state never descends through
    another AST. Containers and state dictionaries still use stdlib deepcopy.

    Custom AST classes, metadata objects and copy protocols retain the original
    stdlib path. Inspect the whole graph before copying anything so that this
    fallback neither invokes a custom hook twice nor exposes partial copies.
    Deep non-AST container nesting retains stdlib's recursion limit.
    The stdlib copy core functions themselves must remain unmodified.
    """
    # A customized table/key must not run a lookup hook during admission.
    tables = (
        vars(copy)["_deepcopy_dispatch"],
        vars(copy)["dispatch_table"],
        copyreg.dispatch_table,
    )
    if any(type(table) is not dict for table in tables) or any(
        not _ordinary_registry_key(key) for table in tables for key in table
    ):
        return copy.deepcopy(value)
    pending: list[Any] = [value]
    seen: set[int] = set()
    nodes: list[ast.AST] = []
    checked_types: set[type[ast.AST]] = set()
    checked_copy_types: set[type] = set()
    while pending:
        item = pending.pop()
        cls = type(item)
        if type(cls) is not type:
            return copy.deepcopy(value)
        if cls not in checked_copy_types:
            if not _ordinary_deepcopy_dispatch(cls):
                return copy.deepcopy(value)
            checked_copy_types.add(cls)
        if cls in _ATOMIC_TYPES or id(item) in seen:
            continue
        seen.add(id(item))
        if cls in _AST_TYPES:
            if cls not in checked_types:
                if not _ordinary_ast_protocols(cls):
                    return copy.deepcopy(value)
                checked_types.add(cls)
            state = item.__dict__
            if type(state) is not dict or any(type(key) is not str for key in state):
                return copy.deepcopy(value)
            if _PROTOCOL_NAMES.intersection(state):
                return copy.deepcopy(value)
            nodes.append(item)
            pending.append(state)
        elif cls is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        elif cls in (list, tuple, set, frozenset):
            pending.extend(item)
        else:
            return copy.deepcopy(value)

    # Native AST.__reduce__ reconstructs with empty arguments. The verified
    # constructors create blank shells without consulting shallow-copy hooks.
    memo = {id(node): type(node)() for node in nodes}
    for node in nodes:
        # Match copy._reconstruct: update the new object's dictionary, rather
        # than installing the copied state dictionary as the object's storage.
        memo[id(node)].__dict__.update(copy.deepcopy(node.__dict__, memo))
    return copy.deepcopy(value, memo)
