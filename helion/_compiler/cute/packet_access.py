"""Matcher and renderer of the memory-access calls the CuTe emitters generate.

The lane-loop, pipelining and vectorization passes run over the generated
device body and have to recognize its loads and stores again.  This module
is the one description of that grammar (``match_packet_access`` is the
analysis, ``render_packet_load`` / ``render_masked_packet_load`` /
``vector_type_source`` the emitter side), replacing the per-pass parsers
of the same call text.  The grammar, pinned by the render corpus of
``test/test_cute_*.py`` (``test/test_cute_packet_access.py`` holds every
call form the corpus contains):

======================================================================  ================
call                                                                    ``kind``
======================================================================  ================
``(PTR).load()``                                                        ``load``
``(PTR).store(VALUE)``                                                  ``store``
``cute.arch.load(PTR, cutlass.T[, cop='cs' | level1_eviction_priority=...])``  ``arch_load``
``cute.arch.load(PTR, ir.VectorType.get([V], cutlass.T.mlir_type)[, kw])``     ``packet``
``<L2 helper>(PTR, ir.VectorType.get([V], cutlass.T.mlir_type))``              ``hinted_packet``
``_cute_load_vNb_if(PTR, PRED, ir.VectorType.get([V], cutlass.T.mlir_type)[, 'prio'])``  ``masked_packet``
``<flush helper>(PTR, VALUES)``                                         ``flush``
======================================================================  ================

``arch_load`` is the scalar ``cute.arch.load``: a cache-hinted 4- or 8-byte
scalar (``cutlass.Float32``), or a byte-packed word carrier
(``cutlass.Uint8`` for an fp8/fp4 element, ``cutlass.Uint32`` /
``cutlass.Uint64`` for a packed fp8 chunk) whose element count the call
does not show.  The L2 helpers are the names of
``cache_policy_loads._CUTE_CACHE_LOAD_HELPER_NAMES``, the predicated loads
those of ``cache_policy_loads.MASKED_PACKET_LOAD_HELPER_NAMES``, the flushes
those of ``packet_flush_names`` (the ``_cute_store_*_vec`` family).

A consumer narrows the match to what it handles (a plain ``packet`` with
no keywords and a ``Uint16`` / ``Uint32`` carrier, say) by reading the
record's fields; the matcher itself only parses.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import Literal

from ._ast_pass_utils import call_path
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .cache_policy_loads import MASKED_PACKET_LOAD_HELPER_NAMES
from .cache_policy_loads import CachePolicy
from .packet_flush_names import PACKET_FLUSH_HELPER_NAMES

PacketKind = Literal[
    "load", "store", "arch_load", "packet", "hinted_packet", "masked_packet", "flush"
]

_ARCH_LOAD = ("cute", "arch", "load")
_ARCH_STORE = ("cute", "arch", "store")
_VECTOR_TYPE = ("ir", "VectorType", "get")


@dataclasses.dataclass(frozen=True)
class PacketAccess:
    """One generated load or store call, parsed.

    ``pointer`` is the addressed operand (the receiver of ``.load()`` /
    ``.store()``, the first argument of every other form).  ``width`` and
    ``carrier`` describe the ``ir.VectorType.get([V], cutlass.T.mlir_type)``
    operand of the vector forms (``carrier`` is ``"cutlass.T"``); for an
    ``arch_load`` ``carrier`` is the scalar type text and ``width`` is None
    (one element, or a packed word).  ``helper`` is the called helper name
    of the hinted / masked / flush forms, ``predicate`` the guard of a
    masked packet, ``keywords`` the call's keywords as written.
    """

    kind: PacketKind
    call: ast.Call
    pointer: ast.expr
    width: int | None = None
    carrier: str | None = None
    helper: str | None = None
    predicate: ast.expr | None = None
    keywords: tuple[ast.keyword, ...] = ()

    @property
    def is_load(self) -> bool:
        return self.kind not in ("store", "flush")

    @property
    def is_vector(self) -> bool:
        """A packet of ``width`` elements (plain, hinted or predicated)."""
        return self.kind in ("packet", "hinted_packet", "masked_packet")

    @property
    def vector_type(self) -> ast.Call | None:
        """The ``ir.VectorType.get(...)`` operand of a vector form."""
        if self.kind == "masked_packet":
            operand = self.call.args[2]
        elif self.is_vector:
            operand = self.call.args[1]
        else:
            return None
        assert isinstance(operand, ast.Call)
        return operand

    def cache_policy(self) -> CachePolicy | None:
        """The hint spelled by the call's keywords (``arch_load`` / ``packet``),
        or None when a keyword is not a string-valued cache hint."""
        return CachePolicy.from_load_keywords(self.keywords)


def vector_type_parts(node: ast.AST) -> tuple[int | None, str | None] | None:
    """``(V, "cutlass.T")`` of ``ir.VectorType.get([V], cutlass.T.mlir_type)``;
    None when ``node`` is not that call, a None component when the shape or
    element operand is not the generated spelling."""
    if not (
        isinstance(node, ast.Call)
        and call_path(node.func) == _VECTOR_TYPE
        and len(node.args) == 2
        and not node.keywords
    ):
        return None
    shape, element = node.args
    width = None
    if (
        isinstance(shape, ast.List)
        and len(shape.elts) == 1
        and isinstance(shape.elts[0], ast.Constant)
        and type(shape.elts[0].value) is int
    ):
        width = shape.elts[0].value
    carrier = None
    if isinstance(element, ast.Attribute) and element.attr == "mlir_type":
        carrier = ast.unparse(element.value)
    return width, carrier


def match_packet_access(node: ast.AST) -> PacketAccess | None:
    """Parse ``node`` as one of the generated load / store call forms."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in ("load", "store"):
        path = call_path(func)
        if path not in (_ARCH_LOAD, _ARCH_STORE):
            if func.attr == "load" and not node.args and not node.keywords:
                return PacketAccess("load", node, func.value)
            if func.attr == "store" and len(node.args) == 1 and not node.keywords:
                return PacketAccess("store", node, func.value)
            return None
        if path == _ARCH_STORE or len(node.args) != 2:
            return None
        pointer, operand = node.args
        keywords = tuple(node.keywords)
        if (parts := vector_type_parts(operand)) is not None:
            width, carrier = parts
            return PacketAccess(
                "packet", node, pointer, width, carrier, None, None, keywords
            )
        if isinstance(operand, (ast.Attribute, ast.Name)):
            return PacketAccess(
                "arch_load",
                node,
                pointer,
                None,
                ast.unparse(operand),
                None,
                None,
                keywords,
            )
        return None
    if not isinstance(func, ast.Name):
        return None
    name = func.id
    if name in _CUTE_CACHE_LOAD_HELPER_NAMES:
        if len(node.args) != 2 or (parts := vector_type_parts(node.args[1])) is None:
            return None
        width, carrier = parts
        return PacketAccess(
            "hinted_packet",
            node,
            node.args[0],
            width,
            carrier,
            name,
            None,
            tuple(node.keywords),
        )
    if name in MASKED_PACKET_LOAD_HELPER_NAMES:
        if (
            len(node.args) not in (3, 4)
            or (parts := vector_type_parts(node.args[2])) is None
        ):
            return None
        width, carrier = parts
        return PacketAccess(
            "masked_packet",
            node,
            node.args[0],
            width,
            carrier,
            name,
            node.args[1],
            tuple(node.keywords),
        )
    if name in PACKET_FLUSH_HELPER_NAMES:
        if len(node.args) != 2 or node.keywords:
            return None
        return PacketAccess("flush", node, node.args[0], None, None, name)
    return None


def vector_type_source(width: int, carrier: str) -> str:
    """``ir.VectorType.get([V], cutlass.T.mlir_type)``."""
    return f"ir.VectorType.get([{width}], {carrier}.mlir_type)"


def render_packet_load(
    pointer: str,
    width: int,
    carrier: str,
    *,
    helper: str | None = None,
    kwargs: str = "",
) -> str:
    """The ``packet`` form (``cute.arch.load`` with ``kwargs``, the
    ``CachePolicy.load_kwargs`` fragment) or, with ``helper``, the
    ``hinted_packet`` form of a ``width`` x ``carrier`` packet at ``pointer``."""
    vector_type = vector_type_source(width, carrier)
    if helper is not None:
        return f"{helper}({pointer}, {vector_type})"
    return f"cute.arch.load({pointer}, {vector_type}{kwargs})"


def render_masked_packet_load(
    helper: str, pointer: str, predicate: str, width: int, carrier: str, priority: str
) -> str:
    """The ``masked_packet`` form: ``helper(pointer, predicate, vector_type[, priority])``."""
    arguments = f"{pointer}, {predicate}, {vector_type_source(width, carrier)}"
    if priority:
        arguments += f", {priority!r}"
    return f"{helper}({arguments})"
